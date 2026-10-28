// HgRT kernels for Mercurius-1-4B decode (hgrt/mercurius_engine.py). Every kernel takes M (or NQ) token rows: M = 1 is
// plain decode, M = k + 1 the speculative verification pass. Each row is computed with exactly the arithmetic of the
// M = 1 case (same reduction orders, explicit _rn intrinsics where a compiler contraction could differ), so a verified
// token is bit-identical to what plain decoding produces: speculative decoding is lossless by construction.
//
//   nf4_gemv     NF4 (bitsandbytes NormalFloat-4, blocks of 64, fp32 absmax) x int8 activations. The 16 NF4 levels are
//                represented as 128 * h + l (h = round(127 L), l = round(16256 L - 128 h), both int8; |error| <= 3.1e-5)
//                looked up with byte permutes from two 16-entry tables; two dp4a per 4 values. Exact to the stored
//                codes and scales (no re-quantization of the weights).
//   rms / quant  producers of the int8 activation groups (natural element order, one fp32 scale per 32).
//   dgemv        small dense bf16 matrices (gates, VeRA factors, rotations) for M rows, any K.
//   gdn2_*       GDN-2: channel-wise decay, erase and write gates, conv, recurrence + gated RMSNorm; verify/commit
//                modes (the state is only written once the accepted count is known).
//   mla_*        absorbed MLA over the TurboQuant 4-bit latent cache: encode, query map, split-KV tensor-core
//                attention straight from the codes, reduction, per-group V up-projection.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define STREAM c10::cuda::getCurrentCUDAStream()
#define BF(t) reinterpret_cast<__nv_bfloat16*>((t).data_ptr())
#define CBF(t) reinterpret_cast<const __nv_bfloat16*>((t).data_ptr())

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
// block sum, fixed order: warp tree, then warp 0..NW-1 sequentially (same result in every block / launch)
template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sh) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) sh[w] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; i++) s = __fadd_rn(s, sh[i]);
  __syncthreads();
  return s;
}
// sin / cos of an fp32 angle as accurately as libm (the reference uses torch's cos of the fp32 product pos * inv_freq):
// Cody-Waite reduction by 2 pi in fp32 (good for |x| < 2^17 rad), then the fast intrinsic on [-pi, pi]. With
// --use_fast_math a plain cosf / sinf is the intrinsic on the raw argument, off by ~1e-2 at 32k rad.
__device__ __forceinline__ void sincos_acc(float x, float* s, float* c) {
  const float k = rintf(x * 0.15915494309189535f);
  float r = __fmaf_rn(-k, 6.28318548202514648f, x);
  r = __fmaf_rn(-k, -1.7484555314695172e-07f, r);
  __sincosf(r, s, c);
}
__device__ __forceinline__ float sigm(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, __expf(-x))); }

// int8 quantization of one 32-value group held by a full warp (lane = element), natural order
__device__ __forceinline__ void warp_quant32(float x, int g, int lane, int8_t* XQ, float* XD) {
  float a = fabsf(x);
#pragma unroll
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  const float d = a / 127.f;
  const int q = d > 0.f ? __float2int_rn(__fdiv_rn(x, d)) : 0;
  XQ[g * 32 + lane] = (int8_t)q;
  if (lane == 0) XD[g] = d;
}


constexpr int MG = 4, MH = 16, RD = 64;


constexpr int AC = 32, AT = 256;


// v2: virtual splits (chunk c belongs to split c % VS; a block runs splits b, b + G, .. one after the other, each with
// its own online softmax and partial), so the grid G can shrink for big NQ without changing any row's arithmetic;
// all query tokens' S tiles in one phase, softmax over all 16 NQ rows in one phase, PV for all tokens in one phase;
// codes read as 32-bit words, values converted in pairs.
template <int NQ, int OT>
__global__ void __launch_bounds__(AT) attn_a(const __half* __restrict__ Q, int Ep, const uint8_t* __restrict__ cache,
                                                  int RW, const __half* __restrict__ R0h, const float* __restrict__ cbl,
                                                  const float* __restrict__ cbr, const long* __restrict__ pos,
                                                  const float* __restrict__ inv_freq, float* __restrict__ part_o,
                                                  float* __restrict__ part_ml, int r, float scale, int VS) {
  extern __shared__ __align__(32) unsigned char smem[];
  const int Es = Ep + 8, rp = Ep - RD;
  constexpr int Sl = AC + 4, Pl = AC + 8, Ul = RD + 8, Rl = RD + 4;
  half* Ks = reinterpret_cast<half*>(smem);                 // [AC][Es]
  half* Uh = Ks + AC * Es;                                  // [RD][Ul]
  float* Ss = reinterpret_cast<float*>(Uh + RD * Ul);       // [NQ][4][16][Sl]; also [AC][Rl] for the rope GEMM
  half* Ps = reinterpret_cast<half*>(Ss + NQ * 4 * 16 * Sl);  // [NQ][16][Pl]
  float* mrow = reinterpret_cast<float*>(Ps + NQ * 16 * Pl);  // [NQ][16]
  float* lrow = mrow + NQ * 16;
  float* arow = lrow + NQ * 16;
  int* posj = reinterpret_cast<int*>(arow + NQ * 16);       // [AC]
  float* invr = reinterpret_cast<float*>(posj + AC);        // [AC][4]
  float* rmap = invr + AC * 4;                              // [256]
  float* cbs = rmap + 256;                                  // [32]
  int* flag = reinterpret_cast<int*>(cbs + 32);             // [2][4]
  uint8_t* rows = reinterpret_cast<uint8_t*>(flag + 8);     // [AC][RW]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, b = blockIdx.x;
  const long p0 = pos[0];
  const int ncell = (int)(p0 + NQ);
  const int nchunk = (ncell + AC - 1) / AC;
  const int nct = rp / 16, nks = Ep / 16;

  for (int i = tid; i < RD * RD; i += AT) Uh[(i / RD) * Ul + i % RD] = R0h[i];
  if (tid < 16) cbs[tid] = cbl[tid];
  else if (tid < 32) cbs[tid] = cbr[tid - 16];
  for (int i = tid; i < 256; i += AT) rmap[i] = (float)(i / 16);
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowf;
  wmma::load_matrix_sync(rowf, rmap, 16, wmma::mem_row_major);
  int ra = (int)rowf.x[0], rb = ra;
#pragma unroll
  for (int i = 0; i < rowf.num_elements; i++) if ((int)rowf.x[i] != ra) rb = (int)rowf.x[i];

  auto stage = [&](int chunk) {
    const int j0 = chunk * AC, nc = min(AC, ncell - j0);
    const uint8_t* src = cache + (size_t)j0 * RW;
    const int n16 = nc * RW / 16;
    for (int s2 = tid; s2 < n16; s2 += AT) {
      const unsigned dst = (unsigned)__cvta_generic_to_shared(rows + 16 * s2);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src + 16 * s2));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  for (int v = b; v < VS; v += G) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O[NQ][OT];
#pragma unroll
    for (int i = 0; i < NQ; i++)
#pragma unroll
      for (int k = 0; k < OT; k++) wmma::fill_fragment(O[i][k], 0.f);
    for (int i = tid; i < NQ * 16; i += AT) { mrow[i] = -INFINITY; lrow[i] = 0.f; arow[i] = 1.f; }
    if (tid < 8) flag[tid] = 0;
    __syncthreads();
    if (v < nchunk) stage(v);
    int it = 0;
    for (int chunk = v; chunk < nchunk; chunk += VS, it++) {
      const int j0 = chunk * AC, nc = min(AC, ncell - j0);
      int* fl = flag + (it & 1) * 4;
      asm volatile("cp.async.wait_all;\n" ::);
      __syncthreads();
      if (tid < 4) flag[((it + 1) & 1) * 4 + tid] = 0;
      // decode: one warp per cell; 8 values per 32-bit code word
      for (int jl = warp; jl < AC; jl += AT / 32) {
        half* kr = Ks + jl * Es;
        if (jl >= nc) {
          for (int w8 = lane; w8 < Ep / 8; w8 += 32) reinterpret_cast<uint4*>(kr)[w8] = make_uint4(0, 0, 0, 0);
          if (lane == 0) posj[jl] = 0x7fffffff;
          if (lane < 4) invr[jl * 4 + lane] = 0.f;
          continue;
        }
        const uint8_t* row = rows + jl * RW;
        if (lane == 0) posj[jl] = *reinterpret_cast<const int*>(row);
        if (lane < 4) invr[jl * 4 + lane] = reinterpret_cast<const float*>(row + 4)[lane];
        const float nl = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + 20)), 1e-12f);
        const float nr = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + 22)), 1e-12f);
        const uint32_t* rcw = reinterpret_cast<const uint32_t*>(row + 24);
        const uint32_t* lcw = reinterpret_cast<const uint32_t*>(row + 24 + RD / 2);
        for (int w8 = lane; w8 < Ep / 8; w8 += 32) {
          const bool lat = w8 >= RD / 8;
          const int e0 = lat ? 8 * w8 - RD : 8 * w8;
          const int n = lat ? r : RD;
          const uint32_t codes = e0 < n ? (lat ? lcw[e0 >> 3] : rcw[w8]) : 0u;
          const float nn = lat ? nl : nr;
          const float* cb = cbs + (lat ? 0 : 16);
          uint32_t o[4];
#pragma unroll
          for (int u = 0; u < 4; u++) {
            float v0 = cb[(codes >> (8 * u)) & 15] * nn, v1 = cb[(codes >> (8 * u + 4)) & 15] * nn;
            if (e0 + 2 * u >= n) v0 = 0.f;
            if (e0 + 2 * u + 1 >= n) v1 = 0.f;
            const __half2 hv = __floats2half2_rn(v0, v1);
            o[u] = *reinterpret_cast<const uint32_t*>(&hv);
          }
          *reinterpret_cast<uint4*>(kr + 8 * w8) = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
      __syncthreads();
      if (chunk + VS < nchunk) stage(chunk + VS);
      {
        const int ct = warp / 4, mt = warp % 4;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll
        for (int k0 = 0; k0 < RD; k0 += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
          wmma::load_matrix_sync(fa, Ks + ct * 16 * Es + k0, Es);
          wmma::load_matrix_sync(fb, Uh + k0 * Ul + mt * 16, Ul);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Ss + ct * 16 * Rl + mt * 16, acc, Rl, wmma::mem_row_major);
      }
      __syncthreads();
      for (int i = tid; i < AC * 32; i += AT) {
        const int jl = i / 32, p = i % 32;
        if (jl >= nc) continue;
        const float x0 = bf(Ss[jl * Rl + p]), x1 = bf(Ss[jl * Rl + p + 32]);
        float sn, cs;
        sincos_acc(__fmul_rn((float)posj[jl], inv_freq[p]), &sn, &cs);
        const float c = bf(cs), s = bf(sn);
        Ks[jl * Es + p] = __float2half(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
        Ks[jl * Es + p + 32] = __float2half(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
      }
      __syncthreads();
      // S = Q K^T for every token: task = (token, quarter, column tile)
      for (int task = warp; task < NQ * 8; task += AT / 32) {
        const int qi = task >> 3, qq = (task & 7) >> 1, ct = task & 1;
        const int k_lo = nks * qq / 4, k_hi = nks * (qq + 1) / 4;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        const half* qb = Q + (size_t)qi * MH * Ep;
        for (int kt = k_lo; kt < k_hi; kt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
          wmma::load_matrix_sync(fa, qb + kt * 16, Ep);
          wmma::load_matrix_sync(fb, Ks + ct * 16 * Es + kt * 16, Es);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Ss + ((qi * 4 + qq) * 16) * Sl + ct * 16, acc, Sl, wmma::mem_row_major);
      }
      __syncthreads();
      // softmax rows (token, head): lane = cell
      for (int rowi = warp; rowi < NQ * 16; rowi += AT / 32) {
        const int qi = rowi >> 4, hh = rowi & 15;
        const long qpos = p0 + qi;
        const float* sq = Ss + (qi * 4 * 16) * Sl;
        const int o = hh * Sl + lane, st = 16 * Sl;
        float sv = -INFINITY;
        if (lane < nc && posj[lane] <= qpos) {
          const float raw = __fadd_rn(__fadd_rn(__fadd_rn(sq[o], sq[st + o]), sq[2 * st + o]), sq[3 * st + o]);
          sv = __fmul_rn(__fmul_rn(raw, invr[lane * 4 + hh / (16 / MG)]), scale);
        }
        const float mx = warp_max(sv);
        const float m_ref = mrow[rowi];
        const bool bump = mx != -INFINITY && (m_ref == -INFINITY || mx > m_ref + 8.f);
        const float m_use = bump ? mx : m_ref;
        const float pr = (m_use == -INFINITY || sv == -INFINITY) ? 0.f : __expf(sv - m_use);
        Ps[(qi * 16 + hh) * Pl + lane] = __float2half(pr);
        const float l = warp_sum(pr);
        __syncwarp();
        if (lane == 0) {
          const float a = bump ? (m_ref == -INFINITY ? 0.f : __expf(m_ref - m_use)) : 1.f;
          arow[rowi] = a;
          lrow[rowi] = __fmaf_rn(lrow[rowi], a, l);
          mrow[rowi] = m_use;
          if (bump && m_ref != -INFINITY) fl[qi] = 1;
        }
      }
      __syncthreads();
#pragma unroll
      for (int qi = 0; qi < NQ; qi++) {
        const bool resc = fl[qi] != 0;
        const float a0 = arow[qi * 16 + ra], a1 = arow[qi * 16 + rb];
#pragma unroll
        for (int k = 0; k < OT; k++) {
          const int ct = warp + (AT / 32) * k;
          if (ct < nct) {
            if (resc) {
#pragma unroll
              for (int i = 0; i < O[qi][k].num_elements; i++) O[qi][k].x[i] *= (rowf.x[i] == (float)ra) ? a0 : a1;
            }
#pragma unroll
            for (int k0 = 0; k0 < AC; k0 += 16) {
              wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
              wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
              wmma::load_matrix_sync(fa, Ps + qi * 16 * Pl + k0, Pl);
              wmma::load_matrix_sync(fb, Ks + k0 * Es + RD + ct * 16, Es);
              wmma::mma_sync(O[qi][k], fa, fb, O[qi][k]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) {
      float* o = part_o + (((size_t)v * NQ + qi) * 16) * rp;
#pragma unroll
      for (int k = 0; k < OT; k++) {
        const int ct = warp + (AT / 32) * k;
        if (ct < nct) wmma::store_matrix_sync(o + ct * 16, O[qi][k], rp, wmma::mem_row_major);
      }
    }
    __syncthreads();
    for (int i = tid; i < NQ * 16; i += AT) {
      part_ml[((size_t)v * NQ * 16 + i) * 2] = mrow[i];
      part_ml[((size_t)v * NQ * 16 + i) * 2 + 1] = lrow[i];
    }
    __syncthreads();
  }
}

static int attn_a_smem(int Ep, int RW, int NQ) {
  const int Es = Ep + 8;
  return AC * Es * 2 + RD * (RD + 8) * 2 + NQ * 4 * 16 * (AC + 4) * 4 + NQ * 16 * (AC + 8) * 2 + 3 * NQ * 16 * 4 +
         AC * 4 + AC * 16 + 256 * 4 + 32 * 4 + 32 + AC * RW + 64;
}


// v2: virtual splits (chunk c belongs to split c % VS; a block runs splits b, b + G, .. one after the other, each with
// its own online softmax and partial), so the grid G can shrink for big NQ without changing any row's arithmetic;
// all query tokens' S tiles in one phase, softmax over all 16 NQ rows in one phase, PV for all tokens in one phase;
// codes read as 32-bit words, values converted in pairs.
template <int NQ, int OT>
__global__ void __launch_bounds__(AT) attn_b(const __half* __restrict__ Q, int Ep, const uint8_t* __restrict__ cache,
                                                  int RW, const __half* __restrict__ R0h, const float* __restrict__ cbl,
                                                  const float* __restrict__ cbr, const long* __restrict__ pos,
                                                  const float* __restrict__ inv_freq, float* __restrict__ part_o,
                                                  float* __restrict__ part_ml, int r, float scale, int VS) {
  extern __shared__ __align__(32) unsigned char smem[];
  const int Es = Ep + 8, rp = Ep - RD;
  constexpr int Sl = AC + 4, Pl = AC + 8, Ul = RD + 8, Rl = RD + 4;
  half* Ks = reinterpret_cast<half*>(smem);                 // [AC][Es]
  half* Uh = Ks + AC * Es;                                  // [RD][Ul]
  float* Ss = reinterpret_cast<float*>(Uh + RD * Ul);       // [NQ][4][16][Sl]; also [AC][Rl] for the rope GEMM
  half* Ps = reinterpret_cast<half*>(Ss + NQ * 4 * 16 * Sl);  // [NQ][16][Pl]
  float* mrow = reinterpret_cast<float*>(Ps + NQ * 16 * Pl);  // [NQ][16]
  float* lrow = mrow + NQ * 16;
  float* arow = lrow + NQ * 16;
  int* posj = reinterpret_cast<int*>(arow + NQ * 16);       // [AC]
  float* invr = reinterpret_cast<float*>(posj + AC);        // [AC][4]
  float* rmap = invr + AC * 4;                              // [256]
  float* cbs = rmap + 256;                                  // [32]
  int* flag = reinterpret_cast<int*>(cbs + 32);             // [2][4]
  uint8_t* rows = reinterpret_cast<uint8_t*>(flag + 8);     // [AC][RW]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, b = blockIdx.x;
  const long p0 = pos[0];
  const int ncell = (int)(p0 + NQ);
  const int nchunk = (ncell + AC - 1) / AC;
  const int nct = rp / 16, nks = Ep / 16;

  for (int i = tid; i < RD * RD; i += AT) Uh[(i / RD) * Ul + i % RD] = R0h[i];
  if (tid < 16) cbs[tid] = cbl[tid];
  else if (tid < 32) cbs[tid] = cbr[tid - 16];
  for (int i = tid; i < 256; i += AT) rmap[i] = (float)(i / 16);
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowf;
  wmma::load_matrix_sync(rowf, rmap, 16, wmma::mem_row_major);
  int ra = (int)rowf.x[0], rb = ra;
#pragma unroll
  for (int i = 0; i < rowf.num_elements; i++) if ((int)rowf.x[i] != ra) rb = (int)rowf.x[i];

  auto stage = [&](int chunk) {
    const int j0 = chunk * AC, nc = min(AC, ncell - j0);
    const uint8_t* src = cache + (size_t)j0 * RW;
    const int n16 = nc * RW / 16;
    for (int s2 = tid; s2 < n16; s2 += AT) {
      const unsigned dst = (unsigned)__cvta_generic_to_shared(rows + 16 * s2);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src + 16 * s2));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  for (int v = b; v < VS; v += G) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O[NQ][OT];
#pragma unroll
    for (int i = 0; i < NQ; i++)
#pragma unroll
      for (int k = 0; k < OT; k++) wmma::fill_fragment(O[i][k], 0.f);
    for (int i = tid; i < NQ * 16; i += AT) { mrow[i] = -INFINITY; lrow[i] = 0.f; arow[i] = 1.f; }
    if (tid < 8) flag[tid] = 0;
    __syncthreads();
    if (v < nchunk) stage(v);
    int it = 0;
    for (int chunk = v; chunk < nchunk; chunk += VS, it++) {
      const int j0 = chunk * AC, nc = min(AC, ncell - j0);
      int* fl = flag + (it & 1) * 4;
      asm volatile("cp.async.wait_all;\n" ::);
      __syncthreads();
      if (tid < 4) flag[((it + 1) & 1) * 4 + tid] = 0;
      // decode (flattened): lane handles (cell, word) items of its warp's 4 cells, 2 at a time (independent loads)
      {
        const int nw = Ep / 8;
        if (lane < 4) {
          const int jl = warp + 8 * lane;
          if (jl < nc) {
            const uint8_t* row = rows + jl * RW;
            posj[jl] = *reinterpret_cast<const int*>(row);
#pragma unroll
            for (int g = 0; g < 4; g++) invr[jl * 4 + g] = reinterpret_cast<const float*>(row + 4)[g];
          } else {
            posj[jl] = 0x7fffffff;
#pragma unroll
            for (int g = 0; g < 4; g++) invr[jl * 4 + g] = 0.f;
          }
        }
        for (int it2 = lane; it2 < 4 * nw; it2 += 32) {
          const int cl = it2 / nw, w8 = it2 - cl * nw, jl = warp + 8 * cl;
          half* kr = Ks + jl * Es;
          if (jl >= nc) { reinterpret_cast<uint4*>(kr)[w8] = make_uint4(0, 0, 0, 0); continue; }
          const uint8_t* row = rows + jl * RW;
          const bool lat = w8 >= RD / 8;
          const int e0 = lat ? 8 * w8 - RD : 8 * w8;
          const int n = lat ? r : RD;
          const uint32_t codes = e0 < n ? *reinterpret_cast<const uint32_t*>(row + 24 + (lat ? RD / 2 + (e0 >> 1) : 4 * w8)) : 0u;
          const float nn = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + (lat ? 20 : 22))), 1e-12f);
          const float* cb = cbs + (lat ? 0 : 16);
          uint32_t o[4];
#pragma unroll
          for (int u = 0; u < 4; u++) {
            float v0 = cb[(codes >> (8 * u)) & 15] * nn, v1 = cb[(codes >> (8 * u + 4)) & 15] * nn;
            if (e0 + 2 * u >= n) v0 = 0.f;
            if (e0 + 2 * u + 1 >= n) v1 = 0.f;
            const __half2 hv = __floats2half2_rn(v0, v1);
            o[u] = *reinterpret_cast<const uint32_t*>(&hv);
          }
          *reinterpret_cast<uint4*>(kr + 8 * w8) = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
      __syncthreads();
      if (chunk + VS < nchunk) stage(chunk + VS);
      {
        const int ct = warp / 4, mt = warp % 4;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll
        for (int k0 = 0; k0 < RD; k0 += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
          wmma::load_matrix_sync(fa, Ks + ct * 16 * Es + k0, Es);
          wmma::load_matrix_sync(fb, Uh + k0 * Ul + mt * 16, Ul);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Ss + ct * 16 * Rl + mt * 16, acc, Rl, wmma::mem_row_major);
      }
      __syncthreads();
      for (int i = tid; i < AC * 32; i += AT) {
        const int jl = i / 32, p = i % 32;
        if (jl >= nc) continue;
        const float x0 = bf(Ss[jl * Rl + p]), x1 = bf(Ss[jl * Rl + p + 32]);
        float sn, cs;
        sincos_acc(__fmul_rn((float)posj[jl], inv_freq[p]), &sn, &cs);
        const float c = bf(cs), s = bf(sn);
        Ks[jl * Es + p] = __float2half(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
        Ks[jl * Es + p + 32] = __float2half(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
      }
      __syncthreads();
      // S = Q K^T for every token: task = (token, quarter, column tile)
      for (int task = warp; task < NQ * 8; task += AT / 32) {
        const int qi = task >> 3, qq = (task & 7) >> 1, ct = task & 1;
        const int k_lo = nks * qq / 4, k_hi = nks * (qq + 1) / 4;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        const half* qb = Q + (size_t)qi * MH * Ep;
        for (int kt = k_lo; kt < k_hi; kt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
          wmma::load_matrix_sync(fa, qb + kt * 16, Ep);
          wmma::load_matrix_sync(fb, Ks + ct * 16 * Es + kt * 16, Es);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Ss + ((qi * 4 + qq) * 16) * Sl + ct * 16, acc, Sl, wmma::mem_row_major);
      }
      __syncthreads();
      // softmax rows (token, head): lane = cell
      for (int rowi = warp; rowi < NQ * 16; rowi += AT / 32) {
        const int qi = rowi >> 4, hh = rowi & 15;
        const long qpos = p0 + qi;
        const float* sq = Ss + (qi * 4 * 16) * Sl;
        const int o = hh * Sl + lane, st = 16 * Sl;
        float sv = -INFINITY;
        if (lane < nc && posj[lane] <= qpos) {
          const float raw = __fadd_rn(__fadd_rn(__fadd_rn(sq[o], sq[st + o]), sq[2 * st + o]), sq[3 * st + o]);
          sv = __fmul_rn(__fmul_rn(raw, invr[lane * 4 + hh / (16 / MG)]), scale);
        }
        const float mx = warp_max(sv);
        const float m_ref = mrow[rowi];
        const bool bump = mx != -INFINITY && (m_ref == -INFINITY || mx > m_ref + 8.f);
        const float m_use = bump ? mx : m_ref;
        const float pr = (m_use == -INFINITY || sv == -INFINITY) ? 0.f : __expf(sv - m_use);
        Ps[(qi * 16 + hh) * Pl + lane] = __float2half(pr);
        const float l = warp_sum(pr);
        __syncwarp();
        if (lane == 0) {
          const float a = bump ? (m_ref == -INFINITY ? 0.f : __expf(m_ref - m_use)) : 1.f;
          arow[rowi] = a;
          lrow[rowi] = __fmaf_rn(lrow[rowi], a, l);
          mrow[rowi] = m_use;
          if (bump && m_ref != -INFINITY) fl[qi] = 1;
        }
      }
      __syncthreads();
#pragma unroll
      for (int qi = 0; qi < NQ; qi++) {
        const bool resc = fl[qi] != 0;
        const float a0 = arow[qi * 16 + ra], a1 = arow[qi * 16 + rb];
#pragma unroll
        for (int k = 0; k < OT; k++) {
          const int ct = warp + (AT / 32) * k;
          if (ct < nct) {
            if (resc) {
#pragma unroll
              for (int i = 0; i < O[qi][k].num_elements; i++) O[qi][k].x[i] *= (rowf.x[i] == (float)ra) ? a0 : a1;
            }
#pragma unroll
            for (int k0 = 0; k0 < AC; k0 += 16) {
              wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
              wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
              wmma::load_matrix_sync(fa, Ps + qi * 16 * Pl + k0, Pl);
              wmma::load_matrix_sync(fb, Ks + k0 * Es + RD + ct * 16, Es);
              wmma::mma_sync(O[qi][k], fa, fb, O[qi][k]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) {
      float* o = part_o + (((size_t)v * NQ + qi) * 16) * rp;
#pragma unroll
      for (int k = 0; k < OT; k++) {
        const int ct = warp + (AT / 32) * k;
        if (ct < nct) wmma::store_matrix_sync(o + ct * 16, O[qi][k], rp, wmma::mem_row_major);
      }
    }
    __syncthreads();
    for (int i = tid; i < NQ * 16; i += AT) {
      part_ml[((size_t)v * NQ * 16 + i) * 2] = mrow[i];
      part_ml[((size_t)v * NQ * 16 + i) * 2 + 1] = lrow[i];
    }
    __syncthreads();
  }
}

static int attn_b_smem(int Ep, int RW, int NQ) {
  const int Es = Ep + 8;
  return AC * Es * 2 + RD * (RD + 8) * 2 + NQ * 4 * 16 * (AC + 4) * 4 + NQ * 16 * (AC + 8) * 2 + 3 * NQ * 16 * 4 +
         AC * 4 + AC * 16 + 256 * 4 + 32 * 4 + 32 + AC * RW + 64;
}


// merge the NS partials (fixed order) -> o_lat (NQ, 16, r) fp32
__global__ void mla_reduce_k(const float* __restrict__ part_o, const float* __restrict__ part_ml, float* __restrict__ out,
                             int NS, int NQ, int r, int rp) {
  const int hq = blockIdx.x;                  // qi * 16 + h
  float M = -INFINITY;
  for (int s = 0; s < NS; s++) M = fmaxf(M, part_ml[((size_t)s * NQ * 16 + hq) * 2]);
  float L = 0.f;
  for (int s = 0; s < NS; s++) {
    const float m = part_ml[((size_t)s * NQ * 16 + hq) * 2];
    if (m != -INFINITY) L = __fmaf_rn(part_ml[((size_t)s * NQ * 16 + hq) * 2 + 1], __expf(m - M), L);
  }
  for (int e = threadIdx.x; e < r; e += blockDim.x) {
    float v = 0.f;
    for (int s = 0; s < NS; s++) {
      const float m = part_ml[((size_t)s * NQ * 16 + hq) * 2];
      if (m != -INFINITY) v = __fmaf_rn(part_o[((size_t)s * NQ * 16 + hq) * rp + e], __expf(m - M), v);
    }
    out[(size_t)hq * r + e] = L > 0.f ? __fdiv_rn(v, L) : 0.f;
  }
}


constexpr int AT2 = 512;
template <int NQ, int OT>
__global__ void __launch_bounds__(AT2) attn_c(const __half* __restrict__ Q, int Ep, const uint8_t* __restrict__ cache,
                                                  int RW, const __half* __restrict__ R0h, const float* __restrict__ cbl,
                                                  const float* __restrict__ cbr, const long* __restrict__ pos,
                                                  const float* __restrict__ inv_freq, float* __restrict__ part_o,
                                                  float* __restrict__ part_ml, int r, float scale, int VS) {
  extern __shared__ __align__(32) unsigned char smem[];
  const int Es = Ep + 8, rp = Ep - RD;
  constexpr int Sl = AC + 4, Pl = AC + 8, Ul = RD + 8, Rl = RD + 4;
  half* Ks = reinterpret_cast<half*>(smem);                 // [AC][Es]
  half* Uh = Ks + AC * Es;                                  // [RD][Ul]
  float* Ss = reinterpret_cast<float*>(Uh + RD * Ul);       // [NQ][4][16][Sl]; also [AC][Rl] for the rope GEMM
  half* Ps = reinterpret_cast<half*>(Ss + NQ * 4 * 16 * Sl);  // [NQ][16][Pl]
  float* mrow = reinterpret_cast<float*>(Ps + NQ * 16 * Pl);  // [NQ][16]
  float* lrow = mrow + NQ * 16;
  float* arow = lrow + NQ * 16;
  int* posj = reinterpret_cast<int*>(arow + NQ * 16);       // [AC]
  float* invr = reinterpret_cast<float*>(posj + AC);        // [AC][4]
  float* rmap = invr + AC * 4;                              // [256]
  float* cbs = rmap + 256;                                  // [32]
  int* flag = reinterpret_cast<int*>(cbs + 32);             // [2][4]
  uint8_t* rows = reinterpret_cast<uint8_t*>(flag + 8);     // [AC][RW]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, b = blockIdx.x;
  const long p0 = pos[0];
  const int ncell = (int)(p0 + NQ);
  const int nchunk = (ncell + AC - 1) / AC;
  const int nct = rp / 16, nks = Ep / 16;

  for (int i = tid; i < RD * RD; i += AT2) Uh[(i / RD) * Ul + i % RD] = R0h[i];
  if (tid < 16) cbs[tid] = cbl[tid];
  else if (tid < 32) cbs[tid] = cbr[tid - 16];
  for (int i = tid; i < 256; i += AT2) rmap[i] = (float)(i / 16);
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowf;
  wmma::load_matrix_sync(rowf, rmap, 16, wmma::mem_row_major);
  int ra = (int)rowf.x[0], rb = ra;
#pragma unroll
  for (int i = 0; i < rowf.num_elements; i++) if ((int)rowf.x[i] != ra) rb = (int)rowf.x[i];

  auto stage = [&](int chunk) {
    const int j0 = chunk * AC, nc = min(AC, ncell - j0);
    const uint8_t* src = cache + (size_t)j0 * RW;
    const int n16 = nc * RW / 16;
    for (int s2 = tid; s2 < n16; s2 += AT2) {
      const unsigned dst = (unsigned)__cvta_generic_to_shared(rows + 16 * s2);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src + 16 * s2));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  for (int v = b; v < VS; v += G) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O[NQ][OT];
#pragma unroll
    for (int i = 0; i < NQ; i++)
#pragma unroll
      for (int k = 0; k < OT; k++) wmma::fill_fragment(O[i][k], 0.f);
    for (int i = tid; i < NQ * 16; i += AT2) { mrow[i] = -INFINITY; lrow[i] = 0.f; arow[i] = 1.f; }
    if (tid < 8) flag[tid] = 0;
    __syncthreads();
    if (v < nchunk) stage(v);
    int it = 0;
    for (int chunk = v; chunk < nchunk; chunk += VS, it++) {
      const int j0 = chunk * AC, nc = min(AC, ncell - j0);
      int* fl = flag + (it & 1) * 4;
      asm volatile("cp.async.wait_all;\n" ::);
      __syncthreads();
      if (tid < 4) flag[((it + 1) & 1) * 4 + tid] = 0;
      // decode: one warp per cell; 8 values per 32-bit code word
      for (int jl = warp; jl < AC; jl += AT2 / 32) {
        half* kr = Ks + jl * Es;
        if (jl >= nc) {
          for (int w8 = lane; w8 < Ep / 8; w8 += 32) reinterpret_cast<uint4*>(kr)[w8] = make_uint4(0, 0, 0, 0);
          if (lane == 0) posj[jl] = 0x7fffffff;
          if (lane < 4) invr[jl * 4 + lane] = 0.f;
          continue;
        }
        const uint8_t* row = rows + jl * RW;
        if (lane == 0) posj[jl] = *reinterpret_cast<const int*>(row);
        if (lane < 4) invr[jl * 4 + lane] = reinterpret_cast<const float*>(row + 4)[lane];
        const float nl = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + 20)), 1e-12f);
        const float nr = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + 22)), 1e-12f);
        const uint32_t* rcw = reinterpret_cast<const uint32_t*>(row + 24);
        const uint32_t* lcw = reinterpret_cast<const uint32_t*>(row + 24 + RD / 2);
        for (int w8 = lane; w8 < Ep / 8; w8 += 32) {
          const bool lat = w8 >= RD / 8;
          const int e0 = lat ? 8 * w8 - RD : 8 * w8;
          const int n = lat ? r : RD;
          const uint32_t codes = e0 < n ? (lat ? lcw[e0 >> 3] : rcw[w8]) : 0u;
          const float nn = lat ? nl : nr;
          const float* cb = cbs + (lat ? 0 : 16);
          uint32_t o[4];
#pragma unroll
          for (int u = 0; u < 4; u++) {
            float v0 = cb[(codes >> (8 * u)) & 15] * nn, v1 = cb[(codes >> (8 * u + 4)) & 15] * nn;
            if (e0 + 2 * u >= n) v0 = 0.f;
            if (e0 + 2 * u + 1 >= n) v1 = 0.f;
            const __half2 hv = __floats2half2_rn(v0, v1);
            o[u] = *reinterpret_cast<const uint32_t*>(&hv);
          }
          *reinterpret_cast<uint4*>(kr + 8 * w8) = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
      __syncthreads();
      if (chunk + VS < nchunk) stage(chunk + VS);
      {
        const int ct = (warp & 7) / 4, mt = warp % 4;
        if (warp < 8) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll
        for (int k0 = 0; k0 < RD; k0 += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
          wmma::load_matrix_sync(fa, Ks + ct * 16 * Es + k0, Es);
          wmma::load_matrix_sync(fb, Uh + k0 * Ul + mt * 16, Ul);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Ss + ct * 16 * Rl + mt * 16, acc, Rl, wmma::mem_row_major);
        }
      }
      __syncthreads();
      for (int i = tid; i < AC * 32; i += AT2) {
        const int jl = i / 32, p = i % 32;
        if (jl >= nc) continue;
        const float x0 = bf(Ss[jl * Rl + p]), x1 = bf(Ss[jl * Rl + p + 32]);
        float sn, cs;
        sincos_acc(__fmul_rn((float)posj[jl], inv_freq[p]), &sn, &cs);
        const float c = bf(cs), s = bf(sn);
        Ks[jl * Es + p] = __float2half(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
        Ks[jl * Es + p + 32] = __float2half(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
      }
      __syncthreads();
      // S = Q K^T for every token: task = (token, quarter, column tile)
      for (int task = warp; task < NQ * 8; task += AT2 / 32) {
        const int qi = task >> 3, qq = (task & 7) >> 1, ct = task & 1;
        const int k_lo = nks * qq / 4, k_hi = nks * (qq + 1) / 4;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        const half* qb = Q + (size_t)qi * MH * Ep;
        for (int kt = k_lo; kt < k_hi; kt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
          wmma::load_matrix_sync(fa, qb + kt * 16, Ep);
          wmma::load_matrix_sync(fb, Ks + ct * 16 * Es + kt * 16, Es);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Ss + ((qi * 4 + qq) * 16) * Sl + ct * 16, acc, Sl, wmma::mem_row_major);
      }
      __syncthreads();
      // softmax rows (token, head): lane = cell
      for (int rowi = warp; rowi < NQ * 16; rowi += AT2 / 32) {
        const int qi = rowi >> 4, hh = rowi & 15;
        const long qpos = p0 + qi;
        const float* sq = Ss + (qi * 4 * 16) * Sl;
        const int o = hh * Sl + lane, st = 16 * Sl;
        float sv = -INFINITY;
        if (lane < nc && posj[lane] <= qpos) {
          const float raw = __fadd_rn(__fadd_rn(__fadd_rn(sq[o], sq[st + o]), sq[2 * st + o]), sq[3 * st + o]);
          sv = __fmul_rn(__fmul_rn(raw, invr[lane * 4 + hh / (16 / MG)]), scale);
        }
        const float mx = warp_max(sv);
        const float m_ref = mrow[rowi];
        const bool bump = mx != -INFINITY && (m_ref == -INFINITY || mx > m_ref + 8.f);
        const float m_use = bump ? mx : m_ref;
        const float pr = (m_use == -INFINITY || sv == -INFINITY) ? 0.f : __expf(sv - m_use);
        Ps[(qi * 16 + hh) * Pl + lane] = __float2half(pr);
        const float l = warp_sum(pr);
        __syncwarp();
        if (lane == 0) {
          const float a = bump ? (m_ref == -INFINITY ? 0.f : __expf(m_ref - m_use)) : 1.f;
          arow[rowi] = a;
          lrow[rowi] = __fmaf_rn(lrow[rowi], a, l);
          mrow[rowi] = m_use;
          if (bump && m_ref != -INFINITY) fl[qi] = 1;
        }
      }
      __syncthreads();
#pragma unroll
      for (int qi = 0; qi < NQ; qi++) {
        const bool resc = fl[qi] != 0;
        const float a0 = arow[qi * 16 + ra], a1 = arow[qi * 16 + rb];
#pragma unroll
        for (int k = 0; k < OT; k++) {
          const int ct = warp + (AT2 / 32) * k;
          if (ct < nct) {
            if (resc) {
#pragma unroll
              for (int i = 0; i < O[qi][k].num_elements; i++) O[qi][k].x[i] *= (rowf.x[i] == (float)ra) ? a0 : a1;
            }
#pragma unroll
            for (int k0 = 0; k0 < AC; k0 += 16) {
              wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
              wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
              wmma::load_matrix_sync(fa, Ps + qi * 16 * Pl + k0, Pl);
              wmma::load_matrix_sync(fb, Ks + k0 * Es + RD + ct * 16, Es);
              wmma::mma_sync(O[qi][k], fa, fb, O[qi][k]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) {
      float* o = part_o + (((size_t)v * NQ + qi) * 16) * rp;
#pragma unroll
      for (int k = 0; k < OT; k++) {
        const int ct = warp + (AT2 / 32) * k;
        if (ct < nct) wmma::store_matrix_sync(o + ct * 16, O[qi][k], rp, wmma::mem_row_major);
      }
    }
    __syncthreads();
    for (int i = tid; i < NQ * 16; i += AT2) {
      part_ml[((size_t)v * NQ * 16 + i) * 2] = mrow[i];
      part_ml[((size_t)v * NQ * 16 + i) * 2 + 1] = lrow[i];
    }
    __syncthreads();
  }
}

static int attn_c_smem(int Ep, int RW, int NQ) {
  const int Es = Ep + 8;
  return AC * Es * 2 + RD * (RD + 8) * 2 + NQ * 4 * 16 * (AC + 4) * 4 + NQ * 16 * (AC + 8) * 2 + 3 * NQ * 16 * 4 +
         AC * 4 + AC * 16 + 256 * 4 + 32 * 4 + 32 + AC * RW + 64;
}

template <int VAR>
void run_lab(torch::Tensor Q, int64_t Ep, torch::Tensor cache, int64_t RW, torch::Tensor R0h, torch::Tensor cbl,
             torch::Tensor cbr, torch::Tensor pos, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml,
             torch::Tensor out, int64_t r, double scale, int64_t NQ) {
  const int VS = part_ml.size(0), rp = Ep - RD, OT = (rp / 16 + 7) / 8;
  const int sm = VAR == 0 ? attn_a_smem(Ep, RW, NQ) : (VAR == 1 ? attn_b_smem(Ep, RW, NQ) : attn_c_smem(Ep, RW, NQ));
  int dev = 0, smem_sm = 0, nsm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int G = min(VS, max(1, nsm * min(max(1, smem_sm / (sm + 1024)), 2)));
#define L(KF, NN, TT) { auto kfn = KF<NN, TT>; cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  kfn<<<G, AT, sm, STREAM>>>(reinterpret_cast<const __half*>(Q.data_ptr()), Ep, cache.data_ptr<uint8_t>(), RW, \
     reinterpret_cast<const __half*>(R0h.data_ptr()), cbl.data_ptr<float>(), cbr.data_ptr<float>(), pos.data_ptr<long>(), \
     inv_freq.data_ptr<float>(), part_o.data_ptr<float>(), part_ml.data_ptr<float>(), r, (float)scale, VS); }
#define LO(KF, NN) if (OT <= 4) L(KF, NN, 4) else if (OT <= 5) L(KF, NN, 5) else L(KF, NN, 6)
  if (VAR == 0) { if (NQ == 1) LO(attn_a, 1) else LO(attn_a, 4) }
  else if (VAR == 1) { if (NQ == 1) LO(attn_b, 1) else LO(attn_b, 4) }
  else {
    const int OT2 = (rp / 16 + 15) / 16;
#define L2(NN, TT) { auto kfn = attn_c<NN, TT>; cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  kfn<<<G, AT2, sm, STREAM>>>(reinterpret_cast<const __half*>(Q.data_ptr()), Ep, cache.data_ptr<uint8_t>(), RW, \
     reinterpret_cast<const __half*>(R0h.data_ptr()), cbl.data_ptr<float>(), cbr.data_ptr<float>(), pos.data_ptr<long>(), \
     inv_freq.data_ptr<float>(), part_o.data_ptr<float>(), part_ml.data_ptr<float>(), r, (float)scale, VS); }
    if (NQ == 1) { if (OT2 <= 2) L2(1, 2) else L2(1, 3) } else { if (OT2 <= 2) L2(4, 2) else L2(4, 3) }
  }
  mla_reduce_k<<<NQ * 16, 128, 0, STREAM>>>(part_o.data_ptr<float>(), part_ml.data_ptr<float>(), out.data_ptr<float>(), VS, NQ, r, rp);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_a", &run_lab<0>); m.def("attn_b", &run_lab<1>); m.def("attn_c", &run_lab<2>); }
