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

template <int NQ, int OT>
__global__ void __launch_bounds__(AT) mla_attn_k(const __half* __restrict__ Q, int Ep, const uint8_t* __restrict__ cache,
                                                 int RW, const __half* __restrict__ R0h, const float* __restrict__ cbl,
                                                 const float* __restrict__ cbr, const long* __restrict__ pos,
                                                 const float* __restrict__ inv_freq, float* __restrict__ part_o,
                                                 float* __restrict__ part_ml, int r, float scale, long long* __restrict__ prof) {
  long long tp = clock64();
#define STAMP(k) if (prof && blockIdx.x == 0 && threadIdx.x == 0) { long long tn = clock64(); prof[k] += tn - tp; tp = tn; }
  extern __shared__ __align__(32) unsigned char smem[];
  const int Es = Ep + 8, rp = Ep - RD;                      // latent columns padded: rp = pad16(r)
  constexpr int Sl = AC + 4, Pl = AC + 8, Ul = RD + 8, Rl = RD + 4;
  half* Ks = reinterpret_cast<half*>(smem);                 // [AC][Es]
  half* Uh = Ks + AC * Es;                                  // [RD][Ul]   R0 (row-major)
  float* Ss = reinterpret_cast<float*>(Uh + RD * Ul);       // [4][16][Sl] score quarters; also [AC][Rl] rope GEMM out
  half* Ps = reinterpret_cast<half*>(Ss + 4 * 16 * Sl);     // [16][Pl]
  float* mrow = reinterpret_cast<float*>(Ps + 16 * Pl);     // [NQ][16]
  float* lrow = mrow + NQ * 16;
  float* arow = lrow + NQ * 16;
  int* posj = reinterpret_cast<int*>(arow + NQ * 16);       // [AC]
  float* invr = reinterpret_cast<float*>(posj + AC);        // [AC][4]
  float* rmap = invr + AC * 4;                              // [256] row index map for accumulator fragments
  float* cbs = rmap + 256;                                  // [32] centroids latent | rope
  int* flag = reinterpret_cast<int*>(cbs + 32);            // [2][4] rescale flags (chunk parity, query token)
  uint8_t* rows = reinterpret_cast<uint8_t*>(flag + 8);     // [AC][RW] (16-byte aligned: see host)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int NS = gridDim.x, b = blockIdx.x;
  const long p0 = pos[0];
  const int ncell = (int)(p0 + NQ);                          // cells [0, ncell) exist
  const int nchunk = (ncell + AC - 1) / AC;
  const int nct = rp / 16;                                   // latent column tiles

  for (int i = tid; i < RD * RD; i += AT) Uh[(i / RD) * Ul + i % RD] = R0h[i];
  if (tid < 16) cbs[tid] = cbl[tid];
  else if (tid < 32) cbs[tid] = cbr[tid - 16];
  for (int i = tid; i < NQ * 16; i += AT) { mrow[i] = -INFINITY; lrow[i] = 0.f; arow[i] = 1.f; }
  for (int i = tid; i < 256; i += AT) rmap[i] = (float)(i / 16);
  if (tid < 8) flag[tid] = 0;
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowf;
  wmma::load_matrix_sync(rowf, rmap, 16, wmma::mem_row_major);
  int ra = (int)rowf.x[0], rb = ra;
#pragma unroll
  for (int i = 0; i < rowf.num_elements; i++) if ((int)rowf.x[i] != ra) rb = (int)rowf.x[i];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> O[NQ][OT];
#pragma unroll
  for (int i = 0; i < NQ; i++)
#pragma unroll
    for (int k = 0; k < OT; k++) wmma::fill_fragment(O[i][k], 0.f);

  auto stage = [&](int chunk) {
    const int j0 = chunk * AC, nc = min(AC, ncell - j0);
    const uint8_t* src = cache + (size_t)j0 * RW;
    const int n16 = nc * RW / 16;
    for (int s = tid; s < n16; s += AT) {
      const unsigned dst = (unsigned)__cvta_generic_to_shared(rows + 16 * s);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src + 16 * s));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  if (b < nchunk) stage(b);
  int it = 0;
  for (int chunk = b; chunk < nchunk; chunk += NS, it++) {
    const int j0 = chunk * AC, nc = min(AC, ncell - j0);
    int* fl = flag + (it & 1) * 4;
    STAMP(0)
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();
    STAMP(1)
    if (tid < 4) flag[((it + 1) & 1) * 4 + tid] = 0;       // next chunk's flags (last read in the previous chunk)
    // decode: one warp per cell (4 cells per warp)
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
      const uint8_t* rc = row + 24;
      const uint8_t* lc = rc + RD / 2;
      // 8 values per lane-iteration from one 32-bit code word; cols [0, 64): rope, [64, Ep): latent (+ zero pad)
      for (int w8 = lane; w8 < Ep / 8; w8 += 32) {
        const bool lat = w8 >= RD / 8;
        const int e0 = lat ? 8 * w8 - RD : 8 * w8;
        const int n = lat ? r : RD;
        uint32_t codes = 0;
        if (e0 < n) {
          const uint8_t* cp = (lat ? lc : rc) + e0 / 2;
          codes = (uint32_t)cp[0] | ((uint32_t)cp[1] << 8) | ((uint32_t)cp[2] << 16) | ((uint32_t)cp[3] << 24);
        }
        const float nn = lat ? nl : nr;
        const int cbo = lat ? 0 : 16;
        uint32_t o[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int c0 = (codes >> (8 * u)) & 15, c1 = (codes >> (8 * u + 4)) & 15;
          const float v0 = e0 + 2 * u < n ? cbs[cbo + c0] * nn : 0.f;
          const float v1 = e0 + 2 * u + 1 < n ? cbs[cbo + c1] * nn : 0.f;
          o[u] = (uint32_t)__half_as_ushort(__float2half(v0)) | ((uint32_t)__half_as_ushort(__float2half(v1)) << 16);
        }
        *reinterpret_cast<uint4*>(kr + 8 * w8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
    __syncthreads();
    STAMP(2)
    if (chunk + NS < nchunk) stage(chunk + NS);
    // RoPE key un-rotation: [AC x 64] raw x R0 -> fp32 (8 tiles of 16 x 16, one per warp)
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
    STAMP(3)
    for (int i = tid; i < AC * 32; i += AT) {
      const int jl = i / 32, p = i % 32;
      if (jl >= nc) continue;
      const float x0 = bf(Ss[jl * Rl + p]), x1 = bf(Ss[jl * Rl + p + 32]);
      const float ang = __fmul_rn((float)posj[jl], inv_freq[p]);
      float sn, cs;
      sincos_acc(ang, &sn, &cs);
      const float c = bf(cs), s = bf(sn);
      Ks[jl * Es + p] = __float2half(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
      Ks[jl * Es + p + 32] = __float2half(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
    }
    __syncthreads();
    STAMP(4)
    const int nks = Ep / 16;
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) {
      const long qpos = p0 + qi;
      // S = Q K^T: warp = (quarter qq, column tile ct)
      {
        const int qq = warp >> 1, ct = warp & 1;
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
        wmma::store_matrix_sync(Ss + qq * 16 * Sl + ct * 16, acc, Sl, wmma::mem_row_major);
      }
      __syncthreads();
      STAMP(5)
      // softmax rows: warp per head (2 heads per warp), lane = cell
      for (int hh = warp; hh < 16; hh += AT / 32) {
        const int o = hh * Sl + lane, st = 16 * Sl;
        float s = -INFINITY;
        if (lane < nc && posj[lane] <= qpos) {
          const float raw = __fadd_rn(__fadd_rn(__fadd_rn(Ss[o], Ss[st + o]), Ss[2 * st + o]), Ss[3 * st + o]);
          s = __fmul_rn(__fmul_rn(raw, invr[lane * 4 + hh / (16 / MG)]), scale);
        }
        const float mx = warp_max(s);
        const float m_ref = mrow[qi * 16 + hh];
        const bool bump = mx != -INFINITY && (m_ref == -INFINITY || mx > m_ref + 8.f);
        const float m_use = bump ? mx : m_ref;
        const float pr = (m_use == -INFINITY || s == -INFINITY) ? 0.f : __expf(s - m_use);
        Ps[hh * Pl + lane] = __float2half(pr);
        const float l = warp_sum(pr);
        __syncwarp();
        if (lane == 0) {
          const float a = bump ? (m_ref == -INFINITY ? 0.f : __expf(m_ref - m_use)) : 1.f;
          arow[qi * 16 + hh] = a;
          lrow[qi * 16 + hh] = __fmaf_rn(lrow[qi * 16 + hh], a, l);
          mrow[qi * 16 + hh] = m_use;
          if (bump && m_ref != -INFINITY) fl[qi] = 1;
        }
      }
      __syncthreads();
      STAMP(6)
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
            wmma::load_matrix_sync(fa, Ps + k0, Pl);
            wmma::load_matrix_sync(fb, Ks + k0 * Es + RD + ct * 16, Es);
            wmma::mma_sync(O[qi][k], fa, fb, O[qi][k]);
          }
        }
      }
      STAMP(7)
    }
  }
  // partials: unnormalized O and (max, sum) per (token, head)
#pragma unroll
  for (int qi = 0; qi < NQ; qi++) {
    float* o = part_o + (((size_t)b * NQ + qi) * 16) * rp;
#pragma unroll
    for (int k = 0; k < OT; k++) {
      const int ct = warp + (AT / 32) * k;
      if (ct < nct) wmma::store_matrix_sync(o + ct * 16, O[qi][k], rp, wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int i = tid; i < NQ * 16; i += AT) {
    part_ml[((size_t)b * NQ * 16 + i) * 2] = mrow[i];
    part_ml[((size_t)b * NQ * 16 + i) * 2 + 1] = lrow[i];
  }
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

static int mla_smem(int Ep, int RW) {
  const int Es = Ep + 8;
  return AC * Es * 2 + RD * (RD + 8) * 2 + 4 * 16 * (AC + 4) * 4 + 16 * (AC + 8) * 2 + 3 * 4 * 16 * 4 + AC * 4 +
         AC * 16 + 256 * 4 + 32 * 4 + 32 + AC * RW + 64;
}

void mla_attn_lab(torch::Tensor Q, int64_t Ep, torch::Tensor cache, int64_t RW, torch::Tensor R0h, torch::Tensor cbl,
              torch::Tensor cbr, torch::Tensor pos, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml,
              torch::Tensor out, int64_t r, double scale, int64_t NQ, torch::Tensor prof) {
  const int NS = part_ml.size(0);
  const int rp = Ep - RD;
  const int OT = (rp / 16 + 7) / 8;
  const int sm = mla_smem(Ep, RW);
  TORCH_CHECK(RW % 16 == 0 && Ep % 16 == 0);
#define MA(NN, TT)                                                                                                 \
  {                                                                                                                \
    auto kfn = mla_attn_k<NN, TT>;                                                                                 \
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);                                   \
    kfn<<<NS, AT, sm, STREAM>>>(reinterpret_cast<const __half*>(Q.data_ptr()), Ep, cache.data_ptr<uint8_t>(), RW,   \
                                reinterpret_cast<const __half*>(R0h.data_ptr()), cbl.data_ptr<float>(),            \
                                cbr.data_ptr<float>(), pos.data_ptr<long>(), inv_freq.data_ptr<float>(),           \
                                part_o.data_ptr<float>(), part_ml.data_ptr<float>(), r, (float)scale, reinterpret_cast<long long*>(prof.data_ptr()));             \
  }
#define MA_OT(NN)                                       \
  if (OT <= 4) MA(NN, 4) else if (OT <= 5) MA(NN, 5) else if (OT <= 6) MA(NN, 6) else TORCH_CHECK(false, "latent too wide");
  switch (NQ) {
    case 1: MA_OT(1) break;
    case 2: MA_OT(2) break;
    case 3: MA_OT(3) break;
    case 4: MA_OT(4) break;
    default: TORCH_CHECK(false, "NQ <= 4");
  }
#undef MA_OT
#undef MA
  mla_reduce_k<<<NQ * 16, 128, 0, STREAM>>>(part_o.data_ptr<float>(), part_ml.data_ptr<float>(), out.data_ptr<float>(),
                                           NS, NQ, r, rp);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mla_attn_lab", &mla_attn_lab); }
