// HgRT kernels for Mercurius-1-4B decode (cairn/mercurius_engine.py). Every kernel takes M (or NQ) token rows: M = 1 is
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


// ---- flash-decoding attention from the codes (v3). 16 warps; chunks of 32 cells, virtual splits as above.
// K is never written to shared memory: every B fragment of the S = Q K^T and O += P C tensor-core MMAs (mma.sync
// m16n8k16, fp32 accumulation) is decoded straight from the 4-bit codes into registers (latent values: codebook x
// norm -> fp16 pair), once, and reused for all NQ query rows. Q is staged once per block and read with ldmatrix. Only
// the 64-dim RoPE key goes through a small shared tile (un-rotation by R0 needs a GEMM, then RoPE). Work split (fixed,
// independent of NQ, so every element is computed identically for any NQ): S task = (8-cell tile, key-width quarter)
// = 16 warps; PV: warp w owns latent dim tiles w, w + 16, ...
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}
constexpr int FTH = 512, FC = 32;

static int mla_fd_smem(int Ep, int RW, int NQ, bool dbuf) {
  const int Es = Ep + 8, RWs = RW + 16;
  return NQ * 16 * Es * 2 + 4 * NQ * 16 * (FC + 4) * 4 + NQ * 16 * (FC + 8) * 2 + FC * 72 * 2 +
         (dbuf ? 2 : 1) * FC * RWs + FC * 4 + FC * 16 + FC * 8 + 3 * NQ * 16 * 4 + 32 * 4 + 32 + 128;
}

template <int NQ, int OT>
__global__ void __launch_bounds__(FTH) mla_fd_k(const __half* __restrict__ Q, int Ep, const uint8_t* __restrict__ cache,
                                                int RW, const __half* __restrict__ R0h, const float* __restrict__ cbl,
                                                const float* __restrict__ cbr, const long* __restrict__ pos,
                                                const float* __restrict__ inv_freq, float* __restrict__ part_o,
                                                float* __restrict__ part_ml, int r, float scale, int VS, int dbuf, long long* __restrict__ prof) {
  long long tp = clock64();
#define STAMP(k) if (prof && blockIdx.x == 0 && threadIdx.x == 0) { long long tn = clock64(); prof[k] += tn - tp; tp = tn; }
  extern __shared__ __align__(128) unsigned char smem[];
  const int Es = Ep + 8, RWs = RW + 16, rp = Ep - RD;
  constexpr int SL = FC + 4, PL = FC + 8, KL = 72;
  half* Qs = reinterpret_cast<half*>(smem);                                  // [NQ*16][Es]
  float* Sp = reinterpret_cast<float*>(Qs + NQ * 16 * Es);                   // [4][NQ][16][SL]; also [FC][68] rope out
  half* Ps = reinterpret_cast<half*>(Sp + 4 * NQ * 16 * SL);                 // [NQ][16][PL]
  half* Kr = Ps + NQ * 16 * PL;                                              // [FC][KL] rope key (raw, then roped)
  uint8_t* rows = reinterpret_cast<uint8_t*>(Kr + FC * KL);                  // [nbuf][FC][RWs]
  int* posj = reinterpret_cast<int*>(rows + (dbuf ? 2 : 1) * FC * RWs);      // [FC]
  float* invr = reinterpret_cast<float*>(posj + FC);                         // [FC][4]
  float* nlj = invr + FC * 4;                                                // [FC]
  float* nrj = nlj + FC;                                                     // [FC]
  float* mrow = nrj + FC;                                                    // [NQ*16]
  float* lrow = mrow + NQ * 16;
  float* arow = lrow + NQ * 16;
  float* cbs = arow + NQ * 16;                                               // [32] latent | rope centroids
  int* flag = reinterpret_cast<int*>(cbs + 32);                              // [2][4]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int G = gridDim.x, b = blockIdx.x;
  const long p0 = pos[0];
  const int ncell = (int)(p0 + NQ), nchunk = (ncell + FC - 1) / FC;
  const int nks = Ep / 16, ndt = rp / 8;
  // Q (fp16, rows (qi, head)) -> shared, once
  for (int i = tid; i < NQ * 16 * (Ep / 8); i += FTH) {
    const int row = i / (Ep / 8), c8 = i % (Ep / 8);
    reinterpret_cast<uint4*>(Qs + row * Es)[c8] = reinterpret_cast<const uint4*>(Q + (size_t)row * Ep)[c8];
  }
  if (tid < 16) cbs[tid] = cbl[tid];
  else if (tid < 32) cbs[tid] = cbr[tid - 16];
  __syncthreads();
  const float* cbL = cbs;

  auto stage = [&](int chunk, int buf) {
    const int j0 = chunk * FC, nc = min(FC, ncell - j0);
    const uint8_t* src = cache + (size_t)j0 * RW;
    const int seg = RW / 16;
    uint8_t* dst0 = rows + buf * FC * RWs;
    for (int s2 = tid; s2 < nc * seg; s2 += FTH) {
      const int c = s2 / seg, sg = s2 % seg;
      const unsigned dst = (unsigned)__cvta_generic_to_shared(dst0 + c * RWs + 16 * sg);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src + (size_t)c * RW + 16 * sg));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  for (int v = b; v < VS; v += G) {
    float O[NQ][OT][4];
#pragma unroll
    for (int qi = 0; qi < NQ; qi++)
#pragma unroll
      for (int j = 0; j < OT; j++) O[qi][j][0] = O[qi][j][1] = O[qi][j][2] = O[qi][j][3] = 0.f;
    for (int i = tid; i < NQ * 16; i += FTH) { mrow[i] = -INFINITY; lrow[i] = 0.f; arow[i] = 1.f; }
    if (tid < 8) flag[tid] = 0;
    __syncthreads();
    if (v < nchunk) stage(v, 0);
    int it = 0;
    for (int chunk = v; chunk < nchunk; chunk += VS, it++) {
      const int j0 = chunk * FC, nc = min(FC, ncell - j0);
      const int cur = dbuf ? (it & 1) : 0;
      const uint8_t* rb = rows + cur * FC * RWs;
      int* fl = flag + (it & 1) * 4;
      asm volatile("cp.async.wait_all;\n" ::);
      STAMP(0)
      __syncthreads();                                                       // S1: rows in place
      STAMP(1)
      if (dbuf && chunk + VS < nchunk) stage(chunk + VS, cur ^ 1);
      if (tid < 4) flag[((it + 1) & 1) * 4 + tid] = 0;
      if (tid < FC) {
        const int c = tid;
        if (c < nc) {
          const uint8_t* row = rb + c * RWs;
          posj[c] = *reinterpret_cast<const int*>(row);
#pragma unroll
          for (int q = 0; q < 4; q++) invr[c * 4 + q] = reinterpret_cast<const float*>(row + 4)[q];
          nlj[c] = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + 20)), 1e-12f);
          nrj[c] = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + 22)), 1e-12f);
        } else {
          posj[c] = 0x7fffffff; nlj[c] = 0.f; nrj[c] = 0.f;
#pragma unroll
          for (int q = 0; q < 4; q++) invr[c * 4 + q] = 0.f;
        }
      }
      // raw RoPE key -> Kr (fp16): 32 cells x 32 code bytes
      for (int i = tid; i < FC * 32; i += FTH) {
        const int c = i >> 5, by = i & 31;
        uint32_t o = 0;
        if (c < nc) {
          const uint8_t* row = rb + c * RWs;
          const float nn = fmaxf(__half2float(*reinterpret_cast<const __half*>(row + 22)), 1e-12f);
          const uint8_t cd = row[24 + by];
          o = pack_h2(cbs[16 + (cd & 15)] * nn, cbs[16 + (cd >> 4)] * nn);
        }
        *reinterpret_cast<uint32_t*>(Kr + c * KL + 2 * by) = o;
      }
      __syncthreads();                                                       // S2
      STAMP(2)
      if (warp < 8) {                                                        // [32 x 64] x R0 -> fp32 (aliases Sp)
        const int ct = warp / 4, mt = warp % 4;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll
        for (int k0 = 0; k0 < RD; k0 += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
          wmma::load_matrix_sync(fa, Kr + ct * 16 * KL + k0, KL);
          wmma::load_matrix_sync(fb, R0h + k0 * RD + mt * 16, RD);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Sp + ct * 16 * 68 + mt * 16, acc, 68, wmma::mem_row_major);
      }
      __syncthreads();                                                       // S3
      STAMP(3)
      for (int i = tid; i < FC * 32; i += FTH) {
        const int c = i >> 5, p = i & 31;
        if (c >= nc) { Kr[c * KL + p] = __float2half(0.f); Kr[c * KL + p + 32] = __float2half(0.f); continue; }
        const float x0 = bf(Sp[c * 68 + p]), x1 = bf(Sp[c * 68 + p + 32]);
        float sn, cs;
        sincos_acc(__fmul_rn((float)posj[c], inv_freq[p]), &sn, &cs);
        const float cc = bf(cs), ss = bf(sn);
        Kr[c * KL + p] = __float2half(__fsub_rn(__fmul_rn(x0, cc), __fmul_rn(x1, ss)));
        Kr[c * KL + p + 32] = __float2half(__fadd_rn(__fmul_rn(x1, cc), __fmul_rn(x0, ss)));
      }
      __syncthreads();                                                       // S4
      STAMP(4)
      // S = Q K^T: task (8-cell tile nt, quarter kq); K fragments decoded in registers, all of the warp's k-steps
      // first (independent loads), then the MMAs
      {
        const int nt = warp & 3, kq = warp >> 2;
        const int k_lo = nks * kq / 4, k_hi = nks * (kq + 1) / 4;
        const int c = nt * 8 + g;
        const uint8_t* lc = rb + c * RWs + 24 + RD / 2;
        const float nl = nlj[c];
        constexpr int KQ = 12;                                                 // max k-steps per quarter (Ep <= 768)
        uint32_t fb0[KQ], fb1[KQ];
#pragma unroll
        for (int u = 0; u < KQ; u++) {
          const int ks = k_lo + u;
          fb0[u] = fb1[u] = 0u;
          if (ks < k_hi) {
            if (ks < RD / 16) {
              fb0[u] = *reinterpret_cast<const uint32_t*>(Kr + c * KL + 16 * ks + 2 * t);
              fb1[u] = *reinterpret_cast<const uint32_t*>(Kr + c * KL + 16 * ks + 8 + 2 * t);
            } else {
              const int e = 16 * ks - RD + 2 * t;
              const uint32_t c0 = e < r ? lc[e >> 1] : 0u, c1 = e + 8 < r ? lc[(e + 8) >> 1] : 0u;
              fb0[u] = pack_h2(e < r ? cbL[c0 & 15] * nl : 0.f, e + 1 < r ? cbL[c0 >> 4] * nl : 0.f);
              fb1[u] = pack_h2(e + 8 < r ? cbL[c1 & 15] * nl : 0.f, e + 9 < r ? cbL[c1 >> 4] * nl : 0.f);
            }
          }
        }
        float acc[NQ][4];
#pragma unroll
        for (int qi = 0; qi < NQ; qi++) acc[qi][0] = acc[qi][1] = acc[qi][2] = acc[qi][3] = 0.f;
#pragma unroll
        for (int u = 0; u < KQ; u++) {
          const int ks = k_lo + u;
          if (ks < k_hi) {
#pragma unroll
            for (int qi = 0; qi < NQ; qi++) {
              uint32_t a0, a1, a2, a3;
              const half* qa = Qs + (qi * 16 + (lane & 7) + 8 * ((lane >> 3) & 1)) * Es + 16 * ks + 8 * (lane >> 4);
              ldsm_x4(a0, a1, a2, a3, qa);
              mma16816(acc[qi], a0, a1, a2, a3, fb0[u], fb1[u]);
            }
          }
        }
#pragma unroll
        for (int qi = 0; qi < NQ; qi++) {
          float* sp = Sp + ((kq * NQ + qi) * 16) * SL + nt * 8 + 2 * t;
          sp[g * SL] = acc[qi][0]; sp[g * SL + 1] = acc[qi][1];
          sp[(g + 8) * SL] = acc[qi][2]; sp[(g + 8) * SL + 1] = acc[qi][3];
        }
      }
      __syncthreads();                                                       // S5
      STAMP(5)
      for (int rowi = warp; rowi < NQ * 16; rowi += FTH / 32) {
        const int qi = rowi >> 4, hh = rowi & 15;
        const long qpos = p0 + qi;
        const int st = NQ * 16 * SL, o = (qi * 16 + hh) * SL + lane;
        float sv = -INFINITY;
        if (lane < nc && posj[lane] <= qpos) {
          const float raw = __fadd_rn(__fadd_rn(__fadd_rn(Sp[o], Sp[st + o]), Sp[2 * st + o]), Sp[3 * st + o]);
          sv = __fmul_rn(__fmul_rn(raw, invr[lane * 4 + hh / (16 / MG)]), scale);
        }
        const float mx = warp_max(sv);
        const float m_ref = mrow[rowi];
        const bool bump = mx != -INFINITY && (m_ref == -INFINITY || mx > m_ref + 8.f);
        const float m_use = bump ? mx : m_ref;
        const float pr = (m_use == -INFINITY || sv == -INFINITY) ? 0.f : __expf(sv - m_use);
        Ps[(qi * 16 + hh) * PL + lane] = __float2half(pr);
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
      __syncthreads();                                                       // S6
      STAMP(6)
      // O += P C: warp owns latent dim tiles warp + 16 j; C fragments decoded in registers
#pragma unroll
      for (int qi = 0; qi < NQ; qi++) {
        if (fl[qi]) {
          const float ag = arow[qi * 16 + g], ag8 = arow[qi * 16 + g + 8];
#pragma unroll
          for (int j = 0; j < OT; j++) { O[qi][j][0] *= ag; O[qi][j][1] *= ag; O[qi][j][2] *= ag8; O[qi][j][3] *= ag8; }
        }
      }
#pragma unroll
      for (int kk = 0; kk < FC / 16; kk++) {
        const int c0 = 16 * kk + 2 * t;
        const uint8_t* r0p = rb + c0 * RWs + 24 + RD / 2;
        const float n0 = nlj[c0], n1 = nlj[c0 + 1], n8 = nlj[c0 + 8], n9 = nlj[c0 + 9];
        uint32_t fb0[OT], fb1[OT];                                           // all of the warp's tiles first
#pragma unroll
        for (int j = 0; j < OT; j++) {
          const int d = 8 * (warp + 16 * j) + g;
          float v0 = 0.f, v1 = 0.f, v8 = 0.f, v9 = 0.f;
          if (d < r) {
            const int by = d >> 1, sh = (d & 1) * 4;
            v0 = cbL[(r0p[by] >> sh) & 15] * n0;
            v1 = cbL[(r0p[RWs + by] >> sh) & 15] * n1;
            v8 = cbL[(r0p[8 * RWs + by] >> sh) & 15] * n8;
            v9 = cbL[(r0p[9 * RWs + by] >> sh) & 15] * n9;
          }
          fb0[j] = pack_h2(v0, v1);
          fb1[j] = pack_h2(v8, v9);
        }
        uint32_t pa[NQ][4];
#pragma unroll
        for (int qi = 0; qi < NQ; qi++)
          ldsm_x4(pa[qi][0], pa[qi][1], pa[qi][2], pa[qi][3],
                  Ps + (qi * 16 + (lane & 7) + 8 * ((lane >> 3) & 1)) * PL + 16 * kk + 8 * (lane >> 4));
#pragma unroll
        for (int j = 0; j < OT; j++) {
          if (warp + 16 * j < ndt) {
#pragma unroll
            for (int qi = 0; qi < NQ; qi++) mma16816(O[qi][j], pa[qi][0], pa[qi][1], pa[qi][2], pa[qi][3], fb0[j], fb1[j]);
          }
        }
      }
      STAMP(7)
      if (!dbuf) {
        __syncthreads();
        if (chunk + VS < nchunk) stage(chunk + VS, 0);
      }
    }
    // partial: unnormalized O (rows (qi, head), cols latent dims) and (max, sum)
#pragma unroll
    for (int qi = 0; qi < NQ; qi++)
#pragma unroll
      for (int j = 0; j < OT; j++) {
        const int dt = warp + 16 * j;
        if (dt < ndt) {
          float* o = part_o + (((size_t)v * NQ + qi) * 16) * rp + 8 * dt + 2 * t;
          o[g * rp] = O[qi][j][0]; o[g * rp + 1] = O[qi][j][1];
          o[(g + 8) * rp] = O[qi][j][2]; o[(g + 8) * rp + 1] = O[qi][j][3];
        }
      }
    __syncthreads();
    for (int i = tid; i < NQ * 16; i += FTH) {
      part_ml[((size_t)v * NQ * 16 + i) * 2] = mrow[i];
      part_ml[((size_t)v * NQ * 16 + i) * 2 + 1] = lrow[i];
    }
    __syncthreads();
  }
}

void mla_attn_fd(torch::Tensor Q, int64_t Ep, torch::Tensor cache, int64_t RW, torch::Tensor R0h, torch::Tensor cbl,
                 torch::Tensor cbr, torch::Tensor pos, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml,
                 torch::Tensor out, int64_t r, double scale, int64_t NQ, torch::Tensor prof) {
  const int VS = part_ml.size(0), rp = Ep - RD;
  const int ndt = rp / 8, OT = (ndt + 15) / 16;
  TORCH_CHECK(RW % 16 == 0 && Ep % 16 == 0 && OT <= 6);
  int dev = 0, smem_sm = 0, smem_blk = 0, nsm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  cudaDeviceGetAttribute(&smem_blk, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const bool dbuf = mla_fd_smem(Ep, RW, NQ, true) <= smem_blk;
  const int sm = mla_fd_smem(Ep, RW, NQ, dbuf);
  TORCH_CHECK(sm <= smem_blk, "mla_attn_fd: shared memory");
  const int per_sm = max(1, min(2, smem_sm / (sm + 1024)));
  const int G = min(VS, nsm * per_sm);
#define FD(NN, TT)                                                                                                 \
  {                                                                                                                \
    auto kfn = mla_fd_k<NN, TT>;                                                                                   \
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);                                   \
    kfn<<<G, FTH, sm, STREAM>>>(reinterpret_cast<const __half*>(Q.data_ptr()), Ep, cache.data_ptr<uint8_t>(), RW,  \
                                reinterpret_cast<const __half*>(R0h.data_ptr()), cbl.data_ptr<float>(),            \
                                cbr.data_ptr<float>(), pos.data_ptr<long>(), inv_freq.data_ptr<float>(),           \
                                part_o.data_ptr<float>(), part_ml.data_ptr<float>(), r, (float)scale, VS, dbuf, reinterpret_cast<long long*>(prof.data_ptr()));   \
  }
#define FD_OT(NN) if (OT <= 4) FD(NN, 4) else if (OT <= 5) FD(NN, 5) else FD(NN, 6)
  switch (NQ) {
    case 1: FD_OT(1) break;
    case 2: FD_OT(2) break;
    case 3: FD_OT(3) break;
    case 4: FD_OT(4) break;
    default: TORCH_CHECK(false, "NQ <= 4");
  }
#undef FD_OT
#undef FD
  mla_reduce_k<<<NQ * 16, 128, 0, STREAM>>>(part_o.data_ptr<float>(), part_ml.data_ptr<float>(), out.data_ptr<float>(),
                                           VS, NQ, r, rp);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fd", &mla_attn_fd); }
