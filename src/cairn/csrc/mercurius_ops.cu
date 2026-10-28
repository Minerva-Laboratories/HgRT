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

// =====================================================================================================================
// NF4 GEMV. W (N, K/32) uint4: 32 values per 16 bytes, byte b = elements 2b (low nibble), 2b + 1 (high nibble), i.e.
// 32-bit word j holds elements 8j..8j+7 in nibbles 0..7. S (N, K/64) fp32 = absmax / 16256. XQ (M, K) int8, XD (M, K/32).
__constant__ uint32_t c_nf4h[4], c_nf4l[4];

__device__ __forceinline__ void lut16(uint32_t w, uint32_t sel, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3,
                                      uint32_t& lo, uint32_t& hi) {
  uint32_t a = __byte_perm(t0, t1, w), b = __byte_perm(t2, t3, w);
  lo = __byte_perm(a, b, sel);
  const uint32_t w2 = w >> 16;
  a = __byte_perm(t0, t1, w2);
  b = __byte_perm(t2, t3, w2);
  hi = __byte_perm(a, b, sel >> 16);
}

constexpr int GW = 4;   // warps per block

// EXACT = false: only the h table (levels rounded to multiples of 1/127, |error| <= 3.9e-3): NOT the deployed weights,
// an IQ4_NL-like approximation of the NF4 grid kept for measuring what the exact lookup costs.
// S sub-rows per warp: 32 / S lanes stride over the groups of one row (S = 2: half-warps, for K = 2560 whose 80 groups
// do not divide by 32). The per-row summation order depends only on S, never on M, R or U, so a matrix keeps one S for
// every M (bit identity of the verification pass).
template <int M, int R, int U, bool F32OUT, bool EXACT, int S>
__global__ void __launch_bounds__(GW * 32) nf4_gemv_k(const uint4* __restrict__ W, const float* __restrict__ Sc,
                                                      const int8_t* __restrict__ XQ, const float* __restrict__ XD,
                                                      const __nv_bfloat16* __restrict__ res, int ldr,
                                                      void* __restrict__ Yv, int ldy, int N, int K) {
  constexpr int L = 32 / S;
  const int lane = threadIdx.x & 31, sub = lane / L, ls = lane % L;
  const int row0 = (blockIdx.x * GW + (threadIdx.x >> 5)) * R * S + sub * R;
  if (row0 - sub * R >= N) return;
  const int G = K >> 5, SB = K >> 6;
  const uint32_t h0 = c_nf4h[0], h1 = c_nf4h[1], h2 = c_nf4h[2], h3 = c_nf4h[3];
  const uint32_t l0 = c_nf4l[0], l1 = c_nf4l[1], l2 = c_nf4l[2], l3 = c_nf4l[3];
  float acc[M][R];
#pragma unroll
  for (int m = 0; m < M; m++)
#pragma unroll
    for (int r = 0; r < R; r++) acc[m][r] = 0.f;
  for (int g0 = ls; g0 < G; g0 += L * U) {
    uint4 p[U][R];
    float sc[U][R];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int g = g0 + L * u;
#pragma unroll
      for (int r = 0; r < R; r++) {
        const int row = min(row0 + r, N - 1);
        if (g < G) {
          p[u][r] = __ldg(W + (size_t)row * G + g);
          sc[u][r] = __ldg(Sc + (size_t)row * SB + (g >> 1));
        } else {
          p[u][r] = make_uint4(0, 0, 0, 0);
          sc[u][r] = 0.f;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int g = g0 + L * u;
      if (g >= G) break;
      int xw[M][8];
      float xd[M];
#pragma unroll
      for (int m = 0; m < M; m++) {
        const int4* xp = reinterpret_cast<const int4*>(XQ + (size_t)m * K + (size_t)g * 32);
        const int4 a = __ldg(xp), b = __ldg(xp + 1);
        xw[m][0] = a.x; xw[m][1] = a.y; xw[m][2] = a.z; xw[m][3] = a.w;
        xw[m][4] = b.x; xw[m][5] = b.y; xw[m][6] = b.z; xw[m][7] = b.w;
        xd[m] = __ldg(XD + (size_t)m * G + g);
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        const uint32_t wd[4] = {p[u][r].x, p[u][r].y, p[u][r].z, p[u][r].w};
        int dh[M], dl[M];
#pragma unroll
        for (int m = 0; m < M; m++) { dh[m] = 0; dl[m] = 0; }
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t w = wd[j];
          const uint32_t sel = 0x32103210u | ((w & 0x88888888u) >> 1);
          uint32_t hl, hh, ll = 0, lh = 0;
          lut16(w, sel, h0, h1, h2, h3, hl, hh);
          if (EXACT) lut16(w, sel, l0, l1, l2, l3, ll, lh);
#pragma unroll
          for (int m = 0; m < M; m++) {
            dh[m] = __dp4a((int)hl, xw[m][2 * j], dh[m]);
            dh[m] = __dp4a((int)hh, xw[m][2 * j + 1], dh[m]);
            if (EXACT) {
              dl[m] = __dp4a((int)ll, xw[m][2 * j], dl[m]);
              dl[m] = __dp4a((int)lh, xw[m][2 * j + 1], dl[m]);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < M; m++)
          acc[m][r] = __fmaf_rn(__fmul_rn(sc[u][r], xd[m]), (float)(EXACT ? dh[m] * 128 + dl[m] : dh[m] * 128), acc[m][r]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < M; m++)
#pragma unroll
    for (int r = 0; r < R; r++) {
      float v = acc[m][r];
#pragma unroll
      for (int o = L / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
      const int row = row0 + r;
      if (ls == 0 && row < N) {
        if (F32OUT) {
          reinterpret_cast<float*>(Yv)[(size_t)m * ldy + row] = v;
        } else {
          const float o = res ? __fadd_rn(v, __bfloat162float(res[(size_t)m * ldr + row])) : v;
          reinterpret_cast<__nv_bfloat16*>(Yv)[(size_t)m * ldy + row] = __float2bfloat16(o);
        }
      }
    }
}

template <int M, int R, int U, bool F, bool X>
static void nf4_launch(const torch::Tensor& codes, const torch::Tensor& scales, const torch::Tensor& xq,
                       const torch::Tensor& xd, const __nv_bfloat16* res, int ldr, void* y, int ldy, int S) {
  const int N = codes.size(0), K = codes.size(1) * 32;
  dim3 grid((N + GW * R * S - 1) / (GW * R * S));
  if (S == 2)
    nf4_gemv_k<M, R, U, F, X, 2><<<grid, GW * 32, 0, STREAM>>>(
        reinterpret_cast<const uint4*>(codes.data_ptr()), scales.data_ptr<float>(), xq.data_ptr<int8_t>(),
        xd.data_ptr<float>(), res, ldr, y, ldy, N, K);
  else
    nf4_gemv_k<M, R, U, F, X, 1><<<grid, GW * 32, 0, STREAM>>>(
        reinterpret_cast<const uint4*>(codes.data_ptr()), scales.data_ptr<float>(), xq.data_ptr<int8_t>(),
        xd.data_ptr<float>(), res, ldr, y, ldy, N, K);
}

template <int M, bool F, bool X>
static void nf4_dispatch_r(int code, const torch::Tensor& c, const torch::Tensor& s, const torch::Tensor& xq,
                           const torch::Tensor& xd, const __nv_bfloat16* res, int ldr, void* y, int ldy) {
  const int S = (code / 1000) == 1 ? 2 : 1, rows = code % 100;
  switch (rows) {   // R * 10 + U
    case 21: nf4_launch<M, 2, 1, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    case 22: nf4_launch<M, 2, 2, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    case 41: nf4_launch<M, 4, 1, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    case 42: nf4_launch<M, 4, 2, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    case 81: nf4_launch<M, 8, 1, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    case 12: nf4_launch<M, 1, 2, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    case 14: nf4_launch<M, 1, 4, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    case 24: nf4_launch<M, 2, 4, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
    default: nf4_launch<M, 4, 1, F, X>(c, s, xq, xd, res, ldr, y, ldy, S); break;
  }
}

// y (M rows, stride ldy) = W x (+ res); out_f32: fp32 logits (no residual). rows = R*10+U, + 100: the approximate
// one-table variant (not the deployed weights; see nf4_gemv_k), + 1000: half-warp rows (S = 2)
void nf4_gemv(torch::Tensor codes, torch::Tensor scales, torch::Tensor xq, torch::Tensor xd,
              c10::optional<torch::Tensor> res, int64_t ldr, torch::Tensor y, int64_t ldy, int64_t M, int64_t rows) {
  const bool f32 = y.scalar_type() == torch::kFloat32;
  const bool exact = (rows % 1000) < 100;
  const __nv_bfloat16* r = res ? reinterpret_cast<const __nv_bfloat16*>(res->data_ptr()) : nullptr;
  void* yp = y.data_ptr();
#define NF4_CASE(MM)                                                                                  \
  case MM:                                                                                            \
    if (exact) {                                                                                      \
      if (f32) nf4_dispatch_r<MM, true, true>(rows - (rows % 1000) + (rows % 100), codes, scales, xq, xd, r, ldr, yp, ldy);    \
      else nf4_dispatch_r<MM, false, true>(rows - (rows % 1000) + (rows % 100), codes, scales, xq, xd, r, ldr, yp, ldy);       \
    } else {                                                                                          \
      if (f32) nf4_dispatch_r<MM, true, false>(rows - (rows % 1000) + (rows % 100), codes, scales, xq, xd, r, ldr, yp, ldy);   \
      else nf4_dispatch_r<MM, false, false>(rows - (rows % 1000) + (rows % 100), codes, scales, xq, xd, r, ldr, yp, ldy);      \
    }                                                                                                 \
    break;
  switch (M) { NF4_CASE(1) NF4_CASE(2) NF4_CASE(3) NF4_CASE(4) default: TORCH_CHECK(false, "M <= 4"); }
#undef NF4_CASE
}

void nf4_set_tables(torch::Tensor hi, torch::Tensor lo) {   // int8 (16,) each, host tensors
  TORCH_CHECK(hi.numel() == 16 && lo.numel() == 16 && hi.device().is_cpu());
  cudaMemcpyToSymbol(c_nf4h, hi.data_ptr(), 16);
  cudaMemcpyToSymbol(c_nf4l, lo.data_ptr(), 16);
}

// =====================================================================================================================
// NF4 GEMM on int8 tensor cores (mma.m16n8k32.s8): the same table decode (h, l int8 levels), then two MMAs per 32-group
// (h and l parts) against up to 8 token columns; the int32 group sums equal the dp4a kernel's exactly. A warp owns 16
// rows over a K slice (KS warps per block split K; partial tiles summed in warp order): every (row, token) value is
// accumulated in the same order whatever the number of tokens M <= 8, so one kernel serves plain decode and
// verification bit-identically.
// Code layout (nf4_mma_layout): per row, per 128 values ("super"), 4 x uint4: thread t's uint4 = for group j = 0..3 the
// 32-bit word [values 32j + 4t .. +3 | values 32j + 16 + 4t .. +3] (nibbles, low first) = its A fragment's codes.
__device__ __forceinline__ void mma_s8(int& d0, int& d1, int& d2, int& d3, uint32_t a0, uint32_t a1, uint32_t a2,
                                       uint32_t a3, int b0, int b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}

template <bool F32OUT, int KS>
__global__ void __launch_bounds__(KS * 32) nf4_mma_k(const uint4* __restrict__ W, const float* __restrict__ Sc,
                                                     const int8_t* __restrict__ XQ, const float* __restrict__ XD,
                                                     const __nv_bfloat16* __restrict__ res, int ldr, void* __restrict__ Yv,
                                                     int ldy, int N, int K, int M) {
  __shared__ float part[KS][16][9];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * 16;
  const int nsup = K >> 7, G = K >> 5, SB = K >> 6;
  const int s_lo = nsup * warp / KS, s_hi = nsup * (warp + 1) / KS;
  const int ra = min(row0 + g, N - 1), rb = min(row0 + g + 8, N - 1);
  const uint32_t h0 = c_nf4h[0], h1 = c_nf4h[1], h2 = c_nf4h[2], h3 = c_nf4h[3];
  const uint32_t l0 = c_nf4l[0], l1 = c_nf4l[1], l2 = c_nf4l[2], l3 = c_nf4l[3];
  const bool bval = g < M, x0v = 2 * t < M, x1v = 2 * t + 1 < M;
  const int8_t* xrow = XQ + (size_t)(bval ? g : 0) * K + 4 * t;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (int s = s_lo; s < s_hi; s++) {
    const uint4 wa = __ldg(W + ((size_t)ra * nsup + s) * 4 + t);
    const uint4 wb = __ldg(W + ((size_t)rb * nsup + s) * 4 + t);
    const float2 sa = __ldg(reinterpret_cast<const float2*>(Sc + (size_t)ra * SB + 2 * s));
    const float2 sb = __ldg(reinterpret_cast<const float2*>(Sc + (size_t)rb * SB + 2 * s));
    const uint32_t wav[4] = {wa.x, wa.y, wa.z, wa.w}, wbv[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int gi = 4 * s + j;
      const uint32_t w1 = wav[j], w2 = wbv[j];
      const uint32_t sel1 = 0x32103210u | ((w1 & 0x88888888u) >> 1), sel2 = 0x32103210u | ((w2 & 0x88888888u) >> 1);
      uint32_t ha_lo, ha_hi, hb_lo, hb_hi, la_lo, la_hi, lb_lo, lb_hi;
      lut16(w1, sel1, h0, h1, h2, h3, ha_lo, ha_hi);
      lut16(w2, sel2, h0, h1, h2, h3, hb_lo, hb_hi);
      lut16(w1, sel1, l0, l1, l2, l3, la_lo, la_hi);
      lut16(w2, sel2, l0, l1, l2, l3, lb_lo, lb_hi);
      const int b0 = bval ? __ldg(reinterpret_cast<const int*>(xrow + gi * 32)) : 0;
      const int b1 = bval ? __ldg(reinterpret_cast<const int*>(xrow + gi * 32 + 16)) : 0;
      int dh0, dh1, dh2, dh3, dl0, dl1, dl2, dl3;
      mma_s8(dh0, dh1, dh2, dh3, ha_lo, hb_lo, ha_hi, hb_hi, b0, b1);
      mma_s8(dl0, dl1, dl2, dl3, la_lo, lb_lo, la_hi, lb_hi, b0, b1);
      const float xd0 = x0v ? __ldg(XD + (size_t)(2 * t) * G + gi) : 0.f;
      const float xd1 = x1v ? __ldg(XD + (size_t)(2 * t + 1) * G + gi) : 0.f;
      const float swa = (j < 2) ? sa.x : sa.y, swb = (j < 2) ? sb.x : sb.y;
      acc0 = __fmaf_rn(__fmul_rn(swa, xd0), (float)(dh0 * 128 + dl0), acc0);
      acc1 = __fmaf_rn(__fmul_rn(swa, xd1), (float)(dh1 * 128 + dl1), acc1);
      acc2 = __fmaf_rn(__fmul_rn(swb, xd0), (float)(dh2 * 128 + dl2), acc2);
      acc3 = __fmaf_rn(__fmul_rn(swb, xd1), (float)(dh3 * 128 + dl3), acc3);
    }
  }
  part[warp][g][2 * t] = acc0;
  part[warp][g][2 * t + 1] = acc1;
  part[warp][g + 8][2 * t] = acc2;
  part[warp][g + 8][2 * t + 1] = acc3;
  __syncthreads();
  for (int i = threadIdx.x; i < 128; i += KS * 32) {
    const int r = i >> 3, n = i & 7, row = row0 + r;
    if (n >= M || row >= N) continue;
    float v = part[0][r][n];
#pragma unroll
    for (int w = 1; w < KS; w++) v = __fadd_rn(v, part[w][r][n]);
    if (F32OUT) {
      reinterpret_cast<float*>(Yv)[(size_t)n * ldy + row] = v;
    } else {
      const float o = res ? __fadd_rn(v, __bfloat162float(res[(size_t)n * ldr + row])) : v;
      reinterpret_cast<__nv_bfloat16*>(Yv)[(size_t)n * ldy + row] = __float2bfloat16(o);
    }
  }
}

// v2: 32 rows per warp (two 16-row tiles share every activation fragment), next 128-value block's weights prefetched
template <bool F32OUT, int KS>
__global__ void __launch_bounds__(KS * 32) nf4_mma2_k(const uint4* __restrict__ W, const float* __restrict__ Sc,
                                                      const int8_t* __restrict__ XQ, const float* __restrict__ XD,
                                                      const __nv_bfloat16* __restrict__ res, int ldr, void* __restrict__ Yv,
                                                      int ldy, int N, int K, int M) {
  __shared__ float part[KS][32][9];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * 32;
  const int nsup = K >> 7, G = K >> 5, SB = K >> 6;
  const int s_lo = nsup * warp / KS, s_hi = nsup * (warp + 1) / KS;
  int rr[4];
#pragma unroll
  for (int q = 0; q < 4; q++) rr[q] = min(row0 + 8 * q + g, N - 1);     // rows g, g+8, g+16, g+24
  const uint32_t h0 = c_nf4h[0], h1 = c_nf4h[1], h2 = c_nf4h[2], h3 = c_nf4h[3];
  const uint32_t l0 = c_nf4l[0], l1 = c_nf4l[1], l2 = c_nf4l[2], l3 = c_nf4l[3];
  const bool bval = g < M, x0v = 2 * t < M, x1v = 2 * t + 1 < M;
  const int8_t* xrow = XQ + (size_t)(bval ? g : 0) * K + 4 * t;
  float acc[4][2];
#pragma unroll
  for (int q = 0; q < 4; q++) acc[q][0] = acc[q][1] = 0.f;
  uint4 wc[4];
  float2 sc[4];
  if (s_lo < s_hi) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      wc[q] = __ldg(W + ((size_t)rr[q] * nsup + s_lo) * 4 + t);
      sc[q] = __ldg(reinterpret_cast<const float2*>(Sc + (size_t)rr[q] * SB + 2 * s_lo));
    }
  }
  for (int s = s_lo; s < s_hi; s++) {
    uint4 wn[4];
    float2 sn[4];
    const int sp = s + 1 < s_hi ? s + 1 : s;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      wn[q] = __ldg(W + ((size_t)rr[q] * nsup + sp) * 4 + t);
      sn[q] = __ldg(reinterpret_cast<const float2*>(Sc + (size_t)rr[q] * SB + 2 * sp));
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int gi = 4 * s + j;
      const int b0 = bval ? __ldg(reinterpret_cast<const int*>(xrow + gi * 32)) : 0;
      const int b1 = bval ? __ldg(reinterpret_cast<const int*>(xrow + gi * 32 + 16)) : 0;
      const float xd0 = x0v ? __ldg(XD + (size_t)(2 * t) * G + gi) : 0.f;
      const float xd1 = x1v ? __ldg(XD + (size_t)(2 * t + 1) * G + gi) : 0.f;
#pragma unroll
      for (int tile = 0; tile < 2; tile++) {
        const uint4 wa4 = wc[2 * tile], wb4 = wc[2 * tile + 1];
        const uint32_t w1 = j == 0 ? wa4.x : j == 1 ? wa4.y : j == 2 ? wa4.z : wa4.w;
        const uint32_t w2 = j == 0 ? wb4.x : j == 1 ? wb4.y : j == 2 ? wb4.z : wb4.w;
        const uint32_t sel1 = 0x32103210u | ((w1 & 0x88888888u) >> 1), sel2 = 0x32103210u | ((w2 & 0x88888888u) >> 1);
        uint32_t ha_lo, ha_hi, hb_lo, hb_hi, la_lo, la_hi, lb_lo, lb_hi;
        lut16(w1, sel1, h0, h1, h2, h3, ha_lo, ha_hi);
        lut16(w2, sel2, h0, h1, h2, h3, hb_lo, hb_hi);
        lut16(w1, sel1, l0, l1, l2, l3, la_lo, la_hi);
        lut16(w2, sel2, l0, l1, l2, l3, lb_lo, lb_hi);
        int dh0, dh1, dh2, dh3, dl0, dl1, dl2, dl3;
        mma_s8(dh0, dh1, dh2, dh3, ha_lo, hb_lo, ha_hi, hb_hi, b0, b1);
        mma_s8(dl0, dl1, dl2, dl3, la_lo, lb_lo, la_hi, lb_hi, b0, b1);
        const float swa = (j < 2) ? sc[2 * tile].x : sc[2 * tile].y, swb = (j < 2) ? sc[2 * tile + 1].x : sc[2 * tile + 1].y;
        acc[2 * tile][0] = __fmaf_rn(__fmul_rn(swa, xd0), (float)(dh0 * 128 + dl0), acc[2 * tile][0]);
        acc[2 * tile][1] = __fmaf_rn(__fmul_rn(swa, xd1), (float)(dh1 * 128 + dl1), acc[2 * tile][1]);
        acc[2 * tile + 1][0] = __fmaf_rn(__fmul_rn(swb, xd0), (float)(dh2 * 128 + dl2), acc[2 * tile + 1][0]);
        acc[2 * tile + 1][1] = __fmaf_rn(__fmul_rn(swb, xd1), (float)(dh3 * 128 + dl3), acc[2 * tile + 1][1]);
      }
    }
#pragma unroll
    for (int q = 0; q < 4; q++) { wc[q] = wn[q]; sc[q] = sn[q]; }
  }
#pragma unroll
  for (int q = 0; q < 4; q++) {
    part[warp][8 * q + g][2 * t] = acc[q][0];
    part[warp][8 * q + g][2 * t + 1] = acc[q][1];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += KS * 32) {
    const int r = i >> 3, n = i & 7, row = row0 + r;
    if (n >= M || row >= N) continue;
    float v = part[0][r][n];
#pragma unroll
    for (int w = 1; w < KS; w++) v = __fadd_rn(v, part[w][r][n]);
    if (F32OUT) {
      reinterpret_cast<float*>(Yv)[(size_t)n * ldy + row] = v;
    } else {
      const float o = res ? __fadd_rn(v, __bfloat162float(res[(size_t)n * ldr + row])) : v;
      reinterpret_cast<__nv_bfloat16*>(Yv)[(size_t)n * ldy + row] = __float2bfloat16(o);
    }
  }
}

void nf4_mma(torch::Tensor codes, torch::Tensor scales, torch::Tensor xq, torch::Tensor xd,
             c10::optional<torch::Tensor> res, int64_t ldr, torch::Tensor y, int64_t ldy, int64_t M, int64_t ks) {
  const int N = codes.size(0), K = codes.size(1) * 128;          // codes (N, K/128, 4, 4) int32
  TORCH_CHECK(M <= 8 && K % 128 == 0);
  const bool f32 = y.scalar_type() == torch::kFloat32;
  const __nv_bfloat16* r = res ? reinterpret_cast<const __nv_bfloat16*>(res->data_ptr()) : nullptr;
  const uint4* W = reinterpret_cast<const uint4*>(codes.data_ptr());
  dim3 grid((N + 15) / 16);
#define NM(KK)                                                                                                    \
  case KK:                                                                                                        \
    if (f32) nf4_mma_k<true, KK><<<grid, KK * 32, 0, STREAM>>>(W, scales.data_ptr<float>(), xq.data_ptr<int8_t>(),  \
                                                               xd.data_ptr<float>(), r, ldr, y.data_ptr(), ldy, N, K, M); \
    else nf4_mma_k<false, KK><<<grid, KK * 32, 0, STREAM>>>(W, scales.data_ptr<float>(), xq.data_ptr<int8_t>(),     \
                                                             xd.data_ptr<float>(), r, ldr, y.data_ptr(), ldy, N, K, M); \
    break;
  if (ks >= 100) {                                   // v2 (32 rows per warp): ks + 100
    dim3 grid2((N + 31) / 32);
#define NM2(KK)                                                                                                   \
  case KK:                                                                                                        \
    if (f32) nf4_mma2_k<true, KK><<<grid2, KK * 32, 0, STREAM>>>(W, scales.data_ptr<float>(), xq.data_ptr<int8_t>(), \
                                                                 xd.data_ptr<float>(), r, ldr, y.data_ptr(), ldy, N, K, M); \
    else nf4_mma2_k<false, KK><<<grid2, KK * 32, 0, STREAM>>>(W, scales.data_ptr<float>(), xq.data_ptr<int8_t>(),    \
                                                               xd.data_ptr<float>(), r, ldr, y.data_ptr(), ldy, N, K, M); \
    break;
    switch (ks - 100) { NM2(1) NM2(2) NM2(4) NM2(8) default: TORCH_CHECK(false, "ks in 1, 2, 4, 8"); }
#undef NM2
    return;
  }
  switch (ks) { NM(1) NM(2) NM(4) NM(8) default: TORCH_CHECK(false, "ks in 1, 2, 4, 8"); }
#undef NM
}

// =====================================================================================================================
// RMSNorm (+ int8 groups) over M rows: y = bf16((x * rsqrt(mean x^2 + eps)) * gain), gain a vector (n) or a scalar (1).
template <bool Q>
__global__ void rms_rows_k(const __nv_bfloat16* __restrict__ x, int ldx, const float* __restrict__ gain, int gn,
                           __nv_bfloat16* __restrict__ y, int ldy, int8_t* __restrict__ XQ, float* __restrict__ XD,
                           int n, float eps) {
  __shared__ float sh[32];
  const int m = blockIdx.x;
  const __nv_bfloat16* xr = x + (size_t)m * ldx;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += 1024) { const float v = __bfloat162float(xr[i]); ss = __fmaf_rn(v, v, ss); }
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum<1024>(ss, sh), (float)n), eps));
  const int lane = threadIdx.x & 31;
  for (int g = threadIdx.x >> 5; g < n / 32; g += 32) {
    const int i = g * 32 + lane;
    const float gg = gn == 1 ? gain[0] : gain[i];
    const __nv_bfloat16 yb = __float2bfloat16(__fmul_rn(__fmul_rn(__bfloat162float(xr[i]), r), gg));
    if (y) y[(size_t)m * ldy + i] = yb;
    if (Q) warp_quant32(__bfloat162float(yb), m * (n / 32) + g, lane, XQ, XD);
  }
}

void rms_rows(torch::Tensor x, int64_t ldx, torch::Tensor gain, c10::optional<torch::Tensor> y, int64_t ldy,
              c10::optional<torch::Tensor> xq, c10::optional<torch::Tensor> xd, int64_t M, int64_t n, double eps) {
  __nv_bfloat16* yp = y ? BF(*y) : nullptr;
  if (xq)
    rms_rows_k<true><<<M, 1024, 0, STREAM>>>(CBF(x), ldx, gain.data_ptr<float>(), gain.numel(), yp, ldy,
                                             xq->data_ptr<int8_t>(), xd->data_ptr<float>(), n, (float)eps);
  else
    rms_rows_k<false><<<M, 1024, 0, STREAM>>>(CBF(x), ldx, gain.data_ptr<float>(), gain.numel(), yp, ldy, nullptr,
                                              nullptr, n, (float)eps);
}

// plain int8 groups of bf16 rows (M, n), stride ldx
__global__ void quant_rows_k(const __nv_bfloat16* __restrict__ x, int ldx, int8_t* XQ, float* XD, int n, int M) {
  const int gidx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const int ng = n / 32;
  if (gidx >= ng * M) return;
  const int m = gidx / ng, g = gidx % ng;
  warp_quant32(__bfloat162float(x[(size_t)m * ldx + g * 32 + lane]), gidx, lane, XQ, XD);
}
void quant_rows(torch::Tensor x, int64_t ldx, torch::Tensor xq, torch::Tensor xd, int64_t M, int64_t n) {
  const int thr = (int)(M * n);
  quant_rows_k<<<(thr + 255) / 256, 256, 0, STREAM>>>(CBF(x), ldx, xq.data_ptr<int8_t>(), xd.data_ptr<float>(), n, M);
}

// MLP activation (gate | up in one GEMV row of 2n): act = bf16(bf16(silu(g)) * u) + int8 groups
__global__ void silu_mul_q_k(const __nv_bfloat16* __restrict__ gu, int ldgu, __nv_bfloat16* __restrict__ y,
                             int8_t* XQ, float* XD, int n, int M) {
  const int gidx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const int ng = n / 32;
  if (gidx >= ng * M) return;
  const int m = gidx / ng, i = (gidx % ng) * 32 + lane;
  const float a = __bfloat162float(gu[(size_t)m * ldgu + i]), u = __bfloat162float(gu[(size_t)m * ldgu + n + i]);
  const float s = bf(__fdiv_rn(a, __fadd_rn(1.f, __expf(-a))));
  const __nv_bfloat16 yb = __float2bfloat16(__fmul_rn(s, u));
  if (y) y[(size_t)m * n + i] = yb;
  warp_quant32(__bfloat162float(yb), gidx, lane, XQ, XD);
}
void silu_mul_q(torch::Tensor gu, int64_t ldgu, c10::optional<torch::Tensor> y, torch::Tensor xq, torch::Tensor xd,
                int64_t M, int64_t n) {
  const int thr = (int)(M * n);
  silu_mul_q_k<<<(thr + 255) / 256, 256, 0, STREAM>>>(CBF(gu), ldgu, y ? BF(*y) : nullptr, xq.data_ptr<int8_t>(),
                                                      xd.data_ptr<float>(), n, M);
}

// =====================================================================================================================
// small dense bf16 GEMV for M rows: y[m][i] = sum_j W[i][j] x[m][j] (fp32 accumulation, ascending j per lane, warp
// tree). W (N, K) bf16 row stride ldw; x bf16 rows (stride ldx). Out: fp32 (optionally rounded to bf16 values) or bf16.
template <int M, bool VEC, int R>
__global__ void __launch_bounds__(256) dgemv_k(const __nv_bfloat16* __restrict__ W, int ldw,
                                               const __nv_bfloat16* __restrict__ x, int ldx, float* __restrict__ yf,
                                               __nv_bfloat16* __restrict__ yb, int ldy, int N, int K, bool round) {
  const int lane = threadIdx.x & 31, row0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) acc[r][m] = 0.f;
  if (VEC) {                                  // K % 8 == 0, 16-byte aligned rows
    for (int j = lane * 8; j < K; j += 256) {
      uint4 wu[R];
#pragma unroll
      for (int r = 0; r < R; r++) wu[r] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)min(row0 + r, N - 1) * ldw + j));
#pragma unroll
      for (int m = 0; m < M; m++) {
        const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * ldx + j));
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xu);
#pragma unroll
        for (int r = 0; r < R; r++) {
          const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wu[r]);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 a = __bfloat1622float2(w2[e]), b = __bfloat1622float2(x2[e]);
            acc[r][m] = __fmaf_rn(a.x, b.x, acc[r][m]);
            acc[r][m] = __fmaf_rn(a.y, b.y, acc[r][m]);
          }
        }
      }
    }
  } else {
    for (int j = lane; j < K; j += 32) {
#pragma unroll
      for (int r = 0; r < R; r++) {
        const float a = __bfloat162float(W[(size_t)min(row0 + r, N - 1) * ldw + j]);
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = __fmaf_rn(a, __bfloat162float(x[(size_t)m * ldx + j]), acc[r][m]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) {
      const float v = warp_sum(acc[r][m]);
      const int row = row0 + r;
      if (lane == 0 && row < N) {
        if (yb) yb[(size_t)m * ldy + row] = __float2bfloat16(v);
        else yf[(size_t)m * ldy + row] = round ? bf(v) : v;
      }
    }
}

void dgemv(torch::Tensor W, int64_t ldw, int64_t N, int64_t K, torch::Tensor x, int64_t ldx, torch::Tensor y,
           int64_t ldy, int64_t M, bool round) {
  const bool vec = (K % 8 == 0) && (ldw % 8 == 0) && (ldx % 8 == 0) &&
                   (reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0) && (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  float* yf = y.scalar_type() == torch::kFloat32 ? y.data_ptr<float>() : nullptr;
  __nv_bfloat16* yb = yf ? nullptr : BF(y);
  const int R = (vec && N >= 512) ? 4 : 1;      // several rows per warp for the big (gate / VeRA) matrices
  dim3 grid((N + 8 * R - 1) / (8 * R));
#define DG(MM)                                                                                                        \
  case MM:                                                                                                            \
    if (vec && R == 4) dgemv_k<MM, true, 4><<<grid, 256, 0, STREAM>>>(CBF(W), ldw, CBF(x), ldx, yf, yb, ldy, N, K, round); \
    else if (vec) dgemv_k<MM, true, 1><<<grid, 256, 0, STREAM>>>(CBF(W), ldw, CBF(x), ldx, yf, yb, ldy, N, K, round);     \
    else dgemv_k<MM, false, 1><<<grid, 256, 0, STREAM>>>(CBF(W), ldw, CBF(x), ldx, yf, yb, ldy, N, K, round);            \
    break;
  switch (M) { DG(1) DG(2) DG(3) DG(4) default: TORCH_CHECK(false, "M <= 4"); }
#undef DG
}

// =====================================================================================================================
// GDN-2 gates. sg (M, ldsg) fp32 from dgemv over [U_a; U_be; U_bw (3 nv rows, bf16-rounded) | A (rank) | lora A (lr,
// bf16-rounded)]. Per output channel j (= h * 128 + c):
//   t_g = bf16(vb_g[j] * sum_r B[j][r] bf16(vd_g[r] u[r]));  pre_g = bf16(base_g[h] + t_g)
//   a = bf16(pre_a + bf16(sum_r Bl[j][r] ul[r]));  g = -exp(A_log) softplus(a + dt_bias);  be, bw = bf16(sigmoid(pre))
// One warp per channel j, all M rows.
constexpr int GRPW = 8;            // output channels per warp (amortizes the staging of the scaled VeRA vectors)
template <int M>
__global__ void __launch_bounds__(256) gdn2_gates_k(const float* __restrict__ sg, int ldsg, int nv, int rank, int lr,
                                                    const __nv_bfloat16* __restrict__ B, const __nv_bfloat16* __restrict__ Bl,
                                                    const float* __restrict__ vd, const float* __restrict__ vb,
                                                    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                                    float* __restrict__ gdec, float* __restrict__ ber,
                                                    float* __restrict__ bwr, int Nout, int dk) {
  extern __shared__ float ush[];                  // [3][M][rank] then [M][lr]
  float* ul = ush + 3 * M * rank;
  for (int i = threadIdx.x; i < 3 * M * rank; i += blockDim.x) {
    const int g = i / (M * rank), m = (i / rank) % M, r = i % rank;
    ush[i] = bf(__fmul_rn(vd[g * rank + r], sg[(size_t)m * ldsg + 3 * nv + r]));
  }
  for (int i = threadIdx.x; i < M * lr; i += blockDim.x) {
    const int m = i / lr, r = i % lr;
    ul[i] = sg[(size_t)m * ldsg + 3 * nv + rank + r];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  for (int jj = 0; jj < GRPW; jj++) {
  const int j = (blockIdx.x * 8 + (threadIdx.x >> 5)) * GRPW + jj;
  if (j >= Nout) return;
  float t[3][M], lo[M];
#pragma unroll
  for (int m = 0; m < M; m++) { t[0][m] = t[1][m] = t[2][m] = 0.f; lo[m] = 0.f; }
  const __nv_bfloat16* br = B + (size_t)j * rank;
  for (int r0 = lane * 8; r0 < rank; r0 += 256) {          // rank % 8 == 0 (host check): 16-byte loads
    const uint4 bu = __ldg(reinterpret_cast<const uint4*>(br + r0));
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bu);
    float bv[8];
#pragma unroll
    for (int e = 0; e < 4; e++) { const float2 f = __bfloat1622float2(b2[e]); bv[2 * e] = f.x; bv[2 * e + 1] = f.y; }
#pragma unroll
    for (int g = 0; g < 3; g++)
#pragma unroll
      for (int m = 0; m < M; m++) {
        const float4* uu = reinterpret_cast<const float4*>(ush + (g * M + m) * rank + r0);
        const float4 a = uu[0], c = uu[1];
        float acc = t[g][m];
        acc = __fmaf_rn(bv[0], a.x, acc); acc = __fmaf_rn(bv[1], a.y, acc);
        acc = __fmaf_rn(bv[2], a.z, acc); acc = __fmaf_rn(bv[3], a.w, acc);
        acc = __fmaf_rn(bv[4], c.x, acc); acc = __fmaf_rn(bv[5], c.y, acc);
        acc = __fmaf_rn(bv[6], c.z, acc); acc = __fmaf_rn(bv[7], c.w, acc);
        t[g][m] = acc;
      }
  }
  if (lr > 0) {
    const __nv_bfloat16* blr = Bl + (size_t)j * lr;
    for (int r = lane; r < lr; r += 32) {
      const float b = __bfloat162float(blr[r]);
#pragma unroll
      for (int m = 0; m < M; m++) lo[m] = __fmaf_rn(b, ul[m * lr + r], lo[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; m++) {
#pragma unroll
    for (int g = 0; g < 3; g++) t[g][m] = warp_sum(t[g][m]);
    lo[m] = warp_sum(lo[m]);
  }
  if (lane < M) {
    float ta = 0.f, tb = 0.f, tw = 0.f, ll = 0.f;
#pragma unroll
    for (int m = 0; m < M; m++)
      if (m == lane) { ta = t[0][m]; tb = t[1][m]; tw = t[2][m]; ll = lo[m]; }
    const int m = lane, h = j / dk;
    const float* s = sg + (size_t)m * ldsg;
    const float pa = bf(__fadd_rn(s[h], bf(__fmul_rn(vb[j], ta))));
    const float pb = bf(__fadd_rn(s[nv + h], bf(__fmul_rn(vb[Nout + j], tb))));
    const float pw = bf(__fadd_rn(s[2 * nv + h], bf(__fmul_rn(vb[2 * Nout + j], tw))));
    const float a = lr > 0 ? bf(__fadd_rn(pa, bf(ll))) : pa;
    const float x = __fadd_rn(a, dt_bias[j]);
    const float sp = x > 20.f ? x : log1pf(__expf(x));
    gdec[(size_t)m * Nout + j] = __fmul_rn(-__expf(A_log[j]), sp);
    ber[(size_t)m * Nout + j] = bf(sigm(pb));
    bwr[(size_t)m * Nout + j] = bf(sigm(pw));
  }
  }
}

void gdn2_gates(torch::Tensor sg, int64_t ldsg, int64_t nv, int64_t rank, int64_t lr, torch::Tensor B,
                c10::optional<torch::Tensor> Bl, torch::Tensor vd, torch::Tensor vb, torch::Tensor A_log,
                torch::Tensor dt_bias, torch::Tensor gdec, torch::Tensor ber, torch::Tensor bwr, int64_t M, int64_t dk) {
  const int Nout = A_log.numel();
  const size_t sm = (size_t)(3 * M * rank + M * lr) * sizeof(float);
  dim3 grid((Nout + 8 * GRPW - 1) / (8 * GRPW));
  const __nv_bfloat16* blp = Bl ? CBF(*Bl) : nullptr;
  TORCH_CHECK(rank % 8 == 0, "VeRA rank must be a multiple of 8");
#define GG(MM)                                                                                                     \
  case MM:                                                                                                         \
    cudaFuncSetAttribute(gdn2_gates_k<MM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);                 \
    gdn2_gates_k<MM><<<grid, 256, sm, STREAM>>>(sg.data_ptr<float>(), ldsg, nv, rank, lr, CBF(B), blp,            \
        vd.data_ptr<float>(), vb.data_ptr<float>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),             \
        gdec.data_ptr<float>(), ber.data_ptr<float>(), bwr.data_ptr<float>(), Nout, dk);                            \
    break;
  switch (M) { GG(1) GG(2) GG(3) GG(4) default: TORCH_CHECK(false, "M <= 4"); }
#undef GG
}

// GDN-2 causal depthwise conv (kernel 4) over M tokens: y_t = bf16(silu(sum_k w[c][k] window_t[k])), window = the last 4
// inputs. write_state: shift the M inputs into the state (plain decode); else keep it and save the inputs (verify).
__global__ void gdn2_conv_k(const __nv_bfloat16* __restrict__ x, int ldx, float* __restrict__ st,
                            const float* __restrict__ w, float* __restrict__ y, __nv_bfloat16* __restrict__ xsave, int C,
                            int M, bool write_state) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float4 s = reinterpret_cast<float4*>(st)[c];
  const float4 k = reinterpret_cast<const float4*>(w)[c];
  for (int t = 0; t < M; t++) {
    const __nv_bfloat16 xb = x[(size_t)t * ldx + c];
    if (xsave) xsave[(size_t)t * C + c] = xb;
    s = make_float4(s.y, s.z, s.w, __bfloat162float(xb));
    float a = __fmul_rn(k.x, s.x);
    a = __fmaf_rn(k.y, s.y, a);
    a = __fmaf_rn(k.z, s.z, a);
    a = __fmaf_rn(k.w, s.w, a);
    y[(size_t)t * C + c] = bf(__fdiv_rn(a, __fadd_rn(1.f, __expf(-a))));
  }
  if (write_state) reinterpret_cast<float4*>(st)[c] = s;
}
// commit after verification: shift the first n + 1 saved inputs into the state (n = accepted drafts, on the GPU)
__global__ void gdn2_conv_commit_k(float* __restrict__ st, const __nv_bfloat16* __restrict__ xsave,
                                   const int* __restrict__ nacc, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float4 s = reinterpret_cast<float4*>(st)[c];
  const int n = nacc[0] + 1;
  for (int t = 0; t < n; t++) s = make_float4(s.y, s.z, s.w, __bfloat162float(xsave[(size_t)t * C + c]));
  reinterpret_cast<float4*>(st)[c] = s;
}
void gdn2_conv(torch::Tensor x, int64_t ldx, torch::Tensor st, torch::Tensor w, torch::Tensor y,
               c10::optional<torch::Tensor> xsave, int64_t M, bool write_state) {
  const int C = st.size(0);
  gdn2_conv_k<<<(C + 255) / 256, 256, 0, STREAM>>>(CBF(x), ldx, st.data_ptr<float>(), w.data_ptr<float>(),
                                                   y.data_ptr<float>(), xsave ? BF(*xsave) : nullptr, C, M, write_state);
}
void gdn2_conv_commit(torch::Tensor st, torch::Tensor xsave, torch::Tensor nacc) {
  const int C = st.size(0);
  gdn2_conv_commit_k<<<(C + 255) / 256, 256, 0, STREAM>>>(st.data_ptr<float>(), CBF(xsave), nacc.data_ptr<int>(), C);
}

// GDN-2 recurrence + gated RMSNorm. One block per value head h (128 threads = value columns j; the state column
// S[:, j] lives in registers). Per step (fla fused_recurrent_gdn2 arithmetic):
//   q, k l2-normalized (x / sqrt(sum x^2 + 1e-6)), q *= dk^-0.5;  S = diag(exp g) S;  e_j = sum_i S_ij b_i k_i;
//   v'_j = w_j v_j - e_j;  S += k v'^T;  o_j = sum_i S_ij q_i
//   out = bf16( bf16( bf16(o) * rsqrt(mean bf16(o)^2 + eps) ) * gn ) * silu(z) -> bf16 (+ int8 groups)
// mode 0: M steps, outputs, write the state. 1 (verify): M steps, outputs, state untouched. 2 (commit): nacc + 1 steps
// from the saved inputs, write the state, no outputs.
constexpr int DK = 128;
__device__ __forceinline__ void gdn2_step(float* s, const float* __restrict__ mixed, const float* __restrict__ gd,
                                          const float* __restrict__ be, const float* __restrict__ bw, int h, int kh,
                                          int key_dim, int j, float* qs, float* bks, float* ks, float* dec, float* sh,
                                          float& o_out, bool want_o, const float* pe_c, int nv) {
  const float q = mixed[kh * DK + j], k = mixed[key_dim + kh * DK + j], v = mixed[2 * key_dim + h * DK + j];
  float qq = q, kk = k;
  if (pe_c) {                                     // decay phase: rotate channel pairs by pe_c * mean pair log-decay
    const float gm = __fmul_rn(__fadd_rn(gd[h * DK + (j & ~1)], gd[h * DK + (j | 1)]), 0.5f);
    const float th = __fmul_rn(gm, pe_c[h]);
    float sn, cs;
    sincosf(th, &sn, &cs);
    const float qo = __shfl_xor_sync(0xffffffff, q, 1), ko = __shfl_xor_sync(0xffffffff, k, 1);
    if ((j & 1) == 0) { qq = bf(cs * q - sn * qo); kk = bf(cs * k - sn * ko); }
    else { qq = bf(sn * qo + cs * q); kk = bf(sn * ko + cs * k); }
  }
  const float rq = block_sum<DK>(__fmul_rn(qq, qq), sh), rk = block_sum<DK>(__fmul_rn(kk, kk), sh);
  const float qn = __fmul_rn(__fdiv_rn(qq, __fsqrt_rn(__fadd_rn(rq, 1e-6f))), 0.08838834764831845f);
  const float kn = __fdiv_rn(kk, __fsqrt_rn(__fadd_rn(rk, 1e-6f)));
  qs[j] = qn;
  ks[j] = kn;
  bks[j] = __fmul_rn(be[h * DK + j], kn);
  dec[j] = __expf(gd[h * DK + j]);
  __syncthreads();
  float e = 0.f;
#pragma unroll
  for (int i = 0; i < DK; i++) { s[i] = __fmul_rn(s[i], dec[i]); e = __fmaf_rn(s[i], bks[i], e); }
  const float vn = __fsub_rn(__fmul_rn(bw[h * DK + j], v), e);
  float o = 0.f;
#pragma unroll
  for (int i = 0; i < DK; i++) { s[i] = __fmaf_rn(ks[i], vn, s[i]); o = __fmaf_rn(s[i], qs[i], o); }
  o_out = o;
  __syncthreads();                                // qs/ks/bks/dec reused by the next step
}

__global__ void __launch_bounds__(DK) gdn2_rec_k(const float* __restrict__ mixed, int C, const __nv_bfloat16* __restrict__ z,
                                                 int ldz, const float* __restrict__ gd, const float* __restrict__ be,
                                                 const float* __restrict__ bw, const __nv_bfloat16* __restrict__ gn,
                                                 float* __restrict__ S, __nv_bfloat16* __restrict__ out,
                                                 int8_t* __restrict__ XQ, float* __restrict__ XD, int nk, int nv,
                                                 float eps, int M, int mode, const int* __restrict__ nacc,
                                                 const float* __restrict__ pe_c) {
  __shared__ float qs[DK], ks[DK], bks[DK], dec[DK], sh[DK / 32];
  const int h = blockIdx.x, j = threadIdx.x, kh = h / (nv / nk), key_dim = nk * DK, Nv = nv * DK;
  float* Sh = S + (size_t)h * DK * DK + j;
  float s[DK];
#pragma unroll
  for (int i = 0; i < DK; i++) s[i] = Sh[i * DK];
  const int steps = mode == 2 ? nacc[0] + 1 : M;
  for (int t = 0; t < steps; t++) {
    float o;
    gdn2_step(s, mixed + (size_t)t * C, gd + (size_t)t * Nv, be + (size_t)t * Nv, bw + (size_t)t * Nv, h, kh, key_dim,
              j, qs, bks, ks, dec, sh, o, mode != 2, pe_c, nv);
    if (mode != 2) {
      const float ob = bf(o);
      const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum<DK>(__fmul_rn(ob, ob), sh), (float)DK), eps));
      const float zz = __bfloat162float(z[(size_t)t * ldz + h * DK + j]);
      const float sz = __fdiv_rn(zz, __fadd_rn(1.f, __expf(-zz)));
      const __nv_bfloat16 y = __float2bfloat16(__fmul_rn(bf(__fmul_rn(bf(__fmul_rn(ob, r)), __bfloat162float(gn[j]))), sz));
      out[(size_t)t * Nv + h * DK + j] = y;
      warp_quant32(__bfloat162float(y), t * (Nv / 32) + ((h * DK + j) >> 5), j & 31, XQ, XD);
    }
  }
  if (mode != 1) {
#pragma unroll
    for (int i = 0; i < DK; i++) Sh[i * DK] = s[i];
  }
}

void gdn2_rec(torch::Tensor mixed, torch::Tensor z, int64_t ldz, torch::Tensor gd, torch::Tensor be, torch::Tensor bw,
              torch::Tensor gn, torch::Tensor S, torch::Tensor out, torch::Tensor xq, torch::Tensor xd, int64_t nk,
              int64_t nv, double eps, int64_t M, int64_t mode, c10::optional<torch::Tensor> nacc,
              c10::optional<torch::Tensor> pe_c) {
  TORCH_CHECK(S.size(1) == DK && S.size(2) == DK);
  gdn2_rec_k<<<nv, DK, 0, STREAM>>>(mixed.data_ptr<float>(), mixed.size(-1), CBF(z), ldz, gd.data_ptr<float>(),
                                    be.data_ptr<float>(), bw.data_ptr<float>(), CBF(gn), S.data_ptr<float>(), BF(out),
                                    xq.data_ptr<int8_t>(), xd.data_ptr<float>(), nk, nv, (float)eps, M, mode,
                                    nacc ? nacc->data_ptr<int>() : nullptr, pe_c ? pe_c->data_ptr<float>() : nullptr);
}

// fused commit + verify (speculative cycles): first replay the previous cycle's nprev + 1 accepted steps from its saved
// inputs (prev set) and write the committed state, then run the M new steps (cur set) with outputs and no state write.
// nprev < 0: nothing pending. Each step is gdn2_step, so the state equals plain decoding bit for bit.
__global__ void __launch_bounds__(DK) gdn2_rec_fused_k(const float* __restrict__ mixed, const float* __restrict__ mixed_p,
                                                       int C, const __nv_bfloat16* __restrict__ z, int ldz,
                                                       const float* __restrict__ gd, const float* __restrict__ be,
                                                       const float* __restrict__ bw, const float* __restrict__ gd_p,
                                                       const float* __restrict__ be_p, const float* __restrict__ bw_p,
                                                       const __nv_bfloat16* __restrict__ gn, float* __restrict__ S,
                                                       __nv_bfloat16* __restrict__ out, int8_t* __restrict__ XQ,
                                                       float* __restrict__ XD, int nk, int nv, float eps, int M,
                                                       const int* __restrict__ nprev, const float* __restrict__ pe_c) {
  __shared__ float qs[DK], ks[DK], bks[DK], dec[DK], sh[DK / 32];
  const int h = blockIdx.x, j = threadIdx.x, kh = h / (nv / nk), key_dim = nk * DK, Nv = nv * DK;
  float* Sh = S + (size_t)h * DK * DK + j;
  float s[DK];
#pragma unroll
  for (int i = 0; i < DK; i++) s[i] = Sh[i * DK];
  const int np = nprev[0] + 1;
  for (int t = 0; t < np; t++) {
    float o;
    gdn2_step(s, mixed_p + (size_t)t * C, gd_p + (size_t)t * Nv, be_p + (size_t)t * Nv, bw_p + (size_t)t * Nv, h, kh,
              key_dim, j, qs, bks, ks, dec, sh, o, false, pe_c, nv);
  }
  if (np > 0) {
#pragma unroll
    for (int i = 0; i < DK; i++) Sh[i * DK] = s[i];
  }
  for (int t = 0; t < M; t++) {
    float o;
    gdn2_step(s, mixed + (size_t)t * C, gd + (size_t)t * Nv, be + (size_t)t * Nv, bw + (size_t)t * Nv, h, kh, key_dim,
              j, qs, bks, ks, dec, sh, o, true, pe_c, nv);
    const float ob = bf(o);
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum<DK>(__fmul_rn(ob, ob), sh), (float)DK), eps));
    const float zz = __bfloat162float(z[(size_t)t * ldz + h * DK + j]);
    const float sz = __fdiv_rn(zz, __fadd_rn(1.f, __expf(-zz)));
    const __nv_bfloat16 y = __float2bfloat16(__fmul_rn(bf(__fmul_rn(bf(__fmul_rn(ob, r)), __bfloat162float(gn[j]))), sz));
    out[(size_t)t * Nv + h * DK + j] = y;
    warp_quant32(__bfloat162float(y), t * (Nv / 32) + ((h * DK + j) >> 5), j & 31, XQ, XD);
  }
}
void gdn2_rec_fused(torch::Tensor mixed, torch::Tensor mixed_p, torch::Tensor z, int64_t ldz, torch::Tensor gd,
                    torch::Tensor be, torch::Tensor bw, torch::Tensor gd_p, torch::Tensor be_p, torch::Tensor bw_p,
                    torch::Tensor gn, torch::Tensor S, torch::Tensor out, torch::Tensor xq, torch::Tensor xd, int64_t nk,
                    int64_t nv, double eps, int64_t M, torch::Tensor nprev, c10::optional<torch::Tensor> pe_c) {
  gdn2_rec_fused_k<<<nv, DK, 0, STREAM>>>(mixed.data_ptr<float>(), mixed_p.data_ptr<float>(), mixed.size(-1), CBF(z), ldz,
      gd.data_ptr<float>(), be.data_ptr<float>(), bw.data_ptr<float>(), gd_p.data_ptr<float>(), be_p.data_ptr<float>(),
      bw_p.data_ptr<float>(), CBF(gn), S.data_ptr<float>(), BF(out), xq.data_ptr<int8_t>(), xd.data_ptr<float>(), nk, nv,
      (float)eps, M, nprev.data_ptr<int>(), pe_c ? pe_c->data_ptr<float>() : nullptr);
}
// conv: shift in the previous cycle's nprev + 1 saved inputs (committed state written), then the M new outputs
__global__ void gdn2_conv_fused_k(const __nv_bfloat16* __restrict__ x, int ldx, float* __restrict__ st,
                                  const float* __restrict__ w, float* __restrict__ y, __nv_bfloat16* __restrict__ xsave,
                                  const __nv_bfloat16* __restrict__ xsave_p, const int* __restrict__ nprev, int C, int M) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float4 s = reinterpret_cast<float4*>(st)[c];
  const int np = nprev[0] + 1;
  for (int t = 0; t < np; t++) s = make_float4(s.y, s.z, s.w, __bfloat162float(xsave_p[(size_t)t * C + c]));
  if (np > 0) reinterpret_cast<float4*>(st)[c] = s;
  const float4 k = reinterpret_cast<const float4*>(w)[c];
  for (int t = 0; t < M; t++) {
    const __nv_bfloat16 xb = x[(size_t)t * ldx + c];
    xsave[(size_t)t * C + c] = xb;
    s = make_float4(s.y, s.z, s.w, __bfloat162float(xb));
    float a = __fmul_rn(k.x, s.x);
    a = __fmaf_rn(k.y, s.y, a);
    a = __fmaf_rn(k.z, s.z, a);
    a = __fmaf_rn(k.w, s.w, a);
    y[(size_t)t * C + c] = bf(__fdiv_rn(a, __fadd_rn(1.f, __expf(-a))));
  }
}
void gdn2_conv_fused(torch::Tensor x, int64_t ldx, torch::Tensor st, torch::Tensor w, torch::Tensor y, torch::Tensor xsave,
                     torch::Tensor xsave_p, torch::Tensor nprev, int64_t M) {
  const int C = st.size(0);
  gdn2_conv_fused_k<<<(C + 255) / 256, 256, 0, STREAM>>>(CBF(x), ldx, st.data_ptr<float>(), w.data_ptr<float>(),
      y.data_ptr<float>(), BF(xsave), CBF(xsave_p), nprev.data_ptr<int>(), C, M);
}

// =====================================================================================================================
// Skinny bf16 GEMM on tensor cores: Y[n][i] = sum_k W[i][k] X[n][k] for n < Mx (Mx <= 16; X has 16 rows allocated,
// rows >= Mx may hold anything: their columns are computed and dropped). One block per 16-row tile of W; the 8 warps
// split K into fixed slices and their partial tiles are summed in warp order (deterministic, independent of Mx: a
// column only ever depends on its own X row). W (N, K) bf16 row stride ldw, K % 16 == 0, N % 16 == 0.
__global__ void __launch_bounds__(256) tc_skinny_k(const __nv_bfloat16* __restrict__ W, int ldw,
                                                   const __nv_bfloat16* __restrict__ X, int ldx, float* __restrict__ Y,
                                                   int ldy, int N, int K, int Mx, bool round) {
  __shared__ __align__(32) float part[8][16 * 16];
  const int warp = threadIdx.x >> 5, row0 = blockIdx.x * 16;
  const int ks = K / 16, k_lo = ks * warp / 8, k_hi = ks * (warp + 1) / 8;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
  wmma::fill_fragment(c0, 0.f);
  wmma::fill_fragment(c1, 0.f);
  int kt = k_lo;
  for (; kt + 1 < k_hi; kt += 2) {                 // two independent chains: more loads in flight
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;
    wmma::load_matrix_sync(a0, W + (size_t)row0 * ldw + kt * 16, ldw);
    wmma::load_matrix_sync(a1, W + (size_t)row0 * ldw + (kt + 1) * 16, ldw);
    wmma::load_matrix_sync(b0, X + kt * 16, ldx);
    wmma::load_matrix_sync(b1, X + (kt + 1) * 16, ldx);
    wmma::mma_sync(c0, a0, b0, c0);
    wmma::mma_sync(c1, a1, b1, c1);
  }
  if (kt < k_hi) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0;
    wmma::load_matrix_sync(a0, W + (size_t)row0 * ldw + kt * 16, ldw);
    wmma::load_matrix_sync(b0, X + kt * 16, ldx);
    wmma::mma_sync(c0, a0, b0, c0);
  }
#pragma unroll
  for (int i = 0; i < c0.num_elements; i++) c0.x[i] = __fadd_rn(c0.x[i], c1.x[i]);
  wmma::store_matrix_sync(part[warp], c0, 16, wmma::mem_row_major);   // part[w][i * 16 + n]
  __syncthreads();
  const int t = threadIdx.x, i = t / 16, n = t % 16;
  if (n < Mx) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; w++) v = __fadd_rn(v, part[w][i * 16 + n]);
    Y[(size_t)n * ldy + row0 + i] = round ? bf(v) : v;
  }
}
// staged variant: the 16 x K row tile streamed through shared memory in 16 x 512 chunks with cp.async (double
// buffered, fully coalesced 16-byte segments); warp w takes k-steps w, w + 8, .. of every chunk (fixed, M-independent)
constexpr int TCC = 256, TST = 4;               // chunk columns, pipeline stages (3 chunks in flight)
// shared memory: W tiles [TST][16][LD] bf16, X tiles [TST][16][LD] bf16 (only rows < Mx loaded; the others zero), part
__global__ void __launch_bounds__(256) tc_skinny2_k(const __nv_bfloat16* __restrict__ W, int ldw,
                                                    const __nv_bfloat16* __restrict__ X, int ldx, float* __restrict__ Y,
                                                    int ldy, int N, int K, int Mx, bool round) {
  extern __shared__ __align__(32) unsigned char tsm[];
  constexpr int LD = TCC + 16;
  __nv_bfloat16* buf = reinterpret_cast<__nv_bfloat16*>(tsm);              // [TST][16][LD]
  __nv_bfloat16* xbuf = buf + TST * 16 * LD;                               // [TST][16][LD]
  float* part = reinterpret_cast<float*>(xbuf + TST * 16 * LD);            // [8][256]
  const int warp = threadIdx.x >> 5, row0 = blockIdx.x * 16, tid = threadIdx.x;
  const int nch = (K + TCC - 1) / TCC;
  for (int i = tid; i < TST * 16 * LD / 8; i += 256) {                       // zero the X tiles once (rows >= Mx stay 0)
    const int row = (i * 8 / LD) % 16;
    if (row >= Mx) reinterpret_cast<uint4*>(xbuf)[i] = make_uint4(0, 0, 0, 0);
  }
  auto stage = [&](int c) {
    if (c < nch) {
      const int k0 = c * TCC, kc = min(TCC, K - k0), seg = kc / 8;
      __nv_bfloat16* b = buf + (size_t)(c % TST) * 16 * LD;
      __nv_bfloat16* xb = xbuf + (size_t)(c % TST) * 16 * LD;
      for (int i = tid; i < (16 + Mx) * seg; i += 256) {
        const int r = i / seg, sg = i % seg;
        if (r < 16) {
          const unsigned dst = (unsigned)__cvta_generic_to_shared(b + r * LD + sg * 8);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(W + (size_t)(row0 + r) * ldw + k0 + sg * 8));
        } else {
          const unsigned dst = (unsigned)__cvta_generic_to_shared(xb + (r - 16) * LD + sg * 8);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(X + (size_t)(r - 16) * ldx + k0 + sg * 8));
        }
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);                         // (empty groups keep the count uniform)
  };
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0;
  wmma::fill_fragment(c0, 0.f);
#pragma unroll
  for (int c = 0; c < TST - 1; c++) stage(c);
  for (int c = 0; c < nch; c++) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(TST - 2));
    __syncthreads();                                  // chunk c landed; chunk c - 1's buffers are free
    stage(c + TST - 1);
    const int ks = min(TCC, K - c * TCC) / 16;
    const __nv_bfloat16* tile = buf + (size_t)(c % TST) * 16 * LD;
    const __nv_bfloat16* xt = xbuf + (size_t)(c % TST) * 16 * LD;
    for (int kt = warp; kt < ks; kt += 8) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bx;
      wmma::load_matrix_sync(a, tile + kt * 16, LD);
      wmma::load_matrix_sync(bx, xt + kt * 16, LD);
      wmma::mma_sync(c0, a, bx, c0);
    }
  }
  wmma::store_matrix_sync(part + warp * 256, c0, 16, wmma::mem_row_major);
  __syncthreads();
  const int i = tid / 16, n = tid % 16;
  if (n < Mx) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; w++) v = __fadd_rn(v, part[w * 256 + i * 16 + n]);
    Y[(size_t)n * ldy + row0 + i] = round ? bf(v) : v;
  }
}
static void tc_skinny_launch(const __nv_bfloat16* W, int ldw, const __nv_bfloat16* X, int ldx, float* Y, int ldy, int N,
                             int K, int Mx, bool round) {
  const int sm = 2 * TST * 16 * (TCC + 16) * 2 + 8 * 256 * 4;
  cudaFuncSetAttribute(tc_skinny2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  tc_skinny2_k<<<N / 16, 256, sm, STREAM>>>(W, ldw, X, ldx, Y, ldy, N, K, Mx, round);
}
void tc_skinny(torch::Tensor W, int64_t ldw, int64_t N, int64_t K, torch::Tensor X, int64_t ldx, torch::Tensor Y,
               int64_t ldy, int64_t Mx, bool round) {
  TORCH_CHECK(N % 16 == 0 && K % 16 == 0 && ldw % 16 == 0 && ldx % 16 == 0 && Mx <= 16 && X.size(0) >= 16);
  tc_skinny_launch(CBF(W), ldw, CBF(X), ldx, Y.data_ptr<float>(), ldy, N, K, Mx, round);
}

// GDN-2 gates with the VeRA product on tensor cores. Stage: W2[g * M + m][r] = bf16(vd_g[r] u_m[r]) (16 rows); then
// tc_skinny(B, W2) -> T[g * M + m][j]; epilogue per (m, j) as gdn2_gates_k.
__global__ void vera_stage_k(const float* __restrict__ sg, int ldsg, int off, const float* __restrict__ vd, int rank,
                             __nv_bfloat16* __restrict__ W2, int M) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * M * rank) return;
  const int row = i / rank, r = i % rank, g = row / M, m = row % M;
  W2[(size_t)row * rank + r] = __float2bfloat16(__fmul_rn(vd[g * rank + r], sg[(size_t)m * ldsg + off + r]));
}
__global__ void gdn2_gates_epi_k(const float* __restrict__ sg, int ldsg, int nv, int rank, int lr,
                                 const float* __restrict__ T, const __nv_bfloat16* __restrict__ Bl,
                                 const float* __restrict__ vb, const float* __restrict__ A_log,
                                 const float* __restrict__ dt_bias, float* __restrict__ gdec, float* __restrict__ ber,
                                 float* __restrict__ bwr, int Nout, int dk, int M) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * Nout) return;
  const int m = i / Nout, j = i % Nout, h = j / dk;
  const float* s = sg + (size_t)m * ldsg;
  float lo = 0.f;
  if (lr > 0) {
    const __nv_bfloat16* blr = Bl + (size_t)j * lr;
    for (int r = 0; r < lr; r++) lo = __fmaf_rn(__bfloat162float(blr[r]), s[3 * nv + rank + r], lo);
  }
  const float ta = T[(size_t)(0 * M + m) * Nout + j], tb = T[(size_t)(1 * M + m) * Nout + j], tw = T[(size_t)(2 * M + m) * Nout + j];
  const float pa = bf(__fadd_rn(s[h], bf(__fmul_rn(vb[j], ta))));
  const float pb = bf(__fadd_rn(s[nv + h], bf(__fmul_rn(vb[Nout + j], tb))));
  const float pw = bf(__fadd_rn(s[2 * nv + h], bf(__fmul_rn(vb[2 * Nout + j], tw))));
  const float a = lr > 0 ? bf(__fadd_rn(pa, bf(lo))) : pa;
  const float x = __fadd_rn(a, dt_bias[j]);
  const float sp = x > 20.f ? x : log1pf(__expf(x));
  gdec[(size_t)m * Nout + j] = __fmul_rn(-__expf(A_log[j]), sp);
  ber[(size_t)m * Nout + j] = bf(sigm(pb));
  bwr[(size_t)m * Nout + j] = bf(sigm(pw));
}
// the gate epilogue alone, for any number of rows M (prefill): T (3M, Nout) fp32 = rows (g * M + m) of the VeRA B products
void gdn2_gates_epi(torch::Tensor sg, int64_t ldsg, int64_t nv, int64_t rank, int64_t lr, torch::Tensor T,
                    c10::optional<torch::Tensor> Bl, torch::Tensor vb, torch::Tensor A_log, torch::Tensor dt_bias,
                    torch::Tensor gdec, torch::Tensor ber, torch::Tensor bwr, int64_t M, int64_t dk) {
  const int Nout = A_log.numel();
  gdn2_gates_epi_k<<<(M * Nout + 255) / 256, 256, 0, STREAM>>>(sg.data_ptr<float>(), ldsg, nv, rank, lr, T.data_ptr<float>(),
      Bl ? CBF(*Bl) : nullptr, vb.data_ptr<float>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      gdec.data_ptr<float>(), ber.data_ptr<float>(), bwr.data_ptr<float>(), Nout, dk, M);
}
void gdn2_gates_tc(torch::Tensor sg, int64_t ldsg, int64_t nv, int64_t rank, int64_t lr, torch::Tensor B,
                   c10::optional<torch::Tensor> Bl, torch::Tensor vd, torch::Tensor vb, torch::Tensor A_log,
                   torch::Tensor dt_bias, torch::Tensor W2, torch::Tensor T, torch::Tensor gdec, torch::Tensor ber,
                   torch::Tensor bwr, int64_t M, int64_t dk) {
  const int Nout = A_log.numel();
  TORCH_CHECK(3 * M <= 16 && W2.size(0) >= 16 && W2.size(1) == rank);
  vera_stage_k<<<(3 * M * rank + 255) / 256, 256, 0, STREAM>>>(sg.data_ptr<float>(), ldsg, 3 * nv, vd.data_ptr<float>(),
                                                               rank, BF(W2), M);
  tc_skinny_launch(CBF(B), rank, CBF(W2), rank, T.data_ptr<float>(), Nout, Nout, rank, 3 * M, false);
  gdn2_gates_epi_k<<<(M * Nout + 255) / 256, 256, 0, STREAM>>>(sg.data_ptr<float>(), ldsg, nv, rank, lr, T.data_ptr<float>(),
      Bl ? CBF(*Bl) : nullptr, vb.data_ptr<float>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      gdec.data_ptr<float>(), ber.data_ptr<float>(), bwr.data_ptr<float>(), Nout, dk, M);
}

// =====================================================================================================================
// Embedding rows from the NF4 table (kernel packing: element 2i in the LOW nibble): bf16(code[q] * absmax)
__global__ void emb_nf4_k(const long* __restrict__ tok, const uint8_t* __restrict__ codes, const float* __restrict__ absmax,
                          const float* __restrict__ code, __nv_bfloat16* __restrict__ y, int ldy, int D) {
  const int m = blockIdx.x;
  const long t = tok[m];
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const uint8_t b = codes[(size_t)t * (D / 2) + i / 2];
    const int q = (i & 1) ? (b >> 4) : (b & 15);
    y[(size_t)m * ldy + i] = __float2bfloat16(__fmul_rn(code[q], absmax[(size_t)t * (D / 64) + i / 64]));
  }
}
// the same from the nf4_mma layout (the LM head's codes): value i of a row sits in super s = i / 128, group j, lane t,
// halfword h, nibble q: word ((row * nsup + s) * 4 + t) * 4 + j, nibble 4 h + q
__global__ void emb_nf4_mma_k(const long* __restrict__ tok, const uint32_t* __restrict__ codes,
                              const float* __restrict__ absmax, const float* __restrict__ code,
                              __nv_bfloat16* __restrict__ y, int ldy, int D) {
  const int m = blockIdx.x;
  const long t = tok[m];
  const int nsup = D / 128;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const int s = i >> 7, j = (i & 127) >> 5, e = i & 31, ln = (e & 15) >> 2, h = e >> 4, q = e & 3;
    const uint32_t w = codes[(((size_t)t * nsup + s) * 4 + ln) * 4 + j];
    const int c = (w >> (4 * (4 * h + q))) & 15;
    y[(size_t)m * ldy + i] = __float2bfloat16(__fmul_rn(code[c], absmax[(size_t)t * (D / 64) + i / 64]));
  }
}
void emb_nf4_mma(torch::Tensor tok, torch::Tensor codes, torch::Tensor absmax, torch::Tensor code, torch::Tensor y,
                 int64_t ldy, int64_t M) {
  const int D = codes.size(1) * 128;
  emb_nf4_mma_k<<<M, 256, 0, STREAM>>>(tok.data_ptr<long>(), reinterpret_cast<const uint32_t*>(codes.data_ptr()),
                                      absmax.data_ptr<float>(), code.data_ptr<float>(), BF(y), ldy, D);
}
void emb_nf4(torch::Tensor tok, torch::Tensor codes, torch::Tensor absmax, torch::Tensor code, torch::Tensor y,
             int64_t ldy, int64_t M) {
  const int D = codes.size(1) * 2;
  emb_nf4_k<<<M, 256, 0, STREAM>>>(tok.data_ptr<long>(), codes.data_ptr<uint8_t>(), absmax.data_ptr<float>(),
                                  code.data_ptr<float>(), BF(y), ldy, D);
}

// argmax over the first V entries of each of M fp32 rows (first index of the maximum, as torch.argmax)
__global__ void argmax_k(const float* __restrict__ x, int ld, int V, long* __restrict__ out) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const int m = blockIdx.x;
  const float* r = x + (size_t)m * ld;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float v = r[i];
    if (v > bv) { bv = v; bi = i; }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffff, bv, o);
    const int oi = __shfl_xor_sync(0xffffffff, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { sv[w] = bv; si[w] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x / 32; i++)
      if (sv[i] > bv || (sv[i] == bv && si[i] < bi)) { bv = sv[i]; bi = si[i]; }
    out[m] = bi;
  }
}
void argmax_rows(torch::Tensor x, int64_t ld, int64_t V, torch::Tensor out, int64_t M) {
  argmax_k<<<M, 1024, 0, STREAM>>>(x.data_ptr<float>(), ld, V, out.data_ptr<long>());
}

// =====================================================================================================================
// MLA. Cache row (bytes, stride RW, a multiple of 16): [pos i32 | inv_r f32 x 4 | latent norm f16 | rope norm f16 |
// rope codes 32 B | latent codes ceil(r/2) B]; codes: index = #{boundaries < x / n}, two per byte (even index low).
constexpr int MG = 4, MH = 16, RD = 64;

__device__ __forceinline__ int tq_index(float u, const float* bnd) {
  int i = 0;
#pragma unroll
  for (int b = 0; b < 15; b++) i += bnd[b] < u;
  return i;
}

// per token: inv_r per group (from k_rms output), TurboQuant codes of the RoPE key (kr, 64) and latent (c_rot, r) ->
// the cache row at position pos[0] + m. One block (256 threads) per token.
__global__ void __launch_bounds__(256) mla_encode_k(const __nv_bfloat16* __restrict__ kk, int ldk,
                                                    const float* __restrict__ kr, const float* __restrict__ cr, int r,
                                                    const float* __restrict__ cbl, const float* __restrict__ cbr,
                                                    const long* __restrict__ pos, uint8_t* __restrict__ cache, int RW,
                                                    float k_eps) {
  __shared__ float sh[8];
  __shared__ float bl[15], br_[15];
  const int m = blockIdx.x, t = threadIdx.x;
  if (t < 15) { bl[t] = cbl[16 + t]; br_[t] = cbr[16 + t]; }
  const long p = pos[0] + m;
  uint8_t* row = cache + (size_t)p * RW;
  float inv[MG];
#pragma unroll
  for (int g = 0; g < MG; g++) {
    const float v = __bfloat162float(kk[(size_t)m * ldk + g * 256 + t]);
    const float ss = block_sum<256>(__fmul_rn(v, v), sh);
    inv[g] = rsqrtf(__fadd_rn(__fdiv_rn(ss, 256.f), k_eps));
  }
  // norms (fp32 sum of squares -> sqrt -> fp16)
  const float* krm = kr + (size_t)m * RD;
  const float* crm = cr + (size_t)m * r;
  float a = t < RD ? __fmul_rn(krm[t], krm[t]) : 0.f;
  const float nr2 = block_sum<256>(a, sh);
  float b = 0.f;
  for (int i = t; i < r; i += 256) b = __fmaf_rn(crm[i], crm[i], b);
  const float nl2 = block_sum<256>(b, sh);
  const __half nrh = __float2half_rn(__fsqrt_rn(nr2)), nlh = __float2half_rn(__fsqrt_rn(nl2));
  const float nrf = fmaxf(__half2float(nrh), 1e-12f), nlf = fmaxf(__half2float(nlh), 1e-12f);
  if (t == 0) {
    *reinterpret_cast<int*>(row) = (int)p;
#pragma unroll
    for (int g = 0; g < MG; g++) reinterpret_cast<float*>(row + 4)[g] = inv[g];
    *reinterpret_cast<__half*>(row + 20) = nlh;
    *reinterpret_cast<__half*>(row + 22) = nrh;
  }
  uint8_t* rc = row + 24;
  uint8_t* lc = rc + RD / 2;
  if (t < RD / 2) {
    const int lo = tq_index(__fdiv_rn(krm[2 * t], nrf), br_), hi = tq_index(__fdiv_rn(krm[2 * t + 1], nrf), br_);
    rc[t] = (uint8_t)(lo | (hi << 4));
  }
  for (int pidx = t; pidx < (r + 1) / 2; pidx += 256) {
    const int i = 2 * pidx;
    const int lo = tq_index(__fdiv_rn(crm[i], nlf), bl);
    const int hi = i + 1 < r ? tq_index(__fdiv_rn(crm[i + 1], nlf), bl) : 0;
    lc[pidx] = (uint8_t)(lo | (hi << 4));
  }
}
void mla_encode(torch::Tensor kk, int64_t ldk, torch::Tensor kr, torch::Tensor cr, int64_t r, torch::Tensor cbl,
                torch::Tensor cbr, torch::Tensor pos, torch::Tensor cache, int64_t RW, double k_eps, int64_t M) {
  mla_encode_k<<<M, 256, 0, STREAM>>>(CBF(kk), ldk, kr.data_ptr<float>(), cr.data_ptr<float>(), r, cbl.data_ptr<float>(),
                                     cbr.data_ptr<float>(), pos.data_ptr<long>(), cache.data_ptr<uint8_t>(), RW, (float)k_eps);
}

// queries: y_h = bf16(RMSNorm(q_h) * qn); q_full = P_h y_h (fp32), RoPE (NeoX) on the first 64 at position pos + m with
// bf16-rounded cos / sin; written as fp16 (M, H, Ep) with zero padding. Grid (H, ceil(Ep / 64)), 256 threads (8 warps,
// 8 rows each); y_h of every token staged in shared memory.
template <int M>
__global__ void __launch_bounds__(256) mla_query_k(const __nv_bfloat16* __restrict__ yq, int ldq,
                                                   const float* __restrict__ qn, float q_eps,
                                                   const __nv_bfloat16* __restrict__ P, int E, int Ep,
                                                   const long* __restrict__ pos, const float* __restrict__ inv_freq,
                                                   __half* __restrict__ Q) {
  __shared__ float ys[M][256];
  __shared__ float outs[M][64];
  __shared__ float sh[8];
  const int h = blockIdx.x, e0 = blockIdx.y * 64, t = threadIdx.x, lane = t & 31, w = t >> 5;
#pragma unroll
  for (int m = 0; m < M; m++) {
    const float v = __bfloat162float(yq[(size_t)m * ldq + h * 512 + t]);      // per head [q (256) | gate (256)]
    const float ss = block_sum<256>(__fmul_rn(v, v), sh);
    const float rr = rsqrtf(__fadd_rn(__fdiv_rn(ss, 256.f), q_eps));
    ys[m][t] = bf(__fmul_rn(__fmul_rn(v, rr), qn[t]));
  }
  __syncthreads();
  for (int k = 0; k < 8; k++) {
    const int el = w * 8 + k, e = e0 + el;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; m++) acc[m] = 0.f;
    if (e < E) {
      const __nv_bfloat16* pr = P + ((size_t)h * E + e) * 256;
      const uint4 pu = __ldg(reinterpret_cast<const uint4*>(pr + lane * 8));
      const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&pu);
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const float2 f = __bfloat1622float2(p2[q]);
#pragma unroll
        for (int m = 0; m < M; m++) {
          acc[m] = __fmaf_rn(f.x, ys[m][lane * 8 + 2 * q], acc[m]);
          acc[m] = __fmaf_rn(f.y, ys[m][lane * 8 + 2 * q + 1], acc[m]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++) acc[m] = warp_sum(acc[m]);
    }
    if (lane == 0)
#pragma unroll
      for (int m = 0; m < M; m++) outs[m][el] = acc[m];
  }
  __syncthreads();
  for (int i = t; i < M * 64; i += 256) {
    const int m = i / 64, el = i % 64, e = e0 + el;
    if (e >= Ep) continue;
    float v = outs[m][el];
    if (e0 == 0) {                                // RoPE on dims 0..63: pairs (p, p + 32)
      const int p = el % 32;
      const float ang = __fmul_rn((float)(pos[0] + m), inv_freq[p]);
      float sn, cs;
      sincos_acc(ang, &sn, &cs);
      const float c = bf(cs), s = bf(sn);
      const float x0 = outs[m][p], x1 = outs[m][p + 32];
      v = el < 32 ? __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)) : __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
    }
    Q[((size_t)m * MH + h) * Ep + e] = __float2half(e < E ? v : 0.f);
  }
}
void mla_query(torch::Tensor yq, int64_t ldq, torch::Tensor qn, double q_eps, torch::Tensor P, int64_t E, int64_t Ep,
               torch::Tensor pos, torch::Tensor inv_freq, torch::Tensor Q, int64_t M) {
  dim3 grid(MH, (Ep + 63) / 64);
#define MQ(MM)                                                                                                       \
  case MM:                                                                                                           \
    mla_query_k<MM><<<grid, 256, 0, STREAM>>>(CBF(yq), ldq, qn.data_ptr<float>(), (float)q_eps, CBF(P), E, Ep,       \
                                             pos.data_ptr<long>(), inv_freq.data_ptr<float>(),                       \
                                             reinterpret_cast<__half*>(Q.data_ptr()));                                \
    break;
  switch (M) { MQ(1) MQ(2) MQ(3) MQ(4) default: TORCH_CHECK(false, "M <= 4"); }
#undef MQ
}

// ---- split-KV attention straight from the codes. Chunks of 32 cells; chunk c belongs to virtual split c % VS (fixed,
// so a query row's arithmetic never depends on how many query rows, cells or blocks there are). Per chunk: decode the
// rows into an fp16 tile K [32][Es] (raw RoPE key in cols 0..63, latent in 64..64+r), RoPE key un-rotation (x R0, tensor
// cores) and NeoX rotation at the cell's position, then S = Q K^T for every query token (4 key-width quarters x 2
// column tiles per token), per-group inv_r, scale, causal mask, online softmax (lazy rescale), O += P C. 256 threads
// for one query row, 512 for several (identical results: the warp assignment changes no element's operation order).
constexpr int AC = 32, AT = 256;

// Virtual splits (chunk c belongs to split c % VS; a block runs splits b, b + G, .. one after the other, each with
// its own online softmax and partial), so the grid G can shrink for big NQ without changing any row's arithmetic;
// all query tokens' S tiles in one phase, softmax over all 16 NQ rows in one phase, PV for all tokens in one phase;
// codes read as 32-bit words, values converted in pairs.
template <int NQ, int OT, int TH>
__global__ void __launch_bounds__(TH) mla_attn2_k(const __half* __restrict__ Q, int Ep, const uint8_t* __restrict__ cache,
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

  for (int i = tid; i < RD * RD; i += TH) Uh[(i / RD) * Ul + i % RD] = R0h[i];
  if (tid < 16) cbs[tid] = cbl[tid];
  else if (tid < 32) cbs[tid] = cbr[tid - 16];
  for (int i = tid; i < 256; i += TH) rmap[i] = (float)(i / 16);
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
    for (int s2 = tid; s2 < n16; s2 += TH) {
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
    for (int i = tid; i < NQ * 16; i += TH) { mrow[i] = -INFINITY; lrow[i] = 0.f; arow[i] = 1.f; }
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
      for (int jl = warp; jl < AC; jl += TH / 32) {
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
        if (warp < 8) {                                     // 8 tiles (extra warps of a 512-thread block idle)
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
      for (int i = tid; i < AC * 32; i += TH) {
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
      for (int task = warp; task < NQ * 8; task += TH / 32) {
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
      for (int rowi = warp; rowi < NQ * 16; rowi += TH / 32) {
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
          const int ct = warp + (TH / 32) * k;
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
        const int ct = warp + (TH / 32) * k;
        if (ct < nct) wmma::store_matrix_sync(o + ct * 16, O[qi][k], rp, wmma::mem_row_major);
      }
    }
    __syncthreads();
    for (int i = tid; i < NQ * 16; i += TH) {
      part_ml[((size_t)v * NQ * 16 + i) * 2] = mrow[i];
      part_ml[((size_t)v * NQ * 16 + i) * 2 + 1] = lrow[i];
    }
    __syncthreads();
  }
}

static int mla2_smem(int Ep, int RW, int NQ) {
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


void mla_attn(torch::Tensor Q, int64_t Ep, torch::Tensor cache, int64_t RW, torch::Tensor R0h, torch::Tensor cbl,
              torch::Tensor cbr, torch::Tensor pos, torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_ml,
              torch::Tensor out, int64_t r, double scale, int64_t NQ) {
  const int VS = part_ml.size(0);                 // virtual splits (fixed per engine: results never depend on the grid)
  const int rp = Ep - RD;
  const int TH = NQ == 1 ? 256 : 512;            // threads per block; results are identical for both (bit), the
  const int OT = (rp / 16 + TH / 32 - 1) / (TH / 32); // wider block hides more latency once several query rows share it
  const int sm = mla2_smem(Ep, RW, NQ);
  TORCH_CHECK(RW % 16 == 0 && Ep % 16 == 0);
  int dev = 0, smem_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  const int per_sm = max(1, smem_sm / (sm + 1024));
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int G = min(VS, max(1, nsm * min(per_sm, TH == 256 ? 2 : 1)));
#define MA(NN, TT, HH)                                                                                             \
  {                                                                                                                \
    auto kfn = mla_attn2_k<NN, TT, HH>;                                                                            \
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);                                   \
    kfn<<<G, HH, sm, STREAM>>>(reinterpret_cast<const __half*>(Q.data_ptr()), Ep, cache.data_ptr<uint8_t>(), RW,     \
                               reinterpret_cast<const __half*>(R0h.data_ptr()), cbl.data_ptr<float>(),             \
                               cbr.data_ptr<float>(), pos.data_ptr<long>(), inv_freq.data_ptr<float>(),            \
                               part_o.data_ptr<float>(), part_ml.data_ptr<float>(), r, (float)scale, VS);          \
  }
  switch (NQ) {
    case 1: if (OT <= 4) MA(1, 4, 256) else if (OT <= 5) MA(1, 5, 256) else if (OT <= 6) MA(1, 6, 256) else TORCH_CHECK(false, "latent too wide"); break;
    case 2: if (OT <= 2) MA(2, 2, 512) else if (OT <= 3) MA(2, 3, 512) else TORCH_CHECK(false, "latent too wide"); break;
    case 3: if (OT <= 2) MA(3, 2, 512) else if (OT <= 3) MA(3, 3, 512) else TORCH_CHECK(false, "latent too wide"); break;
    case 4: if (OT <= 2) MA(4, 2, 512) else if (OT <= 3) MA(4, 3, 512) else TORCH_CHECK(false, "latent too wide"); break;
    default: TORCH_CHECK(false, "NQ <= 4");
  }
#undef MA
  mla_reduce_k<<<NQ * 16, 128, 0, STREAM>>>(part_o.data_ptr<float>(), part_ml.data_ptr<float>(), out.data_ptr<float>(),
                                           VS, NQ, r, rp);
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
                                                float* __restrict__ part_ml, int r, float scale, int VS, int dbuf) {
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
      __syncthreads();                                                       // S1: rows in place
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
                 torch::Tensor out, int64_t r, double scale, int64_t NQ) {
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
                                part_o.data_ptr<float>(), part_ml.data_ptr<float>(), r, (float)scale, VS, dbuf);   \
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

// V up-projection per group + output gate: o[m][h][d] = sum_c Wv[g][d][c] o_lat[m][h][c] (Wv bf16 (G, D, r));
// y = bf16(bf16(o) * bf16(sigmoid(gate))). One warp per (g, d): the 4 heads of group g and all M tokens.
template <int M>
__global__ void __launch_bounds__(256) mla_vup_k(const float* __restrict__ olat, int r, const __nv_bfloat16* __restrict__ Wv,
                                                 const __nv_bfloat16* __restrict__ yq, int ldq, __nv_bfloat16* __restrict__ y) {
  const int lane = threadIdx.x & 31, gd = blockIdx.x * 8 + (threadIdx.x >> 5);   // gd = g * 256 + d
  if (gd >= MG * 256) return;
  const int g = gd / 256, d = gd % 256;
  const __nv_bfloat16* wr = Wv + (size_t)gd * r;
  float acc[M][4];
#pragma unroll
  for (int m = 0; m < M; m++)
#pragma unroll
    for (int k = 0; k < 4; k++) acc[m][k] = 0.f;
  for (int c = lane; c < r; c += 32) {
    const float wv = __bfloat162float(wr[c]);
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
      for (int k = 0; k < 4; k++) acc[m][k] = __fmaf_rn(wv, olat[((size_t)m * MH + g * 4 + k) * r + c], acc[m][k]);
  }
#pragma unroll
  for (int m = 0; m < M; m++)
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const float v = warp_sum(acc[m][k]);
      if (lane == 0) {
        const int h = g * 4 + k;
        const float gt = __bfloat162float(yq[(size_t)m * ldq + h * 512 + 256 + d]);
        y[(size_t)m * MH * 256 + h * 256 + d] = __float2bfloat16(__fmul_rn(bf(v), bf(sigm(gt))));
      }
    }
}
void mla_vup(torch::Tensor olat, int64_t r, torch::Tensor Wv, torch::Tensor yq, int64_t ldq, torch::Tensor y, int64_t M) {
  dim3 grid(MG * 256 / 8);
#define MV(MM) case MM: mla_vup_k<MM><<<grid, 256, 0, STREAM>>>(olat.data_ptr<float>(), r, CBF(Wv), CBF(yq), ldq, BF(y)); break;
  switch (M) { MV(1) MV(2) MV(3) MV(4) default: TORCH_CHECK(false, "M <= 4"); }
#undef MV
}

// =====================================================================================================================
// MTP (stock Qwen3.5 gated attention, 16 q / 4 kv heads of 256, partial NeoX RoPE on 64 dims) for M rows at positions
// pos[0] + m: q/k RMSNorm (w = 1 + weight), RoPE (accurate sin / cos, bf16-rounded like the HF rotary), gate split.
// q -> fp32 (M, 16, 256) unscaled; k, v -> bf16 straight into the head-major cache (4, Tcap, 256); gate -> bf16.
// y rows: [(q | gate) per head (16 x 512) | k (4 x 256) | v (4 x 256)].
__global__ void mtp_attn_pre_k(const __nv_bfloat16* __restrict__ y, int ldy, const float* __restrict__ qn,
                               const float* __restrict__ kn, const long* __restrict__ pos,
                               const float* __restrict__ inv_freq, float* __restrict__ qo, __nv_bfloat16* __restrict__ kb,
                               __nv_bfloat16* __restrict__ vb, int Tcap, __nv_bfloat16* __restrict__ go, float eps) {
  __shared__ float xs[256];
  __shared__ float sh[8];
  const int m = blockIdx.y, b = blockIdx.x, t = threadIdx.x;      // b < 16: q head; 16..19: kv head
  const bool isq = b < 16;
  const int hh = isq ? b : b - 16;
  const __nv_bfloat16* row = y + (size_t)m * ldy;
  const __nv_bfloat16* src = isq ? row + hh * 512 : row + 16 * 512 + hh * 256;
  const float x = __bfloat162float(src[t]);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum<256>(__fmul_rn(x, x), sh), 256.f), eps));
  const float xn = bf(__fmul_rn(__fmul_rn(x, r), isq ? qn[t] : kn[t]));
  xs[t] = xn;
  __syncthreads();
  float o = xn;
  const long p = pos[0] + m;
  if (t < 64) {
    const int f = t % 32;
    float sn, cs;
    sincos_acc(__fmul_rn((float)p, inv_freq[f]), &sn, &cs);
    const float c = bf(cs), s = bf(sn);
    const float other = t < 32 ? -xs[t + 32] : xs[t - 32];
    o = bf(__fadd_rn(__fmul_rn(xn, c), __fmul_rn(other, s)));
  }
  if (isq) {
    qo[((size_t)m * 16 + hh) * 256 + t] = o;
    go[(size_t)m * 4096 + hh * 256 + t] = src[256 + t];
  } else {
    kb[((size_t)hh * Tcap + p) * 256 + t] = __float2bfloat16(o);
    vb[((size_t)hh * Tcap + p) * 256 + t] = row[16 * 512 + 4 * 256 + hh * 256 + t];
  }
}
void mtp_attn_pre(torch::Tensor y, int64_t ldy, torch::Tensor qn, torch::Tensor kn, torch::Tensor pos,
                  torch::Tensor inv_freq, torch::Tensor qo, torch::Tensor kb, torch::Tensor vb, torch::Tensor go,
                  double eps, int64_t M) {
  mtp_attn_pre_k<<<dim3(20, M), 256, 0, STREAM>>>(CBF(y), ldy, qn.data_ptr<float>(), kn.data_ptr<float>(),
      pos.data_ptr<long>(), inv_freq.data_ptr<float>(), qo.data_ptr<float>(), BF(kb), BF(vb), kb.size(1), BF(go), (float)eps);
}

// =====================================================================================================================
// MTP cache in compressed form: per (kv head, token) row of 256 values, symmetric int8 (one fp16 scale per row) or int4
// (one fp16 scale per 32 values, codes two per byte, low nibble first, value = (q - 8) * s). K/V arrays (Hkv, Tcap, row
// bytes), scales (Hkv, Tcap, nsc). quantization: s = absmax / 127 (int8) or absmax / 7 (int4, q = round(x / s) + 8).
template <int BITS>
__device__ __forceinline__ void kvq_store_row(const float* x, int lane, uint8_t* codes, __half* scales) {
  // a full warp holds 256 values, lane owns x[0..7] = values 8 lane .. 8 lane + 7
  constexpr int NSC = BITS == 8 ? 1 : 8;
  float a = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) a = fmaxf(a, fabsf(x[e]));
  if (BITS == 8) {
#pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  } else {
#pragma unroll
    for (int o = 2; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));     // 4 lanes = 32 values
  }
  const __half sh = __float2half(a / (BITS == 8 ? 127.f : 7.f));
  const float sc = __half2float(sh), inv = sc > 0.f ? 1.f / sc : 0.f;
  if (BITS == 8) {
    uint32_t w0 = 0, w1 = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      w0 |= (uint32_t)(uint8_t)(int8_t)max(-127, min(127, __float2int_rn(x[e] * inv))) << (8 * e);
      w1 |= (uint32_t)(uint8_t)(int8_t)max(-127, min(127, __float2int_rn(x[4 + e] * inv))) << (8 * e);
    }
    reinterpret_cast<uint2*>(codes)[lane] = make_uint2(w0, w1);
    if (lane == 0) scales[0] = sh;
  } else {
    uint32_t w = 0;
#pragma unroll
    for (int e = 0; e < 8; e++) w |= (uint32_t)max(0, min(15, __float2int_rn(x[e] * inv) + 8)) << (4 * e);
    reinterpret_cast<uint32_t*>(codes)[lane] = w;
    if ((lane & 3) == 0) scales[lane >> 2] = sh;
  }
  (void)NSC;
}

// MTP attention prep with a quantized cache: as mtp_attn_pre_k but K / V rows are quantized into (kc, ks), (vc, vs)
template <int BITS>
__global__ void mtp_attn_pre_q_k(const __nv_bfloat16* __restrict__ y, int ldy, const float* __restrict__ qn,
                                 const float* __restrict__ kn, const long* __restrict__ pos,
                                 const float* __restrict__ inv_freq, float* __restrict__ qo, uint8_t* __restrict__ kc,
                                 __half* __restrict__ ksc, uint8_t* __restrict__ vc, __half* __restrict__ vsc, int Tcap,
                                 __nv_bfloat16* __restrict__ go, float eps) {
  constexpr int RB = BITS == 8 ? 256 : 128, NSC = BITS == 8 ? 1 : 8;
  __shared__ float xs[256];
  __shared__ float outs[256];
  __shared__ float sh[8];
  const int m = blockIdx.y, b = blockIdx.x, t = threadIdx.x;
  const bool isq = b < 16;
  const int hh = isq ? b : b - 16;
  const __nv_bfloat16* row = y + (size_t)m * ldy;
  const __nv_bfloat16* src = isq ? row + hh * 512 : row + 16 * 512 + hh * 256;
  const float x = __bfloat162float(src[t]);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum<256>(__fmul_rn(x, x), sh), 256.f), eps));
  const float xn = bf(__fmul_rn(__fmul_rn(x, r), isq ? qn[t] : kn[t]));
  xs[t] = xn;
  __syncthreads();
  float o = xn;
  const long p = pos[0] + m;
  if (t < 64) {
    const int f = t % 32;
    float sn, cs;
    sincos_acc(__fmul_rn((float)p, inv_freq[f]), &sn, &cs);
    const float c = bf(cs), s = bf(sn);
    const float other = t < 32 ? -xs[t + 32] : xs[t - 32];
    o = bf(__fadd_rn(__fmul_rn(xn, c), __fmul_rn(other, s)));
  }
  if (isq) {
    qo[((size_t)m * 16 + hh) * 256 + t] = o;
    go[(size_t)m * 4096 + hh * 256 + t] = src[256 + t];
    return;
  }
  outs[t] = __bfloat162float(__float2bfloat16(o));                 // the bf16 key the dense cache would hold
  xs[t] = __bfloat162float(row[16 * 512 + 4 * 256 + hh * 256 + t]);
  __syncthreads();
  if (t < 64) {                                                    // warp 0: key row, warp 1: value row
    const int lane = t & 31;
    const float* srcv = t < 32 ? outs : xs;
    float v8[8];
#pragma unroll
    for (int e = 0; e < 8; e++) v8[e] = srcv[8 * lane + e];
    const size_t ro = (size_t)hh * Tcap + p;
    if (t < 32) kvq_store_row<BITS>(v8, lane, kc + ro * RB, ksc + ro * NSC);
    else kvq_store_row<BITS>(v8, lane, vc + ro * RB, vsc + ro * NSC);
  }
}
void mtp_attn_pre_q(torch::Tensor y, int64_t ldy, torch::Tensor qn, torch::Tensor kn, torch::Tensor pos,
                    torch::Tensor inv_freq, torch::Tensor qo, torch::Tensor kc, torch::Tensor ks, torch::Tensor vc,
                    torch::Tensor vs, torch::Tensor go, double eps, int64_t M, int64_t bits) {
  const int Tcap = kc.size(1);
  auto f = bits == 8 ? mtp_attn_pre_q_k<8> : mtp_attn_pre_q_k<4>;
  f<<<dim3(20, M), 256, 0, STREAM>>>(CBF(y), ldy, qn.data_ptr<float>(), kn.data_ptr<float>(), pos.data_ptr<long>(),
      inv_freq.data_ptr<float>(), qo.data_ptr<float>(), kc.data_ptr<uint8_t>(), reinterpret_cast<__half*>(ks.data_ptr()),
      vc.data_ptr<uint8_t>(), reinterpret_cast<__half*>(vs.data_ptr()), Tcap, BF(go), (float)eps);
}

// decode attention over the quantized MTP cache: q (16, 256) fp32 unscaled, GQA 4; length pos[0] + 1. Split-K: grid
// (NS, 4 kv heads), 4 warps; a warp walks tokens t0 + w, + 4, ..; lane holds 8 dims of the 4 query heads.
template <int BITS>
__global__ void __launch_bounds__(128) kvq_attn_split_k(const float* __restrict__ q, const uint8_t* __restrict__ kc,
                                                        const __half* __restrict__ ksc, const uint8_t* __restrict__ vc,
                                                        const __half* __restrict__ vsc, const long* __restrict__ pos,
                                                        int Tcap, float* __restrict__ pm, float* __restrict__ pl,
                                                        float* __restrict__ pacc, float scale) {
  constexpr int RB = BITS == 8 ? 256 : 128, NSC = BITS == 8 ? 1 : 8;
  __shared__ float sm_m[4][4], sm_l[4][4], sm_acc[4][4][256];
  const int s = blockIdx.x, NS = gridDim.x, h = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (int)pos[0] + 1;
  const int chunk = (n + NS - 1) / NS, t0 = s * chunk, t1 = min(n, t0 + chunk);
  float qv[4][8];
#pragma unroll
  for (int g = 0; g < 4; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) qv[g][e] = q[(h * 4 + g) * 256 + lane * 8 + e] * scale;
  float m[4], l[4], acc[4][8];
#pragma unroll
  for (int g = 0; g < 4; g++) { m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) acc[g][e] = 0.f; }
  for (int t = t0 + w; t < t1; t += 4) {
    const size_t ro = (size_t)h * Tcap + t;
    float kf[8], vf[8];
    if (BITS == 8) {
      const uint2 ku = reinterpret_cast<const uint2*>(kc + ro * RB)[lane], vu = reinterpret_cast<const uint2*>(vc + ro * RB)[lane];
      const float ks = __half2float(ksc[ro]), vs = __half2float(vsc[ro]);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        kf[e] = (float)(int8_t)(ku.x >> (8 * e)) * ks; kf[4 + e] = (float)(int8_t)(ku.y >> (8 * e)) * ks;
        vf[e] = (float)(int8_t)(vu.x >> (8 * e)) * vs; vf[4 + e] = (float)(int8_t)(vu.y >> (8 * e)) * vs;
      }
    } else {
      const uint32_t ku = reinterpret_cast<const uint32_t*>(kc + ro * RB)[lane], vu = reinterpret_cast<const uint32_t*>(vc + ro * RB)[lane];
      const float ks = __half2float(ksc[ro * NSC + (lane >> 2)]), vs = __half2float(vsc[ro * NSC + (lane >> 2)]);
#pragma unroll
      for (int e = 0; e < 8; e++) { kf[e] = (float)((int)((ku >> (4 * e)) & 15) - 8) * ks; vf[e] = (float)((int)((vu >> (4 * e)) & 15) - 8) * vs; }
    }
    float sc[4];
#pragma unroll
    for (int g = 0; g < 4; g++) { float d = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) d = fmaf(qv[g][e], kf[e], d);
      sc[g] = d; }
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int g = 0; g < 4; g++) sc[g] += __shfl_xor_sync(0xffffffff, sc[g], o);
#pragma unroll
    for (int g = 0; g < 4; g++) {
      const float mn = fmaxf(m[g], sc[g]), c = __expf(m[g] - mn), p = __expf(sc[g] - mn);
      l[g] = l[g] * c + p; m[g] = mn;
#pragma unroll
      for (int e = 0; e < 8; e++) acc[g][e] = fmaf(acc[g][e], c, p * vf[e]);
    }
  }
#pragma unroll
  for (int g = 0; g < 4; g++) {
    if (lane == 0) { sm_m[w][g] = m[g]; sm_l[w][g] = l[g]; }
#pragma unroll
    for (int e = 0; e < 8; e++) sm_acc[w][g][lane * 8 + e] = acc[g][e];
  }
  __syncthreads();
  for (int g = 0; g < 4; g++) {
    float M = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; i++) M = fmaxf(M, sm_m[i][g]);
    float L = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float f = sm_m[i][g] == -INFINITY ? 0.f : __expf(sm_m[i][g] - M);
      L += sm_l[i][g] * f; a0 += sm_acc[i][g][threadIdx.x] * f; a1 += sm_acc[i][g][threadIdx.x + 128] * f;
    }
    const size_t o = ((size_t)s * gridDim.y + h) * 4 + g;
    if (threadIdx.x == 0) { pm[o] = M; pl[o] = L; }
    pacc[o * 256 + threadIdx.x] = a0; pacc[o * 256 + threadIdx.x + 128] = a1;
  }
}
__global__ void kvq_combine_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                              float* __restrict__ out, int NS, int HG) {
  const int hg = blockIdx.x, d = threadIdx.x;
  float M = -INFINITY;
  for (int s = 0; s < NS; s++) M = fmaxf(M, pm[s * HG + hg]);
  float L = 0.f, a = 0.f;
  for (int s = 0; s < NS; s++) {
    const float mm = pm[s * HG + hg];
    if (mm == -INFINITY) continue;
    const float f = __expf(mm - M);
    L += pl[s * HG + hg] * f; a += pacc[((size_t)s * HG + hg) * 256 + d] * f;
  }
  out[hg * 256 + d] = a / L;
}
void kvq_attn(torch::Tensor q, torch::Tensor kc, torch::Tensor ks, torch::Tensor vc, torch::Tensor vs, torch::Tensor pos,
              torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc, torch::Tensor out, double scale, int64_t bits) {
  const int NS = pm.size(0), Tcap = kc.size(1);
  dim3 grid(NS, 4);
  if (bits == 8)
    kvq_attn_split_k<8><<<grid, 128, 0, STREAM>>>(q.data_ptr<float>(), kc.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(ks.data_ptr()),
        vc.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(vs.data_ptr()), pos.data_ptr<long>(), Tcap, pm.data_ptr<float>(),
        pl.data_ptr<float>(), pacc.data_ptr<float>(), (float)scale);
  else
    kvq_attn_split_k<4><<<grid, 128, 0, STREAM>>>(q.data_ptr<float>(), kc.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(ks.data_ptr()),
        vc.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(vs.data_ptr()), pos.data_ptr<long>(), Tcap, pm.data_ptr<float>(),
        pl.data_ptr<float>(), pacc.data_ptr<float>(), (float)scale);
  kvq_combine_k<<<16, 256, 0, STREAM>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(), out.data_ptr<float>(), NS, 16);
}

// NF4 (dp4a layout: per row, 16-byte words of 32 values, nibble i of 32-bit word j = value 8j + i) -> bf16, with the
// exact bitsandbytes levels (code table) times the block absmax: bf16(L * absmax) like bnb dequantize_4bit.
__global__ void nf4_dequant_k(const uint32_t* __restrict__ W, const float* __restrict__ absmax,
                              const float* __restrict__ code, __nv_bfloat16* __restrict__ out, long n_words, int K,
                              float amul) {
  const long w = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= n_words) return;
  const uint32_t v = W[w];
  const long e0 = w * 8;
  const long row = e0 / K, col = e0 % K;
  const float a = __fmul_rn(absmax[row * (K / 64) + col / 64], amul);
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const __nv_bfloat16 x0 = __float2bfloat16(__fmul_rn(code[(v >> (8 * i)) & 15], a));
    const __nv_bfloat16 x1 = __float2bfloat16(__fmul_rn(code[(v >> (8 * i + 4)) & 15], a));
    o[i] = (uint32_t)__bfloat16_as_ushort(x0) | ((uint32_t)__bfloat16_as_ushort(x1) << 16);
  }
  reinterpret_cast<uint4*>(out)[w] = make_uint4(o[0], o[1], o[2], o[3]);
}
// absmax = scales x amul (the engine keeps absmax / 16256: amul = 16256, within one fp32 ulp of the stored absmax)
void nf4_dequant(torch::Tensor codes, torch::Tensor absmax, torch::Tensor code, torch::Tensor out, double amul) {
  const long n_words = codes.numel();
  const int K = codes.size(1) * 32;
  nf4_dequant_k<<<(n_words + 255) / 256, 256, 0, STREAM>>>(reinterpret_cast<const uint32_t*>(codes.data_ptr()),
      absmax.data_ptr<float>(), code.data_ptr<float>(), BF(out), n_words, K, (float)amul);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mtp_attn_pre", &mtp_attn_pre); m.def("mtp_attn_pre_q", &mtp_attn_pre_q); m.def("kvq_attn", &kvq_attn);
  m.def("nf4_gemv", &nf4_gemv); m.def("nf4_dequant", &nf4_dequant); m.def("nf4_mma", &nf4_mma); m.def("nf4_set_tables", &nf4_set_tables);
  m.def("rms_rows", &rms_rows); m.def("quant_rows", &quant_rows); m.def("silu_mul_q", &silu_mul_q);
  m.def("dgemv", &dgemv);
  m.def("gdn2_gates", &gdn2_gates); m.def("tc_skinny", &tc_skinny); m.def("gdn2_gates_tc", &gdn2_gates_tc); m.def("gdn2_gates_epi", &gdn2_gates_epi); m.def("gdn2_conv", &gdn2_conv); m.def("gdn2_conv_commit", &gdn2_conv_commit);
  m.def("gdn2_rec", &gdn2_rec); m.def("gdn2_rec_fused", &gdn2_rec_fused); m.def("gdn2_conv_fused", &gdn2_conv_fused);
  m.def("emb_nf4", &emb_nf4); m.def("emb_nf4_mma", &emb_nf4_mma); m.def("argmax_rows", &argmax_rows);
  m.def("mla_encode", &mla_encode); m.def("mla_query", &mla_query); m.def("mla_attn", &mla_attn); m.def("mla_attn_fd", &mla_attn_fd); m.def("mla_vup", &mla_vup);
}
