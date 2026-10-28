// Single-token (M = 1) weight-only GEMVs for decode: y = W x, x and y bf16.
//   w4: int4, groups of 32 along K, one fp16 scale per group, symmetric: w = (q - 8) * s, q in 0..15 (llama.cpp Q4_0
//       numerics). Layout: codes (N, K/32) uint4 (one 16-byte word = one group, nibble i of 32-bit word j = element
//       8j + i), scales (N, K/32) half.
//   w8: int8 per output row: w = q * s[row]. Layout: codes (N, K/16) uint4 (16 int8 each), scales (N,) float.
// One warp computes R rows; lanes stride over K so loads are 512-byte coalesced; x is read once per group into registers
// and reused for the R rows. Optional fused residual: y = W x + res.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr int WARPS = 4;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ __forceinline__ void load_x32(const __nv_bfloat16* x, float* xf) {      // 32 bf16 -> float registers
  const uint4* p = reinterpret_cast<const uint4*>(x);
#pragma unroll
  for (int c = 0; c < 4; c++) {
    uint4 u = __ldg(p + c);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int e = 0; e < 4; e++) { float2 f = __bfloat1622float2(b[e]); xf[c * 8 + 2 * e] = f.x; xf[c * 8 + 2 * e + 1] = f.y; }
  }
}

// nibble -> float without I2F (quarter rate on this GPU): exponent bits of 2^23 OR'ed with q, minus 2^23 (exact).
__device__ __forceinline__ float nib(uint32_t w, int i) {
  return __int_as_float(((w >> (4 * i)) & 15u) | 0x4B000000u) - 8388608.f;
}

template <int R, int U>
__global__ void __launch_bounds__(WARPS * 32) w4_k(const uint4* __restrict__ W, const half* __restrict__ S,
                                                    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ res,
                                                    __nv_bfloat16* __restrict__ Y, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;
  const int G = K >> 5;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  for (int g0 = lane * U; g0 < G; g0 += 32 * U) {           // U consecutive groups per lane per step (G % U == 0)
    uint4 p[U][R];
    float sc[U][R];
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < R; r++) {
        const int row = min(row0 + r, N - 1);
        p[u][r] = __ldg(W + (size_t)row * G + g0 + u);
        sc[u][r] = __half2float(__ldg(S + (size_t)row * G + g0 + u));
      }
#pragma unroll
    for (int u = 0; u < U; u++) {
      float xf[32];
      load_x32(X + (g0 + u) * 32, xf);
      float xs = 0.f;
#pragma unroll
      for (int e = 0; e < 32; e++) xs += xf[e];
#pragma unroll
      for (int r = 0; r < R; r++) {
        const uint32_t wd[4] = {p[u][r].x, p[u][r].y, p[u][r].z, p[u][r].w};
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int i = 0; i < 8; i++) part = fmaf(nib(wd[j], i), xf[j * 8 + i], part);
        acc[r] = fmaf(sc[u][r], part - 8.f * xs, acc[r]);  // sum((q - 8) x) = sum(q x) - 8 sum(x)
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    float v = warp_sum(acc[r]);
    const int row = row0 + r;
    if (lane == 0 && row < N) Y[row] = __float2bfloat16(res ? v + __bfloat162float(res[row]) : v);
  }
}

// ---- llama.cpp-style Q4_0 x Q8 path: x quantized to int8 per 32-group (one warp per group), packed so that word 2j holds
// elements 8j+0,2,4,6 and word 2j+1 elements 8j+1,3,5,7 (matching (w >> 0/4) & 0x0F0F0F0F of the nibble word j).
__global__ void quant_x_k(const __nv_bfloat16* __restrict__ X, int8_t* __restrict__ XQ, float2* __restrict__ XD, int K) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= K) return;
  const float x = __bfloat162float(X[e]);
  float a = fabsf(x);
#pragma unroll
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  const float d = a / 127.f;
  const int q = d > 0.f ? __float2int_rn(x / d) : 0;
  int sq = q;
#pragma unroll
  for (int o = 16; o; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
  const int g = e >> 5, i = e & 7, j = (e & 31) >> 3;
  XQ[g * 32 + j * 8 + (i & 1) * 4 + (i >> 1)] = (int8_t)q;
  if ((e & 31) == 0) XD[g] = make_float2(d, d * (float)sq);
}

template <int R, int U>
__global__ void __launch_bounds__(WARPS * 32) w4q_k(const uint4* __restrict__ W, const half* __restrict__ S,
                                                     const uint4* __restrict__ XQ, const float2* __restrict__ XD,
                                                     const __nv_bfloat16* __restrict__ res, __nv_bfloat16* __restrict__ Y,
                                                     int N, int K) {
  // U groups per lane per step: R * U weight words in flight per lane before any arithmetic
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;
  const int G = K >> 5;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  for (int g0 = lane; g0 < G; g0 += 32 * U) {
    uint4 p[U][R];
    float sc[U][R];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int g = g0 + 32 * u;
#pragma unroll
      for (int r = 0; r < R; r++) {
        const int row = min(row0 + r, N - 1);
        if (g < G) {
          p[u][r] = __ldg(W + (size_t)row * G + g);
          sc[u][r] = __half2float(__ldg(S + (size_t)row * G + g));
        } else {
          p[u][r] = make_uint4(0, 0, 0, 0);
          sc[u][r] = 0.f;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int g = g0 + 32 * u;
      if (g >= G) break;
      const uint4 x0 = __ldg(XQ + 2 * g), x1 = __ldg(XQ + 2 * g + 1);
      const float2 xd = __ldg(XD + g);
      const int xw[8] = {(int)x0.x, (int)x0.y, (int)x0.z, (int)x0.w, (int)x1.x, (int)x1.y, (int)x1.z, (int)x1.w};
#pragma unroll
      for (int r = 0; r < R; r++) {
        const uint32_t wd[4] = {p[u][r].x, p[u][r].y, p[u][r].z, p[u][r].w};
        int d = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          d = __dp4a((int)(wd[j] & 0x0F0F0F0Fu), xw[2 * j], d);
          d = __dp4a((int)((wd[j] >> 4) & 0x0F0F0F0Fu), xw[2 * j + 1], d);
        }
        acc[r] = fmaf(sc[u][r], xd.x * (float)d - 8.f * xd.y, acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    float v = warp_sum(acc[r]);
    const int row = row0 + r;
    if (lane == 0 && row < N) Y[row] = __float2bfloat16(res ? v + __bfloat162float(res[row]) : v);
  }
}

template <int R>
__global__ void __launch_bounds__(WARPS * 32) w8_k(const uint4* __restrict__ W, const float* __restrict__ S,
                                                    const __nv_bfloat16* __restrict__ X, float* __restrict__ Y, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;
  const int G = K >> 4;                                     // 16 int8 per uint4
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  for (int g = lane; g < G; g += 32) {
    float xf[16];
    const uint4* p = reinterpret_cast<const uint4*>(X + g * 16);
#pragma unroll
    for (int c = 0; c < 2; c++) {
      uint4 u = __ldg(p + c);
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int e = 0; e < 4; e++) { float2 f = __bfloat1622float2(b[e]); xf[c * 8 + 2 * e] = f.x; xf[c * 8 + 2 * e + 1] = f.y; }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int row = min(row0 + r, N - 1);
      uint4 q = __ldg(W + (size_t)row * G + g);
      const int8_t* c8 = reinterpret_cast<const int8_t*>(&q);
      float part = 0.f;
#pragma unroll
      for (int e = 0; e < 16; e++) part += (float)c8[e] * xf[e];
      acc[r] += part;
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    float v = warp_sum(acc[r]);
    const int row = row0 + r;
    if (lane == 0 && row < N) Y[row] = v * S[row];
  }
}

template <int R, int U>
static void w4_launch(torch::Tensor& codes, torch::Tensor& scales, torch::Tensor& x, const __nv_bfloat16* res, torch::Tensor& y) {
  const int N = codes.size(0), K = codes.size(1) * 32;
  TORCH_CHECK((K / 32) % U == 0, "K/32 must be a multiple of U");
  dim3 grid((N + WARPS * R - 1) / (WARPS * R));
  w4_k<R, U><<<grid, WARPS * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(codes.data_ptr()), reinterpret_cast<const half*>(scales.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), res, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K);
}

void w4(torch::Tensor codes, torch::Tensor scales, torch::Tensor x, c10::optional<torch::Tensor> res, torch::Tensor y,
        int64_t variant) {
  const __nv_bfloat16* r = res ? reinterpret_cast<const __nv_bfloat16*>(res->data_ptr()) : nullptr;
  switch (variant) {
    case 0: w4_launch<4, 1>(codes, scales, x, r, y); break;
    case 1: w4_launch<4, 2>(codes, scales, x, r, y); break;
    case 2: w4_launch<8, 1>(codes, scales, x, r, y); break;
    case 3: w4_launch<8, 2>(codes, scales, x, r, y); break;
    case 4: w4_launch<2, 4>(codes, scales, x, r, y); break;
    case 5: w4_launch<16, 1>(codes, scales, x, r, y); break;
    default: w4_launch<2, 2>(codes, scales, x, r, y); break;
  }
}

void quant_x(torch::Tensor x, torch::Tensor xq, torch::Tensor xd) {
  const int K = x.numel();
  quant_x_k<<<(K + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), xq.data_ptr<int8_t>(), reinterpret_cast<float2*>(xd.data_ptr()), K);
}

template <int R, int U>
static void w4q_launch(torch::Tensor& codes, torch::Tensor& scales, torch::Tensor& xq, torch::Tensor& xd,
                       const __nv_bfloat16* res, torch::Tensor& y) {
  const int N = codes.size(0), K = codes.size(1) * 32;
  dim3 grid((N + WARPS * R - 1) / (WARPS * R));
  w4q_k<R, U><<<grid, WARPS * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(codes.data_ptr()), reinterpret_cast<const half*>(scales.data_ptr()),
      reinterpret_cast<const uint4*>(xq.data_ptr()), reinterpret_cast<const float2*>(xd.data_ptr()), res,
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K);
}

void w4q(torch::Tensor codes, torch::Tensor scales, torch::Tensor xq, torch::Tensor xd, c10::optional<torch::Tensor> res,
         torch::Tensor y, int64_t rows) {
  const __nv_bfloat16* r = res ? reinterpret_cast<const __nv_bfloat16*>(res->data_ptr()) : nullptr;
  // rows encodes R * 10 + U (U = groups in flight per lane); plain R means U = 1
  switch (rows) {
    case 2: w4q_launch<2, 1>(codes, scales, xq, xd, r, y); break;
    case 4: w4q_launch<4, 1>(codes, scales, xq, xd, r, y); break;
    case 16: w4q_launch<16, 1>(codes, scales, xq, xd, r, y); break;
    case 22: w4q_launch<2, 2>(codes, scales, xq, xd, r, y); break;
    case 24: w4q_launch<2, 4>(codes, scales, xq, xd, r, y); break;
    case 42: w4q_launch<4, 2>(codes, scales, xq, xd, r, y); break;
    case 44: w4q_launch<4, 4>(codes, scales, xq, xd, r, y); break;
    case 82: w4q_launch<8, 2>(codes, scales, xq, xd, r, y); break;
    default: w4q_launch<8, 1>(codes, scales, xq, xd, r, y); break;
  }
}

void w8(torch::Tensor codes, torch::Tensor scales, torch::Tensor x, torch::Tensor y) {
  const int N = codes.size(0), K = codes.size(1);                     // int8 (N, K)
  constexpr int R = 4;
  dim3 grid((N + WARPS * R - 1) / (WARPS * R));
  w8_k<R><<<grid, WARPS * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(codes.data_ptr()), scales.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), y.data_ptr<float>(), N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4", &w4); m.def("w8", &w8); m.def("quant_x", &quant_x); m.def("w4q", &w4q);
}
