// Kernels for mercurius-packed-v1 checkpoints (Mercurius 0.5): Q4_0 / Q8_0 GEMVs for M <= 4 rows and the Q4_0 embedding.
//
// Weight layouts are the checkpoint's own (no repacking):
//   q4_0: weight_packed (N, K/8) int32 = codes (N, K/32) uint4: one 16-byte word per group of 32, nibble i of 32-bit word
//         j = element 8j + i, stored code q in 0..15; weight_scale (N, K/32) fp16; w = (q - 8) * s.
//   q8_0: weight_q8 (N, K) int8 = codes (N, K/16) uint4, two words per group; weight_scale (N, K/32) fp16; w = q * s.
// Every weight is used as bf16((q - 8) * s) (resp. bf16(q * s)): the bf16 tensor the pipeline reconstructs at load and
// the QAT forward trained with. Activations x are bf16, accumulation fp32, output bf16 (or fp32 for logits).
//
// Bit identity across M: each (row, m) dot product runs the same instruction sequence whatever M is (groups in the same
// order, the same warp reduction), so the M-row verification pass of speculative decoding reproduces plain decoding.
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

__device__ __forceinline__ void load_x32(const __nv_bfloat16* x, float* xf) {
  const uint4* p = reinterpret_cast<const uint4*>(x);
#pragma unroll
  for (int c = 0; c < 4; c++) {
    uint4 u = __ldg(p + c);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int e = 0; e < 4; e++) { float2 f = __bfloat1622float2(b[e]); xf[c * 8 + 2 * e] = f.x; xf[c * 8 + 2 * e + 1] = f.y; }
  }
}

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// 32 bf16-rounded weights of one Q4_0 group
__device__ __forceinline__ void q4_group(const uint4& p, float s, float* w) {
  const uint32_t wd[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int i = 0; i < 8; i++) w[j * 8 + i] = bfr((float)((int)((wd[j] >> (4 * i)) & 15u) - 8) * s);
}

__device__ __forceinline__ void q8_group(const uint4& a, const uint4& b, float s, float* w) {
  const int8_t* ca = reinterpret_cast<const int8_t*>(&a);
  const int8_t* cb = reinterpret_cast<const int8_t*>(&b);
#pragma unroll
  for (int e = 0; e < 16; e++) { w[e] = bfr((float)ca[e] * s); w[16 + e] = bfr((float)cb[e] * s); }
}

// One warp computes R rows for all M token rows; lanes stride over the K/32 groups (coalesced 512-byte weight loads).
// FMT 0: q4_0 (one uint4 per group), 1: q8_0 (two). OUT: __nv_bfloat16 or float.
template <int FMT, int R, int M, typename OUT>
__global__ void __launch_bounds__(WARPS * 32) gemv_k(const uint4* __restrict__ W, const half* __restrict__ S,
                                                     const __nv_bfloat16* __restrict__ X, int ldx,
                                                     const __nv_bfloat16* __restrict__ res, int ldr,
                                                     OUT* __restrict__ Y, int ldy, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;
  const int G = K >> 5;
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) acc[r][m] = 0.f;
  for (int g = lane; g < G; g += 32) {
    float w[R][32];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int row = min(row0 + r, N - 1);
      const float s = __half2float(__ldg(S + (size_t)row * G + g));
      if (FMT == 0) q4_group(__ldg(W + (size_t)row * G + g), s, w[r]);
      else q8_group(__ldg(W + ((size_t)row * G + g) * 2), __ldg(W + ((size_t)row * G + g) * 2 + 1), s, w[r]);
    }
#pragma unroll
    for (int m = 0; m < M; m++) {
      float xf[32];
      load_x32(X + (size_t)m * ldx + g * 32, xf);
#pragma unroll
      for (int r = 0; r < R; r++) {
        float part = 0.f;
#pragma unroll
        for (int e = 0; e < 32; e++) part = fmaf(w[r][e], xf[e], part);
        acc[r][m] += part;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) {
      float v = warp_sum(acc[r][m]);
      const int row = row0 + r;
      if (lane == 0 && row < N) {
        if (res) v += __bfloat162float(res[(size_t)m * ldr + row]);
        if constexpr (std::is_same<OUT, float>::value) Y[(size_t)m * ldy + row] = v;
        else Y[(size_t)m * ldy + row] = __float2bfloat16(v);
      }
    }
}

template <int FMT, int R, typename OUT>
static void launch(torch::Tensor& codes, torch::Tensor& scales, torch::Tensor& x, int ldx, const __nv_bfloat16* res,
                   int ldr, torch::Tensor& y, int ldy, int N, int K, int M) {
  dim3 grid((N + WARPS * R - 1) / (WARPS * R));
  auto st = c10::cuda::getCurrentCUDAStream();
  auto W = reinterpret_cast<const uint4*>(codes.data_ptr());
  auto S = reinterpret_cast<const half*>(scales.data_ptr());
  auto X = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto Y = reinterpret_cast<OUT*>(y.data_ptr());
  switch (M) {
    case 1: gemv_k<FMT, R, 1, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
    case 2: gemv_k<FMT, R, 2, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
    case 3: gemv_k<FMT, R, 3, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
    default: gemv_k<FMT, R, 4, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
  }
}

// y[m, :N] (+= res) = W x[m] for m < M. fmt 0 = q4_0, 1 = q8_0. R = rows per warp (1, 2 or 4).
void packed_gemv(torch::Tensor codes, torch::Tensor scales, int64_t fmt, torch::Tensor x, int64_t ldx,
                 c10::optional<torch::Tensor> res, int64_t ldr, torch::Tensor y, int64_t ldy, int64_t M, int64_t R) {
  const int N = scales.size(0), K = scales.size(1) * 32;
  TORCH_CHECK(M >= 1 && M <= 4, "M must be 1..4");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
  const __nv_bfloat16* r = res ? reinterpret_cast<const __nv_bfloat16*>(res->data_ptr()) : nullptr;
  const bool f32 = y.scalar_type() == at::kFloat;
#define GO(F, RR) do { if (f32) launch<F, RR, float>(codes, scales, x, ldx, r, ldr, y, ldy, N, K, M); \
                       else launch<F, RR, __nv_bfloat16>(codes, scales, x, ldx, r, ldr, y, ldy, N, K, M); } while (0)
  if (fmt == 0) { if (R == 1) GO(0, 1); else if (R == 2) GO(0, 2); else GO(0, 4); }
  else { if (R == 1) GO(1, 1); else if (R == 2) GO(1, 2); else GO(1, 4); }
#undef GO
}

// embedding rows of the Q4_0 table: y[m] = bf16((q - 8) * s) for token tok[m]
__global__ void emb_q4_k(const long* __restrict__ tok, const uint4* __restrict__ W, const half* __restrict__ S,
                         __nv_bfloat16* __restrict__ Y, int K) {
  const int m = blockIdx.y, G = K >> 5;
  const long t = tok[m];
  for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < G; g += gridDim.x * blockDim.x) {
    float w[32];
    q4_group(__ldg(W + (size_t)t * G + g), __half2float(__ldg(S + (size_t)t * G + g)), w);
#pragma unroll
    for (int e = 0; e < 32; e++) Y[(size_t)m * K + g * 32 + e] = __float2bfloat16(w[e]);
  }
}

void emb_q4(torch::Tensor tok, torch::Tensor codes, torch::Tensor scales, torch::Tensor y, int64_t M) {
  const int K = scales.size(1) * 32;
  dim3 grid((K / 32 + 63) / 64, M);
  emb_q4_k<<<grid, 64, 0, c10::cuda::getCurrentCUDAStream()>>>(tok.data_ptr<long>(),
      reinterpret_cast<const uint4*>(codes.data_ptr()), reinterpret_cast<const half*>(scales.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), K);
}


// ---- CARE-latent MLA decode attention over the exact per-token cache (k_rope | c | 1/rms per group), all heads at once.
// score[h, t] = scale * ir[t, g(h)] * (qr[h] . kr[t] + ql[h] . c[t]),  o_lat[h] = sum_t softmax(score[h])_t c[t].
// Split over T in chunks of CH tokens (one block each; blocks past the device-side length n exit, so the launch shape is
// static and the kernel lives inside a CUDA graph); a second kernel merges the chunk partials. The latent c and k_rope
// rows are bf16 (their values are bf16 in the model), loaded once per chunk for all heads.
constexpr int AH = 16, ARD = 64, ACH = 32, AWARPS = 8, AHPW = AH / AWARPS;   // heads per warp = 2; rank AR: template

template <int AR>
__global__ void __launch_bounds__(AWARPS * 32) mla_lat_attn_k(
    const float* __restrict__ qr, const float* __restrict__ ql, const __nv_bfloat16* __restrict__ kr,
    const __nv_bfloat16* __restrict__ c, const float* __restrict__ ir, const long* __restrict__ npos, int G, float scale,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
  const int n = (int)npos[0] + 1;                  // tokens 0 .. pos
  const int t0 = blockIdx.x * ACH;
  if (t0 >= n) return;
  const int tn = min(ACH, n - t0);
  __shared__ __nv_bfloat16 sc[ACH][AR];
  __shared__ __nv_bfloat16 sk[ACH][ARD];
  __shared__ float sir[ACH][AH];
  for (int i = threadIdx.x; i < tn * AR / 8; i += blockDim.x) {
    const int tt = i / (AR / 8), cc = i % (AR / 8);
    reinterpret_cast<uint4*>(sc[tt])[cc] = reinterpret_cast<const uint4*>(c + (size_t)(t0 + tt) * AR)[cc];
  }
  for (int i = threadIdx.x; i < tn * ARD / 8; i += blockDim.x) {
    const int tt = i / (ARD / 8), cc = i % (ARD / 8);
    reinterpret_cast<uint4*>(sk[tt])[cc] = reinterpret_cast<const uint4*>(kr + (size_t)(t0 + tt) * ARD)[cc];
  }
  for (int i = threadIdx.x; i < tn * AH; i += blockDim.x) {
    const int tt = i / AH, h = i % AH;
    sir[tt][h] = ir[(size_t)(t0 + tt) * G + h / (AH / G)];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float qv[AHPW][AR / 32], qk[AHPW][ARD / 32];
#pragma unroll
  for (int j = 0; j < AHPW; j++) {
    const int h = warp * AHPW + j;
#pragma unroll
    for (int e = 0; e < AR / 32; e++) qv[j][e] = ql[h * AR + lane + 32 * e];
#pragma unroll
    for (int e = 0; e < ARD / 32; e++) qk[j][e] = qr[h * ARD + lane + 32 * e];
  }
  float s[AHPW][ACH];
  float mx[AHPW];
#pragma unroll
  for (int j = 0; j < AHPW; j++) mx[j] = -INFINITY;
  for (int tt = 0; tt < tn; tt++) {
#pragma unroll
    for (int j = 0; j < AHPW; j++) {
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < AR / 32; e++) d = fmaf(qv[j][e], __bfloat162float(sc[tt][lane + 32 * e]), d);
#pragma unroll
      for (int e = 0; e < ARD / 32; e++) d = fmaf(qk[j][e], __bfloat162float(sk[tt][lane + 32 * e]), d);
      d = warp_sum(d) * scale * sir[tt][warp * AHPW + j];
      s[j][tt] = d;
      mx[j] = fmaxf(mx[j], d);
    }
  }
#pragma unroll
  for (int j = 0; j < AHPW; j++) {
    const int h = warp * AHPW + j;
    float l = 0.f, acc[AR / 32];
#pragma unroll
    for (int e = 0; e < AR / 32; e++) acc[e] = 0.f;
    for (int tt = 0; tt < tn; tt++) {
      const float p = __expf(s[j][tt] - mx[j]);
      l += p;
#pragma unroll
      for (int e = 0; e < AR / 32; e++) acc[e] = fmaf(p, __bfloat162float(sc[tt][lane + 32 * e]), acc[e]);
    }
    const size_t o = (size_t)blockIdx.x * AH + h;
    if (lane == 0) { pm[o] = mx[j]; pl[o] = l; }
#pragma unroll
    for (int e = 0; e < AR / 32; e++) pacc[o * AR + lane + 32 * e] = acc[e];
  }
}

template <int AR>
__global__ void mla_lat_merge_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                const long* __restrict__ npos, float* __restrict__ out) {
  const int h = blockIdx.x;
  const int nch = ((int)npos[0] + 1 + ACH - 1) / ACH;
  float m = -INFINITY;
  for (int k = 0; k < nch; k++) m = fmaxf(m, pm[(size_t)k * AH + h]);
  float l = 0.f;
  for (int k = 0; k < nch; k++) l += pl[(size_t)k * AH + h] * __expf(pm[(size_t)k * AH + h] - m);
  for (int e = threadIdx.x; e < AR; e += blockDim.x) {
    float a = 0.f;
    for (int k = 0; k < nch; k++) a += pacc[((size_t)k * AH + h) * AR + e] * __expf(pm[(size_t)k * AH + h] - m);
    out[h * AR + e] = a / l;
  }
}

// qr (H, 64) fp32, ql (H, 512) fp32, kr (Tmax, 64) bf16, c (Tmax, 512) bf16, ir (Tmax, G) fp32, npos (1,) long = pos
// of the newest token; scratch pm, pl (Tmax/CH, H), pacc (Tmax/CH, H, 512) fp32; out (H, 512) fp32.
template <int AR>
static void mla_lat_launch(torch::Tensor& qr, torch::Tensor& ql, torch::Tensor& kr, torch::Tensor& c, torch::Tensor& ir,
                           torch::Tensor& npos, float scale, torch::Tensor& pm, torch::Tensor& pl, torch::Tensor& pacc,
                           torch::Tensor& out) {
  const int Tmax = c.size(0), G = ir.size(1);
  const int nblk = (Tmax + ACH - 1) / ACH;
  auto st = c10::cuda::getCurrentCUDAStream();
  mla_lat_attn_k<AR><<<nblk, AWARPS * 32, 0, st>>>(qr.data_ptr<float>(), ql.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(kr.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(c.data_ptr()),
      ir.data_ptr<float>(), npos.data_ptr<long>(), G, scale, pm.data_ptr<float>(), pl.data_ptr<float>(),
      pacc.data_ptr<float>());
  mla_lat_merge_k<AR><<<AH, 128, 0, st>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(),
                                           npos.data_ptr<long>(), out.data_ptr<float>());
}

// qr (H, 64) fp32, ql (H, r) fp32, kr (Tmax, 64) bf16, c (Tmax, r) bf16, ir (Tmax, G) fp32, npos (1,) long = position of
// the newest token; scratch pm, pl (>= Tmax/CH * H), pacc (>= Tmax/CH * H * r) fp32; out (>= H * r) fp32. r in {384, 512, 640}.
void mla_lat_attn(torch::Tensor qr, torch::Tensor ql, torch::Tensor kr, torch::Tensor c, torch::Tensor ir,
                  torch::Tensor npos, double scale, torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc, torch::Tensor out) {
  TORCH_CHECK(qr.size(0) == AH && kr.size(1) == ARD, "mla_lat_attn: 16 heads, rope 64");
  const int r = c.size(1);
  TORCH_CHECK(ql.size(1) == r, "q_lat / latent rank mismatch");
  switch (r) {
    case 384: mla_lat_launch<384>(qr, ql, kr, c, ir, npos, (float)scale, pm, pl, pacc, out); break;
    case 512: mla_lat_launch<512>(qr, ql, kr, c, ir, npos, (float)scale, pm, pl, pacc, out); break;
    case 640: mla_lat_launch<640>(qr, ql, kr, c, ir, npos, (float)scale, pm, pl, pacc, out); break;
    default: TORCH_CHECK(false, "mla_lat_attn: unsupported latent rank ", r);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packed_gemv", &packed_gemv);
  m.def("emb_q4", &emb_q4);
  m.def("mla_lat_attn", &mla_lat_attn);
}
