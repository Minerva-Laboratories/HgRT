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

// the full-warp sums of 16 values per lane at once: lane l returns the sum of v[l >> 1] over the warp. The same pairs
// are added in the same order as warp_sum (xor 16, 8, 4, 2, 1; own + partner), so each sum is bit-identical to it.
__device__ __forceinline__ float warp_sum16_t(const float (&v)[16]) {
  const int lane = threadIdx.x & 31;
  float a8[8], a4[4], a2[2];
  const bool u16 = lane & 16, u8 = lane & 8, u4 = lane & 4, u2 = lane & 2;
#pragma unroll
  for (int i = 0; i < 8; i++) a8[i] = (u16 ? v[i + 8] : v[i]) + __shfl_xor_sync(0xffffffff, u16 ? v[i] : v[i + 8], 16);
#pragma unroll
  for (int i = 0; i < 4; i++) a4[i] = (u8 ? a8[i + 4] : a8[i]) + __shfl_xor_sync(0xffffffff, u8 ? a8[i] : a8[i + 4], 8);
#pragma unroll
  for (int i = 0; i < 2; i++) a2[i] = (u4 ? a4[i + 2] : a4[i]) + __shfl_xor_sync(0xffffffff, u4 ? a4[i] : a4[i + 2], 4);
  float a1 = (u2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffff, u2 ? a2[0] : a2[1], 2);
  return a1 + __shfl_xor_sync(0xffffffff, a1, 1);
}

// bf16 hi / lo split of two floats (hi + lo carries ~16 mantissa bits), packed as bf16x2 (low half = first value)
__device__ __forceinline__ void split_bf2(float a, float b, uint32_t& hi, uint32_t& lo) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  const float2 hf = __bfloat1622float2(h);
  __nv_bfloat162 l = __floats2bfloat162_rn(a - hf.x, b - hf.y);
  hi = *reinterpret_cast<uint32_t*>(&h); lo = *reinterpret_cast<uint32_t*>(&l);
}
__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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
    int ns, float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
  const int m = blockIdx.y, n = (int)npos[m] + 1;          // tokens 0 .. pos
  if ((int)blockIdx.x * ACH >= n) return;
  qr += (size_t)m * AH * ARD; ql += (size_t)m * AH * AR;
  __shared__ __nv_bfloat16 sc[ACH][AR];
  __shared__ __nv_bfloat16 sk[ACH][ARD];
  __shared__ float sir[ACH][AH];
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
  float rm[AHPW], rl[AHPW], acc[AHPW][AR / 32];
#pragma unroll
  for (int j = 0; j < AHPW; j++) {
    rm[j] = -INFINITY; rl[j] = 0.f;
#pragma unroll
    for (int e = 0; e < AR / 32; e++) acc[j][e] = 0.f;
  }
  for (int t0 = blockIdx.x * ACH; t0 < n; t0 += ns * ACH) {
    const int tn = min(ACH, n - t0);
    __syncthreads();
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
#pragma unroll
    for (int j = 0; j < AHPW; j++) {
      float s[ACH], mx = -INFINITY;
      for (int tt = 0; tt < tn; tt++) {
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < AR / 32; e++) d = fmaf(qv[j][e], __bfloat162float(sc[tt][lane + 32 * e]), d);
#pragma unroll
        for (int e = 0; e < ARD / 32; e++) d = fmaf(qk[j][e], __bfloat162float(sk[tt][lane + 32 * e]), d);
        d = warp_sum(d) * scale * sir[tt][warp * AHPW + j];
        s[tt] = d;
        mx = fmaxf(mx, d);
      }
      const float nm = fmaxf(rm[j], mx), al = __expf(rm[j] - nm);
      rl[j] *= al;
#pragma unroll
      for (int e = 0; e < AR / 32; e++) acc[j][e] *= al;
      for (int tt = 0; tt < tn; tt++) {
        const float p = __expf(s[tt] - nm);
        rl[j] += p;
#pragma unroll
        for (int e = 0; e < AR / 32; e++) acc[j][e] = fmaf(p, __bfloat162float(sc[tt][lane + 32 * e]), acc[j][e]);
      }
      rm[j] = nm;
    }
  }
#pragma unroll
  for (int j = 0; j < AHPW; j++) {
    const int h = warp * AHPW + j;
    const size_t o = ((size_t)m * ns + blockIdx.x) * AH + h;
    if (lane == 0) { pm[o] = rm[j]; pl[o] = rl[j]; }
#pragma unroll
    for (int e = 0; e < AR / 32; e++) pacc[o * AR + lane + 32 * e] = acc[j][e];
  }
}

// merge the split partials of row m (grid (H, M)); nb = the split blocks that saw tokens
template <int AR>
__global__ void mla_lat_merge_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                const long* __restrict__ npos, int ch, int ns, float* __restrict__ out) {
  const int h = blockIdx.x, m = blockIdx.y;
  const int nb = min(ns, ((int)npos[m] + 1 + ch - 1) / ch);
  pm += (size_t)m * ns * AH; pl += (size_t)m * ns * AH; pacc += (size_t)m * ns * AH * AR; out += (size_t)m * AH * AR;
  float mx = -INFINITY;
  for (int k = 0; k < nb; k++) mx = fmaxf(mx, pm[(size_t)k * AH + h]);
  float l = 0.f;
  for (int k = 0; k < nb; k++) l += pl[(size_t)k * AH + h] * __expf(pm[(size_t)k * AH + h] - mx);
  for (int e = threadIdx.x; e < AR; e += blockDim.x) {
    float a = 0.f;
    for (int k = 0; k < nb; k++) a += pacc[((size_t)k * AH + h) * AR + e] * __expf(pm[(size_t)k * AH + h] - mx);
    out[h * AR + e] = a / l;
  }
}

// qr (M, H, 64) fp32, ql (M, H, r) fp32, kr (Tmax, 64) bf16, c (Tmax, r) bf16, ir (Tmax, G) fp32, npos (>= M,) long = the
// rows' positions; scratch pm, pl (>= M ns H), pacc (>= M ns H r) fp32; out (M, H, r) fp32. r in {384, 512, 640}.
template <int AR>
static void mla_lat_launch(torch::Tensor& qr, torch::Tensor& ql, torch::Tensor& kr, torch::Tensor& c, torch::Tensor& ir,
                           torch::Tensor& npos, int M, int ns, float scale, torch::Tensor& pm, torch::Tensor& pl,
                           torch::Tensor& pacc, torch::Tensor& out) {
  const int G = ir.size(1);
  auto st = c10::cuda::getCurrentCUDAStream();
  mla_lat_attn_k<AR><<<dim3(ns, M), AWARPS * 32, 0, st>>>(qr.data_ptr<float>(), ql.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(kr.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(c.data_ptr()),
      ir.data_ptr<float>(), npos.data_ptr<long>(), G, scale, ns, pm.data_ptr<float>(), pl.data_ptr<float>(),
      pacc.data_ptr<float>());
  mla_lat_merge_k<AR><<<dim3(AH, M), 128, 0, st>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(),
                                                   npos.data_ptr<long>(), ACH, ns, out.data_ptr<float>());
}

void mla_lat_attn(torch::Tensor qr, torch::Tensor ql, torch::Tensor kr, torch::Tensor c, torch::Tensor ir,
                  torch::Tensor npos, int64_t M, int64_t ns, double scale, torch::Tensor pm, torch::Tensor pl,
                  torch::Tensor pacc, torch::Tensor out) {
  TORCH_CHECK(qr.size(-2) == AH && qr.size(-1) == ARD && kr.size(1) == ARD, "mla_lat_attn: 16 heads, rope 64");
  const int r = c.size(1);
  TORCH_CHECK(ql.size(-1) == r, "q_lat / latent rank mismatch");
  TORCH_CHECK(pm.numel() >= M * ns * AH && pacc.numel() >= M * ns * AH * r, "mla_lat_attn: scratch");
  switch (r) {
    case 384: mla_lat_launch<384>(qr, ql, kr, c, ir, npos, M, ns, (float)scale, pm, pl, pacc, out); break;
    case 512: mla_lat_launch<512>(qr, ql, kr, c, ir, npos, M, ns, (float)scale, pm, pl, pacc, out); break;
    case 640: mla_lat_launch<640>(qr, ql, kr, c, ir, npos, M, ns, (float)scale, pm, pl, pacc, out); break;
    default: TORCH_CHECK(false, "mla_lat_attn: unsupported latent rank ", r);
  }
}

// ---- GDN-2 decode step, one token, fused (one block per value head, thread v owns column v of the head's state).
// Same arithmetic as the pipeline's single-step path: conv over the rolling raw-input state (bf16 products, fp32 sum,
// bf16 result, SiLU), g = -exp(A_log) softplus(a + dt_bias) (fp32), b = sigmoid(be), w = sigmoid(bw) (bf16, as the
// module rounds them), l2-normed q / k (eps 1e-6), S <- S diag(exp g); S <- S + k (w v - (b k)^T S); o = (q/sqrt(Dk)) S;
// then the gated RMSNorm: bf16(o) normalized in fp32 -> bf16 -> * gain (fp32) * silu(z) -> bf16.
// Layout: qkv (C) = [q (Hq Dk) | k (Hq Dk) | v (H Dv)], conv state (C, KC) bf16 (oldest first), conv_w (C, KC) bf16,
// gates (3 H Dk) = [a | be | bw] (Dk == Dv), S (H, Dk, Dv) fp32, z (H Dv), out (H Dv) bf16. Dk = Dv = 128, KC = 4.
constexpr int GD = 128, GKC = 4;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float block_sum128(float v, float* red) {   // 128 threads
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  const float t = red[0] + red[1] + red[2] + red[3];
  __syncthreads();
  return t;
}

__device__ __forceinline__ float conv_ch(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* sti,
                                         __nv_bfloat16* sto, const __nv_bfloat16* __restrict__ w, int ch) {
  float full[GKC];
#pragma unroll
  for (int k = 0; k < GKC - 1; k++) full[k] = __bfloat162float(sti[ch * GKC + k + 1]);
  full[GKC - 1] = __bfloat162float(x[ch]);
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < GKC; k++) {
    sto[ch * GKC + k] = __float2bfloat16(full[k]);
    acc += bf(full[k] * __bfloat162float(w[ch * GKC + k]));
  }
  const float y = bf(acc);
  return bf(y / (1.f + __expf(-y)));                       // silu, bf16 out
}

// M rows in one launch, in order: the head's recurrent state stays in registers between rows (fp32, so bit-identical to
// running the rows one by one). Row m reads conv / state from row m - 1 (row 0: csi / Si) and writes them to
// cso + m cs_stride / So + m S_stride (plain decoding: M = 1, in place).
__global__ void __launch_bounds__(GD) gdn2_step_k(const __nv_bfloat16* __restrict__ qkv, int ldq,
    const __nv_bfloat16* csi, __nv_bfloat16* cso, long cs_stride, const __nv_bfloat16* __restrict__ cw,
    const __nv_bfloat16* __restrict__ gates, int ldg, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const float* Si, float* So, long S_stride, const __nv_bfloat16* __restrict__ z, int ldz,
    const float* __restrict__ gn, float gn_eps, __nv_bfloat16* __restrict__ out, int ldo, int H, int Hq, int M) {
  const int h = blockIdx.x, v = threadIdx.x;              // q / k heads expanded to the value heads (Hq == H)
  const int KD = Hq * GD;
  __shared__ float sq[GD], sk[GD], sg[GD], sb[GD], red[4];
  float col[GD];
  {
    const float* Sih = Si + (size_t)h * GD * GD;
#pragma unroll
    for (int k = 0; k < GD; k++) col[k] = Sih[k * GD + v];
  }
  const float eA = __expf(A_log[h * GD + v]), dtb = dt_bias[h * GD + v];
  for (int m = 0; m < M; m++) {
    const __nv_bfloat16* x = qkv + (size_t)m * ldq;
    const __nv_bfloat16* g = gates + (size_t)m * ldg;
    const __nv_bfloat16* ci = m == 0 ? csi : cso + (size_t)(m - 1) * cs_stride;
    __nv_bfloat16* co = cso + (size_t)m * cs_stride;
    const float qv = conv_ch(x, ci, co, cw, h * GD + v);
    const float kv = conv_ch(x, ci, co, cw, KD + h * GD + v);
    const float vv = conv_ch(x, ci, co, cw, 2 * KD + h * GD + v);
    const float q2 = block_sum128(qv * qv, red), k2 = block_sum128(kv * kv, red);
    sq[v] = qv * rsqrtf(q2 + 1e-6f) * rsqrtf((float)GD);
    sk[v] = kv * rsqrtf(k2 + 1e-6f);
    const float a = __bfloat162float(g[h * GD + v]);
    const float sp = a + dtb;
    const float softplus = sp > 20.f ? sp : log1pf(__expf(sp));
    sg[v] = __expf(-eA * softplus);
    sb[v] = bf(1.f / (1.f + __expf(-__bfloat162float(g[H * GD + h * GD + v])))) * sk[v];
    const float wv = bf(1.f / (1.f + __expf(-__bfloat162float(g[2 * H * GD + h * GD + v])))) * __bfloat162float(__float2bfloat16(vv));
    __syncthreads();
    float* Soh = So + (size_t)m * S_stride + (size_t)h * GD * GD;
    float erase = 0.f;
#pragma unroll
    for (int k = 0; k < GD; k++) {
      col[k] = col[k] * sg[k];
      erase = fmaf(sb[k], col[k], erase);
    }
    const float d = wv - erase;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < GD; k++) {
      col[k] = fmaf(sk[k], d, col[k]);
      Soh[k * GD + v] = col[k];
      o = fmaf(sq[k], col[k], o);
    }
    // gated RMSNorm over the head's Dv outputs
    const float ob = bf(o);
    const float ms = block_sum128(ob * ob, red) / GD;
    const float hn = bf(ob * rsqrtf(ms + gn_eps)) * gn[v];
    const float zz = __bfloat162float(z[(size_t)m * ldz + h * GD + v]);
    out[(size_t)m * ldo + h * GD + v] = __float2bfloat16(hn * (zz / (1.f + __expf(-zz))));
    __syncthreads();
  }
}

// rows qkv / gates / z / out (M, ld*); conv / recurrent state read from (csi, Si) and written to (cso, So) + m stride: the
// same tensors (stride 0) for plain decoding, per-row slots for speculative verification. Each block owns one head.
void gdn2_step(torch::Tensor qkv, torch::Tensor csi, torch::Tensor cso, int64_t cs_stride, torch::Tensor cw, torch::Tensor gates,
               torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor Si, torch::Tensor So, int64_t S_stride, torch::Tensor z,
               torch::Tensor gn, double gn_eps, torch::Tensor out, int64_t H, int64_t Hq, int64_t M) {
  TORCH_CHECK(Si.size(-1) == GD && csi.size(-1) == GKC, "gdn2_step: Dk = Dv = 128, conv 4");
  TORCH_CHECK(H == Hq, "gdn2_step: q/k heads must be expanded to the value heads (num_qk_heads == num_heads)");
  gdn2_step_k<<<H, GD, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(csi.data_ptr()), reinterpret_cast<__nv_bfloat16*>(cso.data_ptr()), cs_stride,
      reinterpret_cast<const __nv_bfloat16*>(cw.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gates.data_ptr()),
      gates.stride(0), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), Si.data_ptr<float>(), So.data_ptr<float>(), S_stride,
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()), z.stride(0), gn.data_ptr<float>(), (float)gn_eps,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), H, Hq, M);
}

// ---- GQA decode attention for the MTP layer: q (H, 256) fp32; K, V (Tmax, G, 256) bf16; out (H, 256) fp32.
// Split over T (chunks of 32 tokens) and KV groups; one warp per head of the group; device-side length (graph-safe).
constexpr int QD = 256, QCH = 32;

__global__ void __launch_bounds__(4 * 32) gqa_attn_k(const float* __restrict__ q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ Vc, const long* __restrict__ npos, int H, int G, float scale, int ns,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
  const int m = blockIdx.z, n = (int)npos[m] + 1, g = blockIdx.y;
  if ((int)blockIdx.x * QCH >= n) return;
  const int hpg = H / G;
  q += (size_t)m * H * QD;
  __shared__ __nv_bfloat16 sk[QCH][QD], sv[QCH][QD];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const bool act = warp < hpg;
  const int h = g * hpg + (act ? warp : 0);
  float qv[QD / 32];
#pragma unroll
  for (int e = 0; e < QD / 32; e++) qv[e] = q[h * QD + lane + 32 * e];
  float rm = -INFINITY, rl = 0.f, acc[QD / 32];
#pragma unroll
  for (int e = 0; e < QD / 32; e++) acc[e] = 0.f;
  for (int t0 = blockIdx.x * QCH; t0 < n; t0 += ns * QCH) {
    const int tn = min(QCH, n - t0);
    __syncthreads();
    for (int i = threadIdx.x; i < tn * QD / 8; i += blockDim.x) {
      const int tt = i / (QD / 8), cc = i % (QD / 8);
      reinterpret_cast<uint4*>(sk[tt])[cc] = reinterpret_cast<const uint4*>(K + ((size_t)(t0 + tt) * G + g) * QD)[cc];
      reinterpret_cast<uint4*>(sv[tt])[cc] = reinterpret_cast<const uint4*>(Vc + ((size_t)(t0 + tt) * G + g) * QD)[cc];
    }
    __syncthreads();
    if (!act) continue;
    float s[QCH], mx = -INFINITY;
    for (int tt = 0; tt < tn; tt++) {
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < QD / 32; e++) d = fmaf(qv[e], __bfloat162float(sk[tt][lane + 32 * e]), d);
      s[tt] = warp_sum(d) * scale;
      mx = fmaxf(mx, s[tt]);
    }
    const float nm = fmaxf(rm, mx), al = __expf(rm - nm);
    rl *= al;
#pragma unroll
    for (int e = 0; e < QD / 32; e++) acc[e] *= al;
    for (int tt = 0; tt < tn; tt++) {
      const float p = __expf(s[tt] - nm);
      rl += p;
#pragma unroll
      for (int e = 0; e < QD / 32; e++) acc[e] = fmaf(p, __bfloat162float(sv[tt][lane + 32 * e]), acc[e]);
    }
    rm = nm;
  }
  if (!act) return;
  const size_t o = ((size_t)m * ns + blockIdx.x) * H + h;
  if (lane == 0) { pm[o] = rm; pl[o] = rl; }
#pragma unroll
  for (int e = 0; e < QD / 32; e++) pacc[o * QD + lane + 32 * e] = acc[e];
}

__global__ void gqa_merge_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                            const long* __restrict__ npos, int H, int ns, float* __restrict__ out) {
  const int h = blockIdx.x, m = blockIdx.y, nb = min(ns, ((int)npos[m] + 1 + QCH - 1) / QCH);
  pm += (size_t)m * ns * H; pl += (size_t)m * ns * H; pacc += (size_t)m * ns * H * QD; out += (size_t)m * H * QD;
  float mx = -INFINITY, l = 0.f;
  for (int k = 0; k < nb; k++) mx = fmaxf(mx, pm[(size_t)k * H + h]);
  for (int k = 0; k < nb; k++) l += pl[(size_t)k * H + h] * __expf(pm[(size_t)k * H + h] - mx);
  for (int e = threadIdx.x; e < QD; e += blockDim.x) {
    float a = 0.f;
    for (int k = 0; k < nb; k++) a += pacc[((size_t)k * H + h) * QD + e] * __expf(pm[(size_t)k * H + h] - mx);
    out[h * QD + e] = a / l;
  }
}

// q (M, H, 256) fp32; K, V (Tmax, G, 256) bf16; npos (>= M,) positions; out (M, H, 256) fp32
void gqa_attn(torch::Tensor q, torch::Tensor K, torch::Tensor Vc, torch::Tensor npos, int64_t M, int64_t ns, double scale,
              torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc, torch::Tensor out) {
  const int H = q.size(-2), G = K.size(1);
  TORCH_CHECK(q.size(-1) == QD && K.size(2) == QD && H % G == 0 && H / G <= 4, "gqa_attn: D 256, <= 4 heads per group");
  TORCH_CHECK(pm.numel() >= M * ns * H && pacc.numel() >= M * ns * H * QD, "gqa_attn: scratch");
  auto st = c10::cuda::getCurrentCUDAStream();
  gqa_attn_k<<<dim3(ns, G, M), 4 * 32, 0, st>>>(q.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(Vc.data_ptr()), npos.data_ptr<long>(), H, G, (float)scale, ns,
      pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>());
  gqa_merge_k<<<dim3(H, M), 128, 0, st>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(),
                                          npos.data_ptr<long>(), H, ns, out.data_ptr<float>());
}

__device__ __forceinline__ float warp_max_f(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

// per head: logsumexp over s <= t of sc[h, s]; then lp[s] = max_h (sc[h, s] - lse_h) for s in comp, else -inf.
// Row m = blockIdx.y: scores sc (M, H, Tmax), lse (M, H), lp (M, Tmax).
__global__ void fovea_lse_k(const float* __restrict__ sc, const long* __restrict__ npos, int Tmax, float* __restrict__ lse) {
  const int h = blockIdx.x, m = blockIdx.y, n = (int)npos[m] + 1;
  sc += ((size_t)m * AH + h) * Tmax;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int s = threadIdx.x; s < n; s += blockDim.x) mx = fmaxf(mx, sc[s]);
  mx = warp_max_f(mx);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY; v = warp_max_f(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  const float M = red[0];
  __syncthreads();
  float acc = 0.f;
  for (int s = threadIdx.x; s < n; s += blockDim.x) acc += expf(sc[s] - M);
  acc = warp_sum(acc);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { float tot = 0.f; for (int w = 0; w < (int)blockDim.x / 32; w++) tot += red[w]; lse[m * AH + h] = M + logf(tot); }
}

__global__ void fovea_lp_k(const float* __restrict__ sc, const float* __restrict__ lse, const long* __restrict__ npos,
                           int Tmax, int H, int sinks, int window, float* __restrict__ lp) {
  const int m = blockIdx.y, t = (int)npos[m];
  sc += (size_t)m * H * Tmax; lse += m * H; lp += (size_t)m * Tmax;
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s <= t; s += gridDim.x * blockDim.x) {
    float v = -INFINITY;
    if (s >= sinks && t - s >= window)
      for (int h = 0; h < H; h++) v = fmaxf(v, sc[(size_t)h * Tmax + s] - lse[h]);
    lp[s] = v;
  }
}

// threshold = the k-th largest lp over comp (k = min(ceil(hot_frac (t + 1)), |comp|)); +inf if k = 0 (no hot tokens).
// Exact: radix select over the order-preserving uint32 key of the float, 8 bits per pass from the top (4 passes, one
// block; integer histogram counts, so the result is the exact k-th largest key whatever the atomic order).
__device__ __forceinline__ unsigned fkey(float f) { unsigned u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
__device__ __forceinline__ float kfloat(unsigned k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k); }

__global__ void fovea_select_k(const float* __restrict__ lp, const long* __restrict__ npos, int Tmax, int sinks, int window,
                               double hot_frac, float* __restrict__ thr) {
  const int m = blockIdx.x, t = (int)npos[m];
  lp += (size_t)m * Tmax;
  const int lo_s = sinks, hi_s = t - window;               // comp = [sinks, t - window]
  const int ncomp = max(0, hi_s - lo_s + 1);
  const long kk = min((long)ceil(hot_frac * ((double)t + 1.0)), (long)ncomp);
  __shared__ unsigned hist[256];
  __shared__ unsigned sb, sk;
  if (kk <= 0) { if (threadIdx.x == 0) thr[m] = INFINITY; return; }
  unsigned prefix = 0u, mask = 0u, k = (unsigned)kk;       // the k-th largest key among those matching prefix on mask
  for (int sh = 24; sh >= 0; sh -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
    __syncthreads();
    for (int s = lo_s + threadIdx.x; s <= hi_s; s += blockDim.x) {
      const unsigned key = fkey(lp[s]);
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> sh) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 32) {                                 // lane l: bins 255 - 8 l .. 248 - 8 l (descending)
      const int l = threadIdx.x;
      unsigned c[8], sum = 0u;
#pragma unroll
      for (int j = 0; j < 8; j++) { c[j] = hist[255 - 8 * l - j]; sum += c[j]; }
      unsigned inc = sum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const unsigned v = __shfl_up_sync(0xffffffff, inc, o); if (l >= o) inc += v; }
      const unsigned hit = __ballot_sync(0xffffffff, inc >= k);
      if (l == __ffs(hit) - 1) {
        unsigned above = inc - sum;
        int j = 0;
        while (above + c[j] < k) { above += c[j]; j++; }
        sb = 255u - 8u * l - j; sk = k - above;
      }
    }
    __syncthreads();
    prefix |= sb << sh; mask |= 255u << sh; k = sk;
    __syncthreads();
  }
  if (threadIdx.x == 0) thr[m] = kfloat(prefix);
}

// ---- unrounded Q4_0 / Q8_0 GEMV (llama.cpp / GGUF numerics: w = (q - 8) d exactly, d fp16): the scale factors out of
// each group, y += d (sum q x - 8 sum x), so a weight costs one FMA per row. Same structure and bit identity across M.
__device__ __forceinline__ float nibf(uint32_t w, int i) {          // nibble -> float, exact, no I2F
  return __int_as_float(((w >> (4 * i)) & 15u) | 0x4B000000u) - 8388608.f;
}

template <int FMT, int R, int M, typename OUT>
__global__ void __launch_bounds__(WARPS * 32) gemvu_k(const uint4* __restrict__ W, const half* __restrict__ S,
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
    uint4 pa[R], pb[R];
    float sc[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int row = min(row0 + r, N - 1);
      sc[r] = __half2float(__ldg(S + (size_t)row * G + g));
      if (FMT == 0) pa[r] = __ldg(W + (size_t)row * G + g);
      else { pa[r] = __ldg(W + ((size_t)row * G + g) * 2); pb[r] = __ldg(W + ((size_t)row * G + g) * 2 + 1); }
    }
    // every row's activations for this group; each weight is then unpacked once and used by all M rows (the same per-row
    // FMA order as for M = 1)
    float xf[M][32], xs[M];
#pragma unroll
    for (int m = 0; m < M; m++) {
      load_x32(X + (size_t)m * ldx + g * 32, xf[m]);
      xs[m] = 0.f;
      if (FMT == 0) {
#pragma unroll
        for (int e = 0; e < 32; e++) xs[m] += xf[m][e];
      }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
      float part[M];
#pragma unroll
      for (int m = 0; m < M; m++) part[m] = 0.f;
      if (FMT == 0) {
        const uint32_t wd[4] = {pa[r].x, pa[r].y, pa[r].z, pa[r].w};
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const float w = nibf(wd[j], i);
#pragma unroll
            for (int m = 0; m < M; m++) part[m] = fmaf(w, xf[m][j * 8 + i], part[m]);
          }
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = fmaf(sc[r], part[m] - 8.f * xs[m], acc[r][m]);
      } else {
        const int8_t* ca = reinterpret_cast<const int8_t*>(&pa[r]);
        const int8_t* cb = reinterpret_cast<const int8_t*>(&pb[r]);
#pragma unroll
        for (int e = 0; e < 16; e++) {
          const float wa = (float)ca[e], wb = (float)cb[e];
#pragma unroll
          for (int m = 0; m < M; m++) { part[m] = fmaf(wa, xf[m][e], part[m]); part[m] = fmaf(wb, xf[m][16 + e], part[m]); }
        }
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = fmaf(sc[r], part[m], acc[r][m]);
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
static void launchu(torch::Tensor& codes, torch::Tensor& scales, torch::Tensor& x, int ldx, const __nv_bfloat16* res,
                    int ldr, torch::Tensor& y, int ldy, int N, int K, int M) {
  dim3 grid((N + WARPS * R - 1) / (WARPS * R));
  auto st = c10::cuda::getCurrentCUDAStream();
  auto W = reinterpret_cast<const uint4*>(codes.data_ptr());
  auto S = reinterpret_cast<const half*>(scales.data_ptr());
  auto X = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto Y = reinterpret_cast<OUT*>(y.data_ptr());
  switch (M) {
    case 1: gemvu_k<FMT, R, 1, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
    case 2: gemvu_k<FMT, R, 2, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
    case 3: gemvu_k<FMT, R, 3, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
    default: gemvu_k<FMT, R, 4, OUT><<<grid, WARPS * 32, 0, st>>>(W, S, X, ldx, res, ldr, Y, ldy, N, K); break;
  }
}

void packed_gemv_u(torch::Tensor codes, torch::Tensor scales, int64_t fmt, torch::Tensor x, int64_t ldx,
                   c10::optional<torch::Tensor> res, int64_t ldr, torch::Tensor y, int64_t ldy, int64_t M, int64_t R) {
  const int N = scales.size(0), K = scales.size(1) * 32;
  TORCH_CHECK(M >= 1 && M <= 4, "M must be 1..4");
  const __nv_bfloat16* r = res ? reinterpret_cast<const __nv_bfloat16*>(res->data_ptr()) : nullptr;
  const bool f32 = y.scalar_type() == at::kFloat;
#define GOU(F, RR) do { if (f32) launchu<F, RR, float>(codes, scales, x, ldx, r, ldr, y, ldy, N, K, M); \
                        else launchu<F, RR, __nv_bfloat16>(codes, scales, x, ldx, r, ldr, y, ldy, N, K, M); } while (0)
  if (fmt == 0) { if (R == 1) GOU(0, 1); else if (R == 2) GOU(0, 2); else GOU(0, 4); }
  else { if (R == 1) GOU(1, 1); else if (R == 2) GOU(1, 2); else GOU(1, 4); }
#undef GOU
}

// ---- RMSNorm over M rows in one launch (one block per row, fixed per-row summation order -> identical rows for any M):
// y = bf16( x / sqrt(mean(x^2) + eps) * g ),  g = w (scalar: ScaleNorm, w[0]) or 1 + w[i] (zero-centred RMSNorm).
__global__ void rms_rows_k(const __nv_bfloat16* __restrict__ x, int ldx, const float* __restrict__ w, int scalar, float eps,
                           __nv_bfloat16* __restrict__ y, int ldy, int H) {
  const int m = blockIdx.x;
  __shared__ float red[32];
  float acc = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) { const float v = __bfloat162float(x[(size_t)m * ldx + i]); acc = fmaf(v, v, acc); }
  acc = warp_sum(acc);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0.f; for (int k = 0; k < (int)blockDim.x / 32; k++) t += red[k]; red[0] = rsqrtf(t / H + eps); }
  __syncthreads();
  const float r = red[0];
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const float g = scalar ? w[0] : 1.f + w[i];
    y[(size_t)m * ldy + i] = __float2bfloat16(__bfloat162float(x[(size_t)m * ldx + i]) * r * g);
  }
}

void rms_rows(torch::Tensor x, int64_t ldx, torch::Tensor w, bool scalar, double eps, torch::Tensor y, int64_t ldy,
              int64_t M, int64_t H) {
  rms_rows_k<<<M, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), ldx,
      w.data_ptr<float>(), scalar ? 1 : 0, (float)eps, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), ldy, H);
}

// ---- FoveaKV attention reading the CODES (the memory-saving cache). Per token s of an MLA layer the cache keeps:
//   codes  4-bit fine index per channel of the rotated latent (r/2 bytes) and of the rotated k_rope (32 bytes), nibble-packed,
//          channel 2i in the low nibble; the coarse plane is the index's top 2 bits (FoveaKV's nested codebook)
//   gain   8-bit log-gain code per 128-channel head of the latent and one for k_rope
//   ir     1 / rms per KV group (fp16)        block means M_b (fp16) per 128-token block, latent and k_rope
//   exact  latent / k_rope of the sinks and of a ring of the last RW tokens (bf16): the always-exact set A
// x_hat = M_b + (g lev s) @ R, so for a query q: q . x_hat = q . M_b + g s sum_d (R q)_d lev_d  (rotated query, read codes);
// values accumulate as sum p M_b (per block) + (sum p g s lev) @ R (rotated back once per head at the merge).
// CH-token chunks (CH divides the 128-token block: one block mean per chunk), one warp per head (16 warps); grid (ns, M):
// rows batched, a fixed number of split blocks per row looping over chunks with an online softmax (scratch O(ns)).
struct FoveaCodes {
  const uint8_t* cc; const uint8_t* gc; const uint8_t* cr; const uint8_t* gr;   // codes / gains: latent, k_rope
  const half* Mc; const half* Mr;                                                // block means
  const float* loc; const float* spc; const float* lor; const float* spr;        // gain calibration (per head)
  const __nv_bfloat16* sink_c; const __nv_bfloat16* sink_r;                      // exact sinks (sinks, r) / (sinks, 64)
  const __nv_bfloat16* ring_c; const __nv_bfloat16* ring_r;                      // exact ring (RW, r) / (RW, 64)
  const half* ir; int RW, sinks, window, G, block, nbits;
  float sfk, sfv, sck, scv, srf, src_;                                           // shrinkage: fine / coarse, key / value, rope
};
__constant__ float c_lut16[16];
__constant__ float c_lut4[4];
__constant__ float c_bnd16[15];

__device__ __forceinline__ float fv_gain(int code, float lo, float span, int nbits, float hd) {
  return expf(lo + ((float)code + 0.5f) / (float)(1 << nbits) * span) / sqrtf(hd);
}

template <int AR, int PASS, int CH>
__global__ void __launch_bounds__(16 * 32) fovea_codes_k(const float* __restrict__ qr, const float* __restrict__ ql,
    const float* __restrict__ qrr, const float* __restrict__ qlr, const long* __restrict__ npos, float scale, FoveaCodes F,
    const float* __restrict__ lp, const float* __restrict__ thr, float* __restrict__ sc_out, int Tmax, int ns,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pax, float* __restrict__ par) {
  constexpr int NH = AR / 128;
  const int m = blockIdx.y, t = (int)npos[m], n = t + 1;
  if ((int)blockIdx.x * CH >= n) return;
  qr += (size_t)m * AH * ARD; qrr += (size_t)m * AH * ARD; ql += (size_t)m * AH * AR; qlr += (size_t)m * AH * AR;
  // per chunk, decoded once for all 16 heads: exact rows (tier A), or g * level of the coded rows (shrinkage per tier
  // is applied per token to the head's partial sums). The next chunk's raw data (codes as 32-bit words of 8 nibbles,
  // gains, 1 / rms, ranking keys, block means) is loaded into registers while the current chunk is computed.
  constexpr int PT_ = 16 * 32, WPT = AR / 8, NWC = (CH * WPT + PT_ - 1) / PT_;
  __shared__ float sv[CH][AR + 8], svr[CH][ARD + 8];   // +8: conflict-free mma B-fragment loads
  // scores of coded chunks on tensor cores: 16 heads x 16 tokens x (AR + 64), K split 8 ways x 2 token tiles over the
  // 16 warps (each chunk element read once, not once per head); fp32 operands as bf16 hi + lo (3 mma: lo*hi, hi*lo,
  // hi*hi); the 8 K partials summed in a fixed order from dynamic shared memory (deterministic, no atomics)
  constexpr int KL = AR / 16, KS = KL + ARD / 16, NSTEP = (KS + 7) / 8, RHS = 24;
  extern __shared__ float red[];                 // [8][16 heads][RHS]
  __shared__ float sgc[CH][NH], sgr[CH];
  __shared__ float sir[CH][AH];
  __shared__ int tier[CH];
  __shared__ float slut[20];                     // 16 + 4 levels
  __shared__ __half2 smc[AR / 2], smr[ARD / 2];  // the chunk's block means
  if (threadIdx.x < 20) slut[threadIdx.x] = threadIdx.x < 16 ? c_lut16[threadIdx.x] : c_lut4[threadIdx.x - 16];
  const int tid = threadIdx.x, lane = tid & 31, h = tid >> 5;
  float qv[AR / 32], qk[2];
#pragma unroll
  for (int e = 0; e < AR / 32; e++) qv[e] = ql[h * AR + lane + 32 * e];
#pragma unroll
  for (int e = 0; e < 2; e++) qk[e] = qr[h * ARD + lane + 32 * e];
  const int fg = lane >> 2, f4 = lane & 3, ks = h >> 1, ntl = h & 1;   // mma fragment coordinates, K slice, token tile
  uint32_t qah[NSTEP][4], qal[NSTEP][4];        // A fragments (heads x K) of the rotated query for this warp's K slice
#pragma unroll
  for (int i = 0; i < NSTEP; i++) {
    const int st = ks + 8 * i;
    if (st < KS) {
      const float* q0 = st < KL ? qlr + fg * AR + 16 * st : qrr + fg * ARD + 16 * (st - KL);
      const int hs = st < KL ? 8 * AR : 8 * ARD;
      split_bf2(q0[2 * f4], q0[2 * f4 + 1], qah[i][0], qal[i][0]);
      split_bf2(q0[hs + 2 * f4], q0[hs + 2 * f4 + 1], qah[i][1], qal[i][1]);
      split_bf2(q0[2 * f4 + 8], q0[2 * f4 + 9], qah[i][2], qal[i][2]);
      split_bf2(q0[hs + 2 * f4 + 8], q0[hs + 2 * f4 + 9], qah[i][3], qal[i][3]);
    }
  }
  const float th = PASS == 2 ? thr[m] : 0.f;
  constexpr int NT = AR / 128;                   // value mma: 8-channel tiles per warp (warp h: channels h * AR / 16 ..)
  float rm = -INFINITY, rl = 0.f, ax[AR / 32], arm[NT][4];   // arm: rotated coded-value accumulator, mma C layout
  float* sp = red + 8 * 16 * RHS;                 // [16 heads][RHS] shrunk token weights of the chunk
  float* sal = sp + 16 * RHS;                     // [16] per-head rescale of the running max
#pragma unroll
  for (int j = 0; j < NT; j++) { arm[j][0] = 0.f; arm[j][1] = 0.f; arm[j][2] = 0.f; arm[j][3] = 0.f; }
#pragma unroll
  for (int e = 0; e < AR / 32; e++) ax[e] = 0.f;
  // raw registers of one chunk
  uint32_t cw[NWC], rw = 0, mcw = 0, mrw = 0;
  int gcb = 0, grb = 0;
  float irv = 0.f, lpv = 0.f;
  const uint32_t* CC = reinterpret_cast<const uint32_t*>(F.cc);
  const uint32_t* CR = reinterpret_cast<const uint32_t*>(F.cr);
  auto load = [&](int c0) {
    const int cn = min(CH, n - c0), cb = c0 / F.block;
#pragma unroll
    for (int k = 0; k < NWC; k++) {
      const int wi = tid + k * PT_;
      cw[k] = wi < cn * WPT ? __ldg(CC + (size_t)c0 * WPT + wi) : 0u;
    }
    rw = tid < cn * 8 ? __ldg(CR + (size_t)c0 * 8 + tid) : 0u;
    if (tid < AR / 2) mcw = __ldg(reinterpret_cast<const uint32_t*>(F.Mc) + (size_t)cb * (AR / 2) + tid);
    if (tid < ARD / 2) mrw = __ldg(reinterpret_cast<const uint32_t*>(F.Mr) + (size_t)cb * (ARD / 2) + tid);
    if (tid < cn * NH) gcb = __ldg(F.gc + (size_t)c0 * NH + tid);
    if (tid < cn) {
      grb = __ldg(F.gr + c0 + tid);
      if (PASS == 2) lpv = __ldg(lp + (size_t)m * Tmax + c0 + tid);
    }
    if (tid < cn * AH) irv = __half2float(F.ir[(size_t)(c0 + tid / AH) * F.G + (tid % AH) / (AH / F.G)]);
  };
  load(blockIdx.x * CH);
  for (int t0 = blockIdx.x * CH; t0 < n; t0 += ns * CH) {
    const int tn = min(CH, n - t0);
    const bool hasA = t0 < F.sinks || t - (t0 + tn - 1) < F.window;   // block-uniform: the chunk holds exact tokens
    __syncthreads();
    bool fine = false;
    if (tid < tn) {
      const int sidx = t0 + tid;
      const bool A = sidx < F.sinks || t - sidx < F.window;
      fine = !A && PASS == 2 && lpv >= th;
      tier[tid] = A ? 0 : (fine ? 1 : 2);
      sgr[tid] = fv_gain(grb, F.lor[0], F.spr[0], F.nbits, (float)ARD);
    }
    if (tid < tn * NH) sgc[tid / NH][tid % NH] = fv_gain(gcb, F.loc[tid % NH], F.spc[tid % NH], F.nbits, 128.f);
    if (tid < tn * AH) sir[tid / AH][tid % AH] = irv;
    if (tid < AR / 2) smc[tid] = *reinterpret_cast<__half2*>(&mcw);
    if (tid < ARD / 2) smr[tid] = *reinterpret_cast<__half2*>(&mrw);
    // pass 2, a coded chunk without hot tokens: its scores are pass 1's (same decode, shrinkage and order: bit-identical)
    const int anyf = __syncthreads_or(fine);     // also the barrier after the chunk setup: always executed
    const bool reuse = PASS == 2 && !hasA && !anyf;
#pragma unroll
    for (int k = 0; k < NWC; k++) {
      const int wi = tid + k * PT_;
      if (wi < tn * WPT) {
        const int tt = wi / WPT, c0 = (wi % WPT) * 8, tr = tier[tt];
        if (tr != 0) {
          const float g = sgc[tt][c0 >> 7];
          const int off = tr == 1 ? 0 : 16, sh = tr == 1 ? 0 : 2;
          float4 v0, v1;
          v0.x = slut[off + (((cw[k] >> 0) & 15) >> sh)] * g;  v0.y = slut[off + (((cw[k] >> 4) & 15) >> sh)] * g;
          v0.z = slut[off + (((cw[k] >> 8) & 15) >> sh)] * g;  v0.w = slut[off + (((cw[k] >> 12) & 15) >> sh)] * g;
          v1.x = slut[off + (((cw[k] >> 16) & 15) >> sh)] * g; v1.y = slut[off + (((cw[k] >> 20) & 15) >> sh)] * g;
          v1.z = slut[off + (((cw[k] >> 24) & 15) >> sh)] * g; v1.w = slut[off + (((cw[k] >> 28) & 15) >> sh)] * g;
          *reinterpret_cast<float4*>(&sv[tt][c0]) = v0;
          *reinterpret_cast<float4*>(&sv[tt][c0 + 4]) = v1;
        }
      }
    }
    if (tid < tn * 8) {
      const int tt = tid / 8, c0 = (tid % 8) * 8, tr = tier[tt];
      if (tr != 0) {
        const float g = sgr[tt];
        const int off = tr == 1 ? 0 : 16, sh = tr == 1 ? 0 : 2;
#pragma unroll
        for (int j = 0; j < 8; j++) svr[tt][c0 + j] = slut[off + (((rw >> (4 * j)) & 15) >> sh)] * g;
      }
    }
    if (hasA) {                                  // exact rows: sinks and the recent window (direct loads, rare chunks)
      for (int i = tid; i < tn * AR; i += PT_) {
        const int tt = i / AR, c = i % AR, sidx = t0 + tt;
        if (tier[tt] == 0) {
          const __nv_bfloat16* src = sidx < F.sinks ? F.sink_c + (size_t)sidx * AR : F.ring_c + (size_t)(sidx % F.RW) * AR;
          sv[tt][c] = __bfloat162float(src[c]);
        }
      }
      for (int i = tid; i < tn * ARD; i += PT_) {
        const int tt = i / ARD, c = i % ARD, sidx = t0 + tt;
        if (tier[tt] == 0) {
          const __nv_bfloat16* src = sidx < F.sinks ? F.sink_r + (size_t)sidx * ARD : F.ring_r + (size_t)(sidx % F.RW) * ARD;
          svr[tt][c] = __bfloat162float(src[c]);
        }
      }
    }
    if (t0 + ns * CH < n) load(t0 + ns * CH);     // the next chunk's raw data, in flight during the compute below
    __syncthreads();
    // q . M_b for this chunk's block (shared by every coded token of the chunk)
    float qm = 0.f;
    if (!reuse) {
#pragma unroll
    for (int e = 0; e < AR / 32; e++) qm = fmaf(qv[e], __half2float(reinterpret_cast<const __half*>(smc)[lane + 32 * e]), qm);
#pragma unroll
    for (int e = 0; e < 2; e++) qm = fmaf(qk[e], __half2float(reinterpret_cast<const __half*>(smr)[lane + 32 * e]), qm);
    qm = warp_sum(qm);
    }
    static_assert(CH == 16, "warp_sum16_t / mma: 16 tokens per chunk");
    // lane l owns token l >> 1 for head h (lanes 2i and 2i + 1 hold the same value)
    const int my = lane >> 1;
    const bool valid = my < tn;
    float dm;
    if (reuse) {
      dm = sc_out[((size_t)m * AH + h) * Tmax + t0 + my];
    } else if (!hasA) {                          // every token coded (and tn == CH): tensor-core scores
      const int tok = 8 * ntl + fg;
      float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4] = {0.f, 0.f, 0.f, 0.f};   // latent, rope partials
#pragma unroll
      for (int i = 0; i < NSTEP; i++) {
        const int st = ks + 8 * i;
        if (st < KS) {
          const bool rope = st >= KL;
          const float* row = rope ? &svr[tok][16 * (st - KL)] : &sv[tok][16 * st];
          const float2 x0 = *reinterpret_cast<const float2*>(row + 2 * f4), x1 = *reinterpret_cast<const float2*>(row + 2 * f4 + 8);
          uint32_t bh0, bl0, bh1, bl1;
          split_bf2(x0.x, x0.y, bh0, bl0);
          split_bf2(x1.x, x1.y, bh1, bl1);
          if (rope) {
            mma_bf16_16816(d, qal[i], bh0, bh1); mma_bf16_16816(d, qah[i], bl0, bl1); mma_bf16_16816(d, qah[i], bh0, bh1);
          } else {
            mma_bf16_16816(c, qal[i], bh0, bh1); mma_bf16_16816(c, qah[i], bl0, bl1); mma_bf16_16816(c, qah[i], bh0, bh1);
          }
        }
      }
      const int t0_ = 8 * ntl + 2 * f4;
      const bool f0 = tier[t0_] == 1, f1 = tier[t0_ + 1] == 1;
      const float k0 = f0 ? F.sfk : F.sck, k1 = f1 ? F.sfk : F.sck, r0 = f0 ? F.srf : F.src_, r1 = f1 ? F.srf : F.src_;
      float* rd = red + (ks * 16 + fg) * RHS + t0_;
      *reinterpret_cast<float2*>(rd) = make_float2(c[0] * k0 + d[0] * r0, c[1] * k1 + d[1] * r1);
      *reinterpret_cast<float2*>(rd + 8 * RHS) = make_float2(c[2] * k0 + d[2] * r0, c[3] * k1 + d[3] * r1);
      __syncthreads();
      const float* rp = red + ((lane & 1) * 4 * 16 + h) * RHS + my;
      float v = rp[0];
#pragma unroll
      for (int k = 1; k < 4; k++) v += rp[k * 16 * RHS];
      dm = v + __shfl_xor_sync(0xffffffff, v, 1);
    } else {
      float s[CH];
#pragma unroll
      for (int tt = 0; tt < CH; tt++) {
        s[tt] = 0.f;
        if (tt >= tn) continue;
        const int tr = tier[tt];
        float dc = 0.f, dr = 0.f;
        if (tr == 0) {
#pragma unroll
          for (int e = 0; e < AR / 32; e++) dc = fmaf(qv[e], sv[tt][lane + 32 * e], dc);
#pragma unroll
          for (int e = 0; e < 2; e++) dr = fmaf(qk[e], svr[tt][lane + 32 * e], dr);
          s[tt] = dc + dr;
        } else {
#pragma unroll
          for (int e = 0; e < AR / 32; e++) dc = fmaf(__ldg(qlr + h * AR + lane + 32 * e), sv[tt][lane + 32 * e], dc);
#pragma unroll
          for (int e = 0; e < 2; e++) dr = fmaf(__ldg(qrr + h * ARD + lane + 32 * e), svr[tt][lane + 32 * e], dr);
          s[tt] = dc * (tr == 1 ? F.sfk : F.sck) + dr * (tr == 1 ? F.srf : F.src_);
        }
      }
      dm = warp_sum16_t(s);
    }
    if (!reuse) {
      if (tier[my] != 0) dm += qm;
      dm *= scale * sir[my][h];
    }
    if (PASS == 1) {
      if (!(lane & 1) && valid) sc_out[((size_t)m * AH + h) * Tmax + t0 + my] = dm;
      continue;
    }
    float mx = valid ? dm : -INFINITY;
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    const float nm = fmaxf(rm, mx), al = __expf(rm - nm);
#pragma unroll
    for (int e = 0; e < AR / 32; e++) ax[e] *= al;
    const float pme = valid ? __expf(dm - nm) : 0.f;   // this lane's token weight
    const int tmy = tier[my];
    float pv = (lane & 1) ? 0.f : pme, pc = ((lane & 1) || tmy == 0) ? 0.f : pme;
#pragma unroll
    for (int o = 16; o; o >>= 1) { pv += __shfl_xor_sync(0xffffffff, pv, o); pc += __shfl_xor_sync(0xffffffff, pc, o); }
    rl = rl * al + pv;
    if (!(lane & 1)) sp[h * RHS + my] = (!valid || tmy == 0) ? 0.f : pme * (tmy == 1 ? F.sfv : F.scv);
    if (lane == 0) sal[h] = al;
    if (hasA) {                                  // exact tokens: per head, fp32
#pragma unroll
      for (int tt = 0; tt < CH; tt++) {
        const float p = __shfl_sync(0xffffffff, pme, 2 * tt);
        if (tt < tn && tier[tt] == 0) {
#pragma unroll
          for (int e = 0; e < AR / 32; e++) ax[e] = fmaf(p, sv[tt][lane + 32 * e], ax[e]);
        }
      }
    }
#pragma unroll
    for (int e = 0; e < AR / 32; e++) ax[e] = fmaf(pc, __half2float(reinterpret_cast<const __half*>(smc)[lane + 32 * e]), ax[e]);
    __syncthreads();
    {                                            // coded values on tensor cores: arm += P (heads x tokens) . V (tokens x slice)
      const float a0 = sal[fg], a8 = sal[fg + 8];
      uint32_t pah[4], pal[4];
      const float* p0 = sp + fg * RHS + 2 * f4;
      split_bf2(p0[0], p0[1], pah[0], pal[0]);
      split_bf2(p0[8 * RHS], p0[8 * RHS + 1], pah[1], pal[1]);
      split_bf2(p0[8], p0[9], pah[2], pal[2]);
      split_bf2(p0[8 * RHS + 8], p0[8 * RHS + 9], pah[3], pal[3]);
      const bool z = tn < CH;                    // rows past the chunk end hold stale data: read as 0
#pragma unroll
      for (int j = 0; j < NT; j++) {
        arm[j][0] *= a0; arm[j][1] *= a0; arm[j][2] *= a8; arm[j][3] *= a8;
        const int cb = h * (AR / 16) + 8 * j + fg;
        float x00 = sv[2 * f4][cb], x01 = sv[2 * f4 + 1][cb], x10 = sv[2 * f4 + 8][cb], x11 = sv[2 * f4 + 9][cb];
        if (z) {
          x00 = 2 * f4 < tn ? x00 : 0.f; x01 = 2 * f4 + 1 < tn ? x01 : 0.f;
          x10 = 2 * f4 + 8 < tn ? x10 : 0.f; x11 = 2 * f4 + 9 < tn ? x11 : 0.f;
        }
        uint32_t bh0, bl0, bh1, bl1;
        split_bf2(x00, x01, bh0, bl0);
        split_bf2(x10, x11, bh1, bl1);
        mma_bf16_16816(arm[j], pal, bh0, bh1); mma_bf16_16816(arm[j], pah, bl0, bl1); mma_bf16_16816(arm[j], pah, bh0, bh1);
      }
    }
    rm = nm;
  }
  if (PASS == 1) return;
  const size_t o = ((size_t)m * ns + blockIdx.x) * AH + h;
  if (lane == 0) { pm[o] = rm; pl[o] = rl; }
#pragma unroll
  for (int e = 0; e < AR / 32; e++) pax[o * AR + lane + 32 * e] = ax[e];
  const size_t ob = ((size_t)m * ns + blockIdx.x) * AH;
#pragma unroll
  for (int j = 0; j < NT; j++) {
    const int c = h * (AR / 16) + 8 * j + 2 * f4;
    *reinterpret_cast<float2*>(par + (ob + fg) * AR + c) = make_float2(arm[j][0], arm[j][1]);
    *reinterpret_cast<float2*>(par + (ob + fg + 8) * AR + c) = make_float2(arm[j][2], arm[j][3]);
  }
}

// merge (grid (H, M)): o_lat[h] = (sum_k w_k ax_k + (sum_k w_k ar_k) @ R) / sum_k w_k l_k, R per 128-channel latent head.
template <int AR>
__global__ void fovea_codes_merge_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pax,
    const float* __restrict__ par, const long* __restrict__ npos, int ch, int ns, const float* __restrict__ R, float* __restrict__ out) {
  const int h = blockIdx.x, m = blockIdx.y, nb = min(ns, ((int)npos[m] + 1 + ch - 1) / ch);
  pm += (size_t)m * ns * AH; pl += (size_t)m * ns * AH; pax += (size_t)m * ns * AH * AR; par += (size_t)m * ns * AH * AR;
  out += (size_t)m * AH * AR;
  __shared__ float rot[AR];
  float mx = -INFINITY;
  for (int k = 0; k < nb; k++) mx = fmaxf(mx, pm[(size_t)k * AH + h]);
  float l = 0.f;
  for (int k = 0; k < nb; k++) l += pl[(size_t)k * AH + h] * __expf(pm[(size_t)k * AH + h] - mx);
  for (int e = threadIdx.x; e < AR; e += blockDim.x) {
    float a = 0.f, r = 0.f;
    for (int k = 0; k < nb; k++) {
      const float w = __expf(pm[(size_t)k * AH + h] - mx);
      a += pax[((size_t)k * AH + h) * AR + e] * w;
      r += par[((size_t)k * AH + h) * AR + e] * w;
    }
    out[h * AR + e] = a / l;
    rot[e] = r / l;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < AR; e += blockDim.x) {          // + (rotated accumulator) @ R, per 128-channel head
    const int j = e >> 7, c = e & 127;
    float v = 0.f;
    for (int d = 0; d < 128; d++) v = fmaf(rot[j * 128 + d], R[d * 128 + c], v);
    out[h * AR + e] += v;
  }
}

template <int AR>
static void fovea_codes_launch(torch::Tensor& qr, torch::Tensor& ql, torch::Tensor& qrr, torch::Tensor& qlr, torch::Tensor& npos,
                               int M, int ns, double scale, FoveaCodes& F, double hot_frac, torch::Tensor& R, torch::Tensor& sc,
                               torch::Tensor& lse, torch::Tensor& lp, torch::Tensor& thr, torch::Tensor& pm, torch::Tensor& pl,
                               torch::Tensor& pax, torch::Tensor& par, torch::Tensor& out, int Tmax) {
  constexpr int CH = 16, RED = (9 * 16 * 24 + 16) * sizeof(float);   // dynamic: the mma K-split partials
  static bool attr = false;
  if (!attr) {                                   // static + dynamic shared memory above 48 KB needs the opt-in
    cudaFuncSetAttribute(fovea_codes_k<AR, 1, CH>, cudaFuncAttributeMaxDynamicSharedMemorySize, RED);
    cudaFuncSetAttribute(fovea_codes_k<AR, 2, CH>, cudaFuncAttributeMaxDynamicSharedMemorySize, RED);
    attr = true;
  }
  auto st = c10::cuda::getCurrentCUDAStream();
  fovea_codes_k<AR, 1, CH><<<dim3(ns, M), 16 * 32, RED, st>>>(qr.data_ptr<float>(), ql.data_ptr<float>(), qrr.data_ptr<float>(),
      qlr.data_ptr<float>(), npos.data_ptr<long>(), (float)scale, F, nullptr, nullptr, sc.data_ptr<float>(), Tmax, ns,
      nullptr, nullptr, nullptr, nullptr);
  fovea_lse_k<<<dim3(AH, M), 256, 0, st>>>(sc.data_ptr<float>(), npos.data_ptr<long>(), Tmax, lse.data_ptr<float>());
  fovea_lp_k<<<dim3(16, M), 256, 0, st>>>(sc.data_ptr<float>(), lse.data_ptr<float>(), npos.data_ptr<long>(), Tmax, AH,
                                          F.sinks, F.window, lp.data_ptr<float>());
  fovea_select_k<<<M, 1024, 0, st>>>(lp.data_ptr<float>(), npos.data_ptr<long>(), Tmax, F.sinks, F.window, hot_frac,
                                     thr.data_ptr<float>());
  fovea_codes_k<AR, 2, CH><<<dim3(ns, M), 16 * 32, RED, st>>>(qr.data_ptr<float>(), ql.data_ptr<float>(), qrr.data_ptr<float>(),
      qlr.data_ptr<float>(), npos.data_ptr<long>(), (float)scale, F, lp.data_ptr<float>(), thr.data_ptr<float>(), sc.data_ptr<float>(), Tmax,
      ns, pm.data_ptr<float>(), pl.data_ptr<float>(), pax.data_ptr<float>(), par.data_ptr<float>());
  fovea_codes_merge_k<AR><<<dim3(AH, M), 256, 0, st>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pax.data_ptr<float>(),
      par.data_ptr<float>(), npos.data_ptr<long>(), CH, ns, R.data_ptr<float>(), out.data_ptr<float>());
}

// ---- MLA per-layer query / key preparation for M <= 4 rows in one launch (replaces ~30 small ops per row).
// Blocks 0 .. H-1: head h, every row: q RMSNorm (qk_eps, zero-centred or plain gain) -> bf16, per-head q_map (fp32) -> bf16,
// rorope (q[:rd/2] * U[h, g], q[rd/2:rd] * U[h, g] -> bf16), RoPE of the first n_rope groups (bf16 ops as PyTorch does them),
// q_lat = q_res @ Ures + q_nope @ Unope[g(h)] (fp32); with R given, also the FoveaKV-rotated queries.
// Block H: key side of every row: 1/rms per KV group, k_rope after RoPE, and the cache writes (exact cache at pos, or the
// FoveaKV ring at pos % RW with fp16 1/rms). cos / sin from the device-side positions (fp32 angle, accurate cos, bf16).
// Per-row arithmetic is fixed (the same reduction order for any M), so verification rows reproduce plain decoding.
constexpr int PT = 256;                                   // threads; head_dim = G * rd = 256

__device__ __forceinline__ float bsum256(float v, float* red) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < PT / 32; k++) t += red[k];
  return t;
}

__device__ __forceinline__ void cos_sin_bf(long pos, const float* inv_freq, int i, int half, float& c, float& s) {
  const float ang = (float)pos * inv_freq[i % half];
  c = bfr((float)cos((double)ang));
  s = bfr((float)sin((double)ang));
}

struct MlaPrep {
  const __nv_bfloat16* y; int ldy, M;
  const long* pos; const float* inv_freq;
  const float* qn; const float* qmap; const __nv_bfloat16* U; const __nv_bfloat16* Ures; const __nv_bfloat16* Unope;
  int H, G, D, rd, nr, r, gated, zc, mode, RW, halve;
  float eps;
  float* qr; float* ql; float* qrr; float* qlr; const float* R; const float* R64;
  __nv_bfloat16* kc_r; __nv_bfloat16* kc_c; float* kc_ir; half* kc_irh;
};

__global__ void __launch_bounds__(PT) mla_prep_k(MlaPrep P) {
  __shared__ float red[PT / 32];
  __shared__ float sq[4][PT], sqc[4][PT], sql[4][640], sqr[4][64];
  const int tid = threadIdx.x, half_ = P.rd / 2, D = P.D, nq = P.H * D * (P.gated ? 2 : 1);
  if ((int)blockIdx.x == P.H) {                            // ---- key side
    for (int m = 0; m < P.M; m++) {
      const __nv_bfloat16* y = P.y + (size_t)m * P.ldy;
      const long pos = P.pos[m];
      const __nv_bfloat16* krms = y + nq + P.r;
      const __nv_bfloat16* kr = krms + P.G * D;
      // 1 / rms per group: one warp per group (G <= 8), fixed lane order
      const int w = tid >> 5, lane = tid & 31;
      if (w < P.G) {
        float a = 0.f;
        for (int i = lane; i < D; i += 32) { const float v = __bfloat162float(krms[w * D + i]); a = fmaf(v, v, a); }
        a = warp_sum(a);
        if (lane == 0) {
          float ir = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a, (float)D), P.eps)));
          if (P.halve) ir = __half2float(__float2half_rn(ir));
          if (P.mode == 0) P.kc_ir[(size_t)pos * P.G + w] = ir;
          else P.kc_irh[(size_t)pos * P.G + w] = __float2half_rn(ir);
        }
      }
      const size_t slot = P.mode == 0 ? (size_t)pos : (size_t)(pos % P.RW);
      for (int i = tid; i < P.nr * P.rd; i += PT) {
        const int j = i % P.rd, g0 = i - j;
        float c, s; cos_sin_bf(pos, P.inv_freq, j, half_, c, s);
        const float x = __bfloat162float(kr[i]);
        const float rot = j < half_ ? -__bfloat162float(kr[g0 + j + half_]) : __bfloat162float(kr[g0 + j - half_]);
        P.kc_r[slot * (P.nr * P.rd) + i] = __float2bfloat16_rn(bfr(x * c) + bfr(rot * s));
      }
      for (int i = tid; i < P.r; i += PT) P.kc_c[slot * P.r + i] = y[nq + i];
    }
    return;
  }
  const int h = blockIdx.x, g = h / (P.H / P.G);
  const int hstride = D * (P.gated ? 2 : 1);
  // q RMSNorm
  for (int m = 0; m < P.M; m++) {
    const float v = __bfloat162float(P.y[(size_t)m * P.ldy + h * hstride + tid]);
    const float ss = bsum256(v * v, red);
    const float rs = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)D), P.eps)));
    const float gw = P.zc ? 1.f + P.qn[tid] : P.qn[tid];
    sq[m][tid] = bfr(v * rs * gw);
  }
  __syncthreads();
  if (P.qmap) {                                            // q_e = sum_d q_d qmap[h, d, e]
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    const float* qm = P.qmap + (size_t)h * D * D + tid;
    for (int d = 0; d < D; d++) {
      const float wv = qm[(size_t)d * D];
#pragma unroll
      for (int m = 0; m < 4; m++) if (m < P.M) a[m] = fmaf(sq[m][d], wv, a[m]);
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 4; m++) if (m < P.M) sq[m][tid] = bfr(a[m]);
    __syncthreads();
  }
  {                                                        // rorope: qc[g', i] (G * rd = 256 = PT)
    const int gg = tid / P.rd, i = tid % P.rd;
    const float u = __bfloat162float(P.U[((size_t)h * P.G + gg) * half_ + (i % half_)]);
    for (int m = 0; m < P.M; m++) sqc[m][tid] = bfr(sq[m][i] * u);
  }
  __syncthreads();
  if (tid < P.nr * P.rd) {                                 // q_rope
    const int j = tid % P.rd, g0 = tid - j;
    for (int m = 0; m < P.M; m++) {
      float c, s; cos_sin_bf(P.pos[m], P.inv_freq, j, half_, c, s);
      const float x = sqc[m][tid];
      const float rot = j < half_ ? -sqc[m][g0 + j + half_] : sqc[m][g0 + j - half_];
      const float o = bfr(bfr(x * c) + bfr(rot * s));
      sqr[m][tid] = o;
      P.qr[((size_t)m * P.H + h) * (P.nr * P.rd) + tid] = o;
    }
  }
  // q_lat[j] = sum_k qc[nr rd + k] Ures[k, j] + sum_k q[rd + k] Unope[g, k, j]
  const int nres = (P.G - P.nr) * P.rd, nn = D - P.rd;
  for (int j = tid; j < P.r; j += PT) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < nres; k++) {
      const float wv = __bfloat162float(P.Ures[(size_t)k * P.r + j]);
#pragma unroll
      for (int m = 0; m < 4; m++) if (m < P.M) a[m] = fmaf(sqc[m][P.nr * P.rd + k], wv, a[m]);
    }
    float b[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* un = P.Unope + (size_t)g * nn * P.r + j;
    for (int k = 0; k < nn; k++) {
      const float wv = __bfloat162float(un[(size_t)k * P.r]);
#pragma unroll
      for (int m = 0; m < 4; m++) if (m < P.M) b[m] = fmaf(sq[m][P.rd + k], wv, b[m]);
    }
#pragma unroll
    for (int m = 0; m < 4; m++) if (m < P.M) {
      const float o = a[m] + b[m];
      sql[m][j] = o;
      P.ql[((size_t)m * P.H + h) * P.r + j] = o;
    }
  }
  if (!P.R) return;
  __syncthreads();
  // FoveaKV rotated queries: per 128-channel head of q_lat, (q @ R^T)_j = sum_k q_k R[j, k]; k_rope part with R64
  for (int j = tid; j < P.r; j += PT) {
    const int c0 = j / 128 * 128, jj = j % 128;
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < 128; k++) {
      const float wv = P.R[jj * 128 + k];
#pragma unroll
      for (int m = 0; m < 4; m++) if (m < P.M) a[m] = fmaf(sql[m][c0 + k], wv, a[m]);
    }
#pragma unroll
    for (int m = 0; m < 4; m++) if (m < P.M) P.qlr[((size_t)m * P.H + h) * P.r + j] = a[m];
  }
  if (tid < 64) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < 64; k++) {
      const float wv = P.R64[tid * 64 + k];
#pragma unroll
      for (int m = 0; m < 4; m++) if (m < P.M) a[m] = fmaf(sqr[m][k], wv, a[m]);
    }
#pragma unroll
    for (int m = 0; m < 4; m++) if (m < P.M) P.qrr[((size_t)m * P.H + h) * 64 + tid] = a[m];
  }
}

template <typename T> static T* opt_ptr(const torch::Tensor& t) { return t.numel() ? reinterpret_cast<T*>(t.data_ptr()) : nullptr; }

// y (M, ldy) bf16 = [q|gate per head | c | k_rms | k_rope]; t = [pos, inv_freq, qn, qmap?, U, Ures, Unope, qr, ql, qrr?, qlr?,
// R?, R64?, cache_r, cache_c, cache_ir]; i = [H, G, D, rd, nr, r, gated, zc, mode, halve]; empty tensors = absent.
void mla_prep(torch::Tensor y, int64_t ldy, int64_t M, std::vector<torch::Tensor> t, std::vector<int64_t> i, double eps) {
  TORCH_CHECK(t.size() == 16 && i.size() == 10 && M >= 1, "mla_prep: arguments");
  MlaPrep P;
  P.y = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()); P.ldy = ldy; P.M = M;
  P.pos = t[0].data_ptr<long>(); P.inv_freq = t[1].data_ptr<float>(); P.qn = t[2].data_ptr<float>();
  P.qmap = opt_ptr<const float>(t[3]); P.U = opt_ptr<const __nv_bfloat16>(t[4]);
  P.Ures = opt_ptr<const __nv_bfloat16>(t[5]); P.Unope = opt_ptr<const __nv_bfloat16>(t[6]);
  P.qr = t[7].data_ptr<float>(); P.ql = t[8].data_ptr<float>(); P.qrr = opt_ptr<float>(t[9]); P.qlr = opt_ptr<float>(t[10]);
  P.R = opt_ptr<const float>(t[11]); P.R64 = opt_ptr<const float>(t[12]);
  P.kc_r = opt_ptr<__nv_bfloat16>(t[13]); P.kc_c = opt_ptr<__nv_bfloat16>(t[14]);
  P.H = i[0]; P.G = i[1]; P.D = i[2]; P.rd = i[3]; P.nr = i[4]; P.r = i[5]; P.gated = i[6]; P.zc = i[7]; P.mode = i[8]; P.halve = i[9];
  P.kc_ir = P.mode == 0 ? opt_ptr<float>(t[15]) : nullptr; P.kc_irh = P.mode == 1 ? opt_ptr<half>(t[15]) : nullptr;
  P.RW = P.mode == 1 ? t[13].size(0) : 0; P.eps = eps;
  TORCH_CHECK(P.D == PT && P.G * P.rd == PT && P.r <= 640 && P.nr * P.rd <= 64 && P.G <= 8, "mla_prep: shapes");
  for (int m0 = 0; m0 < M; m0 += 4) {                      // groups of <= 4 rows (prefill: any M)
    MlaPrep Q = P;
    Q.M = (int)std::min<int64_t>(4, M - m0);
    Q.y = P.y + (size_t)m0 * ldy; Q.pos = P.pos + m0;
    Q.qr = P.qr + (size_t)m0 * P.H * 64; Q.ql = P.ql + (size_t)m0 * P.H * P.r;
    if (P.qrr) Q.qrr = P.qrr + (size_t)m0 * P.H * 64;
    if (P.qlr) Q.qlr = P.qlr + (size_t)m0 * P.H * P.r;
    mla_prep_k<<<P.H + 1, PT, 0, c10::cuda::getCurrentCUDAStream()>>>(Q);
  }
}

// ---- MLA output for M rows: o[m, h, d] = bf16(sum_j olat[m, h, j] Wv[g(h), j, d]) (* bf16(sigmoid(gate)) -> bf16).
__global__ void __launch_bounds__(PT) mla_out_k(const float* __restrict__ olat, const __nv_bfloat16* __restrict__ WvT,
    const __nv_bfloat16* __restrict__ y, int ldy, int M, int H, int G, int D, int r, int gated, __nv_bfloat16* __restrict__ o, int ldo) {
  __shared__ float so[4][640];
  const int h = blockIdx.x, g = h / (H / G), tid = threadIdx.x;
  for (int m = 0; m < M; m++)
    for (int j = tid; j < r; j += PT) so[m][j] = olat[((size_t)m * H + h) * r + j];
  __syncthreads();
  for (int d = tid; d < D; d += PT) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* w = WvT + (size_t)g * r * D + d;
    for (int j = 0; j < r; j++) {
      const float wv = __bfloat162float(w[(size_t)j * D]);
#pragma unroll
      for (int m = 0; m < 4; m++) if (m < M) a[m] = fmaf(so[m][j], wv, a[m]);
    }
#pragma unroll
    for (int m = 0; m < 4; m++) if (m < M) {
      float v = bfr(a[m]);
      if (gated) {
        const float gt = __bfloat162float(y[(size_t)m * ldy + h * 2 * D + D + d]);
        v = bfr(v * bfr(__fdiv_rn(1.f, 1.f + (float)exp(-(double)gt))));
      }
      o[(size_t)m * ldo + h * D + d] = __float2bfloat16_rn(v);
    }
  }
}

void mla_out(torch::Tensor olat, torch::Tensor WvT, torch::Tensor y, int64_t ldy, int64_t M, int64_t H, bool gated,
             torch::Tensor o, int64_t ldo) {
  const int G = WvT.size(0), r = WvT.size(1), D = WvT.size(2);
  TORCH_CHECK(M >= 1 && r <= 640, "mla_out: shapes");
  for (int64_t m0 = 0; m0 < M; m0 += 4)                    // groups of <= 4 rows (prefill: any M)
    mla_out_k<<<H, PT, 0, c10::cuda::getCurrentCUDAStream()>>>(olat.data_ptr<float>() + m0 * H * r,
        reinterpret_cast<const __nv_bfloat16*>(WvT.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()) + m0 * ldy,
        ldy, (int)std::min<int64_t>(4, M - m0), H, G, D, r, gated ? 1 : 0,
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()) + m0 * ldo, ldo);
}

// ---- FoveaKV encode in decode (one launch per layer for M rows): row m encodes token s = max(pos_m - window, 0), the one
// leaving the exact window, from the exact ring: per head (the latent's 128-channel heads; k_rope is one 64-channel head):
// block mean (fp16), y = (x - M) @ R^T, g = |y|, index = bucketize(y / g sqrt(hd), bnd) (4 bits), gain code
// floor((log g - lo) / span 2^bits) clamped. Same arithmetic as FoveaCodec.encode_codes (IEEE div / sqrt, accurate log).
struct FoveaEnc {
  const __nv_bfloat16* ring; int RW, C, hd; const float* R; const float* lo; const float* span;
  uint8_t* codes; uint8_t* gains; int gstride; half* means;
};

__global__ void __launch_bounds__(128) fovea_encode_k(FoveaEnc E0, FoveaEnc E1, int nh0, const long* __restrict__ pos,
                                                     int window, int block, int nbits) {
  __shared__ float red[4], sx[128];
  __shared__ int sidx[128];
  const int m = blockIdx.x, hb = blockIdx.y, j = threadIdx.x;
  const FoveaEnc& E = hb < nh0 ? E0 : E1;
  const int hh = hb < nh0 ? hb : 0, hd = E.hd, ch = hh * hd + j;
  const long p = pos[m];
  const long s = p - window > 0 ? p - window : 0, b0 = s / block * block;
  float mean = 0.f;
  if (j < hd) {
    float a = 0.f;
    for (int t = 0; t < block; t++) a += __bfloat162float(E.ring[(size_t)((b0 + t) % E.RW) * E.C + ch]);
    const half mh = __float2half_rn(__fdiv_rn(a, (float)block));
    mean = __half2float(mh);
    E.means[(size_t)(s / block) * E.C + ch] = mh;
    sx[j] = __bfloat162float(E.ring[(size_t)(s % E.RW) * E.C + ch]) - mean;
  }
  __syncthreads();
  float yv = 0.f;
  if (j < hd) for (int k = 0; k < hd; k++) yv = fmaf(sx[k], E.R[j * hd + k], yv);
  float ss = warp_sum(j < hd ? yv * yv : 0.f);
  if ((j & 31) == 0) red[j >> 5] = ss;
  __syncthreads();
  const float g = __fsqrt_rn(red[0] + red[1] + red[2] + red[3]);
  const float gc = fmaxf(g, 1e-30f);
  if (j < hd) {
    const float z = __fmul_rn(__fdiv_rn(yv, gc), __fsqrt_rn((float)hd));
    int idx = 0;
#pragma unroll
    for (int b = 0; b < 15; b++) idx += c_bnd16[b] < z;
    sidx[j] = idx;
  }
  __syncthreads();
  if (j < hd && (j & 1) == 0)
    E.codes[(size_t)s * (E.C / 2) + (hh * hd + j) / 2] = (uint8_t)(sidx[j] | (sidx[j + 1] << 4));
  if (j == 0) {
    const float n = (float)(1 << nbits);
    float c = floorf(__fmul_rn(__fdiv_rn((float)log((double)gc) - E.lo[hh], E.span[hh]), n));
    c = fminf(fmaxf(c, 0.f), n - 1.f);
    E.gains[(size_t)s * E.gstride + hh] = (uint8_t)c;
  }
}

// latent / k_rope encoders: [ring, R, lo, span, codes, gains, means] each; i = [window, block, gain_bits]
void fovea_encode(std::vector<torch::Tensor> a, std::vector<torch::Tensor> b, torch::Tensor pos, int64_t M, std::vector<int64_t> i) {
  auto mk = [](std::vector<torch::Tensor>& v) {
    FoveaEnc E;
    E.ring = reinterpret_cast<const __nv_bfloat16*>(v[0].data_ptr()); E.RW = v[0].size(0); E.C = v[0].size(1);
    E.hd = v[1].size(0); E.R = v[1].data_ptr<float>(); E.lo = v[2].data_ptr<float>(); E.span = v[3].data_ptr<float>();
    E.codes = v[4].data_ptr<uint8_t>(); E.gains = v[5].data_ptr<uint8_t>(); E.gstride = v[5].dim() == 2 ? v[5].size(1) : 1;
    E.means = reinterpret_cast<half*>(v[6].data_ptr());
    TORCH_CHECK(E.hd <= 128 && E.C % E.hd == 0, "fovea_encode: head size");
    return E;
  };
  FoveaEnc E0 = mk(a), E1 = mk(b);
  dim3 grid(M, E0.C / E0.hd + E1.C / E1.hd);
  TORCH_CHECK(E1.C == E1.hd, "fovea_encode: the second family is one head");
  fovea_encode_k<<<grid, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(E0, E1, E0.C / E0.hd, pos.data_ptr<long>(), i[0], i[1], i[2]);
}

void fovea_set_luts(torch::Tensor lut16, torch::Tensor lut4, torch::Tensor bnd16) {
  cudaMemcpyToSymbol(c_lut16, lut16.data_ptr<float>(), 16 * sizeof(float));
  cudaMemcpyToSymbol(c_lut4, lut4.data_ptr<float>(), 4 * sizeof(float));
  cudaMemcpyToSymbol(c_bnd16, bnd16.data_ptr<float>(), 15 * sizeof(float));
}

// t: list of tensors in FoveaCodes order [cc, gc, cr, gr, Mc, Mr, loc, spc, lor, spr, sink_c, sink_r, ring_c, ring_r, ir];
// f: [sfk, sfv, sck, scv, srf, src]; i: [sinks, window, block, gain_bits].
void fovea_codes_attn(torch::Tensor qr, torch::Tensor ql, torch::Tensor qrr, torch::Tensor qlr, torch::Tensor npos, int64_t M,
                      int64_t ns, double scale,
                      std::vector<torch::Tensor> t, std::vector<double> f, std::vector<int64_t> i, double hot_frac,
                      torch::Tensor R, torch::Tensor sc, torch::Tensor lse, torch::Tensor lp, torch::Tensor thr,
                      torch::Tensor pm, torch::Tensor pl, torch::Tensor pax, torch::Tensor par, torch::Tensor out) {
  TORCH_CHECK(t.size() == 15 && f.size() == 6 && i.size() == 4, "fovea_codes_attn: argument lists");
  FoveaCodes F;
  F.cc = t[0].data_ptr<uint8_t>(); F.gc = t[1].data_ptr<uint8_t>(); F.cr = t[2].data_ptr<uint8_t>(); F.gr = t[3].data_ptr<uint8_t>();
  F.Mc = reinterpret_cast<const half*>(t[4].data_ptr()); F.Mr = reinterpret_cast<const half*>(t[5].data_ptr());
  F.loc = t[6].data_ptr<float>(); F.spc = t[7].data_ptr<float>(); F.lor = t[8].data_ptr<float>(); F.spr = t[9].data_ptr<float>();
  F.sink_c = reinterpret_cast<const __nv_bfloat16*>(t[10].data_ptr()); F.sink_r = reinterpret_cast<const __nv_bfloat16*>(t[11].data_ptr());
  F.ring_c = reinterpret_cast<const __nv_bfloat16*>(t[12].data_ptr()); F.ring_r = reinterpret_cast<const __nv_bfloat16*>(t[13].data_ptr());
  F.ir = reinterpret_cast<const half*>(t[14].data_ptr());
  F.RW = t[12].size(0); F.G = t[14].size(1);
  F.sinks = i[0]; F.window = i[1]; F.block = i[2]; F.nbits = i[3];
  F.sfk = f[0]; F.sfv = f[1]; F.sck = f[2]; F.scv = f[3]; F.srf = f[4]; F.src_ = f[5];
  const int r = t[12].size(1), Tmax = t[0].size(0);
  TORCH_CHECK(F.block % 16 == 0 && F.block == 128, "fovea_codes_attn: 128-token blocks");
  TORCH_CHECK(sc.numel() >= M * AH * Tmax && lp.numel() >= M * Tmax && thr.numel() >= M && lse.numel() >= M * AH &&
              pm.numel() >= M * ns * AH && pax.numel() >= M * ns * AH * r && par.numel() >= M * ns * AH * r, "fovea_codes_attn: scratch");
  switch (r) {
    case 384: fovea_codes_launch<384>(qr, ql, qrr, qlr, npos, M, ns, scale, F, hot_frac, R, sc, lse, lp, thr, pm, pl, pax, par, out, Tmax); break;
    case 512: fovea_codes_launch<512>(qr, ql, qrr, qlr, npos, M, ns, scale, F, hot_frac, R, sc, lse, lp, thr, pm, pl, pax, par, out, Tmax); break;
    case 640: fovea_codes_launch<640>(qr, ql, qrr, qlr, npos, M, ns, scale, F, hot_frac, R, sc, lse, lp, thr, pm, pl, pax, par, out, Tmax); break;
    default: TORCH_CHECK(false, "fovea_codes_attn: unsupported latent rank ", r);
  }
}

// ---- packed (N, K) -> bf16((q - 8) d) / bf16(q d): the pipeline's reconstruction, for the prefill GEMMs.
__global__ void dequant_bf16_k(const int32_t* __restrict__ c4, const int8_t* __restrict__ c8, const half* __restrict__ sc,
                               __nv_bfloat16* __restrict__ out, long n, int K) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;          // one thread per 8 weights
  if (i >= n / 8) return;
  const long e = i * 8, row = e / K, k = e % K;
  const float d = __half2float(sc[row * (K / 32) + k / 32]);
  __align__(16) __nv_bfloat16 o[8];
  if (c4) {
    const uint32_t w = (uint32_t)c4[i];
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = __float2bfloat16_rn((float)((int)((w >> (4 * j)) & 15) - 8) * d);
  } else {
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = __float2bfloat16_rn((float)c8[e + j] * d);
  }
  *reinterpret_cast<uint4*>(out + e) = *reinterpret_cast<uint4*>(o);
}

void dequant_bf16(torch::Tensor codes, torch::Tensor scales, int64_t fmt, torch::Tensor out) {
  const int N = scales.size(0), K = scales.size(1) * 32;
  const long n = (long)N * K;
  TORCH_CHECK(out.numel() >= n && out.scalar_type() == at::kBFloat16, "dequant_bf16: out");
  dequant_bf16_k<<<(unsigned)((n / 8 + 255) / 256), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      fmt == 0 ? reinterpret_cast<const int32_t*>(codes.data_ptr()) : nullptr,
      fmt == 1 ? reinterpret_cast<const int8_t*>(codes.data_ptr()) : nullptr,
      reinterpret_cast<const half*>(scales.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_bf16", &dequant_bf16);
  m.def("packed_gemv", &packed_gemv);
  m.def("packed_gemv_u", &packed_gemv_u);
  m.def("rms_rows", &rms_rows);
  m.def("emb_q4", &emb_q4);
  m.def("mla_lat_attn", &mla_lat_attn);
  m.def("gdn2_step", &gdn2_step);
  m.def("gqa_attn", &gqa_attn);
  m.def("fovea_codes_attn", &fovea_codes_attn);
  m.def("fovea_set_luts", &fovea_set_luts);
  m.def("mla_prep", &mla_prep);
  m.def("mla_out", &mla_out);
  m.def("fovea_encode", &fovea_encode);
}
