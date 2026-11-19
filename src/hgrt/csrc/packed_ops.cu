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

__global__ void __launch_bounds__(GD) gdn2_step_k(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* csi,
    __nv_bfloat16* cso, const __nv_bfloat16* __restrict__ cw, const __nv_bfloat16* __restrict__ gates,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias, const float* Si, float* So,
    const __nv_bfloat16* __restrict__ z,
    const float* __restrict__ gn, float gn_eps, __nv_bfloat16* __restrict__ out, int H, int Hq) {
  const int h = blockIdx.x, v = threadIdx.x;              // q / k heads expanded to the value heads (Hq == H)
  const int KD = Hq * GD;
  __shared__ float sq[GD], sk[GD], sg[GD], sb[GD], red[4];
  const float qv = conv_ch(qkv, csi, cso, cw, h * GD + v);
  const float kv = conv_ch(qkv, csi, cso, cw, KD + h * GD + v);
  const float vv = conv_ch(qkv, csi, cso, cw, 2 * KD + h * GD + v);
  const float q2 = block_sum128(qv * qv, red), k2 = block_sum128(kv * kv, red);
  sq[v] = qv * rsqrtf(q2 + 1e-6f) * rsqrtf((float)GD);
  sk[v] = kv * rsqrtf(k2 + 1e-6f);
  const float a = __bfloat162float(gates[h * GD + v]);
  const float sp = a + dt_bias[h * GD + v];
  const float softplus = sp > 20.f ? sp : log1pf(__expf(sp));
  sg[v] = __expf(-__expf(A_log[h * GD + v]) * softplus);
  sb[v] = bf(1.f / (1.f + __expf(-__bfloat162float(gates[H * GD + h * GD + v])))) * sk[v];
  const float wv = bf(1.f / (1.f + __expf(-__bfloat162float(gates[2 * H * GD + h * GD + v])))) * __bfloat162float(__float2bfloat16(vv));
  __syncthreads();
  const float* Sih = Si + (size_t)h * GD * GD;
  float* Soh = So + (size_t)h * GD * GD;
  float col[GD];
  float erase = 0.f;
#pragma unroll
  for (int k = 0; k < GD; k++) {
    col[k] = Sih[k * GD + v] * sg[k];
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
  const float zz = __bfloat162float(z[h * GD + v]);
  out[h * GD + v] = __float2bfloat16(hn * (zz / (1.f + __expf(-zz))));
}

// conv / recurrent state read from (csi, Si) and written to (cso, So): the same tensors for plain decoding, per-row slots
// for speculative verification (row i reads slot i - 1, writes slot i). Each block reads only its own head's state.
void gdn2_step(torch::Tensor qkv, torch::Tensor csi, torch::Tensor cso, torch::Tensor cw, torch::Tensor gates,
               torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor Si, torch::Tensor So, torch::Tensor z,
               torch::Tensor gn, double gn_eps, torch::Tensor out, int64_t H, int64_t Hq) {
  TORCH_CHECK(Si.size(-1) == GD && csi.size(-1) == GKC, "gdn2_step: Dk = Dv = 128, conv 4");
  TORCH_CHECK(H == Hq, "gdn2_step: q/k heads must be expanded to the value heads (num_qk_heads == num_heads)");
  gdn2_step_k<<<H, GD, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(csi.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(cso.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(cw.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gates.data_ptr()), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      Si.data_ptr<float>(), So.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()), gn.data_ptr<float>(), (float)gn_eps,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, Hq);
}

// ---- GQA decode attention for the MTP layer: q (H, 256) fp32; K, V (Tmax, G, 256) bf16; out (H, 256) fp32.
// Split over T (chunks of 32 tokens) and KV groups; one warp per head of the group; device-side length (graph-safe).
constexpr int QD = 256, QCH = 32;

__global__ void __launch_bounds__(4 * 32) gqa_attn_k(const float* __restrict__ q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ Vc, const long* __restrict__ npos, int H, int G, float scale,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
  const int n = (int)npos[0] + 1, t0 = blockIdx.x * QCH, g = blockIdx.y;
  if (t0 >= n) return;
  const int tn = min(QCH, n - t0), hpg = H / G;
  __shared__ __nv_bfloat16 sk[QCH][QD], sv[QCH][QD];
  for (int i = threadIdx.x; i < tn * QD / 8; i += blockDim.x) {
    const int tt = i / (QD / 8), cc = i % (QD / 8);
    reinterpret_cast<uint4*>(sk[tt])[cc] = reinterpret_cast<const uint4*>(K + ((size_t)(t0 + tt) * G + g) * QD)[cc];
    reinterpret_cast<uint4*>(sv[tt])[cc] = reinterpret_cast<const uint4*>(Vc + ((size_t)(t0 + tt) * G + g) * QD)[cc];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp >= hpg) return;
  const int h = g * hpg + warp;
  float qv[QD / 32];
#pragma unroll
  for (int e = 0; e < QD / 32; e++) qv[e] = q[h * QD + lane + 32 * e];
  float s[QCH], mx = -INFINITY;
  for (int tt = 0; tt < tn; tt++) {
    float d = 0.f;
#pragma unroll
    for (int e = 0; e < QD / 32; e++) d = fmaf(qv[e], __bfloat162float(sk[tt][lane + 32 * e]), d);
    s[tt] = warp_sum(d) * scale;
    mx = fmaxf(mx, s[tt]);
  }
  float l = 0.f, acc[QD / 32];
#pragma unroll
  for (int e = 0; e < QD / 32; e++) acc[e] = 0.f;
  for (int tt = 0; tt < tn; tt++) {
    const float p = __expf(s[tt] - mx);
    l += p;
#pragma unroll
    for (int e = 0; e < QD / 32; e++) acc[e] = fmaf(p, __bfloat162float(sv[tt][lane + 32 * e]), acc[e]);
  }
  const size_t o = (size_t)blockIdx.x * H + h;
  if (lane == 0) { pm[o] = mx; pl[o] = l; }
#pragma unroll
  for (int e = 0; e < QD / 32; e++) pacc[o * QD + lane + 32 * e] = acc[e];
}

__global__ void gqa_merge_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                            const long* __restrict__ npos, int H, float* __restrict__ out) {
  const int h = blockIdx.x, nch = ((int)npos[0] + 1 + QCH - 1) / QCH;
  float m = -INFINITY, l = 0.f;
  for (int k = 0; k < nch; k++) m = fmaxf(m, pm[(size_t)k * H + h]);
  for (int k = 0; k < nch; k++) l += pl[(size_t)k * H + h] * __expf(pm[(size_t)k * H + h] - m);
  for (int e = threadIdx.x; e < QD; e += blockDim.x) {
    float a = 0.f;
    for (int k = 0; k < nch; k++) a += pacc[((size_t)k * H + h) * QD + e] * __expf(pm[(size_t)k * H + h] - m);
    out[h * QD + e] = a / l;
  }
}

void gqa_attn(torch::Tensor q, torch::Tensor K, torch::Tensor Vc, torch::Tensor npos, double scale, torch::Tensor pm,
              torch::Tensor pl, torch::Tensor pacc, torch::Tensor out) {
  const int H = q.size(0), Tmax = K.size(0), G = K.size(1);
  TORCH_CHECK(q.size(1) == QD && K.size(2) == QD && H % G == 0 && H / G <= 4, "gqa_attn: D 256, <= 4 heads per group");
  auto st = c10::cuda::getCurrentCUDAStream();
  dim3 grid((Tmax + QCH - 1) / QCH, G);
  gqa_attn_k<<<grid, 4 * 32, 0, st>>>(q.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(Vc.data_ptr()), npos.data_ptr<long>(), H, G, (float)scale,
      pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>());
  gqa_merge_k<<<H, 128, 0, st>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(), npos.data_ptr<long>(),
                                  H, out.data_ptr<float>());
}

__device__ __forceinline__ float warp_max_f(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

template <int AR>
__global__ void mla_lat_merge_ch_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                   const long* __restrict__ npos, int ch, float* __restrict__ out) {
  const int h = blockIdx.x;
  const int nch = ((int)npos[0] + 1 + ch - 1) / ch;
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

// per head: logsumexp over s <= t of sc[h, s]; then lp[s] = max_h (sc[h, s] - lse_h) for s in comp, else -inf.
__global__ void fovea_lse_k(const float* __restrict__ sc, const long* __restrict__ npos, int Tmax, float* __restrict__ lse) {
  const int h = blockIdx.x, n = (int)npos[0] + 1;
  __shared__ float red[32];
  float m = -INFINITY;
  for (int s = threadIdx.x; s < n; s += blockDim.x) m = fmaxf(m, sc[(size_t)h * Tmax + s]);
  m = warp_max_f(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY; v = warp_max_f(v); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  const float M = red[0];
  __syncthreads();
  float acc = 0.f;
  for (int s = threadIdx.x; s < n; s += blockDim.x) acc += expf(sc[(size_t)h * Tmax + s] - M);
  acc = warp_sum(acc);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { float tot = 0.f; for (int w = 0; w < (int)blockDim.x / 32; w++) tot += red[w]; lse[h] = M + logf(tot); }
}

__global__ void fovea_lp_k(const float* __restrict__ sc, const float* __restrict__ lse, const long* __restrict__ npos,
                           int Tmax, int H, int sinks, int window, float* __restrict__ lp) {
  const int t = (int)npos[0];
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < Tmax; s += gridDim.x * blockDim.x) {
    float v = -INFINITY;
    if (s <= t && s >= sinks && t - s >= window)
      for (int h = 0; h < H; h++) v = fmaxf(v, sc[(size_t)h * Tmax + s] - lse[h]);
    lp[s] = v;
  }
}

// threshold = the k-th largest lp over comp (k = min(ceil(hot_frac (t + 1)), |comp|)); +inf if k = 0 (no hot tokens).
// Exact: bisection over the order-preserving uint32 key of the float (32 counting passes, one block).
__device__ __forceinline__ unsigned fkey(float f) { unsigned u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
__device__ __forceinline__ float kfloat(unsigned k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k); }

__global__ void fovea_select_k(const float* __restrict__ lp, const long* __restrict__ npos, int sinks, int window,
                               double hot_frac, float* __restrict__ thr) {
  const int t = (int)npos[0];
  const int lo_s = sinks, hi_s = t - window;               // comp = [sinks, t - window]
  const int ncomp = max(0, hi_s - lo_s + 1);
  const long kk = min((long)ceil(hot_frac * ((double)t + 1.0)), (long)ncomp);
  __shared__ unsigned red[32];
  if (kk <= 0) { if (threadIdx.x == 0) thr[0] = INFINITY; return; }
  unsigned lo = 0u, hi = 0xFFFFFFFFu;                      // find the largest key K with count(key >= K) >= kk
  while (lo < hi) {
    const unsigned mid = lo + (unsigned)(((unsigned long long)hi - lo + 1ull) / 2ull);
    unsigned c = 0;
    for (int s = lo_s + threadIdx.x; s <= hi_s; s += blockDim.x) c += fkey(lp[s]) >= mid;
    for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(0xffffffff, c, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = c;
    __syncthreads();
    unsigned tot = 0;
    for (int w = 0; w < (int)blockDim.x / 32; w++) tot += red[w];
    __syncthreads();
    if (tot >= (unsigned)kk) lo = mid; else hi = mid - 1;
  }
  if (threadIdx.x == 0) thr[0] = kfloat(lo);
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
#pragma unroll
    for (int m = 0; m < M; m++) {
      float xf[32];
      load_x32(X + (size_t)m * ldx + g * 32, xf);
      float xs = 0.f;
      if (FMT == 0) {
#pragma unroll
        for (int e = 0; e < 32; e++) xs += xf[e];
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        float part = 0.f;
        if (FMT == 0) {
          const uint32_t wd[4] = {pa[r].x, pa[r].y, pa[r].z, pa[r].w};
#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int i = 0; i < 8; i++) part = fmaf(nibf(wd[j], i), xf[j * 8 + i], part);
          acc[r][m] = fmaf(sc[r], part - 8.f * xs, acc[r][m]);
        } else {
          const int8_t* ca = reinterpret_cast<const int8_t*>(&pa[r]);
          const int8_t* cb = reinterpret_cast<const int8_t*>(&pb[r]);
#pragma unroll
          for (int e = 0; e < 16; e++) { part = fmaf((float)ca[e], xf[e], part); part = fmaf((float)cb[e], xf[16 + e], part); }
          acc[r][m] = fmaf(sc[r], part, acc[r][m]);
        }
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
// One block per CH-token chunk (CH divides the 128-token block: one block mean per chunk), one warp per head (16 warps).
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

__device__ __forceinline__ float fv_gain(int code, float lo, float span, int nbits, float hd) {
  return expf(lo + ((float)code + 0.5f) / (float)(1 << nbits) * span) / sqrtf(hd);
}

template <int AR, int PASS, int CH>
__global__ void __launch_bounds__(16 * 32) fovea_codes_k(const float* __restrict__ qr, const float* __restrict__ ql,
    const float* __restrict__ qrr, const float* __restrict__ qlr, const long* __restrict__ npos, float scale, FoveaCodes F,
    const float* __restrict__ lp, const float* __restrict__ thr, float* __restrict__ sc_out, int Tmax,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pax, float* __restrict__ par) {
  constexpr int NH = AR / 128;
  const int t = (int)npos[0], n = t + 1;
  const int t0 = blockIdx.x * CH;
  if (t0 >= n) return;
  const int tn = min(CH, n - t0), b = t0 / F.block;
  __shared__ __nv_bfloat16 sxc[CH][AR];          // exact latent rows (tier A)
  __shared__ __nv_bfloat16 sxr[CH][ARD];
  __shared__ uint8_t scc[CH][AR / 2];            // codes
  __shared__ uint8_t scr[CH][ARD / 2];
  __shared__ float sgc[CH][NH], sgr[CH];         // decoded gains
  __shared__ float sMc[AR], sMr[ARD];
  __shared__ float sir[CH][AH];
  __shared__ int tier[CH];
  if (threadIdx.x < tn) {
    const int sidx = t0 + threadIdx.x;
    const bool A = sidx < F.sinks || t - sidx < F.window;
    tier[threadIdx.x] = A ? 0 : (PASS == 2 && lp[sidx] >= thr[0] ? 1 : 2);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < tn * (AR / 8); i += blockDim.x) {
    const int tt = i / (AR / 8), cc = i % (AR / 8), sidx = t0 + tt;
    if (tier[tt] == 0) {
      const __nv_bfloat16* src = sidx < F.sinks ? F.sink_c + (size_t)sidx * AR : F.ring_c + (size_t)(sidx % F.RW) * AR;
      reinterpret_cast<uint4*>(sxc[tt])[cc] = reinterpret_cast<const uint4*>(src)[cc];
    }
  }
  for (int i = threadIdx.x; i < tn * (ARD / 8); i += blockDim.x) {
    const int tt = i / (ARD / 8), cc = i % (ARD / 8), sidx = t0 + tt;
    if (tier[tt] == 0) {
      const __nv_bfloat16* src = sidx < F.sinks ? F.sink_r + (size_t)sidx * ARD : F.ring_r + (size_t)(sidx % F.RW) * ARD;
      reinterpret_cast<uint4*>(sxr[tt])[cc] = reinterpret_cast<const uint4*>(src)[cc];
    }
  }
  for (int i = threadIdx.x; i < tn * (AR / 32); i += blockDim.x) {        // codes: AR/2 bytes = AR/32 uint4 per token
    const int tt = i / (AR / 32), cc = i % (AR / 32);
    if (tier[tt] != 0) reinterpret_cast<uint4*>(scc[tt])[cc] = reinterpret_cast<const uint4*>(F.cc + (size_t)(t0 + tt) * (AR / 2))[cc];
  }
  for (int i = threadIdx.x; i < tn * 2; i += blockDim.x) {
    const int tt = i / 2, cc = i % 2;
    if (tier[tt] != 0) reinterpret_cast<uint4*>(scr[tt])[cc] = reinterpret_cast<const uint4*>(F.cr + (size_t)(t0 + tt) * (ARD / 2))[cc];
  }
  for (int i = threadIdx.x; i < tn * NH; i += blockDim.x) {
    const int tt = i / NH, j = i % NH;
    sgc[tt][j] = fv_gain(F.gc[(size_t)(t0 + tt) * NH + j], F.loc[j], F.spc[j], F.nbits, 128.f);
  }
  if (threadIdx.x < tn) sgr[threadIdx.x] = fv_gain(F.gr[t0 + threadIdx.x], F.lor[0], F.spr[0], F.nbits, (float)ARD);
  for (int i = threadIdx.x; i < AR; i += blockDim.x) sMc[i] = __half2float(F.Mc[(size_t)b * AR + i]);
  if (threadIdx.x < ARD) sMr[threadIdx.x] = __half2float(F.Mr[(size_t)b * ARD + threadIdx.x]);
  for (int i = threadIdx.x; i < tn * AH; i += blockDim.x) {
    const int tt = i / AH, h = i % AH;
    sir[tt][h] = __half2float(F.ir[(size_t)(t0 + tt) * F.G + h / (AH / F.G)]);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, h = threadIdx.x >> 5;
  float qv[AR / 32], qvr[AR / 32], qk[2], qkr[2];
#pragma unroll
  for (int e = 0; e < AR / 32; e++) { qv[e] = ql[h * AR + lane + 32 * e]; qvr[e] = qlr[h * AR + lane + 32 * e]; }
#pragma unroll
  for (int e = 0; e < 2; e++) { qk[e] = qr[h * ARD + lane + 32 * e]; qkr[e] = qrr[h * ARD + lane + 32 * e]; }
  // q . M_b for this chunk's block (shared by every coded token of the chunk)
  float qm = 0.f;
#pragma unroll
  for (int e = 0; e < AR / 32; e++) qm = fmaf(qv[e], sMc[lane + 32 * e], qm);
#pragma unroll
  for (int e = 0; e < 2; e++) qm = fmaf(qk[e], sMr[lane + 32 * e], qm);
  qm = warp_sum(qm);
  float s[CH], mx = -INFINITY;
  for (int tt = 0; tt < tn; tt++) {
    const int tr = tier[tt];
    float d = 0.f;
    if (tr == 0) {
#pragma unroll
      for (int e = 0; e < AR / 32; e++) d = fmaf(qv[e], __bfloat162float(sxc[tt][lane + 32 * e]), d);
#pragma unroll
      for (int e = 0; e < 2; e++) d = fmaf(qk[e], __bfloat162float(sxr[tt][lane + 32 * e]), d);
    } else {
      const float shc = tr == 1 ? F.sfk : F.sck, shr = tr == 1 ? F.srf : F.src_;
#pragma unroll
      for (int e = 0; e < AR / 32; e++) {
        const int ch = lane + 32 * e;
        const int idx = (scc[tt][ch >> 1] >> ((ch & 1) * 4)) & 15;
        const float lev = tr == 1 ? c_lut16[idx] : c_lut4[idx >> 2];
        d = fmaf(qvr[e] * sgc[tt][ch >> 7] * shc, lev, d);
      }
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int ch = lane + 32 * e;
        const int idx = (scr[tt][ch >> 1] >> ((ch & 1) * 4)) & 15;
        const float lev = tr == 1 ? c_lut16[idx] : c_lut4[idx >> 2];
        d = fmaf(qkr[e] * sgr[tt] * shr, lev, d);
      }
    }
    d = warp_sum(d);
    if (tr != 0) d += qm;
    d *= scale * sir[tt][h];
    s[tt] = d;
    mx = fmaxf(mx, d);
  }
  if (PASS == 1) {
    if (lane == 0) for (int tt = 0; tt < tn; tt++) sc_out[(size_t)h * Tmax + t0 + tt] = s[tt];
    return;
  }
  float l = 0.f, ax[AR / 32], ar[AR / 32], wc = 0.f;
#pragma unroll
  for (int e = 0; e < AR / 32; e++) { ax[e] = 0.f; ar[e] = 0.f; }
  for (int tt = 0; tt < tn; tt++) {
    const float p = __expf(s[tt] - mx);
    l += p;
    const int tr = tier[tt];
    if (tr == 0) {
#pragma unroll
      for (int e = 0; e < AR / 32; e++) ax[e] = fmaf(p, __bfloat162float(sxc[tt][lane + 32 * e]), ax[e]);
    } else {
      wc += p;
      const float sh = tr == 1 ? F.sfv : F.scv;
#pragma unroll
      for (int e = 0; e < AR / 32; e++) {
        const int ch = lane + 32 * e;
        const int idx = (scc[tt][ch >> 1] >> ((ch & 1) * 4)) & 15;
        const float lev = tr == 1 ? c_lut16[idx] : c_lut4[idx >> 2];
        ar[e] = fmaf(p * sgc[tt][ch >> 7] * sh, lev, ar[e]);
      }
    }
  }
#pragma unroll
  for (int e = 0; e < AR / 32; e++) ax[e] = fmaf(wc, sMc[lane + 32 * e], ax[e]);      // the block-mean part of coded values
  const size_t o = (size_t)blockIdx.x * AH + h;
  if (lane == 0) { pm[o] = mx; pl[o] = l; }
#pragma unroll
  for (int e = 0; e < AR / 32; e++) { pax[o * AR + lane + 32 * e] = ax[e]; par[o * AR + lane + 32 * e] = ar[e]; }
}

// merge: o_lat[h] = (sum_k w_k ax_k + (sum_k w_k ar_k) @ R) / sum_k w_k l_k,  R per 128-channel head of the latent.
template <int AR>
__global__ void fovea_codes_merge_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pax,
    const float* __restrict__ par, const long* __restrict__ npos, int ch, const float* __restrict__ R, float* __restrict__ out) {
  const int h = blockIdx.x, nch = ((int)npos[0] + 1 + ch - 1) / ch;
  __shared__ float rot[AR];
  float m = -INFINITY;
  for (int k = 0; k < nch; k++) m = fmaxf(m, pm[(size_t)k * AH + h]);
  float l = 0.f;
  for (int k = 0; k < nch; k++) l += pl[(size_t)k * AH + h] * __expf(pm[(size_t)k * AH + h] - m);
  for (int e = threadIdx.x; e < AR; e += blockDim.x) {
    float a = 0.f, r = 0.f;
    for (int k = 0; k < nch; k++) {
      const float w = __expf(pm[(size_t)k * AH + h] - m);
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
                               double scale, FoveaCodes& F, double hot_frac, torch::Tensor& R, torch::Tensor& sc,
                               torch::Tensor& lse, torch::Tensor& lp, torch::Tensor& thr, torch::Tensor& pm, torch::Tensor& pl,
                               torch::Tensor& pax, torch::Tensor& par, torch::Tensor& out, int Tmax) {
  constexpr int CH = 16;
  auto st = c10::cuda::getCurrentCUDAStream();
  const int nblk = (Tmax + CH - 1) / CH;
  fovea_codes_k<AR, 1, CH><<<nblk, 16 * 32, 0, st>>>(qr.data_ptr<float>(), ql.data_ptr<float>(), qrr.data_ptr<float>(),
      qlr.data_ptr<float>(), npos.data_ptr<long>(), (float)scale, F, nullptr, nullptr, sc.data_ptr<float>(), Tmax,
      nullptr, nullptr, nullptr, nullptr);
  fovea_lse_k<<<AH, 256, 0, st>>>(sc.data_ptr<float>(), npos.data_ptr<long>(), Tmax, lse.data_ptr<float>());
  fovea_lp_k<<<(Tmax + 255) / 256, 256, 0, st>>>(sc.data_ptr<float>(), lse.data_ptr<float>(), npos.data_ptr<long>(), Tmax, AH,
                                                  F.sinks, F.window, lp.data_ptr<float>());
  fovea_select_k<<<1, 1024, 0, st>>>(lp.data_ptr<float>(), npos.data_ptr<long>(), F.sinks, F.window, hot_frac, thr.data_ptr<float>());
  fovea_codes_k<AR, 2, CH><<<nblk, 16 * 32, 0, st>>>(qr.data_ptr<float>(), ql.data_ptr<float>(), qrr.data_ptr<float>(),
      qlr.data_ptr<float>(), npos.data_ptr<long>(), (float)scale, F, lp.data_ptr<float>(), thr.data_ptr<float>(), nullptr, Tmax,
      pm.data_ptr<float>(), pl.data_ptr<float>(), pax.data_ptr<float>(), par.data_ptr<float>());
  fovea_codes_merge_k<AR><<<AH, 256, 0, st>>>(pm.data_ptr<float>(), pl.data_ptr<float>(), pax.data_ptr<float>(),
      par.data_ptr<float>(), npos.data_ptr<long>(), CH, R.data_ptr<float>(), out.data_ptr<float>());
}

void fovea_set_luts(torch::Tensor lut16, torch::Tensor lut4) {
  cudaMemcpyToSymbol(c_lut16, lut16.data_ptr<float>(), 16 * sizeof(float));
  cudaMemcpyToSymbol(c_lut4, lut4.data_ptr<float>(), 4 * sizeof(float));
}

// t: list of tensors in FoveaCodes order [cc, gc, cr, gr, Mc, Mr, loc, spc, lor, spr, sink_c, sink_r, ring_c, ring_r, ir];
// f: [sfk, sfv, sck, scv, srf, src]; i: [sinks, window, block, gain_bits].
void fovea_codes_attn(torch::Tensor qr, torch::Tensor ql, torch::Tensor qrr, torch::Tensor qlr, torch::Tensor npos, double scale,
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
  switch (r) {
    case 384: fovea_codes_launch<384>(qr, ql, qrr, qlr, npos, scale, F, hot_frac, R, sc, lse, lp, thr, pm, pl, pax, par, out, Tmax); break;
    case 512: fovea_codes_launch<512>(qr, ql, qrr, qlr, npos, scale, F, hot_frac, R, sc, lse, lp, thr, pm, pl, pax, par, out, Tmax); break;
    case 640: fovea_codes_launch<640>(qr, ql, qrr, qlr, npos, scale, F, hot_frac, R, sc, lse, lp, thr, pm, pl, pax, par, out, Tmax); break;
    default: TORCH_CHECK(false, "fovea_codes_attn: unsupported latent rank ", r);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packed_gemv", &packed_gemv);
  m.def("packed_gemv_u", &packed_gemv_u);
  m.def("rms_rows", &rms_rows);
  m.def("emb_q4", &emb_q4);
  m.def("mla_lat_attn", &mla_lat_attn);
  m.def("gdn2_step", &gdn2_step);
  m.def("gqa_attn", &gqa_attn);
  m.def("fovea_codes_attn", &fovea_codes_attn);
  m.def("fovea_set_luts", &fovea_set_luts);
}
