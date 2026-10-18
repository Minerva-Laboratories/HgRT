// Fused single-token ops for the Qwen3.5 decode engine (cairn/engine.py). Rounding to bf16 follows the Hugging Face
// modules at the same points (after norms, after the recurrence output, after the gated norm), so the fused path matches
// the PyTorch path of the engine to bf16 rounding.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }   // round to bf16

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sh) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) sh[w] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; i++) s += sh[i];
  __syncthreads();
  return s;
}

// ---- RMSNorm (Qwen3.5: x * rsqrt(mean x^2 + eps) * w, w = 1 + weight, in fp32, then bf16); one block, 1024 threads
__global__ void rms_k(const __nv_bfloat16* __restrict__ x, const float* __restrict__ w, __nv_bfloat16* __restrict__ y,
                      int n, float eps) {
  __shared__ float sh[32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += 1024) { float v = __bfloat162float(x[i]); ss += v * v; }
  const float r = rsqrtf(block_sum<1024>(ss, sh) / n + eps);
  for (int i = threadIdx.x; i < n; i += 1024) y[i] = __float2bfloat16(__bfloat162float(x[i]) * r * w[i]);
}

// ---- MLP activation: silu(g) * u, gu = [g | u] (2n,) -> (n,)
__global__ void silu_mul_k(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = __bfloat162float(gu[i]), u = __bfloat162float(gu[n + i]);
  y[i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
}

// ---- Gated DeltaNet, one token. conv: causal depthwise conv (kernel 4) over the q|k|v channels, state (C, 4) fp32 holds
// the last 4 inputs; out = silu(sum_j w[c, j] state[c, j]) rounded to bf16.
__global__ void conv_k(const __nv_bfloat16* __restrict__ x, float* __restrict__ st, const float* __restrict__ w,
                       float* __restrict__ y, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float4 s = reinterpret_cast<float4*>(st)[c];
  const float4 k = reinterpret_cast<const float4*>(w)[c];
  s = make_float4(s.y, s.z, s.w, __bfloat162float(x[c]));
  reinterpret_cast<float4*>(st)[c] = s;
  const float a = k.x * s.x + k.y * s.y + k.z * s.z + k.w * s.w;
  y[c] = bf(a / (1.f + __expf(-a)));
}

// recurrence + gated RMSNorm: one block per value head (128 threads = value columns), the state column in registers.
//   q, k: l2-normalized (q also / sqrt(dk)); decay = exp(-exp(A_log) softplus(a + dt_bias)); beta = sigmoid(b)
//   S = decay S;  delta = (v - S^T k) beta;  S += k delta^T;  o = S^T q
//   out = bf16( bf16( bf16(o) * rsqrt(mean bf16(o)^2 + eps) ) * gn ) * silu(z)
constexpr int DK = 128;
__global__ void __launch_bounds__(DK) delta_k(const float* __restrict__ mixed, const __nv_bfloat16* __restrict__ z,
                                               const __nv_bfloat16* __restrict__ ba, const float* __restrict__ negA,
                                               const float* __restrict__ dt_bias, const float* __restrict__ gn,
                                               float* __restrict__ S, __nv_bfloat16* __restrict__ out, int nk, int nv,
                                               float eps) {
  __shared__ float qs[DK], ks[DK], sh[DK / 32];
  const int h = blockIdx.x, j = threadIdx.x, kh = h / (nv / nk);
  const int key_dim = nk * DK;
  const float q = mixed[kh * DK + j], k = mixed[key_dim + kh * DK + j], v = mixed[2 * key_dim + h * DK + j];
  const float rq = rsqrtf(block_sum<DK>(q * q, sh) + 1e-6f), rk = rsqrtf(block_sum<DK>(k * k, sh) + 1e-6f);
  qs[j] = q * rq * rsqrtf((float)DK);
  ks[j] = k * rk;
  __syncthreads();
  const float b = __bfloat162float(ba[h]), a = __bfloat162float(ba[nv + h]) + dt_bias[h];
  const float beta = 1.f / (1.f + __expf(-b));
  const float sp = a > 20.f ? a : log1pf(__expf(a));
  const float dec = __expf(negA[h] * sp);
  float* Sh = S + (size_t)h * DK * DK + j;
  float s[DK];
  float kv = 0.f;
#pragma unroll
  for (int i = 0; i < DK; i++) { s[i] = Sh[i * DK] * dec; kv = fmaf(s[i], ks[i], kv); }
  const float delta = (v - kv) * beta;
  float o = 0.f;
#pragma unroll
  for (int i = 0; i < DK; i++) { s[i] = fmaf(ks[i], delta, s[i]); Sh[i * DK] = s[i]; o = fmaf(s[i], qs[i], o); }
  o = bf(o);
  const float r = rsqrtf(block_sum<DK>(o * o, sh) / DK + eps);
  const float zz = __bfloat162float(z[h * DK + j]);
  out[h * DK + j] = __float2bfloat16(bf(bf(o * r) * gn[j]) * (zz / (1.f + __expf(-zz))));
}

// ---- full attention, before the cache: q/k RMSNorm (w = 1 + weight), partial RoPE (first `rot` dims, rotate_half),
// gate split. y = [ (q | gate) per q head (Hq x 2D), k (Hkv x D), v (Hkv x D) ]. One block per q head, then per k head.
__global__ void attn_pre_k(const __nv_bfloat16* __restrict__ y, const float* __restrict__ qn, const float* __restrict__ kn,
                           const long* __restrict__ pos, const float* __restrict__ inv_freq, float* __restrict__ qo,
                           float* __restrict__ ko, float* __restrict__ vo, __nv_bfloat16* __restrict__ go, int Hq, int Hkv,
                           int D, int rot, float eps) {
  extern __shared__ float xs[];
  __shared__ float sh[32];
  const int b = blockIdx.x, t = threadIdx.x;
  const bool isq = b < Hq;
  const int hh = isq ? b : b - Hq;
  const __nv_bfloat16* src = isq ? y + (size_t)hh * 2 * D : y + (size_t)Hq * 2 * D + (size_t)hh * D;
  const float x = __bfloat162float(src[t]);
  const float r = rsqrtf(block_sum<256>(x * x, sh) / D + eps);
  const float xn = bf(x * r * (isq ? qn[t] : kn[t]));
  xs[t] = xn;
  __syncthreads();
  float outv = xn;
  if (t < rot) {
    const int half = rot / 2, f = t % half;
    const float ang = (float)pos[0] * inv_freq[f];
    const float c = bf(cosf(ang)), s = bf(sinf(ang));
    const float other = t < half ? -xs[t + half] : xs[t - half];
    outv = bf(xn * c + other * s);
  }
  if (isq) {
    qo[(size_t)hh * D + t] = outv;
    go[(size_t)hh * D + t] = src[D + t];
  } else {
    ko[(size_t)hh * D + t] = outv;
    vo[(size_t)hh * D + t] = __bfloat162float(y[(size_t)Hq * 2 * D + (size_t)Hkv * D + (size_t)hh * D + t]);
  }
}

// after the cache: o (Hq*D fp32 attention output) -> bf16(o) * sigmoid(gate) -> bf16
__global__ void attn_post_k(const float* __restrict__ o, const __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ y,
                            int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gg = __bfloat162float(g[i]);
  y[i] = __float2bfloat16(bf(o[i]) / (1.f + __expf(-gg)));
}

// ---- dense bf16 decode attention, GQA group G = 4, head_dim 256, length read from the GPU (graph-capturable).
// Split-K: grid (NS, Hkv), 4 warps per block, a lane holds 8 dims; online softmax per warp, merged per block, then
// combine_k merges the NS partials. K/V layout (Hkv, Tcap, D) bf16; q (Hkv*G, D) fp32 unscaled; out (Hkv*G, D) fp32.
constexpr int AG = 4, AD = 256, AW = 4;
__global__ void __launch_bounds__(AW * 32) dense_split_k(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
                                                         const __nv_bfloat16* __restrict__ vb, const long* __restrict__ pos,
                                                         const int* __restrict__ off, float* __restrict__ pm,
                                                         float* __restrict__ pl, float* __restrict__ pacc, long hstride,
                                                         long tstride, float scale) {
  __shared__ float sm_m[AW][AG], sm_l[AW][AG], sm_acc[AW][AG][AD];
  const int s = blockIdx.x, NS = gridDim.x, h = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (int)pos[0] + 1 - (off ? off[0] : 0);       // tokens [0, n) of this buffer
  const int chunk = (n + NS - 1) / NS, t0 = s * chunk, t1 = min(n, t0 + chunk);
  float qv[AG][8];
#pragma unroll
  for (int g = 0; g < AG; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) qv[g][e] = q[(h * AG + g) * AD + lane * 8 + e] * scale;
  float m[AG], l[AG], acc[AG][8];
#pragma unroll
  for (int g = 0; g < AG; g++) { m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) acc[g][e] = 0.f; }
  const __nv_bfloat16* K = kb + (size_t)h * hstride + lane * 8;
  const __nv_bfloat16* V = vb + (size_t)h * hstride + lane * 8;
  for (int t = t0 + w; t < t1; t += AW) {
    const uint4 ku = __ldg(reinterpret_cast<const uint4*>(K + (size_t)t * tstride));
    const uint4 vu = __ldg(reinterpret_cast<const uint4*>(V + (size_t)t * tstride));
    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
    float kf[8], vf[8];
#pragma unroll
    for (int e = 0; e < 4; e++) { float2 a = __bfloat1622float2(k2[e]); kf[2*e] = a.x; kf[2*e+1] = a.y;
                                  float2 b = __bfloat1622float2(v2[e]); vf[2*e] = b.x; vf[2*e+1] = b.y; }
    float sc[AG];
#pragma unroll
    for (int g = 0; g < AG; g++) { float d = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) d = fmaf(qv[g][e], kf[e], d);
      sc[g] = d; }
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int g = 0; g < AG; g++) sc[g] += __shfl_xor_sync(0xffffffff, sc[g], o);
#pragma unroll
    for (int g = 0; g < AG; g++) {
      const float mn = fmaxf(m[g], sc[g]), c = __expf(m[g] - mn), p = __expf(sc[g] - mn);
      l[g] = l[g] * c + p; m[g] = mn;
#pragma unroll
      for (int e = 0; e < 8; e++) acc[g][e] = fmaf(acc[g][e], c, p * vf[e]);
    }
  }
#pragma unroll
  for (int g = 0; g < AG; g++) {
    if (lane == 0) { sm_m[w][g] = m[g]; sm_l[w][g] = l[g]; }
#pragma unroll
    for (int e = 0; e < 8; e++) sm_acc[w][g][lane * 8 + e] = acc[g][e];
  }
  __syncthreads();
  // merge the 4 warps: thread t handles dims t and t + 128 for every g
  for (int g = 0; g < AG; g++) {
    float M = -INFINITY;
#pragma unroll
    for (int i = 0; i < AW; i++) M = fmaxf(M, sm_m[i][g]);
    float L = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < AW; i++) {
      const float f = sm_m[i][g] == -INFINITY ? 0.f : __expf(sm_m[i][g] - M);
      L += sm_l[i][g] * f; a0 += sm_acc[i][g][threadIdx.x] * f; a1 += sm_acc[i][g][threadIdx.x + 128] * f;
    }
    const size_t o = ((size_t)s * gridDim.y + h) * AG + g;
    if (threadIdx.x == 0) { pm[o] = M; pl[o] = L; }
    pacc[o * AD + threadIdx.x] = a0; pacc[o * AD + threadIdx.x + 128] = a1;
  }
}

__global__ void dense_combine_k(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                float* __restrict__ out, int NS, int HG) {
  const int hg = blockIdx.x, d = threadIdx.x;
  float M = -INFINITY;
  for (int s = 0; s < NS; s++) M = fmaxf(M, pm[s * HG + hg]);
  float L = 0.f, a = 0.f;
  for (int s = 0; s < NS; s++) {
    const float mm = pm[s * HG + hg];
    if (mm == -INFINITY) continue;
    const float f = __expf(mm - M);
    L += pl[s * HG + hg] * f; a += pacc[((size_t)s * HG + hg) * AD + d] * f;
  }
  out[hg * AD + d] = a / L;
}

// kb/vb: head-major (Hkv, Tcap, D) when token_major is false, else token-major (T, Hkv, D); length = pos + 1 - off.
// out given: finished attention; else only the NS partials (to be merged with other partials by the caller).
void dense_attn(torch::Tensor q, torch::Tensor kb, torch::Tensor vb, torch::Tensor pos, c10::optional<torch::Tensor> off,
                torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc, c10::optional<torch::Tensor> out, bool token_major,
                double scale) {
  const int Hkv = token_major ? kb.size(1) : kb.size(0), NS = pm.size(0);
  const long hstride = token_major ? AD : kb.size(1) * AD, tstride = token_major ? (long)Hkv * AD : AD;
  TORCH_CHECK(kb.size(2) == AD && q.size(0) == Hkv * AG, "dense_attn: D 256, G 4");
  dense_split_k<<<dim3(NS, Hkv), AW * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(kb.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(vb.data_ptr()), pos.data_ptr<long>(),
      off.has_value() ? off->data_ptr<int>() : nullptr, pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(),
      hstride, tstride, (float)scale);
  if (!out.has_value()) return;
  dense_combine_k<<<Hkv * AG, AD, 0, c10::cuda::getCurrentCUDAStream()>>>(pm.data_ptr<float>(), pl.data_ptr<float>(),
      pacc.data_ptr<float>(), out->data_ptr<float>(), NS, Hkv * AG);
}

#define BF(t) reinterpret_cast<__nv_bfloat16*>(t.data_ptr())
#define CBF(t) reinterpret_cast<const __nv_bfloat16*>(t.data_ptr())
#define STREAM c10::cuda::getCurrentCUDAStream()

void rms(torch::Tensor x, torch::Tensor w, torch::Tensor y, double eps) {
  rms_k<<<1, 1024, 0, STREAM>>>(CBF(x), w.data_ptr<float>(), BF(y), x.numel(), (float)eps);
}
void silu_mul(torch::Tensor gu, torch::Tensor y) {
  const int n = y.numel();
  silu_mul_k<<<(n + 255) / 256, 256, 0, STREAM>>>(CBF(gu), BF(y), n);
}
void conv(torch::Tensor x, torch::Tensor st, torch::Tensor w, torch::Tensor y) {
  const int C = st.size(0);
  conv_k<<<(C + 255) / 256, 256, 0, STREAM>>>(CBF(x), st.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), C);
}
void delta(torch::Tensor mixed, torch::Tensor z, torch::Tensor ba, torch::Tensor negA, torch::Tensor dt_bias,
           torch::Tensor gn, torch::Tensor S, torch::Tensor out, int64_t nk, int64_t nv, double eps) {
  TORCH_CHECK(S.size(1) == DK && S.size(2) == DK, "head dims must be 128");
  delta_k<<<nv, DK, 0, STREAM>>>(mixed.data_ptr<float>(), CBF(z), CBF(ba), negA.data_ptr<float>(),
                                 dt_bias.data_ptr<float>(), gn.data_ptr<float>(), S.data_ptr<float>(), BF(out), nk, nv,
                                 (float)eps);
}
void attn_pre(torch::Tensor y, torch::Tensor qn, torch::Tensor kn, torch::Tensor pos, torch::Tensor inv_freq,
              torch::Tensor qo, torch::Tensor ko, torch::Tensor vo, torch::Tensor go, int64_t Hq, int64_t Hkv, int64_t rot,
              double eps) {
  const int D = qo.size(1);
  TORCH_CHECK(D == 256, "head_dim 256");
  attn_pre_k<<<Hq + Hkv, D, D * sizeof(float), STREAM>>>(CBF(y), qn.data_ptr<float>(), kn.data_ptr<float>(),
      pos.data_ptr<long>(), inv_freq.data_ptr<float>(), qo.data_ptr<float>(), ko.data_ptr<float>(), vo.data_ptr<float>(),
      BF(go), Hq, Hkv, D, rot, (float)eps);
}
void attn_post(torch::Tensor o, torch::Tensor g, torch::Tensor y) {
  const int n = y.numel();
  attn_post_k<<<(n + 255) / 256, 256, 0, STREAM>>>(o.data_ptr<float>(), CBF(g), BF(y), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms", &rms); m.def("silu_mul", &silu_mul); m.def("conv", &conv); m.def("delta", &delta);
  m.def("attn_pre", &attn_pre); m.def("attn_post", &attn_post); m.def("dense_attn", &dense_attn);
}
