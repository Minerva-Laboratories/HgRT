// Exact top-k token selection for FoveaKV's dynamic 2+2 decode, graph-capturable (all counts on the GPU).
// Score per token t < n and KV head h: s = max over the GQA group g of (logits[h, g, t] - lse[h, g]) (the estimated
// log-probability; FoveaKV.attend's rule). k = min(ceil(k_frac * (pos + 1)), n). Selection = exact top-k by a 3-level
// radix select on the order-preserving 32-bit key of s (12 + 12 + 8 bits): histogram -> find the bin holding the k-th
// key -> narrow inside it -> exact threshold T; then every token with key > T and the first (k - #above) with key == T
// are written to hot[h, :k]; hot[h, k:k_cap] = -1 (the delta kernel skips them). Ties at T are kept in scan order (any
// tie order is a valid top-k, as torch.topk's). One small kernel per level, multi-block over tokens, any n.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

constexpr int G = 4, NB1 = 4096, NB2 = 4096, NB3 = 256, TPB = 256;

__device__ __forceinline__ unsigned key_of(float s) {          // order-preserving: larger float -> larger key
  const unsigned u = __float_as_uint(s);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// level 1: scores + top-12-bit histogram. state[h]: 0 n, 1 k, 2 prefix (bin path), 3 remaining k inside the bin,
// 4 above-count, 5 threshold key, 6 #gt written, 7 #eq written
__global__ void score_hist(const float* __restrict__ logits, const float* __restrict__ lse, int cap, int b,
                           const int* __restrict__ n_dev, const long* __restrict__ pos, float k_frac,
                           float* __restrict__ sel, unsigned* __restrict__ hist, int* __restrict__ state) {
  const int h = blockIdx.y, n = min(*n_dev, b);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int k = min((int)ceilf(k_frac * (float)(*pos + 1)), n);
    state[h * 8 + 0] = n; state[h * 8 + 1] = k; state[h * 8 + 6] = 0; state[h * 8 + 7] = 0;
  }
  __shared__ unsigned sh[NB1];                                  // per-block histogram (scores cluster: global
  for (int i = threadIdx.x; i < NB1; i += blockDim.x) sh[i] = 0;  // atomics on a few bins would serialize)
  __syncthreads();
  float l[G];
#pragma unroll
  for (int g = 0; g < G; g++) l[g] = lse[h * G + g];
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
    float s = -INFINITY;
#pragma unroll
    for (int g = 0; g < G; g++) s = fmaxf(s, logits[((size_t)h * G + g) * cap + t] - l[g]);
    sel[(size_t)h * b + t] = s;
    atomicAdd(&sh[key_of(s) >> 20], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NB1; i += blockDim.x)
    if (sh[i]) atomicAdd(&hist[h * NB1 + i], sh[i]);
}

// find the bin (from the top) where the cumulative count reaches the remaining k; one block (TPB threads) per head:
// thread i sums its chunk of NB / TPB bins (chunks ordered from the top), a scan over the chunk sums finds the chunk
// that crosses k, and that chunk is walked bin by bin (no serial pass over all NB bins)
template <int NB>
__global__ void find_bin(const unsigned* __restrict__ hist, int* __restrict__ state, int level) {
  constexpr int CH = (NB + TPB - 1) / TPB;
  const int h = blockIdx.x, i = threadIdx.x;
  int* st = state + h * 8;
  const int k = (level == 1) ? st[1] : st[3];
  __shared__ unsigned csum[TPB];
  __shared__ long cpre[TPB];
  unsigned c = 0;
  for (int j = 0; j < CH; j++) {                                 // chunk i = bins [NB-1-i*CH ... NB-(i+1)*CH]
    const int bin = NB - 1 - i * CH - j;
    if (bin >= 0) c += hist[h * NB + bin];
  }
  csum[i] = c;
  __syncthreads();
  if (i == 0) {                                                  // 256 adds: exclusive prefix over chunks
    long run = 0;
    for (int j = 0; j < TPB; j++) { cpre[j] = run; run += csum[j]; }
  }
  __syncthreads();
  if (k <= 0) { if (i == 0) { st[2] = -1; st[3] = 0; } return; }
  const long before = cpre[i];
  if (before < k && before + csum[i] >= k) {                     // exactly one thread crosses k
    long above = before;
    int bin = NB - 1 - i * CH;
    for (int j = 0; j < CH; j++, bin--) {
      const unsigned cb = hist[h * NB + bin];
      if (above + cb >= k) break;
      above += cb;
    }
    const unsigned prefix = (level == 1) ? (unsigned)bin : ((unsigned)st[2] << (level == 2 ? 12 : 8)) | (unsigned)bin;
    st[2] = (int)prefix;
    st[3] = k - (int)above;
    st[4] = ((level == 1) ? 0 : st[4]) + (int)above;
  }
}

// levels 2 and 3: histogram of the next bits among tokens whose key matches the prefix so far
__global__ void narrow_hist(const float* __restrict__ sel, int b, const int* __restrict__ state, int level,
                            unsigned* __restrict__ hist) {
  const int h = blockIdx.y;
  const int* st = state + h * 8;
  const int n = st[0], prefix = st[2];
  if (prefix < 0) return;
  const int nb = (level == 2) ? NB2 : NB3;
  __shared__ unsigned sh[NB2];
  for (int i = threadIdx.x; i < nb; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
    const unsigned key = key_of(sel[(size_t)h * b + t]);
    if (level == 2) { if ((int)(key >> 20) == prefix) atomicAdd(&sh[(key >> 8) & 0xFFF], 1u); }
    else { if ((int)(key >> 8) == prefix) atomicAdd(&sh[key & 0xFF], 1u); }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nb; i += blockDim.x)
    if (sh[i]) atomicAdd(&hist[h * nb + i], sh[i]);
}

// write hot[h, :k]: keys above the threshold, then the first ties
__global__ void compact(const float* __restrict__ sel, int b, int* __restrict__ state, int* __restrict__ hot, int k_cap) {
  const int h = blockIdx.y;
  int* st = state + h * 8;
  const int n = st[0], k = st[1];
  if (k <= 0) return;
  const unsigned T = (unsigned)st[2];                          // full 32-bit key after level 3
  const int n_eq = st[3], n_gt = k - n_eq;
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
    const unsigned key = key_of(sel[(size_t)h * b + t]);
    if (key > T) {
      const int slot = atomicAdd(&st[6], 1);
      if (slot < n_gt) hot[h * k_cap + slot] = t;
    } else if (key == T) {
      const int slot = atomicAdd(&st[7], 1);
      if (slot < n_eq) hot[h * k_cap + n_gt + slot] = t;
    }
  }
}

// hot (Hkv, k_cap) int32, sel scratch (Hkv, b) fp32, hist scratch (Hkv, 4096) uint32 x2 + (Hkv, 256), state (Hkv, 8) int32
void select_topk(torch::Tensor logits, torch::Tensor lse, int64_t b, torch::Tensor n_dev, torch::Tensor pos, double k_frac,
                 torch::Tensor sel, torch::Tensor h1, torch::Tensor h2, torch::Tensor h3, torch::Tensor state,
                 torch::Tensor hot) {
  const int Hkv = logits.size(0), cap = logits.size(2), k_cap = hot.size(1);
  auto st = c10::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(h1.data_ptr(), 0, h1.numel() * 4, st);
  cudaMemsetAsync(h2.data_ptr(), 0, h2.numel() * 4, st);
  cudaMemsetAsync(h3.data_ptr(), 0, h3.numel() * 4, st);
  cudaMemsetAsync(hot.data_ptr(), 0xFF, hot.numel() * 4, st);                // -1
  const int nblk = max(1, min((int)((b + TPB * 8 - 1) / (TPB * 8)), 64));
  const dim3 grid(nblk, Hkv);
  unsigned* u1 = (unsigned*)h1.data_ptr(); unsigned* u2 = (unsigned*)h2.data_ptr(); unsigned* u3 = (unsigned*)h3.data_ptr();
  int* s = state.data_ptr<int>();
  score_hist<<<grid, TPB, 0, st>>>(logits.data_ptr<float>(), lse.data_ptr<float>(), cap, (int)b, n_dev.data_ptr<int>(),
                                   pos.data_ptr<long>(), (float)k_frac, sel.data_ptr<float>(), u1, s);
  find_bin<NB1><<<Hkv, TPB, 0, st>>>(u1, s, 1);
  narrow_hist<<<grid, TPB, 0, st>>>(sel.data_ptr<float>(), (int)b, s, 2, u2);
  find_bin<NB2><<<Hkv, TPB, 0, st>>>(u2, s, 2);
  narrow_hist<<<grid, TPB, 0, st>>>(sel.data_ptr<float>(), (int)b, s, 3, u3);
  find_bin<NB3><<<Hkv, TPB, 0, st>>>(u3, s, 3);
  compact<<<grid, TPB, 0, st>>>(sel.data_ptr<float>(), (int)b, s, hot.data_ptr<int>(), k_cap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("select_topk", &select_topk); }
