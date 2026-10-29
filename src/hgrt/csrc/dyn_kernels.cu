// Kernels of hgrt.dyn (FoveaKV dynamic 2+2 decode in a CUDA graph): the coarse pass on the tensor cores, the delta pass,
// and the step glue (append + query rotation, the merges around selection).
//
// FoveaKV coarse (2-bit) gain-shape pass on the tensor cores: same inputs and partial outputs as foveakv
// kernels.cu attn_pass<F_GS> with kfine = vfine = 0 (uint8 gain codes + per-head gain LUT, optional block means,
// range / device-count / page iteration, optional per-token logits).
//
// Exactness. The 2-bit level tables are symmetric, lev = {-b, -a, a, b} (FoveaKV's nested codebook times a scale), so
//   lev[c] = a * s(c) + (b - a) * s(c) * big(c),   s = +-1 (code >= 2), big = (c == 0 || c == 3)
// and both factors are exact in fp16. Keys: logits = kgain * (A1 . (a q) + A2 . ((b - a) q)) + q . kmu, with A1 / A2
// the {-1, 0, 1} matrices of the codes and (a q), ((b - a) q) split into fp16 hi + lo parts placed in the 8 columns of
// one m16n8k16 B operand (4 query heads x {hi, lo}); fp32 accumulation. Values: the same with the softmax weights
// p * vgain in the B columns. Error vs the fp32 FMA kernel: the hi/lo split (~2^-22 relative) and summation order.
//
// Layout (one warp = 16-token tiles; mma.m16n8k16.row.col f16 -> f32):
//  K: A = codes (16 tokens x 16 dims per k-step). Dims are permuted so that lane group tig reads a contiguous D/16-byte
//     segment of each token row: k-step ks, logical columns {2tig, 2tig+1, 2tig+8, 2tig+9} = dims 4*KS*tig + 4ks + {0..3}
//     (one code byte: low nibble -> a0 / a1, high nibble -> a2 / a3). q is permuted the same way in the B registers.
//  V: A = codes^T (16 dims x 16 tokens), m-tile mt, rows {g, g+8} = dims 4*VS*g + 2mt + {0, 1}, columns = the tile's
//     tokens; the two tokens of a register are interleaved 2 bits at a time, then decoded by one 16-entry table.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#ifndef TC_D
#define TC_D 256
#endif
constexpr int D = TC_D;
static_assert(D == 128 || D == 256, "head_dim must be 128 or 256");
constexpr int G = 4, WARPS = 4, RB = D / 4;        // RB: bytes per 2-bit plane row
constexpr int KS = D / 16;                         // key k-steps per tile = code bytes per lane group and row
constexpr int VS = D / 32;                         // value code bytes per lane row-group and token
constexpr int MT = D / 16;                         // value m-tiles

struct TcParams {
  int Hkv, Tcap, t0, n, tpb;
  const int* n_dev; const int* pages; int pages_stride;
  const float* q;                                  // (Hkv, G, D) fp32, rotated, pre-scaled
  const uint8_t *khi, *vhi, *kg8, *vg8; const float *kglut, *vglut;
  const __half *kmu, *vmu; int mu_block;
  const float *klev, *vlev;                        // 4-entry coarse level tables (symmetric)
  float *pm, *pl, *pacc, *logits;
  float* qoff_out;                                 // optional (Hkv, Tcap / mu_block, G): q . kmu per mean block
};

__device__ __forceinline__ void mma(float (&c)[4], uint2 a0, uint2 a1, uint2 a2, uint2 a3, bool second, unsigned b0,
                                    unsigned b1) {
  const unsigned x0 = second ? a0.y : a0.x, x1 = second ? a1.y : a1.x, x2 = second ? a2.y : a2.x,
                 x3 = second ? a3.y : a3.x;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack2(__half lo, __half hi) {
  return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}
__device__ __forceinline__ __half part_of(float x, int part) {   // fp16 hi part (0) or the residual (1)
  const __half hi = __float2half_rn(x);
  return part ? __float2half_rn(x - __half2float(hi)) : hi;
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}
__device__ __forceinline__ float group_max(float x) {            // over the 8 lanes sharing lane & 3
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
  return x;
}
__device__ __forceinline__ float group_sum(float x) {
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

// one warp's staged tile: 16 consecutive token rows of the key / value 2-bit planes and their gain codes (a tile's
// tokens are contiguous in the cache: range mode, or inside one 128-token page). Value rows are padded by 16 bytes
// (fewer bank conflicts for the 4-token reads of the value pass).
constexpr int STAGES = 3, VSTR = RB + 16;
struct Stage { uint8_t k[16][RB]; uint8_t v[16][VSTR]; uint8_t kg[16]; uint8_t vg[16]; };
constexpr int STAGE_BYTES = WARPS * STAGES * (int)sizeof(Stage), SACC_BYTES = WARPS * G * D * 4;

template <int NW> __device__ __forceinline__ void load_words(const uint8_t* src, unsigned* w) {
  if constexpr (NW == 4) { const uint4 v = *reinterpret_cast<const uint4*>(src); w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w; }
  else if constexpr (NW == 2) { const uint2 v = *reinterpret_cast<const uint2*>(src); w[0] = v.x; w[1] = v.y; }
  else w[0] = *reinterpret_cast<const unsigned*>(src);
}

__global__ void __launch_bounds__(32 * WARPS, 3) coarse_tc(TcParams p) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5, g8 = lane >> 2, tig = lane & 3;
  __shared__ uint2 dec[16];                         // nibble (c_lo | c_hi << 2) -> {A1 half2, A2 half2}
  __shared__ float kgl[256], vgl[256];
  __shared__ __align__(16) float qs[G][D];
  __shared__ float ws[WARPS][16][G];                // per-tile softmax weights p * vgain
  __shared__ __align__(16) unsigned char big[STAGE_BYTES > SACC_BYTES ? STAGE_BYTES : SACC_BYTES];
  Stage* stg_w = reinterpret_cast<Stage*>(big) + w * STAGES;          // this warp's pipeline stages
  float (*sacc)[G][D] = reinterpret_cast<float(*)[G][D]>(big);        // after the loop: per-warp partials
  __shared__ float sm_[WARPS][G], sl_[WARPS][G];
  __shared__ uint2 qf[2][KS][32];                   // key-pass B fragments per lane, both level matrices
  if (threadIdx.x < 16) {
    const int c0 = threadIdx.x & 3, c1 = threadIdx.x >> 2;
    const float s0 = c0 >= 2 ? 1.f : -1.f, s1 = c1 >= 2 ? 1.f : -1.f;
    const bool g0 = (c0 == 0 || c0 == 3), g1 = (c1 == 0 || c1 == 3);
    dec[threadIdx.x] = make_uint2(pack2(__float2half(s0), __float2half(s1)),
                                  pack2(__float2half(g0 ? s0 : 0.f), __float2half(g1 ? s1 : 0.f)));
  }
  for (int i = threadIdx.x; i < 256; i += blockDim.x) { kgl[i] = p.kglut[h * 256 + i]; vgl[i] = p.vglut[h * 256 + i]; }
  for (int i = threadIdx.x; i < G * D; i += blockDim.x) qs[i / D][i % D] = p.q[(size_t)h * G * D + i];
  __syncthreads();

  const float ka = p.klev[2], kd = p.klev[3] - p.klev[2], va = p.vlev[2], vd = p.vlev[3] - p.vlev[2];
  const int bh = g8 & 3, bpart = g8 >> 2;           // this lane's B column: query head bh, hi (0) / lo (1) part
  if (w == 0)
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
      const int d = 4 * KS * tig + 4 * ks;
#pragma unroll
      for (int m = 0; m < 2; m++) {
        const float cf = m ? kd : ka;
        qf[m][ks][lane] = make_uint2(pack2(part_of(cf * qs[bh][d], bpart), part_of(cf * qs[bh][d + 1], bpart)),
                                     pack2(part_of(cf * qs[bh][d + 2], bpart), part_of(cf * qs[bh][d + 3], bpart)));
      }
    }
  __syncthreads();

  const int h0 = 2 * (tig & 1);                     // heads of this lane's score / accumulator columns: h0, h0 + 1
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f}, l_b[2] = {0.f, 0.f}, qoff[2] = {0.f, 0.f};
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
  const bool blockwise = p.kmu != nullptr;
  int cur_b = -1;

  const int per_warp = (p.tpb + WARPS - 1) / WARPS;
  const int b0 = blockIdx.x * p.tpb + w * per_warp;
  const int n_tok = p.n_dev ? *p.n_dev : p.n;
  const int b1 = min(min(b0 + per_warp, (int)(blockIdx.x + 1) * p.tpb), n_tok);
  auto tok_of = [&](int it) -> int {
    return p.pages ? p.pages[h * p.pages_stride + (it >> 7)] * 128 + (it & 127) : p.t0 + it;
  };
  auto flush_vmu = [&](int bb) {                    // block mean of the values: (sum of p in the block) * vmu, hi columns
    if (tig < 2) {
      const __half* vm = p.vmu + ((size_t)h * (p.Tcap / p.mu_block) + bb) * D + 4 * VS * g8;
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        const float2 m2 = __half22float2(*reinterpret_cast<const __half2*>(vm + 2 * mt));
        acc[mt][0] += l_b[0] * m2.x; acc[mt][1] += l_b[1] * m2.x;
        acc[mt][2] += l_b[0] * m2.y; acc[mt][3] += l_b[1] * m2.y;
      }
    }
  };

  // copy tile it0 of this warp into stage st (async; all 16 rows are inside the allocation: capacities are 16-aligned)
  auto issue = [&](int st, int it0) {
    Stage& S = stg_w[st];
    const size_t r0 = (size_t)h * p.Tcap + tok_of(it0);
    const uint8_t* ks = p.khi + r0 * RB;
    const uint8_t* vs = p.vhi + r0 * RB;
#pragma unroll
    for (int c = lane; c < RB; c += 32) {           // RB 16-byte chunks per plane tile (16 rows x RB bytes)
      cp16(&S.k[c / (RB / 16)][(c % (RB / 16)) * 16], ks + c * 16);
      cp16(&S.v[c / (RB / 16)][(c % (RB / 16)) * 16], vs + c * 16);
    }
    if (lane == 0) cp16(S.kg, p.kg8 + r0);
    if (lane == 1) cp16(S.vg, p.vg8 + r0);
  };
  int it_next = b0;
#pragma unroll
  for (int st = 0; st < STAGES - 1; st++) { if (it_next < b1) issue(st, it_next); cp_commit(); it_next += 16; }
  int st = 0;

  for (int it0 = b0; it0 < b1; it0 += 16) {
    if (it_next < b1) issue((st + STAGES - 1) % STAGES, it_next);
    cp_commit(); it_next += 16;
    cp_wait<STAGES - 1>();
    __syncwarp();
    const Stage& S = stg_w[st];
    if (blockwise) {
      const int bb = tok_of(it0) / p.mu_block;
      if (bb != cur_b) {
        if (cur_b >= 0) flush_vmu(cur_b);
        l_b[0] = l_b[1] = 0.f;
        cur_b = bb;
        const __half* km = p.kmu + ((size_t)h * (p.Tcap / p.mu_block) + bb) * D + lane * (D / 32);
        float part[G] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < D / 32; i++) {
          const float x = __half2float(km[i]);
#pragma unroll
          for (int hd = 0; hd < G; hd++) part[hd] += qs[hd][lane * (D / 32) + i] * x;
        }
#pragma unroll
        for (int hd = 0; hd < G; hd++) part[hd] = warp_sum(part[hd]);
        if (p.qoff_out && lane < G) p.qoff_out[((size_t)h * (p.Tcap / p.mu_block) + bb) * G + lane] = part[lane];
        qoff[0] = (h0 == 0) ? part[0] : part[2];
        qoff[1] = (h0 == 0) ? part[1] : part[3];
      }
    }
    // ---------------- keys: rows g8, g8 + 8 of the tile
    const int ia = it0 + g8, ib = ia + 8;
    const int ta = ia < b1 ? tok_of(ia) : -1, tb = ib < b1 ? tok_of(ib) : -1;
    unsigned kwa[KS / 4], kwb[KS / 4];
    load_words<KS / 4>(&S.k[g8][KS * tig], kwa);
    load_words<KS / 4>(&S.k[g8 + 8][KS * tig], kwb);
    float cc[4][4] = {};                            // 4 independent accumulators: short mma dependency chains
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
      const unsigned xa = (kwa[ks >> 2] >> (8 * (ks & 3))) & 0xFF, xb = (kwb[ks >> 2] >> (8 * (ks & 3))) & 0xFF;
      const uint2 a0 = dec[xa & 15], a1 = dec[xb & 15], a2 = dec[xa >> 4], a3 = dec[xb >> 4];
      const uint2 f0 = qf[0][ks][lane], f1 = qf[1][ks][lane];
      mma(cc[(2 * ks) & 3], a0, a1, a2, a3, false, f0.x, f0.y);
      mma(cc[(2 * ks + 1) & 3], a0, a1, a2, a3, true, f1.x, f1.y);
    }
    float c[4];
#pragma unroll
    for (int i = 0; i < 4; i++) c[i] = (cc[0][i] + cc[1][i]) + (cc[2][i] + cc[3][i]);
#pragma unroll
    for (int i = 0; i < 4; i++) c[i] += __shfl_xor_sync(0xffffffff, c[i], 2);    // hi + lo columns
    const float kga = kgl[S.kg[g8]], kgb = kgl[S.kg[g8 + 8]];
    float s[2][2];                                  // [row a / b][head h0 / h0 + 1]
    s[0][0] = ta >= 0 ? c[0] * kga + qoff[0] : -INFINITY;
    s[0][1] = ta >= 0 ? c[1] * kga + qoff[1] : -INFINITY;
    s[1][0] = tb >= 0 ? c[2] * kgb + qoff[0] : -INFINITY;
    s[1][1] = tb >= 0 ? c[3] * kgb + qoff[1] : -INFINITY;
    if (p.logits && tig < 2) {
      if (ta >= 0) { p.logits[((size_t)h * G + h0) * p.Tcap + ta] = s[0][0]; p.logits[((size_t)h * G + h0 + 1) * p.Tcap + ta] = s[0][1]; }
      if (tb >= 0) { p.logits[((size_t)h * G + h0) * p.Tcap + tb] = s[1][0]; p.logits[((size_t)h * G + h0 + 1) * p.Tcap + tb] = s[1][1]; }
    }
    // ---------------- online softmax, once per tile
    float sc[2], pr[2][2];
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const float mt_ = group_max(fmaxf(s[0][hh], s[1][hh]));
      const float mn = fmaxf(m_run[hh], mt_);
      sc[hh] = (mn == -INFINITY) ? 1.f : __expf(m_run[hh] - mn);
      pr[0][hh] = (s[0][hh] == -INFINITY) ? 0.f : __expf(s[0][hh] - mn);
      pr[1][hh] = (s[1][hh] == -INFINITY) ? 0.f : __expf(s[1][hh] - mn);
      const float ps = group_sum(pr[0][hh] + pr[1][hh]);
      l_run[hh] = l_run[hh] * sc[hh] + ps;
      l_b[hh] = l_b[hh] * sc[hh] + ps;
      m_run[hh] = mn;
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++) { acc[mt][0] *= sc[0]; acc[mt][1] *= sc[1]; acc[mt][2] *= sc[0]; acc[mt][3] *= sc[1]; }
    if (tig < 2) {
      const float vga = ta >= 0 ? vgl[S.vg[g8]] : 0.f, vgb = tb >= 0 ? vgl[S.vg[g8 + 8]] : 0.f;
      ws[w][g8][h0] = pr[0][0] * vga; ws[w][g8][h0 + 1] = pr[0][1] * vga;
      ws[w][g8 + 8][h0] = pr[1][0] * vgb; ws[w][g8 + 8][h0 + 1] = pr[1][1] * vgb;
    }
    __syncwarp();
    // ---------------- values: B = weights of tokens {2tig, 2tig+1, 2tig+8, 2tig+9} for column (bh, bpart)
    unsigned vb[2][2];
    {
      const float w0 = ws[w][2 * tig][bh], w1 = ws[w][2 * tig + 1][bh], w2 = ws[w][2 * tig + 8][bh], w3 = ws[w][2 * tig + 9][bh];
#pragma unroll
      for (int m = 0; m < 2; m++) {
        const float cf = m ? vd : va;
        vb[m][0] = pack2(part_of(cf * w0, bpart), part_of(cf * w1, bpart));
        vb[m][1] = pack2(part_of(cf * w2, bpart), part_of(cf * w3, bpart));
      }
    }
    unsigned vw[4][VS / 4 > 0 ? VS / 4 : 1];
#pragma unroll
    for (int x = 0; x < 4; x++) {
      load_words<VS / 4>(&S.v[2 * tig + (x & 1) + 8 * (x >> 1)][VS * g8], vw[x]);
    }
#pragma unroll
    for (int wi = 0; wi < VS / 4; wi++) {
      // token pairs (0, 1) and (2, 3): even codes (dims +0, +2 of each byte) and odd codes (+1, +3), 2-bit interleaved
      const unsigned e01 = (vw[0][wi] & 0x33333333u) | ((vw[1][wi] & 0x33333333u) << 2);
      const unsigned o01 = ((vw[0][wi] >> 2) & 0x33333333u) | (vw[1][wi] & 0xCCCCCCCCu);
      const unsigned e23 = (vw[2][wi] & 0x33333333u) | ((vw[3][wi] & 0x33333333u) << 2);
      const unsigned o23 = ((vw[2][wi] >> 2) & 0x33333333u) | (vw[3][wi] & 0xCCCCCCCCu);
#pragma unroll
      for (int nb = 0; nb < 8; nb++) {              // nibble nb: byte wi * 4 + nb / 2, m-tile 2 * byte + (nb & 1)
        const int mt = 2 * (wi * 4 + (nb >> 1)) + (nb & 1);
        const uint2 a0 = dec[(e01 >> (4 * nb)) & 15], a1 = dec[(o01 >> (4 * nb)) & 15];
        const uint2 a2 = dec[(e23 >> (4 * nb)) & 15], a3 = dec[(o23 >> (4 * nb)) & 15];
        mma(acc[mt], a0, a1, a2, a3, false, vb[0][0], vb[0][1]);
        mma(acc[mt], a0, a1, a2, a3, true, vb[1][0], vb[1][1]);
      }
    }
    __syncwarp();
    st = (st + 1) % STAGES;
  }
  cp_wait<0>();
  if (blockwise && cur_b >= 0) flush_vmu(cur_b);
  // ---------------- block reduction over warps
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[mt][i] += __shfl_xor_sync(0xffffffff, acc[mt][i], 2);
  __syncthreads();                                  // sacc aliases the stages of every warp
  if (tig < 2) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      const int d = 4 * VS * g8 + 2 * mt;
      sacc[w][h0][d] = acc[mt][0]; sacc[w][h0 + 1][d] = acc[mt][1];
      sacc[w][h0][d + 1] = acc[mt][2]; sacc[w][h0 + 1][d + 1] = acc[mt][3];
    }
  }
  if (lane < 2) { sm_[w][h0] = m_run[0]; sm_[w][h0 + 1] = m_run[1]; sl_[w][h0] = l_run[0]; sl_[w][h0 + 1] = l_run[1]; }
  __syncthreads();
  const size_t ob = (size_t)blockIdx.x * p.Hkv + h;
  for (int e = threadIdx.x; e < G * D; e += blockDim.x) {
    const int g = e / D, dd = e % D;
    float M = -INFINITY;
    for (int ww = 0; ww < WARPS; ww++) M = fmaxf(M, sm_[ww][g]);
    float L = 0.f, A = 0.f;
    if (M > -INFINITY)
      for (int ww = 0; ww < WARPS; ww++) {
        if (sm_[ww][g] == -INFINITY) continue;
        const float f = __expf(sm_[ww][g] - M); L += sl_[ww][g] * f; A += sacc[ww][g][dd] * f;
      }
    p.pacc[(ob * G + g) * D + dd] = A;
    if (dd == 0) { p.pm[ob * G + g] = M; p.pl[ob * G + g] = L; }
  }
}

static const void* ptr(const c10::optional<torch::Tensor>& t) { return (t.has_value() && t->defined()) ? t->data_ptr() : nullptr; }

// partials into (pm, pl, pacc) from block `boff`, like foveakv ext().pass_(1, ...) with kfine = vfine = 0
void coarse(torch::Tensor q, int64_t Hkv, int64_t Tcap, int64_t t0, int64_t n, int64_t tpb, int64_t boff,
            torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc, torch::Tensor khi, torch::Tensor vhi,
            torch::Tensor kg8, torch::Tensor vg8, torch::Tensor kglut, torch::Tensor vglut,
            c10::optional<torch::Tensor> kmu, c10::optional<torch::Tensor> vmu, int64_t mu_block,
            torch::Tensor klev, torch::Tensor vlev, c10::optional<torch::Tensor> logits,
            c10::optional<torch::Tensor> pages, c10::optional<torch::Tensor> n_dev, c10::optional<torch::Tensor> qoff_out) {
  TORCH_CHECK(tpb % 32 == 0 && t0 % 16 == 0 && Tcap % 16 == 0 && (!kmu.has_value() || mu_block % 16 == 0), "tile alignment");
  TcParams p{};
  p.Hkv = Hkv; p.Tcap = Tcap; p.t0 = t0; p.n = n; p.tpb = tpb;
  p.n_dev = (const int*)ptr(n_dev); p.pages = (const int*)ptr(pages); p.pages_stride = pages.has_value() ? pages->size(1) : 0;
  p.q = q.data_ptr<float>();
  p.khi = khi.data_ptr<uint8_t>(); p.vhi = vhi.data_ptr<uint8_t>(); p.kg8 = kg8.data_ptr<uint8_t>(); p.vg8 = vg8.data_ptr<uint8_t>();
  p.kglut = kglut.data_ptr<float>(); p.vglut = vglut.data_ptr<float>();
  p.kmu = (const __half*)ptr(kmu); p.vmu = (const __half*)ptr(vmu); p.mu_block = mu_block;
  p.klev = klev.data_ptr<float>(); p.vlev = vlev.data_ptr<float>();
  const int HG = Hkv * G;
  p.pm = pm.data_ptr<float>() + boff * HG; p.pl = pl.data_ptr<float>() + boff * HG; p.pacc = pacc.data_ptr<float>() + boff * HG * D;
  p.logits = (float*)ptr(logits);
  p.qoff_out = (float*)ptr(qoff_out);
  const int nb = (n + tpb - 1) / tpb;
  if (nb == 0) return;
  coarse_tc<<<dim3(nb, Hkv), 32 * WARPS, 0, c10::cuda::getCurrentCUDAStream()>>>(p);
}

// ------------------------------------------------------------------------------------------- delta pass
// FoveaKV's dyn delta (kernels.cu delta_pass) for the hot tokens idx[h, :] (-1 = skip): replace each token's coarse
// contribution by the fine (4-bit) one, relative to the coarse max mref:
//   dl += ef - ec,  dacc += ef * vf - ec * vc,  v = gv * lev + vmu,  e = exp(gk * (q . lev) + q . kmu - mref)
// Same arithmetic; the coarse logit of each hot token is read from the coarse pass's logits (the exact value that pass
// added), q . kmu of the fine logit comes from the per-block table the coarse pass wrote (qoff), the gain tables sit
// in shared memory, and the block mean of the values is read once per token (the generic kernel reads both means
// twice and recomputes the coarse dot product).
// One warp per token, lane = LD consecutive dims; the next token's codes are loaded while the current one computes.
constexpr int LD = D / 32;
struct DParams {
  int Hkv, Tcap, n, idx_stride, tpb, mu_block;
  const int* idx; const float* q; const float* mref; const float* qoff; const float* logits;
  const uint8_t *khi, *klo, *vhi, *vlo, *kg8, *vg8; const float *kglut, *vglut;
  const __half* vmu; const float *k16, *v4, *v16;
  float *pl, *pacc;
};
struct TokData { unsigned kh, kl, vh, vl; float gk, gv; float qo[G], sc[G]; uint4 mu; };

__device__ __forceinline__ unsigned codes_lane(const uint8_t* row, int lane) {      // the 2*LD code bits of a lane
  if constexpr (LD == 8) return reinterpret_cast<const uint16_t*>(row)[lane];
  else return row[lane];
}

__global__ void __launch_bounds__(32 * WARPS) delta_gs(DParams p) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __shared__ float kgl[256], vgl[256], lk16[16], lv4[4], lv16[16];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) { kgl[i] = p.kglut[h * 256 + i]; vgl[i] = p.vglut[h * 256 + i]; }
  if (threadIdx.x < 16) { lk16[threadIdx.x] = p.k16[threadIdx.x]; lv16[threadIdx.x] = p.v16[threadIdx.x]; }
  if (threadIdx.x < 4) lv4[threadIdx.x] = p.v4[threadIdx.x];
  __syncthreads();
  float q[G][LD], mr[G], dl[G], dacc[G][LD];
#pragma unroll
  for (int g = 0; g < G; g++) {
    mr[g] = p.mref[h * G + g]; dl[g] = 0.f;
#pragma unroll
    for (int i = 0; i < LD; i++) { q[g][i] = p.q[(h * G + g) * D + lane * LD + i]; dacc[g][i] = 0.f; }
  }
  const int nmu = p.Tcap / p.mu_block;
  const int per_warp = (p.tpb + WARPS - 1) / WARPS;
  const int b0 = blockIdx.x * p.tpb + w * per_warp;
  const int b1 = min(min(b0 + per_warp, (int)(blockIdx.x + 1) * p.tpb), p.n);
  auto fetch = [&](int t, TokData& d) {
    const size_t row = (size_t)h * p.Tcap + t;
    d.kh = codes_lane(p.khi + row * RB, lane); d.kl = codes_lane(p.klo + row * RB, lane);
    d.vh = codes_lane(p.vhi + row * RB, lane); d.vl = codes_lane(p.vlo + row * RB, lane);
    d.gk = kgl[p.kg8[row]]; d.gv = vgl[p.vg8[row]];
    const size_t mb = (size_t)h * nmu + t / p.mu_block;
#pragma unroll
    for (int g = 0; g < G; g++) { d.qo[g] = p.qoff[mb * G + g]; d.sc[g] = p.logits[((size_t)h * G + g) * p.Tcap + t]; }
    if constexpr (LD == 8) d.mu = *reinterpret_cast<const uint4*>(p.vmu + mb * D + lane * LD);
    else { const uint2 m2 = *reinterpret_cast<const uint2*>(p.vmu + mb * D + lane * LD); d.mu = make_uint4(m2.x, m2.y, 0, 0); }
  };
  int it = b0, t = -1;
  while (it < b1 && (t = p.idx[h * p.idx_stride + it]) < 0) it++;
  TokData cur, nxt;
  if (it < b1) fetch(t, cur);
  while (it < b1) {
    int it2 = it + 1, t2 = -1;
    while (it2 < b1 && (t2 = p.idx[h * p.idx_stride + it2]) < 0) it2++;
    if (it2 < b1) fetch(t2, nxt);
    float b[G] = {0.f, 0.f, 0.f, 0.f}, xc[LD], xf[LD];
    const __half* mh = reinterpret_cast<const __half*>(&cur.mu);
#pragma unroll
    for (int i = 0; i < LD; i++) {
      const int ch = (cur.kh >> (2 * i)) & 3, cl = (cur.kl >> (2 * i)) & 3;
      const float kf = lk16[(ch << 2) | cl];
#pragma unroll
      for (int g = 0; g < G; g++) b[g] += q[g][i] * kf;
      const int vh = (cur.vh >> (2 * i)) & 3, vl = (cur.vl >> (2 * i)) & 3;
      const float mu = __half2float(mh[i]);
      xc[i] = cur.gv * lv4[vh] + mu; xf[i] = cur.gv * lv16[(vh << 2) | vl] + mu;     // coarse / fine value
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
      b[g] = warp_sum(b[g]);
      const float ec = __expf(cur.sc[g] - mr[g]), ef = __expf(cur.gk * b[g] + cur.qo[g] - mr[g]);
      dl[g] += ef - ec;
#pragma unroll
      for (int i = 0; i < LD; i++) dacc[g][i] += ef * xf[i] - ec * xc[i];
    }
    it = it2; cur = nxt;
  }
  __shared__ float sl[WARPS][G];
  __shared__ float sacc[WARPS][G][D];
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < G; g++) sl[w][g] = dl[g];
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int i = 0; i < LD; i++) sacc[w][g][lane * LD + i] = dacc[g][i];
  __syncthreads();
  const size_t ob = (size_t)blockIdx.x * p.Hkv + h;
  for (int e = threadIdx.x; e < G * D; e += blockDim.x) {
    const int g = e / D, dd = e % D;
    float L = 0.f, A = 0.f;
    for (int ww = 0; ww < WARPS; ww++) { L += sl[ww][g]; A += sacc[ww][g][dd]; }
    p.pacc[(ob * G + g) * D + dd] = A;
    if (dd == 0) p.pl[ob * G + g] = L;
  }
}

// same outputs as foveakv ext().delta (partials (pl, pacc) per block of `tpb` hot-list entries); needs the qoff table
// and the logits of the coarse pass that produced mref
void delta(torch::Tensor q, int64_t Hkv, int64_t Tcap, torch::Tensor idx, int64_t tpb, torch::Tensor mref,
           torch::Tensor qoff, torch::Tensor logits, torch::Tensor pl, torch::Tensor pacc, torch::Tensor khi,
           torch::Tensor klo, torch::Tensor vhi, torch::Tensor vlo, torch::Tensor kg8, torch::Tensor vg8,
           torch::Tensor kglut, torch::Tensor vglut, torch::Tensor vmu, int64_t mu_block, torch::Tensor k16,
           torch::Tensor v4, torch::Tensor v16) {
  DParams p{};
  p.Hkv = Hkv; p.Tcap = Tcap; p.n = idx.size(1); p.idx_stride = idx.size(1); p.tpb = tpb; p.mu_block = mu_block;
  p.idx = idx.data_ptr<int>(); p.q = q.data_ptr<float>(); p.mref = mref.data_ptr<float>(); p.qoff = qoff.data_ptr<float>();
  p.logits = logits.data_ptr<float>();
  p.khi = khi.data_ptr<uint8_t>(); p.klo = klo.data_ptr<uint8_t>(); p.vhi = vhi.data_ptr<uint8_t>(); p.vlo = vlo.data_ptr<uint8_t>();
  p.kg8 = kg8.data_ptr<uint8_t>(); p.vg8 = vg8.data_ptr<uint8_t>(); p.kglut = kglut.data_ptr<float>(); p.vglut = vglut.data_ptr<float>();
  p.vmu = (const __half*)vmu.data_ptr(); p.k16 = k16.data_ptr<float>(); p.v4 = v4.data_ptr<float>(); p.v16 = v16.data_ptr<float>();
  p.pl = pl.data_ptr<float>(); p.pacc = pacc.data_ptr<float>();
  const int nb = (p.n + tpb - 1) / tpb;
  if (nb == 0) return;
  delta_gs<<<dim3(nb, Hkv), 32 * WARPS, 0, c10::cuda::getCurrentCUDAStream()>>>(p);
}

// ------------------------------------------------------------------------------------------- delta pass, tensor cores
// The same delta on the tensor cores, 16 hot tokens per tile (gathered with cp.async):
//   keys:   fine logit = gk * (A16 . q) + q . kmu, A16 = lev16 codes as fp16 hi + lo matrices (a 256-entry table maps the
//           2 + 2 code bits of 2 dims to both), q split hi / lo in the B columns (as in coarse_tc);
//           ef = exp(fine - mref), ec = exp(coarse logit - mref) (the coarse pass's logit of that token)
//   values: dacc += (ef gv) . A16v - (ec gv) . (a A1 + (b - a) A2)   (A1 / A2: the coarse sign matrices)
//   means:  wsum[h, b, g] += ef - ec per mean block b (atomics); delta_means adds sum_b wsum * vmu_b and re-zeroes it.
// No running max: every exponent is relative to the fixed coarse max mref (as FoveaKV's delta).
constexpr int DSTG = 2;
struct DStage {
  uint8_t kh[16][RB], kl[16][RB], vh[16][VSTR], vl[16][VSTR];
  unsigned kgw[16], vgw[16];                       // the aligned 4-byte words holding each token's gain code
  float sc[16][G], qo[16][G];
  int t[16];
};
constexpr int DSTAGE_BYTES = WARPS * DSTG * (int)sizeof(DStage);
constexpr int DSMEM = DSTAGE_BYTES > SACC_BYTES ? DSTAGE_BYTES : SACC_BYTES;

struct DTParams {
  int Hkv, Tcap, n, idx_stride, tpb, mu_block;
  const int* idx; const float* q; const float* mref; const float* qoff; const float* logits;
  const uint8_t *khi, *klo, *vhi, *vlo, *kg8, *vg8; const float *kglut, *vglut;
  const float *k16, *v4, *v16;
  float *pl, *pacc, *wsum;
};

__device__ __forceinline__ void cp4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

__global__ void __launch_bounds__(32 * WARPS, 2) delta_tc(DTParams p) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5, g8 = lane >> 2, tig = lane & 3;
  __shared__ uint2 dec[16];                         // 4-level sign matrices (as coarse_tc)
  __shared__ uint2 t16k[256], t16v[256];            // (hi codes 2 dims | lo codes 2 dims << 4) -> {hi half2, lo half2}
  __shared__ float kgl[256], vgl[256];
  __shared__ uint2 qf[KS][32];                      // key B fragments (q hi / lo columns)
  __shared__ float wsf[WARPS][16][G], wsc[WARPS][16][G];
  extern __shared__ __align__(16) unsigned char big[];     // DSMEM bytes: the stages, then the per-warp partials
  DStage* stg_w = reinterpret_cast<DStage*>(big) + w * DSTG;
  float (*sacc)[G][D] = reinterpret_cast<float(*)[G][D]>(big);
  __shared__ float sl_[WARPS][G];
  if (threadIdx.x < 16) {
    const int c0 = threadIdx.x & 3, c1 = threadIdx.x >> 2;
    const float s0 = c0 >= 2 ? 1.f : -1.f, s1 = c1 >= 2 ? 1.f : -1.f;
    const bool g0 = (c0 == 0 || c0 == 3), g1 = (c1 == 0 || c1 == 3);
    dec[threadIdx.x] = make_uint2(pack2(__float2half(s0), __float2half(s1)),
                                  pack2(__float2half(g0 ? s0 : 0.f), __float2half(g1 ? s1 : 0.f)));
  }
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    const int c0 = ((e & 3) << 2) | ((e >> 4) & 3), c1 = (((e >> 2) & 3) << 2) | ((e >> 6) & 3);   // hi << 2 | lo
    const float k0 = p.k16[c0], k1 = p.k16[c1], v0 = p.v16[c0], v1 = p.v16[c1];
    t16k[e] = make_uint2(pack2(part_of(k0, 0), part_of(k1, 0)), pack2(part_of(k0, 1), part_of(k1, 1)));
    t16v[e] = make_uint2(pack2(part_of(v0, 0), part_of(v1, 0)), pack2(part_of(v0, 1), part_of(v1, 1)));
    kgl[e] = p.kglut[h * 256 + e]; vgl[e] = p.vglut[h * 256 + e];
  }
  const int bh = g8 & 3, bpart = g8 >> 2;
  if (w == 0)
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
      const float* qh = p.q + (size_t)(h * G + bh) * D + 4 * KS * tig + 4 * ks;
      qf[ks][lane] = make_uint2(pack2(part_of(qh[0], bpart), part_of(qh[1], bpart)),
                                pack2(part_of(qh[2], bpart), part_of(qh[3], bpart)));
    }
  __syncthreads();
  const float va = p.v4[2], vd = p.v4[3] - p.v4[2];
  const int h0 = 2 * (tig & 1);
  const float mr0 = p.mref[h * G + h0], mr1 = p.mref[h * G + h0 + 1];
  const int nmu = p.Tcap / p.mu_block;
  float dl[2] = {0.f, 0.f}, acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;

  const int per_warp = (p.tpb + WARPS - 1) / WARPS;
  const int b0 = blockIdx.x * p.tpb + w * per_warp;
  const int b1 = min(min(b0 + per_warp, (int)(blockIdx.x + 1) * p.tpb), p.n);
  const int ntile = b1 > b0 ? (b1 - b0 + 15) / 16 : 0;
  int my_t = (lane < 2 * 16 && b0 + lane < b1) ? p.idx[h * p.idx_stride + b0 + lane] : -1;   // tiles 0, 1
  auto issue = [&](int st, int tile) {             // tile's tokens from my_t (lanes 16 * (tile & 1) ...)
    DStage& S = stg_w[st];
    const int base = 16 * (tile & 1);
    constexpr int CH = RB / 16;
#pragma unroll
    for (int c = lane; c < 16 * 4 * CH; c += 32) {
      const int tok = c / (4 * CH), pl = (c / CH) & 3, ch = c % CH;
      int t = __shfl_sync(0xffffffff, my_t, base + tok);
      t = t < 0 ? 0 : t;
      const size_t off = ((size_t)h * p.Tcap + t) * RB + 16 * ch;
      const uint8_t* src = pl == 0 ? p.khi : pl == 1 ? p.klo : pl == 2 ? p.vhi : p.vlo;
      uint8_t* dst = pl == 0 ? &S.kh[tok][16 * ch] : pl == 1 ? &S.kl[tok][16 * ch] : pl == 2 ? &S.vh[tok][16 * ch] : &S.vl[tok][16 * ch];
      cp16(dst, src + off);
    }
    {
      const int tok = lane & 15;
      const int t0 = __shfl_sync(0xffffffff, my_t, base + tok);
      const int t = t0 < 0 ? 0 : t0;
      const size_t row = (size_t)h * p.Tcap + t;
      if (lane < 16) {
        cp4(&S.kgw[tok], p.kg8 + (row & ~(size_t)3)); cp4(&S.vgw[tok], p.vg8 + (row & ~(size_t)3));
        cp16(&S.qo[tok][0], p.qoff + ((size_t)h * nmu + t / p.mu_block) * G);
        S.t[tok] = (b0 + 16 * tile + tok < b1) ? t0 : -1;
      } else {
#pragma unroll
        for (int g = 0; g < G; g++) cp4(&S.sc[tok][g], p.logits + ((size_t)h * G + g) * p.Tcap + t);
      }
    }
  };
  if (ntile > 0) issue(0, 0);
  cp_commit();
  for (int tile = 0; tile < ntile; tile++) {
    const int st = tile & 1;
    if (tile + 1 < ntile) {
      if (((tile + 1) & 1) == 0)                   // next pair of tiles: reload the hot-list entries
        my_t = (b0 + 16 * (tile + 1) + lane < b1) ? p.idx[h * p.idx_stride + b0 + 16 * (tile + 1) + lane] : -1;
      issue(st ^ 1, tile + 1);
    }
    cp_commit();
    cp_wait<1>();
    __syncwarp();
    const DStage& S = stg_w[st];
    // ---------------- fine key logits: rows g8, g8 + 8
    unsigned kha[KS / 4], khb[KS / 4], kla[KS / 4], klb[KS / 4];
    load_words<KS / 4>(&S.kh[g8][KS * tig], kha); load_words<KS / 4>(&S.kh[g8 + 8][KS * tig], khb);
    load_words<KS / 4>(&S.kl[g8][KS * tig], kla); load_words<KS / 4>(&S.kl[g8 + 8][KS * tig], klb);
    float cc[2][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
      const int sh = 8 * (ks & 3);
      const unsigned ha = (kha[ks >> 2] >> sh) & 0xFF, hb = (khb[ks >> 2] >> sh) & 0xFF;
      const unsigned la = (kla[ks >> 2] >> sh) & 0xFF, lb = (klb[ks >> 2] >> sh) & 0xFF;
      const uint2 a0 = t16k[(ha & 15) | ((la & 15) << 4)], a1 = t16k[(hb & 15) | ((lb & 15) << 4)];
      const uint2 a2 = t16k[(ha >> 4) | (la & 0xF0)], a3 = t16k[(hb >> 4) | (lb & 0xF0)];
      const uint2 f = qf[ks][lane];
      mma(cc[0], a0, a1, a2, a3, false, f.x, f.y);
      mma(cc[1], a0, a1, a2, a3, true, f.x, f.y);
    }
    float c[4];
#pragma unroll
    for (int i = 0; i < 4; i++) { c[i] = cc[0][i] + cc[1][i]; c[i] += __shfl_xor_sync(0xffffffff, c[i], 2); }
    const int ta = S.t[g8], tb = S.t[g8 + 8];
    const float gka = kgl[(S.kgw[g8] >> (8 * (ta & 3))) & 0xFF], gkb = kgl[(S.kgw[g8 + 8] >> (8 * (tb & 3))) & 0xFF];
    const float gva = vgl[(S.vgw[g8] >> (8 * (ta & 3))) & 0xFF], gvb = vgl[(S.vgw[g8 + 8] >> (8 * (tb & 3))) & 0xFF];
    float ef[2][2], ec[2][2];
    ef[0][0] = ta >= 0 ? __expf(c[0] * gka + S.qo[g8][h0] - mr0) : 0.f;
    ef[0][1] = ta >= 0 ? __expf(c[1] * gka + S.qo[g8][h0 + 1] - mr1) : 0.f;
    ef[1][0] = tb >= 0 ? __expf(c[2] * gkb + S.qo[g8 + 8][h0] - mr0) : 0.f;
    ef[1][1] = tb >= 0 ? __expf(c[3] * gkb + S.qo[g8 + 8][h0 + 1] - mr1) : 0.f;
    ec[0][0] = ta >= 0 ? __expf(S.sc[g8][h0] - mr0) : 0.f;
    ec[0][1] = ta >= 0 ? __expf(S.sc[g8][h0 + 1] - mr1) : 0.f;
    ec[1][0] = tb >= 0 ? __expf(S.sc[g8 + 8][h0] - mr0) : 0.f;
    ec[1][1] = tb >= 0 ? __expf(S.sc[g8 + 8][h0 + 1] - mr1) : 0.f;
    if (tig < 2) {
      dl[0] += (ef[0][0] - ec[0][0]) + (ef[1][0] - ec[1][0]);
      dl[1] += (ef[0][1] - ec[0][1]) + (ef[1][1] - ec[1][1]);
      wsf[w][g8][h0] = ef[0][0] * gva; wsf[w][g8][h0 + 1] = ef[0][1] * gva;
      wsf[w][g8 + 8][h0] = ef[1][0] * gvb; wsf[w][g8 + 8][h0 + 1] = ef[1][1] * gvb;
      wsc[w][g8][h0] = ec[0][0] * gva; wsc[w][g8][h0 + 1] = ec[0][1] * gva;
      wsc[w][g8 + 8][h0] = ec[1][0] * gvb; wsc[w][g8 + 8][h0 + 1] = ec[1][1] * gvb;
      float* wa = p.wsum + ((size_t)h * nmu + (ta < 0 ? 0 : ta) / p.mu_block) * G + h0;
      float* wb = p.wsum + ((size_t)h * nmu + (tb < 0 ? 0 : tb) / p.mu_block) * G + h0;
      if (ta >= 0) { atomicAdd(wa, ef[0][0] - ec[0][0]); atomicAdd(wa + 1, ef[0][1] - ec[0][1]); }
      if (tb >= 0) { atomicAdd(wb, ef[1][0] - ec[1][0]); atomicAdd(wb + 1, ef[1][1] - ec[1][1]); }
    }
    __syncwarp();
    // ---------------- values: B columns (bh, bpart) for tokens {2tig, 2tig+1, 2tig+8, 2tig+9}
    unsigned bf[2], bc1[2], bc2[2];
    {
      const int k0 = 2 * tig;
      const float f0 = wsf[w][k0][bh], f1 = wsf[w][k0 + 1][bh], f2 = wsf[w][k0 + 8][bh], f3 = wsf[w][k0 + 9][bh];
      const float e0 = wsc[w][k0][bh], e1 = wsc[w][k0 + 1][bh], e2 = wsc[w][k0 + 8][bh], e3 = wsc[w][k0 + 9][bh];
      bf[0] = pack2(part_of(f0, bpart), part_of(f1, bpart)); bf[1] = pack2(part_of(f2, bpart), part_of(f3, bpart));
      bc1[0] = pack2(part_of(-va * e0, bpart), part_of(-va * e1, bpart)); bc1[1] = pack2(part_of(-va * e2, bpart), part_of(-va * e3, bpart));
      bc2[0] = pack2(part_of(-vd * e0, bpart), part_of(-vd * e1, bpart)); bc2[1] = pack2(part_of(-vd * e2, bpart), part_of(-vd * e3, bpart));
    }
    unsigned vhw[4][VS / 4 > 0 ? VS / 4 : 1], vlw[4][VS / 4 > 0 ? VS / 4 : 1];
#pragma unroll
    for (int x = 0; x < 4; x++) {
      const int r = 2 * tig + (x & 1) + 8 * (x >> 1);
      load_words<VS / 4>(&S.vh[r][VS * g8], vhw[x]); load_words<VS / 4>(&S.vl[r][VS * g8], vlw[x]);
    }
#pragma unroll
    for (int wi = 0; wi < VS / 4; wi++) {
      const unsigned he01 = (vhw[0][wi] & 0x33333333u) | ((vhw[1][wi] & 0x33333333u) << 2);
      const unsigned ho01 = ((vhw[0][wi] >> 2) & 0x33333333u) | (vhw[1][wi] & 0xCCCCCCCCu);
      const unsigned he23 = (vhw[2][wi] & 0x33333333u) | ((vhw[3][wi] & 0x33333333u) << 2);
      const unsigned ho23 = ((vhw[2][wi] >> 2) & 0x33333333u) | (vhw[3][wi] & 0xCCCCCCCCu);
      const unsigned le01 = (vlw[0][wi] & 0x33333333u) | ((vlw[1][wi] & 0x33333333u) << 2);
      const unsigned lo01 = ((vlw[0][wi] >> 2) & 0x33333333u) | (vlw[1][wi] & 0xCCCCCCCCu);
      const unsigned le23 = (vlw[2][wi] & 0x33333333u) | ((vlw[3][wi] & 0x33333333u) << 2);
      const unsigned lo23 = ((vlw[2][wi] >> 2) & 0x33333333u) | (vlw[3][wi] & 0xCCCCCCCCu);
#pragma unroll
      for (int nb = 0; nb < 8; nb++) {
        const int mt = 2 * (wi * 4 + (nb >> 1)) + (nb & 1), sh = 4 * nb;
        const unsigned n0 = (he01 >> sh) & 15, n1 = (ho01 >> sh) & 15, n2 = (he23 >> sh) & 15, n3 = (ho23 >> sh) & 15;
        const uint2 f0 = t16v[n0 | (((le01 >> sh) & 15) << 4)], f1 = t16v[n1 | (((lo01 >> sh) & 15) << 4)];
        const uint2 f2 = t16v[n2 | (((le23 >> sh) & 15) << 4)], f3 = t16v[n3 | (((lo23 >> sh) & 15) << 4)];
        mma(acc[mt], f0, f1, f2, f3, false, bf[0], bf[1]);
        mma(acc[mt], f0, f1, f2, f3, true, bf[0], bf[1]);
        const uint2 d0 = dec[n0], d1 = dec[n1], d2 = dec[n2], d3 = dec[n3];
        mma(acc[mt], d0, d1, d2, d3, false, bc1[0], bc1[1]);
        mma(acc[mt], d0, d1, d2, d3, true, bc2[0], bc2[1]);
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[mt][i] += __shfl_xor_sync(0xffffffff, acc[mt][i], 2);
  // dl: sum over the row groups (lanes sharing tig)
#pragma unroll
  for (int i = 0; i < 2; i++) dl[i] = group_sum(dl[i]);
  __syncthreads();
  if (tig < 2) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      const int d = 4 * VS * g8 + 2 * mt;
      sacc[w][h0][d] = acc[mt][0]; sacc[w][h0 + 1][d] = acc[mt][1];
      sacc[w][h0][d + 1] = acc[mt][2]; sacc[w][h0 + 1][d + 1] = acc[mt][3];
    }
  }
  if (lane < 2) { sl_[w][h0] = dl[0]; sl_[w][h0 + 1] = dl[1]; }
  __syncthreads();
  const size_t ob = (size_t)blockIdx.x * p.Hkv + h;
  for (int e = threadIdx.x; e < G * D; e += blockDim.x) {
    const int g = e / D, dd = e % D;
    float L = 0.f, A = 0.f;
    for (int ww = 0; ww < WARPS; ww++) { L += sl_[ww][g]; A += sacc[ww][g][dd]; }
    p.pacc[(ob * G + g) * D + dd] = A;
    if (dd == 0) p.pl[ob * G + g] = L;
  }
}

// the delta's value block-mean term: out[h, g, :] += sum_b wsum[h, b, g] * vmu[h, b, :], then wsum = 0.
// grid (Hkv * G, NSPLIT), block D; each block reduces a slice of the mean blocks and adds atomically.
constexpr int NSPLIT = 8;
__global__ void delta_means_k(float* wsum, const __half* vmu, int nmu, int nmu_live, float* out) {
  const int hg = blockIdx.x, h = hg / G, g = hg % G, d = threadIdx.x;
  const int per = (nmu_live + NSPLIT - 1) / NSPLIT, bs = blockIdx.y * per, be = min(bs + per, nmu_live);
  float a = 0.f;
#pragma unroll 4
  for (int b = bs; b < be; b++)
    a += wsum[((size_t)h * nmu + b) * G + g] * __half2float(vmu[((size_t)h * nmu + b) * D + d]);
  atomicAdd(&out[hg * D + d], a);
  __syncthreads();
  for (int b = bs + d; b < be; b += D) wsum[((size_t)h * nmu + b) * G + g] = 0.f;
}

// partials (pl, pacc) like delta(); the value block-mean term goes to wsum (zero on entry) and is added to
// pacc[0] by delta_means (which re-zeroes wsum). tpb: hot-list entries per block (multiple of 64).
void delta_tc_launch(torch::Tensor q, int64_t Hkv, int64_t Tcap, torch::Tensor idx, int64_t tpb, torch::Tensor mref,
                     torch::Tensor qoff, torch::Tensor logits, torch::Tensor wsum, torch::Tensor pl, torch::Tensor pacc,
                     torch::Tensor khi, torch::Tensor klo, torch::Tensor vhi, torch::Tensor vlo, torch::Tensor kg8,
                     torch::Tensor vg8, torch::Tensor kglut, torch::Tensor vglut, torch::Tensor vmu, int64_t mu_block,
                     int64_t nmu_live, torch::Tensor k16, torch::Tensor v4, torch::Tensor v16) {
  TORCH_CHECK(tpb % 64 == 0 && tpb / WARPS <= 32 * 1000, "tpb");
  DTParams p{};
  p.Hkv = Hkv; p.Tcap = Tcap; p.n = idx.size(1); p.idx_stride = idx.size(1); p.tpb = tpb; p.mu_block = mu_block;
  p.idx = idx.data_ptr<int>(); p.q = q.data_ptr<float>(); p.mref = mref.data_ptr<float>(); p.qoff = qoff.data_ptr<float>();
  p.logits = logits.data_ptr<float>(); p.wsum = wsum.data_ptr<float>();
  p.khi = khi.data_ptr<uint8_t>(); p.klo = klo.data_ptr<uint8_t>(); p.vhi = vhi.data_ptr<uint8_t>(); p.vlo = vlo.data_ptr<uint8_t>();
  p.kg8 = kg8.data_ptr<uint8_t>(); p.vg8 = vg8.data_ptr<uint8_t>(); p.kglut = kglut.data_ptr<float>(); p.vglut = vglut.data_ptr<float>();
  p.k16 = k16.data_ptr<float>(); p.v4 = v4.data_ptr<float>(); p.v16 = v16.data_ptr<float>();
  p.pl = pl.data_ptr<float>(); p.pacc = pacc.data_ptr<float>();
  const int nb = (p.n + tpb - 1) / tpb;
  auto st = c10::cuda::getCurrentCUDAStream();
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(delta_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, DSMEM); attr = true; }
  if (nb > 0) delta_tc<<<dim3(nb, Hkv), 32 * WARPS, DSMEM, st>>>(p);
  delta_means_k<<<dim3(Hkv * G, NSPLIT), D, 0, st>>>(p.wsum, (const __half*)vmu.data_ptr(), Tcap / mu_block, (int)nmu_live,
                                                     p.pacc);
}

// ------------------------------------------------------------------------------------------- step glue
// prologue: append k, v (Hkv, D) fp32 at row pos - ncomp of the bf16 window (token-major, (Tw, Hkv, D)) and
// qk = (q * scale) @ Rt (the rotated query of the compressed passes). grid Hkv * G, block D.
__global__ void prologue_k(const float* q, const float* k, const float* v, const long* pos, const int* ncomp,
                           __nv_bfloat16* kraw, __nv_bfloat16* vraw, int Hkv, const float* Rt, float scale, float* qk) {
  const int hg = blockIdx.x, d = threadIdx.x;
  __shared__ float qs[D];
  qs[d] = q[hg * D + d] * scale;
  if (hg % G == 0) {
    const int h = hg / G;
    const size_t r = ((size_t)(*pos - *ncomp) * Hkv + h) * D + d;
    kraw[r] = __float2bfloat16(k[h * D + d]); vraw[r] = __float2bfloat16(v[h * D + d]);
  }
  __syncthreads();
  float a = 0.f;
  for (int j = 0; j < D; j++) a += qs[j] * Rt[j * D + d];
  qk[hg * D + d] = a;
}

// merge before selection: combine the coarse partials (nb1) and the window partials (nw) -> m1, l1, acc1 (rotated),
// mw, lw, aw, and the log-normalizer lse over both. grid Hkv * G, block D.
__global__ void merge1_k(const float* pm1, const float* pl1, const float* pa1, int nb1, const float* pm2,
                         const float* pl2, const float* pa2, int nw, int HG, float* m1, float* l1, float* acc1,
                         float* mw, float* lw, float* aw, float* lse) {
  const int hg = blockIdx.x, d = threadIdx.x;
  float M1 = -INFINITY, MW = -INFINITY;
  for (int b = 0; b < nb1; b++) M1 = fmaxf(M1, pm1[b * HG + hg]);
  for (int b = 0; b < nw; b++) MW = fmaxf(MW, pm2[b * HG + hg]);
  float L1 = 0.f, A1 = 0.f, LW = 0.f, AW = 0.f;
  for (int b = 0; b < nb1; b++) {
    const float m = pm1[b * HG + hg];
    if (m == -INFINITY) continue;
    const float f = __expf(m - M1);
    L1 += pl1[b * HG + hg] * f; A1 += pa1[((size_t)b * HG + hg) * D + d] * f;
  }
  for (int b = 0; b < nw; b++) {
    const float m = pm2[b * HG + hg];
    if (m == -INFINITY) continue;
    const float f = __expf(m - MW);
    LW += pl2[b * HG + hg] * f; AW += pa2[((size_t)b * HG + hg) * D + d] * f;
  }
  acc1[hg * D + d] = A1; aw[hg * D + d] = AW;
  if (d == 0) {
    m1[hg] = M1; l1[hg] = L1; mw[hg] = MW; lw[hg] = LW;
    const float M = fmaxf(M1, MW);
    lse[hg] = M + logf(L1 * __expf(M1 - M) + LW * __expf(MW - M));
  }
}

// final merge: add the delta partials (nbd) to the coarse sums, rotate the coarse accumulator back (acc1 @ R), merge
// with the window. grid Hkv * G, block D.
__global__ void merge2_k(const float* pld, const float* pad, int nbd, int HG, const float* m1, const float* l1,
                         const float* acc1, const float* mw, const float* lw, const float* aw, const float* R, float* out) {
  const int hg = blockIdx.x, d = threadIdx.x;
  __shared__ float a[D];
  __shared__ float Ls;
  float A = acc1[hg * D + d];
  for (int b = 0; b < nbd; b++) A += pad[((size_t)b * HG + hg) * D + d];
  a[d] = A;
  if (d == 0) { float L = l1[hg]; for (int b = 0; b < nbd; b++) L += pld[b * HG + hg]; Ls = L; }
  __syncthreads();
  float r = 0.f;
  for (int j = 0; j < D; j++) r += a[j] * R[j * D + d];
  const float M = fmaxf(m1[hg], mw[hg]);
  const float fa = (m1[hg] == -INFINITY) ? 0.f : __expf(m1[hg] - M), fb = __expf(mw[hg] - M);
  out[hg * D + d] = (r * fa + aw[hg * D + d] * fb) / (Ls * fa + lw[hg] * fb);
}

void prologue(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor pos, torch::Tensor ncomp,
              torch::Tensor kraw, torch::Tensor vraw, torch::Tensor Rt, double scale, torch::Tensor qk) {
  const int HG = q.size(0);
  TORCH_CHECK(kraw.size(1) * G == HG && kraw.size(2) == D, "kraw (Tw, Hkv, D)");
  prologue_k<<<HG, D, 0, c10::cuda::getCurrentCUDAStream()>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      pos.data_ptr<long>(), ncomp.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(kraw.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(vraw.data_ptr()), kraw.size(1), Rt.data_ptr<float>(), (float)scale, qk.data_ptr<float>());
}

void merge1(torch::Tensor pm1, torch::Tensor pl1, torch::Tensor pa1, int64_t nb1, torch::Tensor pm2, torch::Tensor pl2,
            torch::Tensor pa2, torch::Tensor m1, torch::Tensor l1, torch::Tensor acc1, torch::Tensor mw, torch::Tensor lw,
            torch::Tensor aw, torch::Tensor lse) {
  const int HG = m1.numel(), nw = pm2.size(0);
  merge1_k<<<HG, D, 0, c10::cuda::getCurrentCUDAStream()>>>(pm1.data_ptr<float>(), pl1.data_ptr<float>(), pa1.data_ptr<float>(),
      nb1, pm2.data_ptr<float>(), pl2.data_ptr<float>(), pa2.data_ptr<float>(), nw, HG, m1.data_ptr<float>(),
      l1.data_ptr<float>(), acc1.data_ptr<float>(), mw.data_ptr<float>(), lw.data_ptr<float>(), aw.data_ptr<float>(),
      lse.data_ptr<float>());
}

void merge2(torch::Tensor pld, torch::Tensor pad, int64_t nbd, torch::Tensor m1, torch::Tensor l1, torch::Tensor acc1,
            torch::Tensor mw, torch::Tensor lw, torch::Tensor aw, torch::Tensor R, torch::Tensor out) {
  const int HG = m1.numel();
  merge2_k<<<HG, D, 0, c10::cuda::getCurrentCUDAStream()>>>(pld.data_ptr<float>(), pad.data_ptr<float>(), nbd, HG,
      m1.data_ptr<float>(), l1.data_ptr<float>(), acc1.data_ptr<float>(), mw.data_ptr<float>(), lw.data_ptr<float>(),
      aw.data_ptr<float>(), R.data_ptr<float>(), out.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coarse", &coarse); m.def("delta", &delta); m.def("delta_tc", &delta_tc_launch);
  m.def("prologue", &prologue); m.def("merge1", &merge1); m.def("merge2", &merge2);
}
