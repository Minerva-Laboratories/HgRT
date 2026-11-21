"""Score draft-depth policies exactly on logged fixed-depth blocks: a block stopped after n drafts accepts min(n, nacc)
(drafting and greedy verification are deterministic), costs n draft steps + one verification pass with n + 1 rows.
    python dyn_sim.py log.jsonl [...]"""
import json, math, sys, collections
rows = [json.loads(l) for fn in sys.argv[1:] for l in open(fn)]
blocks = [r for r in rows if "nacc" in r]
V = collections.defaultdict(dict)
for r in rows:
    if r.get("cost") == "verify": V[r["kv"]][(r["T"], r["M"])] = r["s"]
K = max(len(b["p"]) for b in blocks)
def vcost(kv, T, M):                                   # linear interpolation in T
    Ts = sorted({t for t, m in V[kv]})
    lo = max([t for t in Ts if t <= T] or [Ts[0]]); hi = min([t for t in Ts if t >= T] or [Ts[-1]])
    a, b = V[kv][(lo, M)], V[kv][(hi, M)]
    return a if hi == lo else a + (b - a) * (T - lo) / (hi - lo)
dstep = sorted(b["t_draft"] for b in blocks)[len(blocks) // 2] / K   # median time of one draft step
def score(policy, sel):
    tok = t = 0.0
    for b in sel:
        n = policy(b)
        tok += min(n, b["nacc"]) + 1
        t += max(n, 1) * dstep * (n > 0) + vcost(b["kv"], b["T"], n + 1)
    return tok / t
fixed = {f"k={k}": (lambda k: lambda b: k)(k) for k in range(0, K + 1)}
def thr(tau):        # draft while the drafted token's top-1 probability >= tau (the token below tau is dropped)
    def f(b):
        n = 0
        while n < K and b["p"][n] >= tau: n += 1
        return n
    return f
def thr_keep(tau):   # same, but the first draft below tau is still verified
    def f(b):
        n = 0
        while n < K:
            n += 1
            if b["p"][n - 1] < tau: break
        return n
    return f
def surv(tau):       # stop when the product of top-1 probabilities (a survival estimate) falls below tau
    def f(b):
        n, s = 0, 1.0
        while n < K and s * b["p"][n] >= tau: s *= b["p"][n]; n += 1
        return n
    return f
oracle = lambda b: b["nacc"]
class CostAware:
    """AMTP-style, training-free, online: after paying for draft j (seeing its top-1 probability p), keep it if its
    expected extra tokens S * c_j(p) exceed rate * (extra verification time); draft another if the expected value of the
    next one (S * mean acceptance at that position) exceeds rate * (draft step + extra verification). c_j(p): acceptance
    per position and probability bin, learned online from verification outcomes (prior: p). rate: running tokens/s."""
    def __init__(self, Kmax, alpha=0.05, pnext=True):
        self.Kmax, self.alpha, self.pnext = Kmax, alpha, pnext
        self.n = [[2.0] * 10 for _ in range(Kmax)]; self.a = [[2.0 * (i + 0.5) / 10 for i in range(10)] for _ in range(Kmax)]
        self.rate = None
    def c(self, j, p): i = min(int(p * 10), 9); return self.a[j][i] / self.n[j][i]
    def mean(self, j): return sum(self.a[j]) / sum(self.n[j])
    def __call__(self, b):
        kv, T = b["kv"], b["T"]
        rate = self.rate or 1.0 / vcost(kv, T, 1)
        n, S = 0, 1.0
        while n < len(b["p"]):
            dV = vcost(kv, T, n + 2) - vcost(kv, T, n + 1)
            nxt = self.mean(n) if n == 0 or not self.pnext else self.c(n, b["p"][n - 1])   # predicted from the last draft
            if S * nxt < rate * (dstep + dV): break           # not worth drafting the next one
            p = b["p"][n]
            if S * self.c(n, p) < rate * dV: break                        # drafted, but not worth verifying
            S *= self.c(n, p); n += 1
        for j in range(min(n, b["nacc"] + 1)):                            # learn from what verification revealed
            i = min(int(b["p"][j] * 10), 9); self.n[j][i] += 1; self.a[j][i] += j < b["nacc"]
        t = max(n, 1) * dstep * (n > 0) + vcost(kv, T, n + 1); tok = min(n, b["nacc"]) + 1
        self.rate = tok / t if self.rate is None else (1 - self.alpha) * self.rate + self.alpha * tok / t
        return n
def cost_aware(): return CostAware(K)
def cost_aware_mean(): return CostAware(K, pnext=False)
groups = collections.defaultdict(list)
for b in blocks:
    groups[(b["kv"], b["prompt"])].append(b)
print(f"draft step {dstep * 1e3:.2f} ms; blocks {len(blocks)}")
# calibration of top-1 probability vs acceptance (given the prefix was accepted)
cal = collections.defaultdict(lambda: [0, 0])
for b in blocks:
    for j in range(K):
        if j > b["nacc"]: break
        bin_ = min(int(b["p"][j] * 10), 9); cal[(j, bin_)][0] += 1; cal[(j, bin_)][1] += j < b["nacc"]
print("acceptance by draft position and top-1 probability bin (n, rate):")
for j in range(K):
    print(f"  pos {j + 1}: " + " ".join(f"{bn / 10:.1f}:{cal[(j, bn)][0]}/{cal[(j, bn)][1] / max(cal[(j, bn)][0], 1):.2f}" for bn in range(10) if cal[(j, bn)][0]))
pols = dict(fixed)
for tau in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9): pols[f"p>={tau}"] = thr(tau); pols[f"p>={tau}+1"] = thr_keep(tau)
for tau in (0.2, 0.3, 0.4, 0.5, 0.6): pols[f"surv>={tau}"] = surv(tau)
pols["costaware"] = "CA"
pols["ca-mean"] = "CAM"
pols["oracle"] = oracle
keys = sorted(groups)
print("tokens/s by prompt (best fixed marked *):")
print(f"{'policy':12s} " + " ".join(f"{k[0][:3]}/{k[1]:>8s}" for k in keys) + "   total")
mk = lambda f: cost_aware() if f == "CA" else cost_aware_mean() if f == "CAM" else f
res = {p: [score(mk(f), groups[k]) for k in keys] + [score(mk(f), blocks)] for p, f in pols.items()}
best_fixed = [max(res[f"k={k}"][i] for k in range(K + 1)) for i in range(len(keys) + 1)]
for p, v in res.items():
    print(f"{p:12s} " + " ".join(f"{x:12.1f}" + ("*" if p.startswith("k=") and abs(x - best_fixed[i]) < 1e-9 else " ") for i, x in enumerate(v)))
