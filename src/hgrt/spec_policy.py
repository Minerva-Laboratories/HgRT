"""Adaptive draft depth for MTP speculative decoding (training-free, online; in the spirit of AMTP and AdaEDL).

Every depth is lossless (verification keeps exactly the plain greedy tokens), so the policy only trades time. Per block:

- Draft j costs one MTP step (`dstep`); verifying n drafts costs one pass with n + 1 rows, V(n + 1, T) at context T.
- After drafting j the policy sees the draft's top-1 probability p over the draft vocabulary. c_j(p), the chance that
  draft j is accepted given that drafts 1 .. j - 1 were, is learned online per position and probability bin from the
  verification outcomes (prior: p itself). S, the chance that every kept draft survives, is the product of the c's.
- `rate` is a running estimate of tokens per second. A draft is worth drafting if its expected extra tokens
  S * c_j(p of draft j - 1) exceed rate * (dstep + V(n + 2) - V(n + 1)); a drafted token is worth verifying if S * c_j(p) exceeds
  rate * (V(n + 2) - V(n + 1)). Otherwise the block stops.
"""
import time


class DepthPolicy:
    def __init__(self, kmax, vcost, bins=10, alpha=0.05, prior=2.0):
        self.kmax, self.vcost, self.bins, self.alpha = kmax, vcost, bins, alpha
        self.n = [[prior] * bins for _ in range(kmax)]
        self.a = [[prior * (i + 0.5) / bins for i in range(bins)] for _ in range(kmax)]
        self.rate = None
        self.dstep = None
        self.hist = [0] * (kmax + 1)                 # blocks by number of drafts verified

    def _bin(self, p):
        return min(int(p * self.bins), self.bins - 1)

    def c(self, j, p):
        i = self._bin(p)
        return self.a[j][i] / self.n[j][i]

    def mean(self, j):
        return sum(self.a[j]) / sum(self.n[j])

    # ---- per block: begin(T); want_more(n, S, p_last) / keep(n, p, S) while drafting; end(n, nacc, probs, seconds, ..)
    def begin(self, T):
        self.T = T
        if self.dstep is None:                        # an MTP draft step: ~5 % of a decode step until measured
            self.dstep = 0.05 * self.vcost(1, T)
        self.r = (self.rate if self.rate is not None else self.expected_rate(T))

    def expected_rate(self, T):
        """tokens per second of the best fixed depth under the current acceptance statistics (the starting rate of a
        call, before its own blocks are measured)."""
        best, S, tok = 1.0 / self.vcost(1, T), 1.0, 1.0
        for k in range(1, self.kmax + 1):
            S *= self.mean(k - 1)
            tok += S
            best = max(best, tok / (self.vcost(k + 1, T) + k * self.dstep))
        return best

    def want_more(self, n, S, p_last):
        """draft number n + 1 (n drafts kept so far, the last with top-1 probability p_last)? The next draft's
        acceptance is predicted as c_n(p_last): consecutive drafts' confidences are strongly correlated."""
        if n >= self.kmax:
            return False
        if n == 0:
            return True                               # the first draft: always (decided again in keep)
        dV = self.vcost(n + 2, self.T) - self.vcost(n + 1, self.T)
        return S * self.c(n, p_last) >= self.r * ((self.dstep or 0.0) + dV)

    def keep(self, n, p, S):
        """keep draft n + 1 (top-1 probability p) for verification? -> (keep, new S)"""
        dV = self.vcost(n + 2, self.T) - self.vcost(n + 1, self.T)
        c = self.c(n, p)
        return S * c >= self.r * dV, S * c

    def end(self, n, nacc, probs, seconds, draft_seconds, ndrafted):
        for j in range(min(n, nacc + 1)):            # outcomes verification revealed: drafts 1 .. nacc accepted,
            i = self._bin(probs[j])                   # draft nacc + 1 (if verified) rejected
            self.n[j][i] += 1
            self.a[j][i] += j < nacc
        if ndrafted:
            d = draft_seconds / ndrafted
            self.dstep = d if self.dstep is None else 0.9 * self.dstep + 0.1 * d
        r = (min(n, nacc) + 1) / max(seconds, 1e-6)
        self.rate = r if self.rate is None else (1 - self.alpha) * self.rate + self.alpha * r
        self.hist[n] += 1


def now():
    return time.perf_counter()
