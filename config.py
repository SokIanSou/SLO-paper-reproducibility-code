"""Section 4: SLO-constrained configuration (problem (P), Algorithm 1) and the two baseline rules."""
import numpy as np
from slo_model import *


class Problem:
    """Evaluates Pi(x) and the load L(x) for x = (k, T_h) with nested nearest-first candidate sets.
    Row i of the handover matrix depends on the configuration only through (k_i, T_h), so rows are cached."""
    def __init__(self, d, Pdir, Pgap, p0, fam, nu=None, Lambda=0.05, rc=2.5, Cmin=120.0, alive=None):
        self.d, self.Pdir, self.Pgap, self.p0, self.fam = d, Pdir, Pgap, np.asarray(p0), fam
        self.N = len(p0)
        self.alive = list(range(self.N)) if alive is None else alive        # cameras that can see
        self.nu = np.full(self.N, 1 / self.N) if nu is None else np.asarray(nu)
        self.Lambda, self.rc, self.Cmin = Lambda, rc, Cmin
        self.order = [[j for j in np.argsort(d[i], kind="stable") if j != i and j in self.alive] for i in range(self.N)]
        self.rows, self.cache = {}, {}

    def sets(self, k):
        return [set(self.order[i][:k[i]]) for i in range(self.N)]

    def row(self, i, ki, Th):
        key = (i, ki, Th)
        if key not in self.rows:
            self.rows[key] = handover_row(i, set(self.order[i][:ki]), Th, self.Pdir, self.Pgap, self.p0, self.fam)
        return self.rows[key]

    def evaluate(self, k, Th):
        key = (tuple(int(x) for x in k), Th)
        if key not in self.cache:
            rows = [self.row(i, key[0][i], Th) for i in range(self.N)]
            a = np.array([r[0] for r in rows]); w = np.array([r[1] for r in rows])
            m = metrics(a, w, self.sets(key[0]), self.Pdir, self.fam.mean_soj, self.rc)
            slo = slo_ccdf(a, m["q"], self.fam.lstH, self.Cmin)
            self.cache[key] = (float(self.nu @ slo), float(self.Lambda * self.nu @ m["K"]))
        return self.cache[key]


def greedy(prob, T, target, kmax, k0=None, delta=1e-9):
    """Algorithm 1.  Returns (best, pimax, status) with best = (load, k, T_h) or None.
    status: 'ok', 'infeasible' (certified: every live camera already searches all live cameras), or 'none found'."""
    N = prob.N
    cap = np.array([min(kmax, len(prob.order[i])) if i in prob.alive else 0 for i in range(N)])
    full = all(cap[i] == len(prob.order[i]) for i in prob.alive)
    pimax = prob.evaluate(cap, max(T))[0]
    if pimax < target and full:
        return None, pimax, "infeasible"
    best = (np.inf, None, None)

    def augment(k, Th):
        pi, load = prob.evaluate(k, Th)
        while pi < target:
            cand = []
            for i in range(N):
                if k[i] < cap[i]:
                    e = k.copy(); e[i] += 1
                    p2, l2 = prob.evaluate(e, Th)
                    if p2 - pi > 0:
                        cand.append(((p2 - pi) / max(l2 - load, delta), i))
            if not cand:
                break
            k[max(cand)[1]] += 1
            pi, load = prob.evaluate(k, Th)
        return k, pi, load

    for Th in T:
        start = np.zeros(N, int) if k0 is None else np.minimum(np.array(k0, int), cap)
        k, pi, load = augment(start.copy(), Th)
        if pi < target and start.any():                               # a warm start stalled: restart from zero
            k, pi, load = augment(np.zeros(N, int), Th)
        if pi < target:
            continue
        changed = True
        while changed:                                               # pruning
            changed = False
            for i in range(N):
                if k[i] > 0:
                    e = k.copy(); e[i] -= 1
                    p2, l2 = prob.evaluate(e, Th)
                    if p2 >= target and l2 < load:
                        k, pi, load, changed = e, p2, l2, True
        if load < best[0]:
            best = (load, k.copy(), Th)
    if best[1] is None:
        return None, pimax, "none found"
    return best, pimax, "ok"


def uniform(prob, T, target, kmax):
    best = (np.inf, None, None)
    for Th in T:
        for kk in range(kmax + 1):
            pi, load = prob.evaluate(np.full(prob.N, kk), Th)
            if pi >= target and load < best[0]:
                best = (load, np.full(prob.N, kk), Th)
    return best


def proportional(prob, T, target, kmax):
    """|Phi_i| = min(kmax, floor(c P_{i,0} + 1/2)) for c = 0.25, 0.5, ..., 60; cheapest feasible setting"""
    best = (np.inf, None, None)
    vecs = {tuple(np.minimum(kmax, np.floor(c * prob.p0 + 0.5)).astype(int)) for c in np.arange(0.25, 60.01, 0.25)}
    for Th in T:
        for k in vecs:
            pi, load = prob.evaluate(np.array(k), Th)
            if pi >= target and load < best[0]:
                best = (load, np.array(k), Th)
    return best
