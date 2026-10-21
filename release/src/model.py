"""Revised session-level model (Sec. 3): expected metrics, LST-based SLO probability,
and a Monte Carlo simulator of the tracking process for validation."""
import numpy as np
from math import comb
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy import integrate, stats

# ---------------- deployment helpers ----------------
def inv_dist_weights(D, ell=None):
    """Distance-decaying transition kernel: exp(-d/ell) (default) or 1/d if ell is None."""
    N = len(D); W = np.zeros_like(D)
    m = ~np.eye(N, dtype=bool)
    W[m] = np.exp(-D[m] / ell) if ell else 1.0 / D[m]
    return W / W.sum(1, keepdims=True)

def nearest_sets(D, k):
    """k may be int or per-camera array; nested nearest-k sets (excluding i)."""
    N = len(D); k = np.broadcast_to(np.asarray(k), (N,))
    order = [list(np.argsort(D[i])[1:]) for i in range(N)]
    return [order[i][:int(k[i])] for i in range(N)]

class Deployment:
    def __init__(self, xy, P0, mean_soj=30.0, gap="exp", gap_mean=30.0,
                 soj="exp", ps=0.95, rc=2.5, walk=1.2, cv=0.5, exit_p=0.0, ell=15.0,
                 p_back=0.15):
        self.xy = np.asarray(xy, float); self.N = len(xy)
        self.D = np.linalg.norm(self.xy[:, None] - self.xy[None], axis=2)
        self.P0 = np.asarray(P0, float)
        W = inv_dist_weights(self.D, ell)
        self.Pd = W * (1 - self.P0)[:, None]          # direct moves, all cameras (j != i)
        # reappearance after a gap: turn-back into camera i with prob p_back,
        # remaining mass over j != i by the distance kernel
        self.p_back = p_back
        self.Pg = (1 - p_back) * W * (1 - exit_p)
        np.fill_diagonal(self.Pg, p_back * (1 - exit_p))
        self.mean_soj = np.broadcast_to(np.asarray(mean_soj, float), (self.N,)).copy()
        self.gap, self.gap_mean, self.soj = gap, gap_mean, soj
        self.walk, self.cv = walk, cv
        self.ps_cam = np.full(self.N, ps)              # per-camera re-ID prob (0 = failed)
        self.rc = rc
        # gap-time distribution parameters per (i,j)
        if gap == "exp":
            self.gmean = np.full((self.N, self.N), gap_mean)
        else:  # lognormal, mean proportional to distance
            # a turn-back re-enters over the field-of-view scale, taken as half the
            # mean nearest-neighbour distance
            nn = np.sort(self.D, 1)[:, 1].mean()
            self.gmean = np.where(self.D > 0, self.D / walk, 0.5 * nn / walk)
        s2 = np.log(1 + cv**2)
        self.ln_sigma = np.sqrt(s2); self.ln_mu = np.log(self.gmean) - s2 / 2

    # --- gap quantities
    def F(self, Th):
        if self.gap == "exp": return 1 - np.exp(-Th / self.gmean)
        return stats.lognorm.cdf(Th, s=self.ln_sigma, scale=np.exp(self.ln_mu))
    def tau(self, Th):
        if np.isinf(Th): return self.gmean.copy()
        if self.gap == "exp":
            m = self.gmean; a = np.exp(-Th / m); return m * (1 - a) - Th * a
        # partial expectation of lognormal
        mu, s = self.ln_mu, self.ln_sigma
        return np.exp(mu + s**2 / 2) * stats.norm.cdf((np.log(Th) - mu - s**2) / s)
    def Gamma(self, s, Th, i, j):
        if self.gap == "exp":
            lam = 1 / self.gmean[i, j]
            return lam / (lam + s) * (1 - np.exp(-(lam + s) * Th))
        x, w = np.polynomial.legendre.leggauss(160)
        t = 0.5 * Th * (x + 1); w = 0.5 * Th * w
        f = stats.lognorm.pdf(t, s=self.ln_sigma, scale=np.exp(self.ln_mu[i, j]))
        return np.sum(w * f * np.exp(-s * t))
    def H(self, s, i):
        m = self.mean_soj[i]
        if self.soj == "exp": return 1 / (1 + s * m)
        return (1 + s * m / 2) ** -2      # gamma, shape 2

# ---------------- analytical model ----------------
def build(dep, Phi, Th):
    N = dep.N; ps = dep.ps_cam
    F = dep.F(Th) if np.isfinite(Th) else np.ones((N, N))
    tau = dep.tau(Th)
    rows, cols, vals = [], [], []
    r = {k: np.zeros(2 * N) for k in ("C", "L", "K", "D", "Vc", "V0")}
    phi = np.zeros(N); beta = np.zeros(N); m = np.zeros(N)
    for i in range(N):
        P = Phi[i]                                   # candidate set, i not included
        Psi = [i] + list(P)                          # cameras streaming and searched
        nf = 1 + sum(1 for j in P if ps[j] > 0)      # camera i plus surviving candidates
        succ_d = sum(ps[j] * dep.Pd[i, j] for j in P)
        for j in P:
            rows += [i]; cols += [j]; vals += [ps[j] * dep.Pd[i, j]]
        for j in Psi:                                # a search may recover the target at i
            g = ps[j] * dep.Pg[i, j] * F[i, j]
            rows += [N + i]; cols += [j]; vals += [g]
        rows += [i]; cols += [N + i]; vals += [dep.P0[i]]
        phi[i] = 1 - dep.P0[i] - succ_d
        beta[i] = 1 - sum(ps[j] * dep.Pg[i, j] * F[i, j] for j in Psi)
        Thf = Th if np.isfinite(Th) else 0.0
        m[i] = sum(ps[j] * dep.Pg[i, j] * tau[i, j] for j in Psi) + beta[i] * Thf
        if not np.isfinite(Th) and beta[i] > 1e-12: m[i] = np.inf
        # --- untracked time L: time spent OUT OF COVERAGE, truncated at T_h.
        # A direct move to an unsearched camera has zero gap, so it contributes nothing;
        # a search contributes E[min(G, T_h)] whatever its outcome.
        gap_all = sum(dep.Pg[i, j] * tau[i, j] for j in range(N))
        p_late = sum(dep.Pg[i, j] * (1 - F[i, j]) for j in range(N)) + (1 - dep.Pg[i].sum())
        # --- stream time: the searched set streams for the gap on success, else the full T_h
        succ_gap = sum(ps[j] * dep.Pg[i, j] * tau[i, j] for j in Psi)
        stream_cam = phi[i] * Thf              # failed direct move: streams still run T_h
        stream_0 = succ_gap + beta[i] * Thf    # search phase duration
        r["C"][i] = dep.mean_soj[i]
        r["L"][i] = 0.0                        # zero gap on a direct move
        r["L"][N + i] = gap_all + p_late * Thf
        r["K"][i] = dep.rc * dep.mean_soj[i] + stream_cam * nf * dep.rc
        r["K"][N + i] = nf * dep.rc * stream_0
        # --- continuity time D: sojourns plus the gaps of SUCCESSFUL searches only
        r["D"][i] = dep.mean_soj[i]; r["D"][N + i] = succ_gap
        r["Vc"][i] = 1; r["V0"][N + i] = 1
    Q = sparse.csc_matrix((vals, (rows, cols)), shape=(2 * N, 2 * N))
    return Q, r, dict(phi=phi, beta=beta, m=m)

def metrics(dep, Phi, Th):
    N = dep.N
    Q, r, aux = build(dep, Phi, Th)
    lu = splu(sparse.identity(2 * N, format="csc") - Q)
    out = {k: lu.solve(v) for k, v in r.items()}
    out["S"] = out["C"] + out["L"]
    out = {k: v[:N] for k, v in out.items()}
    out["R"] = out["C"] / out["S"]; out["kappa"] = out["K"] / out["C"]
    out["Krate"] = out["K"] / out["S"]; out["H"] = out["Vc"] - 1
    return out

def psi(dep, Phi, Th, s):
    N = dep.N; ps = dep.ps_cam; A = np.zeros((2*N, 2*N), complex); b = np.zeros(2*N, complex)
    F = dep.F(Th)
    for i in range(N):
        Hs = dep.H(s, i); P = Phi[i]; Psi = [i] + list(P)
        succ_d = sum(ps[j] * dep.Pd[i, j] for j in P)
        phi = 1 - dep.P0[i] - succ_d
        beta = 1 - sum(ps[j] * dep.Pg[i, j] * F[i, j] for j in Psi)
        for j in P:
            A[i, j] = Hs * ps[j] * dep.Pd[i, j]
        for j in Psi:
            A[N + i, j] = ps[j] * dep.Pg[i, j] * dep.Gamma(s, Th, i, j)
        A[i, N + i] = Hs * dep.P0[i]
        b[i] = Hs * phi; b[N + i] = beta      # continuity stops at the last sighting
    return np.linalg.solve(np.eye(2*N) - A, b)[:N]

def ccdf(dep, Phi, Th, t, A=18.4, n=15, mm=11):
    """Pr(D > t) for all start cameras (Abate-Whitt Euler inversion)."""
    terms = []
    for k in range(n + mm + 1):
        s = (A + 2j * np.pi * k) / (2 * t)
        f = (1 - psi(dep, Phi, Th, s)) / s
        terms.append(((-1) ** k) * f.real * (0.5 if k == 0 else 1.0))
    terms = np.array(terms); cs = np.cumsum(terms, axis=0)
    val = sum(comb(mm, j) * 2.0 ** -mm * cs[n + j] for j in range(mm + 1))
    return np.clip(np.exp(A / 2) / t * val, 0, 1)

# ---------------- Monte Carlo simulator ----------------
def simulate(dep, Phi, Th, i0, n=20000, seed=0):
    rng = np.random.default_rng(seed); N = dep.N
    Phis = [set(p) for p in Phi]; out = np.zeros((n, 7))
    cumd = np.cumsum(dep.Pd / dep.Pd.sum(1, keepdims=True), 1)
    exitp = 1 - dep.Pg.sum(1)
    cumg = np.cumsum(dep.Pg, 1)
    for k in range(n):
        i = i0; S = C = L = K = D = 0.0; vc = 1; v0 = 0
        while True:
            m = dep.mean_soj[i]
            t = rng.exponential(m) if dep.soj == "exp" else rng.gamma(2, m / 2)
            C += t; D += t; K += dep.rc * t
            nf = 1 + sum(1 for j in Phis[i] if dep.ps_cam[j] > 0)   # camera i streams too
            if rng.random() < dep.P0[i]:
                v0 += 1; u = rng.random()
                j = int(np.searchsorted(cumg[i], u)) if u < cumg[i, -1] else -1
                if j >= 0:
                    g = (rng.exponential(dep.gmean[i, j]) if dep.gap == "exp" else
                         rng.lognormal(dep.ln_mu[i, j], dep.ln_sigma))
                    if (j == i or j in Phis[i]) and g <= Th and rng.random() < dep.ps_cam[j]:
                        L += g; D += g; K += nf * dep.rc * g; i = j; vc += 1; continue
                    L += min(g, Th)                  # out-of-coverage time, truncated
                else:
                    L += Th                          # never reappears
                K += nf * dep.rc * Th; break
            else:
                j = int(np.searchsorted(cumd[i], rng.random()))
                if j in Phis[i] and rng.random() < dep.ps_cam[j]:
                    i = j; vc += 1; continue
                K += nf * dep.rc * Th; break         # zero gap: no untracked time charged
        out[k] = (C + L, C, L, K, vc, v0, D)
    return out

# ---------------- fixed testbed ----------------
TESTBED_XY = np.array([[10, 12], [24, 8], [18, 24], [32, 20], [26, 34],   # dense 1-5
                       [58, 14], [74, 38], [96, 20], [88, 58], [60, 64]], float)  # sparse 6-10
TESTBED_P0 = np.array([0.3, 0.2, 0.2, 0.2, 0.4, 0.6, 0.7, 0.8, 0.7, 0.65])
DENSE, SPARSE = 2, 8   # camera 3 and camera 9 (0-indexed)

def random_deployment(N, tier, rng, density=22/100**2, **kw):
    side = np.sqrt(N / density); xy = rng.uniform(0, side, (N, 2))
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    md = np.sort(D, 1)[:, 1:4].mean(1)
    lo, hi = tier
    P0 = lo + (hi - lo) * (md - md.min()) / (md.max() - md.min() + 1e-12)
    return Deployment(xy, P0, **kw)


# ---------------- fast sparse evaluation (exponential sojourns and gaps) ----------------
class Fast:
    """Precomputes structure for a given (dep, Phi, Th); evaluates expected metrics and
    Pr(S >= Smin) with sparse solves. Assumes dep.soj == dep.gap == 'exp'."""
    def __init__(self, dep, Phi, Th):
        self.dep, self.Phi, self.Th = dep, Phi, Th; N = dep.N; ps = dep.ps_cam
        F = dep.F(Th); r_, c_, v_, kind, lam = [], [], [], [], []
        self.phi = np.zeros(N); self.beta = np.zeros(N)
        for i in range(N):
            Psi = [i] + list(Phi[i])
            for j in Phi[i]:
                r_.append(i); c_.append(j); v_.append(ps[j]*dep.Pd[i, j]); kind.append(0); lam.append(0)
            for j in Psi:
                r_.append(N+i); c_.append(j); v_.append(ps[j]*dep.Pg[i, j]); kind.append(2); lam.append(1/dep.gmean[i, j])
            r_.append(i); c_.append(N+i); v_.append(dep.P0[i]); kind.append(0); lam.append(0)
            self.phi[i] = 1 - dep.P0[i] - sum(ps[j]*dep.Pd[i, j] for j in Phi[i])
            self.beta[i] = 1 - sum(ps[j]*dep.Pg[i, j]*F[i, j] for j in Psi)
        self.r = np.array(r_); self.c = np.array(c_); self.v = np.array(v_)
        self.kind = np.array(kind); self.lam = np.array(lam)
        self.lc = 1 / dep.mean_soj
    def Q(self, s):
        N = self.dep.N; v = self.v.astype(complex).copy()
        cam = self.kind == 0
        v[cam] *= (self.lc[self.r[cam]] / (self.lc[self.r[cam]] + s))
        g = self.kind == 2; l = self.lam[g]
        v[g] *= l / (l + s) * (1 - np.exp(-(l + s) * self.Th))
        return sparse.csc_matrix((v, (self.r, self.c)), shape=(2*N, 2*N))
    def b(self, s):
        N = self.dep.N                                   # D stops at the last sighting
        return np.concatenate([self.lc / (self.lc + s) * self.phi, self.beta])
    def psi(self, s):
        N = self.dep.N
        return splu(sparse.identity(2*N, format="csc", dtype=complex) - self.Q(s)).solve(self.b(s))[:N]
    def ccdf(self, t, A=18.4, n=15, mm=11):
        acc = []
        for k in range(n + mm + 1):
            s = (A + 2j*np.pi*k) / (2*t)
            acc.append(((-1)**k) * ((1 - self.psi(s)) / s).real * (0.5 if k == 0 else 1.0))
        cs = np.cumsum(np.array(acc), 0)
        val = sum(comb(mm, j) * 2.0**-mm * cs[n+j] for j in range(mm+1))
        return np.clip(np.exp(A/2) / t * val, 0, 1)

def slo_and_load(dep, Phi, Th, Smin, nu, Lam):
    """Network-level SLO probability sum_i nu_i Pr(S_i >= Smin) and aggregate load (Prop. 6)."""
    m = metrics(dep, Phi, Th)
    p = Fast(dep, Phi, Th).ccdf(Smin)           # Pr(D >= Dmin), taken on D directly
    return float(nu @ p), float(Lam * (nu @ m["K"])), m, p

def greedy_config(dep, Th, Smin, target, nu, Lam, kmax=None, order=None, k0=None):
    """Greedy SLO-constrained allocation of nested nearest-k candidate sets (Sec. 5)."""
    N = dep.N; kmax = kmax or N - 1
    ordr = order if order is not None else [list(np.argsort(dep.D[i])[1:]) for i in range(N)]
    ordr = [[j for j in o if dep.ps_cam[j] > 0] for o in ordr]
    k = np.zeros(N, int) if k0 is None else np.array(k0, int)
    mk = lambda kk: [ordr[i][:kk[i]] for i in range(N)]
    slo, load, *_ = slo_and_load(dep, mk(k), Th, Smin, nu, Lam)
    while slo < target:
        best = None
        for i in range(N):
            if k[i] >= min(kmax, len(ordr[i])) or nu[i] == 0 and False: continue
            kk = k.copy(); kk[i] += 1
            s2, l2, *_ = slo_and_load(dep, mk(kk), Th, Smin, nu, Lam)
            gain = s2 - slo; dl = max(l2 - load, 1e-9)
            score = gain / dl
            if gain > 1e-12 and (best is None or score > best[0]): best = (score, i, s2, l2)
        if best is None: return None
        _, i, slo, load = best; k[i] += 1
    # pruning pass
    improved = True
    while improved:
        improved = False
        for i in np.argsort(-k):
            if k[i] == 0: continue
            kk = k.copy(); kk[i] -= 1
            s2, l2, *_ = slo_and_load(dep, mk(kk), Th, Smin, nu, Lam)
            if s2 >= target and l2 < load: k, slo, load, improved = kk, s2, l2, True
    return k, slo, load
