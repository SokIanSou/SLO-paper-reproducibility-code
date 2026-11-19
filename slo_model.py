"""Session-level model of candidate-driven camera activation (Section 3 of the paper).

Indices are 0-based: camera i of the paper is index i-1.
  mobility      : Pdir[i,j] = P_{i,j},  p0[i] = P_{i,0},  Pgap[i,j] = P_{0^(i),j}
  policy        : Phi[i] (set of candidates of camera i),  Th (grace period)
  handover      : a[i,j] (eq. a), q[i] (eq. q), w[i] (eq. w)
  session chain : Z = (I - a)^-1 and the rewards of one visit (Section 3.2)
"""
import numpy as np
from scipy.linalg import expm
from math import comb

# ----------------------------------------------------------------------------- deployment
POS = np.array([(10, 12), (24, 8), (18, 24), (32, 20), (26, 34),
                (58, 14), (74, 38), (96, 20), (88, 58), (60, 64)], float)
P0 = np.array([0.3, 0.2, 0.2, 0.2, 0.4, 0.6, 0.7, 0.8, 0.7, 0.65])
DENSE, SPARSE = 2, 8          # cameras 3 and 9
DEFAULTS = dict(mean_soj=30.0, mean_gap=30.0, pback=0.15, ell=15.0, rc=2.5,
                k=5, Th=20.0, Cmin=120.0, Lambda=0.05)


def mobility(pos=POS, p0=P0, ell=15.0, pback=0.15):
    """distance-decaying kernel, eq. (Pij)"""
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    w = np.exp(-d / ell)
    np.fill_diagonal(w, 0.0)
    w /= w.sum(1, keepdims=True)
    Pgap = (1 - pback) * w
    np.fill_diagonal(Pgap, pback)
    Pdir = (1 - np.asarray(p0))[:, None] * w
    return d, Pdir, Pgap


def nearest_sets(d, k):
    N = len(d)
    ks = np.full(N, k, int) if np.isscalar(k) else np.asarray(k, int)
    order = [np.argsort(d[i], kind="stable")[1:] for i in range(N)]
    return [set(order[i][:ks[i]].tolist()) for i in range(N)]


# ----------------------------------------------------------------------------- distributions
class Family:
    """Sojourn and gap laws.  gap_mean[k,m] is the mean of G_{0^(k),m}."""
    def __init__(self, name, mean_soj, gap_mean, soj_shape, gap_shape):
        self.name, self.mean_soj, self.gap_mean = name, mean_soj, np.asarray(gap_mean, float)
        self.soj_shape, self.gap_shape = soj_shape, gap_shape      # gamma shapes (1 = exponential)
        self.exponential = (soj_shape == 1 and gap_shape == 1 and np.ptp(self.gap_mean) == 0)

    def lstH(self, s):
        return (1 + self.mean_soj * s / self.soj_shape) ** (-self.soj_shape)

    def lstF(self, s):                                             # matrix of gap transforms
        return (1 + self.gap_mean * s / self.gap_shape) ** (-self.gap_shape)


def exponential_family(N, mean_soj=30.0, mean_gap=30.0):
    return Family("exponential", mean_soj, np.full((N, N), mean_gap), 1, 1)


def general_family(d, mean_soj=30.0, speed=1.2):
    """gamma sojourns (shape 2); gamma gaps (shape 4, CV 0.5) whose mean is the walking time to the
    destination; a turn-back walks half the average nearest-neighbour spacing."""
    N = len(d)
    nn = np.sort(d + np.diag(np.full(N, np.inf)), axis=1)[:, 0].mean()
    gm = d / speed
    np.fill_diagonal(gm, 0.5 * nn / speed)
    return Family("general", mean_soj, gm, 2, 4)


# ----------------------------------------------------------------------------- inversion
def euler(fhat, t, M=15, A=18.4, n=38):
    """Abate-Whitt Euler algorithm; fhat maps complex s to an array, the inverse is taken at t."""
    ks = np.arange(0, n + M + 1)
    vals = np.array([np.real(fhat(complex(A / (2 * t), np.pi * k / t))) for k in ks])
    vals[0] *= 0.5
    sgn = ((-1.0) ** ks).reshape((-1,) + (1,) * (vals.ndim - 1))
    partial = np.cumsum(sgn * vals, axis=0)
    b = np.array([comb(M, m) for m in range(M + 1)]) / 2.0 ** M
    return np.exp(A / 2) / t * np.tensordot(b, partial[n:n + M + 1], axes=(0, 0))


# ----------------------------------------------------------------------------- handover kernel
def _locations(i, Phi_i, N):
    Psi = sorted(Phi_i | {i})
    non = [k for k in range(N) if k not in Phi_i and k != i]
    gaps = non + [i]
    return Psi, non, gaps


def handover_row(i, Phi_i, Th, Pdir, Pgap, p0, fam):
    """a_{i,.}(T_h) and w_i for one camera: eqs (a), (w), with the untracked target followed through
    non-candidate cameras and out-of-coverage states (Section 3.2.2)."""
    N = len(p0)
    Psi, non, gaps = _locations(i, Phi_i, N)
    nc, n = len(non), len(non) + len(gaps)
    direct = np.array([Pdir[i, j] if j in Phi_i else 0.0 for j in Psi])
    pi = np.zeros(n)
    pi[:nc] = Pdir[i, non]
    pi[n - 1] = p0[i]                                   # 0^(i) is the last location
    a_row = np.zeros(N)
    if fam.exponential:                                 # Remark (exponential case)
        lc, l0 = 1 / fam.mean_soj, 1 / fam.gap_mean[0, 0]
        G = np.zeros((n, n)); g = np.zeros((n, len(Psi)))
        G[:nc, :nc] = lc * Pdir[np.ix_(non, non)]
        for u, k in enumerate(non):
            G[u, nc + u] = lc * p0[k]
        G[nc:, :nc] = l0 * Pgap[np.ix_(gaps, non)]
        G[np.arange(nc), np.arange(nc)] -= lc
        G[np.arange(nc, n), np.arange(nc, n)] -= l0
        g[:nc] = lc * Pdir[np.ix_(non, Psi)]
        g[nc:] = l0 * Pgap[np.ix_(gaps, Psi)]
        Gi = np.linalg.inv(G); E = expm(G * Th); I = np.eye(n)
        cont = pi @ Gi @ (E - I) @ g
        integ = pi @ Gi @ (Gi @ (E - I) - Th * I) @ g
    else:                                               # transform equations (reduced) + inversion
        def a_cont(s):
            # out-of-coverage states eliminated: unknowns are the non-candidate cameras only
            h, PF = fam.lstH(s), Pgap * fam.lstF(s)
            Pt = Pdir + np.asarray(p0)[:, None] * PF            # P~_{k,u}(s): direct, or through one gap
            out = p0[i] * PF[i, Psi].astype(complex)
            if nc:
                theta = np.linalg.solve(np.eye(nc) - h * Pt[np.ix_(non, non)], h * Pt[np.ix_(non, Psi)])
                out = out + (Pdir[i, non] + p0[i] * PF[i, non]) @ theta
            return out
        both = euler(lambda s: np.concatenate([a_cont(s) / s, a_cont(s) / s ** 2]), Th)
        cont, integ = both[:len(Psi)], both[len(Psi):]
    a_row[Psi] = direct + cont
    w = Th - (direct * Th + integ).sum()
    return a_row, w


def handover(Phi, Th, Pdir, Pgap, p0, fam):
    N = len(p0)
    rows = [handover_row(i, Phi[i], Th, Pdir, Pgap, p0, fam) for i in range(N)]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


# ----------------------------------------------------------------------------- session metrics
def metrics(a, w, Phi, Pdir, mean_soj, rc):
    N = len(w)
    q = 1 - a.sum(1)
    Z = np.linalg.inv(np.eye(N) - a)                               # fundamental matrix
    sizes = np.array([len(P) for P in Phi])
    rC = np.full(N, mean_soj)
    rK = rc * mean_soj + (1 + sizes) * rc * w
    rV0 = np.array([1 - sum(Pdir[j, k] for k in Phi[j]) for j in range(N)])
    C, L, K = Z @ rC, Z @ w, Z @ rK
    return dict(a=a, q=q, w=w, Z=Z, C=C, L=L, S=C + L, K=K, Vc=Z.sum(1), V0=Z @ rV0,
                R=C / (C + L), kappa=K / C, f=(np.diag(Z) - 1) / np.diag(Z))


def slo_ccdf(a, q, lstH, t):
    """Pr(C^(i) > t) for every start camera i (Proposition 'SLO satisfaction probability')."""
    N = len(q)
    I = np.eye(N)
    def fhat(s):
        h = lstH(s)
        return (1 - np.linalg.solve(I - h * a, h * q)) / s
    return euler(fhat, t)


def evaluate(Phi, Th, Pdir, Pgap, p0, fam, rc=2.5, Cmin=120.0):
    a, w = handover(Phi, Th, Pdir, Pgap, p0, fam)
    m = metrics(a, w, Phi, Pdir, fam.mean_soj, rc)
    m["slo"] = slo_ccdf(a, m["q"], fam.lstH, Cmin)
    return m
