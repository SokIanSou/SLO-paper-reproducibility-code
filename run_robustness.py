"""Section 5.3: robustness to history-dependent (directional) mobility.

Ground truth: a target with directional inertia.  The next camera after camera i (directly or after a gap)
is drawn with weight exp(-d_ij/ell) * exp(beta * cos phi), phi being the angle between the last displacement
(previous camera -> i) and the move i -> j; beta = 0 is the first-order mobility of the paper.  Turn-back
probability, out-of-coverage probabilities, sojourns and gaps are unchanged.
The first-order model is calibrated from a long trajectory of this target, as a deployed system would from its
logs, and compared with session-level simulation of the inertial target."""
import json, numpy as np
from slo_model import *
from config import Problem, greedy

d, Pdir0, Pgap0 = mobility(); N = 10; fam = exponential_family(N); ELL, PB = 15.0, 0.15


def inertial_kernel(beta):
    """K[p, i, j]: destination weights for a target in camera i whose previous camera was p (p == i: no direction)"""
    K = np.zeros((N, N, N))
    base = np.exp(-d / ELL)
    for p in range(N):
        for i in range(N):
            w = base[i].copy(); w[i] = 0.0
            if p != i and beta:
                u = POS[i] - POS[p]; u /= np.linalg.norm(u)
                for j in range(N):
                    if j != i:
                        v = POS[j] - POS[i]; w[j] *= np.exp(beta * (u @ v) / np.linalg.norm(v))
            K[p, i] = w / w.sum()
    return K


class Target:
    def __init__(self, beta, rng):
        self.K = inertial_kernel(beta); self.rng = rng
        self.cumK = np.cumsum(self.K, axis=2)

    def dest(self, p, i):
        return int(np.searchsorted(self.cumK[p, i], self.rng.random()))

    def leave(self, p, i):
        """leave camera i (previous camera p): returns (kind, dest, gap) with kind 'direct' or 'gap'"""
        if self.rng.random() < P0[i]:
            if self.rng.random() < PB:
                return "gap", i, self.rng.exponential(fam.mean_soj)
            return "gap", self.dest(p, i), self.rng.exponential(fam.gap_mean[0, 0])
        return "direct", self.dest(p, i), 0.0


def calibrate(beta, steps, seed):
    """first-order estimates of P_{i,j}, P_{i,0}, P_{0^(i),j} and the predecessor distribution of each camera"""
    tg = Target(beta, np.random.default_rng(seed))
    nd = np.zeros((N, N)); ng = np.zeros((N, N)); dep = np.zeros(N); pred = np.ones((N, N)) * 1e-9
    p, i = 0, 1
    for _ in range(steps):
        kind, j, _ = tg.leave(p, i); dep[i] += 1
        (nd if kind == "direct" else ng)[i, j] += 1
        pred[j, i] += 1
        p, i = (i, j) if j != i else (i, i)
    p0 = ng.sum(1) / dep
    Pdir = nd / dep[:, None]
    Pgap = ng / ng.sum(1, keepdims=True)
    return Pdir, Pgap, p0, pred / pred.sum(1, keepdims=True)


def simulate_inertial(beta, Phi, Th, start, pred, n, seed):
    tg = Target(beta, np.random.default_rng(seed)); rng = tg.rng
    Psi = [Phi[i] | {i} for i in range(N)]
    C = np.zeros(n)
    for s in range(n):
        i = start; p = int(np.searchsorted(np.cumsum(pred[start]), rng.random()))
        while True:
            C[s] += rng.exponential(fam.mean_soj)
            kind, j, g = tg.leave(p, i)
            if kind == "direct" and j in Phi[i]:
                p, i = i, j; continue
            t, last, prev = g, i, p
            cam = j if kind == "direct" else None
            if kind == "gap":
                if t > Th: break
                if j in Psi[i]: p, i = (i, j) if j != i else (i, i); continue
                cam = j; prev = i
            found = None
            while True:                                   # unseen in camera `cam`, previous camera `prev`
                t += rng.exponential(fam.mean_soj)
                if t > Th: break
                kind, y, g = tg.leave(prev, cam)
                t += g
                if t > Th: break
                if y in Psi[i]: found = (cam if y != cam else cam, y); break
                prev, cam = (cam, y) if y != cam else (cam, cam)
            if found is None: break
            p, i = found
    return C


if __name__ == "__main__":
    out = {"cases": [], "config": []}
    NS = 40000
    for beta in (0.0, 1.0, 2.0):
        Pdir, Pgap, p0, pred = calibrate(beta, 1_500_000, 7)
        print(f"beta={beta}: calibrated |P - P(beta=0)| max {np.abs(Pdir - Pdir0).max():.3f}")
        for k, Th in ((5, 20.0), (3, 60.0)):
            Phi = nearest_sets(d, k)
            m = evaluate(Phi, Th, Pdir, Pgap, p0, fam)
            for st in (DENSE, SPARSE):
                C = simulate_inertial(beta, Phi, Th, st, pred, NS, 100 + st + int(10 * beta) + k)
                r = dict(beta=beta, k=k, Th=Th, start=st + 1, C_model=float(m["C"][st]), C_sim=float(C.mean()),
                         C_se=float(C.std() / np.sqrt(NS)), slo_model=float(m["slo"][st]), slo_sim=float((C >= 120).mean()))
                out["cases"].append(r)
                print(f"  k={k} Th={Th:4.0f} start {st+1}: E[C] model {r['C_model']:7.1f} sim {r['C_sim']:7.1f} ({100*(r['C_model']/r['C_sim']-1):+5.1f}%)"
                      f" | Pr model {r['slo_model']:.3f} sim {r['slo_sim']:.3f}")
        # configuration chosen on the calibrated model: is the SLO met in the inertial world?
        prob = Problem(d, Pdir, Pgap, p0, fam)
        (load, kk, T), _, _ = greedy(prob, [10, 20, 30, 40, 60, 80, 100, 120], 0.5, 9)
        Phi = prob.sets(kk)
        true_pi = float(np.mean([(simulate_inertial(beta, Phi, float(T), st, pred, 15000, 900 + st) >= 120).mean() for st in range(N)]))
        out["config"].append(dict(beta=beta, load=load, T=int(T), k=kk.tolist(), pi_model=prob.evaluate(kk, T)[0], pi_true=true_pi))
        print(f"  config for 0.5 on calibrated model: T={T} k={kk.tolist()} load {load:.1f}; Pi model {prob.evaluate(kk, T)[0]:.3f}, inertial simulation {true_pi:.3f}")
    json.dump(out, open("results/robustness.json", "w"))
