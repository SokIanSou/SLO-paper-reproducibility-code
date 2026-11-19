"""Discrete-event Monte Carlo simulation of the tracking service of Section 3.1.
It samples the trajectory of the target (cameras, sojourns, gaps) and applies the activation policy:
after the target leaves a camera i in which it was tracked, the set Psi_i streams for at most T_h and the
session continues iff the target enters a camera of Psi_i within T_h. None of the model's equations is used."""
import numpy as np


class _Pool:
    def __init__(self, draw):
        self.draw, self.buf, self.i = draw, draw(), 0
    def __call__(self):
        if self.i >= len(self.buf):
            self.buf, self.i = self.draw(), 0
        self.i += 1
        return self.buf[self.i - 1]


def simulate(Phi, Th, Pdir, Pgap, p0, fam, start, n, seed, rc=2.5):
    rng = np.random.default_rng(seed)
    N = len(p0)
    cumdir = np.cumsum(np.hstack([Pdir, np.asarray(p0)[:, None]]), axis=1)     # last column: region 0
    cumgap = np.cumsum(Pgap, axis=1)
    exits = Pgap.sum(1) < 1 - 1e-12
    Psi = [Phi[i] | {i} for i in range(N)]
    B = 200000
    soj = _Pool(lambda: rng.gamma(fam.soj_shape, fam.mean_soj / fam.soj_shape, B))
    ugap = _Pool(lambda: rng.gamma(fam.gap_shape, 1.0 / fam.gap_shape, B))      # unit-mean gap
    uni = _Pool(lambda: rng.random(B))
    C = np.zeros(n); L = np.zeros(n); K = np.zeros(n); Vc = np.zeros(n); V0 = np.zeros(n)
    for s in range(n):
        i = start
        while True:
            T = soj(); C[s] += T; K[s] += rc * T; Vc[s] += 1          # tracked at camera i
            x = int(np.searchsorted(cumdir[i], uni()))
            if x < N and x in Phi[i]:                                 # direct move to a candidate
                i = x
                continue
            V0[s] += 1                                                # a search over Psi_i begins
            t, found = 0.0, None
            cam = x if x < N else None                                # unseen camera, or None = region 0
            last = i
            while True:
                if cam is not None:                                   # in a non-candidate camera
                    t += soj(); last = cam
                    if t > Th:
                        break
                    y = int(np.searchsorted(cumdir[cam], uni()))
                    if y == N:
                        cam = None
                        continue
                else:                                                 # out of coverage, last camera `last`
                    y = int(np.searchsorted(cumgap[last], uni()))
                    if y >= N:                                        # left the monitored area
                        break
                    t += ugap() * fam.gap_mean[last, y]
                    if t > Th:
                        break
                if y in Psi[i]:
                    found = y
                    break
                cam = y
            if found is None:                                         # search expires at the deadline
                L[s] += Th; K[s] += (1 + len(Phi[i])) * rc * Th
                break
            L[s] += t; K[s] += (1 + len(Phi[i])) * rc * t
            i = found
    return dict(C=C, L=L, S=C + L, K=K, Vc=Vc, V0=V0)
