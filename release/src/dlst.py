"""Under convention B the SLO transform is taken on D directly: failure branches
   contribute e^{-s*0}=1 instead of e^{-s*T_h}. D is the same random variable as before,
   so this must reproduce the current Pr(S >= D_min + T_h)."""
import numpy as np
from math import comb
from model import *

def psiD(dep, Phi, Th, s):
    N = dep.N; ps = dep.ps_cam; F = dep.F(Th)
    A = np.zeros((2*N, 2*N), complex); b = np.zeros(2*N, complex)
    for i in range(N):
        Hs = dep.H(s, i); P = list(Phi[i]); Psi = [i] + P
        phi = 1 - dep.P0[i] - sum(ps[j]*dep.Pd[i, j] for j in P)
        beta = 1 - sum(ps[j]*dep.Pg[i, j]*F[i, j] for j in Psi)
        for j in P:
            A[i, j] = Hs * ps[j] * dep.Pd[i, j]
        for j in Psi:
            A[N+i, j] = ps[j] * dep.Pg[i, j] * dep.Gamma(s, Th, i, j)
        A[i, N+i] = Hs * dep.P0[i]
        b[i]   = Hs * phi          # continuity stops at the sighting: no e^{-s T_h}
        b[N+i] = beta              # likewise
    return np.linalg.solve(np.eye(2*N) - A, b)[:N]

def ccdfD(dep, Phi, Th, t, A=18.4, n=15, mm=11):
    acc = []
    for k in range(n + mm + 1):
        s = (A + 2j*np.pi*k) / (2*t)
        acc.append(((-1)**k) * ((1 - psiD(dep, Phi, Th, s))/s).real * (0.5 if k == 0 else 1.0))
    cs = np.cumsum(np.array(acc), 0)
    val = sum(comb(mm, j) * 2.0**-mm * cs[n+j] for j in range(mm+1))
    return np.clip(np.exp(A/2)/t * val, 0, 1)

dep = Deployment(TESTBED_XY, TESTBED_P0); Phi = nearest_sets(dep.D, 5)
print(f"{'Th':>5}{'Dmin':>6}{'cam':>5}{'Pr(D>=Dmin) via S-shift':>25}{'via D-transform':>18}{'diff':>10}")
worst = 0
for Th in (10.0, 20.0, 60.0):
    for Dmin in (60.0, 120.0):
        old = Fast(dep, Phi, Th).ccdf(Dmin + Th)     # current: Pr(S >= Dmin + Th)
        new = ccdfD(dep, Phi, Th, Dmin)              # proposed: Pr(D >= Dmin)
        for c in (DENSE, SPARSE):
            d = abs(old[c] - new[c]); worst = max(worst, d)
            print(f"{Th:>5.0f}{Dmin:>6.0f}{c+1:>5}{old[c]:>25.5f}{new[c]:>18.5f}{d:>10.2e}")
print(f"\nlargest discrepancy: {worst:.2e}")
