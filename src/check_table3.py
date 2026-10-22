"""Independently re-derive every row of Table III from its verbal definition and check
   (a) each panel sums to one, (b) the derived quantities phi_i, g_ij, beta_i, m_i agree,
   (c) the embedded chain rows (Q plus absorption) sum to one."""
import numpy as np
from model import *

def check(dep, Phi, Th, tag):
    N = dep.N; ps = dep.ps_cam; F = dep.F(Th); tau = dep.tau(Th)
    Q, r, aux = build(dep, Phi, Th)
    Qd = np.asarray(Q.todense())
    errs = {k: 0.0 for k in
            ("D sums to 1","S sums to 1","phi","g","beta","m","chain rows","L reward","K reward")}
    for i in range(N):
        P = list(Phi[i]); Psi = [i] + P
        notPsi = [j for j in range(N) if j not in Psi]
        notPhi = [j for j in range(N) if j not in P and j != i]
        Pdef = 1 - dep.Pg[i].sum()                      # P_{0^(i),empty}

        # ---- D panel, each row straight from its verbal definition ----
        D1 = ps[Psi and P].sum() if False else sum(ps[j]*dep.Pd[i,j] for j in P)
        D2 = dep.P0[i]
        D3 = sum(dep.Pd[i,j] for j in notPhi) + sum((1-ps[j])*dep.Pd[i,j] for j in P)
        errs["D sums to 1"] = max(errs["D sums to 1"], abs(D1+D2+D3 - 1))
        errs["phi"] = max(errs["phi"], abs(D3 - aux["phi"][i]))

        # ---- S panel ----
        S1 = sum(ps[j]*dep.Pg[i,j]*F[i,j] for j in Psi)
        S2 = sum((1-ps[j])*dep.Pg[i,j]*F[i,j] for j in Psi)
        S3 = sum(dep.Pg[i,j]*(1-F[i,j]) for j in Psi)
        S4 = sum(dep.Pg[i,j] for j in notPsi)
        S5 = Pdef
        errs["S sums to 1"] = max(errs["S sums to 1"], abs(S1+S2+S3+S4+S5 - 1))
        errs["beta"] = max(errs["beta"], abs((S2+S3+S4+S5) - aux["beta"][i]))
        for j in Psi:   # per-destination success probability
            errs["g"] = max(errs["g"], abs(ps[j]*dep.Pg[i,j]*F[i,j] - Qd[N+i, j]))

        # ---- expected search duration from the case durations ----
        m = sum(ps[j]*dep.Pg[i,j]*tau[i,j] for j in Psi) + (S2+S3+S4+S5)*Th
        errs["m"] = max(errs["m"], abs(m - aux["m"][i]))

        # ---- rewards implied by the table (streams = 1+|Phi_i|) ----
        nf = 1 + sum(1 for j in P if ps[j] > 0)   # a failed camera does not stream (Sec. IV-C)
        errs["L reward"] = max(errs["L reward"], abs(D3*Th - r["L"][i]), abs(m - r["L"][N+i]))
        errs["K reward"] = max(errs["K reward"],
                               abs(dep.rc*dep.mean_soj[i] + D3*nf*dep.rc*Th - r["K"][i]),
                               abs(nf*dep.rc*m - r["K"][N+i]))

        # ---- embedded chain: transient row + absorption = 1 ----
        errs["chain rows"] = max(errs["chain rows"], abs(Qd[i].sum() + D3 - 1),
                                 abs(Qd[N+i].sum() + (S2+S3+S4+S5) - 1))
    worst = max(errs.values())
    print(f"{tag:<46} max abs error {worst:.2e}   " +
          ("OK" if worst < 1e-9 else "FAIL " + str({k: f'{v:.1e}' for k, v in errs.items() if v > 1e-9})))
    return worst

worst = 0.0
rng = np.random.default_rng(0)
# fixed testbed, wide parameter sweep
for k in (0, 1, 3, 5, 9):
    for Th in (5.0, 20.0, 60.0, 200.0):
        for psv in (0.7, 0.95, 1.0):
            for pb in (0.0, 0.15, 0.4):
                for ex in (0.0, 0.1):
                    for gap in ("exp", "lognorm"):
                        dep = Deployment(TESTBED_XY, TESTBED_P0, ps=psv, p_back=pb,
                                         exit_p=ex, gap=gap)
                        worst = max(worst, check(dep, nearest_sets(dep.D, k),
                                                 Th, f"testbed k={k} Th={Th} ps={psv} pb={pb} exit={ex} {gap}")
                                    if False else 0)
# quieter: aggregate instead of printing 720 lines
import io, contextlib
worst = 0.0; n = 0
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for k in (0, 1, 3, 5, 9):
        for Th in (5.0, 20.0, 60.0, 200.0):
            for psv in (0.7, 0.95, 1.0):
                for pb in (0.0, 0.15, 0.4):
                    for ex in (0.0, 0.1):
                        for gap in ("exp", "lognorm"):
                            dep = Deployment(TESTBED_XY, TESTBED_P0, ps=psv, p_back=pb,
                                             exit_p=ex, gap=gap)
                            worst = max(worst, check(dep, nearest_sets(dep.D, k), Th, "x")); n += 1
    # random deployments, heterogeneous per-camera candidate sizes and failed cameras
    for seed in range(30):
        r2 = np.random.default_rng(seed)
        dep = random_deployment(12, (0.2, 0.8), r2, ps=float(r2.uniform(0.6, 1.0)),
                                p_back=float(r2.uniform(0, 0.4)), exit_p=float(r2.uniform(0, 0.15)))
        dep.ps_cam[r2.integers(0, 12, 2)] = 0.0          # failed cameras
        ks = r2.integers(0, 11, 12)
        worst = max(worst, check(dep, nearest_sets(dep.D, ks), float(r2.uniform(5, 150)), "y")); n += 1
print(f"configurations checked: {n}")
print(f"worst absolute error across all checks: {worst:.3e}")
print("VERDICT:", "all rows consistent, both panels sum to 1" if worst < 1e-9 else "MISMATCH")
