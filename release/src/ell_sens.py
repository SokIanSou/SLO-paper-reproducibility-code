"""Sensitivity of the Sec. VI-C conclusion to the mobility-locality length ell."""
from model import *
from common import DMIN
import numpy as np, json
N=10; nu=np.full(N,1.0/N); Lam=0.05; target=0.5
T=[10,20,30,40,60,80,100,120]
rows=[]
for ell in (10.0,15.0,20.0,25.0):
    dep=Deployment(TESTBED_XY,TESTBED_P0,ell=ell)
    ordr=[list(np.argsort(dep.D[i])[1:]) for i in range(N)]
    best_u=None
    for Th in T:                                  # best uniform setting
        for k in range(0,N):
            Phi=[ordr[i][:k] for i in range(N)]
            s,l,*_=slo_and_load(dep,Phi,float(Th),DMIN,nu,Lam)
            if s>=target and (best_u is None or l<best_u[0]): best_u=(l,k,Th)
    best_g=None
    for Th in T:                                  # model-driven
        r=greedy_config(dep,float(Th),DMIN,target,nu,Lam)
        if r is None: continue
        k,s,l=r
        if s>=target and (best_g is None or l<best_g[0]): best_g=(l,Th)
    red=100*(best_u[0]-best_g[0])/best_u[0]
    rows.append(dict(ell=ell,uniform=best_u[0],k_u=best_u[1],Th_u=best_u[2],
                     model=best_g[0],Th_g=best_g[1],reduction=red))
    print("ell=%4.0f m  uniform %6.1f Mbps (k=%d, Th=%d)   model-driven %6.1f (Th=%d)   reduction %4.1f%%"
          %(ell,best_u[0],best_u[1],best_u[2],best_g[0],best_g[1],red))
json.dump(rows,open("res/ell_sens.json","w"),indent=1)
