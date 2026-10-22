from model import *; from common import *; import json, sys, os
dep = Deployment(TESTBED_XY, TESTBED_P0); N=10; nu=np.full(N,1/N); LAM=0.05
THS=[10.,20.,30.,40.,60.,80.,100.,120.]
order=[list(np.argsort(dep.D[i])[1:]) for i in range(N)]
mk=lambda k:[order[i][:k[i]] for i in range(N)]
def best_uniform(target):
    best=None
    for Th in THS:
        for k in range(N):
            s,l,*_=slo_and_load(dep,nearest_sets(dep.D,k),Th,DMIN,nu,LAM)
            if s>=target:
                if best is None or l<best["load"]: best=dict(load=l,Th=Th,k=[k]*N,slo=s)
                break
    return best
def best_prop(target):
    best=None; w=dep.P0/dep.P0.mean()
    for Th in THS:
        for c in np.arange(0.25,9.01,0.25):
            k=np.clip(np.round(c*w),0,N-1).astype(int)
            s,l,*_=slo_and_load(dep,mk(k),Th,DMIN,nu,LAM)
            if s>=target:
                if best is None or l<best["load"]: best=dict(load=l,Th=Th,k=k.tolist(),slo=s)
                break
    return best
def best_greedy(target):
    best=None
    for Th in THS:
        r=greedy_config(dep,Th,DMIN,target,nu,LAM)
        if r and (best is None or r[2]<best["load"]): best=dict(load=r[2],Th=Th,k=r[0].tolist(),slo=r[1])
    return best
target=float(sys.argv[1]); f="res/e5.json"
res=json.load(open(f)) if os.path.exists(f) else {}
res[str(target)]={"uniform":best_uniform(target),"proportional":best_prop(target),"greedy":best_greedy(target)}
json.dump(res,open(f,"w"),indent=1); print(target, json.dumps(res[str(target)]))
