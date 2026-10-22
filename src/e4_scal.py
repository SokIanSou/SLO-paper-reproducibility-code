from model import *; from common import *; import json, time
Ns=[10,20,40,80,160,320,640,1280]; out={}
for N in Ns:
    tm, ts = [], []
    for seed in range(3):
        dep = random_deployment(N, (0.3,0.7), np.random.default_rng(seed))
        Phi = nearest_sets(dep.D, 5)
        a=[];b=[]
        for rep in range(3):
            t0=time.perf_counter(); metrics(dep,Phi,20.0); t1=time.perf_counter()
            Fast(dep,Phi,20.0).ccdf(DMIN); t2=time.perf_counter()
            a.append(t1-t0); b.append(t2-t1)
        tm.append(np.median(a)); ts.append(np.median(b))
    out[N]=(float(np.mean(tm))*1e3, float(np.mean(ts))*1e3); print(N, np.round(out[N],2))
json.dump(out, open("res/e4.json","w"), indent=1)
x=np.array(Ns); y1=np.array([out[n][0] for n in Ns]); y2=np.array([out[n][1] for n in Ns])
sl1=np.polyfit(np.log(x[3:]),np.log(y1[3:]),1)[0]; sl2=np.polyfit(np.log(x[3:]),np.log(y2[3:]),1)[0]
print("slopes", sl1, sl2); json.dump({"times":out,"slope_metrics":sl1,"slope_slo":sl2}, open("res/e4.json","w"), indent=1)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.loglog(x,y1,"o-",color=CD,ms=3,label=f"expected metrics $S,C,L,K,V$ (slope {sl1:.2f})")
ax.loglog(x,y2,"s-",color=CS,ms=3,label=f"SLO probability $\\Pr(D\\geq D_{{\\min}})$ (slope {sl2:.2f})")
ax.set_xlabel("number of cameras $N$"); ax.set_ylabel("evaluation time (ms)"); ax.legend(loc="upper left")
save(fig,"e4_scalability")
