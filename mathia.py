import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

out_dir = Path("./gd_outputs")
out_dir.mkdir(exist_ok=True)

def gd_1d(a, b, c, x0, eta, tol=1e-6, kmax=500):
    grad = lambda x: 2*a*x + b
    f = lambda x: a*float(x)*float(x) + b*float(x) + c
    xs, fs = [x0], [f(x0)]
    for _ in range(kmax):
        x_new = xs[-1] - eta * grad(xs[-1])
        if abs(x_new) > 1e6:  # runaway -> divergence
            return np.array(xs), np.array(fs), False
        xs.append(x_new)
        fs.append(f(x_new))
        if abs(xs[-1] - xs[-2]) < tol:
            return np.array(xs), np.array(fs), True
    return np.array(xs), np.array(fs), False


def gd_2d(m, M, x0, eta, tol=1e-6, kmax=2000):
    Q = np.array([[m,0],[0,M]])
    grad = lambda v: Q @ v
    f = lambda v: 0.5*(m*v[0]**2 + M*v[1]**2)
    xs, fs = [np.array(x0)], [f(x0)]
    for _ in range(kmax):
        v_new = xs[-1] - eta*grad(xs[-1])
        xs.append(v_new)
        fs.append(f(v_new))
        if np.linalg.norm(xs[-1]-xs[-2]) < tol:
            return np.array(xs), np.array(fs), True
    return np.array(xs), np.array(fs), False

def exp_1d(a=2.0, b=-4.0, c=0.0, x0=10.0, etas=None):
    if etas is None: etas = np.linspace(0.02,1.2,60)
    L = 2*a
    recs=[]
    for eta in etas:
        xs,fs,ok = gd_1d(a,b,c,x0,eta)
        recs.append({"eta":eta,"iters":len(xs)-1 if ok else None,"conv":ok,"stable":eta<2/L})
    return pd.DataFrame(recs)

def exp_2d(m=1.0, kappas=(2,10,100), x0=np.array([5.0,-5.0])):
    recs=[]
    for k in kappas:
        M=m*k
        eta=1/(1.5*M)
        xs,fs,ok=gd_2d(m,M,x0,eta)
        recs.append({"kappa":k,"iters":len(xs)-1 if ok else None,"conv":ok,"eta":eta})
    return pd.DataFrame(recs)

df1=exp_1d()
df2=exp_2d()

df1.to_csv(out_dir/"eta_sweep.csv",index=False)
df2.to_csv(out_dir/"kappa_sweep.csv",index=False)

plt.plot(df1[df1.conv]["eta"],df1[df1.conv]["iters"])
plt.xlabel("eta");plt.ylabel("iterations");plt.title("1D: iterations vs eta")
plt.savefig(out_dir/"chart_eta_iters.png");plt.close()

etas_show=[0.05,0.25,0.6,1.0]
a,b,c,x0=2.0,-4.0,0.0,10.0
plt.figure()
for e in etas_show:
    xs,fs,ok=gd_1d(a,b,c,x0,e)
    x_star=-b/(2*a)
    errs=np.abs(xs-x_star);errs=np.clip(errs,1e-16,None)
    plt.plot(np.arange(len(errs)),np.log10(errs),label=f"eta={e}")
plt.xlabel("iteration");plt.ylabel("log10|x-x*|");plt.title("1D error decay")
plt.legend();plt.savefig(out_dir/"chart_eta_decay.png");plt.close()

plt.plot(df2[df2.conv]["kappa"],df2[df2.conv]["iters"])
plt.xlabel("kappa");plt.ylabel("iterations");plt.title("2D: iterations vs kappa")
plt.savefig(out_dir/"chart_kappa_iters.png");plt.close()

plt.figure()
for k in [2,10,100]:
    M=1.0*k
    eta=1/(1.5*M)
    xs,fs,ok=gd_2d(1.0,M,np.array([5.0,-5.0]),eta)
    errs=np.linalg.norm(xs,axis=1);errs=np.clip(errs,1e-16,None)
    plt.plot(np.arange(len(errs)),np.log10(errs),label=f"kappa={k}")
plt.xlabel("iteration");plt.ylabel("log10||x-x*||");plt.title("2D error decay")
plt.legend();plt.savefig(out_dir/"chart_kappa_decay.png");plt.close()
