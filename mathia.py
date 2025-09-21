import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

out_dir = Path("./gd_outputs")
out_dir.mkdir(exist_ok=True)

def gd(Q, x0, eta, tol=1e-6, kmax=5000):
    xs=[x0]; fs=[0.5*x0.T@Q@x0]
    for _ in range(kmax):
        grad = Q @ xs[-1]
        x_new = xs[-1] - eta*grad
        if np.linalg.norm(x_new-xs[-1])<tol:
            xs.append(x_new); fs.append(0.5*x_new.T@Q@x_new)
            return np.array(xs),np.array(fs),True
        if np.linalg.norm(x_new)>1e8:
            return np.array(xs),np.array(fs),False
        xs.append(x_new); fs.append(0.5*x_new.T@Q@x_new)
    return np.array(xs),np.array(fs),False

def run_exp(dim, diag_vals, x0=None):
    if x0 is None: x0=np.ones(dim)*5
    recs=[]
    for diag in diag_vals:
        Q=np.diag(diag)
        L=max(diag); eta=1/(1.5*L)
        xs,fs,ok=gd(Q,x0,eta)
        kappa=max(diag)/min(diag)
        recs.append({"dim":dim,"diag":diag,"kappa":kappa,
                     "iters":len(xs)-1 if ok else None,
                     "conv":ok,"eta":eta})
    return pd.DataFrame(recs)

# 1D
etas=np.linspace(0.02,1.2,60)
df1=[]
for e in etas:
    Q=np.array([[4.0]])
    xs,fs,ok=gd(Q,np.array([10.0]),e)
    df1.append({"eta":e,"iters":len(xs)-1 if ok else None,"conv":ok,"stable":e<0.5})
df1=pd.DataFrame(df1); df1.to_csv(out_dir/"sweep_1d.csv",index=False)

plt.plot(df1[df1.conv]["eta"],df1[df1.conv]["iters"])
plt.xlabel("eta");plt.ylabel("iterations");plt.title("1D: iterations vs eta")
plt.savefig(out_dir/"chart_1d_iters.png");plt.close()

# 2D, 3D, 4D
df2=run_exp(2,[(1,2),(1,10),(1,100)])
df3=run_exp(3,[(1,5,20),(1,3,5),(1,10,50)])
df4=run_exp(4,[(1,2,5,10),(1,3,7,15),(1,10,20,40)])

df2.to_csv(out_dir/"sweep_2d.csv",index=False)
df3.to_csv(out_dir/"sweep_3d.csv",index=False)
df4.to_csv(out_dir/"sweep_4d.csv",index=False)

for df,name in [(df2,"2d"),(df3,"3d"),(df4,"4d")]:
    plt.plot(df["kappa"],df["iters"],'o-')
    plt.xlabel("kappa");plt.ylabel("iterations")
    plt.title(f"{name.upper()}: iterations vs kappa")
    plt.savefig(out_dir/f"chart_{name}_iters.png");plt.close()