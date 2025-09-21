import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

out_dir = Path("./poly_outputs")
out_dir.mkdir(exist_ok=True)

def gd_poly(coeffs, x0, eta, tol=1e-6, kmax=10000):
    poly = np.poly1d(coeffs)
    grad = poly.deriv()
    xs, fs = [x0], [poly(x0)]
    for _ in range(kmax):
        x_new = xs[-1] - eta * grad(xs[-1])
        if abs(x_new - xs[-1]) < tol:
            xs.append(x_new); fs.append(poly(x_new))
            return np.array(xs), np.array(fs), True
        if abs(x_new) > 1e6:
            return np.array(xs), np.array(fs), False
        xs.append(x_new); fs.append(poly(x_new))
    return np.array(xs), np.array(fs), False

def sweep(coeffs, x0, etas, name):
    recs=[]
    for e in etas:
        xs,fs,ok=gd_poly(coeffs,x0,e)
        recs.append({"eta":e,"iters":len(xs)-1 if ok else None,"conv":ok})
    df=pd.DataFrame(recs)
    df.to_csv(out_dir/f"{name}.csv",index=False)

    # chart 1: iterations vs eta
    plt.plot(df[df.conv]["eta"],df[df.conv]["iters"])
    plt.xlabel("eta"); plt.ylabel("iterations")
    plt.title(f"{name.capitalize()}: iterations vs eta")
    plt.savefig(out_dir/f"{name}_iters.png"); plt.close()

    # chart 2: error decay curves (few selected etas)
    etas_show=[0.01,0.05,0.1,0.2]
    plt.figure()
    for e in etas_show:
        xs,fs,ok=gd_poly(coeffs,x0,e)
        if not ok: continue
        errs=np.abs(xs-xs[-1]); errs=np.clip(errs,1e-16,None)
        plt.plot(np.arange(len(errs)),np.log10(errs),label=f"eta={e}")
    plt.xlabel("iteration"); plt.ylabel("log10|x-x*|")
    plt.title(f"{name.capitalize()}: error decay")
    plt.legend()
    plt.savefig(out_dir/f"{name}_decay.png"); plt.close()

# sweep settings
etas = np.linspace(0.001,0.5,50)

# degree 1: f(x)=2x+5
sweep([2,5], x0=10, etas=etas, name="linear")

# degree 2: f(x)=2x^2 -4x
sweep([2,-4,0], x0=10, etas=etas, name="quadratic")

# degree 3: f(x)=x^3 -3x^2 + 2x
sweep([1,-3,2,0], x0=5, etas=etas, name="cubic")

# degree 4: f(x)=x^4 -4x^2 + x
sweep([1,0,-4,1,0], x0=5, etas=etas, name="quartic")

# degree 5: f(x)=x^5 -5x^3 + 4x
sweep([1,0,-5,0,4,0], x0=3, etas=etas, name="quintic")