import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

out_dir = Path("./poly_outputs")
out_dir.mkdir(exist_ok=True)

def gd_poly(coeffs, x0, eta, tol=1e-6, kmax=10000):
    # coeffs: [a_n, a_{n-1}, ..., a_0]
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

def sweep(coeffs, x0, etas):
    recs=[]
    for e in etas:
        xs,fs,ok=gd_poly(coeffs,x0,e)
        recs.append({"eta":e,"iters":len(xs)-1 if ok else None,"conv":ok})
    return pd.DataFrame(recs)

etas = np.linspace(0.001, 1.0, 50)

# degree 1: f(x)=2x+5
df1 = sweep([2,5], x0=10, etas=etas)
df1.to_csv(out_dir/"linear.csv",index=False)

# degree 2: f(x)=2x^2 -4x
df2 = sweep([2,-4,0], x0=10, etas=etas)
df2.to_csv(out_dir/"quadratic.csv",index=False)

# degree 3: f(x)=x^3 -3x^2 + 2x
df3 = sweep([1,-3,2,0], x0=5, etas=etas)
df3.to_csv(out_dir/"cubic.csv",index=False)

# degree 4: f(x)=x^4 -4x^2 + x
df4 = sweep([1,0,-4,1,0], x0=5, etas=etas)
df4.to_csv(out_dir/"quartic.csv",index=False)

# quick plot example for quadratic
plt.plot(df2[df2.conv]["eta"], df2[df2.conv]["iters"])
plt.xlabel("eta"); plt.ylabel("iterations"); plt.title("Quadratic: iterations vs eta")
plt.savefig(out_dir/"quadratic_iters.png"); plt.close()