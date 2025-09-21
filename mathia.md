# Mathematics IA  
**Title:** *An Investigation into the Convergence of Gradient Descent in Quadratic Functions with Varying Learning Rates and Condition Numbers*  

---

## Introduction  

Optimization lies at the heart of modern mathematics and its applications. Whether in physics, economics, or machine learning, we often want to minimize a cost or maximize a reward. Among the algorithms invented to do this, **gradient descent** stands out because of its simplicity and effectiveness.  

While experimenting with small machine learning projects, I noticed that gradient descent sometimes converged quickly, sometimes very slowly, and sometimes exploded into divergence. This puzzled me, and I wanted to explore the mathematics behind this phenomenon.  

**Research Question**  
*How does the convergence rate of gradient descent in quadratic functions depend on the learning rate and the condition number of the Hessian matrix?*  

This problem links directly to the **IB Mathematics AA HL syllabus** under Topic 5 (Calculus: optimization and rates of change). It also connects to artificial intelligence, where training algorithms rely heavily on gradient descent.  

---

## Mathematical Background  

Gradient descent iterates according to  

$$
x_{k+1} = x_k - \eta \nabla f(x_k),
$$  

where \( \eta \) is the learning rate.  

For a quadratic function  

$$
f(x) = \tfrac{1}{2} x^\top Q x - p^\top x + r,
$$  

with \( Q \) symmetric and positive definite, the unique minimizer is  

$$
x^\ast = Q^{-1} p.
$$  

Define the error \( e_k = x_k - x^\ast \). Then  

$$
e_{k+1} = (I - \eta Q)e_k.
$$  

The eigenvalues of the iteration matrix are \( 1 - \eta \lambda_i \), where \( \lambda_i \) are eigenvalues of \( Q \).  

- **Stability condition:**  

$$
0 < \eta < \frac{2}{\lambda_{\max}}.
$$  

- **Convergence factor:**  

$$
\rho = \max_i |1 - \eta \lambda_i|.
$$  

Thus  

$$
\|e_k\| \leq \rho^k \|e_0\|.
$$  

- **Condition number:**  

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}.
$$  

When \( \kappa \) is small, the level sets are circular and convergence is rapid. When \( \kappa \) is large, the function looks like a narrow valley and gradient descent zig-zags, converging slowly.  

---

## Methodology  

### One-Dimensional Quadratic  

I analyzed  

$$
f(x) = 2x^2 - 4x,
$$  

starting at \( x_0 = 10 \).  
Here \( Q = 4 \), so stability requires \( \eta < 0.5 \).  

- Learning rate sweep: \( \eta \in [0.02, 1.2] \).  
- Stopping condition: \( |x_{k+1} - x_k| < 10^{-6} \) or 500 iterations.  

### Two-Dimensional Quadratic  

I extended to  

$$
f(x,y) = \tfrac{1}{2}(m x^2 + M y^2),
$$  

with Hessian \( Q = \mathrm{diag}(m,M) \).  

- Fixed \( m = 1 \).  
- Tested \( M = 2, 10, 100 \) (so \( \kappa = 2, 10, 100 \)).  
- Starting point: \( (5, -5) \).  
- Learning rate: \( \eta = \tfrac{1}{1.5 M} \).  

### Three-Dimensional Quadratic  

Finally, I considered  

$$
f(x,y,z) = \tfrac{1}{2}(l_1 x^2 + l_2 y^2 + l_3 z^2),
$$  

with Hessian \( Q = \mathrm{diag}(l_1,l_2,l_3) \).  

- Tested cases: \( (1,5,20), (1,3,5), (1,10,50) \).  
- Condition numbers: \( \kappa = 20, 5, 50 \).  
- Learning rate: \( \eta = \tfrac{1}{1.5 \max(l_i)} \).  
- Starting point: \( (5,-5,5) \).  

### Data Collection  

For each run I recorded:  

- Learning rate \( \eta \).  
- Number of iterations until convergence.  
- Whether it converged.  
- Whether it matched theoretical stability.  

Data were exported to CSV and plotted.  

---

## Results  

### One-Dimensional  

| η   | iterations | converged | stable |
|-----|------------|-----------|--------|
| 0.02 | 163 | True | True |
| 0.04 | 83  | True | True |
| 0.06 | 55  | True | True |
| 0.08 | 40  | True | True |
| 0.10 | 31  | True | True |
| 0.12 | 25  | True | True |
| 0.14 | 20  | True | True |
| 0.16 | 17  | True | True |
| 0.18 | 14  | True | True |
| 0.20 | 11  | True | True |
| 0.22 | 9   | True | True |
| 0.24 | 6   | True | True |
| 0.26 | 6   | True | True |
| 0.28 | 9   | True | True |
| 0.30 | 12  | True | True |
| 0.32 | 14  | True | True |
| 0.34 | 17  | True | True |
| 0.36 | 21  | True | True |
| 0.38 | 27  | True | True |
| 0.40 | 34  | True | True |
| 0.42 | 44  | True | True |
| 0.44 | 62  | True | True |
| 0.46 | 97  | True | True |
| 0.48 | 201 | True | True |
| 0.50+ | – | False | False |

**Figure 1:** Iterations vs learning rate (1D).  
**Figure 2:** Error decay curves for selected \( \eta \).  

---

### Two-Dimensional  

| κ   | iterations | converged | η used |
|-----|------------|-----------|--------|
| 2   | 37  | True | 0.3333 |
| 10  | 186 | True | 0.0667 |
| 100 | 1558| True | 0.0067 |

**Figure 3:** Iterations vs condition number (2D).  
**Figure 4:** Error decay in log-scale (2D).  

---

### Three-Dimensional  

| (l1,l2,l3) | κ   | iterations | converged | η used |
|------------|-----|------------|-----------|--------|
| (1,5,20)   | 20  | 356        | True      | 0.0333 |
| (1,3,5)    | 5   | 95         | True      | 0.1333 |
| (1,10,50)  | 50  | 829        | True      | 0.0133 |

**Figure 5:** Iterations vs condition number (3D).  
**Figure 6:** Error decay curves for 3D trials.  

---

## Analysis  

### Effect of Learning Rate  

- Stability bound: \( \eta < 0.5 \).  
- Data confirms: convergence until \( \eta = 0.48 \); divergence at \( \eta \geq 0.5 \).  
- Fastest convergence around \( \eta \approx 0.24\)–0.26 with only 6 iterations.  

### Effect of Condition Number  

- In 2D: iterations grow from 37 (\(\kappa=2\)) to 1558 (\(\kappa=100\)).  
- In 3D: moderate κ = 5 still needed 95 iterations; large κ = 50 took 829 iterations.  
- Matches theory: higher κ stretches the valley and forces zig-zagging.  

### Comparison with Theory  

- Predicted optimal step size for 1D: \( \eta_{\text{best}}=0.25\).  
- Experimental fastest rate: \( \eta=0.24\)–0.26. Perfect agreement.  

### Dimensionality  

Going from 2D to 3D did not change the mathematics (still eigenvalues determine convergence). But more dimensions allow intermediate κ values (5, 20, 50) that show clearly how iteration count scales almost linearly with κ.  

---

## Conclusion  

This investigation shows that:  

- **Learning rate \( \eta \):**  
  - Too small → convergence but slow.  
  - Moderate near \( 2/(\lambda_{\min}+\lambda_{\max}) \) → fastest.  
  - Too large → divergence.  

- **Condition number \( \kappa \):**  
  - Small → rapid convergence.  
  - Large → extremely slow even if stable.  

Both the theory of iteration matrices and the experiments in 1D, 2D, and 3D matched perfectly.  

---

## Reflection  

This project gave me both practical coding experience and deeper insight into mathematical optimization. I learned how eigenvalues—abstract objects from linear algebra—directly control convergence speed in an algorithm that powers modern AI.  

It also showed me how small-scale numerical experiments complement theory: instead of just trusting the inequality \( \eta < 2/\lambda_{\max} \), I watched the error curve plunge or blow up depending on η.  

If extended, I would study more advanced algorithms like **momentum**, **Nesterov acceleration**, and **Adam**, to see how they tame ill-conditioning where plain gradient descent struggles.  

---

## Bibliography  

- Nocedal, J. & Wright, S. (2006). *Numerical Optimization*. Springer.  
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.  
- International Baccalaureate Organization. *Mathematics: Analysis and Approaches HL Guide*.  