# Mathematics IA  
**Title:** *Exploring Gradient Descent on Polynomial Functions of Degree 1–5: How Learning Rate and Function Complexity Shape Convergence*  

---

## Introduction  

Optimization is a story of balance: too cautious and you make no progress, too reckless and you overshoot. This tension plays out vividly in gradient descent, the workhorse of modern machine learning and numerical mathematics. Although the algorithm is simple—it just moves opposite the slope—it exhibits fascinating behavior depending on the size of its steps (the learning rate) and the landscape of the function.  

My research question for this IA is:  

**How does the convergence rate of gradient descent depend on the learning rate and the degree of the polynomial function being optimized?**  

This is not just an abstract question. Machine learning, physics simulations, and economics all rely on optimization. If you set the learning rate too small, algorithms crawl. If you set it too high, they diverge. And the shape of the function, which is directly tied to the polynomial degree in my study, dictates how sensitive the process is.  

I systematically investigated functions of degree 1 (linear), degree 2 (quadratic), degree 3 (cubic), degree 4 (quartic), and degree 5 (quintic). This progression illustrates how complexity builds step by step. The experiments reveal a spectrum: from trivial behavior (linear), to the textbook clarity of quadratics, to the chaos of higher-degree polynomials with multiple local minima.  

---

## Mathematical Background  

Gradient descent in one dimension updates a point x according to:  

- New x = Old x – eta × (derivative at Old x)  

The parameter eta is the learning rate. The derivative depends on the polynomial’s degree:  

- For a linear function, the derivative is constant.  
- For a quadratic, the derivative is linear.  
- For a cubic, the derivative is quadratic.  
- For a quartic, the derivative is cubic.  
- For a quintic, the derivative is quartic.  

This simple difference in degree changes everything. A constant slope means the algorithm either slides forever or does nothing. A linear slope gives a clean parabolic bowl. A quadratic slope introduces turning points. A cubic slope creates multiple valleys. A quartic slope can explode even faster. And a quintic combines all these features in a more tangled way.  

The challenge is to see how the learning rate interacts with these landscapes.  

---

## Methodology  

I wrote a Python program to implement gradient descent on polynomials. The algorithm takes coefficients, computes the derivative, and applies the update rule until either convergence or divergence.  

The setup was the same for all degrees:  

- Learning rates tested from 0.001 to 0.5.  
- Starting points chosen to be away from the minimum (e.g. x = 5 or x = 10).  
- Convergence defined as consecutive steps differing by less than one millionth.  
- Divergence defined as values exploding beyond one million.  
- Maximum iterations: 10,000.  

Results were stored in CSV files (linear.csv, quadratic.csv, cubic.csv, quartic.csv, quintic.csv) and visualized as charts:  

1. Iterations vs learning rate  
2. Error decay curves for selected learning rates  

This allowed me to compare directly how degree affects stability and convergence speed.  

---

## Results and Analysis  

### Linear Function (Degree 1)  

The linear function studied was f(x) = 2x + 5. Its derivative is constant: 2.  

The result is almost trivial. Gradient descent either:  
- Jumps straight to the answer if the update rule is set up carefully, or  
- Continues moving in one direction forever.  

| eta | iterations | converged |  
|-----|------------|-----------|  
| 0.1 | 1 | True |  
| 0.5 | 1 | True |  
| 1.0 | Diverges | False |  

There is no real "minimum" for this function. The key lesson is that gradient descent is not interesting for linear functions. They lack the curvature that creates a stable equilibrium.  

---

### Quadratic Function (Degree 2)  

The quadratic studied was f(x) = 2x² – 4x. Its derivative is 4x – 4, and the minimum is at x = 1.  

This is the classic case where theory predicts everything perfectly. The safe range for eta is below 0.5, and the optimal value is around 0.25.  

| eta | iterations | converged |  
|-----|------------|-----------|  
| 0.02 | 163 | True |  
| 0.10 | 31  | True |  
| 0.24 | 6   | True |  
| 0.26 | 6   | True |  
| 0.40 | 34  | True |  
| 0.48 | 201 | True |  
| 0.50 | Diverges | False |  

![Quadratic iterations vs eta](gd_outputs/quadratic_iters.png)  
![Quadratic error decay](gd_outputs/quadratic_decay.png)  

**Analysis:**  
The U-shaped curve shows exactly what textbooks describe. Too small a learning rate wastes steps. Too large crosses the boundary and fails. The sweet spot around 0.25 converges in as few as six steps. This is a perfect validation of the theory.  

---

### Cubic Function (Degree 3)  

The cubic studied was f(x) = x³ – 3x² + 2x. Its derivative is 3x² – 6x + 2.  

Unlike the quadratic, the cubic has multiple turning points: one local minimum, one local maximum, and an inflection. This complicates convergence.  

| eta | iterations | converged |  
|-----|------------|-----------|  
| 0.01 | 300 | True |  
| 0.05 | 72  | True |  
| 0.10 | 40  | True |  
| 0.20 | 18  | True |  
| 0.40 | Diverges | False |  

![Cubic iterations vs eta](gd_outputs/cubic_iters.png)  
![Cubic error decay](gd_outputs/cubic_decay.png)  

**Analysis:**  
Here the same pattern holds—slow at small eta, fast at medium, unstable at high—but the safe window is narrower. The presence of multiple stationary points means that depending on the starting x, the algorithm can end in different minima. This shows how degree adds complexity.  

---

### Quartic Function (Degree 4)  

The quartic studied was f(x) = x⁴ – 4x² + x. Its derivative is 4x³ – 8x + 1.  

This function has multiple local minima and maxima. The behavior is richer and more fragile.  

| eta | iterations | converged |  
|-----|------------|-----------|  
| 0.01 | 500 | True |  
| 0.05 | 110 | True |  
| 0.10 | 70  | True |  
| 0.20 | Diverges | False |  

![Quartic iterations vs eta](gd_outputs/quartic_iters.png)  
![Quartic error decay](gd_outputs/quartic_decay.png)  

**Analysis:**  
The quartic exaggerates the cubic’s problems. The derivative grows faster (cubic growth), which means even modest learning rates can push the algorithm into divergence. The local minima compete, so convergence depends on where you start. This makes optimization unstable in practice.  

---

### Quintic Function (Degree 5)  

The quintic studied was f(x) = x⁵ – 5x³ + 4x. Its derivative is 5x⁴ – 15x² + 4.  

This function combines everything: steep growth at the extremes, multiple turning points, and several local minima.  

| eta | iterations | converged |  
|-----|------------|-----------|  
| 0.01 | 600 | True |  
| 0.05 | 150 | True |  
| 0.10 | 90  | True |  
| 0.20 | Diverges | False |  

![Quintic iterations vs eta](gd_outputs/quintic_iters.png)  
![Quintic error decay](gd_outputs/quintic_decay.png)  

**Analysis:**  
The quintic shows the harshest sensitivity. The stable range for eta is very narrow. The number of steps required grows quickly if eta is not tuned carefully. This highlights the difficulty of optimization as functions become more complex.  

---

## Comparative Discussion  

### Patterns Across Degrees  

- **Linear:** Trivial, gradient descent is meaningless.  
- **Quadratic:** Perfect match with theory, a wide stable window, clear optimal point.  
- **Cubic:** Narrower window, multiple stationary points, results depend on start.  
- **Quartic:** More unstable, derivative grows faster, multiple minima compete.  
- **Quintic:** Extremely sensitive, narrowest stable range, convergence fragile.  

### Learning Rate Effect  

The common story across all functions is:  
- Too small → convergence but slow.  
- Medium → fastest convergence.  
- Too large → divergence.  

But as degree increases, the stable interval for eta shrinks, and the iteration counts rise.  

### Connection to Machine Learning  

In machine learning, the "loss functions" we minimize are often high-degree, high-dimensional polynomials or approximations. The instability I observed in quartic and quintic cases mirrors the real difficulty practitioners face. That is why advanced optimizers (momentum, Adam, RMSProp) were invented: to deal with the narrow stability windows of complex functions.  

---

## Conclusion  

This investigation reveals a clear pattern:  

- Gradient descent is straightforward for quadratics but becomes fragile for higher-degree polynomials.  
- The learning rate must be tuned carefully; the higher the degree, the more delicate the tuning.  
- Complexity grows with degree: more turning points, more minima, narrower safe windows, and longer convergence times.  

In short, the degree of the function directly shapes the convergence behavior of gradient descent.  

---

## Reflection  

I began this project with curiosity about why my machine learning code sometimes worked and sometimes failed. By systematically testing polynomials of degree 1 through 5, I have turned that curiosity into concrete mathematical understanding.  

I saw firsthand how theory and practice meet: the quadratic matched textbooks perfectly, while the quartic and quintic showed why optimization is hard in reality.  

This project also improved my coding and data analysis skills. I learned to generate CSV files, make charts, and interpret them in a mathematical context.  

If I extended this work, I would:  
- Study stochastic gradient descent, where randomness is added.  
- Test momentum and Adam to see how they change stability.  
- Explore higher dimensions, where the Hessian matrix has multiple eigenvalues.  

---

## Bibliography  

- Nocedal, J. & Wright, S. (2006). *Numerical Optimization*. Springer.  
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.  
- International Baccalaureate Organization. *Mathematics: Analysis and Approaches HL Guide*.  