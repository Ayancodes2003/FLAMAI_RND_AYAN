# Scientific Evaluation Notes & Metric Methodology

---

## 1. Why CSV Row-Wise Comparison is Mathematically Invalid
The provided `xy_data.csv` contains $1,500$ point coordinates $(x_i, y_i)$ that are spatial observations of the curve. Programmatic inspection established that these points are **unordered** along the parameter $t$. 

Evaluating an error metric via naive row-by-row subtraction:
$$\mathcal{L}_{\text{naive}} = \frac{1}{N}\sum_{i=1}^N \left( |x_{\text{obs}, i} - x_{\text{pred}, i}| + |y_{\text{obs}, i} - y_{\text{pred}, i}| \right)$$
implicitly assumes that row $i$ corresponds strictly to a predetermined parameter $t_i = 6 + \frac{54(i-1)}{N-1}$. Because consecutive rows in `xy_data.csv` exhibit spatial jumps of up to $52.69$ coordinate units, this row-correspondence assumption is factually incorrect and would produce an arbitrarily large, meaningless error score regardless of curve accuracy.

---

## 2. Uniform Sampling in $t \in [6, 60]$
The assignment defines the evaluation criteria as:
> *"The L1 distance between uniformly sampled points between expected and predicted curve"*

To satisfy this requirement reproducibly:
- We sample $N$ uniformly spaced points across the domain $t \in [6, 60]$:
  $$t_j = 6.0 + j \cdot \frac{54.0}{N - 1}, \quad j = 0, 1, \dots, N-1$$
- The forward parametric curve $\mathbf{r}(t_j; \hat{\theta}, \hat{M}, \hat{X})$ is evaluated at these points to generate the predicted curve discrete point set $\mathcal{C}_{\text{pred}} = \{(x_{\text{pred}, j}, y_{\text{pred}, j})\}_{j=1}^N$.

---

## 3. Exact $L_1$ (Manhattan) Distance Definition
The spatial distance between two points $p = (x_1, y_1)$ and $q = (x_2, y_2)$ is defined strictly using the $L_1$ (Manhattan / Taxicab) norm:
$$d_{L_1}(p, q) = |x_1 - x_2| + |y_1 - y_2|$$
*Note:* We explicitly verify and enforce that $L_1$ is computed, avoiding Euclidean distance approximations ($\sqrt{\Delta x^2 + \Delta y^2}$).

For each observed point $p_i \in \mathcal{P}_{\text{obs}}$, its distance to the predicted curve representation $\mathcal{C}_{\text{pred}}$ is:
$$d(p_i, \mathcal{C}_{\text{pred}}) = \min_{q_j \in \mathcal{C}_{\text{pred}}} \left( |x_{\text{obs}, i} - x_{\text{pred}, j}| + |y_{\text{obs}, i} - y_{\text{pred}, j}| \right)$$

---

## 4. Bidirectional Evaluations (Observed $\to$ Predicted and Predicted $\to$ Observed)
A rigorous curve comparison must evaluate both directions:
1. **Observed $\to$ Predicted ($\mathcal{P}_{\text{obs}} \to \mathcal{C}_{\text{pred}}$):**
   Measures *fidelity*—how closely the reconstructed curve passes near every actual observation point.
2. **Predicted $\to$ Observed ($\mathcal{C}_{\text{pred}} \to \mathcal{P}_{\text{obs}}$):**
   Measures *coverage / spuriousness*—verifies that the predicted curve does not wander into empty regions of space where no data exists.

---

## 5. Symmetric $L_1$ Formulation
The symmetric mean $L_1$ distance is defined as the arithmetic mean of the two directional distances:
$$\overline{L}_{1, \text{sym}} = \frac{1}{2} \left( \frac{1}{N_{\text{obs}}} \sum_{i=1}^{N_{\text{obs}}} d(p_i, \mathcal{C}_{\text{pred}}) + \frac{1}{N_{\text{pred}}} \sum_{j=1}^{N_{\text{pred}}} d(q_j, \mathcal{P}_{\text{obs}}) \right)$$
Additionally, the symmetric maximum distance corresponds to the discrete **Hausdorff-$L_1$ distance**:
$$H_{L_1}(\mathcal{P}_{\text{obs}}, \mathcal{C}_{\text{pred}}) = \max \left( \max_{i} d(p_i, \mathcal{C}_{\text{pred}}), \max_{j} d(q_j, \mathcal{P}_{\text{obs}}) \right)$$

---

## 6. Sampling Resolution Convergence
When a continuous curve $\mathbf{r}(t)$ is represented by $N$ discrete points, the maximum distance between adjacent samples along the curve is $\approx \|\mathbf{r}'(t)\|_1 \frac{\Delta t}{N}$.
Consequently, discrete nearest-neighbor queries introduce a small discretization bias $\mathcal{O}(1/N)$.
To establish that our reported metric is not an artifact of discretization:
- We test sampling resolutions $N \in [500, 1000, 1500, 5000, 10000]$.
- As $N \to \infty$, the discretization bias decays to zero, and the observed-to-predicted distance converges strictly to the continuous point-to-manifold distance ($< 0.002$ coordinate units).

---

## 7. Distinction Between Objective Functions and Final Evaluation Metrics
It is critical to distinguish the mathematical roles of different metrics:

| Metric | Context / Formulation | Purpose |
| :--- | :--- | :--- |
| **Phase 3 Fitting Objective** | $\min_{\theta, M, X} \frac{1}{N}\sum \min_{t \in \text{grid}} \|p_i - \mathbf{r}(t)\|_1$ | Loss function minimized during baseline optimization. |
| **Phase 4 Transverse Residual** | $\frac{1}{N_{\text{valid}}}\sum |v_i - e^{M u_i}\sin(0.3 u_i)|$ | Geometric alignment loss over rotated coordinate frame. |
| **Final Curve Evaluation Metric** | $\overline{L}_{1, \text{sym}}$ over $N$-point uniform samples | Independent post-fit benchmark for curve reconstruction accuracy. |

These metrics serve different roles and are not directly interchangeable. In Part E, both Phase 3 and Phase 4 parameter sets are evaluated on the **exact same final curve evaluation metric** for an unbiased, scientific comparison.

---

## 8. Limitations & Scope of Assignment Specification
Because the take-home problem description does not specify an exact reference evaluation script, our evaluation pipeline represents a **reproducible local surrogate evaluation** strictly consistent with the assignment's stated criteria. All numerical results, sampling tables, and diagnostic plots are generated directly from the repository code without manual intervention.
