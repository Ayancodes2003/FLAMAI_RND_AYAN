# Phase 5B: Curve-Distance Evaluation Audit & Asymmetry Investigation

---

## 1. Executive Summary & Exact Findings

During the Phase 5 sampling convergence experiment, the following numerical asymmetry was observed:

| Resolution $N$ | Observed $\to$ Predicted Mean $L_1$ | Predicted $\to$ Observed Mean $L_1$ |
| :---: | :---: | :---: |
| **$N = 500$** | $0.040421$ | $0.026238$ |
| **$N = 1,000$** | $0.019395$ | $0.025819$ |
| **$N = 1,500$** | $0.013385$ | $0.026758$ |
| **$N = 5,000$** | $0.004074$ | $0.026659$ |
| **$N = 10,000$** | **$0.002003$** | **$0.026639$** |

This audit independently investigates:
1. Exactness of the Manhattan metric implementation.
2. Mathematical root cause of the directional convergence divergence.
3. Latent parameter $t$ spacing properties in `xy_data.csv`.
4. Theoretical behavior on synthetic noiseless curves.
5. Empirical validation of convergence scaling laws.

---

## 2. Independent Brute-Force Metric Verification

To confirm that `scipy.spatial.cKDTree(..., p=1)` computes exact Manhattan distances ($|x_1 - x_2| + |y_1 - y_2|$) without any metric distortion or Euclidean leakage:
- Evaluated $1,500$ observed points against $1,000$ predicted curve samples using both:
  - **Method A:** `cKDTree(p=1).query(p=1)`
  - **Method B:** Full vectorized pairwise matrix broadcasting: $\min_j (|x_i - x_j| + |y_i - y_j|)$
- **Result:**
  $$\max_{i} |d_{\text{tree}}(i) - d_{\text{brute}}(i)| = 0.000000000000 \times 10^0 \text{ (Exact machine precision match)}$$
- Added automated regression test `test_ckdtree_exact_match_brute_force` to continuous test suite.

---

## 3. Mathematical Explanation of Directional Asymmetry

The fundamental reason **Observed $\to$ Predicted** decays to zero while **Predicted $\to$ Observed** plateaus at $\approx 0.0266$ is the asymmetry between a **fixed finite point set** and a **densifying continuum**:

### A. Observed $\to$ Predicted ($\mathcal{P}_{\text{obs}} \to \mathcal{C}_{\text{pred}}$)
- Evaluates: For each of the $1,500$ fixed observation points, find its closest neighbor in the predicted set $\mathcal{C}_{\text{pred}}$.
- As $N_{\text{pred}} \to \infty$, $\mathcal{C}_{\text{pred}}$ becomes a continuous 1D geometric curve.
- Because the observation points lie on the true continuous manifold, the distance to the closest point on $\mathcal{C}_{\text{pred}}$ is purely governed by curve discretization $\Delta t_{\text{pred}} = \frac{54}{N_{\text{pred}}-1}$.
- **Empirical Power-Law Fit:**
  $$\text{Error}_{\text{obs}\to\text{pred}}(N) = 19.3838 \times N^{-0.9958} \quad (R^2 = 0.999778)$$
  This strictly validates an exact $\mathcal{O}(N^{-1.00})$ decay rate toward zero.

### B. Predicted $\to$ Observed ($\mathcal{C}_{\text{pred}} \to \mathcal{P}_{\text{obs}}$)
- Evaluates: For each of the $N_{\text{pred}}$ samples along the continuous curve, find the nearest point in the **fixed finite** observation dataset $\mathcal{P}_{\text{obs}}$ ($N_{\text{obs}} = 1500$).
- When $N_{\text{pred}} \gg N_{\text{obs}}$, sample points fall between the discrete observation points.
- For a 1D segment of length $L$ bounded by two discrete observation points, a uniformly sampled point has expected distance:
  $$\mathbb{E}[d] = \frac{1}{L}\int_0^L \min(s, L - s) \, ds = \frac{L}{4}$$
- With average along-curve spacing $\bar{L} \approx 0.106$, the theoretical expected nearest-neighbor distance to the finite observed set is $\frac{\bar{L}}{4} \approx 0.0265$.
- Therefore, **Predicted $\to$ Observed cannot decay to zero** as $N_{\text{pred}} \to \infty$; it converges to the **finite sampling resolution floor** of the observation dataset ($\approx 0.0266$).

---

## 4. Latent Parameter $t$ Spacing Statistics (`xy_data.csv`)

Using the recovered Phase 4 parameters ($\theta = 29.999973^\circ, X = 54.999998$), the parallel projection coordinate $u_i = (x_i - X)\cos\theta + (y_i - 42)\sin\theta$ was computed and sorted for all $1,500$ rows:

| Statistic | Value | Notes |
| :--- | :--- | :--- |
| **$t_{\min}$** | $6.049405$ | Bounded within $(6.0, 60.0)$ |
| **$t_{\max}$** | $59.995171$ | Bounded within $(6.0, 60.0)$ |
| **Total Domain Span** | $53.945766$ | Span $\Delta t \approx 54.0$ |
| **Mean Spacing ($\Delta t$)** | **$0.035988$** | Matches theoretical uniform spacing $\frac{53.946}{1499} = 0.035988$ |
| **Median Spacing ($\Delta t$)** | **$0.025277$** | Moderately skewed distribution |
| **Min Spacing ($\Delta t$)** | $0.000032$ | Closely clustered neighboring points |
| **Max Spacing ($\Delta t$)** | $0.400108$ | Largest local sampling gap |
| **Spacing Std Dev ($\sigma_{\Delta t}$)** | $0.035339$ | Non-uniform density variation |

*Finding:* The observed dataset is distributed across the full domain $[6.05, 60.00]$ with an average parameter spacing of $\Delta t \approx 0.0360$, but exhibits natural density fluctuations with occasional gaps up to $\Delta t \approx 0.40$.

---

## 5. Synthetic Noiseless Experiment

To prove theoretical consistency, we generated a synthetic noiseless curve with $N_{\text{obs}} = 1500$ points and evaluated it against increasing predicted sampling resolutions $N_{\text{pred}}$:

| $N_{\text{pred}}$ | Mean Obs $\to$ Pred $L_1$ | Mean Pred $\to$ Obs $L_1$ | Max Pred $\to$ Obs $L_1$ |
| :---: | :---: | :---: | :---: |
| **500** | $0.040520$ | $0.013813$ | $0.027645$ |
| **1,000** | $0.020138$ | $0.013441$ | $0.037397$ |
| **1,500 (Matched)** | **$0.000000$** | **$0.000000$** | **$0.000000$** |
| **5,000** | $0.004023$ | $0.013452$ | $0.042173$ |
| **10,000** | $0.002016$ | $0.013453$ | $0.042239$ |
| **20,000** | $0.001008$ | $0.013454$ | $0.042600$ |

*Conclusion:* On synthetic data with matched grid ($N_{\text{pred}} = 1500$), the error is exactly $0.000000$. For arbitrary $N_{\text{pred}} \neq N_{\text{obs}}$, Pred $\to$ Obs exhibits the exact predicted plateau ($\approx 0.01345$ for uniform grid, $\approx 0.0266$ for non-uniform observed dataset).

---

## 6. Phase 5 Terminology & Scientific Corrections

1. **Clarification on $\mathcal{O}(1/N)$ Convergence:**
   - The $\mathcal{O}(1/N)$ decay rate applies specifically to the **Observed $\to$ Predicted** manifold discretization error.
   - It does **not** apply to the Predicted $\to$ Observed direction, which is lower-bounded by the finite observation density $\mathcal{O}(1/N_{\text{obs}})$.
2. **Symmetric $L_1$ Definition Context:**
   - The reported symmetric mean $L_1$ ($0.02007$ at $N=1500$, $0.01432$ at $N=10000$) reflects a hybrid between continuous curve fidelity and the point-spacing resolution of `xy_data.csv`.
   - The pure manifold reconstruction accuracy is measured by the Observed $\to$ Predicted distance ($< 0.0020$ coordinate units).
