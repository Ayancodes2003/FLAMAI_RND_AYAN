# Phase 6: Robustness, Noise Sensitivity & Limitation Analysis Report

---

## 1. Executive Summary

This report documents controlled empirical evaluations of the geometry-based analytical parameter estimation pipeline. Using synthetic ground-truth parameters ($\theta^* = 30.0^\circ, M^* = 0.03, X^* = 55.0$), we systematically analyze:
1. **Noise Sensitivity:** Gaussian perturbations $\sigma \in [0, 0.10]$ with 10 independent trials per non-zero level.
2. **Parameter Sensitivity:** One-at-a-time parameter sweeps around the global optimum.
3. **Observation Density:** Performance across sample sizes $N \in [100, 1500]$.
4. **Limitation & Failure Regimes:** Extreme noise ($\sigma \le 2.0$), severe sparsity ($N \le 50$), and parameter boundary evaluations ($M = \pm 0.048$).

---

## 2. Noise Robustness Experiments

Independent Gaussian noise $\epsilon_x, \epsilon_y \sim \mathcal{N}(0, \sigma^2)$ was added to both coordinates of $1,500$ observations before point shuffling. Ten independent random seeds ($100, \dots, 109$) were evaluated per noise level:

| Noise Level ($\sigma$) | Mean $|\Delta\theta|$ (deg) | Std $|\Delta\theta|$ | Mean $|\Delta M|$ | Std $|\Delta M|$ | Mean $|\Delta X|$ | Std $|\Delta X|$ | Mean Curve $L_1$ | Max Curve $L_1$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\sigma = 0.000$** | $0.000000^\circ$ | $0.000000^\circ$ | $0.000000$ | $0.000000$ | $0.000000$ | $0.000000$ | $0.000000$ | $0.000000$ |
| **$\sigma = 0.001$** | $0.000070^\circ$ | $0.000045^\circ$ | $0.000001$ | $0.000001$ | $0.000059$ | $0.000041$ | $0.000112$ | $0.000196$ |
| **$\sigma = 0.005$** | $0.000373^\circ$ | $0.000212^\circ$ | $0.000003$ | $0.000002$ | $0.000326$ | $0.000185$ | $0.000576$ | $0.000962$ |
| **$\sigma = 0.010$** | $0.000673^\circ$ | $0.000424^\circ$ | $0.000005$ | $0.000004$ | $0.000587$ | $0.000366$ | $0.001086$ | $0.001905$ |
| **$\sigma = 0.050$** | $0.003567^\circ$ | $0.002013^\circ$ | $0.000032$ | $0.000020$ | $0.003547$ | $0.001784$ | $0.006645$ | $0.009844$ |
| **$\sigma = 0.100$** | $0.006228^\circ$ | $0.003451^\circ$ | $0.000109$ | $0.000084$ | $0.005203$ | $0.002951$ | $0.015073$ | $0.021544$ |

### Findings:
- Across all tested noise levels $\sigma \le 0.10$, parameter error scales linearly with noise amplitude without catastrophic divergence.
- At $\sigma = 0.10$ (a substantial spatial jitter of $\pm 0.1$ units on a curve with transverse amplitude $\approx 2.5$), the mean angular error remains $< 0.007^\circ$, $M$ error $< 0.0002$, and offset error $< 0.006$ units.

---

## 3. Parameter Sensitivity Analysis

One-at-a-time perturbations around the ground truth demonstrate how reconstruction $L_1$ distance responds to parameter variations:

### A. Rotation Angle ($\theta$ around $30.0^\circ$)
| Perturbation $\Delta\theta$ | Candidate $\theta$ | Reconstructed Curve $L_1$ Error |
| :---: | :---: | :---: |
| $-1.0^\circ$ | $29.0^\circ$ | $0.540979$ |
| $-0.5^\circ$ | $29.5^\circ$ | $0.272841$ |
| $-0.1^\circ$ | $29.9^\circ$ | $0.060334$ |
| **$0.0^\circ$** | **$30.0^\circ$** | **$0.000000$** |
| $+0.1^\circ$ | $30.1^\circ$ | $0.060342$ |
| $+0.5^\circ$ | $30.5^\circ$ | $0.273740$ |
| $+1.0^\circ$ | $31.0^\circ$ | $0.543627$ |

*Response:* Quadratic-like symmetric basin ($\approx 0.54$ error per degree of angular deviation).

### B. Growth Rate ($M$ around $0.030$)
| Perturbation $\Delta M$ | Candidate $M$ | Reconstructed Curve $L_1$ Error |
| :---: | :---: | :---: |
| $-0.005$ | $0.025$ | $0.387701$ |
| $-0.002$ | $0.028$ | $0.161652$ |
| $-0.001$ | $0.029$ | $0.083990$ |
| **$0.000$** | **$0.030$** | **$0.000000$** |
| $+0.001$ | $0.031$ | $0.084080$ |
| $+0.002$ | $0.032$ | $0.161553$ |
| $+0.005$ | $0.035$ | $0.393953$ |

### C. Offset ($X$ around $55.0$)
| Perturbation $\Delta X$ | Candidate $X$ | Reconstructed Curve $L_1$ Error |
| :---: | :---: | :---: |
| $-1.0$ | $54.0$ | $0.557783$ |
| $-0.5$ | $54.5$ | $0.280772$ |
| $-0.1$ | $54.9$ | $0.061571$ |
| **$0.0$** | **$55.0$** | **$0.000000$** |
| $+0.1$ | $55.1$ | $0.061540$ |
| $+0.5$ | $55.5$ | $0.279451$ |
| $+1.0$ | $56.0$ | $0.552322$ |

*Sensitivity Comparison:* Spatial offset $X$ and angle $\theta$ produce the steepest initial gradient in coordinate space, while $M$ error grows exponentially towards larger $t$.

---

## 4. Observation Density Analysis

Noiseless synthetic datasets with varying point counts $N$:

| Point Count $N$ | Estimated $\theta$ (deg) | Estimated $M$ | Estimated $X$ | Absolute Error $(|\Delta\theta|, |\Delta M|, |\Delta X|)$ |
| :---: | :---: | :---: | :---: | :---: |
| **$100$** | $30.000000^\circ$ | $0.030000$ | $55.000000$ | $(<10^{-5}, <10^{-6}, <10^{-5})$ |
| **$250$** | $30.000000^\circ$ | $0.030000$ | $55.000000$ | $(<10^{-5}, <10^{-6}, <10^{-5})$ |
| **$500$** | $30.000000^\circ$ | $0.030000$ | $55.000000$ | $(<10^{-5}, <10^{-6}, <10^{-5})$ |
| **$1,000$** | $30.000000^\circ$ | $0.030000$ | $55.000000$ | $(<10^{-5}, <10^{-6}, <10^{-5})$ |
| **$1,500$** | $30.000000^\circ$ | $0.030000$ | $55.000000$ | $(<10^{-5}, <10^{-6}, <10^{-5})$ |

*Finding:* Under noiseless conditions, the geometry solver recovers exact parameters with as few as $100$ points ($< 20$ points per oscillation cycle).

---

## 5. Limitation & Failure Regime Analysis

### Regime 1: Severe Sparsity ($N \le 50$)
When point counts drop below $N=50$ ($< 10$ points per full oscillation cycle):
- At $N=50$: $\theta_{\text{est}} = 27.91^\circ$ ($|\Delta\theta| \approx 2.09^\circ$), $M_{\text{est}} = 0.0235$.
- At $N=10$: $\theta_{\text{est}} = 27.35^\circ$ ($|\Delta\theta| \approx 2.65^\circ$), $M_{\text{est}} = 0.0000$.
- **Failure Cause:** PCA principal axis initialization is sensitive to spatial under-sampling when fewer than 2-3 points sample individual oscillation crests/troughs.

### Regime 2: Severe Noise ($\sigma \ge 1.0$)
- At $\sigma = 0.50$: $|\Delta\theta| \approx 0.06^\circ, |\Delta M| \approx 0.0014, |\Delta X| \approx 0.011$.
- At $\sigma = 1.00$: $|\Delta\theta| \approx 0.08^\circ, |\Delta M| \approx 0.0020, |\Delta X| \approx 0.12$.
- At $\sigma = 2.00$: $|\Delta\theta| \approx 0.03^\circ, |\Delta M| \approx 0.0022, |\Delta X| \approx 0.04$.
- **Observation:** Statistical averaging over 1,500 points keeps parameter error remarkably bounded even when noise variance approaches the oscillation amplitude.

### Regime 3: Boundary & Zero $M$ Parameter Values
Testing across $M \in [-0.048, +0.048]$:
- $M = -0.048 \implies |\Delta M| < 10^{-6}, |\Delta\theta| < 10^{-5}$.
- $M = 0.000 \implies |\Delta M| < 10^{-6}, |\Delta\theta| < 10^{-5}$.
- $M = +0.048 \implies |\Delta M| < 10^{-6}, |\Delta\theta| < 10^{-5}$.
- **Conclusion:** The log-envelope regression estimator handles decaying, constant, and growing amplitudes with uniform fidelity.

---

## 6. Generated Figures

Diagnostic plots saved to `results/figures/`:
1. [`noise_parameter_error.png`](file:///D:/PROJECTS%20%20GITHUB/FLAMAI_RND_AYAN/results/figures/noise_parameter_error.png) — Parameter error scaling vs $\sigma$.
2. [`noise_reconstruction_error.png`](file:///D:/PROJECTS%20%20GITHUB/FLAMAI_RND_AYAN/results/figures/noise_reconstruction_error.png) — Mean and max curve $L_1$ error vs $\sigma$.
3. [`observation_density.png`](file:///D:/PROJECTS%20%20GITHUB/FLAMAI_RND_AYAN/results/figures/observation_density.png) — Parameter error vs observation sample count $N$.
4. [`parameter_sensitivity_theta.png`](file:///D:/PROJECTS%20%20GITHUB/FLAMAI_RND_AYAN/results/figures/parameter_sensitivity_theta.png) — $L_1$ response to $\Delta\theta$.
5. [`parameter_sensitivity_M.png`](file:///D:/PROJECTS%20%20GITHUB/FLAMAI_RND_AYAN/results/figures/parameter_sensitivity_M.png) — $L_1$ response to $\Delta M$.
6. [`parameter_sensitivity_X.png`](file:///D:/PROJECTS%20%20GITHUB/FLAMAI_RND_AYAN/results/figures/parameter_sensitivity_X.png) — $L_1$ response to $\Delta X$.
