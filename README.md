<div align="center">

# 🧠 Neural Codes of Uncertainty in Visual Perception

**Bayesian inference, Monte Carlo sampling, and high-dimensional distributional codes for adjudicating how biological neural circuits encode uncertainty**

[![Cambridge](https://img.shields.io/badge/University%20of%20Cambridge-MEng%20Thesis-blue?style=flat-square)](https://www.cam.ac.uk)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)]()


*Supervised by Professor Máté Lengyel · Co-supervised by Puria Radmard · Department of Engineering*

---

**Temporal vs spatial uncertainty encoding in V1 · Hierarchical Bayesian inversion · Monte Carlo neural data synthesis · Distributed Distributional Codes · 248-dim latent space · 500K synthetic training images**

</div>

---

## The Problem

Perception is inference under uncertainty. When the primary visual cortex (V1) receives a noisy, ambiguous stimulus, it doesn't produce a point estimate — it maintains a **full posterior distribution** over the state of the world. But how is that distribution physically realised in neural activity?

Two competing hypotheses:

| | **Spatial Coding** | **Temporal Coding** |
|---|---|---|
| **Mechanism** | Instantaneous population vector encodes the full posterior | Neural responses over time are Monte Carlo samples from the posterior |
| **Variability** | Noise — a nuisance to be averaged out | Signal — carries uncertainty information |
| **Theory** | PPCs (Ma et al. 2006), DDCs (Vértes & Sahani 2018) | Neural Sampling Hypothesis (Orbán et al. 2016) |
| **Readout** | Single-shot | Requires temporal integration |

This project constructs the full computational pipeline — generative model, Bayesian inversion, Monte Carlo data synthesis, neural decoder training, and statistical hypothesis testing — to rigorously adjudicate between them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  HIERARCHICAL GENERATIVE MODEL                      │
│                                                                     │
│   d ──→ θ ──→ y ──→ x        x = z A y + ε                        │
│   │     │     ↑     ↑                                              │
│   │     │     z     ε         d ∈ {-1,+1}  decision                │
│   │     │                     θ ∈ {1°..180°} orientation           │
│   │     └── p(y|θ) = N(μ_θ, C_θ)          248-dim latents         │
│   └──── p(θ|d) = Uniform(Θ_d)             categorical constraint  │
│                                                                     │
│         z ~ Gamma(k_Γ, θ_Γ)               contrast gain            │
│         ε ~ N(0, σ²_x I)                  observation noise        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    Bayesian Inversion
                    (closed-form + numerical quadrature)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    IDEAL OBSERVER POSTERIORS                         │
│                                                                     │
│   p(θ|x)  ── perceptual posterior (180-dim discrete)               │
│   p(d|x)  ── decision posterior (binary)                           │
│   p(y|x)  ── latent posterior (248-dim Gaussian mixture)           │
│               → Monte Carlo ancestral sampling → synthetic data    │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│    TEMPORAL DECODER       │   │    SPATIAL DECODER        │
│                           │   │                           │
│  y⁽¹⁾, y⁽²⁾, y⁽³⁾       │   │  ȳ = ¹⁄₃ Σ y⁽ⁱ⁾         │
│  → 3 forward passes      │   │  → 1 forward pass         │
│  → aggregate in P-space  │   │  → single prediction      │
│                           │   │                           │
│  Tests: does variability  │   │  Tests: does averaging    │
│  carry information?       │   │  preserve information?    │
└──────────────────────────┘   └──────────────────────────┘
              │                               │
              └───────────┬───────────────────┘
                          ▼
              Paired permutation testing
              (5,000 permutations, α = 0.05)
```

---

## Mathematical Framework

### I. Generative Model — Gaussian Scale Mixture

Images in V1 are modelled as latent neural features projected through a Gabor filter bank, scaled by contrast, and corrupted by noise:

$$\mathbf{x} = z \, A \, \mathbf{y} + \boldsymbol{\epsilon}, \qquad \mathbf{y} \in \mathbb{R}^{D_y},\; \mathbf{x} \in \mathbb{R}^{D_x},\; A \in \mathbb{R}^{D_x \times D_y}$$

where $A$ encodes a bank of oriented Gabor filters (computational analogue to V1 receptive fields), $z \sim \text{Gamma}(k_\Gamma, \theta_\Gamma)$ is the contrast gain, and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_X^2 I)$.

### II. Orientation-Conditioned Priors *(novel contribution)*

The original model uses an unstructured prior $p(\mathbf{y}) = \mathcal{N}(\mathbf{0}, C)$. We replace this with orientation-conditioned priors fitted from empirical posterior statistics:

$$p(\mathbf{y} \mid \theta) = \mathcal{N}\!\left(\mathbf{y};\, \boldsymbol{\mu}_\theta,\, C_\theta\right), \qquad \boldsymbol{\mu}_\theta = \frac{1}{N}\sum_{k=1}^{N}\mathbf{y}_k, \quad C_\theta = \frac{1}{N}\sum_{k=1}^{N}\left(\mathbf{y}_k - \boldsymbol{\mu}_\theta\right)\!\left(\mathbf{y}_k - \boldsymbol{\mu}_\theta\right)^{\!T}$$

This modification is what makes the entire downstream pipeline possible — it introduces orientation-dependent structure into latent space, validated via pairwise PCA reconstruction error analysis confirming that nearby orientations share similar subspaces while distant orientations diverge.

### III. Bayesian Inversion — Perceptual Posterior

The posterior over stimulus orientation requires a two-stage marginalisation. First, the likelihood under each orientation integrates out latent features analytically (exploiting the linear-Gaussian structure):

$$p(\mathbf{x} \mid z, \theta) = \mathcal{N}\!\left(\mathbf{x};\; z\,A\,\boldsymbol{\mu}_\theta,\; \sigma_X^2 I + z^2\, A\,C_\theta\,A^T\right)$$

Then contrast is marginalised via numerical quadrature over a discretised grid $\{z_j\}_{j=1}^{D_z}$:

$$p(\theta \mid \mathbf{x}) = \frac{\displaystyle\sum_{j=1}^{D_z} \mathcal{N}\!\left(\mathbf{x};\; z_j A\boldsymbol{\mu}_\theta,\; \sigma_X^2 I + z_j^2 A C_\theta A^T\right) p(z_j)}{\displaystyle\sum_{\theta'}\sum_{j=1}^{D_z} \mathcal{N}\!\left(\mathbf{x};\; z_j A\boldsymbol{\mu}_{\theta'},\; \sigma_X^2 I + z_j^2 A C_{\theta'} A^T\right) p(z_j)}$$

This 180-dimensional discrete distribution is the **ground-truth supervision signal** for all downstream decoder training.

### IV. Hierarchical Decision Model

A binary decision variable $d \in \{-1, +1\}$ constrains orientation to categorical half-spaces $\Theta_d$, adding a further marginalisation layer:

$$p(d \mid \mathbf{x}) = \frac{p(d)\;\dfrac{1}{n_d}\displaystyle\sum_{\theta \in \Theta_d} p(\mathbf{x} \mid \theta)}{\displaystyle\sum_{d'} p(d')\;\dfrac{1}{n_{d'}}\displaystyle\sum_{\theta \in \Theta_{d'}} p(\mathbf{x} \mid \theta)}$$

The full posterior over latents now requires triple marginalisation over decision, orientation, and contrast:

$$p(\mathbf{y} \mid \mathbf{x}) = \sum_{d}\, p(d \mid \mathbf{x}) \sum_{\theta \in \Theta_d} p(\theta \mid \mathbf{x}, d) \int p(\mathbf{y} \mid z, \theta, \mathbf{x})\; p(z \mid \theta, \mathbf{x})\; dz$$

where each innermost component has closed-form Gaussian parameters:

$$\Sigma_z^{-1} = C_\theta^{-1} + \frac{z^2}{\sigma_X^2}\,A^T\!A, \qquad \boldsymbol{\mu}_z = \Sigma_z\!\left(C_\theta^{-1}\boldsymbol{\mu}_\theta + \frac{z}{\sigma_X^2}\,A^T\mathbf{x}\right)$$

### V. Monte Carlo Neural Data Generation — Hierarchical Ancestral Sampling

Synthetic temporal neural data is generated by ancestral sampling through the full graphical model. Each sample traces the complete causal chain:

$$d^{(i)} \sim p(d \mid \mathbf{x}) \;\longrightarrow\; \theta^{(i)} \sim p(\theta \mid \mathbf{x}, d^{(i)}) \;\longrightarrow\; z^{(i)} \sim p(z \mid \mathbf{x}, \theta^{(i)}) \;\longrightarrow\; \mathbf{y}^{(i)} \sim \mathcal{N}\!\left(\boldsymbol{\mu}_{z^{(i)}},\, \Sigma_{z^{(i)}}\right)$$

Under the temporal coding hypothesis, each $\mathbf{y}^{(i)}$ represents neural activity at a successive time point — a single Monte Carlo sample from the posterior, not a noisy point estimate. The number of samples is constrained to $N = 3$, reflecting the effective sample size imposed by GCaMP6s calcium imaging temporal autocorrelation.

### VI. Temporal Decoding — Aggregation in Probability Space

The temporal decoder processes each sample independently through a shared network, then averages predictions in **probability space** (not activation space):

$$\hat{p}(\theta \mid \mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} f_{\text{NN}}\!\left(\mathbf{y}^{(i)}\right)_\theta$$

This is distinct from the spatial decoder which averages in **activation space** before a single forward pass: $\hat{p}(\theta \mid \mathbf{x}) = f_{\text{NN}}\!\left(\frac{1}{N}\sum_i \mathbf{y}^{(i)}\right)_\theta$. The critical question is whether inter-sample variability $\{\mathbf{y}^{(1)}, \mathbf{y}^{(2)}, \mathbf{y}^{(3)}\}$ carries exploitable uncertainty structure or is destructive noise.

### VII. Distributed Distributional Codes — Kernel Embeddings of Posteriors

For genuine spatial encoding (beyond the proxy of averaged temporal samples), we implement DDCs where the posterior is represented as expected values of $K = 1024$ nonlinear encoding functions:

$$r^{(i)}(\mathbf{x}) = \mathbb{E}_{p(\mathbf{y}|\mathbf{x})}\!\Big[\sigma\!\left(\mathbf{w}^{(i)T}\mathbf{y} + b^{(i)}\right)\Big], \qquad i = 1, \ldots, K$$

with $\mathbf{w}^{(i)} \sim \mathcal{N}(\mathbf{0}, I)$ and $b^{(i)} \sim \mathcal{N}(0, 1)$. These expectations define a maximum-entropy distribution:

$$q(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\boldsymbol{\eta}(\mathbf{x}))} \exp\!\left(\sum_{i=1}^{K} \eta^{(i)}(\mathbf{x})\; \sigma\!\left(\mathbf{w}^{(i)T}\mathbf{y} + b^{(i)}\right)\right)$$

In the limit $K \to \infty$, this converges to a **mean embedding of the posterior in a reproducing kernel Hilbert space** — theoretically guaranteeing arbitrary approximation of any posterior distribution, including non-Gaussian, multimodal forms that PPCs cannot represent.

### VIII. Analytical Quadrature for DDC Expectations

Computing $r^{(i)}(\mathbf{x})$ over the Gaussian mixture posterior exploits the fact that each scalar projection $y_i' = \mathbf{w}^{(i)T}\mathbf{y} + b^{(i)}$ follows a univariate Gaussian under each mixture component:

$$y_i' \sim \mathcal{N}\!\left(m_{i,z,\theta},\; v_{i,z,\theta}\right), \quad m_{i,z,\theta} = \mathbf{w}^{(i)T}\boldsymbol{\mu}_{z,\theta} + b^{(i)}, \quad v_{i,z,\theta} = \mathbf{w}^{(i)T}\Sigma_{z,\theta}\,\mathbf{w}^{(i)}$$

The sigmoid expectation under each Gaussian component is computed via standardised quadrature on $\mathcal{N}(0,1)$, exploiting the invariance of normalised weights under affine transforms:

$$\mathbb{E}\!\left[\sigma(y_i')\right] \approx \sum_{j=1}^{N_g} w_j \cdot \sigma\!\left(m_{i,z,\theta} + \sqrt{v_{i,z,\theta}}\cdot k_j\right)$$

The full DDC vector assembles contributions across the entire mixture:

$$r^{(i)}(\mathbf{x}) = \sum_{\theta}\, p(\theta \mid \mathbf{x}) \sum_{z}\, p(z \mid \mathbf{x}, \theta)\,\Delta z \;\cdot\; \sum_{j=1}^{N_g} w_j \cdot \sigma\!\left(m_{i,z,\theta} + \sqrt{v_{i,z,\theta}}\cdot k_j\right)$$

This yields a 1024-dimensional population vector that encodes the full posterior **instantaneously** — no temporal sampling required — providing the theoretically grounded spatial encoding counterpart to the Monte Carlo temporal scheme.

### IX. Training Objective

All decoders are trained by minimising KL divergence against the ideal observer posterior:

$$\mathcal{L}(\mathbf{x}) = D_{\text{KL}}\!\Big[p(\theta \mid \mathbf{x}) \;\big\|\; \hat{p}(\theta \mid \mathbf{x})\Big] = \sum_\theta p(\theta \mid \mathbf{x}) \log \frac{p(\theta \mid \mathbf{x})}{\hat{p}(\theta \mid \mathbf{x})}$$

Trained on 500K synthetic images spanning the full combinatorial space of orientation $\times$ contrast $\times$ aperture. Statistical comparison via paired permutation testing (5,000 permutations, two-tailed, $\alpha = 0.05$) across 1,260 held-out test stimuli.

---

## Key Contributions

1. **Orientation-conditioned priors** — Extended the GSM model with $p(\mathbf{y}|\theta) = \mathcal{N}(\boldsymbol{\mu}_\theta, C_\theta)$, enabling ground-truth posterior computation via Bayesian inversion over the full hierarchical generative model.

2. **Monte Carlo neural data synthesis** — Hierarchical ancestral sampling through $d \to \theta \to z \to \mathbf{y}$, biologically constrained to ESS = 3 from calcium imaging autocorrelation analysis.

3. **Systematic decoder comparison** — Temporal decoding significantly outperforms spatial for perceptual uncertainty ($p < 0.001$), independently validating Koblinger et al. (2024) on synthetic data with known ground truth.

4. **Variance-specific advantage of temporal integration** — Ablation study reveals three-sample decoders improve variance estimation by $\Delta r = +0.192$ while mean estimation improves only $\Delta r = +0.045$, supporting the view that neural variability carries genuine uncertainty information.

5. **DDC framework with analytical quadrature** — 1024-dimensional nonlinear projections with RKHS convergence guarantees, computed via standardised Gauss-Hermite quadrature over the mixture posterior. Establishes the methodology for testing genuine spatial codes.

---

- **Python** — NumPy, SciPy, PyTorch


## References

- Orbán, G., Berkes, P., Fiser, J., & Lengyel, M. (2016). Neural Variability and Sampling-Based Probabilistic Representations in the Visual Cortex. *Neuron*, 92(2), 530–543.
- Ma, W. J., Beck, J. M., Latham, P. E., & Pouget, A. (2006). Bayesian Inference with Probabilistic Population Codes. *Nature Neuroscience*, 9(11), 1432–1438.
- Vértes, E., & Sahani, M. (2018). Flexible and accurate inference and learning for deep generative models. *NeurIPS 2018*.
- Koblinger, Á., Amvrosiadis, T., Rochefort, N. L., & Lengyel, M. (2024). A Data-Driven Approach for Testing Specific Hypotheses About Probabilistic Representations. *Cosyne 2024*.
- Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2020). Cortical-like dynamics in recurrent circuits optimised for sampling-based probabilistic inference. *Nature Neuroscience*, 23(9), 1138–1149.

---

<div align="center">

*Submitted in partial fulfilment of the requirements for the degree of Master of Engineering at the University of Cambridge (June 2025).*

**For the full codebase, contact [nr490@cantab.ac.uk](mailto:nr490@cantab.ac.uk)**

</div>
