# Neural Codes of Uncertainty in Visual Perception

**MEng Thesis — University of Cambridge, Department of Engineering**

Niket Rajeevan · Supervisor: Professor Máté Lengyel · Co-Supervisor: Puria Radmard

---

## Overview

How does the brain represent uncertainty about what it sees? This project develops a computational framework to investigate whether the primary visual cortex (V1) encodes uncertainty about sensory stimuli through **spatial coding** (static population activity at single time points) or **temporal coding** (dynamic stochastic sampling over time).

We build on the Gaussian Scale Mixture (GSM) model of [Orbán et al. (2016)](https://doi.org/10.1016/j.neuron.2016.09.038) to construct a full Bayesian ideal observer pipeline — from synthetic neural data generation to neural network-based decoding — providing computational tools that can ultimately be applied to real biological data from mouse V1.

## Key Contributions

1. **Orientation-conditioned priors** — Extended the GSM generative model with priors $p(\mathbf{y}|\theta)$ that condition latent neural features on stimulus orientation, enabling computation of ground-truth posteriors $p(\theta|\mathbf{x})$ and $p(d|\mathbf{x})$ via Bayesian inversion.

2. **Biologically-constrained synthetic data generator** — Hierarchical ancestral sampling procedure respecting the effective sample size (ESS = 3) imposed by calcium imaging temporal resolution.

3. **Systematic decoder comparison** — Temporal decoding significantly outperforms spatial decoding for perceptual uncertainty (normalised KL: 0.103 vs 0.128, $p < 0.001$), independently validating [Koblinger et al. (2024)](https://doi.org/10.1101/2024.01.01.000000). An unexpected reversal for decision uncertainty (spatial outperforms temporal) raises questions explored in the thesis.

4. **Ablation study on sample integration** — Three-sample temporal decoders dramatically improve variance estimation over single-sample decoders ($r = 0.928$ vs $0.736$), supporting the view that neural variability carries uncertainty information rather than being mere noise.

5. **Preliminary DDC framework** — Implemented 1024-dimensional Distributed Distributional Codes as groundwork for testing genuine spatial encoding schemes.

## Method

```
Generative Model (extended GSM)
─────────────────────────────────
  d → θ → y → x
         ↑       ↑
         z       ε

  d : binary decision variable (left/right)
  θ : grating orientation (1°–180°)
  y : latent neural features (248-dim)
  z : contrast gain (Gamma prior)
  x : observed image (16×16 pixels)
  ε : observation noise
```

The pipeline works as follows:

- **Ideal observer** — Invert the generative model to compute ground-truth posteriors over orientation and decision via numerical marginalisation.
- **Temporal encoding** — Sample $\mathbf{y}^{(1)}, \mathbf{y}^{(2)}, \mathbf{y}^{(3)} \sim p(\mathbf{y}|\mathbf{x})$ through hierarchical ancestral sampling.
- **Decoding** — Train feedforward neural networks (2×512 hidden units, ReLU) to recover posteriors from either sequential samples (temporal) or averaged samples (spatial).
- **Evaluation** — Compare decoders via normalised KL divergence with paired permutation testing (5,000 permutations).

## Results at a Glance

| Decoder | Target | Normalised KL | Winner |
|---------|--------|---------------|--------|
| Perceptual | $p(\theta\|\mathbf{x})$ | Temporal: 0.103 ± 0.094 vs Spatial: 0.128 ± 0.165 | **Temporal** |
| Decision | $p(d\|\mathbf{x})$ | Temporal: 0.065 ± 0.109 vs Spatial: 0.052 ± 0.100 | **Spatial** |

All comparisons significant at $p < 0.001$.


## References

- Orbán, G., Berkes, P., Fiser, J., & Lengyel, M. (2016). *Neural Variability and Sampling-Based Probabilistic Representations in the Visual Cortex.* Neuron, 92(2), 530–543.
- Ma, W. J., Beck, J. M., Latham, P. E., & Pouget, A. (2006). *Bayesian Inference with Probabilistic Population Codes.* Nature Neuroscience, 9(11), 1432–1438.
- Vértes, E., & Sahani, M. (2018). *Flexible and accurate inference and learning for deep generative models.* NeurIPS 2018.
- Koblinger, Á., Amvrosiadis, T., Rochefort, N. L., & Lengyel, M. (2024). *A Data-Driven Approach for Testing Specific Hypotheses About Probabilistic Representations.* Cosyne 2024.

---

This project was submitted in partial fulfilment of the requirements for the degree of Master of Engineering at the University of Cambridge (June 2025).

For the full codebase, please contact **nr490@cantab.ac.uk**.
