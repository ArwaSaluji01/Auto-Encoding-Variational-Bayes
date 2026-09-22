# Auto-Encoding Variational Bayes — Project Summary

## Overview

This project presents a from-scratch implementation of **Auto-Encoding Variational Bayes (AEVB)** based on the work of Diederik P. Kingma and Max Welling.

The objective was to move from the mathematical formulation of variational inference to a complete working generative model implemented in TensorFlow/Keras.

The implementation was developed progressively, covering the probabilistic encoder, reparameterization trick, probabilistic decoder, ELBO objective, stochastic optimization, latent-space visualization, generative sampling, and marginal likelihood estimation.

---

## Core Implementation

The primary MNIST model uses:

- 784-dimensional input
- 500 hidden units
- 20-dimensional latent space
- Tanh hidden activations
- Bernoulli decoder
- Gaussian latent prior
- Reparameterization trick
- ELBO optimization
- Adagrad optimization
- Batch size of 100
- One latent sample per datapoint

---

## Key Results

The 20-dimensional model was trained for 20 epochs.

```text
Training negative ELBO:
166.42 → 115.93

Test negative ELBO:
114.74

Mean KL divergence:
26.14

Reconstruction term:
-88.61

Reconstruction MSE:
0.115173
```

The model produced recognizable MNIST reconstructions and successfully generated new digit-like samples from latent vectors sampled from the standard normal prior.

A separate 2D model was trained to visualize the latent representation. The resulting embedding demonstrated continuous latent structure with overlapping digit classes.

A separate 3D model was also implemented to explore the marginal likelihood estimator described in the paper. Hybrid Monte Carlo sampling and density estimation were implemented as an experimental reproduction of the paper's evaluation procedure.

---

## Main Learning Outcomes

The project provided practical experience with:

Variational inference
ELBO derivation and optimization
KL divergence
Bernoulli likelihoods
Gaussian latent-variable models
The reparameterization trick
Stochastic gradient optimization
Latent-space visualization
Generative sampling
MCMC-based posterior inference
Research-paper implementation and experimental validation

A particularly important takeaway was the distinction between reproducing an algorithm conceptually and reproducing a research paper's numerical experiments exactly.

---

## Future Work

Potential extensions include:
- Explicit implementation of the paper's MAP/weight-decay objective
- Systematic hyperparameter studies
- Convolutional VAE architectures
- Improved HMC sampling and convergence diagnostics
- More robust marginal likelihood estimation
- Reproduction of the paper's Wake-Sleep and MCEM comparisons
- Experiments on additional datasets

---

## Reference

**Kingma, D. P., & Welling, M.**
**Auto-Encoding Variational Bayes.**
arXiv:1312.6114.
