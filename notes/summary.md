# Auto-Encoding Variational Bayes — Project Summary

## Overview

This project presents a from-scratch implementation of **Auto-Encoding Variational Bayes (AEVB)** based on the work of Diederik P. Kingma and Max Welling.

The objective was to move from the mathematical formulation of variational inference to a complete working generative model implemented in TensorFlow/Keras.

The implementation progressively covers the probabilistic encoder, reparameterization trick, probabilistic decoder, ELBO objective, stochastic optimization, latent-space visualization, generative sampling, latent-dimension experiments, wake-sleep comparison, and marginal likelihood estimation.

The implementation is designed as a **miniature research reproduction**, with reduced experimental configurations where necessary to remain practical in Google Colab.

---

## Core Implementation

The primary MNIST model uses:

* 784-dimensional input
* 500 hidden units
* 20-dimensional latent space
* Tanh hidden activations
* Bernoulli decoder
* Gaussian latent prior
* Reparameterization trick
* ELBO optimization
* Adagrad optimization
* Batch size of 100
* One latent sample per datapoint

The MNIST inputs are binarized to match the Bernoulli observation model.

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

The trained model produces recognizable MNIST reconstructions and can generate new digit-like samples by sampling latent vectors from the standard normal prior.

A separate 2D model was trained to visualize the latent representation. The resulting embedding demonstrates continuous latent structure with overlapping digit classes.

Additional experiments include a latent-dimension sweep and a lightweight wake-sleep comparison.

A separate low-dimensional model is used to investigate marginal likelihood estimation. The experiment uses HMC for posterior-sampling analysis and Monte Carlo importance sampling using samples from the learned variational posterior.

Final marginal-likelihood results:

```text
Mean ELBO:                  [PLACEHOLDER]
Mean estimated log p(x):    [PLACEHOLDER]
Mean estimated gap:         [PLACEHOLDER]
Median estimated gap:       [PLACEHOLDER]
Minimum estimated gap:      [PLACEHOLDER]
Maximum estimated gap:      [PLACEHOLDER]
```

---

## Main Learning Outcomes

The project provided practical experience with:

* Variational inference
* ELBO derivation and optimization
* KL divergence
* Bernoulli likelihoods
* Gaussian latent-variable models
* The reparameterization trick
* Stochastic gradient optimization
* Latent-space visualization
* Generative sampling
* Latent-dimension analysis
* MCMC-based posterior inference
* Monte Carlo importance sampling
* Research-paper implementation and experimental validation

A particularly important takeaway was the distinction between implementing a research method conceptually and reproducing the original paper's numerical experiments exactly.

---

## Future Work

Potential extensions include:

* Explicit implementation of the paper's MAP/weight-decay objective
* Systematic hyperparameter studies
* Convolutional VAE architectures
* Improved HMC sampling and convergence diagnostics
* More robust marginal likelihood estimators
* Larger Monte Carlo sample sizes
* More complete Wake-Sleep and MCEM comparisons
* Experiments on additional datasets
* Larger-scale reproduction of the original experimental setup

---

## Reference

**Kingma, D. P., & Welling.**
**Auto-Encoding Variational Bayes.**
arXiv:1312.6114.
