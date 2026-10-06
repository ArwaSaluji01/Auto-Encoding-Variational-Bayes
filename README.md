# Auto-Encoding Variational Bayes

A from-scratch TensorFlow/Keras implementation of **Auto-Encoding Variational Bayes (AEVB)** based on the paper by Diederik P. Kingma and Max Welling.

The project implements the core ideas behind the Variational Autoencoder, including probabilistic encoding, the reparameterization trick, ELBO optimization, latent-space representation, generative sampling, latent-dimension experiments, wake-sleep comparison, and marginal likelihood estimation.

The implementation is structured as a miniature research reproduction, with reduced experimental configurations.

---

## Project Overview

Variational inference provides a way to approximate otherwise intractable posterior distributions in probabilistic generative models.

The AEVB framework combines:

* Variational inference
* Stochastic Gradient Variational Bayes (SGVB)
* The reparameterization trick
* Neural-network-based recognition and generative models

The resulting architecture is commonly known as the **Variational Autoencoder (VAE)**.

This project implements the method from its underlying mathematical formulation rather than relying on a pre-built VAE implementation.

### Main Pipeline

```text
MNIST image
     ↓
Probabilistic Encoder
     ↓
μ, log(σ²)
     ↓
Reparameterization
     ↓
z
     ↓
Probabilistic Decoder
     ↓
Reconstructed image
```

For generation:

```text
z ~ N(0, I)
     ↓
Decoder
     ↓
Generated MNIST image
```

---

## Original Paper

**Diederik P. Kingma and Max Welling**

**Auto-Encoding Variational Bayes**

[Original Paper — arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

---

## Dataset

### MNIST

The project uses the MNIST handwritten digit dataset containing 28 × 28 grayscale images of handwritten digits from 0 to 9.

Each image is:

* Normalized to `[0, 1]`
* Binarized using a threshold of `0.5`
* Flattened from 28 × 28 to 784 dimensions
* Modelled using a Bernoulli decoder

The standard TensorFlow/Keras MNIST train/test split is used.

---

## Model Architecture

### Main AEVB Model

| Component                    | Configuration |
| ---------------------------- | ------------- |
| Input dimension              | 784           |
| Hidden units                 | 500           |
| Latent dimensions            | 20            |
| Decoder output               | 784           |
| Hidden activation            | Tanh          |
| Observation model            | Bernoulli     |
| Batch size                   | 100           |
| Latent samples per datapoint | 1             |
| Optimizer                    | Adagrad       |
| Learning rate                | 0.01          |

The main configuration follows the paper's MNIST setup closely while keeping the implementation practical for Colab.

---

## Core Mathematical Objective

The model optimizes the Evidence Lower Bound (ELBO):

```text
L(θ, φ; x)
=
E_q[log pθ(x,z) - log qφ(z|x)]
```

which can be written as:

```text
L(θ, φ; x)
=
E_q[log pθ(x|z)]
-
DKL(qφ(z|x) || p(z))
```

The loss minimized during training is therefore:

```text
Negative ELBO
=
KL divergence
-
Reconstruction log likelihood
```

The encoder produces:

```text
μ
log(σ²)
```

and the latent variable is sampled using:

```text
z = μ + σ ⊙ ε

ε ~ N(0, I)
```

This separates the stochastic component from the learnable parameters and allows gradient-based optimization.

---

## Implementation Levels

### Level 1 — Environment and Data Preprocessing

* TensorFlow setup
* Random seeds
* MNIST loading
* Normalization
* Binarization
* Flattening
* Dataset batching

### Level 2 — Probabilistic Encoder

* Neural-network encoder
* Mean estimation
* Log-variance estimation

### Level 3 — Reparameterization

* Gaussian latent sampling
* Reparameterization trick
* Stochasticity verification

### Level 4 — Probabilistic Decoder

* Neural-network decoder
* Bernoulli output distribution
* Image reconstruction

### Level 5 — ELBO and Loss

* KL divergence
* Bernoulli reconstruction likelihood
* ELBO computation
* Negative ELBO loss

### Level 6 — AEVB Training

* Gradient-based optimization
* Adagrad
* Mini-batch training
* ELBO monitoring

### Level 7 — Evaluation

* Test-set ELBO
* KL divergence
* Reconstruction likelihood
* Reconstruction quality
* MSE as an auxiliary metric

### Level 8 — Latent Dimension Analysis

The model is evaluated using multiple latent dimensions:

```text
2, 5, 10, 20, 50
```

This provides a small-scale comparison of representation capacity versus latent dimensionality.

### Level 9 — 2D Latent Space and Generation

A separate 2D model is trained to visualize the learned latent representation.

The notebook also explores:

* Generated MNIST images from random latent vectors
* Nearby latent points
* Continuous structure in the learned latent space

### Level 10 — Wake-Sleep Comparison

A lightweight wake-sleep baseline is implemented to compare its behavior against standard AEVB training.

This is a simplified experimental comparison rather than an exact reproduction of the historical wake-sleep experiments.

### Level 11 — Marginal Likelihood

A separate low-dimensional model is used to investigate marginal likelihood estimation.

The experiment uses:

* 100 hidden units
* 3 latent variables
* Reduced sampling configuration for Colab

HMC is used for posterior-sampling analysis.

For the marginal likelihood estimate itself, samples are drawn directly from the learned variational posterior:

```text
z ~ q(z|x)
```

and Monte Carlo importance sampling is used:

```text
p(x) ≈
1/N Σ [p(x,z) / q(z|x)]
```

This avoids fitting an additional KDE to the posterior samples and keeps the experiment computationally manageable.

---

## Results

### Main 20D AEVB Model

The model was trained for 20 epochs.

| Metric                            |   Result |
| --------------------------------- | -------: |
| Training Negative ELBO — Epoch 1  |   166.42 |
| Training Negative ELBO — Epoch 20 |   115.93 |
| Test Negative ELBO                |   114.74 |
| Mean KL divergence                |    26.14 |
| Mean reconstruction term          |   -88.61 |
| Reconstruction MSE                | 0.115173 |

The negative ELBO decreased consistently during training.

The reconstruction objective is based on the Bernoulli log likelihood; MSE is reported only as an auxiliary diagnostic metric.

### 2D Latent Representation

The separate 2D model was trained for 15 epochs.

| Epoch | Negative ELBO |
| ----: | ------------: |
|     1 |        201.35 |
|     5 |        179.40 |
|    10 |        176.26 |
|    15 |        173.60 |

The resulting latent representation demonstrates continuous structure, with different MNIST digit classes occupying partially distinct regions while retaining some overlap.

### Image Generation

The trained 20-dimensional decoder was sampled using:

```text
z ~ N(0, I)
```

Generated latent vectors:

```text
(20, 20)
```

Generated images:

```text
(20, 784)
```

The decoder produces digit-like samples without requiring an input image.

### Marginal Likelihood Experiment

The final miniature experiment uses a 3-dimensional latent model.

| Metric                      |          Result |
| --------------------------- | --------------: |
| Number of evaluation images |              20 |
| Mean ELBO                   | `[PLACEHOLDER]` |
| Mean estimated log p(x)     | `[PLACEHOLDER]` |
| Mean estimated gap          | `[PLACEHOLDER]` |
| Median estimated gap        | `[PLACEHOLDER]` |
| Minimum estimated gap       | `[PLACEHOLDER]` |
| Maximum estimated gap       | `[PLACEHOLDER]` |

---

## Comparison with the Paper

| Aspect                       | Original Paper             | This Implementation                  |
| ---------------------------- | -------------------------- | ------------------------------------ |
| Dataset                      | MNIST                      | MNIST                                |
| Main hidden units            | 500                        | 500                                  |
| Main latent representation   | Multiple configurations    | 20D                                  |
| Batch size                   | 100                        | 100                                  |
| Latent samples               | L = 1                      | L = 1                                |
| Optimizer                    | Adagrad                    | Adagrad                              |
| Decoder                      | Bernoulli for MNIST        | Bernoulli                            |
| Reparameterization           | Gaussian                   | Gaussian                             |
| 2D visualization             | Yes                        | Yes                                  |
| Image generation             | Yes                        | Yes                                  |
| Latent dimension experiments | Yes                        | Yes                                  |
| Wake-Sleep comparison        | Yes                        | Lightweight baseline                 |
| Marginal likelihood          | Low-dimensional experiment | Low-dimensional miniature experiment |
| Exact numerical reproduction | —                          | Not claimed                          |

The project focuses on reproducing the core methodology and understanding the underlying implementation rather than claiming an exact numerical reproduction of every experiment.

The marginal likelihood experiment is particularly constrained by computational resources and should therefore be interpreted as an experimental implementation rather than a validated reproduction of the paper's reported numerical results.

---

## What I Learned

### 1. Variational Inference

The true posterior `p(z|x)` is often intractable. A parameterized distribution `qφ(z|x)` can instead be optimized to approximate it.

### 2. ELBO

The ELBO provides a tractable objective combining:

* Reconstruction quality
* Regularization of the latent distribution

This connects probabilistic inference with neural-network optimization.

### 3. Reparameterization Trick

The transformation:

```text
z = μ + σ ⊙ ε
```

separates randomness from the learnable parameters and enables gradient-based optimization.

### 4. Latent Representations

A VAE learns a probability distribution over latent representations rather than simply mapping each input to a deterministic vector.

### 5. Generative Modelling

Once the latent distribution is learned, new observations can be generated by sampling from the prior and passing those samples through the decoder.

### 6. Latent-Space Continuity

Nearby latent points tend to produce related outputs, providing an interpretable geometric view of the learned generative model.

### 7. Evaluation Beyond Reconstruction

Good reconstruction quality alone does not guarantee a well-calibrated probabilistic generative model. ELBO and marginal likelihood provide additional probabilistic perspectives.

### 8. Research Implementation

Implementing the paper from its mathematical formulation highlighted the distinction between:

* Reproducing the core algorithm
* Reproducing an experimental setup
* Reproducing exact numerical results

---

## Future Improvements

* Explicit implementation of the paper's MAP/weight-decay objective
* Systematic hyperparameter studies
* Convolutional VAE architectures
* More extensive HMC convergence diagnostics
* More robust marginal likelihood estimators
* Larger Monte Carlo sample sizes
* More complete Wake-Sleep and MCEM comparisons
* Experiments on additional datasets
* Larger-scale reproduction of the original experiments

---

## Repository Structure

```text
Auto-Encoding-Variational-Bayes/
│
├── notes/
│   └── summary.md
│
├── src/
│   └── Auto_Encoding_Variational_Bayes.ipynb
│
├── paper/
│   └── 1312.6114v11.pdf
│
└── README.md
```

* [Project Repository](https://github.com/ArwaSaluji01/Auto-Encoding-Variational-Bayes)
* [Notebook](https://github.com/ArwaSaluji01/Auto-Encoding-Variational-Bayes/blob/main/src/Auto_Encoding_Variational_Bayes.ipynb)
* [Original Paper PDF](https://github.com/ArwaSaluji01/Auto-Encoding-Variational-Bayes/blob/main/paper/1312.6114v11.pdf)

---

## References

**Kingma, D. P., & Welling, M. (2013).**
*Auto-Encoding Variational Bayes.*
arXiv:1312.6114.

**Kingma, D. P., & Welling, M. (2014).**
*Auto-Encoding Variational Bayes.*
International Conference on Learning Representations (ICLR).
