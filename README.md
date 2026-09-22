# Auto-Encoding Variational Bayes

A from-scratch TensorFlow/Keras implementation of **Auto-Encoding Variational Bayes (AEVB)** based on the paper by Diederik P. Kingma and Max Welling.

The project implements the core ideas behind the Variational Autoencoder (VAE), including probabilistic encoding, the reparameterization trick, ELBO optimization, latent-space representation, generative sampling, and marginal likelihood estimation.

The implementation is developed progressively from the mathematical formulation in the original paper to a complete working model on MNIST.

---

## Project Overview

Variational inference provides a way to approximate otherwise intractable posterior distributions in probabilistic generative models.

The AEVB framework introduced by Kingma and Welling combines:

- Variational inference
- The Stochastic Gradient Variational Bayes (SGVB) estimator
- The reparameterization trick
- Neural-network-based recognition and generative models

The resulting architecture is commonly known as the **Variational Autoencoder (VAE)**.

This project implements AEVB from the underlying equations rather than relying on a pre-built VAE implementation.

The complete pipeline is:

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

**For generation:**

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

arXiv:1312.6114

Original paper — Auto-Encoding Variational Bayes

The implementation follows the mathematical formulation and experimental ideas presented in the original paper.

---

## Dataset
**MNIST**

The project uses the **MNIST handwritten digit dataset** containing 28 × 28 grayscale images of handwritten digits from 0 to 9.

Each image is:

- Normalized to the range [0, 1]
- Flattened from 28 × 28 to 784 dimensions
- Modelled using a Bernoulli decoder

The standard TensorFlow/Keras MNIST train/test split is used.

---

## Model Architecture
**Main AEVB Model**

The primary model uses:

| Component | Configuration |
|---|---|
| Input dimension | 784 |
| Hidden units | 500 |
| Latent dimensions | 20 |
| Decoder output | 784 |
| Decoder activation | Sigmoid |
| Hidden activation | Tanh |
| Batch size | 100 |
| Latent samples per datapoint | 1 |
| Optimizer | Adagrad |
| Learning rate | 0.01 |

The architecture follows the main MNIST configuration described in the paper, including 500 hidden units, minibatches of 100, one latent sample per datapoint, and Adagrad optimization.

---

## Core Mathematical Objective

The model optimizes the Evidence Lower Bound (ELBO):

L(θ, φ; x)
=
E_q[log pθ(x,z) - log qφ(z|x)]

which can be written as:

L(θ, φ; x)
=
-DKL(qφ(z|x) || pθ(z))
+
Eq[log pθ(x|z)]

The loss minimized during training is therefore:

Negative ELBO
=
KL divergence
-
Reconstruction log likelihood

The encoder produces:

μ
log(σ²)

and the latent variable is sampled using the reparameterization:

z = μ + σ ⊙ ε

ε ~ N(0, I)

This makes the stochastic sampling operation differentiable with respect to the encoder parameters.

---

## Implementation Levels

The notebook is organized progressively:

**Level 1 — Environment and Data Preprocessing**
- TensorFlow setup
- Random seeds
- MNIST loading
- Normalization
- Flattening
- Dataset batching

**Level 2 — Probabilistic Encoder**
- Neural-network encoder
- Mean estimation
- Log-variance estimation

**Level 3 — Reparameterization**
- Gaussian latent sampling
- Reparameterization trick
- Stochasticity verification

**Level 4 — Probabilistic Decoder**
- Neural-network decoder
- Bernoulli output distribution
- Image reconstruction

**Level 5 — ELBO and Loss**
- KL divergence
- Bernoulli reconstruction likelihood
- ELBO computation
- Negative ELBO loss

**Level 6 — AEVB Training**
- Gradient-based optimization
- Adagrad
- Mini-batch training
- ELBO monitoring

**Level 7 — Evaluation**
- Test-set ELBO
- KL divergence
- Reconstruction likelihood
- Reconstruction quality
- MSE as an auxiliary metric

**Level 8 — 2D Latent Space**
A separate 2-dimensional VAE is trained to visualize how MNIST samples are organized in latent space.

**Level 9 — Generation**
The trained decoder is used to:
- Generate new MNIST images from random latent vectors
- Generate related images from nearby latent points
- Explore generation using the 2D latent model

**Level 10 — Marginal Likelihood**
A separate low-dimensional model with:
- 100 hidden units
- 3 latent variables
is used to explore the paper's MCMC-based marginal likelihood estimator.

The paper specifically uses this low-dimensional configuration because its estimator becomes unreliable for higher-dimensional latent spaces.

---

## Results

**Main 20D AEVB Model**
The model was trained for 20 epochs.
Training Negative ELBO
- Epoch 01: 166.42
- Epoch 05: 126.01
- Epoch 10: 121.06
- Epoch 15: 118.13
- Epoch 20: 115.93

The negative ELBO decreased consistently throughout training.
- Test Evaluation
- Test Negative ELBO: 114.74
- Mean KL divergence: 26.14
- Mean reconstruction term: -88.61
- ELBO: -114.75
- Reconstruction

The trained model produces recognizable reconstructions of MNIST digits.

The reconstruction objective is based on the Bernoulli log likelihood rather than MSE. MSE was calculated only as an additional diagnostic metric.

Reconstruction MSE: 0.115173

--- 

**2D Latent Representation**
A separate 2-dimensional model was trained for latent-space visualization.
- Epoch 01/15: 201.35
- Epoch 05/15: 179.40
- Epoch 10/15: 176.26
- Epoch 15/15: 173.60

The resulting latent representation contains a continuous structure in which different MNIST digits occupy different regions, although substantial overlap remains.

This experiment demonstrates the use of the learned recognition model for low-dimensional representation and visualization.

---

**Image Generation**
The trained 20-dimensional decoder was sampled using:
z ~ N(0, I)

**Results:**

- Generated latent vectors: (20, 20)
- Generated images:         (20, 784)

The generated samples demonstrate that the decoder can produce new digit-like images without receiving an input image.

Sampling nearby latent vectors also produced visually related outputs, demonstrating the continuous structure of the learned latent space.

---

**Marginal Likelihood Experiment**
A separate 3-dimensional model was trained for the marginal likelihood experiment.
- Hidden units: 100
- Latent dimensions: 3
- Training epochs: 15

Training negative ELBO:
- 203.20 → 165.48

Hybrid Monte Carlo sampling produced:
- Posterior samples: (50, 3)
- HMC acceptance rate: 1.0

The implemented estimator produced:
Estimated log marginal likelihood:
-124.52

ELBO estimate:
-115.24

---

**Interpretation**
The marginal-likelihood estimator should not be treated as a validated numerical reproduction of the paper's reported results.

The estimated value was lower than the ELBO, whereas the theoretical marginal log likelihood should be greater than or equal to the ELBO.

This indicates that the practical MCMC/KDE estimator used in this implementation requires further tuning and validation.

The paper itself notes that the estimator requires sufficiently many samples and is intended for very low-dimensional latent spaces.

---

## Comparison with the Paper

| Aspect | Original Paper | This Implementation |
|---|---|---|
| **Dataset** | MNIST | MNIST |
| **Main hidden units** | 500 | 500 |
| **Main latent representation** | Multiple configurations | 20D |
| **Batch size** | 100 | 100 |
| **Latent samples** | $L = 1$ | $L = 1$ |
| **Optimizer** | Adagrad | Adagrad |
| **Decoder** | Bernoulli for MNIST | Bernoulli |
| **Reparameterization** | Gaussian | Gaussian |
| **2D visualization** | Yes | Yes |
| **Image generation** | Yes | Yes |
| **Marginal likelihood** | 3D / 100 hidden units | 3D / 100 hidden units |
| **HMC marginal-likelihood experiment** | Yes | Implemented experimentally |
| **Exact numerical reproduction** | — | Not claimed |

The implementation focuses on reproducing the core methodology and learning behaviour rather than claiming exact reproduction of every numerical result from the original experiments.

The paper reports that its marginal-likelihood estimator used 50 posterior samples and 4 HMC leapfrog steps, which were also used here as the starting configuration.

---

## What I Learned

This implementation provided a practical understanding of variational inference and generative modelling from both mathematical and implementation perspectives.

**1. Variational Inference**

The true posterior p(z|x) is often intractable. A parameterized distribution qφ(z|x) can instead be optimized to approximate it.

**2. ELBO**

The ELBO provides a tractable objective that combines:

Reconstruction quality
Regularization of the latent distribution

This connects probabilistic inference with neural-network optimization.

**3. Reparameterization Trick**

Sampling directly from a parameterized Gaussian prevents straightforward backpropagation.

The transformation:

z = μ + σ ⊙ ε

separates randomness from the learnable parameters and makes gradient-based optimization possible.

**4. Latent Representations**

A VAE does not simply compress an image into a deterministic vector. It learns a probability distribution over latent representations.

**5. Generative Modelling**

Once the latent distribution is learned, new observations can be generated by sampling from the prior and passing the samples through the decoder.

**6. Latent-Space Continuity**

Nearby latent points tend to produce related outputs, providing an interpretable geometric view of the learned generative model.

**7. Evaluation Beyond Reconstruction**

A model that reconstructs images well is not necessarily a well-calibrated probabilistic generative model. ELBO and marginal likelihood provide probabilistic evaluation perspectives beyond reconstruction error.

**8. Practical Research Implementation**

Implementing the paper from its mathematical formulation highlighted the difference between:
- reproducing the core algorithm.
- reproducing an experimental setup.
- and reproducing exact numerical results.

---

## Conclusion

This project implements the central ideas of Auto-Encoding Variational Bayes from the mathematical formulation through training and generative evaluation.

The implementation demonstrates how:

```text

Variational Inference
        +
Reparameterization
        +
Neural Networks
        ↓
Efficient Approximate Posterior Inference
        +
Generative Modelling

```

can be combined into the architecture now widely known as the Variational Autoencoder.

The project also highlights an important aspect of research-oriented implementation: distinguishing between a successful reproduction of the core method and an exact reproduction of every experimental result.

---

## References

**Kingma, D. P., & Welling, M. (2013).**
Auto-Encoding Variational Bayes.
arXiv:1312.6114.

**Kingma, D. P., & Welling, M. (2014).**
Auto-Encoding Variational Bayes.
International Conference on Learning Representations (ICLR).
