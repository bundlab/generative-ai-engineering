# Chapter 01: Generative AI Fundamentals

## Overview & Learning Objectives
This chapter covers the mathematical foundations and core operational mechanics of Generative AI. By the end of this chapter, you will understand:
1. The mathematical distinction between Discriminative and Generative AI models.
2. The exact probability formulations behind autoregressive text generation.
3. How logits are transformed using activation functions and sampling algorithms.
4. The mechanics and mathematical parameters controlling generation randomness ($T$, top-$k$, top-$p$).

---

## 1. Discriminative vs. Generative Models

Machine learning tasks generally partition into two fundamental statistical modeling paradigms:

### Discriminative Models
Discriminative models learn the conditional probability distribution $P(Y \mid X)$, modeling the boundary between classes given input features $X$. They optimize directly for decision boundaries without attempting to model how data features are distributed.

$$\text{Discriminative Objective: } \max_{\theta} P(Y \mid X; \theta)$$

* **Examples:** Logistic Regression, Support Vector Machines (SVMs), Convolutional Neural Networks (CNNs) for classification, Decision Trees.
* **Primary Tasks:** Spam detection, sentiment classification, object recognition.

### Generative Models
Generative models capture the joint probability distribution $P(X, Y)$ or the marginal distribution $P(X)$ over the input space. By learning the underlying data distribution, generative models can sample new, synthetic data points that resemble the training distribution $P_{\text{data}}(X)$.

$$\text{Generative Objective: } \max_{\theta} P(X, Y; \theta) \quad \text{or} \quad \max_{\theta} P(X; \theta)$$

Via Bayes' Theorem, a generative model can also derive conditional probabilities:

$$P(Y \mid X) = \frac{P(X, Y)}{P(X)} = \frac{P(X \mid Y)P(Y)}{\sum_{y'} P(X \mid y')P(y')}$$

* **Examples:** Autoregressive Transformers (GPT-4, Llama), Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models.
* **Primary Tasks:** Text completion, image synthesis, audio generation, data augmentation.

---

## 2. Autoregressive Language Modeling

Modern Large Language Models (LLMs) operate as **autoregressive generative models**. Given a sequence of $N$ tokens $X = (x_1, x_2, \dots, x_N)$, the joint probability of the sequence is factorized into a product of conditional probabilities using the probability chain rule:

$$P(X) = P(x_1, x_2, \dots, x_N) = \prod_{i=1}^{N} P(x_i \mid x_1, x_2, \dots, x_{i-1})$$

During inference, the model predicts the probability distribution for token $x_i$ conditioned on all previous tokens $x_{<i}$. Once token $x_i$ is sampled, it is appended to the context window to predict token $x_{i+1}$.

---

## 3. Logits and the Softmax Function

An LLM's final neural network layer produces a vector of unnormalized log-probabilities called **logits** $\mathbf{z} \in \mathbb{R}^{|V|}$, where $|V|$ represents the size of the token vocabulary.

To convert logits into a valid categorical probability distribution $\mathbf{p} \in \mathbb{R}^{|V|}$ where $\sum_{j=1}^{|V|} p_j = 1$, the **Softmax** function is applied:

$$p_i = \text{Softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{|V|} \exp(z_j)}$$

---

## 4. Token Sampling & Decoding Strategies

Raw probabilities obtained from Softmax can be sampled using several strategy variations to balance deterministic precision against creative diversity.

### A. Temperature Scaling ($T$)
Temperature $T > 0$ is a hyperparameter that scales the raw logits $\mathbf{z}$ before passing them into the Softmax transformation:

$$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$


```

```
      Logits (z) ---> [ Divide by Temperature T ] ---> Softmax ---> Probabilities (p)

```

```

* **Low Temperature ($T \to 0$):** Sharpens the distribution. Differences between raw logits are amplified, forcing $p_{\max} \to 1.0$. The model becomes deterministic and focused.
* **Default Temperature ($T = 1.0$):** Preserves the exact learned probability distribution of the model.
* **High Temperature ($T > 1.0$):** Flattens the distribution towards a uniform distribution. Low-probability tokens receive a higher relative chance of being sampled, introducing creativity or noise.

### B. Greedy Decoding vs. Random Sampling
* **Greedy Decoding:** Selects the token with the absolute maximum probability: $x_i = \arg\max_{j} (p_j)$. This is equivalent to setting $T \to 0$.
* **Random Ancestral Sampling:** Samples token $x_i$ directly according to categorical distribution $\mathbf{p}$.

### C. Top-$k$ Sampling
Top-$k$ sampling restricts token selection strictly to the $k$ most probable tokens in the vocabulary:

1. Sort logits or probabilities in descending order: $p_1 \ge p_2 \ge \dots \ge p_{|V|}$.
2. Truncate the vocabulary distribution set to top $k$ tokens $V_k = \{x_1, x_2, \dots, x_k\}$.
3. Discard all tokens outside $V_k$ by setting their probabilities to zero.
4. Renormalize remaining probabilities so their sum equals $1.0$:

$$p'_i = \frac{p_i}{\sum_{x_j \in V_k} p_j} \quad \text{for } x_i \in V_k$$

### D. Top-$p$ (Nucleus) Sampling
Top-$p$ sampling dynamically adjusts the candidate token set based on cumulative confidence rather than a fixed integer count:

1. Sort probabilities in descending order: $p_1 \ge p_2 \ge \dots \ge p_{|V|}$.
2. Find the smallest index $M$ such that the cumulative sum exceeds threshold $p \in (0, 1]$:

$$\sum_{j=1}^{M} p_j \ge p$$

3. Define candidate subset $V_p = \{x_1, x_2, \dots, x_M\}$.
4. Discard tokens outside $V_p$ and renormalize the candidate set.

---

## 5. Visualizing Decoding Transformations

| Method | Parameters | Effect on Probability Distribution | Ideal Use Case |
| :--- | :--- | :--- | :--- |
| **Greedy Search** | $T \to 0$ | Collapses distribution into a single one-hot peak. | Code generation, math, factual lookup. |
| **Temperature** | $T = 0.2$ to $0.7$ | Sharpened peak; suppresses tail distribution. | Structured technical writing, QA. |
| **Temperature** | $T = 0.8$ to $1.2$ | Smoothed peak; elevates mid/tail tokens. | Brainstorming, creative story generation. |
| **Top-$k$** | $k = 40$ | Truncates tail at fixed boundary $k$. | Preventing nonsensical/out-of-distribution tokens. |
| **Top-$p$** | $p = 0.9$ | Truncates dynamically based on confidence mass. | Natural conversational dialogues. |

---

## 6. Mathematical Example: Sampling Calculation

Given a simplified vocabulary $V = \{\text{"AI"}, \text{"ML"}, \text{"Code"}, \text{"Data"}\}$ with raw logits $\mathbf{z} = [4.0, 2.0, 1.0, 0.0]$:

### 1. Unscaled Softmax ($T = 1.0$)
Exponentials: $\exp(\mathbf{z}) = [e^4, e^2, e^1, e^0] \approx [54.60, 7.39, 2.72, 1.00]$
Sum $= 65.71$
$$\mathbf{p}_{T=1.0} = [0.831, 0.112, 0.041, 0.015]$$

### 2. Temperature Scaled Softmax ($T = 0.5$)
Scaled Logits $\mathbf{z} / 0.5 = [8.0, 4.0, 2.0, 0.0]$
Exponentials: $\exp(\mathbf{z}/0.5) = [e^8, e^4, e^2, e^0] \approx [2980.96, 54.60, 7.39, 1.00]$
Sum $= 3043.95$
$$\mathbf{p}_{T=0.5} = [0.979, 0.018, 0.002, 0.0003]$$
*(Notice how lowering $T$ sharply concentrates probability on the top token).*

---