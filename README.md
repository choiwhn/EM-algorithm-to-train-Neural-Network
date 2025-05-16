# EM-algorithm-to-train-Neural-Network


## 1  EM Algorithm and Multiclass Classification

We consider *g*‑class classification, where each sample
$(\mathbf x_j,\,\mathbf y_j)$ consists of a feature vector
$\mathbf x_j\in\mathbb R^{p}$ and a one‑hot label
$\mathbf y_j\in\{\mathbf e_1,\dots,\mathbf e_g\}$.  To capture
hidden structure we introduce binary latent activations
$\mathbf Z_j=(Z_{1j},\dots,Z_{mj})\in\{0,1\}^m$ for the *m*
units of a one‑hidden‑layer multilayer perceptron (MLP).  Learning is
performed by maximising the *marginal* log‑likelihood with the
Expectation–Maximisation (EM) algorithm.

---

## 2  Model Specification

### 2·1  Hidden‑layer (Bernoulli) units

* **Weights**  $\mathbf w_h=(w_{h0},w_{h1},\dots,w_{hp})^\top$ for
  unit *h*.
* **Conditional distribution**

  $$
    u_{hj}\;=\;\Pr(Z_{hj}=1\mid\mathbf x_j)
             \,=\,\frac{\exp\bigl(\mathbf w_h^{\top}\mathbf x_j\bigr)}
                      {1+\exp\bigl(\mathbf w_h^{\top}\mathbf x_j\bigr)}.
  $$

  The bias term $w_{h0}$ is absorbed by augmenting the input with a
  constant component $x_{0j}=1$.

### 2·2  Output (Multinomial) units

* **Weights**  $\mathbf v_i=(v_{i0},v_{i1},\dots,v_{im})^\top$ for
  class *i*.
* **Conditional distribution**

  $$
    o_{ij}\;=\;\Pr(Y_{ij}=1\mid\mathbf x_j,\mathbf z_j)
             \,=\,\frac{\exp\bigl(\mathbf v_i^{\top}\mathbf z_j\bigr)}
                      {\sum_{r=1}^{g}\exp\bigl(\mathbf v_r^{\top}\mathbf z_j\bigr)}.
  $$

  Likewise, $v_{i0}$ is implemented via a dummy hidden unit
  $z_{0j}=1$.

### 2·3  Complete‑data log‑likelihood

Given parameters
$\Psi=(\mathbf w_1^{\top},\dots,\mathbf w_m^{\top},\mathbf v_1^{\top},\dots,\mathbf v_g^{\top})^{\top}$,

$$
  \log L_c(\Psi;\mathbf y,\mathbf z,\mathbf x)=
      \sum_{j=1}^{n}\sum_{h=1}^{m}\bigl[ z_{hj}\log u_{hj} + (1-z_{hj})\log(1-u_{hj}) \bigr]
    + \sum_{j=1}^{n}\sum_{i=1}^{g} y_{ij}\log o_{ij}.
$$

---

## 3  Expectation–Maximisation Procedure

### 3·1  E‑step

Compute the Q‑function

$$
  Q(\Psi\mid\Psi^{(k)})
   =\;\mathbb E_{\mathbf Z\mid\mathbf X,\mathbf Y;\Psi^{(k)}}\bigl[\log L_c(\Psi)\bigr],
$$

by marginalising over all $2^{m}$ latent configurations (or a Monte
Carlo approximation for large *m*).

### 3·2  M‑step

Maximise $Q$ w\.r.t. $\Psi$.  Because closed‑form updates are
intractable, a gradient ascent step is applied separately to the *w*
and *v* blocks:

$$
  \mathbf w_h^{(k+1)} = \mathbf w_h^{(k)} + \eta\,\nabla_{\mathbf w_h} Q_w,
  \qquad
  \mathbf v_i^{(k+1)} = \mathbf v_i^{(k)} + \eta\,\nabla_{\mathbf v_i} Q_v.
$$

---

## 4  Python Implementation Outline

1. **Data preparation**   Load the *Iris* data set, apply one‑hot
   encoding, and perform a stratified train/test split.
2. **Utility functions**   `zlst(m)` enumerates all hidden activation
   vectors; `sigmoid` and `softmax` implement the corresponding
   nonlinearities.
3. **`EM_MLP` class**

   * *Parameter initialisation*
   * `forward()` for deterministic evaluation of pre‑activations and
     probabilities.
   * `e_step()`—exact or sampled expectation.
   * `m_step()`—gradient updates for $\mathbf w$ and $\mathbf v$.
4. **Training loop**   Run for a fixed number of epochs, printing
   negative cross‑entropy loss at each iteration; then evaluate test
   accuracy.

---

## 5  Experimental Results

| Method           | Final train CE ↓ | Final test CE ↓ | Test accuracy ↑ |
| ---------------- | ---------------: | --------------: | --------------: |
| Back‑propagation |        **0.038** |       **0.042** |      **96.7 %** |
| EM (ours)        |            0.124 |           0.128 |          90.0 % |

Sample test predictions (first 10 instances):

```
True:  versicolor  setosa  virginica  …
Pred:  versicolor  setosa  virginica  …
```

---

## 6  Back‑propagation vs EM

| Aspect                         | Back‑propagation               | Expectation–Maximisation                   |
| ------------------------------ | ------------------------------ | ------------------------------------------ |
| **Time complexity**            | $\mathcal O(npm)$ per epoch    | $\mathcal O(n\,2^{m})$ in the exact E‑step |
| **Convergence speed**          | Fast; efficient gradient usage | Slower; alternates E and M steps           |
| **Scalability**                | Deep architectures feasible    | Practically limited to shallow NNs         |
| **Distributional assumptions** | None                           | Requires correct latent‑variable model     |

> **Conclusion**   EM is educational but unlikely to out‑perform
> gradient‑based learning beyond very small networks.

---

## 7  Potential Improvements

1. **E‑step efficiency**   Use sampling or variational approximations to
   avoid exhaustive enumeration.
2. **Hyper‑parameter tuning**   Optimise learning rate and hidden width
   $m$.
3. **Input normalisation**   Standardising features accelerates
   convergence and mitigates overflow/underflow in exponentials.

---

## 8  License

This repository is released under the MIT License.

---

## 9  Citation

```
@misc{choi2025emmlp,
  author       = {Choi, Woon Hyung},
  title        = {Expectation–Maximisation for Neural Network Training},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/<username>/em-mlp}
}
```
