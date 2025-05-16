# EM-algorithm-to-train-Neural-Network


## EM Algorithm and Multiclass Classification
Assume multiclass classification with $g$ groups, 

Problem: Infer the unknown membership of an unclassified entity with feature vector of $p$-dimensions 

Let $(x_1^T, y_1^T)^T,\;\dots,\;(x_n^T, y_n^T)^T$ be the $n$ examples available for training the neural network and $z$ be missing data or latent variable



### E‑step

Compute the Q‑function as:

$$log L_c(\mathbf{\psi};y,z,x) \propto log P(Y,Z|x;\boldsymbol{\psi}) = log P(Y|x,z;\boldsymbol{\psi}) + log P(Z|x;\boldsymbol{\psi})$$

$$Q(\boldsymbol{\psi},\boldsymbol{\psi}^{(k)}) = E_{\boldsymbol{\psi}^{(k)}}\{log L_c(\boldsymbol{\psi};y,z,x)|y,x\}$$


by marginalising over all $2^{m}$ latent configurations (or a Monte
Carlo approximation for large *m*).

### M‑step

Maximise $Q$ w\.r.t. $\Psi$.  Because closed‑form updates are
intractable, a gradient ascent step is applied separately to the *w*
and *v* blocks:

$$
  \mathbf w_h^{(k+1)} = \mathbf w_h^{(k)} + \eta\,\nabla_{\mathbf w_h} Q_w,
  \qquad
  \mathbf v_i^{(k+1)} = \mathbf v_i^{(k)} + \eta\,\nabla_{\mathbf v_i} Q_v.
$$

---

## Python Implementation Outline

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

## Experimental Results

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

## Back‑propagation vs EM

| Aspect                         | Back‑propagation               | Expectation–Maximisation                   |
| ------------------------------ | ------------------------------ | ------------------------------------------ |
| **Time complexity**            | $\mathcal O(npm)$ per epoch    | $\mathcal O(n\,2^{m})$ in the exact E‑step |
| **Convergence speed**          | Fast; efficient gradient usage | Slower; alternates E and M steps           |
| **Scalability**                | Deep architectures feasible    | Practically limited to shallow NNs         |
| **Distributional assumptions** | None                           | Requires correct latent‑variable model     |

> **Conclusion**   EM is educational but unlikely to out‑perform
> gradient‑based learning beyond very small networks.

---

## Potential Improvements

1. **E‑step efficiency**   Use sampling or variational approximations to
   avoid exhaustive enumeration.
2. **Hyper‑parameter tuning**   Optimise learning rate and hidden width
   $m$.
3. **Input normalisation**   Standardising features accelerates
   convergence and mitigates overflow/underflow in exponentials.

