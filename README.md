# EM-algorithm-to-train-Neural-Network


## EM Algorithm and Multiclass Classification
Assume multiclass classification with $g$ groups, $G_1, ..., G_g$

Problem: Infer the unknown membership of an unclassified entity with feature vector of $p$-dimensions 

Let $(x_1^T, y_1^T)^T,\;\dots,\;(x_n^T, y_n^T)^T$ be the $n$ examples available for training the neural network and $z$ be missing data or latent variable



### E‑step

Compute the Q‑function as:

$$Q(\boldsymbol{\Psi},\boldsymbol{\Psi}^{(k)}) = E_{\boldsymbol{\Psi}^{(k)}}[log L_c(\boldsymbol{\Psi};y,z,x)|y,x]$$

$$log L_c(\boldsymbol{\Psi};y,z,x) \propto log P(Y,Z|x;\boldsymbol{\Psi}) = log P(Y|x,z;\boldsymbol{\Psi}) + log P(Z|x;\boldsymbol{\Psi})$$



### M‑step

$\Psi^{(k)}$ is updated by taking $\Psi^{(k+1)}$ be the value of $\Psi$ that maximizes $Q$-function

$$\Psi^{(k+1)} = argmax_{\Psi}(Q(\Psi;\Psi^{(k)}))$$

## EM in MLP
consider MLP(Multi-Layer Perceptron) neural network with one hidden layer of m units.

Note that sigmoid and softmax function is used as activation function for each layer respectively.

![image](https://github.com/user-attachments/assets/54cd486c-841e-4079-8696-966fda31ff42)

assume $z_{hj}$ be the realization of the zero-one random variable $Z_{hj}$. $h=1,...,m, j=1,...,n$

let Synaptic weight of the $h$ th hidden unit as: $w_h = (w_{h0},w_{h1},...,w_{hp},)$, where bias term $w_{h0}$ is included in $w_h$ by adding a constant input $x_{0j} = 1$

then, the 4conditional distribution of $Z_{hj}$ given $x_j$ is as:

$$P(Z_{hj}=1|x_j) = \frac{exp(w_h^Tx_j)}{1+exp(w_h^Tx_j)}$$

![image](https://github.com/user-attachments/assets/750c75d5-a8f3-4727-854b-8df02d1790f6)

similarly, let Synaptic weight of the $i$ th output unit as: $v_i = (v_{i0},v_{i1},...,v_{im},), i=1,...,g$

then, the conditional distribution of $Y_{ij}$ given $x_j, z_j$ is as:

$$P(Y_{ij}=1|x_j,z_j) = \frac{exp(v_i^Tz_j)}{\sum_{r=1}^g exp(v_r^Tz_j)}$$

**Goal:** Find ML estimate for unknown parameters $\Psi = (w_1^T,w_2^T,\ldots,w_m^T,v_1^T,v_2^T,\ldots,v_{g-1}^T)^T$ through **EM steps** using complete-data log likelihood $L_c(\Psi;y,z,x)$

Recall that

$$\log L_c{\Psi;y,z,x} \propto \log pr(Y,Z|x;\Psi) = \log pr(Y|x,z;\Psi) + \log pr(Z|x; \Psi)$$

Then, the complete-data log likelihood for $\Psi$ is

$$\sum_{j=1}^n\ㅣㄷㄽ[ \sum_{h=1}^m [z_{hj}\  \log {u_{hj} \over 1-u_{hj}} + \log(1-u_{hj}) + \sum_{i=1}^g y_{ij}\ \log o_{ij} ]\right]$$




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

