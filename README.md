# EM-algorithm-to-train-Neural-Network

This repository contains an implementation of the multi-class neural network trained via the EM algorithm as an alternative to backpropagation a model. we implemented both models entirely with NumPy and evaluated their performance on the Iris dataset, comparing EM-based training with stochastic gradient descent(SGD)

This project

   - implements a one‑hidden‑layer multi‑layer perceptron (MLP) trained by EM,

   - provides an equivalent MLP trained by SGD,

   - keeps both implementations entirely in NumPy (no deep‑learning frameworks), and

   - evaluates learning curves, convergence speed, and classification accuracy on the Iris data.

---

## EM Algorithm and Multiclass Classification

Assume multiclass classification with $g$ groups, $G_1, ..., G_g$

Problem: Infer the unknown membership of an unclassified entity with feature vector of $p$-dimensions 

Let $(x_1^T, y_1^T)^T,\;\dots,\;(x_n^T, y_n^T)^T$ be the $n$ examples available for training the neural network and $z$ be missing data or latent variable



### E‑step

Compute the Q‑function as:

$$Q(\boldsymbol{\Psi},\boldsymbol{\Psi}^{(k)}) = E_{\boldsymbol{\Psi}^{(k)}}[\log (L_c(\boldsymbol{\Psi};y,z,x))|y,x]$$

$$\log (L_c(\boldsymbol{\Psi};y,z,x)) \propto \log (p(Y,Z|x;\boldsymbol{\Psi})) = \log (p(Y|x,z;\boldsymbol{\Psi})) + \log (p(Z|x;\boldsymbol{\Psi}))$$



### M‑step

$\Psi^{(k)}$ is updated by taking $\Psi^{(k+1)}$ be the value of $\Psi$ that maximizes $Q$-function

$$\Psi^{(k+1)} = argmax_{\Psi}(Q(\Psi;\Psi^{(k)})) = argmax_{\Psi}(\log (p(Y|x,z;\boldsymbol{\Psi})) + \log (p(Z|x;\boldsymbol{\Psi})))$$

---

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

---

**Goal:** Find ML estimate for unknown parameters $\Psi = (w_1^T,w_2^T,\ldots,w_m^T,v_1^T,v_2^T,\ldots,v_{g-1}^T)^T$ through **EM steps** using complete-data log likelihood $L_c(\Psi;y,z,x)$

Recall that

$$\log (L_c{\Psi;y,z,x}) \propto \log (p(Y,Z|x;\Psi)) = \log (p(Y|x,z;\Psi)) + \log (p(Z|x; \Psi))$$

Then, the complete-data log likelihood for $\Psi$ is

$$\sum_{j=1}^n\left[ \sum_{h=1}^m [z_{hj}\  \log {u_{hj} \over 1-u_{hj}} + \log(1-u_{hj})] + \sum_{i=1}^g y_{ij}\ \log o_{ij} \right],$$

where $z_{hj}\  \log ({u_{hj} (1-u_{hj})^{-1}}) + \log(1-u_{hj})$ is linear in $z$ whereas $\sum_{i=1}^g y_{ij}\ \log o_{ij}$ is nonlinear in $z$. 

We will calculate the expectation of the complete-data log likelihood $\log (L_c(\Psi;y,z,x))$ conditional on the current estimate $\Psi^{(k)}$ and the observed input and output vectors.

## E-step & M-step

- **E-step :**

Compute the Q-function 

$$Q(\Psi;\Psi^{(k)}) = E_{\Psi^{(k)}} \left[ \log (L_c(\Psi;y,z,x)) | y,x\right] = \sum_{j=1}^n \sum_{h=1}^m \left[ E_{\Psi^{(k)}} (Z_{hj} | y,x) \log{u_{hj} \over 1-u_{hj}} + \log(1-u_{hj})\right] + \sum_{j=1}^{n}\sum_{i=1}^g y_{ij} E_{\Psi^{(k)}}(o_{ij}|y,x) =: Q_w + Q_v$$

Marginalizing out all possible $Z$ in complete-data log likelihood yields the following Q-function. 

We can update $w$ and $v$ by finding such $w$ and $v$ which maximize the $Q_w$ and $Q_v$, respectively.

- **M-step :**

Set the differentiationa of $Q_w$ with respect to $w$ as 0..

Then, we take $w_h^{(k+1)}=argmax Q_w$.

$${\nabla_{w_h} Q_w} = \sum_{j=1}\left[E_{\Psi^{(k)}} (Z_{hj}|y,x) - u_{hj} \right]x_j =0, \quad (h=1,\ldots,m),$$

where 

$$E_{\Psi^{(k)}} (Z_{hj}|y,x) = { \sum_{z_j : z_{hj} =1 }p_{\Psi^{(k)}}(x_j,y_j,z_j)  \over \sum_{z_j}p_{\Psi^{(k)}}(x_j,y_j,z_j) }$$

and 

$$p_{\Psi^{(k)}}(x_j,y_j,z_j) = \prod_{h=1}^m u_{hj}^{z_{hj}}(1-u_{hj})^{(1-z_{hj})} \prod_{i=1}^g o_{ij}^{y_{ij}}.$$

Set the gradient of $Q_v$ with respect to $v$ as 0, and then we take $v_i^{(k+1)} = argmax Q_v$.

$$\nabla_{v_i}Q_v = \sum_{j=1}^n \left[y_{ij} E_{\Psi^{(k)}} (Z_{hj}|y,x) - {\sum_{z_j : z_{hj} =1 } o_{ij} p_{\Psi^{(k)}}(x_j,y_j,z_j)  \over \sum_{z_j}p_{\Psi^{(k)}}(x_j,y_j,z_j)} \right]=0.$$

We use the gradient descent method since we cannot obtain our new parameters as a closed form.

---

## implementation details

   - Iris labels are one-hot encoded.

   - t

   - Activations: sigmoid in the hidden layer; softmax in the output layer.

   - the method '**E_step_W**' computes $\nabla_{w_h}Q_w$ for the hidden weights. likewise, the method '**E_step_V**' computes $\nabla_{v_i}Q_v$ for the output weights.

   - the method '**M_step**' updates all parameters via gradient descent.

---

## Experiment result

### hyperparameter for each model & convergence rate comparison plot
![image](https://github.com/user-attachments/assets/b88d8ce2-170e-4fd4-9e81-5d0bdc68a188)

note that convergence rate via EM algorithm is remarkblely slow than backpropagation.

### Accuracy & new prediction sampling (10 from test dataset)
![image](https://github.com/user-attachments/assets/ff98baab-ef64-4508-9a6a-2bcc1bfa3bcb)

Accuracy(SGD) = 96.67%  & Accuracy(EM) = 93.33%

---

## Discussion and Conclusion

- **Areas for Improvement in the Code Implementation Process**

1. Computational efficiency in E-step.
   - vectorized operations like matrix multiplication are difficult to apply when implementing the E-step
   - Unnecessary repetition was introduced during the code implementation

2. Hyperparameter tuning.
   - for optimizing hyperparameters, Using Cross-validation would be a way.

- **Comparison of the Backpropagation Method and Neural Network Training**

Backpropagation(SGD)

1. Fast convergence: 
   - Gradient-descent methods typically converge more quickly and handle complex loss landscapes well.

2. Efficient gradient computation:
   - The algorithm repeatedly computes the gradient of the loss function, making it generally efficient.

3. Distribution-agnostic error signal:
   - Because the loss depends only on the difference between predicted and true values, it works regardless of the data’s underlying distribution.

Expectation–Maximization (EM)

1. Slower, alternating updates:
   - Alternating between E-steps and M-steps can slow convergence—especially in high-dimensional problems where finding a stable solution may be difficult.

2. High computational cost:
   - At each iteration, EM must evaluate all possible combinations of the latent variables—leading to a time complexity of $O(2^m)$ — which becomes prohibitively expensive and inefficient on large datasets
   - Optimizing the updated parameters in the M-step via gradient descent incurs additional computational cost

3. Model‐assumption sensitivity:
   - Since EM relies on estimating latent variables under assumed distributions, performance can degrade if those assumptions poorly reflect the true data.

- **Conclusion**

Hence, EM becomes infeasible beyond shallow networks in Deep Learning.

1. Explosive time complexity:
In a shallow network with $m$ hidden units, the E-step already incurs an $O(2^m*)$ cost to enumerate all possible hidden-unit configurations. Adding more layers multiplies these possibilities, so runtime grows explosively.

2. Slow Convergence:
Given its substantial time complexity, the EM algorithm becomes even less efficient when applied to deep learning.
