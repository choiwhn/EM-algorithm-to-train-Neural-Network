# EM-algorithm-to-train-Neural-Network

Overview

This repository implements a one‑hidden‑layer Multilayer Perceptron whose parameters are learned with the Expectation–Maximisation (EM) algorithm instead of the usual back‑propagation.  From page 8 onward of the accompanying project report we formalise the MLP as a latent‑variable model and derive an EM routine that alternates between probabilistic inference over hidden units (E‑step) and maximum‑likelihood parameter updates (M‑step).



Mathematical Formulation

We tackle a g‑class classification problem with input dimension $p$ and hidden width $m$.  Denote

training set: $\mathcal{D}={(\mathbf{x}_j,\mathbf{y}j)}{j=1}^n$ where $\mathbf{x}_j\in\mathbb{R}^{p}$ and $\mathbf{y}_j\in{\mathbf{e}_1,\dots,\mathbf{e}_g}$;

latent binary activations: $\mathbf{Z}j=(Z{1j},\dots,Z_{mj})\in{0,1}^m$.

Hidden layer



where $\tilde{\mathbf{x}}_j=[1,\mathbf{x}_j^{\top}]^{\top}$ incorporates a bias term and $\sigma(z)=\frac{e^{z}}{1+e^{z}}$ is the logistic sigmoid.

Output layer

Conditioned on $\mathbf{Z}_j$ we assume



with augmented hidden vector $\tilde{\mathbf{Z}}_j=[1,\mathbf{Z}_j^{\top}]^{\top}$ and parameter vectors $\mathbf{v}_i\in\mathbb{R}^{m+1}$.

Complete‑data log‑likelihood

Let $u_{hj}=P(Z_{hj}=1\mid\mathbf{x}j)$ and $o{ij}=P(Y_{ij}=1\mid\mathbf{Z}_j)$.  The log‑likelihood of the complete data $(\mathbf{Y},\mathbf{Z})$ is



where $\Psi;{=};(\mathbf{w}_1,\dots,\mathbf{w}_m,\mathbf{v}1,\dots,\mathbf{v}{g-1})$ collects all weights (the $g$‑th $\mathbf{v}_g$ is fixed to zero for identifiability).

