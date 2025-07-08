### Installation (CUDA Support)
```
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Run
```
(venv) ➜  jax-vpg git:(master) ✗ python3 main.py
```

## VPG Implementation Enhancements

This implementation extends vanilla VPG (Vanilla Policy Gradient) with several key improvements for better training stability and performance:

### 1. **Baseline Subtraction**
Instead of using raw returns R_t in the policy gradient, we compute advantages A_t = R_t - b where b = E[R_t] is the baseline. This reduces variance in the gradient estimates:

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A_t]
```

Implemented as `baseline = jnp.mean(jnp.array(returns))` and `advantage = ret - baseline`.

### 2. **Gradient Clipping**
Added gradient clipping to prevent gradient explosion. The gradient norm is clipped to a maximum value of 1.0:

```
grad = grad · min(1.0, 1.0/||grad||_2)
```

Implemented using `optax.clip_by_global_norm(1.0)` in the optimizer chain.

### 3. **Per-Episode Return Normalization**
Returns are normalized per episode to have zero mean and unit variance:

```
R'_t = (R_t - μ_e) / (σ_e + ε)
```

where μ_e and σ_e are the mean and standard deviation of returns in episode e. This prevents bias from episode length differences.

### 4. **Entropy Regularization**
Added entropy regularization to encourage exploration. The loss function becomes:

```
L(θ) = -log π_θ(a|s) · A_t - β · H(π_θ)
```

where H(π_θ) = -Σ_a π_θ(a|s) log π_θ(a|s) is the policy entropy and β = 0.01 is the entropy coefficient.