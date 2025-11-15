# ------------------------------
# PyTorch vs Optax AdamW check
# ------------------------------
import torch
import optax
import jax
import jax.numpy as jnp
import numpy as np

# --- Config ---
lr = 1e-3
betas = (0.9, 0.95)
eps = 1e-8
eps_root = 0.0
weight_decay = 0.1

# --- Fake parameters and grads ---
np.random.seed(0)
param_init = np.random.randn(3, 3).astype(np.float32)
grad = np.random.randn(3, 3).astype(np.float32)

# -------------------------------
# PyTorch AdamW
# -------------------------------
p = torch.tensor(param_init.copy(), requires_grad=True)
opt = torch.optim.AdamW([p], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# do one step
p.grad = torch.tensor(grad.copy())
opt.step()
opt.zero_grad()

torch_updated = p.detach().numpy()

# -------------------------------
# Optax AdamW
# -------------------------------
optax_opt = optax.adamw(
    learning_rate=lr,
    b1=betas[0],
    b2=betas[1],
    eps=eps,
    eps_root=eps_root,
    weight_decay=weight_decay,
)

# init state
params_jax = jnp.array(param_init)
opt_state = optax_opt.init(params_jax)

# apply one step
updates, opt_state = optax_opt.update(jnp.array(grad), opt_state, params_jax)
jax_updated = optax.apply_updates(params_jax, updates)

# -------------------------------
# Compare
# -------------------------------
print("Torch updated:\n", torch_updated)
print("JAX updated:\n", np.array(jax_updated))
print("Diff (max abs):", np.max(np.abs(torch_updated - np.array(jax_updated))))