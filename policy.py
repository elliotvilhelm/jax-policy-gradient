import jax
import jax.numpy as jnp


def init_policy_params(key, state_dim, action_dim, hidden_dim=128):
    k1, k2 = jax.random.split(key)
    params = {
        # Xavier/Glorot initialization: sqrt(2/n) where n is the number of inputs
        "W1": jax.random.normal(k1, (state_dim, hidden_dim)) * jnp.sqrt(2.0 / state_dim) * 0.1,
        "b1": jax.numpy.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, action_dim)) * jnp.sqrt(2.0 / hidden_dim) * 0.1,
        "b2": jax.numpy.zeros((action_dim,)),
    }
    return params


def policy_forward(params, state):
    x = jnp.dot(state, params["W1"]) + params["b1"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, params["W2"]) + params["b2"]
    return x
