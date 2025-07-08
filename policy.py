import jax
import jax.numpy as jnp


def init_policy_params(key, state_dim, action_dim, hidden_dim=128):
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        # He initialization: sqrt(2/fan_in) for ReLU networks
        "W1": jax.random.normal(k1, (state_dim, hidden_dim))
        * jnp.sqrt(2.0 / state_dim),
        "b1": jax.numpy.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, hidden_dim))
        * jnp.sqrt(2.0 / hidden_dim),
        "b2": jax.numpy.zeros((hidden_dim,)),
        "W3": jax.random.normal(k3, (hidden_dim, action_dim))
        * jnp.sqrt(2.0 / hidden_dim),
        "b3": jax.numpy.zeros((action_dim,)),
    }
    return params


def policy_forward(params, state):
    x = jnp.dot(state, params["W1"]) + params["b1"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, params["W2"]) + params["b2"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, params["W3"]) + params["b3"]

    return x
