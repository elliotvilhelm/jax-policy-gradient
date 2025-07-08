import jax
import jax.numpy as jnp
from typing import List


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns for each timestep.

    Formula: R_t = r_t + γ * R_{t+1}
    where R_t is the return at time t, r_t is the reward at time t, and γ is the discount factor.

    Args:
        rewards: List of rewards from an episode
        gamma: Discount factor

    Returns:
        returns: List of discounted returns for each timestep
    """
    returns = []
    R = 0.0

    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)

    return returns


def compute_policy_loss(
    policy_params,
    policy_forward,
    states,
    actions,
    returns,
    use_baseline=True,
    use_entropy=True,
    entropy_coef=0.01,
):
    """
    Compute the policy gradient loss using REINFORCE with configurable features.

    REINFORCE Policy Gradient Formula:
    ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * (R - b)]

    Loss function:
    L(θ) = -log π_θ(a|s) * (R - b) - β * H(π_θ)
    where π_θ(a|s) is the policy probability of action a in state s,
    R is the discounted return, b is a baseline, and H(π_θ) is the entropy.

    Args:
        policy_params: Policy network parameters
        policy_forward: Policy forward function
        states: List of states
        actions: List of actions taken
        returns: List of returns for each timestep
        use_baseline: Whether to use baseline subtraction
        use_entropy: Whether to use entropy regularization
        entropy_coef: Entropy regularization coefficient

    Returns:
        loss: Policy gradient loss
        log_probs: Log probabilities for debugging
    """
    total_loss = 0.0
    total_entropy = 0.0
    log_probs = []

    # Compute baseline if enabled
    if use_baseline:
        baseline = jnp.mean(jnp.array(returns))
    else:
        baseline = 0.0

    for state, action, ret in zip(states, actions, returns):
        logits = policy_forward(policy_params, state)

        log_probs_action = jax.nn.log_softmax(logits)
        log_prob = log_probs_action[action]

        # Compute entropy for regularization if enabled
        if use_entropy:
            probs = jax.nn.softmax(logits)
            entropy = -jnp.sum(probs * log_probs_action)
        else:
            entropy = 0.0

        # Use advantage (return - baseline) instead of raw return
        advantage = ret - baseline
        policy_loss = -log_prob * advantage

        # Add entropy regularization if enabled
        if use_entropy:
            loss = policy_loss - entropy_coef * entropy
        else:
            loss = policy_loss

        total_loss += loss
        total_entropy += entropy
        log_probs.append(log_prob)

    return total_loss, log_probs
