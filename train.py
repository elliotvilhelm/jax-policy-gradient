import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from policy import init_policy_params, policy_forward
from env import sample_trajectories
from loss import compute_returns, compute_policy_loss


def train_vpg(
    num_episodes=1000,
    episodes_per_update=10,
    learning_rate=0.001,
    gamma=0.99,
    max_steps=500,
):
    """
    Train VPG on CartPole-v1.
    """
    # Initialize environment
    env = gym.make("CartPole-v1")

    # Initialize policy
    key = jax.random.PRNGKey(0)
    policy_params = init_policy_params(key, state_dim=4, action_dim=2, hidden_dim=128)

    # Initialize optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(policy_params)

    # Training loop
    for episode in range(0, num_episodes, episodes_per_update):
        # Sample trajectories
        states, actions, rewards, episode_lengths = sample_trajectories(
            policy_params,
            policy_forward,
            env,
            num_episodes=episodes_per_update,
            max_steps=max_steps,
        )

        # Compute returns for all timesteps
        all_returns = []
        start_idx = 0
        for length in episode_lengths:
            episode_rewards = rewards[start_idx : start_idx + length]
            episode_returns = compute_returns(episode_rewards, gamma)
            # Normalize returns per episode instead of across all episodes
            episode_returns = jnp.array(episode_returns)
            episode_returns = (episode_returns - jnp.mean(episode_returns)) / (jnp.std(episode_returns) + 1e-8)
            all_returns.extend(episode_returns.tolist())
            start_idx += length
        
        # Convert to array for loss computation
        all_returns = jnp.array(all_returns)

        # Compute loss
        loss, _ = compute_policy_loss(
            policy_params, policy_forward, states, actions, all_returns
        )

        # Compute gradients and update parameters
        grads = jax.grad(
            lambda p: compute_policy_loss(
                p, policy_forward, states, actions, all_returns
            )[0]
        )(policy_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        policy_params = optax.apply_updates(policy_params, updates)

        # Print progress
        avg_reward = sum(rewards) / len(episode_lengths)
        print(
            f"Episode {episode}-{episode + episodes_per_update - 1}: "
            f"Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}"
        )

    env.close()
    return policy_params
