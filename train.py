import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from policy import init_policy_params, policy_forward
from env import sample_trajectories
from loss import compute_returns, compute_policy_loss
import matplotlib.pyplot as plt
import numpy as np


def train_vpg(
    num_episodes=1000,
    episodes_per_update=10,
    learning_rate=0.001,
    gamma=0.99,
    max_steps=500,
    use_baseline=True,
    use_entropy=True,
    use_gradient_clipping=True,
    use_per_episode_norm=True,
    entropy_coef=0.01,
    clip_norm=1.0,
):
    """
    Train VPG on CartPole-v1 with configurable features.
    """
    # Initialize environment
    env = gym.make("CartPole-v1")

    # Initialize policy
    key = jax.random.PRNGKey(0)
    policy_params = init_policy_params(key, state_dim=4, action_dim=2, hidden_dim=128)

    # Initialize optimizer
    if use_gradient_clipping:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(learning_rate)
        )
    else:
        optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(policy_params)

    # Training history
    episode_rewards = []
    episode_losses = []

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
            episode_rewards_batch = rewards[start_idx : start_idx + length]
            episode_returns = compute_returns(episode_rewards_batch, gamma)
            
            # Apply per-episode normalization if enabled
            if use_per_episode_norm:
                episode_returns = jnp.array(episode_returns)
                episode_returns = (episode_returns - jnp.mean(episode_returns)) / (jnp.std(episode_returns) + 1e-8)
            
            all_returns.extend(episode_returns.tolist())
            start_idx += length
        
        # Convert to array for loss computation
        all_returns = jnp.array(all_returns)

        # Compute loss with configurable features
        loss, _ = compute_policy_loss(
            policy_params, 
            policy_forward, 
            states, 
            actions, 
            all_returns,
            use_baseline=use_baseline,
            use_entropy=use_entropy,
            entropy_coef=entropy_coef
        )

        # Compute gradients and update parameters
        grads = jax.grad(
            lambda p: compute_policy_loss(
                p, policy_forward, states, actions, all_returns,
                use_baseline=use_baseline,
                use_entropy=use_entropy,
                entropy_coef=entropy_coef
            )[0]
        )(policy_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        policy_params = optax.apply_updates(policy_params, updates)

        # Record metrics
        avg_reward = sum(rewards) / len(episode_lengths)
        episode_rewards.append(avg_reward)
        episode_losses.append(float(loss))

        # Print progress
        print(
            f"Episode {episode}-{episode + episodes_per_update - 1}: "
            f"Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}"
        )

    env.close()
    return policy_params, episode_rewards, episode_losses


def compare_vpg_variants():
    """
    Compare different VPG variants and plot results.
    """
    variants = {
        "Vanilla VPG": {
            "use_baseline": False,
            "use_entropy": False,
            "use_gradient_clipping": False,
            "use_per_episode_norm": False,
        },
        "With Baseline": {
            "use_baseline": True,
            "use_entropy": False,
            "use_gradient_clipping": False,
            "use_per_episode_norm": False,
        },
        "With Entropy": {
            "use_baseline": True,
            "use_entropy": True,
            "use_gradient_clipping": False,
            "use_per_episode_norm": False,
        },
        "Full Enhanced": {
            "use_baseline": True,
            "use_entropy": True,
            "use_gradient_clipping": True,
            "use_per_episode_norm": True,
        }
    }
    
    results = {}
    
    for name, config in variants.items():
        print(f"\n=== Training {name} ===")
        _, rewards, losses = train_vpg(
            num_episodes=500,  # Shorter for comparison
            episodes_per_update=10,
            **config
        )
        results[name] = {"rewards": rewards, "losses": losses}
    
    # Plot results
    plot_comparison(results)


def plot_comparison(results):
    """
    Create comparison plots for different VPG variants.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rewards
    for name, data in results.items():
        episodes = np.arange(len(data["rewards"])) * 10
        ax1.plot(episodes, data["rewards"], label=name, linewidth=2)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Training Progress: Average Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    for name, data in results.items():
        episodes = np.arange(len(data["losses"])) * 10
        ax2.plot(episodes, data["losses"], label=name, linewidth=2)
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Progress: Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("vpg_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """
    Main training function for single run.
    """
    print("Training VPG on CartPole-v1...")

    trained_params, rewards, losses = train_vpg(
        num_episodes=1000,
        episodes_per_update=50,
        learning_rate=0.001,
        gamma=0.99,
        max_steps=500,
    )

    print("\n🎯 Training complete! Now evaluating with rendering...")

    env = gym.make("CartPole-v1", render_mode="human")
    avg_reward = evaluate_policy(trained_params, policy_forward, env, num_episodes=3)

    print(f"\n✅ Final average reward: {avg_reward:.2f}")
    print("🎉 VPG training and evaluation complete!")

    env.close()
    
    return trained_params, rewards, losses
