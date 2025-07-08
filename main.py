import jax
import jax.numpy as jnp
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import numpy as np
from policy import policy_forward
from train import train_vpg


def evaluate_policy(policy_params, policy_forward, env, num_episodes=5, render=True):
    """
    Evaluate the trained policy with rendering.
    """
    total_rewards = []

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}:")

        state, _ = env.reset()
        state = jnp.array(state, dtype=jnp.float32)
        total_reward = 0
        step = 0

        while True:
            if render:
                env.render()

            logits = policy_forward(policy_params, state)
            probs = jax.nn.softmax(logits)
            print(f"Action probs: {probs}")
            action = jax.random.categorical(jax.random.PRNGKey(42), logits)

            next_state, reward, terminated, truncated, _ = env.step(int(action))
            next_state = jnp.array(next_state, dtype=jnp.float32)

            total_reward += reward
            state = next_state
            step += 1

            if terminated or truncated:
                break

        total_rewards.append(total_reward)
        print(f"  Total reward: {total_reward}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def plot_training_curves(rewards, losses, save_path="training_curves.png"):
    """
    Plot training curves and save to file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    episodes = np.arange(len(rewards)) * 50  # Assuming episodes_per_update=50
    
    # Plot rewards
    ax1.plot(episodes, rewards, 'b-', linewidth=2, label='Average Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Progress: Average Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot losses
    ax2.plot(episodes, losses, 'r-', linewidth=2, label='Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Progress: Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train VPG on CartPole-v1 with configurable features')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--episodes-per-update', type=int, default=50, help='Episodes per policy update')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    
    # Feature flags
    parser.add_argument('--no-baseline', action='store_true', help='Disable baseline subtraction')
    parser.add_argument('--no-entropy', action='store_true', help='Disable entropy regularization')
    parser.add_argument('--no-gradient-clipping', action='store_true', help='Disable gradient clipping')
    parser.add_argument('--no-per-episode-norm', action='store_true', help='Disable per-episode normalization')
    
    # Entropy coefficient
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy regularization coefficient')
    
    # Evaluation
    parser.add_argument('--eval-episodes', type=int, default=3, help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering during evaluation')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting training curves')
    
    args = parser.parse_args()
    
    config = {
        'use_baseline': not args.no_baseline,
        'use_entropy': not args.no_entropy,
        'use_gradient_clipping': not args.no_gradient_clipping,
        'use_per_episode_norm': not args.no_per_episode_norm,
        'entropy_coef': args.entropy_coef,
    }
    
    print("🚀 Training VPG on CartPole-v1...")
    print(f"Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Episodes per update: {args.episodes_per_update}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Baseline subtraction: {config['use_baseline']}")
    print(f"  Entropy regularization: {config['use_entropy']} (coef: {config['entropy_coef']})")
    print(f"  Gradient clipping: {config['use_gradient_clipping']}")
    print(f"  Per-episode normalization: {config['use_per_episode_norm']}")
    print()

    trained_params, rewards, losses = train_vpg(
        num_episodes=args.episodes,
        episodes_per_update=args.episodes_per_update,
        learning_rate=args.lr,
        gamma=args.gamma,
        max_steps=args.max_steps,
        **config
    )

    print("\n🎯 Training complete! Now evaluating...")
    env = gym.make("CartPole-v1", render_mode="human" if not args.no_render else None)
    avg_reward = evaluate_policy(trained_params, policy_forward, env, num_episodes=args.eval_episodes, render=not args.no_render)

    print(f"\n✅ Final average reward: {avg_reward:.2f}")
    print("🎉 VPG training and evaluation complete!")

    env.close()
    
    if not args.no_plot:
        plot_training_curves(rewards, losses)
    
    return trained_params, rewards, losses


if __name__ == "__main__":
    main()
