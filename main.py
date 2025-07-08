import jax
import jax.numpy as jnp
import gymnasium as gym
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


def main():
    print("Training VPG on CartPole-v1...")

    trained_params = train_vpg(
        num_episodes=500,
        episodes_per_update=10,
        learning_rate=0.01,
        gamma=0.99,
        max_steps=200,
    )

    print("\n🎯 Training complete! Now evaluating with rendering...")

    env = gym.make("CartPole-v1", render_mode="human")
    avg_reward = evaluate_policy(trained_params, policy_forward, env, num_episodes=3)

    print(f"\n✅ Final average reward: {avg_reward:.2f}")
    print("🎉 VPG training and evaluation complete!")

    env.close()


if __name__ == "__main__":
    main()
