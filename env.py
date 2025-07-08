import jax
import jax.numpy as jnp
import gymnasium as gym
from typing import List, Tuple, Dict, Any
import numpy as np


def sample_episode(policy_params, policy_forward, env, max_steps=500, episode_seed=None):
    """
    Sample a single episode using the current policy.

    Returns:
        states: List of states
        actions: List of actions taken
        rewards: List of rewards received
        done: Whether episode ended
    """
    states, actions, rewards = [], [], []

    state, _ = env.reset()
    state = jnp.array(state, dtype=jnp.float32)

    for step in range(max_steps):
        states.append(state)
        logits = policy_forward(policy_params, state)
        
        # Use episode-specific seed to ensure different episodes take different actions
        action_key = jax.random.PRNGKey(episode_seed * 1000 + step)
        action = jax.random.categorical(action_key, logits)
        actions.append(int(action))

        next_state, reward, terminated, truncated, _ = env.step(int(action))
        next_state = jnp.array(next_state, dtype=jnp.float32)
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    return states, actions, rewards


def sample_trajectories(
    policy_params, policy_forward, env, num_episodes=10, max_steps=500
):
    """
    Sample multiple episodes and return trajectory data.

    Returns:
        all_states: List of all states from all episodes
        all_actions: List of all actions from all episodes
        all_rewards: List of all rewards from all episodes
        episode_lengths: List of episode lengths
    """
    all_states, all_actions, all_rewards = [], [], []
    episode_lengths = []

    for episode in range(num_episodes):
        states, actions, rewards = sample_episode(
            policy_params, policy_forward, env, max_steps, episode_seed=episode
        )

        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        episode_lengths.append(len(states))

    return all_states, all_actions, all_rewards, episode_lengths
