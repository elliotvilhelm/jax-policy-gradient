import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(results):
    """
    Create comparison plots for all variants.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

    for i, (name, data) in enumerate(results.items()):
        if data["rewards"] is not None:
            episodes = np.arange(len(data["rewards"])) * 10

            # Plot rewards
            ax.plot(
                episodes,
                data["rewards"],
                color=colors[i % len(colors)],
                linewidth=2,
                label=name,
                alpha=0.8,
            )

    # Customize reward plot
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("VPG Variants Comparison: Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.8)

    plt.tight_layout()
    plt.grid(True, alpha=0.8)
    plt.savefig("images/vpg_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\n📊 Comparison plot saved to images/vpg_comparison.png")


def plot_training_curves(rewards, losses, save_path="images/training_curves.png"):
    """
    Plot training curves and save to file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    episodes = np.arange(len(rewards)) * 50  # Assuming episodes_per_update=50

    # Plot rewards
    ax1.plot(episodes, rewards, "b-", linewidth=2, label="Average Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Training Progress: Average Reward")
    ax1.grid(True, alpha=0.8)
    ax1.legend()

    # Plot losses
    ax2.plot(episodes, losses, "r-", linewidth=2, label="Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Progress: Loss")
    ax2.grid(True, alpha=0.8)
    ax2.legend()

    plt.tight_layout()
    plt.grid(True, alpha=0.8)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Training curves saved to {save_path}")
