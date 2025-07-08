"""
Compare multiple VPG variants on the same plot.
"""

import matplotlib.pyplot as plt
import numpy as np
from train import train_vpg
from variants import get_variants
from plot import plot_comparison


def run_comparison(episodes=500, episodes_per_update=10, save_plot=True):
    """
    Run comparison of all variants defined in variants.py
    """
    variants = get_variants()

    print("🎯 Starting VPG Variant Comparison...")
    print(f"Running {len(variants)} variants with {episodes} episodes each")
    print()

    # Run each variant
    results = {}
    for name, config in variants.items():
        print(f"\n🚀 Running {name}...")
        print(f"Description: {config.get('description', 'No description')}")

        try:
            # Extract training parameters from config
            training_config = {
                "use_baseline": config.get("use_baseline", True),
                "use_entropy": config.get("use_entropy", True),
                "use_gradient_clipping": config.get("use_gradient_clipping", True),
                "use_per_episode_norm": config.get("use_per_episode_norm", True),
                "entropy_coef": config.get("entropy_coef", 0.01),
                "learning_rate": config.get("learning_rate", 0.001),
            }

            # Run training
            _, rewards, losses = train_vpg(
                num_episodes=episodes,
                episodes_per_update=episodes_per_update,
                **training_config,
            )

            results[name] = {"rewards": rewards, "losses": losses}
            print(f"✅ {name} completed successfully")

        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {"rewards": None, "losses": None}

    # Generate comparison plot
    if save_plot:
        plot_comparison(results)

    # Print summary
    print("\n📊 Summary:")
    for name, data in results.items():
        if data["rewards"] is not None:
            final_reward = data["rewards"][-1] if data["rewards"] else "N/A"
            print(f"  {name}: Final reward = {final_reward:.2f}")
        else:
            print(f"  {name}: Failed")

    return results
