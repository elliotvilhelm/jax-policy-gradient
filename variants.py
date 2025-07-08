"""
Define VPG variants for comparison.
Users can modify this file to add their own variants or change existing ones.
"""

# Default variants for comparison
DEFAULT_VARIANTS = {
    "VPG (Basic)": {
        "use_baseline": False,
        "use_entropy": False,
        "use_gradient_clipping": False,
        "use_per_episode_norm": False,
        "description": "Basic VPG with no enhancements",
    },
    "With Baseline": {
        "use_baseline": True,
        "use_entropy": False,
        "use_gradient_clipping": False,
        "use_per_episode_norm": False,
        "description": "VPG with baseline subtraction for variance reduction",
    },
    "With Entropy": {
        "use_baseline": True,
        "use_entropy": True,
        "use_gradient_clipping": False,
        "use_per_episode_norm": False,
        "description": "VPG with baseline and entropy regularization",
    },
    "Full Enhanced": {
        "use_baseline": True,
        "use_entropy": True,
        "use_gradient_clipping": True,
        "use_per_episode_norm": True,
        "description": "VPG with all enhancements",
    },
}

# Example of how users can add custom variants:
# CUSTOM_VARIANTS = {
#     "High Entropy": {
#         "use_baseline": True,
#         "use_entropy": True,
#         "use_gradient_clipping": True,
#         "use_per_episode_norm": True,
#         "entropy_coef": 0.05,  # Higher entropy coefficient
#         "description": "VPG with high entropy regularization"
#     },
#     "Low Learning Rate": {
#         "use_baseline": True,
#         "use_entropy": True,
#         "use_gradient_clipping": True,
#         "use_per_episode_norm": True,
#         "learning_rate": 0.0001,  # Lower learning rate
#         "description": "VPG with conservative learning rate"
#     }
# }


def get_variants():
    """
    Get the variants to run in comparison.
    Users can override this function to return their own variants.
    """
    return DEFAULT_VARIANTS


def print_variants():
    """Print available variants with descriptions."""
    variants = get_variants()
    print("Available variants:")
    for i, (name, config) in enumerate(variants.items(), 1):
        desc = config.get("description", "No description")
        print(f"  {i}. {name}: {desc}")
    print()
