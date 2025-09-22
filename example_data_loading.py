#!/usr/bin/env python3
"""
Example demonstrating how to load and analyze saved LatticeMC simulation data.

This script shows how to load data from working folders created by LatticeMC
simulations and perform basic analysis. This is useful for post-processing
analysis or creating custom visualizations.
"""

import json
import pathlib
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_simulation_summary(working_folder: pathlib.Path) -> Dict[str, Any]:
    """Load the JSON summary from a simulation working folder."""
    json_path = working_folder / "summary.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"No simulation summary found at {json_path}")


def load_order_parameters(working_folder: pathlib.Path) -> np.ndarray:
    """Load order parameters history from a simulation working folder."""
    npz_path = working_folder / "data" / "order_parameters.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if 'order_parameters' in data:
            return data['order_parameters']
        else:
            raise ValueError("No 'order_parameters' array found in the NPZ file")
    else:
        raise FileNotFoundError(f"No order parameters file found at {npz_path}")


def analyze_simulation_data(working_folder: pathlib.Path):
    """Analyze simulation data from a working folder."""
    print(f"📁 Analyzing data from: {working_folder}")

    try:
        # Load summary
        summary = load_simulation_summary(working_folder)
        print(f"✅ Loaded simulation summary")
        print(f"   Steps completed: {summary['current_step']}")
        print(f"   Total cycles: {summary['total_cycles']}")
        print(f"   Parameters: {summary['parameters']}")

        # Load order parameters
        order_params = load_order_parameters(working_folder)
        print(f"✅ Loaded order parameters ({len(order_params)} data points)")

        # Basic statistics
        print(f"\n📊 Statistical Analysis:")
        print(f"   Energy: {order_params['energy'].mean():.4f} ± {order_params['energy'].std():.4f}")
        print(f"   Range: [{order_params['energy'].min():.4f}, {order_params['energy'].max():.4f}]")
        print(f"   q0: {order_params['q0'].mean():.4f} ± {order_params['q0'].std():.4f}")
        print(f"   q2: {order_params['q2'].mean():.4f} ± {order_params['q2'].std():.4f}")

        # Create visualization
        create_analysis_plots(order_params, working_folder)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def create_analysis_plots(order_params: np.ndarray, working_folder: pathlib.Path):
    """Create analysis plots from loaded order parameters."""
    print(f"\n📈 Creating analysis plots...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Time series plots
    steps = np.arange(len(order_params))

    ax1.plot(steps, order_params['energy'], 'b-', alpha=0.7)
    ax1.set_xlabel('Monte Carlo Step')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, order_params['q0'], 'r-', alpha=0.7, label='q₀')
    ax2.plot(steps, order_params['q2'], 'g-', alpha=0.7, label='q₂')
    ax2.set_xlabel('Monte Carlo Step')
    ax2.set_ylabel('Order Parameter')
    ax2.set_title('Orientational Order')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Distribution plots
    ax3.hist(order_params['energy'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Energy')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Energy Distribution')
    ax3.grid(True, alpha=0.3)

    # Correlation plot
    ax4.scatter(order_params['q0'], order_params['q2'], alpha=0.6, s=1, color='purple')
    ax4.set_xlabel('q₀ Order Parameter')
    ax4.set_ylabel('q₂ Order Parameter')
    ax4.set_title('Order Parameter Correlation')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = working_folder / "analysis_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved plots to: {plot_path}")

    plt.show()


def main():
    """Main function to demonstrate data loading and analysis."""
    print("🔬 LatticeMC Data Loading and Analysis Example")
    print("=" * 50)

    # Example working folders to check
    example_folders = [
        pathlib.Path("simulation_output"),
        pathlib.Path("notebook_simulation_output"),
        pathlib.Path("parallel_tempering_output"),
    ]

    found_data = False

    for folder in example_folders:
        if folder.exists():
            print(f"\n📂 Found working folder: {folder}")

            # Check if it's a parallel tempering folder (contains subfolders)
            temp_folders = [f for f in folder.iterdir() if f.is_dir() and f.name.startswith('T_')]

            if temp_folders:
                print(f"   Found {len(temp_folders)} temperature replicas")
                for temp_folder in sorted(temp_folders)[:2]:  # Analyze first 2 temperatures
                    print(f"\n   🌡️  Analyzing {temp_folder.name}:")
                    analyze_simulation_data(temp_folder)
            else:
                analyze_simulation_data(folder)

            found_data = True
        else:
            print(f"❌ Working folder not found: {folder}")

    if not found_data:
        print("\n💡 No simulation data found!")
        print("   Run one of the example scripts first:")
        print("   - python example.py")
        print("   - python example_synchronized_pt.py")
        print("   - Run cells in example.ipynb")


if __name__ == "__main__":
    main()
