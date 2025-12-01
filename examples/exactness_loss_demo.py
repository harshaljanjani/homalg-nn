import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from homalg_nn.losses import ExactnessLoss, ChainAxiomLoss
from homalg_nn.core import ChainComplex


def optimize_chain_complex(
    dimensions,
    num_steps=1000,
    lr=0.01,
    seed=42
):
    """
    Optimize a chain complex toward exactness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # d_i: C_i -> C_{i-1}
    boundary_maps = []
    for i in range(len(dimensions) - 1):
        d_i = torch.randn(
            dimensions[i],
            dimensions[i+1],
            requires_grad=True,
            dtype=torch.float64
        )
        boundary_maps.append(d_i)
    exactness_loss_fn = ExactnessLoss(epsilon=1e-6, normalize=True)
    chain_axiom_loss_fn = ChainAxiomLoss(normalize=True)
    optimizer = torch.optim.Adam(boundary_maps, lr=lr)
    history = {
        'exactness_loss': [],
        'chain_axiom_loss': [],
        'total_loss': [],
        'betti_numbers': [],
    }
    print(f"Optimizing chain complex: {' <- '.join(map(str, dimensions))}")
    print(f"Number of boundary maps: {len(boundary_maps)}")
    print(f"Total parameters: {sum(d.numel() for d in boundary_maps)}")
    print()
    # initial Betti numbers
    with torch.no_grad():
        chain = ChainComplex(
            dimensions=dimensions,
            boundary_maps=[d.numpy() for d in boundary_maps]
        )
        initial_betti = chain.get_betti_numbers()
        print(f"Initial Betti numbers: {initial_betti}")
        print(f"Initial homology rank: {sum(initial_betti)}")
        print()

    # optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        exactness_loss = exactness_loss_fn(boundary_maps)
        chain_axiom_loss = chain_axiom_loss_fn(boundary_maps)
        total_loss = exactness_loss + 0.1 * chain_axiom_loss
        total_loss.backward()
        optimizer.step()
        history['exactness_loss'].append(exactness_loss.item())
        history['chain_axiom_loss'].append(chain_axiom_loss.item())
        history['total_loss'].append(total_loss.item())
        if step % 100 == 0:
            with torch.no_grad():
                chain = ChainComplex(
                    dimensions=dimensions,
                    boundary_maps=[d.numpy() for d in boundary_maps]
                )
                betti = chain.get_betti_numbers()
                history['betti_numbers'].append((step, betti, sum(betti)))
                print(f"Step {step:4d}: "
                      f"Loss={total_loss.item():.4f}, "
                      f"Exactness={exactness_loss.item():.4f}, "
                      f"ChainAxiom={chain_axiom_loss.item():.6f}, "
                      f"Betti={betti}, "
                      f"Rank={sum(betti)}")
    with torch.no_grad():
        chain = ChainComplex(
            dimensions=dimensions,
            boundary_maps=[d.numpy() for d in boundary_maps]
        )
        final_betti = chain.get_betti_numbers()
        print()
        print(f"Final Betti numbers: {final_betti}")
        print(f"Final homology rank: {sum(final_betti)}")
        print(f"Reduction in rank: {sum(initial_betti)} → {sum(final_betti)}")

    return history, boundary_maps


def plot_optimization_history(history, save_path=None):
    """Plot optimization metrics over time."""
    _, axes = plt.subplots(2, 2)

    # exactness loss
    ax = axes[0, 0]
    ax.plot(history['exactness_loss'])
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Exactness Loss')
    ax.set_title('Exactness Loss Over Time')
    ax.grid(True)
    ax.set_yscale('log')

    # chain axiom loss
    ax = axes[0, 1]
    ax.plot(history['chain_axiom_loss'])
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Chain Axiom Loss')
    ax.set_title('Chain Axiom Loss Over Time')
    ax.grid(True)
    ax.set_yscale('log')

    # total loss
    ax = axes[1, 0]
    ax.plot(history['total_loss'])
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Over Time')
    ax.grid(True)
    ax.set_yscale('log')

    # betti number evolution
    ax = axes[1, 1]
    steps, _, ranks = zip(*history['betti_numbers'])
    ax.plot(steps, ranks)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Total Homology Rank')
    ax.set_title('Betti Number Sum')
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        txt_path = os.path.splitext(save_path)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write("Betti evolution:\n")
            for step, betti, rank in history['betti_numbers']:
                f.write(f"Step {step}: {betti} (Rank: {rank})\n")
        print(f"Stats logged to {txt_path}")
    else:
        plt.show()


if __name__ == '__main__':
    print("EXACTNESS LOSS OPTIMIZATION DEMO")
    print()
    # small chain complex
    print("1: Short Exact Sequence Approximation")
    dimensions_1 = [5, 8, 10, 8, 5]
    history_1, maps_1 = optimize_chain_complex(
        dimensions=dimensions_1,
        num_steps=500,
        lr=0.01,
        seed=42
    )
    # long chain complex
    print("2: Longer Chain Complex")
    dimensions_2 = [3, 6, 10, 12, 10, 6, 3]
    history_2, maps_2 = optimize_chain_complex(
        dimensions=dimensions_2,
        num_steps=500,
        lr=0.01,
        seed=123
    )
    print("VISUALIZATION")
    os.makedirs('examples/outputs', exist_ok=True)
    plot_optimization_history(
        history_1,
        save_path='examples/outputs/02_exactness_optimization_example1.png'
    )
    plot_optimization_history(
        history_2,
        save_path='examples/outputs/02_exactness_optimization_example2.png'
    )
    print("\nDemo complete! Plots and logs saved to examples/outputs/")
