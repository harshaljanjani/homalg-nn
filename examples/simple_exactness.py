import torch
import numpy as np
from homalg_nn.losses import ExactnessLoss, ChainAxiomLoss
from homalg_nn.core import ChainComplex

def main():
    print("EXACTNESS LOSS OPTIMIZATION DEMO")
    print()
    torch.manual_seed(42)
    np.random.seed(42)
    # chain complex: 5 <- 8 <- 10 <- 8 <- 5
    dimensions = [5, 8, 10, 8, 5]
    # random boundary maps
    boundary_maps = []
    for i in range(len(dimensions) - 1):
        d_i = torch.randn(
            dimensions[i],
            dimensions[i+1],
            requires_grad=True,
            dtype=torch.float64
        )
        boundary_maps.append(d_i)
    print(f"Chain complex: {' <- '.join(map(str, dimensions))}")
    print(f"Number of boundary maps: {len(boundary_maps)}")
    print(f"Total parameters: {sum(d.numel() for d in boundary_maps)}")
    print()
    # initial state
    with torch.no_grad():
        chain = ChainComplex(
            dimensions=dimensions,
            boundary_maps=[d.numpy() for d in boundary_maps]
        )
        initial_betti = chain.get_betti_numbers()
        print(f"Initial Betti numbers: {initial_betti}")
        print(f"Initial homology rank: {sum(initial_betti)}")
        print()

    exactness_loss_fn = ExactnessLoss(epsilon=1e-6, normalize=True)
    chain_axiom_loss_fn = ChainAxiomLoss(normalize=True)
    optimizer = torch.optim.Adam(boundary_maps, lr=0.01)
    print("Optimization Progress:")
    num_steps = 500
    for step in range(num_steps):
        optimizer.zero_grad()
        exactness_loss = exactness_loss_fn(boundary_maps)
        chain_axiom_loss = chain_axiom_loss_fn(boundary_maps)
        total_loss = exactness_loss + 0.1 * chain_axiom_loss
        total_loss.backward()
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                chain = ChainComplex(
                    dimensions=dimensions,
                    boundary_maps=[d.numpy() for d in boundary_maps]
                )
                betti = chain.get_betti_numbers()
                print(f"Step {step:4d}: "
                      f"Loss={total_loss.item():.4f}, "
                      f"Exactness={exactness_loss.item():.4f}, "
                      f"ChainAxiom={chain_axiom_loss.item():.6f}")
                print(f"           Betti={betti}, Rank={sum(betti)}")

    # final state
    print()
    print("RESULTS")
    with torch.no_grad():
        chain = ChainComplex(
            dimensions=dimensions,
            boundary_maps=[d.numpy() for d in boundary_maps]
        )
        final_betti = chain.get_betti_numbers()
        print(f"Initial Betti numbers: {initial_betti} (rank={sum(initial_betti)})")
        print(f"Final Betti numbers:   {final_betti} (rank={sum(final_betti)})")
        print(f"Rank reduction: {sum(initial_betti)} -> {sum(final_betti)}")
        print()
        final_exactness = exactness_loss_fn(boundary_maps).item()
        final_chain_axiom = chain_axiom_loss_fn(boundary_maps).item()
        print(f"Final exactness loss: {final_exactness:.6f}")
        print(f"Final chain axiom loss: {final_chain_axiom:.6f}")
        print()
        if sum(final_betti) < sum(initial_betti):
            print("SUCCESS: Homology rank decreased!")
            print("The chain complex moved toward exactness.")
        else:
            print("Homology rank did not decrease, but loss was minimized.")
    print()
    print("Demo complete!")


if __name__ == '__main__':
    main()
