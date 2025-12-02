import torch
import numpy as np
import os
from homalg_nn.nn import ChainModule
from homalg_nn.monitoring import HomologyMonitor


def basic_training():
    """Basic ChainModule training."""
    print("Basic ChainModule Training")
    torch.manual_seed(42)
    np.random.seed(42)
    # chain: 5 <- 8 <- 10 <- 8 <- 5
    dimensions = [5, 8, 10, 8, 5]
    chain = ChainModule(
        dimensions=dimensions,
        init_method='xavier_normal',
        exactness_weight=1.0,
        chain_axiom_weight=0.5,
        sparsity_weight=0.1,
        epsilon=1e-3,
        dtype=torch.float64
    )
    print(f"\nInitial state:")
    print(f"  Dimensions: {dimensions}")
    try:
        betti = chain.get_betti_numbers()
        print(f"  Betti: {betti}")
    except Exception as e:
        print(f"  Betti: (skipped due to environment issue)")
    print(f"  Parameters: {sum(p.numel() for p in chain.parameters())}")
    print()
    monitor = HomologyMonitor(
        log_interval=200,
        compute_betti=True,
        compute_defects=True,
        verbose=True
    )
    optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
    num_steps = 2000
    print(f"Training for {num_steps} steps.")
    for step in range(num_steps):
        optimizer.zero_grad()
        loss_exact = chain.compute_exactness_loss(mode='exactness')
        loss_axiom = chain.compute_exactness_loss(mode='chain_axiom')
        total_loss = loss_exact + 0.5 * loss_axiom
        total_loss.backward()
        optimizer.step()
        if step % 200 == 0:
            monitor.on_step(
                step=step,
                chain_module=chain,
                exactness_loss=loss_exact.item(),
                chain_axiom_loss=loss_axiom.item(),
                total_loss=total_loss.item()
            )
    print()
    print(monitor.summary())
    os.makedirs('examples/outputs', exist_ok=True)
    try:
        monitor.plot(save_path='examples/outputs/chain_module.png', show=False)
    except ImportError:
        print("Matplotlib not available - skipping plot")
    return chain, monitor


def initialization_comparison():
    """Compare initialization methods."""
    print("\nInitialization Method Comparison")
    methods = ['normal', 'xavier_normal', 'he_normal', 'orthogonal']
    results = {}
    for method in methods:
        print(f"\nTraining with {method} initialization.")
        torch.manual_seed(42)  # same seed, repro + fair comp.
        chain = ChainModule(
            dimensions=[5, 8, 10, 8, 5],
            init_method=method,
            epsilon=1e-3,
            dtype=torch.float64
        )
        optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01)
        initial_loss = chain.compute_exactness_loss().item()
        for _ in range(1000):
            optimizer.zero_grad()
            loss = chain.compute_exactness_loss()
            loss.backward()
            optimizer.step()
        final_loss = chain.compute_exactness_loss().item()
        results[method] = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'reduction': (initial_loss - final_loss) / initial_loss * 100
        }
    print("\nRESULTS COMPARISON")
    print(f"{'Method':<20} {'Initial Loss':<15} {'Final Loss':<15} {'Reduction %':<15}")
    for method, res in results.items():
        print(f"{method:<20} {res['initial_loss']:<15.4f} "
              f"{res['final_loss']:<15.4f} {res['reduction']:<15.1f}")
    return results


def task_integration():
    """Chain as regularizer in autoencoder."""
    print("\nTask Integration (Autoencoder + Exactness)")
    chain = ChainModule(
        dimensions=[10, 20, 30, 20, 10],
        exactness_weight=0.1,  # lower weight -- task is primary
        epsilon=1e-3,
        dtype=torch.float64
    )

    # make an autoencoder using boundary maps
    class ChainAutoencoder(torch.nn.Module):
        def __init__(self, chain_module):
            super().__init__()
            self.chain = chain_module
            self.activation = torch.nn.ReLU()

        def forward(self, x):
            # encode using boundary maps (reversed)
            h = x
            for d in reversed(self.chain.boundary_maps[1:]):
                h = self.activation(h @ d.T)
            out = torch.sigmoid(h @ self.chain.boundary_maps[0].T)
            return out

    model = ChainAutoencoder(chain)
    torch.manual_seed(42)
    X_train = torch.randn(100, 10, dtype=torch.float64)
    # training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    monitor = HomologyMonitor(
        log_interval=200,
        compute_betti=False,
        compute_defects=False,
        verbose=True
    )
    print("\nTraining autoencoder with exactness regularization.")
    for step in range(1000):
        optimizer.zero_grad()
        # reconstruction
        X_recon = model(X_train)
        recon_loss = torch.nn.functional.mse_loss(X_recon, X_train)
        # regularization (exactness)
        exact_loss = chain.compute_exactness_loss()
        # combined objective
        total_loss = recon_loss + 0.1 * exact_loss
        total_loss.backward()
        optimizer.step()
        if step % 200 == 0:
            monitor.on_step(
                step,
                chain,
                exactness_loss=exact_loss.item(),
                total_loss=total_loss.item(),
                extra_metrics={'recon_loss': recon_loss.item()}
            )
    print()
    print(f"Final reconstruction loss: {recon_loss.item():.4f}")
    print(f"Final exactness loss: {exact_loss.item():.4f}")


if __name__ == '__main__':
    print("TRAINING DEMONSTRATION\n")
    basic_training()
    initialization_comparison()
    task_integration()
    print("\nAll examples completed successfully!")
