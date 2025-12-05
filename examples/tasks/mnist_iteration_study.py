import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from mnist_classification import (
    MNISTChainNet, evaluate_model
)


def quick_train_exact(model, train_loader, test_loader, epochs=10, lr=1e-3,
    lambda_exact=0.1, device='cpu'):
    """Streamlined training for iteration experiments."""
    from homalg_nn.nn import create_recommended_scheduler
    from homalg_nn.monitoring import HomologyMonitor

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = create_recommended_scheduler(total_steps)
    monitor = HomologyMonitor(log_interval=len(train_loader), verbose=False)
    history = {
        'task_loss': [],
        'exactness_loss': [],
        'chain_axiom_loss': [],
        'test_accuracy': [],
        'betti_numbers': [],
        'final_betti': None,
        'converged_to_exact': False
    }
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_task = 0
        epoch_exact = 0
        epoch_axiom = 0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if next(model.parameters()).dtype == torch.float64:
                data = data.double()
            optimizer.zero_grad()
            output = model(data)
            loss_task = F.cross_entropy(output, target)
            ex_w, ax_w = scheduler.get_weights(step)
            loss_exact = model.chain.compute_exactness_loss(mode='exactness')
            loss_axiom = model.chain.compute_exactness_loss(mode='chain_axiom')
            loss_chain = ex_w * loss_exact + ax_w * loss_axiom
            loss = loss_task + lambda_exact * loss_chain
            loss.backward()
            optimizer.step()
            epoch_task += loss_task.item()
            epoch_exact += loss_exact.item()
            epoch_axiom += loss_axiom.item()
            step += 1
        n_batches = len(train_loader)
        history['task_loss'].append(epoch_task / n_batches)
        history['exactness_loss'].append(epoch_exact / n_batches)
        history['chain_axiom_loss'].append(epoch_axiom / n_batches)
        test_acc = evaluate_model(model, test_loader, device)
        history['test_accuracy'].append(test_acc)
        monitor.on_step(
            step=step,
            chain_module=model.chain,
            exactness_loss=epoch_exact / n_batches,
            chain_axiom_loss=epoch_axiom / n_batches,
            total_loss=(epoch_task + epoch_exact + epoch_axiom) / n_batches
        )
        if monitor.history['betti_numbers']:
            _, betti = monitor.history['betti_numbers'][-1]
            history['betti_numbers'].append(betti)
        if epoch % 5 == 0 or epoch == epochs - 1:
            betti_str = str(history['betti_numbers'][-1]) if history['betti_numbers'] else "N/A"
            print(f"  Epoch {epoch+1}/{epochs} - Task: {history['task_loss'][-1]:.4f}, "
                  f"Exact: {history['exactness_loss'][-1]:.4f}, "
                  f"Acc: {test_acc:.4f}, Betti: {betti_str}")
    if history['betti_numbers']:
        history['final_betti'] = history['betti_numbers'][-1]
        history['converged_to_exact'] = sum(history['final_betti']) == 0
    return history


def experiment_lambda_sweep(train_loader, test_loader, device='cpu', epochs=15):
    """Test different lambda values."""
    print("\nLAMBDA SWEEP")
    print(f"Testing lambda in [0.1, 0.3, 0.5, 1.0, 2.0] for {epochs} epochs")
    lambdas = [0.1, 0.3, 0.5, 1.0, 2.0]
    results = {}
    for lam in lambdas:
        print(f"\nTesting lambda = {lam}")
        model = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
        history = quick_train_exact(
            model, train_loader, test_loader,
            epochs=epochs, lambda_exact=lam, device=device
        )
        results[f"lambda_{lam}"] = {
            'lambda': lam,
            'final_accuracy': history['test_accuracy'][-1],
            'final_task_loss': history['task_loss'][-1],
            'final_exactness_loss': history['exactness_loss'][-1],
            'final_betti': history['final_betti'],
            'converged': history['converged_to_exact'],
            'betti_evolution': history['betti_numbers']
        }
        betti = history['final_betti']
        converged = "[OK] EXACT" if history['converged_to_exact'] else "[X] Not exact"
        print(f"\n  Result: Acc={history['test_accuracy'][-1]:.4f}, "
              f"Betti={betti}, {converged}")
    return results


def experiment_extended_training(train_loader, test_loader, device='cpu'):
    """Extended training (30 epochs) with optimal lambda."""
    print("\nEXTENDED TRAINING")
    print("Training for 30 epochs with lambda = 0.5\n")
    model = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
    history = quick_train_exact(
        model, train_loader, test_loader,
        epochs=30, lambda_exact=0.5, device=device
    )
    result = {
        'final_accuracy': history['test_accuracy'][-1],
        'final_task_loss': history['task_loss'][-1],
        'final_exactness_loss': history['exactness_loss'][-1],
        'final_betti': history['final_betti'],
        'converged': history['converged_to_exact'],
        'history': history
    }
    return result


def experiment_symmetric_chain(train_loader, test_loader, device='cpu', epochs=15):
    """Symmetric chain architecture."""
    print("\nSYMMETRIC CHAIN ARCHITECTURE")
    print("Testing symmetric chain [16,32,64,32,16] with lambda = 0.5\n")

    class MNISTSymmetricChain(nn.Module):
        def __init__(self, dtype=torch.float64):
            super().__init__()
            from homalg_nn.nn import ChainModule

            self.encoder = nn.Linear(784, 512, dtype=dtype)
            self.bridge_in = nn.Linear(512, 256, dtype=dtype)
            # [16, 32, 64, 32, 16]
            self.chain = ChainModule(
                dimensions=[16, 32, 64, 32, 16],
                epsilon=1e-3,
                dtype=dtype,
                init_method='orthogonal'
            )
            self.bridge_out = nn.Linear(16, 128, dtype=dtype)
            self.decoder = nn.Linear(128, 10, dtype=dtype)

        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.encoder(x))
            x = F.relu(self.bridge_in(x))

            # 256 -> 16 via custom projection
            # chain expects input of size 16 (last dimension)
            x_16 = x[:, :16]
            x = self.chain(x_16)
            x = F.relu(self.bridge_out(x))
            x = self.decoder(x)
            return x

    model = MNISTSymmetricChain(dtype=torch.float64)
    history = quick_train_exact(
        model, train_loader, test_loader,
        epochs=epochs, lambda_exact=0.5, device=device
    )
    result = {
        'architecture': '[16,32,64,32,16] symmetric',
        'final_accuracy': history['test_accuracy'][-1],
        'final_betti': history['final_betti'],
        'converged': history['converged_to_exact'],
        'history': history
    }
    return result


def experiment_smaller_chain(train_loader, test_loader, device='cpu', epochs=15):
    """Smaller chain for faster convergence."""
    print("\nSMALLER CHAIN")
    print("Testing smaller chain [16,32,48,32,16] with lambda = 0.5\n")

    class MNISTSmallChain(nn.Module):
        def __init__(self, dtype=torch.float64):
            super().__init__()
            from homalg_nn.nn import ChainModule

            self.encoder = nn.Linear(784, 256, dtype=dtype)
            self.bridge_in = nn.Linear(256, 16, dtype=dtype)
            # [16,32,48,32,16]
            self.chain = ChainModule(
                dimensions=[16, 32, 48, 32, 16],
                epsilon=1e-3,
                dtype=dtype,
                init_method='orthogonal'
            )
            self.bridge_out = nn.Linear(16, 128, dtype=dtype)
            self.decoder = nn.Linear(128, 10, dtype=dtype)

        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.encoder(x))
            x = F.relu(self.bridge_in(x))
            x = self.chain(x)
            x = F.relu(self.bridge_out(x))
            x = self.decoder(x)
            return x

    model = MNISTSmallChain(dtype=torch.float64)
    history = quick_train_exact(
        model, train_loader, test_loader,
        epochs=epochs, lambda_exact=0.5, device=device
    )
    result = {
        'architecture': '[16,32,48,32,16] small',
        'final_accuracy': history['test_accuracy'][-1],
        'final_betti': history['final_betti'],
        'converged': history['converged_to_exact'],
        'history': history
    }
    return result


def visualize_iteration_results(lambda_results, extended_result=None):
    """Create comprehensive visualization of iteration experiments."""
    fig = plt.figure(figsize=(16, 10))

    # final betti vs lambda
    ax1 = plt.subplot(2, 3, 1)
    lambdas = []
    betti_sums = []
    converged = []
    for _, result in lambda_results.items():
        lambdas.append(result['lambda'])
        betti_sums.append(sum(result['final_betti']))
        converged.append(result['converged'])
    colors = ['green' if c else 'red' for c in converged]
    ax1.bar(range(len(lambdas)), betti_sums, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(lambdas)))
    ax1.set_xticklabels([f"lambda={l}" for l in lambdas])
    ax1.set_ylabel('Sum of Betti Numbers')
    ax1.set_title('Betti Convergence vs Lambda')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # accuracy vs lambda
    ax2 = plt.subplot(2, 3, 2)
    accuracies = [result['final_accuracy'] for result in lambda_results.values()]
    ax2.plot(lambdas, accuracies, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy vs Lambda')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.90, 1.0])

    # exactness loss vs lambda
    ax3 = plt.subplot(2, 3, 3)
    exact_losses = [result['final_exactness_loss'] for result in lambda_results.values()]
    ax3.plot(lambdas, exact_losses, 's-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Lambda')
    ax3.set_ylabel('Final Exactness Loss')
    ax3.set_title('Exactness Loss vs Lambda')
    ax3.grid(True, alpha=0.3)

    # betti evolution for best lambda
    ax4 = plt.subplot(2, 3, 4)
    best_key = min(lambda_results.keys(), key=lambda k: sum(lambda_results[k]['final_betti']))
    best_result = lambda_results[best_key]
    if best_result['betti_evolution']:
        betti_array = np.array(best_result['betti_evolution'])
        epochs = range(1, len(betti_array) + 1)
        for i in range(betti_array.shape[1]):
            ax4.plot(epochs, betti_array[:, i], label=f'Betti {i}', marker='o')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Betti Number')
        ax4.set_title(f'Betti Evolution (lambda={best_result["lambda"]})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # extended training (if available)
    if extended_result:
        ax5 = plt.subplot(2, 3, 5)
        hist = extended_result['history']
        epochs = range(1, len(hist['exactness_loss']) + 1)
        ax5.plot(epochs, hist['exactness_loss'], label='Exactness', marker='o')
        ax5.plot(epochs, hist['chain_axiom_loss'], label='Chain Axiom', marker='s')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title('Extended Training (30 epochs)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
        ax6 = plt.subplot(2, 3, 6)
        if hist['betti_numbers']:
            betti_array = np.array(hist['betti_numbers'])
            epochs = range(1, len(betti_array) + 1)
            for i in range(betti_array.shape[1]):
                ax6.plot(epochs, betti_array[:, i], label=f'Betti {i}', marker='o')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Betti Number')
            ax6.set_title('Betti Numbers (Extended)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    """Run comprehensive iteration study."""
    print("MNIST ITERATION STUDY")
    device = 'cpu'
    print("Loading MNIST dataset.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    results = {}
    # lambda sweep (15 epochs each)
    lambda_results = experiment_lambda_sweep(train_loader, test_loader, device, epochs=15)
    results['lambda_sweep'] = lambda_results
    best_lambda_key = min(lambda_results.keys(), key=lambda k: sum(lambda_results[k]['final_betti']))
    best_lambda = lambda_results[best_lambda_key]['lambda']    
    print(f"\nBEST LAMBDA FROM SWEEP: {best_lambda}")
    # extended training with best lambda
    extended_result = experiment_extended_training(train_loader, test_loader, device)
    results['extended_training'] = extended_result
    # alternative architectures
    symmetric_result = experiment_symmetric_chain(train_loader, test_loader, device)
    results['symmetric_chain'] = symmetric_result
    small_result = experiment_smaller_chain(train_loader, test_loader, device)
    results['small_chain'] = small_result
    # summary
    print("\nITERATION STUDY SUMMARY")
    print("\nLambda Sweep Results:")
    print(f"{'Lambda':<10} {'Accuracy':<12} {'Final Betti':<20} {'Exact?':<10}")
    print("-" * 60)
    for _, result in lambda_results.items():
        lam = result['lambda']
        acc = result['final_accuracy']
        betti = str(result['final_betti'])
        exact = "[OK] Yes" if result['converged'] else "[X] No"
        print(f"{lam:<10} {acc:<12.4f} {betti:<20} {exact:<10}")
    print(f"\nExtended Training (30 epochs, lambda={0.5}):")
    print(f"  Accuracy: {extended_result['final_accuracy']:.4f}")
    print(f"  Betti: {extended_result['final_betti']}")
    print(f"  Exact: {'[OK] Yes' if extended_result['converged'] else '[X] No'}")
    print(f"\nAlternative Architectures:")
    print(f"  Symmetric: Betti={symmetric_result['final_betti']}, "
          f"Exact={'[OK]' if symmetric_result['converged'] else '[X]'}")
    print(f"  Small: Betti={small_result['final_betti']}, "
          f"Exact={'[OK]' if small_result['converged'] else '[X]'}")
    any_exact = (
        any(r['converged'] for r in lambda_results.values()) or
        extended_result['converged'] or
        symmetric_result['converged'] or
        small_result['converged']
    )
    if any_exact:
        print("\n[OK] SUCCESS: At least one configuration achieved exactness!")
    else:
        print("\n[X] No configuration achieved full exactness - need more investigation")
    output_dir = Path('./results/mnist_iteration')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = visualize_iteration_results(lambda_results, extended_result)
    fig.savefig(output_dir / 'iteration_study.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir / 'iteration_study.png'}")

    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    results_serializable = convert_to_serializable(results)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")
    print("ITERATION STUDY COMPLETE")


if __name__ == '__main__':
    main()
