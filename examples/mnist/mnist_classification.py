import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from homalg_nn.nn import ChainModule, create_recommended_scheduler
from homalg_nn.monitoring import HomologyMonitor
from homalg_nn.analysis import RepresentationMetrics, FERDetector


class MNISTChainNet(nn.Module):
    """
    MNIST classifier with chain structure.
    """
    def __init__(self, use_chain=True, epsilon=1e-3, dtype=torch.float64):
        super().__init__()
        self.use_chain = use_chain
        self.encoder = nn.Linear(784, 512, dtype=dtype)
        self.bridge_in = nn.Linear(512, 256, dtype=dtype)
        if use_chain:
            # `maps (batch, 256) -> (batch, 16)`
            self.chain = ChainModule(
                dimensions=[16, 32, 64, 128, 256],
                epsilon=epsilon,
                dtype=dtype,
                init_method='orthogonal'
            )
        else:
            self.chain = nn.Sequential(
                nn.Linear(256, 128, dtype=dtype),
                nn.ReLU(),
                nn.Linear(128, 64, dtype=dtype),
                nn.ReLU(),
                nn.Linear(64, 32, dtype=dtype),
                nn.ReLU(),
                nn.Linear(32, 16, dtype=dtype),
                nn.ReLU()
            )
        self.bridge_out = nn.Linear(16, 128, dtype=dtype)
        self.decoder = nn.Linear(128, 10, dtype=dtype)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.encoder(x))
        x = F.relu(self.bridge_in(x))
        if self.use_chain:
            # ChainModule maps from last dim to first: `(batch, 256)` -> `(batch, 16)`
            x = self.chain(x)
        else:
            x = self.chain(x)
        x = F.relu(self.bridge_out(x))
        x = self.decoder(x)
        return x


class MNISTBaselineNet(nn.Module):
    """Baseline MNIST classifier (no chain structure)."""
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512, dtype=dtype),
            nn.ReLU(),
            nn.Linear(512, 256, dtype=dtype),
            nn.ReLU(),
            nn.Linear(256, 128, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 64, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 32, dtype=dtype),
            nn.ReLU(),
            nn.Linear(32, 16, dtype=dtype),
            nn.ReLU(),
            nn.Linear(16, 128, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, dtype=dtype)
        )

    def forward(self, x):
        return self.network(x)

def train_baseline(model, train_loader, test_loader, epochs=10, lr=1e-3, weight_decay=0.0, device='cpu', save_checkpoints=False):
    """
    Train baseline model without exactness constraints.
    """
    print(f"\nTraining Baseline Model")
    print(f"  Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
    print(f"  Device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {
        'train_loss': [],
        'test_accuracy': [],
        'checkpoints': []
    }
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        test_acc = evaluate_model(model, test_loader, device)
        history['test_accuracy'].append(test_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")
        # save checkpoint
        if save_checkpoints:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_acc
            }
            history['checkpoints'].append(checkpoint)
    return history


def train_exact(model, train_loader, test_loader, epochs=10, lr=1e-3,
    lambda_exact=0.1, device='cpu', save_checkpoints=False,
    monitor_homology=True):
    """
    Train model with exactness constraints and annealing.
    """
    print(f"\nTraining Exact Model")
    print(f"  Epochs: {epochs}, LR: {lr}, Lambda: {lambda_exact}")
    print(f"  Device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = create_recommended_scheduler(total_steps)
    if monitor_homology:
        monitor = HomologyMonitor(
            log_interval=len(train_loader),
            verbose=False
        )
    history = {
        'train_loss': [],
        'task_loss': [],
        'exactness_loss': [],
        'chain_axiom_loss': [],
        'exactness_weight': [],
        'chain_axiom_weight': [],
        'test_accuracy': [],
        'betti_numbers': [],
        'checkpoints': []
    }
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_metrics = {
            'total_loss': 0,
            'task_loss': 0,
            'exact_loss': 0,
            'axiom_loss': 0
        }
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if next(model.parameters()).dtype == torch.float64:
                data = data.double()
            optimizer.zero_grad()
            output = model(data)
            loss_task = F.cross_entropy(output, target)
            # exactness loss with annealing
            ex_w, ax_w = scheduler.get_weights(step)
            loss_exact = model.chain.compute_exactness_loss(mode='exactness')
            loss_axiom = model.chain.compute_exactness_loss(mode='chain_axiom')
            loss_chain = ex_w * loss_exact + ax_w * loss_axiom
            # combined loss
            loss = loss_task + lambda_exact * loss_chain
            loss.backward()
            optimizer.step()
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['task_loss'] += loss_task.item()
            epoch_metrics['exact_loss'] += loss_exact.item()
            epoch_metrics['axiom_loss'] += loss_axiom.item()
            step += 1
        n_batches = len(train_loader)
        avg_total = epoch_metrics['total_loss'] / n_batches
        avg_task = epoch_metrics['task_loss'] / n_batches
        avg_exact = epoch_metrics['exact_loss'] / n_batches
        avg_axiom = epoch_metrics['axiom_loss'] / n_batches
        history['train_loss'].append(avg_total)
        history['task_loss'].append(avg_task)
        history['exactness_loss'].append(avg_exact)
        history['chain_axiom_loss'].append(avg_axiom)
        history['exactness_weight'].append(ex_w)
        history['chain_axiom_weight'].append(ax_w)
        test_acc = evaluate_model(model, test_loader, device)
        history['test_accuracy'].append(test_acc)
        # homology monitoring
        if monitor_homology:
            monitor.on_step(
                step=step,
                chain_module=model.chain,
                exactness_loss=avg_exact,
                chain_axiom_loss=avg_axiom,
                total_loss=avg_total
            )
            if monitor.history['betti_numbers']:
                _, betti = monitor.history['betti_numbers'][-1]
                history['betti_numbers'].append(betti)
                betti_str = str(betti)
            else:
                betti_str = "N/A"
        else:
            betti_str = "N/A"
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_total:.4f} "
            f"(Task: {avg_task:.4f}, Exact: {avg_exact:.4f}, Axiom: {avg_axiom:.4f}) "
            f"- Test Acc: {test_acc:.4f} - Betti: {betti_str}")
        # save checkpoint
        if save_checkpoints:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_acc,
                'betti_numbers': history['betti_numbers'][-1] if monitor_homology else None
            }
            history['checkpoints'].append(checkpoint)
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model accuracy on test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if next(model.parameters()).dtype == torch.float64:
                data = data.double()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def analyze_fer(model, test_loader, layer_name='bridge_out', device='cpu',
    sample_size=1000):
    """
    Analyze FER metrics for a trained model.
    """
    print(f"\nAnalyzing FER metrics for layer '{layer_name}'.")
    model = model.to(device).float()
    model.eval()
    detector = FERDetector(
        model=model,
        dataloader=test_loader,
        layer_names=[layer_name],
        device=device,
        sample_size=sample_size
    )
    representations, labels = detector.extract_representations(layer_name, return_labels=True)
    metrics = RepresentationMetrics(sample_size=sample_size, random_state=42)
    probe_result = metrics.linear_probe_accuracy(representations, labels, n_splits=5)
    mi_matrix = metrics.mutual_information_matrix(representations, n_bins=20, normalize=True)
    mask = ~np.eye(mi_matrix.shape[0], dtype=bool)
    entanglement = np.mean(mi_matrix[mask])
    rank_result = metrics.representation_rank(representations, epsilon=1e-3)
    fer_result = metrics.compute_fer_score(representations, labels=labels)
    results = {
        'probe_accuracy': probe_result['mean_accuracy'],
        'probe_std': probe_result['std_accuracy'],
        'probe_complexity': probe_result['probe_complexity'],
        'entanglement': entanglement,
        'effective_rank': rank_result['effective_rank'],
        'rank_ratio': rank_result['rank_ratio'],
        'participation_ratio': rank_result['participation_ratio'],
        'fer_score': fer_result['fer_score']
    }
    print(f"  Probe Accuracy: {results['probe_accuracy']:.4f} +/- {results['probe_std']:.4f}")
    print(f"  Entanglement:   {results['entanglement']:.4f}")
    print(f"  Rank Ratio:     {results['rank_ratio']:.4f}")
    print(f"  FER Score:      {results['fer_score']:.4f}")
    return results


def compare_models(baseline_model, exact_model, test_loader, device='cpu',
    layer_name='bridge_out', n_runs=10):
    """
    Compare FER metrics between baseline and exact models.
    """
    print("\nCOMPARING BASELINE VS EXACT MODELS")
    baseline_metrics = analyze_fer(baseline_model, test_loader, layer_name, device)
    exact_metrics = analyze_fer(exact_model, test_loader, layer_name, device)
    improvements = {}
    for key in baseline_metrics:
        baseline_val = baseline_metrics[key]
        exact_val = exact_metrics[key]
        if key in ['fer_score', 'entanglement']:
            improvement = (baseline_val - exact_val) / baseline_val * 100
        else:
            improvement = (exact_val - baseline_val) / baseline_val * 100
        improvements[key] = improvement
    results = {
        'baseline': baseline_metrics,
        'exact': exact_metrics,
        'improvements': improvements
    }
    print("\nIMPROVEMENT SUMMARY")
    print(f"FER Score:      {improvements['fer_score']:+.2f}%")
    print(f"Entanglement:   {improvements['entanglement']:+.2f}%")
    print(f"Rank Ratio:     {improvements['rank_ratio']:+.2f}%")
    print(f"Probe Accuracy: {improvements['probe_accuracy']:+.2f}%")
    return results


def plot_training_comparison(baseline_history, exact_history, save_path=None):
    """
    Plot training curves for baseline vs exact models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs_baseline = range(1, len(baseline_history['train_loss']) + 1)
    epochs_exact = range(1, len(exact_history['train_loss']) + 1)
    axes[0, 0].plot(epochs_baseline, baseline_history['train_loss'],
    label='Baseline', marker='o')
    axes[0, 0].plot(epochs_exact, exact_history['train_loss'],
    label='Exact', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs_baseline, baseline_history['test_accuracy'],
    label='Baseline', marker='o')
    axes[0, 1].plot(epochs_exact, exact_history['test_accuracy'],
    label='Exact', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Test Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    if 'exactness_loss' in exact_history:
        axes[1, 0].plot(epochs_exact, exact_history['exactness_loss'],
        label='Exactness Loss', marker='o')
        axes[1, 0].plot(epochs_exact, exact_history['chain_axiom_loss'],
        label='Chain Axiom Loss', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Exactness Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    if 'betti_numbers' in exact_history and exact_history['betti_numbers']:
        betti_array = np.array(exact_history['betti_numbers'])
        for i in range(betti_array.shape[1]):
            axes[1, 1].plot(epochs_exact, betti_array[:, i],
            label=f'Betti {i}', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Betti Number')
        axes[1, 1].set_title('Betti Numbers Over Training')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()


def main():
    """Run MNIST classification experiment with baseline vs exact comparison."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 128
    lr = 1e-3
    lambda_exact = 0.1
    print("MNIST CLASSIFICATION: BASELINE VS EXACT")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Lambda (exactness): {lambda_exact}")
    print("\nLoading MNIST dataset.\n")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print("TRAINING BASELINE MODEL")
    baseline_model = MNISTChainNet(use_chain=False, dtype=torch.float32)
    baseline_history = train_baseline(
        baseline_model, train_loader, test_loader,
        epochs=epochs, lr=lr, device=device,
        save_checkpoints=True
    )
    # exact model
    print("\nTRAINING EXACT MODEL")
    exact_model = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
    exact_history = train_exact(
        exact_model, train_loader, test_loader,
        epochs=epochs, lr=lr, lambda_exact=lambda_exact, device=device,
        save_checkpoints=True, monitor_homology=True
    )
    comparison = compare_models(
        baseline_model, exact_model, test_loader,
        device=device, layer_name='bridge_out'
    )
    output_dir = Path('./results/mnist_classification')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_comparison(
        baseline_history, exact_history,
        save_path=output_dir / 'training_comparison.png'
    )
    results = {
        'baseline_history': {k: v for k, v in baseline_history.items() if k != 'checkpoints'},
        'exact_history': {k: v for k, v in exact_history.items() if k != 'checkpoints'},
        'comparison': comparison
    }

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    results = convert_to_serializable(results)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nEXPERIMENT COMPLETE")
    print(f"Results saved to {output_dir}")
    print(f"\nFinal Test Accuracy:")
    print(f"  Baseline: {baseline_history['test_accuracy'][-1]:.4f}")
    print(f"  Exact:    {exact_history['test_accuracy'][-1]:.4f}")
    print(f"\nFER Score:")
    print(f"  Baseline: {comparison['baseline']['fer_score']:.4f}")
    print(f"  Exact:    {comparison['exact']['fer_score']:.4f}")
    print(f"  Improvement: {comparison['improvements']['fer_score']:+.2f}%")


if __name__ == '__main__':
    main()
