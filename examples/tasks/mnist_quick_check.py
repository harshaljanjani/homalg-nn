import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_classification import MNISTChainNet
from mnist_iteration_study import quick_train_exact
from mnist_chain_diagnostics import ChainDiagnostics


def test_imports():
    """Verify all imports work."""
    print("\nChecking imports.")
    try:
        from homalg_nn.nn import ChainModule, create_recommended_scheduler
        from homalg_nn.monitoring import HomologyMonitor
        from homalg_nn.analysis import RepresentationMetrics, FERDetector
        print("  [PASS] All imports successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_model_creation():
    """Verify model instantiation."""
    print("\nCreating models.")
    try:
        model1 = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
        x = torch.randn(4, 1, 28, 28, dtype=torch.float64)
        y = model1(x)
        assert y.shape == (4, 10), f"Expected (4, 10), got {y.shape}"
        betti = model1.chain.get_betti_numbers()
        print(f"  [PASS] Model created, initial Betti: {betti}")
        return True
    except Exception as e:
        print(f"  [FAIL] Model creation failed: {e}")
        return False


def test_training_one_epoch():
    """Run one epoch of training."""
    print("\nTraining for 1 epoch.")
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        model = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
        history = quick_train_exact(
            model, train_loader, test_loader,
            epochs=1, lambda_exact=0.5, device='cpu'
        )
        final_acc = history['test_accuracy'][-1]
        final_betti = history['final_betti']
        print(f"  [PASS] Training successful")
        print(f"    Accuracy: {final_acc:.4f}")
        print(f"    Betti: {final_betti}")
        return True
    except Exception as e:
        print(f"  [FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diagnostics():
    """Verify diagnostics work."""
    print("\nTesting diagnostics.")
    try:
        model = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
        diagnostics = ChainDiagnostics(model.chain, log_interval=10)
        for step in [0, 10, 20]:
            diagnostics.log_step(step)
        print(f"  [PASS] Diagnostics working")
        print(f"    Logged {len(diagnostics.history['steps'])} steps")
        return True
    except Exception as e:
        print(f"  [FAIL] Diagnostics failed: {e}")
        return False


def main():
    """Run all sanity checks."""
    print("MNIST ITERATION SANITY CHECK")
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Training (1 epoch)", test_training_one_epoch()))
    results.append(("Diagnostics", test_diagnostics()))
    print("\nSUMMARY")
    all_pass = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:<25} {status}")
        if not passed:
            all_pass = False
    if all_pass:
        print("\n[SUCCESS] All tests passed! Ready to run full iteration study.")
        print("\nTo run full experiment:")
        print("  cd examples/tasks")
        print("  python mnist_iteration_study.py")
        print("\nTo run diagnostics only:")
        print("  python mnist_chain_diagnostics.py")
    else:
        print("\n[FAILURE] Some tests failed. Please fix before running full study.")
    return all_pass


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
