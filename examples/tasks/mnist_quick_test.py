from mnist_classification import *

if __name__ == '__main__':
    device = 'cpu'
    epochs = 3
    batch_size = 256
    lr = 1e-3
    lambda_exact = 0.1
    print("MNIST QUICK TEST: BASELINE VS EXACT (3 EPOCHS)")
    print("Loading MNIST dataset.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    baseline_model = MNISTChainNet(use_chain=False, dtype=torch.float32)
    baseline_history = train_baseline(
        baseline_model, train_loader, test_loader,
        epochs=epochs, lr=lr, device=device
    )
    exact_model = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
    exact_history = train_exact(
        exact_model, train_loader, test_loader,
        epochs=epochs, lr=lr, lambda_exact=lambda_exact, device=device,
        monitor_homology=True
    )
    comparison = compare_models(
        baseline_model, exact_model, test_loader,
        device=device, layer_name='bridge_out'
    )
    print("\nRESULTS SUMMARY")
    print(f"\nFinal Test Accuracy:")
    print(f"  Baseline: {baseline_history['test_accuracy'][-1]:.4f}")
    print(f"  Exact:    {exact_history['test_accuracy'][-1]:.4f}")
    print(f"\nFinal Betti Numbers (Exact): {exact_history['betti_numbers'][-1] if exact_history['betti_numbers'] else 'N/A'}")
    print(f"\nFER Scores:")
    print(f"  Baseline: {comparison['baseline']['fer_score']:.4f}")
    print(f"  Exact:    {comparison['exact']['fer_score']:.4f}")
    print(f"  Improvement: {comparison['improvements']['fer_score']:+.2f}%")
    print(f"\nEntanglement:")
    print(f"  Baseline: {comparison['baseline']['entanglement']:.4f}")
    print(f"  Exact:    {comparison['exact']['entanglement']:.4f}")
    print(f"  Improvement: {comparison['improvements']['entanglement']:+.2f}%")
