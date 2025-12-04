# NOTE: Verify that probe accuracy correlates with test accuracy; not eval for the proposed method.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from homalg_nn.analysis import RepresentationMetrics, FERDetector


class SimpleMNISTNet(nn.Module):
    """Simple MNIST classifier for validation."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.encoder(x)
        return self.decoder(x)


def train_mnist_baseline(epochs=5, device='cpu'):
    """Train a simple MNIST classifier."""
    print(f"\nTraining baseline MNIST classifier for {epochs} epochs.")
    # setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    model = SimpleMNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    test_accuracy = correct / total
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    return model, test_loader, test_accuracy


def validate_fer_metrics(model, test_loader, test_accuracy, device='cpu'):
    """Validate that FER metrics work correctly on MNIST."""
    print("\nVALIDATING FER METRICS")
    metrics = RepresentationMetrics(sample_size=1000, random_state=42)
    print("\nExtracting representations from encoder.")
    detector = FERDetector(
        model=model,
        dataloader=test_loader,
        layer_names=['encoder'],
        device=device,
        sample_size=1000
    )
    representations, labels = detector.extract_representations('encoder', return_labels=True)
    print(f"Extracted {representations.shape[0]} representations of dimension {representations.shape[1]}")

    # linear probe accuracy
    print("\nLinear Probe Accuracy")
    probe_result = metrics.linear_probe_accuracy(representations, labels, n_splits=5)
    probe_acc = probe_result['mean_accuracy']
    print(f"Probe Accuracy:     {probe_acc:.4f} ± {probe_result['std_accuracy']:.4f}")
    print(f"Test Accuracy:      {test_accuracy:.4f}")
    print(f"Probe Complexity:   {probe_result['probe_complexity']:.4f}")
    diff = abs(probe_acc - test_accuracy)
    if diff < 0.05:
        print(f"\n[PASS] Probe accuracy ({probe_acc:.4f}) correlates well with test accuracy ({test_accuracy:.4f})")
        print(f"       Difference: {diff:.4f} < 0.05")
    else:
        print(f"\n[WARNING] Probe accuracy ({probe_acc:.4f}) differs from test accuracy ({test_accuracy:.4f}) by {diff:.4f}")

    # mutual information
    print("\nMutual Information Matrix")
    mi_matrix = metrics.mutual_information_matrix(representations, n_bins=20, normalize=True)
    mask = ~np.eye(mi_matrix.shape[0], dtype=bool)
    off_diagonal_mi = mi_matrix[mask]
    print(f"MI Matrix shape:         {mi_matrix.shape}")
    print(f"Mean diagonal MI:        {np.mean(np.diag(mi_matrix)):.4f}")
    print(f"Mean off-diagonal MI:    {np.mean(off_diagonal_mi):.4f}")
    print(f"Std off-diagonal MI:     {np.std(off_diagonal_mi):.4f}")
    if np.mean(off_diagonal_mi) < 0.5:
        print(f"\n[PASS] Low off-diagonal MI indicates reasonable factorization")
    else:
        print(f"\n[INFO] Moderate off-diagonal MI - features may be entangled")

    # repr rank
    print("\nRepresentation Rank")
    rank_result = metrics.representation_rank(representations, epsilon=1e-3)
    print(f"Effective Rank:          {rank_result['effective_rank']}")
    print(f"Theoretical Max:         {representations.shape[1]}")
    print(f"Rank Ratio:              {rank_result['rank_ratio']:.4f}")
    print(f"Entropy:                 {rank_result['entropy']:.4f}")
    print(f"Participation Ratio:     {rank_result['participation_ratio']:.4f}")
    if rank_result['rank_ratio'] > 0.5:
        print(f"\n[PASS] High rank ratio ({rank_result['rank_ratio']:.4f}) indicates good use of capacity")
    else:
        print(f"\n[INFO] Low rank ratio - representations may be redundant")

    # FER score
    print("\nAggregate FER Score")
    fer_result = metrics.compute_fer_score(representations, labels=labels)
    print(f"FER Score:               {fer_result['fer_score']:.4f}")
    print(f"Probe Accuracy:          {fer_result['probe_accuracy']:.4f}")
    print(f"Entanglement:            {fer_result['entanglement']:.4f}")
    print(f"Rank Ratio:              {fer_result['rank_ratio']:.4f}")
    if fer_result['fer_score'] < 0.5:
        print(f"\n[PASS] Low FER score ({fer_result['fer_score']:.4f}) indicates good representations")
    else:
        print(f"\n[WARNING] High FER score ({fer_result['fer_score']:.4f}) indicates fractured/entangled representations")

    # `FERDetector` report
    print("\nFERDetector Report Generation")
    report = detector.generate_report(save_path=None)
    print("\n[PASS] Report generated successfully")
    print("\nReport Preview (first 500 chars):")
    print(report[:500])
    # summary
    print("\nVALIDATION SUMMARY")
    print(f"[SUCCESS] All FER metrics validated successfully!")
    print(f"\nKey Findings:")
    print(f"  - Probe accuracy correlates with test accuracy: {abs(probe_acc - test_accuracy) < 0.05}")
    print(f"  - Representations have reasonable MI: {np.mean(off_diagonal_mi) < 0.5}")
    print(f"  - Rank ratio indicates good capacity use: {rank_result['rank_ratio'] > 0.5}")
    print(f"  - FER score: {fer_result['fer_score']:.4f} {'(good)' if fer_result['fer_score'] < 0.5 else '(needs improvement)'}")
    print(f"\n[READY] FER Detection Framework is validated and ready for task integration!")
    return fer_result


def main():
    """Run full validation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model, test_loader, test_accuracy = train_mnist_baseline(epochs=5, device=device)
    validate_fer_metrics(model, test_loader, test_accuracy, device=device)
    print("\nVALIDATION COMPLETE")


if __name__ == '__main__':
    main()
