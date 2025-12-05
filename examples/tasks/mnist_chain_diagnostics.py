import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mnist_classification import MNISTChainNet, evaluate_model
from homalg_nn.nn import create_recommended_scheduler


class ChainDiagnostics:
    def __init__(self, chain_module, log_interval=50):
        self.chain = chain_module
        self.log_interval = log_interval
        self.history = {
            'steps': [],
            'singular_values': [],  # List[`S_0`, `S_1`, `S_2`, `S_3`] (per step)
            'gradient_norms': [],
            'exactness_defects': [],
            'betti_numbers': [],
            'epsilon': chain_module.epsilon
        }

    def log_step(self, step):
        """Record diagnostics for current step."""
        if step % self.log_interval != 0:
            return
        self.history['steps'].append(step)
        # singular values
        svs = []
        with torch.no_grad():
            for d in self.chain.boundary_maps:
                U, S, Vh = torch.linalg.svd(d, full_matrices=False)
                svs.append(S.cpu().numpy())
        self.history['singular_values'].append(svs)
        # gradient norms
        grad_norms = []
        for d in self.chain.boundary_maps:
            if d.grad is not None:
                grad_norms.append(torch.norm(d.grad).item())
            else:
                grad_norms.append(0.0)
        self.history['gradient_norms'].append(grad_norms)
        # exactness defects
        defects = self.chain.get_exactness_defects()
        self.history['exactness_defects'].append(defects)
        # betti numbers
        betti = self.chain.get_betti_numbers()
        self.history['betti_numbers'].append(betti)

    def plot_diagnostics(self, save_path=None):
        """Create comprehensive diagnostic plots."""
        fig = plt.figure(figsize=(18, 12))
        steps = self.history['steps']
        epsilon = self.history['epsilon']

        # singular values over time (first boundary map)
        ax1 = plt.subplot(3, 3, 1)
        svs_0 = [sv[0] for sv in self.history['singular_values']]
        for i in range(min(10, len(svs_0[0]))):
            sv_track = [sv[i] for sv in svs_0]
            ax1.plot(steps, sv_track, alpha=0.7, label=f'σ_{i}')
        ax1.axhline(epsilon, color='red', linestyle='--', label=f'ε={epsilon}')
        ax1.set_yscale('log')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('Singular Values: d_0 (16x32)')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        # singular values (second boundary map)
        ax2 = plt.subplot(3, 3, 2)
        svs_1 = [sv[1] for sv in self.history['singular_values']]
        for i in range(min(10, len(svs_1[0]))):
            sv_track = [sv[i] for sv in svs_1]
            ax2.plot(steps, sv_track, alpha=0.7, label=f'σ_{i}')
        ax2.axhline(epsilon, color='red', linestyle='--', label=f'ε={epsilon}')
        ax2.set_yscale('log')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Singular Value')
        ax2.set_title('Singular Values: d_1 (32x64)')
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)

        # singular values (third boundary map)
        ax3 = plt.subplot(3, 3, 3)
        svs_2 = [sv[2] for sv in self.history['singular_values']]
        for i in range(min(10, len(svs_2[0]))):
            sv_track = [sv[i] for sv in svs_2]
            ax3.plot(steps, sv_track, alpha=0.7, label=f'σ_{i}')
        ax3.axhline(epsilon, color='red', linestyle='--', label=f'ε={epsilon}')
        ax3.set_yscale('log')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Singular Value')
        ax3.set_title('Singular Values: d_2 (64x128)')
        ax3.legend(fontsize=7, ncol=2)
        ax3.grid(True, alpha=0.3)

        # singular values (fourth boundary map)
        ax4 = plt.subplot(3, 3, 4)
        svs_3 = [sv[3] for sv in self.history['singular_values']]
        for i in range(min(10, len(svs_3[0]))):
            sv_track = [sv[i] for sv in svs_3]
            ax4.plot(steps, sv_track, alpha=0.7, label=f'σ_{i}')
        ax4.axhline(epsilon, color='red', linestyle='--', label=f'ε={epsilon}')
        ax4.set_yscale('log')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Singular Value')
        ax4.set_title('Singular Values: d_3 (128x256)')
        ax4.legend(fontsize=7, ncol=2)
        ax4.grid(True, alpha=0.3)

        # number of singular values below epsilon
        ax5 = plt.subplot(3, 3, 5)
        for i, label in enumerate(['d_0', 'd_1', 'd_2', 'd_3']):
            below_eps = []
            for sv_list in self.history['singular_values']:
                sv = sv_list[i]
                below_eps.append(np.sum(sv < epsilon))
            ax5.plot(steps, below_eps, marker='o', label=label)
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('# SVs < ε')
        ax5.set_title('Kernel Dimension Growth')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # gradient norms
        ax6 = plt.subplot(3, 3, 6)
        for i in range(len(self.history['gradient_norms'][0])):
            grad_track = [gn[i] for gn in self.history['gradient_norms']]
            ax6.plot(steps, grad_track, marker='o', label=f'd_{i}')
        ax6.set_yscale('log')
        ax6.set_xlabel('Training Step')
        ax6.set_ylabel('Gradient Norm')
        ax6.set_title('Gradient Flow Through Boundary Maps')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # exactness defects
        ax7 = plt.subplot(3, 3, 7)
        for i in range(len(self.history['exactness_defects'][0])):
            defect_track = [d[i] for d in self.history['exactness_defects']]
            ax7.plot(steps, defect_track, marker='o', label=f'defect_{i}')
        ax7.set_yscale('log')
        ax7.set_xlabel('Training Step')
        ax7.set_ylabel('Exactness Defect')
        ax7.set_title('Exactness Defects: ||im(d_{i+1}) - ker(d_i)||')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # betti numbers
        ax8 = plt.subplot(3, 3, 8)
        betti_array = np.array(self.history['betti_numbers'])
        for i in range(betti_array.shape[1]):
            ax8.plot(steps, betti_array[:, i], marker='o', label=f'Betti_{i}')
        ax8.set_xlabel('Training Step')
        ax8.set_ylabel('Betti Number')
        ax8.set_title('Betti Numbers Over Training')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # rank of each boundary map
        ax9 = plt.subplot(3, 3, 9)
        for i in range(4):
            rank_track = []
            for sv_list in self.history['singular_values']:
                sv = sv_list[i]
                rank = np.sum(sv > epsilon)
                rank_track.append(rank)
            ax9.plot(steps, rank_track, marker='o', label=f'rank(d_{i})')
        ax9.set_xlabel('Training Step')
        ax9.set_ylabel('Rank')
        ax9.set_title('Rank Evolution of Boundary Maps')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nDiagnostics saved to {save_path}")
        return fig

    def print_summary(self):
        """Print diagnostic summary."""
        print("\nCHAIN DIAGNOSTICS SUMMARY")
        if not self.history['steps']:
            print("No data collected")
            return
        final_betti = self.history['betti_numbers'][-1]
        print(f"\nFinal Betti Numbers: {final_betti}")
        print(f"Sum: {sum(final_betti)} (target: 0)")
        print("\nFinal Singular Value Analysis:")
        final_svs = self.history['singular_values'][-1]
        epsilon = self.history['epsilon']
        for i, sv in enumerate(final_svs):
            n_below = np.sum(sv < epsilon)
            n_above = np.sum(sv > epsilon)
            min_sv = np.min(sv)
            max_sv = np.max(sv)
            print(f"  d_{i}: {n_above} SVs > ε, {n_below} SVs < ε (min={min_sv:.6f}, max={max_sv:.2f})")
        print("\nFinal Exactness Defects:")
        final_defects = self.history['exactness_defects'][-1]
        for i, defect in enumerate(final_defects):
            print(f"  ||im(d_{i+1}) - ker(d_{i})||: {defect:.6f}")
        # analyze why `Betti_3` is stuck
        if final_betti[3] > 0:
            print(f"\nANALYSIS: Betti_3 = {final_betti[3]}")
            print("   This means ker(d_3) has large dimension")
            sv_3 = final_svs[3]
            n_small = np.sum(sv_3 < epsilon)
            print(f"   → d_3 has {n_small} singular values < {epsilon}")
            print(f"   → This creates a {n_small}-dimensional kernel")
            print(f"   → For exactness, need im(d_4) to span this kernel")
            print(f"   → But no d_4 in this chain (boundary condition)")
            print("\n   DIAGNOSIS: Chain needs either:")
            print("     1. Stronger exactness loss to reduce kernel of d_3")
            print("     2. Add regularization to increase rank of d_3")
            print("     3. Use different architecture (symmetric chain)")


def train_with_diagnostics(model, train_loader, test_loader, epochs=10,
    lambda_exact=0.5, device='cpu'):
    """Train model while collecting detailed diagnostics."""
    print(f"\nTraining with diagnostics for {epochs} epochs")
    print(f"Lambda: {lambda_exact}, Device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_steps = len(train_loader) * epochs
    scheduler = create_recommended_scheduler(total_steps)
    diagnostics = ChainDiagnostics(model.chain, log_interval=50)
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_task = 0
        epoch_exact = 0
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
            diagnostics.log_step(step)
            step += 1
        test_acc = evaluate_model(model, test_loader, device)
        betti = model.chain.get_betti_numbers()
        print(f"Epoch {epoch+1}/{epochs} - Task: {epoch_task/len(train_loader):.4f}, "
              f"Exact: {epoch_exact/len(train_loader):.4f}, "
              f"Acc: {test_acc:.4f}, Betti: {betti}")
    return diagnostics


def main():
    """Run diagnostic analysis."""
    print("MNIST CHAIN DIAGNOSTICS")
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
    for lam in [0.5, 1.0]:
        print(f"\nDIAGNOSTIC RUN: lambda = {lam}")
        model = MNISTChainNet(use_chain=True, epsilon=1e-3, dtype=torch.float64)
        diagnostics = train_with_diagnostics(
            model, train_loader, test_loader,
            epochs=10, lambda_exact=lam, device=device
        )
        diagnostics.print_summary()
        output_dir = Path('./results/mnist_diagnostics')
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics.plot_diagnostics(
            save_path=output_dir / f'diagnostics_lambda_{lam}.png'
        )
    print("\nDIAGNOSTICS COMPLETE")


if __name__ == '__main__':
    main()
