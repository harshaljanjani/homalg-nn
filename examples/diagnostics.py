import torch
import numpy as np
import os
from typing import List, Dict, Any, Optional
from homalg_nn.nn import ChainModule


class DiagnosticMonitor:
    """
    > Singular values of all boundary maps
    > Gradient magnitudes per boundary map
    > Individual loss components
    > Betti transitions with context
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = {
            'steps': [],
            'betti_numbers': [],
            'singular_values': [],   # List[(step, [S_0, S_1, ...])]
            'gradient_norms': [],    # List[(step, [||grad_0||, ||grad_1||, ...])]
            'exactness_loss': [],
            'chain_axiom_loss': [],
            'total_loss': [],
            'betti_transitions': []  # List[(step, old_betti, new_betti)]
        }
        self._last_betti = None

    def on_step(
        self,
        step: int,
        chain_module: ChainModule,
        exactness_loss: float,
        chain_axiom_loss: float,
        total_loss: float
    ):
        """Record diagnostic information for this step."""
        self.history['steps'].append(step)
        self.history['exactness_loss'].append(exactness_loss)
        self.history['chain_axiom_loss'].append(chain_axiom_loss)
        self.history['total_loss'].append(total_loss)
        betti = chain_module.get_betti_numbers()
        self.history['betti_numbers'].append((step, betti))
        if self._last_betti is not None and betti != self._last_betti:
            self.history['betti_transitions'].append((step, self._last_betti, betti))
            if self.verbose:
                print(f"\n* BETTI TRANSITION at step {step} *")
                print(f"    {self._last_betti} -> {betti}")
                print(f"    Exactness loss: {exactness_loss:.6f}")
                print(f"    Chain axiom loss: {chain_axiom_loss:.6f}")
        self._last_betti = betti
        singular_values = []
        with torch.no_grad():
            for i, d in enumerate(chain_module.boundary_maps):
                try:
                    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
                    singular_values.append(S.cpu().numpy())
                except Exception as e:
                    singular_values.append(None)
        self.history['singular_values'].append((step, singular_values))
        grad_norms = []
        for i, d in enumerate(chain_module.boundary_maps):
            if d.grad is not None:
                grad_norm = torch.norm(d.grad).item()
                grad_norms.append(grad_norm)
            else:
                grad_norms.append(0.0)
        self.history['gradient_norms'].append((step, grad_norms))

    def analyze_transitions(self) -> str:
        lines = ["BETTI TRANSITION ANALYSIS"]
        if not self.history['betti_transitions']:
            lines.append("No Betti transitions detected.")
            return "\n".join(lines)
        lines.append(f"\nTotal transitions: {len(self.history['betti_transitions'])}")
        lines.append("\nDetailed transitions:")
        for step, old_betti, new_betti in self.history['betti_transitions']:
            old_rank = sum(old_betti)
            new_rank = sum(new_betti)
            idx = self.history['steps'].index(step)
            ex_loss = self.history['exactness_loss'][idx]
            ca_loss = self.history['chain_axiom_loss'][idx]
            lines.append(f"\n  Step {step}:")
            lines.append(f"    Betti: {old_betti} -> {new_betti} (rank {old_rank} -> {new_rank})")
            lines.append(f"    Exactness loss: {ex_loss:.6f}")
            lines.append(f"    Chain axiom loss: {ca_loss:.6f}")
            step_idx = self.history['steps'].index(step)
            _, svs = self.history['singular_values'][step_idx]
            for i, S in enumerate(svs):
                if S is not None:
                    epsilon = 1e-3
                    near_epsilon = np.sum(np.abs(S - epsilon) < 0.001)
                    if near_epsilon > 0:
                        lines.append(f"    Boundary map {i}: {near_epsilon} singular values near epsilon")
        return "\n".join(lines)

    def plot_diagnostics(self, save_path: Optional[str] = None, show: bool = False):
        """Create comprehensive diagnostic plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available - skipping plots")
            return
        fig = plt.figure(figsize=(16, 10))
        ax1 = plt.subplot(2, 3, 1)
        steps = self.history['steps']
        ax1.plot(steps, self.history['exactness_loss'], 'b-', linewidth=1.5, alpha=0.7)
        ax1.set_yscale('log')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Exactness Loss (log)')
        ax1.set_title('Exactness Loss Evolution')
        ax1.grid(True, alpha=0.3)
        for trans_step, _, _ in self.history['betti_transitions']:
            ax1.axvline(trans_step, color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(steps, self.history['chain_axiom_loss'], 'r-', linewidth=1.5, alpha=0.7)
        ax2.set_yscale('log')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Chain Axiom Loss (log)')
        ax2.set_title('Chain Axiom Loss Evolution')
        ax2.grid(True, alpha=0.3)
        for trans_step, _, _ in self.history['betti_transitions']:
            ax2.axvline(trans_step, color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(steps, self.history['total_loss'], 'g-', linewidth=1.5, alpha=0.7)
        ax3.set_yscale('log')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Total Loss (log)')
        ax3.set_title('Total Loss Evolution')
        ax3.grid(True, alpha=0.3)
        for trans_step, _, _ in self.history['betti_transitions']:
            ax3.axvline(trans_step, color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        ax4 = plt.subplot(2, 3, 4)
        _, first_svs = self.history['singular_values'][0]
        if first_svs[0] is not None:
            num_svs = len(first_svs[0])
            sv_evolution = [[] for _ in range(min(5, num_svs))]
            for _, svs_list in self.history['singular_values']:
                if svs_list[0] is not None:
                    for i in range(min(5, num_svs)):
                        sv_evolution[i].append(svs_list[0][i])
            for i, sv_track in enumerate(sv_evolution):
                ax4.plot(steps[:len(sv_track)], sv_track, label=f'σ_{i}', linewidth=1.5, alpha=0.7)
            ax4.axhline(1e-3, color='black', linestyle='--', alpha=0.5, label='epsilon=1e-3')
            ax4.set_yscale('log')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Singular Value')
            ax4.set_title('Singular Values (Boundary Map 0)')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        ax5 = plt.subplot(2, 3, 5)
        betti_ranks = [sum(betti) for _, betti in self.history['betti_numbers']]
        ax5.plot(steps, betti_ranks, 'o-', markersize=3, linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Homology Rank (sum of Betti)')
        ax5.set_title('Betti Number Evolution')
        ax5.grid(True, alpha=0.3)
        for trans_step, old_betti, new_betti in self.history['betti_transitions']:
            ax5.axvline(trans_step, color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        ax6 = plt.subplot(2, 3, 6)
        num_maps = len(self.history['gradient_norms'][0][1])
        for i in range(num_maps):
            grad_track = [norms[i] for _, norms in self.history['gradient_norms']]
            ax6.plot(steps[:len(grad_track)], grad_track, label=f'∂_map_{i}', linewidth=1.5, alpha=0.7)
        ax6.set_yscale('log')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Gradient Norm (log)')
        ax6.set_title('Gradient Norms per Boundary Map')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nDiagnostic plots saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()


def experiment_baseline(seed: int = 42, num_steps: int = 2000) -> Dict[str, Any]:
    """
    Baseline experiment: Current default settings.
    """
    print("\nExperiment: Baseline (Current Settings)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chain = ChainModule(
        dimensions=[5, 8, 10, 8, 5],
        init_method='xavier_normal',
        exactness_weight=1.0,
        chain_axiom_weight=0.5,
        sparsity_weight=0.1,
        epsilon=1e-3,
        dtype=torch.float64
    )
    optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
    monitor = DiagnosticMonitor(verbose=False)
    print(f"\nInitial Betti: {chain.get_betti_numbers()}")
    print(f"Training for {num_steps} steps...")
    for step in range(num_steps):
        optimizer.zero_grad()
        loss_exact = chain.compute_exactness_loss(mode='exactness')
        loss_axiom = chain.compute_exactness_loss(mode='chain_axiom')
        total_loss = loss_exact + 0.5 * loss_axiom
        total_loss.backward()
        optimizer.step()
        monitor.on_step(
            step=step,
            chain_module=chain,
            exactness_loss=loss_exact.item(),
            chain_axiom_loss=loss_axiom.item(),
            total_loss=total_loss.item()
        )
        if step % 200 == 0:
            betti = chain.get_betti_numbers()
            print(f"Step {step:4d}: Betti={betti}, ExLoss={loss_exact.item():.4f}, "
                  f"AxLoss={loss_axiom.item():.4f}")
    final_betti = chain.get_betti_numbers()
    print(f"\nFinal Betti: {final_betti}")
    print(monitor.analyze_transitions())
    monitor.plot_diagnostics(
        save_path='examples/outputs/baseline_diagnostics.png',
        show=False
    )
    return {
        'name': 'baseline',
        'monitor': monitor,
        'chain': chain,
        'final_betti': final_betti
    }


def experiment_annealing_schedule(
    seed: int = 42,
    num_steps: int = 2000,
    schedule_type: str = 'linear'
) -> Dict[str, Any]:
    """
    Test annealing schedule for loss weights.
    """
    print(f"\nExperiment: Annealing Schedule ({schedule_type})")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chain = ChainModule(
        dimensions=[5, 8, 10, 8, 5],
        init_method='xavier_normal',
        exactness_weight=1.0,
        chain_axiom_weight=0.5,
        sparsity_weight=0.1,
        epsilon=1e-3,
        dtype=torch.float64
    )
    optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
    monitor = DiagnosticMonitor(verbose=False)
    print(f"\nInitial Betti: {chain.get_betti_numbers()}")
    print(f"Training for {num_steps} steps with {schedule_type} annealing...")
    for step in range(num_steps):
        optimizer.zero_grad()
        progress = step / num_steps
        if schedule_type == 'linear':
            exactness_weight = 0.1 + 0.9 * progress    # 0.1 -> 1.0
            chain_axiom_weight = 2.0 - 1.5 * progress  # 2.0 -> 0.5
        elif schedule_type == 'exponential':
            exactness_weight = 0.1 * (10 ** progress)
            chain_axiom_weight = 2.0 * (0.25 ** progress)
        elif schedule_type == 'cosine':
            exactness_weight = 0.1 + 0.45 * (1 - np.cos(np.pi * progress))
            chain_axiom_weight = 2.0 - 0.75 * (1 - np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        loss_exact = chain.compute_exactness_loss(mode='exactness')
        loss_axiom = chain.compute_exactness_loss(mode='chain_axiom')
        total_loss = exactness_weight * loss_exact + chain_axiom_weight * loss_axiom
        total_loss.backward()
        optimizer.step()
        monitor.on_step(
            step=step,
            chain_module=chain,
            exactness_loss=loss_exact.item(),
            chain_axiom_loss=loss_axiom.item(),
            total_loss=total_loss.item()
        )
        if step % 200 == 0:
            betti = chain.get_betti_numbers()
            print(f"Step {step:4d}: Betti={betti}, ExW={exactness_weight:.3f}, "
                  f"AxW={chain_axiom_weight:.3f}, ExLoss={loss_exact.item():.4f}")
    final_betti = chain.get_betti_numbers()
    print(f"\nFinal Betti: {final_betti}")
    print(monitor.analyze_transitions())
    monitor.plot_diagnostics(
        save_path=f'examples/outputs/annealing_{schedule_type}_diagnostics.png',
        show=False
    )
    return {
        'name': f'annealing_{schedule_type}',
        'monitor': monitor,
        'chain': chain,
        'final_betti': final_betti
    }


def experiment_two_stage_training(seed: int = 42) -> Dict[str, Any]:
    """
    Two-stage training strategy:
    Stage 1 (1000 steps): Focus purely on chain axiom (enforce `d ∘ d = 0`)
    Stage 2 (1000 steps): Focus on exactness (enforce `Betti=0`)
    """
    print("\nExperiment: Two-Stage Training")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chain = ChainModule(
        dimensions=[5, 8, 10, 8, 5],
        init_method='xavier_normal',
        exactness_weight=1.0,
        chain_axiom_weight=0.5,
        sparsity_weight=0.1,
        epsilon=1e-3,
        dtype=torch.float64
    )
    optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
    monitor = DiagnosticMonitor(verbose=False)
    print(f"\nInitial Betti: {chain.get_betti_numbers()}")
    print("\nStage 1: Chain Axiom Training (1000 steps)")
    for step in range(1000):
        optimizer.zero_grad()
        loss_axiom = chain.compute_exactness_loss(mode='chain_axiom')
        total_loss = loss_axiom
        total_loss.backward()
        optimizer.step()
        loss_exact = chain.compute_exactness_loss(mode='exactness')
        monitor.on_step(
            step=step,
            chain_module=chain,
            exactness_loss=loss_exact.item(),
            chain_axiom_loss=loss_axiom.item(),
            total_loss=total_loss.item()
        )
        if step % 200 == 0:
            betti = chain.get_betti_numbers()
            print(f"Step {step:4d}: Betti={betti}, AxLoss={loss_axiom.item():.4f}")
    stage1_betti = chain.get_betti_numbers()
    print(f"After Stage 1 Betti: {stage1_betti}")
    print("\nStage 2: Exactness Training (1000 steps)")
    for step in range(1000, 2000):
        optimizer.zero_grad()
        loss_exact = chain.compute_exactness_loss(mode='exactness')
        loss_axiom = chain.compute_exactness_loss(mode='chain_axiom')
        total_loss = 2.0 * loss_exact + 0.1 * loss_axiom
        total_loss.backward()
        optimizer.step()
        monitor.on_step(
            step=step,
            chain_module=chain,
            exactness_loss=loss_exact.item(),
            chain_axiom_loss=loss_axiom.item(),
            total_loss=total_loss.item()
        )
        if step % 200 == 0:
            betti = chain.get_betti_numbers()
            print(f"Step {step:4d}: Betti={betti}, ExLoss={loss_exact.item():.4f}")
    final_betti = chain.get_betti_numbers()
    print(f"\nFinal Betti: {final_betti}")
    print(monitor.analyze_transitions())
    monitor.plot_diagnostics(
        save_path='examples/outputs/two_stage_diagnostics.png',
        show=False
    )
    return {
        'name': 'two_stage',
        'monitor': monitor,
        'chain': chain,
        'final_betti': final_betti
    }


def experiment_higher_epsilon(seed: int = 42, epsilon: float = 1e-2) -> Dict[str, Any]:
    """
    Test with higher epsilon to reduce threshold sensitivity.
    """
    print(f"\nExperiment: Higher Epsilon (epsilon={epsilon})")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chain = ChainModule(
        dimensions=[5, 8, 10, 8, 5],
        init_method='xavier_normal',
        exactness_weight=1.0,
        chain_axiom_weight=0.5,
        sparsity_weight=0.1,
        epsilon=epsilon,
        dtype=torch.float64
    )
    optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
    monitor = DiagnosticMonitor(verbose=False)
    print(f"\nInitial Betti: {chain.get_betti_numbers()}")
    print(f"Training for 2000 steps with epsilon={epsilon}...")
    for step in range(2000):
        optimizer.zero_grad()
        loss_exact = chain.compute_exactness_loss(mode='exactness')
        loss_axiom = chain.compute_exactness_loss(mode='chain_axiom')
        total_loss = loss_exact + 0.5 * loss_axiom
        total_loss.backward()
        optimizer.step()
        monitor.on_step(
            step=step,
            chain_module=chain,
            exactness_loss=loss_exact.item(),
            chain_axiom_loss=loss_axiom.item(),
            total_loss=total_loss.item()
        )
        if step % 200 == 0:
            betti = chain.get_betti_numbers()
            print(f"Step {step:4d}: Betti={betti}, ExLoss={loss_exact.item():.4f}")
    final_betti = chain.get_betti_numbers()
    print(f"\nFinal Betti: {final_betti}")
    print(monitor.analyze_transitions())
    monitor.plot_diagnostics(
        save_path=f'examples/outputs/epsilon_{epsilon}_diagnostics.png',
        show=False
    )
    return {
        'name': f'epsilon_{epsilon}',
        'monitor': monitor,
        'chain': chain,
        'final_betti': final_betti
    }


def compare_experiments(results: List[Dict[str, Any]]):
    """Compare all experimental results."""
    print("\nExperimental Comparison")
    print(f"\n{'Experiment':<25} {'Final Betti':<20} {'Transitions':<15} {'Final Exact?'}")
    for result in results:
        name = result['name']
        final_betti = result['final_betti']
        num_transitions = len(result['monitor'].history['betti_transitions'])
        is_exact = sum(final_betti) == 0
        print(f"{name:<25} {str(final_betti):<20} {num_transitions:<15} {is_exact}")
    exact_results = [r for r in results if sum(r['final_betti']) == 0]
    if exact_results:
        print(f"\n[SUCCESS] {len(exact_results)} experiments achieved final exactness (Betti=0)")
        best = min(exact_results, key=lambda r: len(r['monitor'].history['betti_transitions']))
        print(f"\nMost stable exact solution: {best['name']}")
        print(f"  Transitions: {len(best['monitor'].history['betti_transitions'])}")
    else:
        print("\n[FAILURE] No experiments achieved final exactness")
        best = min(results, key=lambda r: sum(r['final_betti']))
        print(f"\nClosest to exact: {best['name']}")
        print(f"  Final Betti: {best['final_betti']}")


if __name__ == '__main__':
    print("DIAGNOSTIC EXPERIMENTS")
    print("Investigating Betti Number Oscillation")
    os.makedirs('examples/outputs', exist_ok=True)
    results = []
    results.append(experiment_baseline(seed=42, num_steps=2000))
    results.append(experiment_annealing_schedule(seed=42, schedule_type='linear'))
    results.append(experiment_annealing_schedule(seed=42, schedule_type='exponential'))
    results.append(experiment_annealing_schedule(seed=42, schedule_type='cosine'))
    results.append(experiment_two_stage_training(seed=42))
    results.append(experiment_higher_epsilon(seed=42, epsilon=1e-2))
    compare_experiments(results)
    print("\nAll diagnostic experiments completed!")
    print("Check examples/outputs/ for diagnostic plots")
