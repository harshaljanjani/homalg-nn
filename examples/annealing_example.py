import torch
import numpy as np
import os
from homalg_nn.nn import ChainModule, AnnealingScheduler, create_recommended_scheduler
from homalg_nn.monitoring import HomologyMonitor


def train_without_annealing(seed: int = 42):
    """Baseline: Train without annealing."""
    print("\nBaseline: Training without annealing")
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
    monitor = HomologyMonitor(log_interval=200, verbose=True)
    num_steps = 2000
    print(f"\nInitial Betti: {chain.get_betti_numbers()}")
    print(f"Training for {num_steps} steps...\n")
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = chain.compute_exactness_loss(mode='combined')
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            monitor.on_step(
                step=step,
                chain_module=chain,
                exactness_loss=chain.compute_exactness_loss(mode='exactness').item(),
                chain_axiom_loss=chain.compute_exactness_loss(mode='chain_axiom').item(),
                total_loss=loss.item()
            )
    final_betti = chain.get_betti_numbers()
    print(f"Final Betti: {final_betti}")
    print(f"Is exact: {sum(final_betti) == 0}")
    monitor.plot(save_path='examples/outputs/annealing_baseline.png', show=False)
    return {'final_betti': final_betti, 'monitor': monitor}


def train_with_annealing(seed: int = 42, schedule: str = 'exponential'):
    """Train with annealing schedule for stable convergence."""
    print(f"\nTrain with annealing: {schedule.upper()} schedule")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chain = ChainModule(
        dimensions=[5, 8, 10, 8, 5],
        init_method='xavier_normal',
        epsilon=1e-3,
        dtype=torch.float64
    )
    optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
    monitor = HomologyMonitor(log_interval=200, verbose=True)
    num_steps = 2000
    scheduler = AnnealingScheduler(
        schedule=schedule,
        total_steps=num_steps,
        exactness_range=(0.1, 1.0),
        chain_axiom_range=(2.0, 0.5)
    )
    print(f"\nInitial Betti: {chain.get_betti_numbers()}")
    print(f"\n{scheduler.get_schedule_info()}")
    print(f"\nTraining for {num_steps} steps...\n")
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = chain.compute_loss_with_annealing(
            step=step,
            total_steps=num_steps,
            schedule=schedule
        )
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            ex_w, ax_w = scheduler.get_weights(step)
            monitor.on_step(
                step=step,
                chain_module=chain,
                exactness_loss=chain.compute_exactness_loss(mode='exactness').item(),
                chain_axiom_loss=chain.compute_exactness_loss(mode='chain_axiom').item(),
                total_loss=loss.item(),
                extra_metrics={
                    'exactness_weight': ex_w,
                    'chain_axiom_weight': ax_w
                }
            )
    final_betti = chain.get_betti_numbers()
    print(f"Final Betti: {final_betti}")
    print(f"Is exact: {sum(final_betti) == 0}")
    monitor.plot(
        save_path=f'examples/outputs/annealing_{schedule}.png',
        show=False
    )
    return {'final_betti': final_betti, 'monitor': monitor}


def train_with_recommended_settings(seed: int = 42):
    """
    Train using recommended settings for best results.
    """
    print("\nRecommended: Using `create_recommended_scheduler()`")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chain = ChainModule(
        dimensions=[5, 8, 10, 8, 5],
        init_method='xavier_normal',
        epsilon=1e-3,
        dtype=torch.float64
    )
    optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
    num_steps = 2000
    scheduler = create_recommended_scheduler(num_steps)
    print(f"\nInitial Betti: {chain.get_betti_numbers()}")
    print(f"\n{scheduler.get_schedule_info()}")
    print(f"\nTraining for {num_steps} steps...\n")
    for step in range(num_steps):
        optimizer.zero_grad()
        ex_w, ax_w = scheduler.get_weights(step)
        loss_exact = chain.compute_exactness_loss(mode='exactness')
        loss_axiom = chain.compute_exactness_loss(mode='chain_axiom')
        loss = ex_w * loss_exact + ax_w * loss_axiom
        loss.backward()
        optimizer.step()
        if step % 400 == 0:
            betti = chain.get_betti_numbers()
            print(f"Step {step:4d}: Betti={betti}, ExW={ex_w:.3f}, AxW={ax_w:.3f}")
    final_betti = chain.get_betti_numbers()
    print(f"Final Betti: {final_betti}")
    print(f"Is exact: {sum(final_betti) == 0}")
    return {'final_betti': final_betti}


def compare_all_methods():
    """Compare baseline vs annealing approaches."""
    print("\nComparison: Baseline vs Annealing")
    results = {}
    results['baseline'] = train_without_annealing(seed=42)
    results['exponential'] = train_with_annealing(seed=42, schedule='exponential')
    results['cosine'] = train_with_annealing(seed=42, schedule='cosine')
    results['recommended'] = train_with_recommended_settings(seed=42)
    print("\nFINAL RESULTS SUMMARY")
    print(f"{'Method':<20} {'Final Betti':<20} {'Exact?'}")
    for method, result in results.items():
        final_betti = result['final_betti']
        is_exact = sum(final_betti) == 0
        exact_str = "[EXACT]" if is_exact else "[NOT EXACT]"
        print(f"{method:<20} {str(final_betti):<20} {exact_str}")

if __name__ == '__main__':    
    print("ANNEALING DEMONSTRATION")
    print("Achieving Stable Convergence to Exactness (Betti=0)")
    os.makedirs('examples/outputs', exist_ok=True)
    compare_all_methods()
    print("\nDemo complete! Check examples/outputs/ for plots.")
