import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from examples.arc import (
    ARCChainSolver,
    BaselineARCSolver,
    ARCDataset,
    ARCTaskDataset,
    BaselineEvaluator
)
from homalg_nn.nn import create_recommended_scheduler


def print_header(text):
    """Print formatted section header."""
    print(text)


def test_data_loading():
    """Data loading."""
    print_header("Data Loading")
    data_path = Path("data/arc/subset")
    if not data_path.exists():
        print("✗ Dataset not found. Run: python scripts/download_arc_data.py")
        return False
    try:
        dataset = ARCDataset(
            str(data_path),
            split='train',
            max_grid_size=30
        )
        print(f"[PASS] Loaded dataset: {len(dataset)} examples")
        input_grid, output_grid, mask = dataset[0]
        print(f"  Example 0:")
        print(f"    Input shape:  {input_grid.shape}")
        print(f"    Output shape: {output_grid.shape}")
        print(f"    Mask shape:   {mask.shape}")
        print(f"    Input values: {input_grid.unique().tolist()}")
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"  Batch:")
        print(f"    Input:  {batch[0].shape}")
        print(f"    Output: {batch[1].shape}")
        print(f"    Mask:   {batch[2].shape}")
        print("[PASS] Data loading works!")
        return True
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Model forward pass."""
    print_header("Model Forward Pass")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    try:
        chain_model = ARCChainSolver(
            max_grid_size=30,
            chain_dims=[16, 32, 64, 128, 256],
            epsilon=1e-3,
            dtype=torch.float64
        ).to(device)
        baseline_model = BaselineARCSolver(
            max_grid_size=30,
            embed_dim=256,
            hidden_dim=512,
            dtype=torch.float64
        ).to(device)
        print(f"[PASS] Created models")
        print(f"  Chain model:    {sum(p.numel() for p in chain_model.parameters()):,} params")
        print(f"  Baseline model: {sum(p.numel() for p in baseline_model.parameters()):,} params")
        batch_size = 2
        h, w = 10, 10
        input_grid = torch.randint(0, 10, (batch_size, h, w), device=device)
        mask = torch.ones(batch_size, h, w, dtype=torch.bool, device=device)
        chain_model.eval()
        with torch.no_grad():
            logits_chain = chain_model(input_grid, grid_mask=mask)
        print(f"  Chain forward:")
        print(f"    Input:  {input_grid.shape}")
        print(f"    Output: {logits_chain.shape}")
        print(f"    Output range: [{logits_chain.min():.2f}, {logits_chain.max():.2f}]")
        baseline_model.eval()
        with torch.no_grad():
            logits_baseline = baseline_model(input_grid, grid_mask=mask)
        print(f"  Baseline forward:")
        print(f"    Output: {logits_baseline.shape}")
        print(f"    Output range: [{logits_baseline.min():.2f}, {logits_baseline.max():.2f}]")
        print("[PASS] Model forward pass works!")
        return True
    except Exception as e:
        print(f"[FAIL] Model forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Loss computation."""
    print_header("Loss Computation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = ARCChainSolver(
            max_grid_size=30,
            chain_dims=[16, 32, 64, 128, 256],
            epsilon=1e-3,
            dtype=torch.float64
        ).to(device)
        batch_size = 2
        h, w = 10, 10
        input_grid = torch.randint(0, 10, (batch_size, h, w), device=device)
        output_grid = torch.randint(0, 10, (batch_size, h, w), device=device)
        mask = torch.ones(batch_size, h, w, dtype=torch.bool, device=device)
        model.train()
        logits = model(input_grid, grid_mask=mask)
        loss_task = F.cross_entropy(
            logits.permute(0, 3, 1, 2),  # `(batch, 10, h, w)`
            output_grid,
            reduction='none'
        )
        loss_task = (loss_task * mask.float()).sum() / mask.sum()
        print(f"  Task loss: {loss_task.item():.4f}")
        loss_exact = model.compute_exactness_loss(mode='exactness')
        loss_axiom = model.compute_exactness_loss(mode='chain_axiom')
        print(f"  Exactness loss: {loss_exact.item():.4f}")
        print(f"  Chain axiom loss: {loss_axiom.item():.4f}")
        lambda_exact = 0.5
        loss_total = loss_task + lambda_exact * (loss_exact + loss_axiom)
        print(f"  Total loss: {loss_total.item():.4f}")
        print("[PASS] Loss computation works!")
        return True
    except Exception as e:
        print(f"[FAIL] Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Single training step."""
    print_header("Training Step")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = ARCChainSolver(
            max_grid_size=30,
            chain_dims=[16, 32, 64, 128, 256],
            epsilon=1e-3,
            dtype=torch.float64
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        total_steps = 100
        scheduler = create_recommended_scheduler(total_steps)
        batch_size = 2
        h, w = 10, 10
        input_grid = torch.randint(0, 10, (batch_size, h, w), device=device)
        output_grid = torch.randint(0, 10, (batch_size, h, w), device=device)
        mask = torch.ones(batch_size, h, w, dtype=torch.bool, device=device)
        model.train()
        optimizer.zero_grad()
        logits = model(input_grid, grid_mask=mask)
        loss_task = F.cross_entropy(
            logits.permute(0, 3, 1, 2),
            output_grid,
            reduction='none'
        )
        loss_task = (loss_task * mask.float()).sum() / mask.sum()
        # exactness loss with annealing
        step = 0
        ex_w, ax_w = scheduler.get_weights(step)
        loss_exact = model.compute_exactness_loss(mode='exactness')
        loss_axiom = model.compute_exactness_loss(mode='chain_axiom')
        loss_chain = ex_w * loss_exact + ax_w * loss_axiom
        # combine
        lambda_exact = 0.5
        loss = loss_task + lambda_exact * loss_chain
        print(f"  Before step:")
        print(f"    Loss: {loss.item():.4f}")
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"    Gradient norm: {total_norm:.4f}")
        optimizer.step()
        with torch.no_grad():
            logits_after = model(input_grid, grid_mask=mask)
            loss_after = F.cross_entropy(
                logits_after.permute(0, 3, 1, 2),
                output_grid,
                reduction='none'
            )
            loss_after = (loss_after * mask.float()).sum() / mask.sum()
        print(f"  After step:")
        print(f"    Loss: {loss_after.item():.4f}")
        print(f"    Change: {loss_after.item() - loss_task.item():.6f}")
        print("[PASS] Training step works!")
        return True

    except Exception as e:
        print(f"[FAIL] Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Evaluation."""
    print_header("Evaluation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = Path("data/arc/subset")
    if not data_path.exists():
        print("X Dataset not found. Skipping evaluation test.")
        return True  # >> NOTE: not a failure, just skip
    try:
        model = ARCChainSolver(
            max_grid_size=30,
            chain_dims=[16, 32, 64, 128, 256],
            epsilon=1e-3,
            dtype=torch.float64
        ).to(device)
        dataset = ARCTaskDataset(
            str(data_path),
            max_grid_size=30
        )
        print(f"  Loaded {len(dataset)} tasks")
        dataset.tasks = dataset.tasks[:2]
        evaluator = BaselineEvaluator(
            model,
            device=str(device)
        )
        print(f"  Running evaluation on {len(dataset)} tasks...")
        results = evaluator.evaluate(dataset, verbose=False)
        print(f"  Results:")
        print(f"    Accuracy: {results['overall_accuracy']:.2%}")
        print(f"    Correct: {results['num_correct']}/{results['num_tasks']}")
        print("[PASS] Evaluation works!")
        return True
    except Exception as e:
        print(f"[FAIL] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print_header("ARC-AGI Pipeline Smoke Test")
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Forward Pass", test_model_forward),
        ("Loss Computation", test_loss_computation),
        ("Training Step", test_training_step),
        ("Evaluation", test_evaluation)
    ]
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    print_header("Summary")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    for name, success in results:
        status = "[PASS] PASS" if success else "[FAIL] FAIL"
        print(f"  {status:8s} {name}")
    print(f"\n  Total: {passed}/{total} tests passed")
    if passed == total:
        print("\n[PASS] All smoke tests passed! Pipeline is ready.")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please fix errors above before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
