import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from examples.arc import (
    ARCChainSolver,
    ARCTaskDataset
)


def parse_args():
    parser = argparse.ArgumentParser(description='Lightweight adaptation experiments')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data-path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--num-tasks', type=int, default=10, help='Number of tasks to evaluate (default: 10)')
    parser.add_argument('--adaptation-steps', type=int, default=50, help='Adaptation steps (default: 50)')
    parser.add_argument('--use-exactness', action='store_true', help='Use exactness loss during adaptation')
    parser.add_argument('--lambda-exact', type=float, default=0.3, help='Exactness weight (default: 0.3)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='results/adaptation_lightweight')
    return parser.parse_args()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pad_grid(grid, max_size=30):
    """Pad grid to `max_size x max_size`."""
    h, w = grid.shape
    if h > max_size or w > max_size:
        raise ValueError(f"Grid size ({h}, {w}) exceeds max_size ({max_size})")
    padded = torch.zeros((max_size, max_size), dtype=torch.long)
    padded[:h, :w] = torch.from_numpy(grid) if isinstance(grid, torch.Tensor) == False else grid
    return padded


def load_model(model_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get('args', {})
    model_config = {
        'max_grid_size': model_args.get('max_grid_size', 30),
        'chain_dims': model_args.get('chain_dims', [16, 32, 64, 128, 256]),
        'embed_dim': model_args.get('embed_dim', 256),
        'hidden_dim': model_args.get('hidden_dim', 512)
    }
    model = ARCChainSolver(
        max_grid_size=model_config['max_grid_size'],
        chain_dims=model_config['chain_dims'],
        embed_dim=model_config['embed_dim'],
        hidden_dim=model_config['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    del checkpoint
    clear_memory()
    return model, model_config


def adapt_and_evaluate_task(model, task, adaptation_steps, use_exactness, lambda_exact, device, model_config):
    """
    Adapt model to task demonstrations and evaluate on test inputs.
    """
    adapted_model = ARCChainSolver(
        max_grid_size=model_config['max_grid_size'],
        chain_dims=model_config['chain_dims'],
        embed_dim=model_config['embed_dim'],
        hidden_dim=model_config['hidden_dim']
    ).to(device)
    adapted_model.load_state_dict(model.state_dict())
    adapted_model.train()
    optimizer = optim.Adam(adapted_model.parameters(), lr=1e-4)
    demonstrations = task.train_pairs  # `List[(input_grid, output_grid)]`
    if adaptation_steps > 0 and len(demonstrations) > 0:
        for _ in range(adaptation_steps):
            total_loss = 0.0
            for input_grid_np, output_grid_np in demonstrations:
                # pad grids to max_size
                input_grid = pad_grid(input_grid_np, max_size=30).unsqueeze(0).to(device)
                target_grid = pad_grid(output_grid_np, max_size=30).unsqueeze(0).to(device)
                output = adapted_model(input_grid)
                task_loss = nn.CrossEntropyLoss()(
                    output.view(-1, 10),
                    target_grid.view(-1)
                )
                if use_exactness:
                    exact_loss = adapted_model.compute_exactness_loss(mode='exactness')
                    axiom_loss = adapted_model.compute_exactness_loss(mode='chain_axiom')
                    chain_loss = exact_loss + axiom_loss
                    loss = task_loss + lambda_exact * chain_loss
                else:
                    loss = task_loss
                total_loss += loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    adapted_model.eval()
    test_examples = task.test_pairs  # `List[(input_grid, output_grid)]`
    num_correct = 0
    num_total = len(test_examples)
    with torch.no_grad():
        for input_grid_np, output_grid_np in test_examples:
            input_grid = pad_grid(input_grid_np, max_size=30).unsqueeze(0).to(device)
            output = adapted_model(input_grid)
            pred_grid = output.argmax(dim=1).squeeze(0).cpu()
            h, w = output_grid_np.shape
            pred_actual = pred_grid[:h, :w]
            target_actual = torch.from_numpy(output_grid_np)
            if torch.equal(pred_actual, target_actual):
                num_correct += 1
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    del adapted_model, optimizer
    clear_memory()
    return {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_total': num_total
    }


def run_experiment(args):
    """Run single adaptation experiment."""
    device = torch.device(args.device)
    print("\nLightweight Test-Time Adaptation Experiment")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Tasks: {args.num_tasks}")
    print(f"Adaptation steps: {args.adaptation_steps}")
    print(f"Use exactness: {args.use_exactness}")
    if args.use_exactness:
        print(f"Lambda: {args.lambda_exact}")
    print("Loading model...")
    model, model_config = load_model(args.model_path, device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"\nLoading dataset from {args.data_path}...")
    dataset = ARCTaskDataset(args.data_path, max_grid_size=30)
    if args.num_tasks and len(dataset) > args.num_tasks:
        dataset.tasks = dataset.tasks[:args.num_tasks]
    print(f"Dataset loaded: {len(dataset)} tasks\n")
    results = []
    total_correct = 0
    total_tests = 0
    print("Evaluating tasks:")
    for _, task in enumerate(tqdm(dataset.tasks, desc="Tasks")):
        result = adapt_and_evaluate_task(
            model=model,
            task=task,
            adaptation_steps=args.adaptation_steps,
            use_exactness=args.use_exactness,
            lambda_exact=args.lambda_exact,
            device=device,
            model_config=model_config
        )
        results.append({
            'task_id': task.task_id,
            'accuracy': result['accuracy'],
            'num_correct': result['num_correct'],
            'num_total': result['num_total']
        })
        total_correct += result['num_correct']
        total_tests += result['num_total']
    overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
    print("\nRESULTS")
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_tests})")
    print(f"Tasks Solved: {sum(1 for r in results if r['accuracy'] == 1.0)}/{len(results)}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"adapt_{args.adaptation_steps}steps"
    if args.use_exactness:
        exp_name += f"_exact{args.lambda_exact}"
    exp_name += f"_{timestamp}"
    results_path = save_dir / f'{exp_name}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'model_path': args.model_path,
                'data_path': args.data_path,
                'num_tasks': args.num_tasks,
                'adaptation_steps': args.adaptation_steps,
                'use_exactness': args.use_exactness,
                'lambda_exact': args.lambda_exact
            },
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_tests': total_tests,
            'tasks_solved': sum(1 for r in results if r['accuracy'] == 1.0),
            'per_task_results': results
        }, f, indent=2)
    print(f"Results saved to: {results_path}\n")
    return overall_accuracy


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
