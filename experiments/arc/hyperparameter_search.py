# >> lambda_exact: [0.1, 0.3, 0.5, 0.7, 1.0]
# >> chain_dims: Multiple symmetric and asymmetric configurations
# >> lr: [1e-4, 5e-4, 1e-3]
# >> num_adaptation_steps: [10, 50, 100]
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
import itertools
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from examples.arc import (
    ARCChainSolver,
    ARCTaskDataset,
    ARCEvaluator
)


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter search for ARC solver')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ARC data (subset for search)')
    parser.add_argument('--max-grid-size', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='results/hyperparam_search')
    parser.add_argument('--num-tasks', type=int, default=20, help='Number of tasks to evaluate on')
    parser.add_argument('--quick', action='store_true', help='Quick search with reduced search space')
    return parser.parse_args()


class HyperparameterSearch:
    """Grid search over hyperparameters."""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = self.save_dir / f'search_{timestamp}'
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = ARCTaskDataset(
            args.data_path,
            max_grid_size=args.max_grid_size
        )
        # ffor now, let's just place a limit for speeding things up.
        if len(self.dataset) > args.num_tasks:
            indices = list(range(min(args.num_tasks, len(self.dataset))))
            self.dataset.tasks = [self.dataset.tasks[i] for i in indices]
        print(f"Hyperparameter search on {len(self.dataset)} tasks")
        if args.quick:
            self.search_space = {
                'lambda_exact': [0.3, 0.5],
                'chain_dims': [
                    [16, 32, 64, 128, 256],  # asymmetric
                    [16, 32, 64, 32, 16]     # symmetric
                ],
                'lr': [5e-4, 1e-3],
                'adaptation_steps': [25, 50]
            }
        else:
            self.search_space = {
                'lambda_exact': [0.1, 0.3, 0.5, 0.7, 1.0],
                'chain_dims': [
                    [16, 32, 64, 128, 256],       # asymmetric (baseline)
                    [8, 16, 32, 64, 128],         # smaller
                    [32, 64, 128, 256, 512],      # larger
                    [16, 32, 64, 32, 16]          # fully symmetric
                ],
                'lr': [1e-4, 5e-4, 1e-3],
                'adaptation_steps': [10, 50, 100]
            }
        print(f"\nSearch space:")
        for key, values in self.search_space.items():
            print(f"  {key}: {values}")
        self.total_configs = 1
        for values in self.search_space.values():
            self.total_configs *= len(values)
        print(f"\nTotal configurations: {self.total_configs}")

    def evaluate_config(self, config):
        """Evaluate a single configuration."""
        model = ARCChainSolver(
            max_grid_size=self.args.max_grid_size,
            chain_dims=config['chain_dims'],
            epsilon=1e-3,
            dtype=torch.float64
        )
        model = model.to(self.device)
        evaluator = ARCEvaluator(
            model,
            device=self.args.device,
            num_attempts=3,
            adaptation_steps=config['adaptation_steps'],
            adaptation_lr=config['lr'],
            use_exactness_loss=True,
            lambda_exact=config['lambda_exact']
        )
        results = evaluator.evaluate(self.dataset, verbose=False)
        return {
            'accuracy': results['overall_accuracy'],
            'num_correct': results['num_correct'],
            'num_tasks': results['num_tasks']
        }

    def search(self):
        """Run grid search."""
        print("\nStarting hyperparameter search")
        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        all_results = []
        pbar = tqdm(
            itertools.product(*values),
            total=self.total_configs,
            desc="Grid search"
        )
        for config_values in pbar:
            config = dict(zip(keys, config_values))
            try:
                result = self.evaluate_config(config)
                config_result = {
                    'config': config,
                    'result': result
                }
                all_results.append(config_result)
                pbar.set_postfix({
                    'acc': f"{result['accuracy']:.2%}",
                    'best': f"{max(r['result']['accuracy'] for r in all_results):.2%}"
                })
            except Exception as e:
                print(f"\nError evaluating config {config}: {e}")
                continue
        best_result = max(all_results, key=lambda x: x['result']['accuracy'])
        print("\nSearch complete!")
        print("Best configuration:")
        print(json.dumps(best_result['config'], indent=2))
        print(f"\nBest accuracy: {best_result['result']['accuracy']:.2%}")
        self._save_results(all_results, best_result)
        return best_result

    def _save_results(self, all_results, best_result):
        """Save search results."""
        all_results_path = self.exp_dir / 'all_results.json'
        with open(all_results_path, 'w') as f:
            serializable_results = []
            for r in all_results:
                serializable_results.append({
                    'config': r['config'],
                    'result': r['result']
                })
            json.dump(serializable_results, f, indent=2)
        best_config_path = self.exp_dir / 'best_config.json'
        with open(best_config_path, 'w') as f:
            json.dump({
                'config': best_result['config'],
                'result': best_result['result']
            }, f, indent=2)
        summary_path = self.exp_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Hyperparameter Search Results\n")
            sorted_results = sorted(all_results, key=lambda x: x['result']['accuracy'], reverse=True)
            f.write("Top 10 configurations:\n")
            for i, r in enumerate(sorted_results[:10]):
                f.write(f"\n{i+1}. Accuracy: {r['result']['accuracy']:.2%}\n")
                f.write(f"   lambda_exact: {r['config']['lambda_exact']}\n")
                f.write(f"   chain_dims: {r['config']['chain_dims']}\n")
                f.write(f"   lr: {r['config']['lr']}\n")
                f.write(f"   adaptation_steps: {r['config']['adaptation_steps']}\n")
        print(f"\nResults saved to: {self.exp_dir}")


def main():
    args = parse_args()
    searcher = HyperparameterSearch(args)
    searcher.search()


if __name__ == '__main__':
    main()
