import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from examples.arc import (
    ARCChainSolver,
    ARCTaskDataset,
    ARCEvaluator,
    BaselineEvaluator
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test-time adaptation experiments')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data-path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--max-grid-size', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='results/adaptation_experiments')
    parser.add_argument('--num-tasks', type=int, default=None, help='Limit number of tasks (default: all)')
    return parser.parse_args()


class AdaptationExperiments:
    """Run test-time adaptation experiments."""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = self.save_dir / f'adaptation_{timestamp}'
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=self.device, weights_only=False)
        model_args = checkpoint.get('args', {})
        self.model = ARCChainSolver(
            max_grid_size=model_args.get('max_grid_size', 30),
            chain_dims=model_args.get('chain_dims', [16, 32, 64, 128, 256]),
            embed_dim=model_args.get('embed_dim', 256),
            hidden_dim=model_args.get('hidden_dim', 512)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
        self.dataset = ARCTaskDataset(
            args.data_path,
            max_grid_size=args.max_grid_size
        )
        if args.num_tasks and len(self.dataset) > args.num_tasks:
            self.dataset.tasks = self.dataset.tasks[:args.num_tasks]

        print(f"Evaluation dataset: {len(self.dataset)} tasks")
        self.experiments = self._define_experiments()
        print(f"\nRunning {len(self.experiments)} experiments")

    def _define_experiments(self):
        """Define adaptation experiments."""
        experiments = []
        # 1. no adaptation (baseline)
        experiments.append({
            'name': 'no_adaptation',
            'adaptation_steps': 0,
            'use_exactness': False,
            'description': 'No adaptation (baseline)'
        })
        # 2. task-only adaptation (different steps)
        for steps in [10, 25, 50, 100]:
            experiments.append({
                'name': f'task_only_{steps}steps',
                'adaptation_steps': steps,
                'use_exactness': False,
                'description': f'Task-only ({steps} steps)'
            })
        # 3. task + exactness adaptation (different steps)
        for steps in [10, 25, 50, 100]:
            experiments.append({
                'name': f'task_exact_{steps}steps',
                'adaptation_steps': steps,
                'use_exactness': True,
                'lambda_exact': 0.5,
                'description': f'Task+Exactness ({steps} steps)'
            })
        # 4. extended adaptation (200 steps)
        experiments.append({
            'name': 'extended_adaptation',
            'adaptation_steps': 200,
            'use_exactness': True,
            'lambda_exact': 0.5,
            'description': 'Extended adaptation (200 steps)'
        })
        return experiments

    def run_experiment(self, config):
        """Run single adaptation experiment."""
        print(f"\nRunning: {config['description']}")
        if config['adaptation_steps'] == 0:
            evaluator = BaselineEvaluator(
                self.model,
                device=self.args.device
            )
        else:
            evaluator = ARCEvaluator(
                self.model,
                device=self.args.device,
                num_attempts=3,
                adaptation_steps=config['adaptation_steps'],
                adaptation_lr=1e-4,
                use_exactness_loss=config['use_exactness'],
                lambda_exact=config.get('lambda_exact', 0.5)
            )
        results = evaluator.evaluate(self.dataset, verbose=True)
        return {
            'config': config,
            'accuracy': results['overall_accuracy'],
            'num_correct': results['num_correct'],
            'num_tasks': results['num_tasks']
        }

    def run_all(self):
        """Run all experiments."""
        print("\nTest-Time Adaptation Experiments")
        all_results = []
        for config in self.experiments:
            result = self.run_experiment(config)
            all_results.append(result)
            print(f"  Result: {result['accuracy']:.2%}")
        self._analyze_and_save(all_results)

    def _analyze_and_save(self, all_results):
        """Analyze and save experiment results."""
        print("\nRESULTS SUMMARY")
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        for i, r in enumerate(sorted_results):
            print(f"{i+1}. {r['config']['description']:40s} {r['accuracy']:6.2%}")
        best = sorted_results[0]
        baseline = next(r for r in all_results if r['config']['name'] == 'no_adaptation')
        improvement = best['accuracy'] - baseline['accuracy']
        print(f"Best: {best['config']['description']}")
        print(f"  Accuracy: {best['accuracy']:.2%}")
        print(f"  Improvement over baseline: {improvement:.2%} ({improvement/baseline['accuracy']*100:.1f}%)")
        # task-only vs task + exactness
        print("\nTask-only vs Task+Exactness:")
        for steps in [10, 25, 50, 100]:
            task_only = next((r for r in all_results if r['config']['name'] == f'task_only_{steps}steps'), None)
            task_exact = next((r for r in all_results if r['config']['name'] == f'task_exact_{steps}steps'), None)
            if task_only and task_exact:
                diff = task_exact['accuracy'] - task_only['accuracy']
                print(f"  {steps:3d} steps: Task-only={task_only['accuracy']:.2%}, "
                      f"Task+Exact={task_exact['accuracy']:.2%}, "
                      f"Diff={diff:+.2%}")
        self._save_results(all_results, best, baseline)

    def _save_results(self, all_results, best, baseline):
        """Save experiment results."""
        results_path = self.exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'all_results': all_results,
                'best': best,
                'baseline': baseline,
                'improvement': best['accuracy'] - baseline['accuracy']
            }, f, indent=2)
        summary_path = self.exp_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Test-Time Adaptation Experiments\n")
            f.write("All Results (sorted by accuracy):\n")
            sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
            for i, r in enumerate(sorted_results):
                f.write(f"\n{i+1}. {r['config']['description']}\n")
                f.write(f"   Accuracy: {r['accuracy']:.2%}\n")
                f.write(f"   Steps: {r['config']['adaptation_steps']}\n")
                f.write(f"   Exactness: {r['config']['use_exactness']}\n")
            f.write(f"Best: {best['config']['description']}\n")
            f.write(f"Accuracy: {best['accuracy']:.2%}\n")
            f.write(f"Improvement over baseline: {(best['accuracy'] - baseline['accuracy']):.2%}\n")
        print(f"\nResults saved to: {self.exp_dir}")


def main():
    args = parse_args()
    experiments = AdaptationExperiments(args)
    experiments.run_all()


if __name__ == '__main__':
    main()
