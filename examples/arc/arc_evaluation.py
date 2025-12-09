import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import copy
from tqdm import tqdm
from .arc_data_loader import ARCTaskDataset


class ARCEvaluator:
    """
    Evaluate ARC solver on tasks.
    >>> evaluator = ARCEvaluator(model, device='cuda')
    >>> accuracy = evaluator.evaluate(eval_dataset)
    >>> print(f"Accuracy: {accuracy:.2%}")
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        num_attempts: int = 3,
        adaptation_steps: int = 50,
        adaptation_lr: float = 1e-4,
        use_exactness_loss: bool = True,
        lambda_exact: float = 0.5
    ):
        self.model = model
        self.device = device
        self.num_attempts = num_attempts
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        self.use_exactness_loss = use_exactness_loss
        self.lambda_exact = lambda_exact

    def adapt_to_task(
        self,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
        train_masks: torch.Tensor
    ) -> nn.Module:
        """
        Fine-tune model on task demonstration examples.
        """
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        adapted_model = adapted_model.to(self.device)
        train_inputs = train_inputs.to(self.device)
        train_outputs = train_outputs.to(self.device)
        train_masks = train_masks.to(self.device)
        # optim
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.adaptation_lr)
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad()
            logits = adapted_model(train_inputs, grid_mask=train_masks)
            # task loss (cross-entropy)
            # logits: `(batch, H, W, 10)`, train_outputs: `(batch, H, W)`
            loss_task = F.cross_entropy(
                logits.permute(0, 3, 1, 2),  # `(batch, 10, H, W)`
                train_outputs,
                reduction='none'
            )
            loss_task = (loss_task * train_masks.float()).sum() / train_masks.sum()
            if self.use_exactness_loss and hasattr(adapted_model, 'compute_exactness_loss'):
                try:
                    loss_exact = adapted_model.compute_exactness_loss(mode='exactness')
                    loss_axiom = adapted_model.compute_exactness_loss(mode='chain_axiom')
                    loss_chain = loss_exact + loss_axiom
                    loss = loss_task + self.lambda_exact * loss_chain
                except:
                    loss = loss_task
            else:
                loss = loss_task
            loss.backward()
            optimizer.step()
        adapted_model.eval()
        return adapted_model

    def generate_predictions(
        self,
        model: nn.Module,
        test_input: torch.Tensor,
        test_mask: torch.Tensor,
        num_attempts: int
    ) -> List[torch.Tensor]:
        """
        Generate multiple prediction attempts.
        """
        predictions = []
        model.eval()
        with torch.no_grad():
            test_input_batch = test_input.unsqueeze(0).to(self.device)
            test_mask_batch = test_mask.unsqueeze(0).to(self.device)
            for _ in range(num_attempts):
                logits = model(test_input_batch, grid_mask=test_mask_batch)
                pred = logits.argmax(dim=-1).squeeze(0)  # `(H, W)`
                pred = pred * test_mask.to(pred.device)
                predictions.append(pred.cpu())
        return predictions

    def evaluate_task(
        self,
        task_data: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """
        Evaluate model on a single task.
        """
        train_inputs = task_data['train_inputs']
        train_outputs = task_data['train_outputs']
        train_masks = task_data['train_masks']
        test_inputs = task_data['test_inputs']
        test_outputs = task_data['test_outputs']
        test_masks = task_data['test_masks']
        adapted_model = self.adapt_to_task(train_inputs, train_outputs, train_masks)
        results = []
        for i in range(len(test_inputs)):
            test_input = test_inputs[i]
            test_output = test_outputs[i]
            test_mask = test_masks[i]
            predictions = self.generate_predictions(
                adapted_model,
                test_input,
                test_mask,
                self.num_attempts
            )
            correct = False
            for pred in predictions:
                valid_match = (pred[test_mask] == test_output[test_mask].cpu()).all()
                if valid_match:
                    correct = True
                    break
            results.append({
                'test_idx': i,
                'correct': correct,
                'predictions': predictions,
                'ground_truth': test_output
            })
        return {
            'task_id': task_data['task_id'],
            'results': results,
            'accuracy': sum(r['correct'] for r in results) / len(results)
        }

    def evaluate(
        self,
        dataset: ARCTaskDataset,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Evaluate on entire dataset.
        """
        task_results = []
        num_correct = 0
        num_total = 0
        iterator = tqdm(dataset, desc="Evaluating") if verbose else dataset
        for task_data in iterator:
            result = self.evaluate_task(task_data)
            task_correct = any(r['correct'] for r in result['results'])
            if task_correct:
                num_correct += 1
            num_total += 1
            task_results.append(result)
            if verbose:
                iterator.set_postfix({'accuracy': f"{num_correct/num_total:.2%}"})
        overall_accuracy = num_correct / num_total if num_total > 0 else 0.0
        return {
            'overall_accuracy': overall_accuracy,
            'per_task_results': task_results,
            'num_tasks': num_total,
            'num_correct': num_correct
        }


class BaselineEvaluator(ARCEvaluator):
    """
    Evaluator for baseline model (no test-time adaptation).
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        num_attempts: int = 3
    ):
        super().__init__(
            model=model,
            device=device,
            num_attempts=num_attempts,
            adaptation_steps=0  # no adaptation
        )

    def adapt_to_task(self, train_inputs, train_outputs, train_masks):
        """No adaptation -- just return original model."""
        return self.model


def compare_models(
    model_chain: nn.Module,
    model_baseline: nn.Module,
    dataset: ARCTaskDataset,
    device: str = 'cpu',
    adaptation_steps: int = 50,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare chain model vs baseline on dataset.
    """
    print("Evaluating chain model.")
    evaluator_chain = ARCEvaluator(
        model_chain,
        device=device,
        adaptation_steps=adaptation_steps
    )
    results_chain = evaluator_chain.evaluate(dataset, verbose=verbose)
    print("\nEvaluating baseline model.")
    evaluator_baseline = ARCEvaluator(
        model_baseline,
        device=device,
        adaptation_steps=adaptation_steps
    )
    results_baseline = evaluator_baseline.evaluate(dataset, verbose=verbose)
    improvement = results_chain['overall_accuracy'] - results_baseline['overall_accuracy']
    comparison = {
        'chain_accuracy': results_chain['overall_accuracy'],
        'baseline_accuracy': results_baseline['overall_accuracy'],
        'improvement': improvement,
        'improvement_percent': improvement / results_baseline['overall_accuracy'] * 100 if results_baseline['overall_accuracy'] > 0 else 0,
        'chain_results': results_chain,
        'baseline_results': results_baseline
    }
    if verbose:
        print("\nCOMPARISON RESULTS")
        print(f"Chain model:    {results_chain['overall_accuracy']:.2%}")
        print(f"Baseline model: {results_baseline['overall_accuracy']:.2%}")
        print(f"Improvement:    {improvement:.2%} ({comparison['improvement_percent']:.1f}%)")
    return comparison


def analyze_errors(
    results: Dict[str, any],
    dataset: ARCTaskDataset
) -> Dict[str, any]:
    """
    Analyze prediction errors to identify failure modes.
    """
    error_analysis = {
        'total_tasks': results['num_tasks'],
        'correct_tasks': results['num_correct'],
        'failed_tasks': results['num_tasks'] - results['num_correct'],
        'failed_task_ids': []
    }
    for task_result in results['per_task_results']:
        task_correct = any(r['correct'] for r in task_result['results'])
        if not task_correct:
            error_analysis['failed_task_ids'].append(task_result['task_id'])
    return error_analysis
