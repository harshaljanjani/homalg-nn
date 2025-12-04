import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from homalg_nn.analysis.representation_metrics import RepresentationMetrics


class FERDetector:
    """
    High-level interface for FER detection in trained models.
    >>> detector = FERDetector(model, dataloader, layer_names=['encoder', 'decoder'])
    >>> fer_score = detector.compute_fer_score()
    >>> print(f"FER Score: {fer_score['fer_score']:.3f}")
    >>> # Compare models
    >>> comparison = detector.compare_models(baseline_model, exact_model)
    >>> print(comparison)
    >>> # Generate report
    >>> detector.generate_report(save_path='fer_analysis.txt')
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer_names: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_cache: bool = True,
        sample_size: Optional[int] = 1000
    ):
        """
        Initialize FER detector.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.layer_names = layer_names
        self.device = device
        self.sample_size = sample_size
        self.metrics = RepresentationMetrics(
            use_cache=use_cache,
            sample_size=sample_size
        )
        self._cached_representations = {}
        self._cached_labels = None

    def extract_representations(
        self,
        layer_name: str,
        return_labels: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract representations from a specific layer using PyTorch hooks.
        >>> repr, labels = detector.extract_representations('encoder')
        >>> print(f"Extracted {repr.shape[0]} representations of dim {repr.shape[1]}")
        """
        # cache check!
        cache_key = f"{layer_name}_{return_labels}"
        if cache_key in self._cached_representations:
            cached = self._cached_representations[cache_key]
            if return_labels:
                return cached['representations'], cached['labels']
            else:
                return cached['representations'], None
        layer = self._get_layer_by_name(self.model, layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        # hook hook hook
        activations = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                act = output.detach()
            elif isinstance(output, tuple):
                act = output[0].detach()
            else:
                act = output
            if len(act.shape) > 2:
                act = act.view(act.size(0), -1)
            activations.append(act.cpu())

        # register hook + extract reprs
        handle = layer.register_forward_hook(hook_fn)
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(self.dataloader):
                if isinstance(batch, (tuple, list)):
                    data, labels = batch[0].to(self.device), batch[1]
                else:
                    data = batch.to(self.device)
                    labels = None
                _ = self.model(data)
                if labels is not None:
                    all_labels.append(labels)
                if self.sample_size is not None:
                    total_samples = sum(a.size(0) for a in activations)
                    if total_samples >= self.sample_size:
                        break
        handle.remove()
        representations = torch.cat(activations, dim=0)
        if self.sample_size is not None and representations.size(0) > self.sample_size:
            representations = representations[:self.sample_size]
        labels_tensor = None
        if len(all_labels) > 0:
            labels_tensor = torch.cat(all_labels, dim=0)
            if self.sample_size is not None and labels_tensor.size(0) > self.sample_size:
                labels_tensor = labels_tensor[:self.sample_size]
        self._cached_representations[cache_key] = {
            'representations': representations,
            'labels': labels_tensor
        }
        if return_labels:
            return representations, labels_tensor
        else:
            return representations, None

    def _get_layer_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get layer by dotted name path."""
        parts = name.split('.')
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                found = False
                for module_name, module in current.named_modules():
                    if module_name == part or module_name.endswith(f".{part}"):
                        current = module
                        found = True
                        break
                if not found:
                    return None
        return current

    def compute_fer_score(
        self,
        layer_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute FER score for a specific layer or all layers.
        """
        if layer_name is not None:
            representations, labels = self.extract_representations(layer_name)
            return self.metrics.compute_fer_score(representations, labels=labels)
        else:
            all_scores = {}
            for name in self.layer_names:
                try:
                    representations, labels = self.extract_representations(name)
                    score = self.metrics.compute_fer_score(representations, labels=labels)
                    all_scores[name] = score
                except Exception as e:
                    print(f"Warning: Could not compute FER for layer '{name}': {e}")
            if len(all_scores) == 0:
                raise ValueError("Could not compute FER for any layer")
            aggregated = {
                'fer_score': np.mean([s['fer_score'] for s in all_scores.values()]),
                'probe_accuracy': np.mean([s['probe_accuracy'] for s in all_scores.values() if s['probe_accuracy'] is not None]),
                'entanglement': np.mean([s['entanglement'] for s in all_scores.values()]),
                'rank_ratio': np.mean([s['rank_ratio'] for s in all_scores.values()])
            }
            return aggregated

    def compare_models(
        self,
        baseline_model: nn.Module,
        exact_model: nn.Module,
        n_runs: int = 10
    ) -> pd.DataFrame:
        """
        Compare FER metrics between baseline and exactness-constrained models.
        """
        baseline_detector = FERDetector(
            baseline_model, self.dataloader, self.layer_names,
            device=self.device, sample_size=self.sample_size
        )
        exact_detector = FERDetector(
            exact_model, self.dataloader, self.layer_names,
            device=self.device, sample_size=self.sample_size
        )
        # metrics for each layer
        results = []
        for layer_name in self.layer_names:
            try:
                baseline_repr, labels = baseline_detector.extract_representations(layer_name)
                exact_repr, _ = exact_detector.extract_representations(layer_name)
                baseline_scores = []
                exact_scores = []
                n_samples = baseline_repr.size(0)
                bootstrap_size = min(n_samples, 200)
                for _ in range(n_runs):
                    # random subsample
                    indices = torch.randperm(n_samples)[:bootstrap_size]
                    baseline_subsample = baseline_repr[indices]
                    exact_subsample = exact_repr[indices]
                    label_subsample = labels[indices] if labels is not None else None
                    # FER scores
                    baseline_score = self.metrics.compute_fer_score(
                        baseline_subsample, labels=label_subsample
                    )
                    exact_score = self.metrics.compute_fer_score(
                        exact_subsample, labels=label_subsample
                    )
                    baseline_scores.append(baseline_score)
                    exact_scores.append(exact_score)
                # aggregate metrics
                for metric_name in ['fer_score', 'probe_accuracy', 'entanglement', 'rank_ratio']:
                    baseline_values = [s[metric_name] for s in baseline_scores if s[metric_name] is not None]
                    exact_values = [s[metric_name] for s in exact_scores if s[metric_name] is not None]
                    if len(baseline_values) == 0 or len(exact_values) == 0:
                        continue
                    baseline_mean = np.mean(baseline_values)
                    baseline_std = np.std(baseline_values)
                    exact_mean = np.mean(exact_values)
                    exact_std = np.std(exact_values)
                    if metric_name in ['fer_score', 'entanglement']:
                        # lower == better
                        improvement = (baseline_mean - exact_mean) / baseline_mean * 100
                    else:
                        # ^^^ == better (probe_accuracy, rank_ratio)
                        improvement = (exact_mean - baseline_mean) / baseline_mean * 100
                    # t-test approximation for p-value
                    pooled_std = np.sqrt((baseline_std**2 + exact_std**2) / 2)
                    if pooled_std > 0:
                        t_stat = abs(exact_mean - baseline_mean) / (pooled_std / np.sqrt(n_runs))
                        # p-value approximation (two-tailed)
                        p_value = 2 * (1 - self._normal_cdf(t_stat))
                    else:
                        p_value = 1.0
                    significant = p_value < 0.05
                    results.append({
                        'layer': layer_name,
                        'metric': metric_name,
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'exact_mean': exact_mean,
                        'exact_std': exact_std,
                        'improvement': improvement,
                        'p_value': p_value,
                        'significant': significant
                    })
            except Exception as e:
                print(f"Warning: Could not compare layer '{layer_name}': {e}")
        return pd.DataFrame(results)

    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def plot_subspace_alignment(
        self,
        layer_name: str,
        training_checkpoints: Optional[List[Tuple[int, nn.Module]]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot how representations become exact over training (subspace alignment).
        >>> checkpoints = [(0, model_init), (1000, model_mid), (2000, model_final)]
        >>> detector.plot_subspace_alignment('chain', checkpoints, 'alignment.png')
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        if training_checkpoints is None:
            raise ValueError("Must provide training_checkpoints: list of (step, model) tuples")
        fig, ax = plt.subplots(figsize=(10, 6))
        all_angles = []
        all_steps = []
        for step, model in training_checkpoints:
            model.eval()
            chain_layer = self._get_layer_by_name(model, layer_name)
            if chain_layer is None or not hasattr(chain_layer, 'boundary_maps'):
                print(f"Warning: Layer '{layer_name}' not found or has no boundary_maps")
                continue
            # principal angles between consecutive kernels and images
            boundary_maps = [d.detach().cpu().numpy() for d in chain_layer.boundary_maps]
            for i in range(len(boundary_maps) - 1):
                d_i = boundary_maps[i]
                d_ip1 = boundary_maps[i + 1]
                # kernel of `d_i` and image of `d_{i+1}`
                try:
                    U_i, S_i, Vh_i = np.linalg.svd(d_i, full_matrices=False)
                    U_ip1, S_ip1, Vh_ip1 = np.linalg.svd(d_ip1, full_matrices=False)
                    eps = 1e-3
                    kernel_indices = S_i < eps
                    if np.sum(kernel_indices) == 0:
                        continue
                    ker_basis = Vh_i[kernel_indices, :].T
                    # `im (d_{i+1})`: left singular vectors with large singular values
                    image_indices = S_ip1 > eps
                    if np.sum(image_indices) == 0:
                        continue
                    im_basis = U_ip1[:, image_indices]
                    # QR factorization
                    Q_ker, _ = np.linalg.qr(ker_basis)
                    Q_im, _ = np.linalg.qr(im_basis)
                    # SVD (`Q_ker.T` @ `Q_im`)
                    _, S, _ = np.linalg.svd(Q_ker.T @ Q_im)
                    S = np.clip(S, -1, 1)
                    principal_angles = np.arccos(S)
                    for angle in principal_angles:
                        all_angles.append(angle)
                        all_steps.append(step)
                except np.linalg.LinAlgError:
                    continue

        # plot
        if len(all_angles) > 0:
            ax.scatter(all_steps, all_angles, alpha=0.5, s=20)
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Principal Angle (radians)', fontsize=12)
            ax.set_title(f'Subspace Alignment: ker(d_i) vs im(d_{{i+1}})\nLayer: {layer_name}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.axhline(np.pi/2, color='red', linestyle='--', alpha=0.5, label='π/2 (orthogonal)')
            ax.axhline(0, color='green', linestyle='--', alpha=0.5, label='0 (exact)')
            ax.legend()
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            if show:
                plt.show()
            else:
                plt.close()

    def generate_report(
        self,
        save_path: Optional[str] = None,
        # include_visualizations: bool = True
    ) -> str:
        """
        Generate comprehensive FER analysis report.
        """
        lines = []
        lines.append("\nFER DETECTION REPORT")
        lines.append("")
        # model info
        lines.append(f"Model: {self.model.__class__.__name__}")
        total_params = sum(p.numel() for p in self.model.parameters())
        lines.append(f"Total parameters: {total_params:,}")
        lines.append(f"Layers analyzed: {', '.join(self.layer_names)}")
        lines.append(f"Sample size: {self.sample_size}")
        lines.append("")
        # FER scores per layer
        lines.append("FER SCORES BY LAYER")
        for layer_name in self.layer_names:
            try:
                representations, labels = self.extract_representations(layer_name)
                fer_result = self.metrics.compute_fer_score(representations, labels=labels)
                lines.append(f"\n{layer_name}:")
                lines.append(f"  FER Score:        {fer_result['fer_score']:.4f}")
                if fer_result['probe_accuracy'] is not None:
                    lines.append(f"  Probe Accuracy:   {fer_result['probe_accuracy']:.4f}")
                lines.append(f"  Entanglement:     {fer_result['entanglement']:.4f}")
                lines.append(f"  Rank Ratio:       {fer_result['rank_ratio']:.4f}")
                if fer_result['fer_score'] < 0.3:
                    interpretation = "Good (low FER)"
                elif fer_result['fer_score'] < 0.6:
                    interpretation = "Moderate FER"
                else:
                    interpretation = "High FER (concerning)"
                lines.append(f"  Interpretation:   {interpretation}")
            except Exception as e:
                lines.append(f"\n{layer_name}: ERROR - {e}")

        # overall assessment
        lines.append("OVERALL ASSESSMENT")
        try:
            overall_fer = self.compute_fer_score()
            lines.append(f"\nAverage FER Score: {overall_fer['fer_score']:.4f}")
            if overall_fer['fer_score'] < 0.4:
                lines.append("Status: ✓ Representations appear well-factored")
                lines.append("Recommendation: Current architecture is good")
            elif overall_fer['fer_score'] < 0.7:
                lines.append("Status: ⚠ Moderate FER detected")
                lines.append("Recommendation: Consider exactness constraints or regularization")
            else:
                lines.append("Status: ✗ High FER detected")
                lines.append("Recommendation: Strongly consider exactness constraints")
        except Exception as e:
            lines.append(f"\nERROR computing overall FER: {e}")
        lines.append("\n")
        report_text = "\n".join(lines)
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        return report_text

    def clear_cache(self):
        self._cached_representations = {}
        self._cached_labels = None
        self.metrics.clear_cache()
