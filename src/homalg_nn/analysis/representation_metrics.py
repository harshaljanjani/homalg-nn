import numpy as np
import torch
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings


class RepresentationMetrics:
    """
    Quantify representation quality to detect FER.
    >>> metrics = RepresentationMetrics()
    >>> representations = model.encode(data)  # (N, D)
    >>> labels = data.labels  # (N,)
    >>>
    >>> # Test factorization quality
    >>> probe_result = metrics.linear_probe_accuracy(representations, labels)
    >>> print(f"Linear probe accuracy: {probe_result['mean_accuracy']:.3f}")
    >>>
    >>> # Test entanglement
    >>> mi_matrix = metrics.mutual_information_matrix(representations)
    >>> print(f"Mean off-diagonal MI: {mi_matrix.mean():.3f}")
    """
    def __init__(
        self,
        use_cache: bool = True,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize representation metrics calculator.
        """
        self.use_cache = use_cache
        self.sample_size = sample_size
        self.random_state = random_state
        self._cache = {}

    def linear_probe_accuracy(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
        n_splits: int = 5,
        max_iter: int = 1000
    ) -> Dict[str, float]:
        """
        Train linear classifier on frozen representations via cross-validation.
        """
        if isinstance(representations, torch.Tensor):
            X = representations.detach().cpu().numpy()
        else:
            X = np.array(representations)
        if isinstance(labels, torch.Tensor):
            y = labels.detach().cpu().numpy()
        else:
            y = np.array(labels)
        # standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # k-fold CV
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_accuracies = []
        probe_weights = []
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # train linear probe
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probe = LogisticRegression(
                    max_iter=max_iter,
                    random_state=self.random_state,
                    solver='lbfgs',
                    multi_class='multinomial'
                )
                probe.fit(X_train, y_train)
            # eval
            acc = probe.score(X_test, y_test)
            fold_accuracies.append(acc)
            if hasattr(probe, 'coef_'):
                probe_weights.append(np.linalg.norm(probe.coef_))
        return {
            'mean_accuracy': float(np.mean(fold_accuracies)),
            'std_accuracy': float(np.std(fold_accuracies)),
            'probe_complexity': float(np.mean(probe_weights)) if probe_weights else 0.0,
            'per_fold_accuracy': fold_accuracies
        }

    def mutual_information_matrix(
        self,
        representations: torch.Tensor,
        n_bins: int = 20,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute mutual information between all pairs of representation dimensions.
        """
        if isinstance(representations, torch.Tensor):
            Z = representations.detach().cpu().numpy()
        else:
            Z = np.array(representations)
        if self.sample_size is not None and Z.shape[0] > self.sample_size:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(Z.shape[0], self.sample_size, replace=False)
            Z = Z[indices]
        N, D = Z.shape
        MI = np.zeros((D, D))
        for i in range(D):
            for j in range(i, D):  # symmetric matrix
                mi = self._compute_mi_histogram(Z[:, i], Z[:, j], n_bins, normalize)
                MI[i, j] = mi
                MI[j, i] = mi
        return MI

    def _compute_mi_histogram(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int,
        normalize: bool
    ) -> float:
        """Estimate MI via histogram method."""
        # bins
        x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), n_bins))
        y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), n_bins))
        # joint + marginal histograms
        joint_hist, _, _ = np.histogram2d(x_binned, y_binned, bins=n_bins)
        joint_hist = joint_hist / joint_hist.sum()
        marginal_x = joint_hist.sum(axis=1)
        marginal_y = joint_hist.sum(axis=0)
        # Σ p(x,y) * log(p(x,y) / (p(x)p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_hist[i, j] > 0:
                    mi += joint_hist[i, j] * np.log2(
                        joint_hist[i, j] / (marginal_x[i] * marginal_y[j] + 1e-10)
                    )
        if normalize:
            joint_entropy = -np.sum(joint_hist * np.log2(joint_hist + 1e-10))
            mi = mi / (joint_entropy + 1e-10)
        return float(mi)

    def subspace_orthogonality(
        self,
        repr_task_a: torch.Tensor,
        repr_task_b: torch.Tensor,
        method: str = 'cca'
    ) -> float:
        """
        Measure alignment between task-specific representations.
        """
        if isinstance(repr_task_a, torch.Tensor):
            A = repr_task_a.detach().cpu().numpy()
        else:
            A = np.array(repr_task_a)
        if isinstance(repr_task_b, torch.Tensor):
            B = repr_task_b.detach().cpu().numpy()
        else:
            B = np.array(repr_task_b)
        if self.sample_size is not None:
            min_size = min(A.shape[0], B.shape[0])
            if min_size > self.sample_size:
                rng = np.random.RandomState(self.random_state)
                indices = rng.choice(min_size, self.sample_size, replace=False)
                A = A[indices]
                B = B[indices]
        A_centered = A - A.mean(axis=0, keepdims=True)
        B_centered = B - B.mean(axis=0, keepdims=True)
        if method == 'cca':
            # canonical correlation analysis via SVD
            # CCA finds maximally correlated projections
            cov_AB = A_centered.T @ B_centered / (A_centered.shape[0] - 1)
            cov_AA = A_centered.T @ A_centered / (A_centered.shape[0] - 1)
            cov_BB = B_centered.T @ B_centered / (B_centered.shape[0] - 1)
            eps = 1e-6
            cov_AA += eps * np.eye(cov_AA.shape[0])
            cov_BB += eps * np.eye(cov_BB.shape[0])
            # generalized eigenvalue problem
            # solve: `cov_AA^{-1/2}` @ `cov_AB` @ `cov_BB^{-1}` @ `cov_AB.T` @ `cov_AA^{-1/2}`
            # https://en.wikipedia.org/wiki/Canonical_correlation
            # https://www.cs.cmu.edu/~tom/10701_sp11/slides/CCA_tutorial.pdf
            try:
                L_AA = np.linalg.cholesky(cov_AA)
                L_BB = np.linalg.cholesky(cov_BB)
                inv_L_AA = np.linalg.inv(L_AA)
                inv_L_BB = np.linalg.inv(L_BB)
                M = inv_L_AA.T @ cov_AB @ inv_L_BB @ inv_L_BB.T @ cov_AB.T @ inv_L_AA
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.clip(eigvals, 0, 1)
                canonical_corrs = np.sqrt(eigvals)
                alignment_score = float(np.mean(canonical_corrs))
            except np.linalg.LinAlgError:
                # if cholesky fails, fall back to simpler metric
                alignment_score = float(np.abs(np.corrcoef(
                    A_centered.flatten(), B_centered.flatten()
                )[0, 1]))
        elif method == 'principal_angles':
            # QR factorization to get orthonormal bases
            Q_A, _ = np.linalg.qr(A_centered)
            Q_B, _ = np.linalg.qr(B_centered)
            # SVD of `Q_A.T @ Q_B` gives principal angles
            _, S, _ = np.linalg.svd(Q_A.T @ Q_B)
            S = np.clip(S, -1, 1)
            # princ angles in `[0, π/2]`
            principal_angles = np.arccos(S)
            alignment_score = float(np.mean(np.cos(principal_angles)))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'cca' or 'principal_angles'.")
        return alignment_score

    def representation_rank(
        self,
        representations: torch.Tensor,
        epsilon: float = 1e-3
    ) -> Dict[str, float]:
        """
        Compute effective dimensionality via SVD.
        """
        if isinstance(representations, torch.Tensor):
            Z = representations.detach().cpu().numpy()
        else:
            Z = np.array(representations)
        Z_centered = Z - Z.mean(axis=0, keepdims=True)
        Z_torch = torch.from_numpy(Z_centered).float()
        try:
            U, S, Vh = torch.linalg.svd(Z_torch, full_matrices=False)
            S = S.numpy()
        except:
            _, S, _ = np.linalg.svd(Z_centered, full_matrices=False)
        effective_rank = int(np.sum(S > epsilon))
        rank_ratio = effective_rank / len(S)
        # https://en.wikipedia.org/wiki/Entropy_%28information_theory%29
        S_normalized = S / (S.sum() + 1e-10)
        entropy = -np.sum(S_normalized * np.log2(S_normalized + 1e-10))
        # inverse participation ratio
        participation_ratio = (S.sum() ** 2) / (np.sum(S ** 2) + 1e-10)
        return {
            'effective_rank': effective_rank,
            'rank_ratio': float(rank_ratio),
            'entropy': float(entropy),
            'participation_ratio': float(participation_ratio),
            'singular_values': S.tolist()
        }

    def measure_redundancy(
        self,
        repr_task_a: torch.Tensor,
        repr_task_b: torch.Tensor,
        method: str = 'cca'
    ) -> float:
        """
        Measure if two tasks redundantly learned the same features (fracture).
        >>> Redundancy score in `[0, 1]`:
        >>> High (> 0.7) = tasks learned similar features (potential fracture)
        >>> Low ( < 0.3) = tasks learned distinct features (good separation)
        """
        return self.subspace_orthogonality(repr_task_a, repr_task_b, method=method)

    def compute_fer_score(
        self,
        representations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        task_representations: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Aggregate all metrics into a single FER score in `[0, 1]`.
        `FER = (1 - probe_acc) * entanglement_norm * (1 - rank_ratio)`
        """
        result = {}
        # 1. linear probe accuracy (if labels provided)
        if labels is not None:
            probe_result = self.linear_probe_accuracy(representations, labels)
            probe_acc = probe_result['mean_accuracy']
            result['probe_accuracy'] = probe_acc
        else:
            probe_acc = 0.5
            result['probe_accuracy'] = None
        # 2. entanglement via MI
        mi_matrix = self.mutual_information_matrix(representations, normalize=True)
        D = mi_matrix.shape[0]
        mask = ~np.eye(D, dtype=bool)
        off_diagonal_mi = mi_matrix[mask]
        entanglement = float(np.mean(off_diagonal_mi))
        result['entanglement'] = entanglement
        entanglement_norm = min(entanglement, 1.0)
        # 3. rank ratio
        rank_result = self.representation_rank(representations)
        rank_ratio = rank_result['rank_ratio']
        result['rank_ratio'] = rank_ratio
        # 4. redundancy (if multi-task)
        if task_representations is not None and len(task_representations) >= 2:
            task_names = list(task_representations.keys())
            redundancies = []
            for i in range(len(task_names)):
                for j in range(i + 1, len(task_names)):
                    redundancy = self.measure_redundancy(
                        task_representations[task_names[i]],
                        task_representations[task_names[j]]
                    )
                    redundancies.append(redundancy)
            result['redundancy'] = float(np.mean(redundancies))
        else:
            result['redundancy'] = None
        # 5. agg FER score
        # `FER = (1 - probe_acc) * entanglement_norm * (1 - rank_ratio)`
        fer_score = (1 - probe_acc) * entanglement_norm * (1 - rank_ratio)
        result['fer_score'] = float(fer_score)
        return result

    def clear_cache(self):
        """Clear cached computations."""
        self._cache = {}
