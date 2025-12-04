import numpy as np
import torch
from homalg_nn.analysis import RepresentationMetrics


class TestLinearProbeAccuracy:
    """Test linear probe metric on synthetic data."""
    def test_linearly_separable_data(self):
        metrics = RepresentationMetrics(random_state=42)
        # gen linearly separable data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        X_class0 = np.random.randn(n_samples // 2, n_features) + np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        X_class1 = np.random.randn(n_samples // 2, n_features) - np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).long()
        result = metrics.linear_probe_accuracy(X_torch, y_torch, n_splits=5)
        assert result['mean_accuracy'] > 0.95, f"Got {result['mean_accuracy']:.3f}"
        assert result['std_accuracy'] < 0.1
        assert 'probe_complexity' in result
        assert 'per_fold_accuracy' in result
        assert len(result['per_fold_accuracy']) == 5

    def test_random_labels(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).long()
        result = metrics.linear_probe_accuracy(X_torch, y_torch, n_splits=5)
        assert 0.4 < result['mean_accuracy'] < 0.6, f"Got {result['mean_accuracy']:.3f}"

    def test_perfect_separation(self):
        metrics = RepresentationMetrics(random_state=42)
        n_samples = 100
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples, dtype=int)
        X[:50, 0] = 10  # c0 has f0
        X[50:, 1] = 10  # c1 has f1
        y[50:] = 1
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).long()
        result = metrics.linear_probe_accuracy(X_torch, y_torch, n_splits=3)
        assert result['mean_accuracy'] > 0.99

    def test_multi_class(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_classes = 5
        n_per_class = 40
        n_features = 10
        X_list = []
        y_list = []
        for c in range(n_classes):
            center = np.zeros(n_features)
            center[c % n_features] = 5
            X_class = np.random.randn(n_per_class, n_features) + center
            X_list.append(X_class)
            y_list.append(np.full(n_per_class, c))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).long()
        result = metrics.linear_probe_accuracy(X_torch, y_torch, n_splits=5)
        assert result['mean_accuracy'] > 0.7


class TestMutualInformationMatrix:
    """Test MI metric."""
    def test_independent_gaussians(self):
        metrics = RepresentationMetrics(random_state=42, sample_size=500)
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        X_torch = torch.from_numpy(X).float()
        mi_matrix = metrics.mutual_information_matrix(X_torch, n_bins=20, normalize=True)
        for i in range(n_features):
            assert mi_matrix[i, i] > 0.5, f"Diagonal MI[{i},{i}] = {mi_matrix[i,i]:.3f}"
        mask = ~np.eye(n_features, dtype=bool)
        off_diag = mi_matrix[mask]
        mean_off_diag = np.mean(off_diag)
        assert mean_off_diag < 0.2, f"Mean off-diagonal MI = {mean_off_diag:.3f} (should be <0.2)"

    def test_perfectly_correlated(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_samples = 500
        x1 = np.random.randn(n_samples)
        x2 = x1 + np.random.randn(n_samples) * 0.05
        X = np.column_stack([x1, x2, np.random.randn(n_samples)])
        X_torch = torch.from_numpy(X).float()
        mi_matrix = metrics.mutual_information_matrix(X_torch, n_bins=20, normalize=True)
        assert mi_matrix[0, 1] > 0.5, f"MI[0,1] = {mi_matrix[0,1]:.3f}"
        assert mi_matrix[0, 2] < 0.3, f"MI[0,2] = {mi_matrix[0,2]:.3f}"

    def test_symmetric(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X_torch = torch.from_numpy(X).float()
        mi_matrix = metrics.mutual_information_matrix(X_torch)
        assert np.allclose(mi_matrix, mi_matrix.T), "MI matrix should be symmetric"


class TestSubspaceOrthogonality:
    """Test subspace orthogonality metric."""
    def test_identical_subspaces(self):
        metrics = RepresentationMetrics(random_state=42, sample_size=200)
        np.random.seed(42)
        n_samples = 300
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        X_torch = torch.from_numpy(X).float()
        orthogonality = metrics.subspace_orthogonality(X_torch, X_torch, method='cca')
        assert orthogonality > 0.95, f"Identical subspaces should have orthogonality ~1, got {orthogonality:.3f}"

    def test_orthogonal_subspaces(self):
        metrics = RepresentationMetrics(random_state=42, sample_size=200)
        np.random.seed(42)
        n_samples = 300
        n_features = 10
        X_A = np.random.randn(n_samples, n_features)
        X_A[:, 5:] = 0
        X_B = np.random.randn(n_samples, n_features)
        X_B[:, :5] = 0
        X_A_torch = torch.from_numpy(X_A).float()
        X_B_torch = torch.from_numpy(X_B).float()
        orthogonality = metrics.subspace_orthogonality(X_A_torch, X_B_torch, method='cca')
        assert orthogonality < 0.2, f"Orthogonal subspaces should have orthogonality ~0, got {orthogonality:.3f}"

    def test_principal_angles_method(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        X_A = np.random.randn(100, 10)
        X_B = np.random.randn(100, 10)
        X_A_torch = torch.from_numpy(X_A).float()
        X_B_torch = torch.from_numpy(X_B).float()
        orthogonality = metrics.subspace_orthogonality(X_A_torch, X_B_torch, method='principal_angles')
        assert 0 <= orthogonality <= 1, f"Orthogonality should be in [0,1], got {orthogonality}"


class TestRepresentationRank:
    """Test representation rank metric."""
    def test_full_rank_matrix(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        X_torch = torch.from_numpy(X).float()
        result = metrics.representation_rank(X_torch, epsilon=1e-3)
        assert result['rank_ratio'] > 0.8, f"Full rank should have ratio ~1, got {result['rank_ratio']:.3f}"
        assert result['effective_rank'] >= 8

    def test_low_rank_matrix(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        bases = np.random.randn(n_samples, 2)
        weights = np.random.randn(2, n_features)
        X = bases @ weights
        X_torch = torch.from_numpy(X).float()
        result = metrics.representation_rank(X_torch, epsilon=1e-2)
        assert result['rank_ratio'] < 0.4, f"Rank-2 matrix should have low ratio, got {result['rank_ratio']:.3f}"
        assert result['effective_rank'] <= 3

    def test_entropy(self):
        """Entropy should measure spectrum uniformity."""
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        # uniform spectrum (all singular values equal)
        X_uniform = np.random.randn(100, 10)
        result_uniform = metrics.representation_rank(torch.from_numpy(X_uniform).float())
        # non-uniform spectrum (one dominant singular value)
        X_nonuniform = np.random.randn(100, 10) * 0.1
        X_nonuniform[:, 0] += np.random.randn(100) * 10
        result_nonuniform = metrics.representation_rank(torch.from_numpy(X_nonuniform).float())
        assert result_uniform['entropy'] > result_nonuniform['entropy'], \
            f"Uniform entropy {result_uniform['entropy']:.3f} should be > non-uniform {result_nonuniform['entropy']:.3f}"


class TestMeasureRedundancy:
    """Test redundancy metric (fracture detection)."""
    def test_redundant_tasks(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        # both tasks learn the same repr
        base_repr = np.random.randn(n_samples, n_features)
        task_a = base_repr + np.random.randn(n_samples, n_features) * 0.1
        task_b = base_repr + np.random.randn(n_samples, n_features) * 0.1
        task_a_torch = torch.from_numpy(task_a).float()
        task_b_torch = torch.from_numpy(task_b).float()
        redundancy = metrics.measure_redundancy(task_a_torch, task_b_torch)
        assert redundancy > 0.7, f"Redundant tasks should have high redundancy, got {redundancy:.3f}"

    def test_distinct_tasks(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        task_a = np.random.randn(n_samples, n_features)
        task_a[:, 5:] = 0
        task_b = np.random.randn(n_samples, n_features)
        task_b[:, :5] = 0
        task_a_torch = torch.from_numpy(task_a).float()
        task_b_torch = torch.from_numpy(task_b).float()
        redundancy = metrics.measure_redundancy(task_a_torch, task_b_torch)
        assert redundancy < 0.3, f"Distinct tasks should have low redundancy, got {redundancy:.3f}"


class TestComputeFERScore:
    """Test aggregate FER score computation."""
    def test_fer_score_range(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).long()
        result = metrics.compute_fer_score(X_torch, labels=y_torch)
        assert 0 <= result['fer_score'] <= 1, f"FER score {result['fer_score']} not in [0,1]"

    def test_good_vs_bad_representations(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        n_samples = 200
        # good repr: linearly separable, low entanglement, full rank
        good_repr = np.random.randn(n_samples, 10)
        good_repr[:100, 0] += 5
        good_repr[100:, 1] += 5
        good_labels = np.array([0] * 100 + [1] * 100)
        base = np.random.randn(n_samples, 2)
        bad_repr = base @ np.random.randn(2, 10)
        bad_labels = np.random.randint(0, 2, n_samples)
        good_result = metrics.compute_fer_score(
            torch.from_numpy(good_repr).float(),
            labels=torch.from_numpy(good_labels).long()
        )
        bad_result = metrics.compute_fer_score(
            torch.from_numpy(bad_repr).float(),
            labels=torch.from_numpy(bad_labels).long()
        )
        assert bad_result['fer_score'] > good_result['fer_score'], \
            f"Bad FER {bad_result['fer_score']:.3f} should be > good FER {good_result['fer_score']:.3f}"

    def test_all_metrics_computed(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).long()
        result = metrics.compute_fer_score(X_torch, labels=y_torch)
        required_keys = ['fer_score', 'probe_accuracy', 'entanglement', 'rank_ratio']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
            if result[key] is not None:
                assert isinstance(result[key], (int, float)), f"{key} should be numeric"

    def test_multi_task_redundancy(self):
        metrics = RepresentationMetrics(random_state=42)
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        task_reprs = {
            'task_a': torch.from_numpy(X + np.random.randn(100, 10) * 0.1).float(),
            'task_b': torch.from_numpy(X + np.random.randn(100, 10) * 0.1).float()
        }
        result = metrics.compute_fer_score(
            torch.from_numpy(X).float(),
            labels=torch.from_numpy(y).long(),
            task_representations=task_reprs
        )
        assert 'redundancy' in result
        assert result['redundancy'] is not None
        assert 0 <= result['redundancy'] <= 1


class TestComputationalOptimizations:
    """Test sampling and caching optimizations."""
    def test_sampling(self):
        metrics = RepresentationMetrics(random_state=42, sample_size=100)
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        X_torch = torch.from_numpy(X).float()
        mi_matrix = metrics.mutual_information_matrix(X_torch)
        assert mi_matrix.shape == (10, 10)

    def test_no_errors_on_edge_cases(self):
        metrics = RepresentationMetrics(random_state=42)
        X_const = np.ones((100, 5))
        result = metrics.representation_rank(torch.from_numpy(X_const).float())
        assert result['effective_rank'] == 0
        # very few samples -- need at least `n_splits * n_classes` samples
        X_small = np.random.randn(20, 10)
        # balanced classes
        y_small = np.array([0] * 10 + [1] * 10)
        probe_result = metrics.linear_probe_accuracy(
            torch.from_numpy(X_small).float(),
            torch.from_numpy(y_small).long(),
            n_splits=2
        )
        assert 'mean_accuracy' in probe_result
