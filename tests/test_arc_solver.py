import pytest
import torch
import torch.nn.functional as F
import json
from examples.arc import (
    ARCChainSolver,
    BaselineARCSolver,
    create_arc_solver,
    ARCTask,
    ARCDataset,
    ARCTaskDataset,
    ARCAugmentationPipeline,
    Rotate90,
    FlipHorizontal,
    FlipVertical,
    PermuteColors,
    DeterministicAugmentation,
    ARCEvaluator,
    BaselineEvaluator
)


class TestARCChainSolver:
    """Test suite for ARCChainSolver."""
    def test_basic_functionality(self):
        solver = ARCChainSolver(
            max_grid_size=30,
            chain_dims=[16, 32, 64, 128, 256]
        )
        grid = torch.randint(0, 10, (4, 15, 15))
        logits = solver(grid)
        assert logits.shape == (4, 15, 15, 10)
        assert logits.dtype == torch.float64

    def test_variable_grid_sizes(self):
        solver = ARCChainSolver(max_grid_size=30, chain_dims=[8, 16, 32, 64])
        for h, w in [(5, 5), (10, 15), (30, 30), (1, 1)]:
            grid = torch.randint(0, 10, (2, h, w))
            logits = solver(grid)
            assert logits.shape == (2, h, w, 10)

    def test_with_mask(self):
        solver = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        grid = torch.randint(0, 10, (2, 5, 5))
        mask = torch.ones(2, 5, 5, dtype=torch.bool)
        mask[:, 3:, :] = False
        logits = solver(grid, grid_mask=mask)
        assert torch.allclose(logits[:, 3:, :, :], torch.zeros_like(logits[:, 3:, :, :]))

    def test_predict(self):
        solver = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        grid = torch.randint(0, 10, (4, 5, 5))
        predictions = solver.predict(grid)
        assert predictions.shape == (4, 5, 5)
        assert predictions.dtype == torch.int64
        assert (predictions >= 0).all() and (predictions < 10).all()

    def test_exactness_loss(self):
        solver = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32, 64])
        loss_exact = solver.compute_exactness_loss('exactness')
        loss_axiom = solver.compute_exactness_loss('chain_axiom')
        assert isinstance(loss_exact, torch.Tensor)
        assert isinstance(loss_axiom, torch.Tensor)
        assert loss_exact >= 0
        assert loss_axiom >= 0

    def test_betti_numbers(self):
        solver = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        betti = solver.get_betti_numbers()
        assert isinstance(betti, list)
        assert len(betti) == len(solver.chain_dims) - 1
        assert all(b >= 0 for b in betti)

    def test_return_features(self):
        solver = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        grid = torch.randint(0, 10, (2, 5, 5))
        _, features = solver(grid, return_features=True)
        assert 'embedded' in features
        assert 'encoded' in features
        assert 'chain_input' in features
        assert 'chain_output' in features
        assert 'decoded' in features

    def test_gradient_flow(self):
        solver = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        grid = torch.randint(0, 10, (2, 5, 5))
        logits = solver(grid)
        loss = logits.sum()
        loss.backward()
        assert solver.spatial_embed.value_embed.weight.grad is not None
        assert solver.encoder[0].weight.grad is not None
        assert solver.to_grid.weight.grad is not None


class TestBaselineARCSolver:
    """Test suite for BaselineARCSolver."""
    def test_basic_functionality(self):
        solver = BaselineARCSolver(max_grid_size=30)
        grid = torch.randint(0, 10, (4, 15, 15))
        logits = solver(grid)
        assert logits.shape == (4, 15, 15, 10)

    def test_predict(self):
        solver = BaselineARCSolver(max_grid_size=10)
        grid = torch.randint(0, 10, (2, 5, 5))
        predictions = solver.predict(grid)
        assert predictions.shape == (2, 5, 5)
        assert (predictions >= 0).all() and (predictions < 10).all()

    def test_gradient_flow(self):
        solver = BaselineARCSolver(max_grid_size=10)
        grid = torch.randint(0, 10, (2, 5, 5))
        logits = solver(grid)
        loss = logits.sum()
        loss.backward()
        assert solver.spatial_embed.value_embed.weight.grad is not None


class TestARCDataLoading:
    """Test suite for ARC data loading."""
    def create_dummy_task(self, tmp_path, task_id="test_task"):
        task_data = {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[5, 6], [7, 8]]
                },
                {
                    "input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                    "output": [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
                }
            ],
            "test": [
                {
                    "input": [[1, 1], [2, 2]],
                    "output": [[3, 3], [4, 4]]
                }
            ]
        }
        task_file = tmp_path / f"{task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f)
        return task_file

    def test_arc_task_creation(self, tmp_path):
        task_file = self.create_dummy_task(tmp_path)
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        task = ARCTask("test_task", task_data)
        assert task.task_id == "test_task"
        assert len(task.train_pairs) == 2
        assert len(task.test_pairs) == 1

    def test_arc_dataset_single_file(self, tmp_path):
        task_file = self.create_dummy_task(tmp_path)
        dataset = ARCDataset(task_file, split='train', max_grid_size=10)
        assert len(dataset) == 2
        input_grid, output_grid, mask = dataset[0]
        assert input_grid.shape == (10, 10)
        assert output_grid.shape == (10, 10)
        assert mask.shape == (10, 10)

    def test_arc_dataset_directory(self, tmp_path):
        self.create_dummy_task(tmp_path, "task1")
        self.create_dummy_task(tmp_path, "task2")
        dataset = ARCDataset(tmp_path, split='train', max_grid_size=10)
        assert len(dataset) == 4

    def test_padding(self, tmp_path):
        task_file = self.create_dummy_task(tmp_path)
        dataset = ARCDataset(task_file, split='train', max_grid_size=10)
        input_grid, _, mask = dataset[0]
        assert input_grid.shape == (10, 10)
        assert mask[:2, :2].all()     # valid region
        assert not mask[2:, :].any()  # padding
        assert not mask[:, 2:].any()  # padding

    def test_arc_task_dataset(self, tmp_path):
        self.create_dummy_task(tmp_path)
        dataset = ARCTaskDataset(tmp_path, max_grid_size=10)
        assert len(dataset) == 1
        task_data = dataset[0]
        assert 'task_id' in task_data
        assert 'train_inputs' in task_data
        assert 'train_outputs' in task_data
        assert 'test_inputs' in task_data
        assert task_data['train_inputs'].shape[0] == 2


class TestARCAugmentations:
    """Test suite for ARC augmentations."""
    def test_rotate90(self):
        grid = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        aug = Rotate90(k=1)
        rotated, _ = aug(grid, grid)
        expected = torch.tensor([
            [3, 6, 9],
            [2, 5, 8],
            [1, 4, 7]
        ])
        assert torch.equal(rotated, expected)

    def test_flip_horizontal(self):
        grid = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ])
        aug = FlipHorizontal()
        flipped, _ = aug(grid, grid)
        expected = torch.tensor([
            [3, 2, 1],
            [6, 5, 4]
        ])
        assert torch.equal(flipped, expected)

    def test_flip_vertical(self):
        grid = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ])
        aug = FlipVertical()
        flipped, _ = aug(grid, grid)
        expected = torch.tensor([
            [4, 5, 6],
            [1, 2, 3]
        ])
        assert torch.equal(flipped, expected)

    def test_color_permutation(self):
        grid = torch.tensor([
            [0, 1, 2],
            [3, 4, 5]
        ])
        perm = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        aug = PermuteColors(permutation=perm)
        permuted, _ = aug(grid, grid)
        expected = torch.tensor([
            [9, 8, 7],
            [6, 5, 4]
        ])
        assert torch.equal(permuted, expected)

    def test_deterministic_augmentation(self):
        grid = torch.tensor([
            [1, 2],
            [3, 4]
        ])
        aug = DeterministicAugmentation()
        all_versions, _ = aug(grid, grid)
        assert all_versions.shape == (8, 2, 2)
        assert torch.equal(all_versions[0], grid)

    def test_augmentation_pipeline(self):
        grid = torch.randint(0, 10, (5, 5))
        pipeline = ARCAugmentationPipeline(
            use_rotations=True,
            use_flips=True,
            rotation_prob=1.0,
            flip_prob=1.0
        )
        aug_grid, _ = pipeline(grid, grid)
        assert aug_grid.shape == grid.shape


class TestARCEvaluation:
    """Test suite for ARC evaluation."""
    def test_evaluator_creation(self):
        model = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        evaluator = ARCEvaluator(model, device='cpu')
        assert evaluator.num_attempts == 3
        assert evaluator.adaptation_steps == 50

    def test_prediction_generation(self):
        model = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        evaluator = ARCEvaluator(model, device='cpu', num_attempts=3)
        test_input = torch.randint(0, 10, (5, 5))
        test_mask = torch.ones(5, 5, dtype=torch.bool)
        predictions = evaluator.generate_predictions(
            model,
            test_input,
            test_mask,
            num_attempts=3
        )
        assert len(predictions) == 3
        assert all(p.shape == (5, 5) for p in predictions)

    def test_baseline_evaluator(self):
        model = BaselineARCSolver(max_grid_size=10)
        evaluator = BaselineEvaluator(model, device='cpu')
        assert evaluator.adaptation_steps == 0

    def test_test_time_adaptation(self):
        model = ARCChainSolver(max_grid_size=10, chain_dims=[8, 16, 32])
        evaluator = ARCEvaluator(
            model,
            device='cpu',
            adaptation_steps=5
        )
        train_inputs = torch.randint(0, 10, (2, 5, 5))
        train_outputs = torch.randint(0, 10, (2, 5, 5))
        train_masks = torch.ones(2, 5, 5, dtype=torch.bool)
        adapted_model = evaluator.adapt_to_task(train_inputs, train_outputs, train_masks)
        assert adapted_model is not model
        test_input = torch.randint(0, 10, (1, 5, 5))
        logits = adapted_model(test_input)
        assert logits.shape == (1, 5, 5, 10)


class TestFactory:
    """Test factory functions."""
    def test_create_arc_solver_chain(self):
        solver = create_arc_solver(use_chain=True, max_grid_size=20)
        assert isinstance(solver, ARCChainSolver)
        assert solver.max_grid_size == 20

    def test_create_arc_solver_baseline(self):
        solver = create_arc_solver(use_chain=False, max_grid_size=20)
        assert isinstance(solver, BaselineARCSolver)
        assert solver.max_grid_size == 20


class TestIntegration:
    """Integration tests combining multiple components."""
    def test_end_to_end_forward_pass(self):
        solver = ARCChainSolver(
            max_grid_size=15,
            chain_dims=[16, 32, 64, 32, 16],
            hidden_dim=256
        )
        grid = torch.randint(0, 10, (4, 10, 10))
        logits = solver(grid)
        assert logits.shape == (4, 10, 10, 10)
        assert not torch.isnan(logits).any()
        loss_exact = solver.compute_exactness_loss('exactness')
        loss_axiom = solver.compute_exactness_loss('chain_axiom')
        assert torch.isfinite(loss_exact)
        assert torch.isfinite(loss_axiom)

    def test_training_step_simulation(self):
        from homalg_nn.nn import create_recommended_scheduler
        solver = ARCChainSolver(
            max_grid_size=10,
            chain_dims=[8, 16, 32, 16, 8]
        )
        optimizer = torch.optim.Adam(solver.parameters(), lr=1e-3)
        scheduler = create_recommended_scheduler(total_steps=10)
        for step in range(3):
            inputs = torch.randint(0, 10, (2, 5, 5))
            outputs = torch.randint(0, 10, (2, 5, 5))
            optimizer.zero_grad()
            logits = solver(inputs)
            loss_task = F.cross_entropy(
                logits.permute(0, 3, 1, 2),
                outputs
            )
            ex_w, ax_w = scheduler.get_weights(step)
            loss_exact = solver.compute_exactness_loss('exactness')
            loss_axiom = solver.compute_exactness_loss('chain_axiom')
            loss_chain = ex_w * loss_exact + ax_w * loss_axiom
            loss = loss_task + 0.5 * loss_chain
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
