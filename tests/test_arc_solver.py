import torch
from examples.arc import (
    ARCChainSolver,
    BaselineARCSolver,
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
