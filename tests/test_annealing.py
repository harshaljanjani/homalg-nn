import pytest
import torch
import numpy as np
from homalg_nn.nn import (
    ChainModule,
    AnnealingScheduler,
    create_recommended_scheduler
)


class TestAnnealingScheduler:
    """Test AnnealingScheduler class."""
    def test_initialization(self):
        scheduler = AnnealingScheduler(
            schedule='exponential',
            total_steps=1000,
            exactness_range=(0.1, 1.0),
            chain_axiom_range=(2.0, 0.5)
        )
        assert scheduler.schedule == 'exponential'
        assert scheduler.total_steps == 1000
        assert scheduler.ex_start == 0.1
        assert scheduler.ex_end == 1.0
        assert scheduler.ax_start == 2.0
        assert scheduler.ax_end == 0.5

    def test_invalid_schedule(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            AnnealingScheduler(schedule='invalid')

    def test_constant_schedule(self):
        scheduler = AnnealingScheduler(
            schedule='constant',
            total_steps=1000,
            exactness_range=(0.1, 1.0),
            chain_axiom_range=(2.0, 0.5)
        )
        for step in [0, 500, 999]:
            ex_w, ax_w = scheduler.get_weights(step)
            assert ex_w == 1.0
            assert ax_w == 0.5

    def test_linear_schedule(self):
        scheduler = AnnealingScheduler(
            schedule='linear',
            total_steps=1000,
            exactness_range=(0.0, 1.0),
            chain_axiom_range=(2.0, 0.0)
        )
        ex_w, ax_w = scheduler.get_weights(0)
        assert np.isclose(ex_w, 0.0, atol=1e-6)
        assert np.isclose(ax_w, 2.0, atol=1e-6)
        ex_w, ax_w = scheduler.get_weights(500)
        assert np.isclose(ex_w, 0.5, atol=1e-6)
        assert np.isclose(ax_w, 1.0, atol=1e-6)
        ex_w, ax_w = scheduler.get_weights(999)
        assert np.isclose(ex_w, 0.999, atol=1e-3)
        assert np.isclose(ax_w, 0.001, atol=1e-3)

    def test_exponential_schedule(self):
        scheduler = AnnealingScheduler(
            schedule='exponential',
            total_steps=1000,
            exactness_range=(0.1, 1.0),
            chain_axiom_range=(2.0, 0.5)
        )
        ex_w, ax_w = scheduler.get_weights(0)
        assert np.isclose(ex_w, 0.1, atol=1e-6)
        assert np.isclose(ax_w, 2.0, atol=1e-6)
        ex_w, ax_w = scheduler.get_weights(999)
        assert np.isclose(ex_w, 1.0, atol=1e-2)
        assert np.isclose(ax_w, 0.5, atol=1e-2)
        prev_ex_w = 0.0
        prev_ax_w = 3.0
        for step in range(0, 1000, 100):
            ex_w, ax_w = scheduler.get_weights(step)
            assert ex_w >= prev_ex_w
            assert ax_w <= prev_ax_w
            prev_ex_w = ex_w
            prev_ax_w = ax_w

    def test_cosine_schedule(self):
        scheduler = AnnealingScheduler(
            schedule='cosine',
            total_steps=1000,
            exactness_range=(0.1, 1.0),
            chain_axiom_range=(2.0, 0.5)
        )
        ex_w, ax_w = scheduler.get_weights(0)
        assert np.isclose(ex_w, 0.1, atol=1e-6)
        assert np.isclose(ax_w, 2.0, atol=1e-6)
        ex_w, ax_w = scheduler.get_weights(999)
        assert np.isclose(ex_w, 1.0, atol=1e-2)
        assert np.isclose(ax_w, 0.5, atol=1e-2)

    def test_two_stage_schedule(self):
        scheduler = AnnealingScheduler(
            schedule='two_stage',
            total_steps=1000,
            exactness_range=(0.0, 1.0),
            chain_axiom_range=(2.0, 0.5),
            two_stage_split=0.5
        )
        # stage 1 (steps 0-499): chain axiom only
        for step in [0, 250, 499]:
            ex_w, ax_w = scheduler.get_weights(step)
            assert ex_w == 0.0
            assert ax_w == 2.0
        # stage 2 (steps 500-999): balanced
        for step in [500, 750, 999]:
            ex_w, ax_w = scheduler.get_weights(step)
            assert ex_w == 1.0
            assert ax_w == 0.5

    def test_beyond_total_steps(self):
        scheduler = AnnealingScheduler(
            schedule='linear',
            total_steps=1000,
            exactness_range=(0.0, 1.0),
            chain_axiom_range=(2.0, 0.5)
        )
        ex_w, ax_w = scheduler.get_weights(2000)
        assert ex_w == 1.0
        assert ax_w == 0.5

    def test_get_schedule_info(self):
        scheduler = AnnealingScheduler(
            schedule='exponential',
            total_steps=2000
        )
        info = scheduler.get_schedule_info()
        assert 'exponential' in info.lower()
        assert '2000' in info
        assert isinstance(info, str)
        assert len(info) > 0


class TestCreateRecommendedScheduler:
    """Test create_recommended_scheduler helper."""
    def test_creates_exponential_scheduler(self):
        scheduler = create_recommended_scheduler(total_steps=2000)
        assert scheduler.schedule == 'exponential'
        assert scheduler.total_steps == 2000
        assert scheduler.ex_start == 0.1
        assert scheduler.ex_end == 1.0
        assert scheduler.ax_start == 2.0
        assert scheduler.ax_end == 0.5


class TestChainModuleAnnealing:
    """Test ChainModule integration with annealing."""
    def test_compute_loss_with_annealing(self):
        chain = ChainModule([5, 8, 10], dtype=torch.float64)
        total_steps = 1000
        for schedule in ['exponential', 'cosine', 'linear', 'constant']:
            for step in [0, 500, 999]:
                loss = chain.compute_loss_with_annealing(
                    step=step,
                    total_steps=total_steps,
                    schedule=schedule
                )
                assert isinstance(loss, torch.Tensor)
                assert loss.ndim == 0
                assert loss.requires_grad
                assert loss.item() >= 0

    def test_annealing_gradient_flow(self):
        chain = ChainModule([5, 8, 10], dtype=torch.float64)
        loss = chain.compute_loss_with_annealing(
            step=0,
            total_steps=1000,
            schedule='exponential'
        )
        loss.backward()
        for d in chain.boundary_maps:
            assert d.grad is not None
            assert not torch.all(d.grad == 0)

    def test_annealing_improves_convergence(self):
        torch.manual_seed(42)
        np.random.seed(42)
        chain_annealed = ChainModule(
            [5, 8, 10, 8, 5],
            init_method='xavier_normal',
            epsilon=1e-3,
            dtype=torch.float64
        )
        optimizer = torch.optim.AdamW(chain_annealed.parameters(), lr=0.01)
        total_steps = 1000
        for step in range(total_steps):
            optimizer.zero_grad()
            loss = chain_annealed.compute_loss_with_annealing(
                step=step,
                total_steps=total_steps,
                schedule='exponential'
            )
            loss.backward()
            optimizer.step()
        final_betti_annealed = chain_annealed.get_betti_numbers()
        final_rank_annealed = sum(final_betti_annealed)
        torch.manual_seed(42)
        np.random.seed(42)
        chain_baseline = ChainModule(
            [5, 8, 10, 8, 5],
            init_method='xavier_normal',
            exactness_weight=1.0,
            chain_axiom_weight=0.5,
            epsilon=1e-3,
            dtype=torch.float64
        )
        optimizer = torch.optim.AdamW(chain_baseline.parameters(), lr=0.01)
        for step in range(total_steps):
            optimizer.zero_grad()
            loss = chain_baseline.compute_exactness_loss(mode='combined')
            loss.backward()
            optimizer.step()
        final_betti_baseline = chain_baseline.get_betti_numbers()
        final_rank_baseline = sum(final_betti_baseline)
        assert final_rank_annealed <= final_rank_baseline + 1

    def test_custom_weight_ranges(self):
        chain = ChainModule([5, 8, 10], dtype=torch.float64)
        loss = chain.compute_loss_with_annealing(
            step=0,
            total_steps=1000,
            schedule='linear',
            exactness_range=(0.5, 2.0),
            chain_axiom_range=(1.0, 0.1)
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
