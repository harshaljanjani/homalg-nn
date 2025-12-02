import pytest
import torch
from homalg_nn.nn import ChainModule


class TestChainModuleInitialization:
    """Test ChainModule initialization."""
    def test_basic_init(self):
        chain = ChainModule([5, 8, 10])
        assert len(chain.boundary_maps) == 2
        assert chain.boundary_maps[0].shape == (5, 8)
        assert chain.boundary_maps[1].shape == (8, 10)
        assert chain.boundary_maps[0].requires_grad

    def test_init_methods(self):
        methods = ['normal', 'xavier_normal', 'he_normal',
        'xavier_uniform', 'he_uniform', 'orthogonal', 'zero']
        for method in methods:
            chain = ChainModule([5, 8, 10], init_method=method)
            assert len(chain.boundary_maps) == 2

    def test_custom_boundary_maps(self):
        d0 = torch.randn(5, 8)
        d1 = torch.randn(8, 10)
        chain = ChainModule([5, 8, 10], boundary_maps=[d0, d1])
        assert torch.allclose(chain.boundary_maps[0], d0.to(dtype=torch.float64))
        assert torch.allclose(chain.boundary_maps[1], d1.to(dtype=torch.float64))

    def test_dtype_device(self):
        chain32 = ChainModule([5, 8], dtype=torch.float32)
        chain64 = ChainModule([5, 8], dtype=torch.float64)
        assert chain32.boundary_maps[0].dtype == torch.float32
        assert chain64.boundary_maps[0].dtype == torch.float64

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            ChainModule([5])            # too short
        with pytest.raises(ValueError):
            ChainModule([5, -1, 10])    # negative dimension

    def test_wrong_number_of_maps(self):
        d0 = torch.randn(5, 8)
        with pytest.raises(ValueError):
            # needs 2 maps
            ChainModule([5, 8, 10], boundary_maps=[d0])

    def test_wrong_shape_maps(self):
        d0 = torch.randn(5, 8)
        d1 = torch.randn(10, 8)         # wrong shape
        with pytest.raises(ValueError):
            ChainModule([5, 8, 10], boundary_maps=[d0, d1])


class TestChainModuleLoss:
    """Test loss computation."""
    def test_loss_modes(self):
        chain = ChainModule([5, 8, 10])
        for mode in ['exactness', 'chain_axiom', 'combined']:
            loss = chain.compute_exactness_loss(mode=mode)
            assert loss.ndim == 0       # scalar
            assert loss.requires_grad
            assert loss.item() >= 0

    def test_gradient_flow(self):
        chain = ChainModule([5, 8, 10])
        loss = chain.compute_exactness_loss()
        loss.backward()
        for d in chain.boundary_maps:
            assert d.grad is not None
            # non-trivial gradient
            assert not torch.all(d.grad == 0)

    def test_invalid_mode(self):
        chain = ChainModule([5, 8, 10])
        with pytest.raises(ValueError):
            chain.compute_exactness_loss(mode='invalid')


class TestChainModuleBetti:
    """Test Betti number computation."""
    def test_betti_computation(self):
        chain = ChainModule([5, 8, 10, 8, 5])
        betti = chain.get_betti_numbers()
        assert isinstance(betti, list)
        assert len(betti) == 4
        assert all(b >= 0 for b in betti)

    def test_betti_decreases_with_training(self):
        torch.manual_seed(42)
        chain = ChainModule([5, 8, 10, 8, 5], dtype=torch.float64, epsilon=1e-3)
        initial_rank = sum(chain.get_betti_numbers())
        optimizer = torch.optim.AdamW(chain.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(500):
            optimizer.zero_grad()
            loss = chain.compute_exactness_loss()
            loss.backward()
            optimizer.step()
        final_rank = sum(chain.get_betti_numbers())
        assert final_rank <= initial_rank

    def test_to_chain_complex(self):
        chain_module = ChainModule([5, 8, 10])
        chain_complex = chain_module.to_chain_complex()
        assert chain_complex.dimensions == [5, 8, 10]
        assert len(chain_complex.boundary_maps) == 2

    def test_exactness_defects(self):
        chain = ChainModule([5, 8, 10, 8, 5])
        defects = chain.get_exactness_defects()
        assert isinstance(defects, list)
        assert len(defects) == 3         # n-1 defects for n boundary maps
        assert all(d >= 0 for d in defects)


class TestChainModuleForward:
    """Test forward pass."""
    def test_forward_with_input(self):
        chain = ChainModule([5, 8, 10])
        x = torch.randn(3, 10, dtype=torch.float64)
        out = chain(x)
        assert out.shape == (3, 5)

    def test_forward_none(self):
        chain = ChainModule([5, 8, 10])
        out = chain(None)
        assert out is None


class TestChainModuleAnalysis:
    """Test analysis methods."""
    def test_is_exact(self):
        chain = ChainModule([5, 8, 10])
        result = chain.is_exact()
        assert isinstance(result, bool)

    def test_validate_chain_axiom(self):
        chain = ChainModule([5, 8, 10])
        result = chain.validate_chain_axiom(tolerance=1e-3)
        assert isinstance(result, bool)

    def test_summary(self):
        chain = ChainModule([5, 8, 10, 8, 5])
        summary = chain.summary()
        assert isinstance(summary, dict)
        assert 'dimensions' in summary
        assert 'num_parameters' in summary
        assert 'betti_numbers' in summary
        assert 'is_exact' in summary
        assert 'exactness_defects' in summary
        assert 'chain_axiom_valid' in summary
        assert 'device' in summary
        assert 'dtype' in summary
        assert summary['dimensions'] == [5, 8, 10, 8, 5]
        assert summary['num_parameters'] == 5*8 + 8*10 + 10*8 + 8*5

    def test_repr(self):
        chain = ChainModule([5, 8, 10])
        repr_str = repr(chain)
        assert isinstance(repr_str, str)
        assert 'ChainModule' in repr_str
        assert '5' in repr_str
        assert '8' in repr_str
        assert '10' in repr_str


class TestIntegration:
    """Full integration tests."""
    def test_full_training_loop(self):
        torch.manual_seed(42)
        chain = ChainModule([5, 8, 10, 8, 5], dtype=torch.float64, epsilon=1e-3)
        optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01, weight_decay=1e-4)
        initial_loss = chain.compute_exactness_loss().item()
        for _ in range(200):
            optimizer.zero_grad()
            loss = chain.compute_exactness_loss()
            loss.backward()
            optimizer.step()
        final_loss = chain.compute_exactness_loss().item()
        assert final_loss < initial_loss

    def test_loss_integration(self):
        chain = ChainModule([5, 8, 10])
        exactness_loss = chain.exactness_loss_fn(list(chain.boundary_maps))
        chain_axiom_loss = chain.chain_axiom_loss_fn(list(chain.boundary_maps))
        assert exactness_loss.requires_grad
        assert chain_axiom_loss.requires_grad

    def test_chain_module_integration(self):
        chain_module = ChainModule([5, 8, 10, 8, 5])
        chain_complex = chain_module.to_chain_complex()
        betti = chain_complex.get_betti_numbers()
        is_valid = chain_complex.validate_chain_axiom()
        defects = chain_complex.get_all_exactness_defects()
        assert isinstance(betti, list)
        assert isinstance(is_valid, bool)
        assert isinstance(defects, list)
