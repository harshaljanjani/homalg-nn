import pytest
import torch
from homalg_nn.losses import (
    ExactnessLoss,
    ChainAxiomLoss,
    compute_kernel_projection,
    compute_image_projection,
    compute_kernel_basis,
    compute_image_basis,
)
from homalg_nn.core import ChainComplex


class TestSVDUtilities:
    """Test SVD-based projection computations."""
    def test_kernel_projection_identity(self):
        I = torch.eye(5)
        P_ker = compute_kernel_projection(I)
        assert torch.allclose(P_ker, torch.zeros(5, 5), atol=1e-4)

    def test_kernel_projection_zero_matrix(self):
        Z = torch.zeros(3, 5)
        P_ker = compute_kernel_projection(Z)
        assert torch.allclose(P_ker, torch.eye(5), atol=1e-4)

    def test_image_projection_identity(self):
        I = torch.eye(4)
        P_im = compute_image_projection(I)
        assert torch.allclose(P_im, torch.eye(4), atol=1e-4)

    def test_image_projection_zero_matrix(self):
        Z = torch.zeros(5, 3)
        P_im = compute_image_projection(Z)
        assert torch.allclose(P_im, torch.zeros(5, 5), atol=1e-4)

    def test_projection_idempotency(self):
        torch.manual_seed(42)
        A = torch.randn(6, 10)
        P_ker = compute_kernel_projection(A)
        P_ker_squared = P_ker @ P_ker
        assert torch.allclose(P_ker_squared, P_ker, atol=0.1)
        assert torch.allclose(P_ker.T, P_ker, atol=1e-5)

        P_im = compute_image_projection(A)
        P_im_squared = P_im @ P_im
        assert torch.allclose(P_im_squared, P_im, atol=0.1)
        assert torch.allclose(P_im.T, P_im, atol=1e-5)

    def test_kernel_image_complementary(self):
        torch.manual_seed(42)
        A = torch.randn(8, 8)
        ker_dim, _ = compute_kernel_basis(A)
        im_dim, _ = compute_image_basis(A)
        assert ker_dim + im_dim == 8

    def test_basis_orthonormality(self):
        torch.manual_seed(42)
        A = torch.randn(5, 10)
        ker_dim, K = compute_kernel_basis(A)
        if ker_dim > 0:
            # `K^T K` should be identity
            assert torch.allclose(K.T @ K, torch.eye(ker_dim), atol=1e-5)
        im_dim, U = compute_image_basis(A)
        if im_dim > 0:
            # `U^T U` should be identity
            assert torch.allclose(U.T @ U, torch.eye(im_dim), atol=1e-5)


class TestExactnessLoss:
    """Test ExactnessLoss computation and properties."""
    def test_exact_sequence_zero_loss(self):
        i = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        p = torch.tensor([[0.0, 0.0, 1.0]])
        loss_fn = ExactnessLoss()
        loss = loss_fn([p, i])
        assert loss.item() < 1e-5

    def test_non_exact_sequence_nonzero_loss(self):
        d_0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        d_1 = torch.zeros(3, 2)
        loss_fn = ExactnessLoss()
        loss = loss_fn([d_0, d_1])
        assert loss.item() > 0.1

    def test_loss_decreases_with_optimization(self):
        torch.manual_seed(42)
        d_0 = torch.randn(5, 10, requires_grad=True)
        d_1 = torch.randn(10, 8, requires_grad=True)
        loss_fn = ExactnessLoss()
        optimizer = torch.optim.SGD([d_0, d_1], lr=0.01)
        initial_loss = loss_fn([d_0, d_1]).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = loss_fn([d_0, d_1])
            loss.backward()
            optimizer.step()
        final_loss = loss_fn([d_0, d_1]).item()
        assert final_loss < initial_loss * 0.9

    def test_different_modes_consistent(self):
        torch.manual_seed(42)
        i = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        p = torch.tensor([[0.0, 0.0, 1.0]])
        loss_proj = ExactnessLoss(mode='projection_norm')([p, i])
        loss_angle = ExactnessLoss(mode='subspace_angle')([p, i])
        loss_overlap = ExactnessLoss(mode='overlap')([p, i])
        assert loss_proj.item() < 1e-4
        assert loss_angle.item() < 1e-4
        assert loss_overlap.item() < 1e-4

    def test_normalization_effect(self):
        torch.manual_seed(42)
        d_0 = torch.randn(5, 10)
        d_1 = torch.randn(10, 8)
        d_2 = torch.randn(8, 6)
        loss_unnorm = ExactnessLoss(normalize=False)([d_0, d_1, d_2])
        loss_norm = ExactnessLoss(normalize=True)([d_0, d_1, d_2])
        assert torch.allclose(loss_norm * 2, loss_unnorm, atol=1e-5)


class TestGradients:
    """Test gradient flow through exactness loss."""
    def test_gradients_exist(self):
        torch.manual_seed(42)
        d_0 = torch.randn(5, 10, requires_grad=True)
        d_1 = torch.randn(10, 8, requires_grad=True)
        loss_fn = ExactnessLoss()
        loss = loss_fn([d_0, d_1])
        loss.backward()
        assert d_0.grad is not None
        assert d_1.grad is not None
        assert not torch.isnan(d_0.grad).any()
        assert not torch.isnan(d_1.grad).any()

    def test_finite_difference_gradient_check(self):
        torch.manual_seed(42)
        d_0 = torch.randn(3, 5, requires_grad=True, dtype=torch.float64)
        d_1 = torch.randn(5, 4, requires_grad=True, dtype=torch.float64)
        loss_fn = ExactnessLoss(epsilon=1e-8)
        loss = loss_fn([d_0, d_1])
        loss.backward()
        analytical_grad = d_0.grad.clone()
        epsilon = 1e-6
        finite_diff_grad = torch.zeros_like(d_0)
        with torch.no_grad():
            for i in range(d_0.shape[0]):
                for j in range(d_0.shape[1]):
                    d_0_data = d_0.detach().clone()
                    d_0_data[i, j] += epsilon
                    loss_plus = loss_fn([d_0_data, d_1.detach()])
                    d_0_data = d_0.detach().clone()
                    d_0_data[i, j] -= epsilon
                    loss_minus = loss_fn([d_0_data, d_1.detach()])
                    finite_diff_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
        relative_error = torch.norm(analytical_grad - finite_diff_grad) / torch.norm(finite_diff_grad)
        assert relative_error < 1e-4, f"Relative error: {relative_error.item()}"

    def test_gradient_descent_converges(self):
        torch.manual_seed(42)
        d_0 = torch.randn(4, 8, requires_grad=True)
        d_1 = torch.randn(8, 6, requires_grad=True)
        loss_fn = ExactnessLoss()
        optimizer = torch.optim.Adam([d_0, d_1], lr=0.01)
        losses = []
        for step in range(500):
            optimizer.zero_grad()
            loss = loss_fn([d_0, d_1])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0] * 0.25
        assert losses[-1] < 1.0


class TestNumericalStability:
    """Test numerical stability with edge cases."""
    def test_near_zero_singular_values(self):
        torch.manual_seed(42)
        U = torch.randn(6, 4)
        U, _ = torch.linalg.qr(U)
        S = torch.tensor([10.0, 1.0, 1e-7, 1e-9])
        V = torch.randn(5, 4)
        V, _ = torch.linalg.qr(V)
        A = (U @ torch.diag(S) @ V.T).requires_grad_()
        loss_fn = ExactnessLoss(epsilon=1e-6)
        loss = loss_fn([A, torch.randn(5, 3)])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        loss.backward()
        assert not torch.isnan(A.grad).any()
        assert not torch.isinf(A.grad).any()

    def test_empty_boundary_maps(self):
        d_0 = torch.randn(5, 10)
        loss_fn = ExactnessLoss()
        loss = loss_fn([d_0])
        assert loss.item() == 0.0

    def test_large_matrices(self):
        torch.manual_seed(42)
        d_0 = torch.randn(50, 100, requires_grad=True)
        d_1 = torch.randn(100, 80, requires_grad=True)
        loss_fn = ExactnessLoss()
        loss = loss_fn([d_0, d_1])
        assert not torch.isnan(loss)
        loss.backward()
        assert not torch.isnan(d_0.grad).any()


class TestChainAxiomLoss:
    """Test chain axiom loss."""
    def test_zero_composition_zero_loss(self):
        d_0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        d_1 = torch.tensor([[0.0], [0.0]])
        loss_fn = ChainAxiomLoss()
        loss = loss_fn([d_0, d_1])
        composition = d_0 @ d_1
        assert torch.allclose(composition, torch.zeros_like(composition))
        assert loss.item() < 1e-10

    def test_nonzero_composition_positive_loss(self):
        d_0 = torch.eye(3)
        d_1 = torch.eye(3)
        loss_fn = ChainAxiomLoss()
        loss = loss_fn([d_0, d_1])
        assert loss.item() > 0.5


class TestIntegrationWithNumpy:
    """Test consistency with numpy-based homology computation."""
    def test_loss_matches_betti_numbers(self):
        torch.manual_seed(42)
        d_0 = torch.randn(5, 10, requires_grad=True)
        d_1 = torch.randn(10, 8, requires_grad=True)
        loss_fn = ExactnessLoss()
        optimizer = torch.optim.Adam([d_0, d_1], lr=0.01)
        initial_loss = loss_fn([d_0, d_1]).item()
        with torch.no_grad():
            chain = ChainComplex(
                dimensions=[5, 10, 8],
                boundary_maps=[d_0.numpy(), d_1.numpy()]
            )
            initial_betti = chain.get_betti_numbers()
        for _ in range(300):
            optimizer.zero_grad()
            loss = loss_fn([d_0, d_1])
            loss.backward()
            optimizer.step()
        final_loss = loss_fn([d_0, d_1]).item()
        with torch.no_grad():
            chain = ChainComplex(
                dimensions=[5, 10, 8],
                boundary_maps=[d_0.numpy(), d_1.numpy()]
            )
            final_betti = chain.get_betti_numbers()
        assert final_loss < initial_loss * 0.1
        assert sum(final_betti) <= sum(initial_betti)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
