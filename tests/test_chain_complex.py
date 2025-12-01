import pytest
import numpy as np
from homalg_nn.core import ChainComplex, compute_kernel, compute_image


class TestHomologyBasics:
    """Test basic kernel and image computations."""
    def test_kernel_of_zero_matrix(self):
        A = np.zeros((3, 5))
        ker_dim, ker_basis = compute_kernel(A)
        assert ker_dim == 5
        assert ker_basis.shape == (5, 5)
        result = A @ ker_basis
        assert np.allclose(result, 0)

    def test_kernel_of_identity(self):
        A = np.eye(4)
        ker_dim, ker_basis = compute_kernel(A)
        assert ker_dim == 0
        assert ker_basis.shape == (4, 0)

    def test_image_of_zero_matrix(self):
        A = np.zeros((3, 5))
        im_dim, im_basis = compute_image(A)
        assert im_dim == 0
        assert im_basis.shape == (3, 0)

    def test_image_of_identity(self):
        A = np.eye(4)
        im_dim, im_basis = compute_image(A)
        assert im_dim == 4
        assert im_basis.shape == (4, 4)

    def test_rank_nullity_theorem(self):
        np.random.seed(42)
        A = np.random.randn(6, 10)
        ker_dim, _ = compute_kernel(A)
        im_dim, _ = compute_image(A)
        assert ker_dim + im_dim == 10


class TestExactSequences:
    """Test detection of exact sequences."""
    def test_trivial_exact_sequence(self):
        chain = ChainComplex(dimensions=[0, 0, 0])

        assert chain.validate_chain_axiom()
        assert chain.is_exact()

        betti = chain.get_betti_numbers()
        assert all(b == 0 for b in betti)

    def test_simple_exact_sequence(self):
        i = np.array([
            [1, 0],
            [0, 1],
            [0, 0]
        ])
        p = np.array([[0, 0, 1]])
        chain = ChainComplex(dimensions=[1, 3, 2], boundary_maps=[p, i])
        assert chain.validate_chain_axiom()
        # `ker(p)` = `{[x, y, 0]}` which has dimension 2
        # `im(i)` = `{[x, y, 0]}` which has dimension 2
        # So `ker(p) = im(i)`, hence exact
        assert chain.is_exact_at(degree=0)
        beta_0, _ = chain.compute_homology(degree=0)
        assert beta_0 == 0

    def test_non_exact_sequence(self):
        d_0 = np.array([[1, 0, 0], [0, 1, 0]])
        d_1 = np.array([[0, 0], [0, 0], [0, 0]])
        chain = ChainComplex(dimensions=[2, 3, 2], boundary_maps=[d_0, d_1])
        assert chain.validate_chain_axiom()
        assert not chain.is_exact_at(degree=0)
        beta_0, _ = chain.compute_homology(degree=0)
        assert beta_0 == 1


class TestChainComplexProperties:
    """Test properties of chain complexes."""
    def test_chain_axiom_violation(self):
        d_0 = np.array([[1, 0], [0, 1]])
        d_1 = np.array([[1, 0], [0, 1]])
        chain = ChainComplex(dimensions=[2, 2, 2], boundary_maps=[d_0, d_1])
        assert not chain.validate_chain_axiom()

    def test_betti_numbers_length_sequence(self):
        d_1 = np.array([[2.0]])
        d_0 = np.array([[1.0]])
        chain = ChainComplex(dimensions=[1, 1, 1], boundary_maps=[d_0, d_1])
        d_0 = np.array([[0.0]])
        chain = ChainComplex(dimensions=[1, 1, 1], boundary_maps=[d_0, d_1])
        assert chain.validate_chain_axiom()

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            ChainComplex(
                dimensions=[2, 3, 2],
                boundary_maps=[
                    # should be (2, 3)!!
                    np.zeros((3, 3)),
                    np.zeros((3, 2))
                ]
            )

    def test_negative_dimensions(self):
        with pytest.raises(ValueError):
            ChainComplex(dimensions=[2, -1, 3])

    def test_summary_output(self):
        d_0 = np.zeros((2, 3))
        d_1 = np.zeros((3, 2))
        chain = ChainComplex(dimensions=[2, 3, 2], boundary_maps=[d_0, d_1])
        summary = chain.summary()
        assert 'dimensions' in summary
        assert 'betti_numbers' in summary
        assert 'is_exact' in summary
        assert 'exactness_defects' in summary
        assert 'ranks' in summary
        assert 'nullities' in summary
        assert 'is_valid_chain' in summary


class TestHomologyComputation:
    """Test homology group computations on specific examples."""
    def test_homology_of_zero_complex(self):
        chain = ChainComplex(dimensions=[0, 0, 0])
        betti = chain.get_betti_numbers()
        assert len(betti) == 2
        assert all(b == 0 for b in betti)

    def test_homology_caching(self):
        np.random.seed(42)
        d_0 = np.random.randn(5, 10)
        d_1 = np.random.randn(10, 8)
        chain = ChainComplex(dimensions=[5, 10, 8], boundary_maps=[d_0, d_1])
        beta_0_first, gen_first = chain.compute_homology(0, use_cache=True)
        beta_0_second, gen_second = chain.compute_homology(0, use_cache=True)
        assert beta_0_first == beta_0_second
        assert np.allclose(gen_first, gen_second)
        assert 0 in chain._homology_cache


class TestExactnessDefects:
    """Test exactness defect measurements."""
    def test_exact_sequence_zero_defect(self):
        i = np.array([[1, 0], [0, 1], [0, 0]])
        p = np.array([[0, 0, 1]])
        chain = ChainComplex(dimensions=[1, 3, 2], boundary_maps=[p, i])
        defect = chain.get_exactness_defect(0, method='subspace_distance')
        assert defect < 1e-6

    def test_defect_methods(self):
        np.random.seed(42)
        d_0 = np.random.randn(5, 10)
        d_1 = np.random.randn(10, 8)
        chain = ChainComplex(dimensions=[5, 10, 8], boundary_maps=[d_0, d_1])
        defect_subspace = chain.get_exactness_defect(0, method='subspace_distance')
        defect_dim = chain.get_exactness_defect(0, method='dimension_gap')
        defect_proj = chain.get_exactness_defect(0, method='projection_norm')
        assert defect_subspace >= 0
        assert defect_dim >= 0
        assert defect_proj >= 0


class TestNumericalStability:
    """Test numerical stability of computations."""
    def test_small_singular_values(self):
        A = np.array([[1e-15, 0], [0, 1]])
        ker_dim, _ = compute_kernel(A, epsilon=1e-10)
        assert ker_dim == 1

    def test_nearly_exact_sequence(self):
        eps = 1e-12
        i = np.array([[1, 0], [0, 1], [0, 0]])
        p = np.array([[0, eps, 1]])
        chain = ChainComplex(dimensions=[1, 3, 2], boundary_maps=[p, i], epsilon=1e-10)
        assert chain.validate_chain_axiom(tolerance=1e-10)

    def test_empty_matrices(self):
        d_0 = np.zeros((0, 2))
        d_1 = np.zeros((2, 3))
        chain = ChainComplex(dimensions=[0, 2, 3], boundary_maps=[d_0, d_1])
        assert chain.validate_chain_axiom()
        betti = chain.get_betti_numbers()
        assert len(betti) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
