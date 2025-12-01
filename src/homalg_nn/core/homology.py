from typing import Tuple
import numpy as np
from scipy import linalg


def compute_kernel(matrix: np.ndarray, epsilon: float = 1e-10) -> Tuple[int, np.ndarray]:
    """
    Compute the kernel (null space) of a linear map using SVD.
    For a matrix A: R^n → R^m, ker(A) = {v ∈ R^n : Av = 0}
    - If A = U Σ V^T (SVD), then ker(A) = span of columns of V corresponding to σ_i < ε
    """
    if matrix.size == 0:
        return 0, np.array([]).reshape(matrix.shape[1], 0)
    # SVD: A = U Σ V^T
    # U: m×m, Σ: m×n (diagonal), V^T: n×n
    try:
        U, singular_values, Vt = linalg.svd(matrix, full_matrices=True)
    except linalg.LinAlgError as e:
        raise ValueError(f"SVD failed: {e}")
    # V(n × n); we want columns corresponding to small singular values
    V = Vt.T
    n = matrix.shape[1]
    null_mask = singular_values < epsilon
    kernel_dim = n - np.sum(~null_mask)
    if kernel_dim == 0:
        return 0, np.array([]).reshape(n, 0)
    kernel_basis = V[:, -kernel_dim:]
    return kernel_dim, kernel_basis


def compute_image(matrix: np.ndarray, epsilon: float = 1e-10) -> Tuple[int, np.ndarray]:
    """
    Compute the image (column space) of a linear map using SVD.
    For a matrix A: R^n → R^m, im(A) = {Av : v ∈ R^n}
    - If A = U Σ V^T (SVD), then im(A) = span of columns of U corresponding to σ_i ≥ ε
    """
    if matrix.size == 0:
        return 0, np.array([]).reshape(matrix.shape[0], 0)
    try:
        U, singular_values, Vt = linalg.svd(matrix, full_matrices=True)
    except linalg.LinAlgError as e:
        raise ValueError(f"SVD failed: {e}")
    nonzero_mask = singular_values >= epsilon
    image_dim = np.sum(nonzero_mask)
    if image_dim == 0:
        return 0, np.array([]).reshape(matrix.shape[0], 0)
    image_basis = U[:, :image_dim]
    return image_dim, image_basis


def compute_homology_group(
    d_n: np.ndarray,
    d_n_plus_1: np.ndarray,
    epsilon: float = 1e-10
) -> Tuple[int, np.ndarray]:
    """
    Compute the n-th homology group `H_n = ker(d_n) / im(d_{n+1})`.
    - `H_n` is the quotient vector space `ker(d_n) / im(d_{n+1})`.
    The n-th Betti number `β_n = dim(H_n)` measures "n-dimensional holes".
    In an exact sequence, `H_n = 0 (i.e., β_n = 0)` everywhere.
    Non-zero homology indicates failure of exactness.
    """
    ker_dim, ker_basis = compute_kernel(d_n, epsilon)
    im_dim, im_basis = compute_image(d_n_plus_1, epsilon)
    assert d_n.shape[1] == d_n_plus_1.shape[0], \
        f"Dimension mismatch: d_n has codomain dim {d_n.shape[1]}, but d_{{n+1}} has domain dim {d_n_plus_1.shape[0]}"
    if ker_dim == 0:
        # if ker(d_n) is trivial, then H_n = 0
        return 0, np.array([]).reshape(d_n.shape[1], 0)
    if im_dim == 0:
        # if im(d_{n+1}) is trivial, then H_n ≅ ker(d_n)
        return ker_dim, ker_basis
    # compute quotient ker(d_n) / im(d_{n+1})
    # find a basis for ker(d_n) that's orthogonal to im(d_{n+1})
    P_im = im_basis @ im_basis.T
    ker_orthogonal = ker_basis - P_im @ ker_basis
    # QR decomposition; https://en.wikipedia.org/wiki/QR_decomposition
    if ker_orthogonal.shape[1] > 0:
        Q, R = linalg.qr(ker_orthogonal, mode='economic')
        r_diagonal = np.abs(np.diag(R))
        rank = np.sum(r_diagonal > epsilon)
        if rank == 0:
            return 0, np.array([]).reshape(d_n.shape[1], 0)
        homology_generators = Q[:, :rank]
        betti_number = rank
    else:
        betti_number = 0
        homology_generators = np.array([]).reshape(d_n.shape[1], 0)
    return betti_number, homology_generators


def compute_betti_numbers(boundary_maps: list[np.ndarray], epsilon: float = 1e-10) -> list[int]:
    """
    Compute all Betti numbers for a chain complex.
    Tidbit:
    For a chain complex `C_0 ← C_1 ← C_2 ← ... ← C_n`, we compute:
    - `H_0` = `ker(d_0) / im(d_1)`
    - `H_i` = `ker(d_i) / im(d_{i+1})` for `0 < i < n`
    - `H_n = ker(d_n) / {0} = ker(d_n)`
    """
    if not boundary_maps:
        return []
    n = len(boundary_maps)
    betti_numbers = []
    for i in range(n):
        d_i = boundary_maps[i]
        if i < n - 1:
            d_i_plus_1 = boundary_maps[i + 1]
        else:
            # ffor the last degree, im(d_{n+1}) = {0}
            d_i_plus_1 = np.zeros((d_i.shape[1], 1))
        betti_i, _ = compute_homology_group(d_i, d_i_plus_1, epsilon)
        betti_numbers.append(betti_i)
    return betti_numbers
