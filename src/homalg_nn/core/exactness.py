from typing import Tuple
import numpy as np
from scipy import linalg
from homalg_nn.core.homology import compute_kernel, compute_image


def check_exactness(
    d_n: np.ndarray,
    d_n_plus_1: np.ndarray,
    epsilon: float = 1e-10
) -> bool:
    """
    Check if a chain complex is exact at degree n.
    >> True if exact (`homology H_n = 0`), `False` otherwise
    - The sequence `... → C_{n+1} →^{d_{n+1}} C_n →^{d_n} C_{n-1} → ...`
    is exact at `C_n` if `ker(d_n) = im(d_{n+1})`.
    Equivalent to `H_n = ker(d_n) / im(d_{n+1}) = 0`.
    """
    ker_dim, ker_basis = compute_kernel(d_n, epsilon)
    im_dim, im_basis = compute_image(d_n_plus_1, epsilon)
    if ker_dim != im_dim:
        return False
    if ker_dim == 0:
        # both trivial => exact
        return True
    # ker_basis ⊆ span(im_basis)
    P_im = im_basis @ im_basis.T
    projected_ker = P_im @ ker_basis
    # ker_basis ⊆ im_basis
    diff = np.linalg.norm(ker_basis - projected_ker, ord='fro')
    return diff < epsilon * np.sqrt(ker_dim)


def compute_exactness_defect(
    d_n: np.ndarray,
    d_n_plus_1: np.ndarray,
    epsilon: float = 1e-10,
    method: str = 'subspace_distance'
) -> float:
    """
    Measure how far a chain complex is from being exact at degree n.
    - Exactness requires ker(d_n) = im(d_{n+1}) as subspaces.
    """
    ker_dim, ker_basis = compute_kernel(d_n, epsilon)
    im_dim, im_basis = compute_image(d_n_plus_1, epsilon)
    if method == 'dimension_gap':
        return abs(ker_dim - im_dim)
    elif method == 'subspace_distance':
        # distance between the two subspaces
        if ker_dim == 0 and im_dim == 0:
            return 0.0
        if ker_dim == 0 or im_dim == 0:
            # one trivial, one non-trivial => maximum distance
            return float(max(ker_dim, im_dim))
        # `angle_i = arccos(σ_i)` where `σ_i` are singular values of `ker_basis ^ T @ im_basis`
        overlap = ker_basis.T @ im_basis  # [(ker_dim, im_dim)]
        try:
            singular_values = linalg.svdvals(overlap)
        except linalg.LinAlgError:
            return abs(ker_dim - im_dim)
        singular_values = np.clip(singular_values, 0.0, 1.0)
        principal_angles = np.arccos(singular_values)
        # https://web.ma.utexas.edu/users/vandyke/notes/deep_learning_presentation/presentation.pdf
        defect = np.sum(np.sin(principal_angles) ** 2)
        if ker_dim != im_dim:
            dim_gap = abs(ker_dim - im_dim)
            defect += dim_gap
        return float(defect)

    elif method == 'projection_norm':
        # ||P_ker - P_im||_F where P are projection matrices
        n_dim = d_n.shape[1]
        if ker_dim > 0:
            P_ker = ker_basis @ ker_basis.T
        else:
            P_ker = np.zeros((n_dim, n_dim))

        if im_dim > 0:
            P_im = im_basis @ im_basis.T
        else:
            P_im = np.zeros((n_dim, n_dim))
        # Frobenius norm of diff
        defect = np.linalg.norm(P_ker - P_im, ord='fro')
        return float(defect)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_all_exactness_defects(
    boundary_maps: list[np.ndarray],
    epsilon: float = 1e-10,
    method: str = 'subspace_distance'
) -> list[float]:
    """
    Compute exactness defects at all degrees in a chain complex.
    Note: We cannot measure exactness at the last degree n without `d_{n+1}`
    so we return n defects for `n + 1` boundary maps.
    """
    if len(boundary_maps) < 2:
        return []
    defects = []
    for i in range(len(boundary_maps) - 1):
        d_i = boundary_maps[i]
        d_i_plus_1 = boundary_maps[i + 1]
        defect = compute_exactness_defect(d_i, d_i_plus_1, epsilon, method)
        defects.append(defect)
    return defects


def is_chain_complex(boundary_maps: list[np.ndarray], epsilon: float = 1e-6) -> Tuple[bool, list[float]]:
    """
    Check if a sequence of linear maps forms a valid chain complex.
    - A sequence is a chain complex if `d_{i-1} ∘ d_i = 0` for all i.
    - The fundamental requirement for a chain complex is that the composition
    of consecutive boundary maps is zero. This ensures that `im(d_i) ⊆ ker(d_{i-1})`,
    which is necessary for homology to be well-defined.
    """
    if len(boundary_maps) < 2:
        return True, []
    composition_norms = []
    is_valid = True
    for i in range(len(boundary_maps) - 1):
        d_i = boundary_maps[i]  # `C_i → C_{i-1}`
        d_i_plus_1 = boundary_maps[i + 1]  # `C_{i+1} → C_i`
        composition = d_i @ d_i_plus_1
        # check if `composition` is zero
        comp_norm = np.linalg.norm(composition, ord='fro')
        composition_norms.append(float(comp_norm))
        if comp_norm > epsilon:
            is_valid = False
    return is_valid, composition_norms
