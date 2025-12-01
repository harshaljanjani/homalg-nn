from typing import Tuple
import torch


def safe_svd(
    matrix: torch.Tensor,
    epsilon: float = 1e-10,
    full_matrices: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Numerically stable SVD with gradient support.
    """
    if matrix.numel() == 0:
        m, n = matrix.shape
        min_dim = min(m, n)
        U = torch.eye(m, device=matrix.device, dtype=matrix.dtype)
        S = torch.zeros(min_dim, device=matrix.device, dtype=matrix.dtype)
        Vh = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        return U, S, Vh
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=full_matrices)
        return U, S, Vh
    except RuntimeError as e:
        # if SVD fails, try with regularization
        m, n = matrix.shape
        if m >= n:
            # regularize A^T A
            regularized = matrix.T @ matrix + epsilon * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
            eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
            S = torch.sqrt(torch.clamp(eigenvalues, min=0))
            Vh = eigenvectors.T
            U = matrix @ eigenvectors / (S.unsqueeze(0) + epsilon)
        else:
            # regularize A A^T
            regularized = matrix @ matrix.T + epsilon * torch.eye(m, device=matrix.device, dtype=matrix.dtype)
            eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
            S = torch.sqrt(torch.clamp(eigenvalues, min=0))
            U = eigenvectors
            Vh = (eigenvectors.T @ matrix) / (S.unsqueeze(1) + epsilon)
        return U, S, Vh


def compute_kernel_projection(
    matrix: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute projection matrix onto ker(matrix) using SVD.
    Uses the property: `P_ker = I - P_im(A^T)`. This approach
    uses reduced SVD (`Im(A^T)`) to avoid the gradient instability
    associated with the non-unique null space vectors returned by 
    full SVD.
    Properties:
    >> P_ker @ P_ker = P_ker (idempotent)
    >> P_ker^T = P_ker (symmetric)
    >> P_ker @ v = v for v in ker(A)
    >> P_ker @ v = 0 for v in im(A^T)
    """
    _, n = matrix.shape
    _, S, Vh = safe_svd(matrix, epsilon=epsilon * 1e-4, full_matrices=False)
    V = Vh.T  # (n, min(m, n))
    # use smooth sigmoid approximation instead of hard threshold
    # sigmoid((S - epsilon) / temperature) ≈ 1 when S >> epsilon
    # ≈ 0 when S << epsilon
    S_scale = torch.max(S) if len(S) > 0 else torch.tensor(1.0, device=matrix.device, dtype=matrix.dtype)
    temperature = torch.clamp(S_scale * 0.1, min=epsilon * 0.1, max=1.0)
    row_space_mask = torch.sigmoid((S - epsilon) / temperature)
    # projection using mask: P = V @ diag(mask) @ V^T; preserves gradients through smooth sigmoid
    # `P_row` = `V @ diag(mask) @ V^T`
    P_row = V @ torch.diag(row_space_mask) @ V.T
    # `P_ker = I - P_row`
    eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    P_ker = eye - P_row
    return P_ker

    # ===========================================
    # abandoned (for now): direct null space proj
    # >> the SVD of a rectangular matrix gives non-
    # unique null space basis vectors (arbitrary 
    # rotation); autograd cannot compute stable 
    # gradients for arbitrary vectors -- switched
    # to the rank-nullity method (above) which uses
    # the stable row-space projection.
    # ===========================================
    # m, n = matrix.shape
    # U, S, Vh = safe_svd(matrix, epsilon=epsilon * 1e-4, full_matrices=True)
    # V = Vh.T  # (n, n)
    # min_dim = min(m, n)
    # # use smooth sigmoid approximation instead of hard threshold
    # # sigmoid((epsilon - S) / temperature) ≈ 1 when S << epsilon, ≈ 0 when S >> epsilon
    # S_scale = torch.max(S) if len(S) > 0 else torch.tensor(1.0, device=matrix.device, dtype=matrix.dtype)
    # temperature = torch.clamp(S_scale * 0.1, min=epsilon * 0.1, max=1.0)
    # kernel_mask = torch.sigmoid((epsilon - S) / temperature)
    # # build a mask of length n (avoid in-place ops to preserve gradients)!!!
    # if n > m:
    #     # if n > m, the last (n - m) columns are automatically in kernel
    #     extra_dims_mask = torch.ones(n - min_dim, device=matrix.device, dtype=matrix.dtype)
    #     mask = torch.cat([kernel_mask, extra_dims_mask])
    # else:
    #     if len(kernel_mask) < n:
    #         padding = torch.zeros(n - len(kernel_mask), device=matrix.device, dtype=matrix.dtype)
    #         mask = torch.cat([kernel_mask, padding])
    #     else:
    #         mask = kernel_mask
    # # projection using mask: P = V @ diag(mask) @ V^T; preserves gradients through smooth sigmoid
    # P_ker = V @ torch.diag(mask) @ V.T
    # return P_ker

def compute_image_projection(
    matrix: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute projection matrix onto im(matrix) using SVD.
    The image (column space / range) of A consists of vectors Av for all v.
    Using SVD: `A = U Σ V^T`, `im(A)` = span of columns of U corresponding to σ_i ≥ ε.
    Properties:
    >> `P_im @ P_im = P_im` (idempotent)
    >> `P_im^T = P_im` (symmetric)
    >> `P_im @ v = v` for v in im(A)
    >> `P_im @ v = 0` for v in ker(A^T)
    """
    _, _ = matrix.shape
    U, S, _ = safe_svd(matrix, epsilon=epsilon * 1e-4, full_matrices=False)
    # use smooth sigmoid approximation instead of hard threshold
    # sigmoid((epsilon - S) / temperature) ≈ 1 when S << epsilon, ≈ 0 when S >> epsilon
    S_scale = torch.max(S) if len(S) > 0 else torch.tensor(1.0, device=matrix.device, dtype=matrix.dtype)
    temperature = torch.clamp(S_scale * 0.1, min=epsilon * 0.1, max=1.0)
    image_mask = torch.sigmoid((S - epsilon) / temperature)
    P_im = U @ torch.diag(image_mask) @ U.T
    return P_im

    # ===========================================
    # abandoned (for now): unnecessary full matrices
    # >> consistent with the kernel projection change,
    # we switched to `full_matrices = False` here as 
    # well -- (caveat) image projection was less 
    # unstable, but using reduced SVD is more efficient
    # and mathematically sufficient for the range.
    # ===========================================
    # m, _ = matrix.shape
    # U, S, _ = safe_svd(matrix, epsilon=epsilon * 1e-4, full_matrices=True)
    # # use smooth sigmoid approximation instead of hard threshold
    # # sigmoid((epsilon - S) / temperature) ≈ 1 when S << epsilon, ≈ 0 when S >> epsilon
    # S_scale = torch.max(S) if len(S) > 0 else torch.tensor(1.0, device=matrix.device, dtype=matrix.dtype)
    # temperature = torch.clamp(S_scale * 0.1, min=epsilon * 0.1, max=1.0)
    # image_mask = torch.sigmoid((S - epsilon) / temperature)
    # if len(image_mask) < m:
    #     padding = torch.zeros(m - len(image_mask), device=matrix.device, dtype=matrix.dtype)
    #     mask = torch.cat([image_mask, padding])
    # else:
    #     mask = image_mask
    # P_im = U @ torch.diag(mask) @ U.T
    # return P_im

def compute_kernel_basis(
    matrix: torch.Tensor,
    epsilon: float = 1e-6
) -> Tuple[int, torch.Tensor]:
    """
    Compute orthonormal basis for `ker(matrix)`.
    """
    m, n = matrix.shape
    _, S, Vh = safe_svd(matrix, epsilon=epsilon * 1e-4, full_matrices=True)
    kernel_mask = S < epsilon
    min_dim = min(m, n)
    if n > m:
        kernel_indices = torch.cat([
            torch.where(kernel_mask)[0],
            torch.arange(min_dim, n, device=matrix.device)
        ])
    else:
        kernel_indices = torch.where(kernel_mask)[0]
    kernel_dim = len(kernel_indices)
    if kernel_dim == 0:
        return 0, torch.empty(n, 0, device=matrix.device, dtype=matrix.dtype)
    V = Vh.T
    kernel_basis = V[:, kernel_indices]
    return kernel_dim, kernel_basis


def compute_image_basis(
    matrix: torch.Tensor,
    epsilon: float = 1e-6
) -> Tuple[int, torch.Tensor]:
    """
    Compute orthonormal basis for im(matrix).
    """
    m, _ = matrix.shape
    U, S, _ = safe_svd(matrix, epsilon=epsilon * 1e-4, full_matrices=True)
    image_mask = S >= epsilon
    image_indices = torch.where(image_mask)[0]
    image_dim = len(image_indices)
    if image_dim == 0:
        return 0, torch.empty(m, 0, device=matrix.device, dtype=matrix.dtype)
    image_basis = U[:, image_indices]
    return image_dim, image_basis


def verify_projection_properties(P: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """
    Verify that P is a valid projection matrix.
    A projection matrix must satisfy:
    >> P^2 = P (idempotent)
    >> P^T = P (symmetric)
    """
    # idempotency: P @ P = P
    P_squared = P @ P
    idempotent = torch.allclose(P_squared, P, atol=tolerance)
    # symmetry: P^T = P
    symmetric = torch.allclose(P.T, P, atol=tolerance)
    return idempotent and symmetric
