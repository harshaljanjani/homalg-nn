from typing import List
import torch
import torch.nn as nn
from homalg_nn.losses.svd_utils import (
    compute_kernel_projection,
    compute_image_projection,
    compute_kernel_basis,
    compute_image_basis
)


class ExactnessLoss(nn.Module):
    """
    Differentiable loss measuring deviation from exactness.
    For a chain complex with boundary maps d_0, d_1, ..., d_{n-1}:
        `C_0 <-- d_0 -- C_1 <-- d_1 -- C_2 <-- d_2 -- ... <-- d_{n-1} -- C_n`
    Exactness at `C_i` means: `ker(d_{i-1}) = im(d_i)`
    The loss penalizes this deviation:
        `L = Σ_{i=1}^{n-1} ||P_ker(d_{i-1}) - P_im(d_i)||²_F`
    where:
    - `P_ker(d)` -> projection onto ker(d)
    - `P_im(d)` -> projection onto im(d)
    - `||·||_F` -> Frobenius norm
    - Exact sequences have `H_i = ker(d_{i-1}) / im(d_i) = 0` for all i.
    - This loss directly penalizes non-zero homology by measuring the
    "distance" between ker and im subspaces.
    """
    def __init__(
        self,
        epsilon: float = 1e-6,
        mode: str = 'projection_norm',
        normalize: bool = True,
        weight_by_dimension: bool = False
    ):
        """
        Initialize ExactnessLoss.
        """
        super().__init__()
        self.epsilon = epsilon
        self.mode = mode
        self.normalize = normalize
        self.weight_by_dimension = weight_by_dimension
        if mode not in ['projection_norm', 'subspace_angle', 'overlap']:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'projection_norm', 'subspace_angle', 'overlap'")

    def forward(self, boundary_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute exactness loss for a chain complex.
        Example:
        >>> d_0 = torch.randn(5, 10, requires_grad=True)   # C_1 -> C_0
        >>> d_1 = torch.randn(10, 8, requires_grad=True)   # C_2 -> C_1
        >>> loss_fn = ExactnessLoss()
        >>> loss = loss_fn([d_0, d_1])
        >>> loss.backward()  # Gradients flow to d_0 and d_1
        """
        if len(boundary_maps) < 2:
            return torch.tensor(0.0, device=boundary_maps[0].device, dtype=boundary_maps[0].dtype)
        total_loss = torch.tensor(0.0, device=boundary_maps[0].device, dtype=boundary_maps[0].dtype)
        num_interfaces = 0

        # for each consecutive pair (`d_i`, `d_{i + 1}`), check exactness at `C_{i+1}`
        # we want `ker(d_i) = im(d_{i + 1})`
        # both subspaces live in `C_{i+1}` (dimension = `d_i.shape[1]` = `d_{i+1}.shape[0]`)
        for i in range(len(boundary_maps) - 1):
            d_i = boundary_maps[i]  # `C_{i+1} -> C_i`
            d_i_plus_1 = boundary_maps[i + 1]  # `C_{i+2} -> C_{i+1}`
            defect = self.compute_defect_at_degree(d_i, d_i_plus_1)
            if self.weight_by_dimension:
                dimension = d_i.shape[1]  # = `d_{i+1}.shape[0]`
                defect = defect / dimension
            total_loss = total_loss + defect
            num_interfaces += 1
        if self.normalize and num_interfaces > 0:
            total_loss = total_loss / num_interfaces
        return total_loss

    def compute_defect_at_degree(
        self,
        d_i: torch.Tensor,
        d_i_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute exactness defect at a single degree.
        """
        assert d_i.shape[1] == d_i_plus_1.shape[0], \
            f"Dimension mismatch: d_i codomain {d_i.shape[1]} != d_{{i+1}} domain {d_i_plus_1.shape[0]}"
        if self.mode == 'projection_norm':
            return self._projection_norm_defect(d_i, d_i_plus_1)
        elif self.mode == 'subspace_angle':
            return self._subspace_angle_defect(d_i, d_i_plus_1)
        elif self.mode == 'overlap':
            return self._overlap_defect(d_i, d_i_plus_1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _projection_norm_defect(
        self,
        d_i: torch.Tensor,
        d_i_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute defect as `||P_ker(d_i) - P_im(d_{i+1})||²_F`
        This is the most straightforward measure.
        """
        P_ker = compute_kernel_projection(d_i, self.epsilon)
        P_im = compute_image_projection(d_i_plus_1, self.epsilon)
        diff = P_ker - P_im
        defect = torch.sum(diff ** 2)
        return defect

    def _subspace_angle_defect(
        self,
        d_i: torch.Tensor,
        d_i_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute defect based on principal angles between subspaces.
        >> defect = Σ sin²(θ_i)
        where `θ_i` are principal angles between `ker(d_i)` and `im(d_{i+1})`.
        Principal angles are defined via:
        - `cos(θ_i)` = `σ_i(K^T U)`
        where K, U are orthonormal bases for the two subspaces.
        """
        ker_dim, K = compute_kernel_basis(d_i, self.epsilon)
        im_dim, U = compute_image_basis(d_i_plus_1, self.epsilon)
        if ker_dim == 0 and im_dim == 0:
            # both trivial => exact
            return torch.tensor(0.0, device=d_i.device, dtype=d_i.dtype)
        if ker_dim == 0 or im_dim == 0:
            # one trivial, one non-trivial => non-exact
            return torch.tensor(float(max(ker_dim, im_dim)), device=d_i.device, dtype=d_i.dtype)

        # overlap matrix K^T @ U
        overlap = K.T @ U  # (ker_dim, im_dim)
        singular_values = torch.linalg.svdvals(overlap)
        cos_angles = torch.clamp(singular_values, 0.0, 1.0)
        sin_angles = torch.sqrt(1 - cos_angles ** 2)
        defect = torch.sum(sin_angles ** 2)
        if ker_dim != im_dim:
            dim_gap = abs(ker_dim - im_dim)
            defect = defect + dim_gap
        return defect

    def _overlap_defect(
        self,
        d_i: torch.Tensor,
        d_i_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute defect based on overlap between subspaces.
        Measures how much of `ker(d_i)` is explained by `im(d_{i+1})`:
        >> defect = Tr(I - U U^T K K^T)
        where K is ker basis and U is im basis.
        -> Exact if and only if `U U^T K = K` (i.e., `ker ⊆ im`) and dimensions match.
        """
        ker_dim, K = compute_kernel_basis(d_i, self.epsilon)
        im_dim, U = compute_image_basis(d_i_plus_1, self.epsilon)
        if ker_dim == 0 and im_dim == 0:
            return torch.tensor(0.0, device=d_i.device, dtype=d_i.dtype)
        if ker_dim == 0:
            # kernel trivial => automatically exact
            if im_dim == 0:
                return torch.tensor(0.0, device=d_i.device, dtype=d_i.dtype)
            else:
                return torch.tensor(float(im_dim), device=d_i.device, dtype=d_i.dtype)
        if im_dim == 0:
            # im trivial but kernel non-trivial => not exact
            return torch.tensor(float(ker_dim), device=d_i.device, dtype=d_i.dtype)
        # U U^T is projection onto im(d_{i+1})
        # K K^T is projection onto ker(d_i)
        P_im = U @ U.T
        P_ker = K @ K.T
        # U U^T K K^T
        overlap = P_im @ P_ker
        # "ideally", overlap = P_ker (full containment)
        # defect = `Tr(P_ker - overlap)` = `Tr(P_ker) - Tr(overlap)`
        trace_ker = torch.trace(P_ker)  # = ker_dim
        trace_overlap = torch.trace(overlap)
        defect = trace_ker - trace_overlap
        if ker_dim != im_dim:
            defect = defect + abs(ker_dim - im_dim)
        return torch.clamp(defect, min=0.0)


class ChainAxiomLoss(nn.Module):
    """
    Loss that enforces the chain axiom: `d_{i-1} ∘ d_i = 0`.
    (cool art): https://tomcircle.wordpress.com/2017/01/03/homology-why-boundary-of-boundary-0/
    """
    def __init__(self, normalize: bool = True):
        """
        Initialize ChainAxiomLoss.
        """
        super().__init__()
        self.normalize = normalize

    def forward(self, boundary_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute chain axiom violation.
        >> Loss = `Σ ||d_{i-1} ∘ d_i||²_F`
        """
        if len(boundary_maps) < 2:
            return torch.tensor(0.0, device=boundary_maps[0].device, dtype=boundary_maps[0].dtype)
        total_loss = torch.tensor(0.0, device=boundary_maps[0].device, dtype=boundary_maps[0].dtype)
        num_compositions = 0
        for i in range(len(boundary_maps) - 1):
            d_i = boundary_maps[i]  # `C_{i+1} -> C_i`
            d_i_plus_1 = boundary_maps[i + 1]  # `C_{i+2} -> C_{i+1}`
            # composition: `d_i @ d_{i+1}: C_{i+2} -> C_i`
            composition = d_i @ d_i_plus_1
            loss = torch.sum(composition ** 2)
            total_loss = total_loss + loss
            num_compositions += 1
        if self.normalize and num_compositions > 0:
            total_loss = total_loss / num_compositions
        return total_loss
