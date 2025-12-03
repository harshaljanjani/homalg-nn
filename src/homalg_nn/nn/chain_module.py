from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from homalg_nn.core import ChainComplex
from homalg_nn.losses import ExactnessLoss, ChainAxiomLoss


class ChainModule(nn.Module):
    """
    PyTorch nn.Module for trainable chain complexes.
    > Learnable boundary maps as nn.Parameter
    > Integrated exactness + chain axiom losses
    > Betti number computation via NumPy backend
    > Multiple initialization methods
    >>> chain = ChainModule([5, 8, 10, 8, 5])
    >>> optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01)
    >>> for step in range(1000):
    >>>     optimizer.zero_grad()
    >>>     loss = chain.compute_exactness_loss()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>> print(chain.get_betti_numbers())
    """
    def __init__(
        self,
        dimensions: List[int],
        boundary_maps: Optional[List[torch.Tensor]] = None,
        init_method: str = 'normal',
        init_scale: float = 0.1,
        exactness_weight: float = 1.0,
        chain_axiom_weight: float = 0.5,
        sparsity_weight: float = 0.0,
        epsilon: float = 1e-3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize ChainModule.
        """
        # === Config ===
        super().__init__()
        if not dimensions or len(dimensions) < 2:
            raise ValueError("dimensions must have at least 2 elements")
        if any(d <= 0 for d in dimensions):
            raise ValueError("All dimensions must be positive")
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.exactness_weight = exactness_weight
        self.chain_axiom_weight = chain_axiom_weight
        if boundary_maps is None:
            self.boundary_maps = nn.ParameterList([
                nn.Parameter(
                    self._init_boundary_map(
                        dimensions[i],
                        dimensions[i+1],
                        init_method,
                        init_scale
                    ).to(device=device, dtype=dtype)
                )
                for i in range(len(dimensions) - 1)
            ])
        else:
            if len(boundary_maps) != len(dimensions) - 1:
                raise ValueError(
                    f"Expected {len(dimensions) - 1} boundary maps, "
                    f"got {len(boundary_maps)}"
                )
            for i, d_map in enumerate(boundary_maps):
                expected_shape = (dimensions[i], dimensions[i+1])
                if d_map.shape != expected_shape:
                    raise ValueError(
                        f"Boundary map {i} has shape {d_map.shape}, "
                        f"expected {expected_shape}"
                    )
            self.boundary_maps = nn.ParameterList([
                nn.Parameter(d.clone().detach().to(device=device, dtype=dtype))
                for d in boundary_maps
            ])

        # === Loss Function ===
        self.exactness_loss_fn = ExactnessLoss(
            epsilon=epsilon,
            mode='projection_norm',
            normalize=True,
            sparsity_weight=sparsity_weight
        )
        self.chain_axiom_loss_fn = ChainAxiomLoss(normalize=True)

    def _init_boundary_map(
        self, m: int, n: int, method: str, scale: float
    ) -> torch.Tensor:
        """
        Initialize single boundary map d: R^n -> R^m.
        """
        if method == 'normal':
            return torch.randn(m, n) * scale
        elif method == 'xavier_uniform':
            d = torch.empty(m, n)
            nn.init.xavier_uniform_(d)
            return d
        elif method == 'xavier_normal':
            d = torch.empty(m, n)
            nn.init.xavier_normal_(d)
            return d
        elif method == 'he_uniform':
            d = torch.empty(m, n)
            nn.init.kaiming_uniform_(d, mode='fan_in', nonlinearity='linear')
            return d
        elif method == 'he_normal':
            d = torch.empty(m, n)
            nn.init.kaiming_normal_(d, mode='fan_in', nonlinearity='linear')
            return d
        elif method == 'orthogonal':
            d = torch.empty(m, n)
            nn.init.orthogonal_(d)
            return d
        elif method == 'zero':
            return torch.zeros(m, n)
        else:
            raise ValueError(f"Unknown init_method: {method}")

    def forward(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Apply chain composition: d_0 @ d_1 @ ... @ d_{n-1} @ x
        """
        if x is None:
            return None
        result = x
        for d in reversed(self.boundary_maps):
            result = result @ d.T
        return result

    def compute_exactness_loss(self, mode: str = 'combined') -> torch.Tensor:
        """
        Compute exactness-related loss.
        """
        if mode == 'exactness':
            return self.exactness_loss_fn(list(self.boundary_maps))
        elif mode == 'chain_axiom':
            return self.chain_axiom_loss_fn(list(self.boundary_maps))
        elif mode == 'combined':
            ex_loss = self.exactness_loss_fn(list(self.boundary_maps))
            ca_loss = self.chain_axiom_loss_fn(list(self.boundary_maps))
            total = (self.exactness_weight * ex_loss +
                    self.chain_axiom_weight * ca_loss)
            return total
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_loss_with_annealing(
        self,
        step: int,
        total_steps: int,
        schedule: str = 'exponential',
        exactness_range: tuple = (0.1, 1.0),
        chain_axiom_range: tuple = (2.0, 0.5)
    ) -> torch.Tensor:
        """
        Compute loss with annealing schedule for stable convergence.
        >>> chain = ChainModule([5, 8, 10])
        >>> optimizer = torch.optim.AdamW(chain.parameters(), lr=0.01)
        >>> total_steps = 2000
        >>> for step in range(total_steps):
        >>>     optimizer.zero_grad()
        >>>     loss = chain.compute_loss_with_annealing(
        >>>         step, total_steps, schedule='exponential'
        >>>     )
        >>>     loss.backward()
        >>>     optimizer.step()
        >>> # Achieves stable `betti = 0` with fewer oscillations
        """
        from homalg_nn.nn.annealing import AnnealingScheduler
        scheduler = AnnealingScheduler(
            schedule=schedule,
            total_steps=total_steps,
            exactness_range=exactness_range,
            chain_axiom_range=chain_axiom_range
        )
        exactness_weight, chain_axiom_weight = scheduler.get_weights(step)
        loss_exact = self.exactness_loss_fn(list(self.boundary_maps))
        loss_axiom = self.chain_axiom_loss_fn(list(self.boundary_maps))
        total_loss = exactness_weight * loss_exact + chain_axiom_weight * loss_axiom
        return total_loss

    def to_chain_complex(self) -> ChainComplex:
        """
        Convert to ChainComplex for analysis.
        """
        with torch.no_grad():
            maps_numpy = [d.cpu().numpy() for d in self.boundary_maps]
        return ChainComplex(
            dimensions=self.dimensions,
            boundary_maps=maps_numpy,
            epsilon=self.epsilon
        )

    def get_betti_numbers(self) -> List[int]:
        """
        Compute Betti numbers using NumPy backend.
        """
        chain = self.to_chain_complex()
        return chain.get_betti_numbers()

    def get_exactness_defects(self) -> List[float]:
        """
        Get exactness defects at all degrees.
        """
        chain = self.to_chain_complex()
        return chain.get_all_exactness_defects()

    def is_exact(self, tolerance: float = 1e-6) -> bool:
        """
        Check if approximately exact (all Betti numbers = 0).
        """
        betti = self.get_betti_numbers()
        return all(b == 0 for b in betti)

    def validate_chain_axiom(self, tolerance: float = 1e-6) -> bool:
        """
        Verify `d_{i-1} @ d_i ≈ 0` for all consecutive pairs.
        """
        chain = self.to_chain_complex()
        return chain.validate_chain_axiom(tolerance)

    def summary(self) -> Dict[str, Any]:
        """
        Comprehensive summary of the chain module.
        """
        return {
            'dimensions': self.dimensions,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'betti_numbers': self.get_betti_numbers(),
            'is_exact': self.is_exact(),
            'exactness_defects': self.get_exactness_defects(),
            'chain_axiom_valid': self.validate_chain_axiom(),
            'device': str(self.boundary_maps[0].device),
            'dtype': str(self.boundary_maps[0].dtype)
        }

    def __repr__(self) -> str:
        """String repr of the chain module."""
        dims_str = ' <- '.join(map(str, self.dimensions))
        return f"ChainModule({dims_str})"
