from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from homalg_nn.core.homology import (
    compute_kernel,
    compute_image,
    compute_homology_group,
    compute_betti_numbers
)
from homalg_nn.core.exactness import (
    check_exactness,
    compute_exactness_defect,
    compute_all_exactness_defects,
    is_chain_complex
)


class ChainComplex:
    """
    inspo.
    https://gudhi.inria.fr/python/latest/
    https://doc.sagemath.org/html/en/reference/homology/
    """

    def __init__(
        self,
        dimensions: List[int],
        boundary_maps: Optional[List[np.ndarray]] = None,
        epsilon: float = 1e-10
    ):
        """
        Initialize a chain complex.
        """
        if not dimensions or any(d < 0 for d in dimensions):
            raise ValueError("Dimensions must be non-negative integers")
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.length = len(dimensions)
        # boundary maps
        if boundary_maps is None:
            # zero boundary maps (trivial chain complex)
            self.boundary_maps = [
                np.zeros((dimensions[i], dimensions[i + 1]))
                for i in range(len(dimensions) - 1)
            ]
        else:
            # validate provided boundary maps
            if len(boundary_maps) != len(dimensions) - 1:
                raise ValueError(
                    f"Expected {len(dimensions) - 1} boundary maps for {len(dimensions)} spaces, "
                    f"got {len(boundary_maps)}"
                )
            for i, d_map in enumerate(boundary_maps):
                expected_shape = (dimensions[i], dimensions[i + 1])
                if d_map.shape != expected_shape:
                    raise ValueError(
                        f"Boundary map d_{i} has shape {d_map.shape}, "
                        f"expected {expected_shape}"
                    )
            self.boundary_maps = [np.array(d, dtype=float) for d in boundary_maps]
        self._homology_cache: Dict[int, Tuple[int, np.ndarray]] = {}

    def validate_chain_axiom(self, tolerance: Optional[float] = None) -> bool:
        """
        Check whether the composition of two consecutive boundary maps is zero.
        In a genuine chain complex, everything that comes out of the map `d_i` must 
        land inside the kernel of the next map `d_{i-1}`. In other words, applying 
        d_i and then immediately applying `d_{i-1}` should always give the zero vector.
        """
        if tolerance is None:
            tolerance = self.epsilon
        is_valid, comp_norms = is_chain_complex(self.boundary_maps, tolerance)
        if not is_valid:
            print(f"Chain axiom violated. Composition norms: {comp_norms}")
        return is_valid

    def compute_homology(self, degree: int, use_cache: bool = True) -> Tuple[int, np.ndarray]:
        """
        Compute the n-th homology group `H_n = ker(d_n) / im(d_{n+1})`.
        H_n measures "n-dimensional holes" in the chain complex.
        - `β_n = 0`: No holes (exact at degree n)
        - `β_n > 0`: Presence of non-trivial homology
        """
        if degree < 0 or degree >= len(self.boundary_maps):
            raise IndexError(f"Degree {degree} out of range [0, {len(self.boundary_maps) - 1}]")
        if use_cache and degree in self._homology_cache:
            return self._homology_cache[degree]
        d_n = self.boundary_maps[degree]
        # get d_{n+1} (if it exists)
        if degree < len(self.boundary_maps) - 1:
            d_n_plus_1 = self.boundary_maps[degree + 1]
        else:
            # at highest degree: `im(d_{n+1}) = {0}`
            d_n_plus_1 = np.zeros((self.dimensions[degree + 1], 1))
        betti_number, generators = compute_homology_group(d_n, d_n_plus_1, self.epsilon)
        if use_cache:
            self._homology_cache[degree] = (betti_number, generators)
        return betti_number, generators

    def get_betti_numbers(self) -> List[int]:
        """
        Compute all Betti numbers `[β_0, β_1, ..., β_{n-1}]`.
        - For an exact sequence, all Betti numbers are zero.
        - For a sequence with holes, some Betti numbers are non-zero.
        """
        return compute_betti_numbers(self.boundary_maps, self.epsilon)

    def is_exact_at(self, degree: int) -> bool:
        """
        Check if the chain complex is exact at degree n.
        Exactness at degree n means: ker(d_n) = im(d_{n+1})
        Equivalently: `H_n = 0 (β_n = 0)`
        """
        if degree < 0 or degree >= len(self.boundary_maps) - 1:
            raise IndexError(f"Cannot check exactness at degree {degree}")
        d_n = self.boundary_maps[degree]
        d_n_plus_1 = self.boundary_maps[degree + 1]
        return check_exactness(d_n, d_n_plus_1, self.epsilon)

    def is_exact(self) -> bool:
        """
        Check if the chain complex is exact everywhere.
        - A chain complex is exact if all homology groups vanish (all Betti numbers are zero).
        - == Information flows through the chain without loss or redundancy.
        """
        betti_numbers = self.get_betti_numbers()
        return all(beta == 0 for beta in betti_numbers)

    def get_exactness_defect(self, degree: int, method: str = 'subspace_distance') -> float:
        """
        Measure deviation from exactness at degree n.
        """
        if degree < 0 or degree >= len(self.boundary_maps) - 1:
            raise IndexError(f"Cannot compute defect at degree {degree}")

        d_n = self.boundary_maps[degree]
        d_n_plus_1 = self.boundary_maps[degree + 1]

        return compute_exactness_defect(d_n, d_n_plus_1, self.epsilon, method)

    def get_all_exactness_defects(self, method: str = 'subspace_distance') -> List[float]:
        """
        Compute exactness defects at all degrees.
        Tidbit: We can only compute `n-1` defects for n boundary maps, since the last
        degree requires `d_{n+1}` which doesn't exist.
        """
        return compute_all_exactness_defects(self.boundary_maps, self.epsilon, method)

    def get_ranks(self) -> List[int]:
        """
        Get the ranks of all boundary maps.
        - rank(d_i) = dim(im(d_i)) measures how much information passes through d_i.
        In an exact sequence, rank(d_{i+1}) + rank(d_i) = dim(C_i).
        """
        ranks = []
        for d_map in self.boundary_maps:
            im_dim, _ = compute_image(d_map, self.epsilon)
            ranks.append(im_dim)
        return ranks

    def get_nullities(self) -> List[int]:
        """
        Get the nullities (kernel dimensions) of all boundary maps.
        -- Gives us List `[nullity(d_0), nullity(d_1), ..., nullity(d_{n-1})]`
        - `nullity(d_i) = dim(ker(d_i))` measures how much information is lost by d_i.
        By rank-nullity theorem: `rank(d_i) + nullity(d_i) = dim(C_i)`.
        """
        nullities = []
        for _, d_map in enumerate(self.boundary_maps):
            ker_dim, _ = compute_kernel(d_map, self.epsilon)
            nullities.append(ker_dim)
        return nullities

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of the chain complex.
        """
        return {
            'dimensions': self.dimensions,
            'betti_numbers': self.get_betti_numbers(),
            'is_exact': self.is_exact(),
            'exactness_defects': self.get_all_exactness_defects(),
            'ranks': self.get_ranks(),
            'nullities': self.get_nullities(),
            'is_valid_chain': self.validate_chain_axiom(),
        }

    def __repr__(self) -> str:
        """Str representation of the chain complex."""
        dims_str = ' <- '.join(f'C_{i}({d})' for i, d in enumerate(self.dimensions))
        return f"ChainComplex: {dims_str}"

    def __str__(self) -> str:
        """Human-readable string repr."""
        summary = self.summary()
        lines = [
            repr(self),
            f"Valid chain complex: {summary['is_valid_chain']}",
            f"Betti numbers: {summary['betti_numbers']}",
            f"Exact: {summary['is_exact']}",
            f"Ranks: {summary['ranks']}",
        ]
        return '\n'.join(lines)
