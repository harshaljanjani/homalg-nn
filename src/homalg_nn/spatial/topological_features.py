import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalFeatureExtractor(nn.Module):
    """
    Extract topological features from 2D grids.
    > `β₀ (Betti_0)`: Number of connected components
    > `β₁ (Betti_1)`: Number of holes/cycles
    > Symmetry detection (horizontal, vertical, rotational)
    > Compactness/spatial distribution
    >>> extractor = TopologicalFeatureExtractor(num_colors=10, feature_dim=32)
    >>> grid = torch.randint(0, 10, (4, 15, 15))  # batch=4
    >>> features = extractor(grid)  # (4, 32)
    """
    def __init__(
        self,
        num_colors=10,
        feature_dim=32,
        use_learnable_aggregation=True,
        dtype=torch.float32
    ):
        super().__init__()
        self.num_colors = num_colors
        self.feature_dim = feature_dim
        self.use_learnable_aggregation = use_learnable_aggregation
        self.dtype = dtype
        # `[β₀, β₁, symmetry flags (3), compactness (1)]`
        self.raw_feature_dim = 6 * num_colors
        if use_learnable_aggregation:
            self.feature_mlp = nn.Sequential(
                nn.Linear(self.raw_feature_dim, 128, dtype=dtype),
                nn.ReLU(),
                nn.LayerNorm(128, dtype=dtype),
                nn.Linear(128, 64, dtype=dtype),
                nn.ReLU(),
                nn.LayerNorm(64, dtype=dtype),
                nn.Linear(64, feature_dim, dtype=dtype)
            )
        else:
            self.feature_mlp = nn.Linear(
                self.raw_feature_dim,
                feature_dim,
                dtype=dtype
            )

    def compute_betti_0(self, binary_grid):
        """
        Compute `Betti_0` (number of connected components) using flood-fill.
        """
        if binary_grid.sum() == 0:
            return 0
        grid_np = binary_grid.cpu().numpy().astype(bool)
        h, w = grid_np.shape
        visited = torch.zeros_like(binary_grid, dtype=torch.bool)
        num_components = 0

        def flood_fill(start_r, start_c):
            stack = [(start_r, start_c)]
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= h or c < 0 or c >= w:
                    continue
                if visited[r, c] or not binary_grid[r, c]:
                    continue
                visited[r, c] = True
                # 4-connected neighbors.
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])

        for r in range(h):
            for c in range(w):
                if binary_grid[r, c] and not visited[r, c]:
                    flood_fill(r, c)
                    num_components += 1
        return num_components

    def compute_betti_1(self, binary_grid):
        """
        Estimate Betti_1 (number of holes) using Euler characteristic.
        For a 2D grid with 4-connectivity:
        > χ = V - E + F
        > β₁ = E - V - F + 1
        """
        if binary_grid.sum() == 0:
            return 0
        h, w = binary_grid.shape
        # vertices (active cells)
        V = binary_grid.sum().item()
        # count edges (4-connected)
        E = 0
        for r in range(h):
            for c in range(w):
                if binary_grid[r, c]:
                    if r + 1 < h and binary_grid[r + 1, c]:
                        E += 1
                    if c + 1 < w and binary_grid[r, c + 1]:
                        E += 1
        # > euler characteristic for planar graph
        # > for grid: χ = β₀ - β₁
        # > β₁ = β₀ - χ
        # > approximation: χ ≈ V - E
        beta_0 = self.compute_betti_0(binary_grid)
        euler_char = V - E
        beta_1 = max(0, beta_0 - euler_char)
        return beta_1

    def detect_symmetry(self, binary_grid):
        """
        Detect symmetries in the binary grid.
        """
        h, w = binary_grid.shape
        # horizontal symmetry
        flipped_h = torch.flip(binary_grid, dims=[1])
        horizontal_sym = (binary_grid == flipped_h).float().mean().item()
        # vertical symmetry
        flipped_v = torch.flip(binary_grid, dims=[0])
        vertical_sym = (binary_grid == flipped_v).float().mean().item()
        # rotational symmetry (180-degree)
        rotated = torch.rot90(binary_grid, k=2, dims=(0, 1))
        # crop to same size if dimensions differ
        min_h, min_w = min(h, rotated.shape[0]), min(w, rotated.shape[1])
        rotational_sym = (
            binary_grid[:min_h, :min_w] == rotated[:min_h, :min_w]
        ).float().mean().item()
        return horizontal_sym, vertical_sym, rotational_sym

    def compute_compactness(self, binary_grid):
        """
        Compute spatial compactness of active cells.
        """
        if binary_grid.sum() == 0:
            return 0.0
        # area = number of active cells
        area = binary_grid.sum().item()
        # perimeter = number of edges touching inactive cells
        h, w = binary_grid.shape
        perimeter = 0
        for r in range(h):
            for c in range(w):
                if binary_grid[r, c]:
                    if r == 0 or not binary_grid[r - 1, c]:
                        perimeter += 1
                    if r == h - 1 or not binary_grid[r + 1, c]:
                        perimeter += 1
                    if c == 0 or not binary_grid[r, c - 1]:
                        perimeter += 1
                    if c == w - 1 or not binary_grid[r, c + 1]:
                        perimeter += 1
        if perimeter == 0:
            return 1.0
        compactness = 4 * torch.pi * area / (perimeter ** 2)
        compactness = min(1.0, compactness.item() if isinstance(compactness, torch.Tensor) else compactness)
        return compactness

    def extract_color_features(self, grid, color):
        """
        Extract all topological features for a single color.
        > `[β₀, β₁, h_sym, v_sym, r_sym, compactness]`
        """
        binary_grid = (grid == color)
        beta_0 = self.compute_betti_0(binary_grid)
        beta_1 = self.compute_betti_1(binary_grid)
        h_sym, v_sym, r_sym = self.detect_symmetry(binary_grid)
        compactness = self.compute_compactness(binary_grid)
        return [
            float(beta_0),
            float(beta_1),
            h_sym,
            v_sym,
            r_sym,
            compactness
        ]

    def forward(self, grid):
        """
        Extract topological features from grid.
        """
        batch_size = grid.shape[0]
        device = grid.device
        batch_features = []
        for b in range(batch_size):
            grid_b = grid[b]  # `(height, width)`
            all_features = []
            for color in range(self.num_colors):
                color_features = self.extract_color_features(grid_b, color)
                all_features.extend(color_features)
            batch_features.append(all_features)
        features_tensor = torch.tensor(
            batch_features,
            dtype=self.dtype,
            device=device
        )
        if self.use_learnable_aggregation:
            aggregated = self.feature_mlp(features_tensor)
        else:
            aggregated = features_tensor
        return aggregated

    def extra_repr(self):
        return (
            f'num_colors={self.num_colors}, feature_dim={self.feature_dim}, '
            f'raw_feature_dim={self.raw_feature_dim}'
        )


class SimplifiedTopologicalFeatures(nn.Module):
    """
    Simple version computing only basic counts.
    >>> extractor = SimplifiedTopologicalFeatures(num_colors=10, feature_dim=20)
    >>> grid = torch.randint(0, 10, (4, 15, 15))
    >>> features = extractor(grid)  # (4, 20)
    """
    def __init__(self, num_colors=10, feature_dim=20, dtype=torch.float32):
        super().__init__()
        self.num_colors = num_colors
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.mlp = nn.Sequential(
            nn.Linear(num_colors * 2, 64, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, feature_dim, dtype=dtype)
        )

    def forward(self, grid):
        """
        Extract simplified topological features.
        """
        _, h, w = grid.shape
        features = []
        for color in range(self.num_colors):
            color_mask = (grid == color).float()
            count = color_mask.sum(dim=(1, 2)) / (h * w)
            pooled = F.max_pool2d(
                color_mask.unsqueeze(1),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze(1)
            pooled_count = pooled.sum(dim=(1, 2)) / (h * w)
            features.append(count)
            features.append(pooled_count)
        features_tensor = torch.stack(features, dim=1)  # `(batch, num_colors * 2)`
        return self.mlp(features_tensor)

    def extra_repr(self):
        return f'num_colors={self.num_colors}, feature_dim={self.feature_dim}, simplified=True'
