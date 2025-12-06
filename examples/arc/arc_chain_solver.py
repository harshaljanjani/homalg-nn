import torch
import torch.nn as nn
from homalg_nn.spatial import SpatialEmbedding
from homalg_nn.nn import ChainModule


class ARCChainSolver(nn.Module):
    """
    ARC grid transformation solver with chain complex architecture.
    >>> solver = ARCChainSolver(max_grid_size=30, chain_dims=[16,32,64,128,256])
    >>> grid = torch.randint(0, 10, (4, 15, 15))  # Batch of 4
    >>> logits = solver(grid)  # (4, 15, 15, 10)
    >>> predictions = logits.argmax(dim=-1)  # (4, 15, 15)
    """
    def __init__(
        self,
        max_grid_size=30,
        chain_dims=[16, 32, 64, 128, 256],
        embed_dim=256,
        hidden_dim=512,
        epsilon=1e-3,
        dtype=torch.float64,
        use_topo_features=False,
        dropout=0.1
    ):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.chain_dims = chain_dims
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.dtype = dtype
        self.use_topo_features = use_topo_features
        # grid values + positions → vectors
        self.spatial_embed = SpatialEmbedding(
            grid_size=(max_grid_size, max_grid_size),
            embed_dim=embed_dim,
            num_values=10,
            dtype=torch.float32
        )
        # spatial features → latent representation
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim, dtype=dtype),
            nn.Dropout(dropout)
        )
        self.bridge_in = nn.Linear(hidden_dim, chain_dims[-1], dtype=dtype)
        self.chain = ChainModule(
            dimensions=chain_dims,
            epsilon=epsilon,
            dtype=dtype,
            init_method='orthogonal'
        )
        self.bridge_out = nn.Linear(chain_dims[0], hidden_dim // 2, dtype=dtype)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim, dtype=dtype),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, dtype=dtype)
        )
        self.to_grid = nn.Linear(embed_dim, 10, dtype=dtype)

    def forward(
        self,
        grid,
        grid_mask=None,
        return_features=False
    ):
        """
        Process grid through solver.
        """
        batch_size, h, w = grid.shape
        embedded = self.spatial_embed(grid)  # `(batch, h, w, embed_dim)`
        embedded_flat = embedded.view(batch_size * h * w, -1)
        embedded_flat = embedded_flat.to(dtype=self.dtype)
        encoded = self.encoder(embedded_flat)  # `(batch*h*w, hidden_dim)`
        chain_input = self.bridge_in(encoded)  # `(batch*h*w, chain_dims[-1])`
        chain_output = self.chain(chain_input)  # `(batch*h*w, chain_dims[0])`
        bridged = self.bridge_out(chain_output)  # `(batch*h*w, hidden_dim//2)`
        decoded = self.decoder(bridged)  # `(batch*h*w, embed_dim)`
        logits_flat = self.to_grid(decoded)  # `(batch*h*w, 10)`
        logits = logits_flat.view(batch_size, h, w, 10)
        if grid_mask is not None:
            mask_expanded = grid_mask.unsqueeze(-1).to(dtype=logits.dtype)
            logits = logits * mask_expanded
        if return_features:
            features = {
                'embedded': embedded,
                'encoded': encoded.view(batch_size, h, w, -1),
                'chain_input': chain_input.view(batch_size, h, w, -1),
                'chain_output': chain_output.view(batch_size, h, w, -1),
                'decoded': decoded.view(batch_size, h, w, -1)
            }
            return logits, features
        return logits

    def compute_exactness_loss(self, mode='exactness'):
        """
        Compute exactness or chain axiom loss.
        """
        return self.chain.compute_exactness_loss(mode=mode)

    def get_betti_numbers(self):
        """
        Get current Betti numbers from chain complex.
        """
        return self.chain.get_betti_numbers()

    def predict(self, grid, grid_mask=None):
        """
        Generate predictions (argmax of logits).
        """
        logits = self.forward(grid, grid_mask)
        predictions = logits.argmax(dim=-1)
        return predictions

    def extra_repr(self):
        return (
            f'max_grid_size={self.max_grid_size}, '
            f'chain_dims={self.chain_dims}, '
            f'embed_dim={self.embed_dim}, '
            f'hidden_dim={self.hidden_dim}'
        )


class BaselineARCSolver(nn.Module):
    """
    Baseline ARC solver without chain complex.
    """
    def __init__(
        self,
        max_grid_size=30,
        hidden_dim=512,
        embed_dim=256,
        dropout=0.1,
        dtype=torch.float64
    ):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.spatial_embed = SpatialEmbedding(
            grid_size=(max_grid_size, max_grid_size),
            embed_dim=embed_dim,
            num_values=10,
            dtype=torch.float32
        )
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim, dtype=dtype),
            nn.Dropout(dropout)
        )
        # chain has: `256→128 + 128→64 + 64→32 + 32→16 = ~50K params`
        # match with: `512→256→128`
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2, dtype=dtype),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, dtype=dtype),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4, dtype=dtype)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim, dtype=dtype),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, dtype=dtype)
        )
        self.to_grid = nn.Linear(embed_dim, 10, dtype=dtype)

    def forward(self, grid, grid_mask=None):
        """
        Process grid through baseline solver.
        """
        batch_size, h, w = grid.shape
        embedded = self.spatial_embed(grid)
        embedded_flat = embedded.view(batch_size * h * w, -1).to(dtype=self.dtype)
        encoded = self.encoder(embedded_flat)
        processed = self.mlp(encoded)
        decoded = self.decoder(processed)
        logits_flat = self.to_grid(decoded)
        logits = logits_flat.view(batch_size, h, w, 10)
        if grid_mask is not None:
            mask_expanded = grid_mask.unsqueeze(-1).to(dtype=logits.dtype)
            logits = logits * mask_expanded
        return logits

    def predict(self, grid, grid_mask=None):
        """Generate predictions."""
        logits = self.forward(grid, grid_mask)
        return logits.argmax(dim=-1)

    def extra_repr(self):
        return (
            f'max_grid_size={self.max_grid_size}, '
            f'hidden_dim={self.hidden_dim}, '
            f'baseline=True (no chain)'
        )


def create_arc_solver(use_chain=True, **kwargs):
    """
    Function to create ARC solver.
    """
    if use_chain:
        return ARCChainSolver(**kwargs)
    else:
        return BaselineARCSolver(**kwargs)
