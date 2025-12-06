import torch
import torch.nn as nn
from ..nn import ChainModule
from .spatial_embeddings import SpatialEmbedding
from .topological_features import TopologicalFeatureExtractor


class GridChainModule(nn.Module):
    """Chain complex module for 2D grid transformations.
    >>> module = GridChainModule(
    ...     grid_size=(30, 30),
    ...     chain_dims=[16, 32, 64, 128, 256],
    ...     embed_dim=256,
    ...     out_dim=128
    ... )
    >>> grid = torch.randint(0, 10, (4, 15, 15))  # Variable size
    >>> output = module(grid)  # (4, 15, 15, 128)
    """
    def __init__(
        self,
        grid_size=(30, 30),
        chain_dims=[16, 32, 64, 128, 256],
        embed_dim=256,
        out_dim=128,
        epsilon=1e-3,
        dtype=torch.float64,
        use_topo_features=False,
        init_method='orthogonal'
    ):
        super().__init__()
        self.grid_size = grid_size
        self.chain_dims = chain_dims
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        self.dtype = dtype
        self.use_topo_features = use_topo_features
        self.max_cells = grid_size[0] * grid_size[1]
        # grid values + positions → vectors
        self.spatial_embed = SpatialEmbedding(
            grid_size=grid_size,
            embed_dim=embed_dim,
            num_values=10,
            dtype=torch.float32
        )
        if use_topo_features:
            self.topo_extractor = TopologicalFeatureExtractor(
                num_colors=10,
                feature_dim=32,
                dtype=torch.float32
            )
            self.topo_dim = 32
        else:
            self.topo_extractor = None
            self.topo_dim = 0
        input_to_chain = embed_dim + self.topo_dim
        self.pre_chain = nn.Linear(
            input_to_chain,
            chain_dims[-1],
            dtype=dtype
        )
        # NOTE: maps from dimensions[-1] → dimensions[0]
        self.chain = ChainModule(
            dimensions=chain_dims,
            epsilon=epsilon,
            dtype=dtype,
            init_method=init_method
        )
        self.post_chain = nn.Linear(
            chain_dims[0],
            out_dim,
            dtype=dtype
        )

    def forward(
        self,
        grid,
        grid_mask=None,
        return_chain_features=False
    ):
        """
        Process grid through chain complex.
        """
        batch_size, h, w = grid.shape
        # `(batch, h, w)` → `(batch, h, w, embed_dim)`
        embedded = self.spatial_embed(grid)
        # 2. extract and broadcast topological features
        if self.use_topo_features:
            topo_features = self.topo_extractor(grid)  # `(batch, topo_dim)`
            topo_broadcast = topo_features.unsqueeze(1).unsqueeze(2).expand(
                batch_size, h, w, self.topo_dim
            )
            embedded = torch.cat([embedded, topo_broadcast], dim=-1)
        # `(batch, h, w, embed_dim)` → `(batch*h*w, embed_dim)`
        embedded_flat = embedded.view(batch_size * h * w, -1)
        embedded_flat = embedded_flat.to(dtype=self.dtype)
        chain_input = self.pre_chain(embedded_flat)  # `(batch*h*w, chain_dims[-1])`
        chain_output = self.chain(chain_input)
        output_flat = self.post_chain(chain_output)  # `(batch*h*w, out_dim)`
        output = output_flat.view(batch_size, h, w, self.out_dim)
        if grid_mask is not None:
            # grid_mask: `(batch, h, w) → (batch, h, w, 1)`
            mask_expanded = grid_mask.unsqueeze(-1).to(dtype=output.dtype)
            output = output * mask_expanded
        if return_chain_features:
            chain_features = {
                'embedded': embedded,
                'chain_input': chain_input.view(batch_size, h, w, -1),
                'chain_output': chain_output.view(batch_size, h, w, -1),
                'topo_features': topo_features if self.use_topo_features else None
            }
            return output, chain_features
        return output

    def compute_exactness_loss(self, mode='exactness'):
        """
        Compute exactness or chain axiom loss from the internal chain.
        """
        return self.chain.compute_exactness_loss(mode=mode)

    def get_betti_numbers(self):
        """
        Get current Betti numbers from the chain complex.
        """
        return self.chain.get_betti_numbers()

    def extra_repr(self):
        return (
            f'grid_size={self.grid_size}, '
            f'chain_dims={self.chain_dims}, '
            f'embed_dim={self.embed_dim}, '
            f'out_dim={self.out_dim}, '
            f'epsilon={self.epsilon}, '
            f'use_topo_features={self.use_topo_features}'
        )


class SimpleGridChain(nn.Module):
    """Simple grid chain without topological features.
    >>> module = SimpleGridChain(grid_size=(10, 10), chain_dims=[8,16,32,16,8])
    >>> grid = torch.randint(0, 10, (4, 10, 10))
    >>> output = module(grid)  # (4, 10, 10, 10) - logits for 10 colors
    """
    def __init__(
        self,
        grid_size=(30, 30),
        chain_dims=[8, 16, 32, 64, 128],
        hidden_dim=128,
        dtype=torch.float64
    ):
        super().__init__()
        self.grid_size = grid_size
        self.chain_dims = chain_dims
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.value_embed = nn.Embedding(10, hidden_dim, dtype=torch.float32)
        self.chain = ChainModule(
            dimensions=chain_dims,
            epsilon=1e-3,
            dtype=dtype,
            init_method='orthogonal'
        )
        self.pre_chain = nn.Linear(hidden_dim, chain_dims[-1], dtype=dtype)
        self.post_chain = nn.Linear(chain_dims[0], 10, dtype=dtype)

    def forward(self, grid):
        """
        Process grid through simple chain.
        """
        batch_size, h, w = grid.shape
        embedded = self.value_embed(grid)  # `(batch, h, w, hidden_dim)`
        embedded_flat = embedded.view(batch_size * h * w, -1).to(dtype=self.dtype)
        x = self.pre_chain(embedded_flat)
        x = self.chain(x)
        logits_flat = self.post_chain(x)
        logits = logits_flat.view(batch_size, h, w, 10)
        return logits

    def compute_exactness_loss(self, mode='exactness'):
        return self.chain.compute_exactness_loss(mode=mode)

    def extra_repr(self):
        return f'grid_size={self.grid_size}, chain_dims={self.chain_dims}, simplified=True'
