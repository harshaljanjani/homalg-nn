import math
import torch
import torch.nn as nn


class SpatialEmbedding(nn.Module):
    """
    Encode grid values + spatial positions.
    >>> embed = SpatialEmbedding(grid_size=(30, 30), embed_dim=256)
    >>> grid = torch.randint(0, 10, (4, 15, 15))  # batch=4, `15x15` grid
    >>> embedded = embed(grid)  # (4, 15, 15, 256)
    """
    def __init__(
        self,
        grid_size=(30, 30),
        embed_dim=256,
        num_values=10,
        dtype=torch.float32
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_values = num_values
        self.dtype = dtype
        self.value_dim = embed_dim // 2
        self.pos_dim = embed_dim - self.value_dim
        self.value_embed = nn.Embedding(num_values, self.value_dim, dtype=dtype)
        pos_encoding = self._create_position_encoding(grid_size, self.pos_dim)
        self.register_buffer('pos_encoding', pos_encoding)

    def _create_position_encoding(self, grid_size, pos_dim):
        """
        Create 2D sinusoidal position encoding.
        """
        height, width = grid_size
        row_dim = pos_dim // 2
        col_dim = pos_dim - row_dim
        row_pos = torch.arange(height, dtype=self.dtype).unsqueeze(1)  # `(height, 1)`
        col_pos = torch.arange(width, dtype=self.dtype).unsqueeze(0)   # `(1, width)`
        row_div_term = torch.exp(
            torch.arange(0, row_dim, 2, dtype=self.dtype) *
            (-math.log(10000.0) / row_dim)
        )
        col_div_term = torch.exp(
            torch.arange(0, col_dim, 2, dtype=self.dtype) *
            (-math.log(10000.0) / col_dim)
        )
        # row encoding
        row_encoding = torch.zeros(height, row_dim, dtype=self.dtype)
        row_encoding[:, 0::2] = torch.sin(row_pos * row_div_term)
        if row_dim > 1:
            row_encoding[:, 1::2] = torch.cos(row_pos * row_div_term)
        # column encoding
        col_encoding = torch.zeros(width, col_dim, dtype=self.dtype)
        col_encoding[:, 0::2] = torch.sin(col_pos.T * col_div_term)
        if col_dim > 1:
            col_encoding[:, 1::2] = torch.cos(col_pos.T * col_div_term)
        # `(height, width, pos_dim)`
        row_encoding_2d = row_encoding.unsqueeze(1).expand(height, width, row_dim)
        col_encoding_2d = col_encoding.unsqueeze(0).expand(height, width, col_dim)
        pos_encoding = torch.cat([row_encoding_2d, col_encoding_2d], dim=-1)
        return pos_encoding

    def forward(self, grid):
        """
        Embed grid with value + position information.
        """
        if grid.max() >= self.num_values or grid.min() < 0:
            raise ValueError(
                f"Grid values must be in [0, {self.num_values}), "
                f"got range [{grid.min()}, {grid.max()}]"
            )
        batch_size, h, w = grid.shape
        # `(batch, h, w) -> (batch, h, w, value_dim)`
        val_emb = self.value_embed(grid)
        # `(max_height, max_width, pos_dim)`
        pos_emb = self.pos_encoding[:h, :w]  # `(h, w, pos_dim)`
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # `(batch, h, w, embed_dim)`
        embedding = torch.cat([val_emb, pos_emb], dim=-1)
        return embedding

    def extra_repr(self):
        """String repr for print(model)."""
        return (
            f'grid_size={self.grid_size}, embed_dim={self.embed_dim}, '
            f'num_values={self.num_values}, value_dim={self.value_dim}, '
            f'pos_dim={self.pos_dim}'
        )


class LearnedPositionEmbedding(nn.Module):
    """
    Alt: Fully learned position embeddings.
    >>> embed = LearnedPositionEmbedding(grid_size=(10, 10), embed_dim=128)
    >>> grid = torch.randint(0, 10, (4, 10, 10))
    >>> embedded = embed(grid)  # (4, 10, 10, 128)
    """
    def __init__(
        self,
        grid_size=(30, 30),
        embed_dim=256,
        num_values=10,
        dtype=torch.float32
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_values = num_values
        self.dtype = dtype
        self.value_dim = embed_dim // 2
        self.pos_dim = embed_dim - self.value_dim
        self.value_embed = nn.Embedding(num_values, self.value_dim, dtype=dtype)
        height, width = grid_size
        row_pos_dim = self.pos_dim // 2
        col_pos_dim = self.pos_dim - row_pos_dim
        self.row_pos_embed = nn.Embedding(height, row_pos_dim, dtype=dtype)
        self.col_pos_embed = nn.Embedding(width, col_pos_dim, dtype=dtype)

    def forward(self, grid):
        """
        Embed grid with value + learned position embeddings.
        """
        batch_size, h, w = grid.shape
        val_emb = self.value_embed(grid)  # `(batch, h, w, value_dim)`
        row_indices = torch.arange(h, device=grid.device).unsqueeze(1).expand(h, w)
        col_indices = torch.arange(w, device=grid.device).unsqueeze(0).expand(h, w)
        row_emb = self.row_pos_embed(row_indices)  # `(h, w, row_pos_dim)`
        col_emb = self.col_pos_embed(col_indices)  # `(h, w, col_pos_dim)`
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # `(h, w, pos_dim)`
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        embedding = torch.cat([val_emb, pos_emb], dim=-1)
        return embedding

    def extra_repr(self):
        return (
            f'grid_size={self.grid_size}, embed_dim={self.embed_dim}, '
            f'num_values={self.num_values}, learned_positions=True'
        )
