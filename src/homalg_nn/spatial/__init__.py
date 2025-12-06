# >>> from homalg_nn.spatial import SpatialEmbedding, GridChainModule
# >>> embed = SpatialEmbedding(grid_size=(30, 30), embed_dim=256)
# >>> grid_chain = GridChainModule(grid_size=(30, 30), chain_dims=[16,32,64,128,256])
from .spatial_embeddings import SpatialEmbedding, LearnedPositionEmbedding
from .topological_features import (
    TopologicalFeatureExtractor,
    SimplifiedTopologicalFeatures
)
from .grid_chain_module import GridChainModule, SimpleGridChain

__all__ = [
    'SpatialEmbedding',
    'LearnedPositionEmbedding',
    'TopologicalFeatureExtractor',
    'SimplifiedTopologicalFeatures',
    'GridChainModule',
    'SimpleGridChain',
]
