# >>> from examples.arc import ARCChainSolver, ARCDataset, ARCEvaluator
# >>> solver = ARCChainSolver(max_grid_size=30, chain_dims=[16,32,64,128,256])
# >>> dataset = ARCDataset('data/arc/training', max_grid_size=30)
# >>> evaluator = ARCEvaluator(solver, device='cuda')
# >>> results = evaluator.evaluate(dataset)
from .arc_chain_solver import (
    ARCChainSolver,
    BaselineARCSolver,
    create_arc_solver
)
from .arc_data_loader import (
    ARCTask,
    ARCDataset,
    ARCTaskDataset,
    create_arc_dataloaders
)
from .arc_augmentations import (
    ARCAugmentationPipeline,
    Rotate90,
    FlipHorizontal,
    FlipVertical,
    PermuteColors,
    DeterministicAugmentation,
    augment_task_examples
)
from .arc_evaluation import (
    ARCEvaluator,
    BaselineEvaluator,
    compare_models,
    analyze_errors
)

__all__ = [
    # solvers
    'ARCChainSolver',
    'BaselineARCSolver',
    'create_arc_solver',
    # data loading
    'ARCTask',
    'ARCDataset',
    'ARCTaskDataset',
    'create_arc_dataloaders',
    # augmentations
    'ARCAugmentationPipeline',
    'Rotate90',
    'FlipHorizontal',
    'FlipVertical',
    'PermuteColors',
    'DeterministicAugmentation',
    'augment_task_examples',
    # evaluation
    'ARCEvaluator',
    'BaselineEvaluator',
    'compare_models',
    'analyze_errors',
]
