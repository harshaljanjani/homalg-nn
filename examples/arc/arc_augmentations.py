import torch
import random
from typing import Tuple, List, Optional


class ARCAugmentation:
    """Base class for ARC augmentations."""
    def __call__(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Rotate90(ARCAugmentation):
    """
    Rotate grids by 90 degrees clockwise.
    """
    def __init__(self, k: int = 1):
        assert k in [1, 2, 3], "k must be 1, 2, or 3"
        self.k = k

    def __call__(self, input_grid, output_grid):
        input_rot = torch.rot90(input_grid, k=self.k, dims=(0, 1))
        output_rot = torch.rot90(output_grid, k=self.k, dims=(0, 1))
        return input_rot, output_rot


class FlipHorizontal(ARCAugmentation):
    """Flip grids horizontally (left-right)."""
    def __call__(self, input_grid, output_grid):
        input_flip = torch.flip(input_grid, dims=[1])
        output_flip = torch.flip(output_grid, dims=[1])
        return input_flip, output_flip


class FlipVertical(ARCAugmentation):
    """Flip grids vertically (top-bottom)."""
    def __call__(self, input_grid, output_grid):
        input_flip = torch.flip(input_grid, dims=[0])
        output_flip = torch.flip(output_grid, dims=[0])
        return input_flip, output_flip


class PermuteColors(ARCAugmentation):
    """
    Permute color IDs randomly.
    """
    def __init__(self, permutation: Optional[List[int]] = None):
        if permutation is not None:
            assert len(permutation) == 10, "Permutation must have 10 elements"
            assert set(permutation) == set(range(10)), "Permutation must be 0-9"
            self.permutation = permutation
        else:
            self.permutation = None

    def __call__(self, input_grid, output_grid):
        if self.permutation is None:
            perm = torch.randperm(10).tolist()
        else:
            perm = self.permutation
        input_perm = self._apply_permutation(input_grid, perm)
        output_perm = self._apply_permutation(output_grid, perm)
        return input_perm, output_perm

    def _apply_permutation(self, grid, perm):
        result = grid.clone()
        for old_color, new_color in enumerate(perm):
            result[grid == old_color] = new_color
        return result


class RandomAugmentation(ARCAugmentation):
    """
    Apply random augmentation from a list.
    """
    def __init__(self, augmentations: List[ARCAugmentation], p: float = 0.5):
        self.augmentations = augmentations
        self.p = p

    def __call__(self, input_grid, output_grid):
        for aug in self.augmentations:
            if random.random() < self.p:
                input_grid, output_grid = aug(input_grid, output_grid)
        return input_grid, output_grid


class Compose(ARCAugmentation):
    """
    Compose multiple augmentations sequentially.
    """
    def __init__(self, transforms: List[ARCAugmentation]):
        self.transforms = transforms

    def __call__(self, input_grid, output_grid):
        for transform in self.transforms:
            input_grid, output_grid = transform(input_grid, output_grid)
        return input_grid, output_grid


class ARCAugmentationPipeline:
    """
    Complete augmentation pipeline for ARC training.
    >>> pipeline = ARCAugmentationPipeline()
    >>> input_aug, output_aug = pipeline(input_grid, output_grid)
    """
    def __init__(
        self,
        use_rotations: bool = True,
        use_flips: bool = True,
        use_color_permutations: bool = False,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        color_perm_prob: float = 0.2
    ):
        self.use_rotations = use_rotations
        self.use_flips = use_flips
        self.use_color_permutations = use_color_permutations
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.color_perm_prob = color_perm_prob

    def __call__(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations.
        """
        if self.use_rotations and random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])
            aug = Rotate90(k=k)
            input_grid, output_grid = aug(input_grid, output_grid)
        if self.use_flips and random.random() < self.flip_prob:
            aug = FlipHorizontal()
            input_grid, output_grid = aug(input_grid, output_grid)
        if self.use_flips and random.random() < self.flip_prob:
            aug = FlipVertical()
            input_grid, output_grid = aug(input_grid, output_grid)
        if self.use_color_permutations and random.random() < self.color_perm_prob:
            aug = PermuteColors()
            input_grid, output_grid = aug(input_grid, output_grid)
        return input_grid, output_grid


def augment_task_examples(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    num_augmentations: int = 4,
    pipeline: Optional[ARCAugmentationPipeline] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment a batch of task examples.
    >>Tuple of (augmented_inputs, augmented_outputs)
    >> Each has shape (N*(1+num_augmentations), H, W)
    """
    if pipeline is None:
        pipeline = ARCAugmentationPipeline()
    all_inputs = [inputs]
    all_outputs = [outputs]
    for _ in range(num_augmentations):
        aug_inputs = []
        aug_outputs = []
        for inp, out in zip(inputs, outputs):
            inp_aug, out_aug = pipeline(inp, out)
            aug_inputs.append(inp_aug)
            aug_outputs.append(out_aug)
        all_inputs.append(torch.stack(aug_inputs))
        all_outputs.append(torch.stack(aug_outputs))
    augmented_inputs = torch.cat(all_inputs, dim=0)
    augmented_outputs = torch.cat(all_outputs, dim=0)
    return augmented_inputs, augmented_outputs


class DeterministicAugmentation:
    """
    Apply all deterministic augmentations systematically.
    >>> aug = DeterministicAugmentation()
    >>> inputs, outputs = aug(input_grid, output_grid)
    >>> print(inputs.shape)  # (8, H, W) - all 8 symmetries
    """
    def __call__(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate all 8 symmetric versions.
        """
        inputs = []
        outputs = []
        inputs.append(input_grid)
        outputs.append(output_grid)
        for k in [1, 2, 3]:
            aug = Rotate90(k=k)
            inp_rot, out_rot = aug(input_grid, output_grid)
            inputs.append(inp_rot)
            outputs.append(out_rot)
        flip_h = FlipHorizontal()
        inp_flip, out_flip = flip_h(input_grid, output_grid)
        inputs.append(inp_flip)
        outputs.append(out_flip)
        for k in [1, 2, 3]:
            aug = Rotate90(k=k)
            inp_rot, out_rot = aug(inp_flip, out_flip)
            inputs.append(inp_rot)
            outputs.append(out_rot)
        return torch.stack(inputs), torch.stack(outputs)


def test_augmentations():
    grid = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print("Original grid:")
    print(grid)
    print("\nRotate 90:")
    aug = Rotate90(k=1)
    rotated, _ = aug(grid, grid)
    print(rotated)
    print("\nFlip horizontal:")
    aug = FlipHorizontal()
    flipped, _ = aug(grid, grid)
    print(flipped)
    print("\nColor permutation:")
    aug = PermuteColors(permutation=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    perm, _ = aug(grid, grid)
    print(perm)
    print("\nAll 8 symmetries:")
    aug = DeterministicAugmentation()
    all_syms, _ = aug(grid, grid)
    print(f"Generated {all_syms.shape[0]} versions")
    for i, sym in enumerate(all_syms):
        print(f"\nSymmetry {i}:")
        print(sym)


if __name__ == '__main__':
    test_augmentations()
