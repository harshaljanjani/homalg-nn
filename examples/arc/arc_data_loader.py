# {
#     "train": [
#         {"input": [[...]], "output": [[...]]},
#         ...
#     ],
#     "test": [
#         {"input": [[...]], "output": [[...]]}
#     ]
# }
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ARCTask:
    """
    Represents a single ARC task.
    """
    def __init__(self, task_id: str, task_data: Dict):
        self.task_id = task_id
        self.train_pairs = [
            (np.array(pair['input']), np.array(pair['output']))
            for pair in task_data['train']
        ]
        self.test_pairs = [
            (np.array(pair['input']), np.array(pair['output']))
            for pair in task_data['test']
        ]

    def __repr__(self):
        return (
            f"ARCTask(id='{self.task_id}', "
            f"train_examples={len(self.train_pairs)}, "
            f"test_examples={len(self.test_pairs)})"
        )


class ARCDataset(Dataset):
    """
    Dataset for ARC tasks.
    >>> dataset = ARCDataset('data/arc/training', split='train', max_grid_size=30)
    >>> input_grid, output_grid, mask = dataset[0]
    >>> print(input_grid.shape)  # (30, 30)
    """
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        max_grid_size: int = 30,
        transform=None,
        return_task_id: bool = False
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.max_grid_size = max_grid_size
        self.transform = transform
        self.return_task_id = return_task_id
        self.tasks = self._load_tasks()
        # `(task_idx, example_idx)`
        self.examples = []
        for task_idx, task in enumerate(self.tasks):
            pairs = task.train_pairs if split == 'train' else task.test_pairs
            for example_idx in range(len(pairs)):
                self.examples.append((task_idx, example_idx))

    def _load_tasks(self) -> List[ARCTask]:
        """Load all tasks from data path."""
        tasks = []
        if self.data_path.is_file():
            task_id = self.data_path.stem
            with open(self.data_path, 'r') as f:
                task_data = json.load(f)
            tasks.append(ARCTask(task_id, task_data))
        elif self.data_path.is_dir():
            for json_file in sorted(self.data_path.glob('*.json')):
                task_id = json_file.stem
                with open(json_file, 'r') as f:
                    task_data = json.load(f)
                tasks.append(ARCTask(task_id, task_data))
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
        if len(tasks) == 0:
            raise ValueError(f"No tasks found in {self.data_path}")
        return tasks

    def _pad_grid(self, grid: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad grid to max_grid_size and create mask.
        """
        h, w = grid.shape
        if h > self.max_grid_size or w > self.max_grid_size:
            raise ValueError(
                f"Grid size ({h}, {w}) exceeds max_grid_size ({self.max_grid_size})"
            )
        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int64)
        padded[:h, :w] = grid
        mask = np.zeros((self.max_grid_size, self.max_grid_size), dtype=bool)
        mask[:h, :w] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get a single example.
        """
        task_idx, example_idx = self.examples[idx]
        task = self.tasks[task_idx]
        pairs = task.train_pairs if self.split == 'train' else task.test_pairs
        input_grid, output_grid = pairs[example_idx]
        input_padded, input_mask = self._pad_grid(input_grid)
        output_padded, _ = self._pad_grid(output_grid)
        if self.transform:
            input_padded, output_padded = self.transform(input_padded, output_padded)
        if self.return_task_id:
            return input_padded, output_padded, input_mask, task.task_id
        else:
            return input_padded, output_padded, input_mask

    def get_task(self, task_id: str) -> Optional[ARCTask]:
        """
        Get task by ID.
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_task_by_index(self, task_idx: int) -> ARCTask:
        """
        Get task by index.
        """
        return self.tasks[task_idx]


class ARCTaskDataset(Dataset):
    """
    Dataset that returns entire tasks (all examples together).
    >>> dataset = ARCTaskDataset('data/arc/training')
    >>> task_data = dataset[0]
    >>> print(task_data['train_inputs'].shape)  # (num_train, 30, 30)
    """
    def __init__(
        self,
        data_path: Union[str, Path],
        max_grid_size: int = 30,
        max_examples: int = 10
    ):
        self.data_path = Path(data_path)
        self.max_grid_size = max_grid_size
        self.max_examples = max_examples
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> List[ARCTask]:
        """Load all tasks."""
        tasks = []
        if self.data_path.is_file():
            task_id = self.data_path.stem
            with open(self.data_path, 'r') as f:
                task_data = json.load(f)
            tasks.append(ARCTask(task_id, task_data))
        elif self.data_path.is_dir():
            for json_file in sorted(self.data_path.glob('*.json')):
                task_id = json_file.stem
                with open(json_file, 'r') as f:
                    task_data = json.load(f)
                tasks.append(ARCTask(task_id, task_data))
        return tasks

    def _pad_grid(self, grid: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad grid and create mask."""
        h, w = grid.shape
        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int64)
        padded[:h, :w] = grid
        mask = np.zeros((self.max_grid_size, self.max_grid_size), dtype=bool)
        mask[:h, :w] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        """
        Get entire task.
        """
        task = self.tasks[idx]
        # training examples
        train_inputs, train_outputs, train_masks = [], [], []
        for inp, out in task.train_pairs[:self.max_examples]:
            inp_pad, inp_mask = self._pad_grid(inp)
            out_pad, _ = self._pad_grid(out)
            train_inputs.append(inp_pad)
            train_outputs.append(out_pad)
            train_masks.append(inp_mask)
        # test examples
        test_inputs, test_outputs, test_masks = [], [], []
        for inp, out in task.test_pairs[:self.max_examples]:
            inp_pad, inp_mask = self._pad_grid(inp)
            out_pad, _ = self._pad_grid(out)
            test_inputs.append(inp_pad)
            test_outputs.append(out_pad)
            test_masks.append(inp_mask)
        return {
            'task_id': task.task_id,
            'train_inputs': torch.stack(train_inputs),
            'train_outputs': torch.stack(train_outputs),
            'train_masks': torch.stack(train_masks),
            'test_inputs': torch.stack(test_inputs),
            'test_outputs': torch.stack(test_outputs),
            'test_masks': torch.stack(test_masks)
        }


def create_arc_dataloaders(
    train_path: Union[str, Path],
    eval_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    max_grid_size: int = 30,
    num_workers: int = 0,
    shuffle_train: bool = True
):
    """
    Create train and eval dataloaders for ARC.
    """
    train_dataset = ARCDataset(
        train_path,
        split='train',
        max_grid_size=max_grid_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    if eval_path is None:
        return train_loader
    eval_dataset = ARCDataset(
        eval_path,
        split='test',
        max_grid_size=max_grid_size
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, eval_loader
