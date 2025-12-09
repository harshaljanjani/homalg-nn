import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from examples.arc import (
    ARCChainSolver,
    BaselineARCSolver,
    ARCDataset
)
from homalg_nn.nn import create_recommended_scheduler
from homalg_nn.monitoring import HomologyMonitor
from homalg_nn.analysis import RepresentationMetrics


def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ARC-AGI solver')
    parser.add_argument('--no-chain', action='store_true', default=False, help='Disable chain complex (use baseline model)')
    parser.add_argument('--chain-dims', type=int, nargs='+', default=[16, 32, 64, 128, 256], help='Chain dimensions (default: [16,32,64,128,256])')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension (default: 512)')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension (default: 256)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--lambda-exact', type=float, default=0.5, help='Exactness loss weight (default: 0.5 from MNIST)')
    # data
    parser.add_argument('--train-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--val-path', type=str, default=None, help='Path to validation data')
    parser.add_argument('--max-grid-size', type=int, default=30, help='Maximum grid size (default: 30)')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--use-augmentation', action='store_true', default=True, help='Use data augmentation')
    parser.add_argument('--aug-prob', type=float, default=0.5, help='Augmentation probability')
    # monitoring and checkpointing
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval (steps)')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval (steps)')
    parser.add_argument('--save-dir', type=str, default='results/arc_training', help='Directory to save results')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Checkpoint interval (epochs)')
    # device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    return parser.parse_args()


class ARCTrainer:
    """
    Trainer for ARC solver with chain complexes.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if args.name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chain_str = 'chain' if not args.no_chain else 'baseline'
            self.exp_name = f'{chain_str}_{timestamp}'
        else:
            self.exp_name = args.name
        self.exp_dir = self.save_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        print(f"\nModel architecture:")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        total_steps = len(self.train_loader) * args.epochs
        self.annealing_scheduler = create_recommended_scheduler(total_steps)
        if not args.no_chain:
            self.homology_monitor = HomologyMonitor(
                log_interval=args.log_interval,
                verbose=True
            )
        else:
            self.homology_monitor = None
        self.metrics = RepresentationMetrics()
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_task_loss': [],
            'train_exact_loss': [],
            'train_axiom_loss': [],
            'val_loss': [],
            'betti_numbers': [],
            'learning_rate': []
        }

    def _create_model(self):
        """Create model based on args."""
        if not self.args.no_chain:
            model = ARCChainSolver(
                max_grid_size=self.args.max_grid_size,
                chain_dims=self.args.chain_dims,
                embed_dim=self.args.embed_dim,
                hidden_dim=self.args.hidden_dim,
                epsilon=1e-3,
                dtype=torch.float64
            )
        else:
            model = BaselineARCSolver(
                max_grid_size=self.args.max_grid_size,
                embed_dim=self.args.embed_dim,
                hidden_dim=self.args.hidden_dim,
                dtype=torch.float64
            )
        return model

    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        train_dataset = ARCDataset(
            self.args.train_path,
            split='train',
            max_grid_size=self.args.max_grid_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        print(f"\nData:")
        print(f"  Training examples: {len(train_dataset)}")
        print(f"  Batches per epoch: {len(train_loader)}")
        if self.args.val_path:
            val_dataset = ARCDataset(
                self.args.val_path,
                split='test',
                max_grid_size=self.args.max_grid_size
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            print(f"  Validation examples: {len(val_dataset)}")
        else:
            val_loader = None
        return train_loader, val_loader

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'task_loss': 0.0,
            'exact_loss': 0.0,
            'axiom_loss': 0.0
        }
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.args.epochs}')
        for _, (input_grids, output_grids, masks) in enumerate(pbar):
            input_grids = input_grids.to(self.device)
            output_grids = output_grids.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_grids, grid_mask=masks)
            # task loss (cross-entropy)
            loss_task = F.cross_entropy(
                logits.permute(0, 3, 1, 2),  # `(batch, 10, H, W)`
                output_grids,
                reduction='none'
            )
            loss_task = (loss_task * masks.float()).sum() / masks.sum()
            if not self.args.no_chain:
                ex_w, ax_w = self.annealing_scheduler.get_weights(self.global_step)
                loss_exact = self.model.compute_exactness_loss(mode='exactness')
                loss_axiom = self.model.compute_exactness_loss(mode='chain_axiom')
                loss_chain = ex_w * loss_exact + ax_w * loss_axiom
                loss = loss_task + self.args.lambda_exact * loss_chain
            else:
                loss_exact = torch.tensor(0.0)
                loss_axiom = torch.tensor(0.0)
                loss = loss_task
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['task_loss'] += loss_task.item()
            epoch_metrics['exact_loss'] += loss_exact.item()
            epoch_metrics['axiom_loss'] += loss_axiom.item()
            if not self.args.no_chain and self.global_step % self.args.log_interval == 0:
                self.homology_monitor.on_step(
                    self.global_step,
                    self.model.chain,
                    loss_exact,
                    loss_axiom
                )
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'task': f'{loss_task.item():.4f}',
                'exact': f'{loss_exact.item():.4f}' if not self.args.no_chain else 'N/A'
            })
            self.global_step += 1
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        return epoch_metrics

    def validate(self):
        """Validate on validation set."""
        if self.val_loader is None:
            return None
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_grids, output_grids, masks in self.val_loader:
                input_grids = input_grids.to(self.device)
                output_grids = output_grids.to(self.device)
                masks = masks.to(self.device)
                logits = self.model(input_grids, grid_mask=masks)
                loss = F.cross_entropy(
                    logits.permute(0, 3, 1, 2),
                    output_grids,
                    reduction='none'
                )
                loss = (loss * masks.float()).sum() / masks.sum()
                val_loss += loss.item()
        val_loss /= len(self.val_loader)
        return val_loss

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'args': vars(self.args)
        }
        checkpoint_path = self.exp_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.exp_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (val_loss={self.best_val_loss:.4f})")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training: {self.exp_name}")
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_loss = self.validate()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_task_loss'].append(train_metrics['task_loss'])
            self.history['train_exact_loss'].append(train_metrics['exact_loss'])
            self.history['train_axiom_loss'].append(train_metrics['axiom_loss'])
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
            if not self.args.no_chain:
                betti = self.model.get_betti_numbers()
                self.history['betti_numbers'].append(betti)
            print(f"\nEpoch {epoch+1}/{self.args.epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Task Loss:  {train_metrics['task_loss']:.4f}")
            if not self.args.no_chain:
                print(f"  Exact Loss: {train_metrics['exact_loss']:.4f}")
                print(f"  Axiom Loss: {train_metrics['axiom_loss']:.4f}")
                print(f"  Betti:      {betti}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(is_best=is_best)
        self._save_results()
        print(f"\nTraining complete!")
        print(f"Results saved to: {self.exp_dir}")

    def _save_results(self):
        """Save training results."""
        results = {
            'args': vars(self.args),
            'history': self.history,
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.history['train_loss'][-1],
                'total_steps': self.global_step
            }
        }
        results = convert_to_json_serializable(results)
        results_path = self.exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    args = parse_args()
    print("ARC-AGI Solver Training")
    print(f"  Device: {args.device}")
    print(f"  Model: {'Chain' if not args.no_chain else 'Baseline'}")
    if not args.no_chain:
        print(f"  Chain dims: {args.chain_dims}")
        print(f"  Lambda: {args.lambda_exact}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    trainer = ARCTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
