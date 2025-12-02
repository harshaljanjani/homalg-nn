from typing import Dict, List, Tuple, Optional, Any
import warnings


class HomologyMonitor:
    """
    Monitor homology evolution during training.
    > Betti numbers per degree
    > Exactness defects
    > Loss values (exactness, chain axiom, total)
    > Homology rank (sum of Betti numbers)
    >>> from homalg_nn.nn import ChainModule
    >>> from homalg_nn.monitoring import HomologyMonitor
    >>>
    >>> chain = ChainModule([5, 8, 10, 8, 5])
    >>> monitor = HomologyMonitor(log_interval=100)
    >>>
    >>> for step in range(1000):
    >>>     optimizer.zero_grad()
    >>>     loss = chain.compute_exactness_loss()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>
    >>>     monitor.on_step(step, chain, exactness_loss=loss.item())
    >>>
    >>> print(monitor.summary())
    >>> monitor.plot(save_path='training.png')
    """
    def __init__(
        self,
        log_interval: int = 100,
        compute_betti: bool = True,
        compute_defects: bool = True,
        log_to_tensorboard: bool = False,
        tensorboard_writer: Optional[Any] = None,
        log_to_wandb: bool = False,
        verbose: bool = True
    ):
        """
        Initialize HomologyMonitor.
        """
        self.log_interval = log_interval
        self.compute_betti = compute_betti
        self.compute_defects = compute_defects
        self.verbose = verbose
        self.history = {
            'steps': [],                    # List[int]
            'betti_numbers': [],            # List[Tuple[int, List[int]]]
            'exactness_defects': [],        # List[Tuple[int, List[float]]]
            'exactness_loss': [],           # List[Tuple[int, float]]
            'chain_axiom_loss': [],         # List[Tuple[int, float]]
            'total_loss': [],               # List[Tuple[int, float]]
            'homology_rank': [],            # List[Tuple[int, int]]
            'extra_metrics': {},            # Dict[str, List[Tuple[int, float]]]
        }
        self.tb_writer = None
        self.use_wandb = False
        if log_to_tensorboard:
            if tensorboard_writer is not None:
                self.tb_writer = tensorboard_writer
            else:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.tb_writer = SummaryWriter()
                except ImportError:
                    warnings.warn(
                        "tensorboard requested but torch.utils.tensorboard not available. "
                        "Install with: pip install tensorboard"
                    )
        if log_to_wandb:
            try:
                import wandb
                if wandb.run is None:
                    warnings.warn(
                        "wandb logging requested but wandb.init() not called. "
                        "Call wandb.init() before creating HomologyMonitor."
                    )
                else:
                    self.use_wandb = True
            except ImportError:
                warnings.warn(
                    "wandb requested but not installed. "
                    "Install with: pip install wandb"
                )

    def on_step(
        self,
        step: int,
        chain_module,
        exactness_loss: Optional[float] = None,
        chain_axiom_loss: Optional[float] = None,
        total_loss: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Callback to run after each training step.
        >>> for step in range(num_steps):
        >>>     optimizer.zero_grad()
        >>>     loss = chain.compute_exactness_loss()
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>
        >>>     monitor.on_step(
        >>>         step=step,
        >>>         chain_module=chain,
        >>>         exactness_loss=loss.item()
        >>>     )
        """
        if step % self.log_interval != 0:
            return
        # rec step
        self.history['steps'].append(step)
        # betti numbers
        if self.compute_betti:
            betti = chain_module.get_betti_numbers()
            rank = sum(betti)
            self.history['betti_numbers'].append((step, betti))
            self.history['homology_rank'].append((step, rank))
            # optional logging
            self._log_tensorboard('homology/rank', rank, step)
            self._log_wandb({'homology/rank': rank}, step)
            for i, b in enumerate(betti):
                self._log_tensorboard(f'homology/betti_{i}', b, step)
                self._log_wandb({f'homology/betti_{i}': b}, step)

        # exactness defects
        if self.compute_defects:
            defects = chain_module.get_exactness_defects()
            self.history['exactness_defects'].append((step, defects))

            for i, d in enumerate(defects):
                self._log_tensorboard(f'exactness/defect_{i}', d, step)
                self._log_wandb({f'exactness/defect_{i}': d}, step)

        # losses
        if exactness_loss is not None:
            self.history['exactness_loss'].append((step, exactness_loss))
            self._log_tensorboard('loss/exactness', exactness_loss, step)
            self._log_wandb({'loss/exactness': exactness_loss}, step)
        if chain_axiom_loss is not None:
            self.history['chain_axiom_loss'].append((step, chain_axiom_loss))
            self._log_tensorboard('loss/chain_axiom', chain_axiom_loss, step)
            self._log_wandb({'loss/chain_axiom': chain_axiom_loss}, step)
        if total_loss is not None:
            self.history['total_loss'].append((step, total_loss))
            self._log_tensorboard('loss/total', total_loss, step)
            self._log_wandb({'loss/total': total_loss}, step)
        if extra_metrics:
            for key, value in extra_metrics.items():
                if key not in self.history['extra_metrics']:
                    self.history['extra_metrics'][key] = []
                self.history['extra_metrics'][key].append((step, value))
                self._log_tensorboard(f'extra/{key}', value, step)
                self._log_wandb({f'extra/{key}': value}, step)
        if self.verbose:
            self._print_step(
                step, exactness_loss, chain_axiom_loss, total_loss,
                betti if self.compute_betti else None
            )

    def _log_tensorboard(self, tag: str, value: float, step: int):
        """Log to TensorBoard if available."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

    def _log_wandb(self, metrics: Dict[str, float], step: int):
        """Log to Wandb if available."""
        # TODO: stub for now.
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                pass

    def _print_step(
        self,
        step: int,
        exactness_loss: Optional[float],
        chain_axiom_loss: Optional[float],
        total_loss: Optional[float],
        betti: Optional[List[int]]
    ):
        """Print step information to console."""
        parts = [f"Step {step:5d}:"]
        if total_loss is not None:
            parts.append(f"Loss={total_loss:.4f}")
        if exactness_loss is not None:
            parts.append(f"Exact={exactness_loss:.4f}")
        if chain_axiom_loss is not None:
            parts.append(f"ChainAx={chain_axiom_loss:.6f}")
        if betti is not None:
            parts.append(f"Betti={betti}")
        print("  ".join(parts))

    def summary(self) -> str:
        lines = ["TRAINING SUMMARY"]
        # betti number evolution
        if self.history['betti_numbers']:
            _, initial_betti = self.history['betti_numbers'][0]
            _, final_betti = self.history['betti_numbers'][-1]
            initial_rank = sum(initial_betti)
            final_rank = sum(final_betti)
            lines.append(f"Initial Betti: {initial_betti} (rank={initial_rank})")
            lines.append(f"Final Betti:   {final_betti} (rank={final_rank})")
            lines.append(f"Rank reduction: {initial_rank} -> {final_rank}")
            # check if ever achieved exactness
            all_ranks = [sum(betti) for _, betti in self.history['betti_numbers']]
            if 0 in all_ranks:
                first_exact_idx = all_ranks.index(0)
                first_exact_step, _ = self.history['betti_numbers'][first_exact_idx]
                lines.append(f"First achieved exactness at step {first_exact_step}")
            lines.append("")
        # loss evolution
        if self.history['exactness_loss']:
            _, initial_loss = self.history['exactness_loss'][0]
            _, final_loss = self.history['exactness_loss'][-1]
            if initial_loss > 0:
                reduction = (initial_loss - final_loss) / initial_loss * 100
                lines.append(f"Exactness loss: {initial_loss:.4f} -> {final_loss:.4f}")
                lines.append(f"Reduction: {reduction:.1f}%")
            else:
                lines.append(f"Exactness loss: {initial_loss:.4f} -> {final_loss:.4f}")
            lines.append("")
        if self.history['chain_axiom_loss']:
            _, initial_ca = self.history['chain_axiom_loss'][0]
            _, final_ca = self.history['chain_axiom_loss'][-1]
            if initial_ca > 0:
                reduction = (initial_ca - final_ca) / initial_ca * 100
                lines.append(f"Chain axiom loss: {initial_ca:.6f} -> {final_ca:.6f}")
                lines.append(f"Reduction: {reduction:.1f}%")
            else:
                lines.append(f"Chain axiom loss: {initial_ca:.6f} -> {final_ca:.6f}")
            lines.append("")
        return "\n".join(lines)

    def plot(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[int, int]] = None
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting.")
        fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 8))

        # exactness loss
        if self.history['exactness_loss']:
            steps, losses = zip(*self.history['exactness_loss'])
            axes[0, 0].plot(steps, losses)
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Exactness Loss')
            axes[0, 0].set_title('Exactness Loss Over Time')
            axes[0, 0].grid(True)

        # chain axiom loss
        if self.history['chain_axiom_loss']:
            steps, losses = zip(*self.history['chain_axiom_loss'])
            axes[0, 1].plot(steps, losses)
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Chain Axiom Loss')
            axes[0, 1].set_title('Chain Axiom Loss Over Time')
            axes[0, 1].grid(True)

        # total loss
        if self.history['total_loss']:
            steps, losses = zip(*self.history['total_loss'])
            axes[1, 0].plot(steps, losses)
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Total Loss')
            axes[1, 0].set_title('Total Loss Over Time')
            axes[1, 0].grid(True)

        # homology rank (with betti markers)
        if self.history['homology_rank']:
            steps, ranks = zip(*self.history['homology_rank'])
            axes[1, 1].plot(steps, ranks)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Total Homology Rank')
            axes[1, 1].set_title('Betti Number Sum')
            axes[1, 1].set_ylim(bottom=-0.5)
            # annotate when rank hits 0 for first time.
            if 0 in ranks:
                first_zero_idx = ranks.index(0)
                first_zero_step = steps[first_zero_idx]
                axes[1, 1].axvline(
                    first_zero_step, color='r', linestyle='--', alpha=0.7,
                    label=f'First Exact (step {first_zero_step})'
                )
                axes[1, 1].legend()
            axes[1, 1].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
