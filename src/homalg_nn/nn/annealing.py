import numpy as np
from typing import Tuple


class AnnealingScheduler:
    """
    > 'exponential': Smooth exponential transition (recommended)
    > 'cosine': Cosine annealing with smooth acceleration
    > 'linear': Linear interpolation
    > 'two_stage': Discrete two-stage approach
    > 'constant': No annealing (baseline)
    >>> scheduler = AnnealingScheduler(
    >>>     schedule='exponential',
    >>>     total_steps=2000,
    >>>     exactness_range=(0.1, 1.0),
    >>>     chain_axiom_range=(2.0, 0.5)
    >>> )
    >>> for step in range(2000):
    >>>     ex_w, ax_w = scheduler.get_weights(step)
    >>>     loss = ex_w * loss_exact + ax_w * loss_axiom
    >>>     # >>> training step
    """
    def __init__(
        self,
        schedule: str = 'exponential',
        total_steps: int = 2000,
        exactness_range: Tuple[float, float] = (0.1, 1.0),
        chain_axiom_range: Tuple[float, float] = (2.0, 0.5),
        two_stage_split: float = 0.5
    ):
        """
        Initialize annealing scheduler.
        """
        self.schedule = schedule
        self.total_steps = total_steps
        self.ex_start, self.ex_end = exactness_range
        self.ax_start, self.ax_end = chain_axiom_range
        self.two_stage_split = two_stage_split
        if schedule not in ['exponential', 'cosine', 'linear', 'two_stage', 'constant']:
            raise ValueError(f"Unknown schedule: {schedule}")

    def get_weights(self, step: int) -> Tuple[float, float]:
        """
        Get loss weights for current training step.
        """
        if step >= self.total_steps:
            return self.ex_end, self.ax_end
        progress = step / self.total_steps
        if self.schedule == 'constant':
            return self.ex_end, self.ax_end
        elif self.schedule == 'linear':
            exactness_weight = self.ex_start + (self.ex_end - self.ex_start) * progress
            chain_axiom_weight = self.ax_start + (self.ax_end - self.ax_start) * progress
        elif self.schedule == 'exponential':
            if self.ex_start > 0:
                ratio = self.ex_end / self.ex_start
                exactness_weight = self.ex_start * (ratio ** progress)
            else:
                exactness_weight = self.ex_end * progress
            if self.ax_end > 0:
                ratio = self.ax_end / self.ax_start
                chain_axiom_weight = self.ax_start * (ratio ** progress)
            else:
                chain_axiom_weight = self.ax_start * (1 - progress)
        elif self.schedule == 'cosine':
            cosine_factor = 0.5 * (1 - np.cos(np.pi * progress))
            exactness_weight = self.ex_start + (self.ex_end - self.ex_start) * cosine_factor
            chain_axiom_weight = self.ax_start + (self.ax_end - self.ax_start) * cosine_factor
        elif self.schedule == 'two_stage':
            split_step = int(self.two_stage_split * self.total_steps)
            if step < split_step:
                exactness_weight = 0.0
                chain_axiom_weight = self.ax_start
            else:
                exactness_weight = self.ex_end
                chain_axiom_weight = self.ax_end
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        return exactness_weight, chain_axiom_weight

    def get_schedule_info(self) -> str:
        """
        Get human-readable description of schedule.
        """
        lines = [
            f"Annealing Schedule: {self.schedule}",
            f"Total steps: {self.total_steps}",
            f"Exactness weight: {self.ex_start:.3f} -> {self.ex_end:.3f}",
            f"Chain axiom weight: {self.ax_start:.3f} -> {self.ax_end:.3f}"
        ]
        if self.schedule == 'two_stage':
            split_step = int(self.two_stage_split * self.total_steps)
            lines.append(f"Stage 1 (steps 0-{split_step}): Chain axiom only")
            lines.append(f"Stage 2 (steps {split_step}-{self.total_steps}): Balanced")
        return "\n".join(lines)


def create_recommended_scheduler(total_steps: int) -> AnnealingScheduler:
    return AnnealingScheduler(
        schedule='exponential',
        total_steps=total_steps,
        exactness_range=(0.1, 1.0),
        chain_axiom_range=(2.0, 0.5)
    )
