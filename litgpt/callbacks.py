"""Custom callbacks for LitGPT training."""

import math
import time
import logging
from typing import Any, Dict, Optional, Union

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.fabric.wrappers import _FabricModule


class AlphaSchedulerCallback(Callback):
    """
    Callback to adjust AdaSplash alpha parameter during training.

    This dynamically adjusts the alpha parameter in AdaSplash attention modules
    according to a schedule (linear, cosine, etc.).
    """

    def __init__(
            self,
            initial_alpha: float = 0.5,
            final_alpha: float = 2.0,
            max_steps: Optional[int] = None,
            strategy: str = "linear",
            power: float = 2.0,
            step_size: int = 1000,
            increment: float = 0.1,
            k: float = 0.1,
            layer_indices: Optional[list] = None,
    ):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.max_steps = max_steps  # Will be set in setup if None
        self.strategy = strategy
        self.power = power
        self.step_size = step_size
        self.increment = increment
        self.k = k
        self.layer_indices = layer_indices  # Specific layer indices to update
        self.scheduler = None  # Will be initialized in setup
        self.fabric = None
        self.current_step = 0

    def setup(self, trainer: L.Trainer, pl_module: Any, stage: str) -> None:
        """Initialize the scheduler during setup."""
        if self.max_steps is None:
            # Use trainer's max steps if not provided
            self.max_steps = trainer.max_steps
            if self.max_steps == -1:
                # If max_steps is not set in trainer, estimate from epochs
                steps_per_epoch = len(trainer.train_dataloader) // trainer.accumulate_grad_batches
                self.max_steps = steps_per_epoch * trainer.max_epochs
                logging.warning(f"Max steps not provided. Estimated: {self.max_steps}")

        # Set the initial alpha value in model
        self._update_model_alpha(pl_module, self.initial_alpha)

        # Store fabric reference for logging
        if hasattr(trainer, "fabric"):
            self.fabric = trainer.fabric

    def _update_model_alpha(self, model: Union[torch.nn.Module, _FabricModule], alpha: float) -> None:
        """Update the alpha parameter in AdaSplash attention modules."""
        # Handle fabric-wrapped models
        if isinstance(model, _FabricModule):
            model = model._module

        # Handle models wrapped in DDP or other wrappers
        if hasattr(model, "module"):
            model = model.module

        # Special case when model is directly a LitGPT model
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            blocks = model.transformer.h

            # Update alpha in specific layers or all layers
            layers_to_update = self.layer_indices if self.layer_indices is not None else range(len(blocks))

            update_count = 0
            for layer_idx in layers_to_update:
                if layer_idx >= len(blocks):
                    continue

                block = blocks[layer_idx]
                if hasattr(block, "attn") and hasattr(block.attn, "alpha"):
                    block.attn.alpha = alpha
                    update_count += 1

            if self.fabric and update_count > 0:
                self.fabric.print(f"Updated AdaSplash alpha to {alpha:.6f} in {update_count} layers")

    def on_train_batch_end(
            self, trainer: L.Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Update alpha after each training batch."""
        self.current_step += 1
        progress = min(self.current_step / self.max_steps, 1.0)

        # Calculate new alpha value based on schedule
        if self.strategy == "linear":
            # Linear annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        elif self.strategy == "exponential":
            # Exponential annealing
            new_alpha = self.initial_alpha * (self.final_alpha / self.initial_alpha) ** progress
        elif self.strategy == "cosine":
            # Cosine annealing
            new_alpha = self.final_alpha - (self.final_alpha - self.initial_alpha) * (
                    1 + math.cos(math.pi * progress)) / 2
        elif self.strategy == "polynomial":
            # Polynomial annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * (progress ** self.power)
        elif self.strategy == "stepwise":
            # Stepwise annealing
            new_alpha = self.initial_alpha + (self.current_step // self.step_size) * self.increment
            new_alpha = min(new_alpha, self.final_alpha)
        elif self.strategy == "sigmoid":
            # Sigmoid annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) / (
                    1 + math.exp(-self.k * (self.current_step - self.max_steps / 2)))
        else:
            raise ValueError(f"Unknown annealing strategy: {self.strategy}")

        # Update alpha and ensure it doesn't exceed final_alpha
        new_alpha = min(new_alpha, self.final_alpha)

        # Update model's alpha value
        self._update_model_alpha(pl_module, new_alpha)

        # Log alpha value
        if trainer.global_step % 10 == 0:  # Log every 10 steps
            metrics = {"adasplash/alpha": new_alpha}

            # Log to all loggers
            if hasattr(trainer, "loggers") and trainer.loggers:
                for logger in trainer.loggers:
                    logger.log_metrics(metrics, step=trainer.global_step)


class TokenMonitorCallback(Callback):
    """
    Callback that tracks the number of tokens processed during training.

    This helps monitor progress towards a token limit and provides ETA for completion.
    """

    def __init__(
            self,
            target_tokens: int = 10_000_000_000,  # 10B tokens
            log_interval: int = 100,
            time_window_size: int = 100,
    ):
        """
        Initialize the TokenMonitorCallback.

        Args:
            target_tokens: Target number of tokens to process during training
            log_interval: Log progress every N steps
            time_window_size: Window size for calculating token throughput
        """
        super().__init__()
        self.target_tokens = target_tokens
        self.log_interval = log_interval
        self.time_window_size = time_window_size

        # Initialize counters
        self.total_tokens = 0
        self.batch_sizes = []
        self.sequence_lengths = []
        self.step_times = []
        self.start_time = None

        # For calculating ETA
        self.tokens_per_second = 0
        self.last_log_time = 0

    def setup(self, trainer: L.Trainer, pl_module: Any, stage: str) -> None:
        """Setup callback when training begins."""
        self.start_time = time.time()
        self.fabric = trainer.fabric if hasattr(trainer, "fabric") else None

    def on_train_batch_end(
            self, trainer: L.Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Track tokens after each batch."""
        # Record timing
        now = time.time()
        step_time = now - (self.last_log_time or self.start_time)
        self.last_log_time = now
        self.step_times.append(step_time)

        # Keep the window size fixed
        if len(self.step_times) > self.time_window_size:
            self.step_times.pop(0)

        # Calculate tokens in this batch
        batch_size = batch["input_ids"].size(0) if isinstance(batch, dict) else batch[0].size(0)
        seq_length = batch["input_ids"].size(1) if isinstance(batch, dict) else batch[0].size(1)

        # Track batch and sequence information
        self.batch_sizes.append(batch_size)
        self.sequence_lengths.append(seq_length)

        # Keep window size fixed
        if len(self.batch_sizes) > self.time_window_size:
            self.batch_sizes.pop(0)
        if len(self.sequence_lengths) > self.time_window_size:
            self.sequence_lengths.pop(0)

        # Calculate tokens in this batch
        tokens_in_batch = batch_size * seq_length

        # Update total tokens
        self.total_tokens += tokens_in_batch

        # Log progress at intervals
        if trainer.global_step % self.log_interval == 0:
            # Calculate tokens per second based on recent batches
            avg_step_time = sum(self.step_times) / max(1, len(self.step_times))
            avg_batch_size = sum(self.batch_sizes) / max(1, len(self.batch_sizes))
            avg_seq_length = sum(self.sequence_lengths) / max(1, len(self.sequence_lengths))
            tokens_per_step = avg_batch_size * avg_seq_length

            self.tokens_per_second = tokens_per_step / max(0.1, avg_step_time)

            # Calculate ETA
            remaining_tokens = max(0, self.target_tokens - self.total_tokens)
            eta_seconds = remaining_tokens / max(1, self.tokens_per_second)

            # Convert ETA to a readable format
            eta_hours, remainder = divmod(eta_seconds, 3600)
            eta_minutes, eta_seconds = divmod(remainder, 60)
            eta_str = f"{int(eta_hours)}h {int(eta_minutes)}m {int(eta_seconds)}s"

            # Calculate progress percentage
            progress = min(100, (self.total_tokens / self.target_tokens) * 100)

            # Log metrics
            metrics = {
                "tokens/total": self.total_tokens,
                "tokens/percent_complete": progress,
                "tokens/tokens_per_second": self.tokens_per_second,
                "tokens/eta_seconds": eta_seconds,
            }

            # Log to all loggers
            if hasattr(trainer, "loggers") and trainer.loggers:
                for logger in trainer.loggers:
                    logger.log_metrics(metrics, step=trainer.global_step)

            # Print progress
            if self.fabric:
                self.fabric.print(
                    f"Tokens: {self.total_tokens:,}/{self.target_tokens:,} ({progress:.2f}%) | "
                    f"Speed: {self.tokens_per_second:.2f} tokens/s | "
                    f"ETA: {eta_str}"
                )

            # Check if we've reached the target
            if self.total_tokens >= self.target_tokens:
                if self.fabric:
                    self.fabric.print(f"Reached target of {self.target_tokens:,} tokens!")
                trainer.should_stop = True

    def on_train_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Final report when training ends."""
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        if self.fabric:
            self.fabric.print(
                f"Training completed! "
                f"Processed {self.total_tokens:,} tokens in {time_str} "
                f"({self.total_tokens / total_time:.2f} tokens/s average)"
            )
