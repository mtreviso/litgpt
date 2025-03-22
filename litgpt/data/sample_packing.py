"""
Sample packing implementation for efficient training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import partial
from typing import Dict, Any, List, Tuple, Optional, Union, Callable


class PackedDataset(Dataset):
    """
    Dataset that packs multiple sequences together to maximize GPU utilization.

    This creates efficient training samples by packing multiple sequences into
    a single sequence of the target length, with proper attention masking.
    """

    def __init__(
            self,
            base_dataset: Dataset,
            tokenizer,
            max_seq_length: int = 2048,
            pad_token_id: Optional[int] = None,
            packing_efficiency_factor: float = 1.2,
            max_samples_to_pack: int = 8,
            drop_last: bool = False,
    ):
        """
        Initialize the PackedDataset.

        Args:
            base_dataset: The base dataset providing sequences to pack
            tokenizer: The tokenizer used for the model
            max_seq_length: The target sequence length after packing
            pad_token_id: The token ID to use for padding (defaults to tokenizer.pad_token_id)
            packing_efficiency_factor: Scale target seq length by this factor when collecting samples
            max_samples_to_pack: Maximum number of sequences to pack into one sample
            drop_last: Whether to drop samples that don't fill to max_seq_length
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        self.packing_efficiency_factor = packing_efficiency_factor
        self.max_samples_to_pack = max_samples_to_pack
        self.drop_last = drop_last

        # Generate packed samples
        self.packed_samples = self._create_packed_samples()

    def _create_packed_samples(self) -> List[Dict[str, torch.Tensor]]:
        """
        Create packed samples from the base dataset.
        """
        packed_samples = []
        current_tokens = []
        current_positions = []
        current_length = 0
        sample_count = 0

        # Calculate efficient sequence length for collection
        collection_seq_length = int(self.max_seq_length * self.packing_efficiency_factor)

        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]

            # Extract tokens
            if isinstance(sample, dict) and "input_ids" in sample:
                tokens = sample["input_ids"]
            elif isinstance(sample, torch.Tensor):
                tokens = sample
            else:
                raise ValueError(f"Unsupported sample type: {type(sample)}")

            if isinstance(tokens, list):
                tokens = torch.tensor(tokens)

            # Skip sample if it's too long on its own
            if len(tokens) > self.max_seq_length:
                continue

            # If adding this sample would exceed collection length, create a packed sample
            if current_length + len(tokens) > collection_seq_length or sample_count >= self.max_samples_to_pack:
                if current_length > 0:  # Only create a sample if we have tokens
                    packed_sample = self._create_single_packed_sample(
                        current_tokens, current_positions, current_length
                    )
                    packed_samples.append(packed_sample)

                # Reset for new packed sample
                current_tokens = []
                current_positions = []
                current_length = 0
                sample_count = 0

            # Add this sample to the current pack
            current_tokens.append(tokens)
            current_positions.append(current_length)
            current_length += len(tokens)
            sample_count += 1

        # Handle any remaining tokens
        if current_length > 0 and (not self.drop_last or current_length >= self.max_seq_length):
            packed_sample = self._create_single_packed_sample(
                current_tokens, current_positions, current_length
            )
            packed_samples.append(packed_sample)

        return packed_samples

    def _create_single_packed_sample(
            self, tokens_list: List[torch.Tensor], positions: List[int], total_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        Create a single packed sample from a list of token sequences.
        """
        # Initialize tensors with pad tokens
        input_ids = torch.full((self.max_seq_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.bool)

        # Fill in the packed tokens
        token_idx = 0
        for i, (tokens, start_pos) in enumerate(zip(tokens_list, positions)):
            seq_len = len(tokens)

            # Only use tokens that fit in max_seq_length
            tokens_to_use = min(seq_len, self.max_seq_length - token_idx)
            if tokens_to_use <= 0:
                break

            # Copy tokens
            input_ids[token_idx:token_idx + tokens_to_use] = tokens[:tokens_to_use]

            # Set attention mask for this sequence
            for j in range(tokens_to_use):
                attention_mask[token_idx + j, token_idx:token_idx + j + 1] = True

            token_idx += tokens_to_use

            # Stop if we've filled the whole sequence
            if token_idx >= self.max_seq_length:
                break

        # Create labels (for LM training, labels = input_ids)
        labels = input_ids.clone()

        # Set padding positions to -100 (ignore in loss calculation)
        valid_mask = input_ids != self.pad_token_id
        labels[~valid_mask] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.packed_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.packed_samples[idx]


def create_packed_dataloader(
        dataset: Dataset,
        tokenizer,
        batch_size: int = 8,
        max_seq_length: int = 2048,
        shuffle: bool = True,
        num_workers: int = 4,
        drop_last: bool = False,
        packing_efficiency_factor: float = 1.2,
        max_samples_to_pack: int = 8,
        collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Create a DataLoader with packed samples for efficient training.

    Args:
        dataset: Base dataset to pack
        tokenizer: The tokenizer to use
        batch_size: Batch size for the dataloader
        max_seq_length: Maximum sequence length after packing
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for the dataloader
        drop_last: Whether to drop the last batch if it's incomplete
        packing_efficiency_factor: Scale factor for sample collection
        max_samples_to_pack: Maximum number of sequences to combine
        collate_fn: Optional custom collate function

    Returns:
        DataLoader with packed samples
    """
    packed_dataset = PackedDataset(
        base_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        packing_efficiency_factor=packing_efficiency_factor,
        max_samples_to_pack=max_samples_to_pack,
        drop_last=drop_last
    )

    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
