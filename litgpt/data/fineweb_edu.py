# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, Any, List, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from litgpt.data import DataModule, get_sft_collate_fn
from litgpt.tokenizer import Tokenizer
from litgpt.utils import chunked_cross_entropy


class FinewebPretrainingDataset(IterableDataset):
    """Dataset for pretraining on the Fineweb-Edu dataset using streaming mode."""

    def __init__(
            self,
            tokenizer: Tokenizer,
            dataset_name: str = "allenai/fineweb-edu",
            subset: Optional[str] = None,
            split: str = "train",
            max_seq_length: int = 4096,
            add_eos_token: bool = True,
            max_samples: Optional[int] = None,
            seed: int = 42,
            streaming: bool = True,
            shuffle: bool = False,
            preprocess_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_length = max_seq_length
        self.add_eos_token = add_eos_token
        self.max_samples = max_samples
        self.seed = seed
        self.streaming = streaming
        self.subset = subset
        self.preprocess_fn = preprocess_fn

        # Load dataset in streaming mode
        from datasets import load_dataset

        self.dataset = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=self.streaming,
        )

        # Shuffle with a buffer
        if shuffle:
            self.dataset = self.dataset.shuffle(
                seed=self.seed,
                buffer_size=10000,
            )

        # Apply custom preprocessing if provided
        if self.preprocess_fn:
            self.dataset = self.dataset.map(self.preprocess_fn)

        # Set counter for limiting samples if needed
        self.sample_counter = 0

    def __iter__(self):
        """Iterate through the dataset in streaming mode with optional sample limit."""
        self.sample_counter = 0

        for sample in self.dataset:
            if self.max_samples is not None and self.sample_counter >= self.max_samples:
                break

            # Skip empty examples
            if not sample.get("text", "").strip():
                continue

            self.sample_counter += 1

            # Get text data
            text = sample.get("text", "").strip()

            # Tokenize
            tokens = self.tokenizer.encode(text, eos=self.add_eos_token)

            # Truncate if necessary
            if 0 < self.max_seq_length < len(tokens):
                tokens = tokens[:self.max_seq_length]

                # If we're adding EOS, make sure the last token is EOS
                if self.add_eos_token:
                    tokens[-1] = self.tokenizer.eos_id

            # For pretraining, input_ids and labels are identical
            input_ids = torch.tensor(tokens, dtype=torch.long) if isinstance(tokens, list) else tokens
            labels = input_ids.clone()

            # Count tokens
            raw_count = len(tokens)

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "token_counts": {
                    "raw": raw_count,
                    "raw_plus_prompt_template": raw_count,  # No prompt template in pretraining
                },
            }


def get_pretrain_collate_fn(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100):
    """Returns the collate function for pretraining (needed in the DataLoader)."""
    return partial(_pretrain_collate_fn, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)


def _pretrain_collate_fn(
        samples: List[Dict[str, Union[torch.Tensor, Dict]]],
        max_seq_length: int = -1,
        pad_id: int = 0,
        ignore_index: int = -100
) -> Dict[str, Union[torch.Tensor, Dict]]:
    """Collate function that matches LitGPT's expected format for batches."""
    if not samples:
        return {}

    batched = {}
    for key in ("input_ids", "labels"):
        pad_value = pad_id if key == "input_ids" else ignore_index

        # Pad right based on the longest sequence
        batched[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in samples], batch_first=True, padding_value=pad_value
        )

        # Truncate if needed
        if max_seq_length is not None and max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length+1]

    # make sure labels are shifted by one
    batched["labels"] = batched["labels"][:, 1:]

    # adjust input_ids to have the same length as labels
    batched["input_ids"] = batched["input_ids"][:, :-1]

    # Handle token counts
    batched["token_counts"] = {}
    batched["token_counts"]["raw"] = torch.tensor(
        [sample["token_counts"]["raw"] for sample in samples], dtype=torch.int64
    ).unsqueeze(1)
    batched["token_counts"]["raw_plus_prompt_template"] = torch.tensor(
        [sample["token_counts"]["raw_plus_prompt_template"] for sample in samples], dtype=torch.int64
    ).unsqueeze(1)

    return batched


@dataclass
class FinewebEdu(DataModule):
    """Fineweb-Edu data module for pretraining."""

    dataset_name: str = "HuggingFace/fineweb-edu"
    """The Hugging Face dataset repository ID."""

    subset: Optional[str] = None
    """The subset of the dataset to use (e.g., "sample-10T")."""

    split: str = "train"
    """The dataset split to use."""

    val_split_fraction: float = 0.001
    """Fraction of the dataset to use for validation."""

    add_eos_token: bool = True
    """Whether to add an EOS token to each example."""

    seed: int = 42
    """Random seed for shuffling and splitting."""

    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""

    num_workers: int = 4
    """Number of workers for data loading."""

    streaming: bool = True
    """Whether to load the dataset in streaming mode."""

    max_dataset_length: Optional[int] = None
    """Maximum number of examples to load (useful for large datasets)."""

    preprocess_fn: Optional[Callable] = None
    """Custom preprocessing function to apply to each example."""

    use_sample_packing: bool = False
    """Whether to use sample packing to maximize GPU utilization."""

    packing_efficiency_factor: float = 1.2
    """Factor to scale target sequence length when collecting samples for packing."""

    max_samples_to_pack: int = 8
    """Maximum number of samples to pack into a single example."""

    access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))
    """The Hugging Face API token for accessing private datasets. Can be set via HF_TOKEN env var."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[Dataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[Dataset] = field(default=None, init=False, repr=False)

    def connect(
            self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def prepare_data(self) -> None:
        """Download the dataset if needed."""
        from datasets import load_dataset

        # Just check that the dataset exists
        load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=self.streaming,
            token=self.access_token,
        )

    def setup(self, stage: str = "") -> None:
        """Set up the dataset for training and validation."""
        # Create the base streaming dataset
        dataset = FinewebPretrainingDataset(
            tokenizer=self.tokenizer,
            dataset_name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            max_seq_length=self.max_seq_length,
            add_eos_token=self.add_eos_token,
            max_samples=self.max_dataset_length,
            seed=self.seed,
            streaming=self.streaming,
        )

        # For streaming datasets, we handle the val_split in the dataloader
        self.train_dataset = dataset

        # Create a separate validation dataset with different seed
        # Just a small fraction for validation
        val_size = int(self.max_dataset_length * self.val_split_fraction) if self.max_dataset_length else 1000

        self.val_dataset = FinewebPretrainingDataset(
            tokenizer=self.tokenizer,
            dataset_name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            max_seq_length=self.max_seq_length,
            add_eos_token=self.add_eos_token,
            max_samples=val_size,
            seed=self.seed + 1,  # Use different seed for validation
            streaming=self.streaming,
        )

        # Apply sample packing if requested
        if self.use_sample_packing:
            from litgpt.data.sample_packing import PackedDataset

            # We can only pack the dataset if it's not streaming
            # For streaming datasets, we'll need a different approach
            if not self.streaming:
                self.train_dataset = PackedDataset(
                    base_dataset=self.train_dataset,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_seq_length,
                    packing_efficiency_factor=self.packing_efficiency_factor,
                    max_samples_to_pack=self.max_samples_to_pack,
                )

                self.val_dataset = PackedDataset(
                    base_dataset=self.val_dataset,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_seq_length,
                    packing_efficiency_factor=self.packing_efficiency_factor,
                    max_samples_to_pack=self.max_samples_to_pack,
                )
            else:
                # For streaming, we'll handle packing in the DataLoader's collate function
                print("Warning: Sample packing with streaming datasets is not fully supported yet.")
                print("Using standard batching instead.")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Streaming dataset handles shuffling
            num_workers=self.num_workers,
            collate_fn=get_pretrain_collate_fn(
                max_seq_length=self.max_seq_length, pad_id=self.tokenizer.pad_id, ignore_index=self.ignore_index
            ),
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_pretrain_collate_fn(
                max_seq_length=self.max_seq_length, pad_id=self.tokenizer.pad_id, ignore_index=self.ignore_index
            ),
        )


# Utility function for preprocessing fineweb data
def fineweb_preprocessing(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Default preprocessing function for fineweb-edu dataset.

    This can be used as a preprocess_fn in the FinewebEdu data module.
    """
    # Ensure text field exists
    if "text" not in example:
        return {"text": ""}

    # Clean up text (remove excessive whitespace, etc.)
    text = example["text"].strip()

    # Return processed example
    return {"text": text}
