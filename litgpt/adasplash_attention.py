"""AdaSplash attention implementation for LitGPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from litgpt.model import CausalSelfAttention

try:
    from adasplash import adasplash, adasplash_no_block_mask
except ImportError:
    raise ImportError("Please install adasplash first via `pip install adasplash`")


class AdaSplashCausalSelfAttention(CausalSelfAttention):
    """
    AdaSplash implementation of Causal Self Attention for LitGPT.
    Works as a drop-in replacement for the standard CausalSelfAttention.
    """

    def __init__(self, config, block_idx: int) -> None:
        super().__init__(config, block_idx)
        self.alpha = getattr(config, "adasplash_alpha", 1.5)
        self.block_sparse = getattr(config, "block_sparse", "auto")

    def scaled_dot_product_attention(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Calculate varlen for varying sequence lengths (needed by AdaSplash)
        if mask is not None:
            valid_mask = mask.eq(True) if mask.dtype == torch.bool else mask.eq(0)
            varlen = valid_mask.sum(-1).long().contiguous()
        else:
            B, T = q.shape[:2]
            varlen = torch.full((B,), T, dtype=torch.long, device=q.device)

        # Choose AdaSplash function based on alpha value
        if self.block_sparse == "auto":
            # Use block mask for alpha >= 1.5, which is more sparse and thus faster
            adasplash_fn = adasplash if self.alpha >= 1.5 else adasplash_no_block_mask
        else:
            adasplash_fn = adasplash if self.block_sparse else adasplash_no_block_mask

        # Apply AdaSplash attention (causal by default for LLM training)
        y = adasplash_fn(
            q, k, v,
            alpha=self.alpha,
            is_causal=True,
            varlen=varlen,
            niter=4,
        )
        return y.transpose(1, 2)
