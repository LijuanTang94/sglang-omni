# SPDX-License-Identifier: Apache-2.0
"""Torch MPS runner for Whisper."""

from __future__ import annotations

import torch

from sglang_omni.model_runner.base import ModelRunner


class WhisperTorchMpsModelRunner(ModelRunner):
    """Whisper's Torch/MPS step, with grad held off across the whole step.

    Omni's scheduler loops do not carry SGLang's ``@DynamicGradMode()``, so
    without this every request retains its autograd graph: ~4.6 GB of live
    tensors per request for large-v3, which exhausts the MPS watermark after
    five. ``no_grad`` rather than ``inference_mode`` because this scope covers
    the sample-before-post block, and the sampler updates logits that
    ``inference_mode`` would not let it touch.
    """

    model_name = "Whisper"

    @torch.no_grad()
    def _prepare_and_forward(
        self,
        forward_batch,
        schedule_batch,
        requests,
        is_prefill,
        *,
        is_lookahead: bool = False,
    ):
        return super()._prepare_and_forward(
            forward_batch,
            schedule_batch,
            requests,
            is_prefill,
            is_lookahead=is_lookahead,
        )


__all__ = ["WhisperTorchMpsModelRunner"]
