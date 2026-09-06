# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Whisper.

Whisper is the first encoder-decoder model on this path, so it does not fit
SGLang's MLX cache discovery: that discovery classifies a layer by looking for
an attention module exposing ``("q_proj", "k_proj", "v_proj", "o_proj",
"rope")``, and Whisper has no RoPE — it uses learned absolute positions — and
names its output projection ``out_proj``. Discovery therefore finds no layers
and ``MlxModelRunner.__init__`` rejects the model. Declaring the layout up
front (see ``_declared_cache_layout``) skips the classifier entirely, so
nothing here has to pretend Whisper has a rotary embedding.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any
from unittest import mock

import mlx.core as mx

logger = logging.getLogger(__name__)

_RUNNER_MODULE = "sglang.srt.hardware_backend.mlx.model_runner"


def _whisper_attention_layout(model: Any) -> tuple[list[Any], list[str]]:
    """Report the decoder stack as one attention layer per decoder block.

    Each block also owns an ``encoder_attn``, but its keys and values live in
    the per-layer cross-attention cache rather than in SGLang's KV pool, so the
    pool only needs to size the self-attention half.
    """
    layers = list(model.model.decoder.layers)
    return layers, ["self_attn"] * len(layers)


@contextlib.contextmanager
def _declared_cache_layout():
    """Replace attention discovery for the duration of base ``__init__``.

    ``patch_model_attention`` is also disabled: it wraps each discovered
    attention in ``MLXAttentionWrapper`` for batched decode, which assumes the
    rotary, single-attention-per-layer shape Whisper does not have.
    """
    with (
        mock.patch(
            f"{_RUNNER_MODULE}.find_attention_layers",
            side_effect=_whisper_attention_layout,
        ),
        mock.patch(f"{_RUNNER_MODULE}.patch_model_attention", return_value=0),
    ):
        yield


class WhisperMlxModelRunner:
    """Whisper support layered on SGLang's native MLX model runner.

    The base runner keeps ownership of pool sizing, request bookkeeping and
    batched decode. This mixin supplies the model, the encoder-decoder cache
    shape, and the audio prefill; decode steps need no override because
    ``WhisperMlxModel.__call__`` decodes from tokens and a populated cache.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _declared_cache_layout():
            super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import WhisperMlxModel

        model_path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info("Loading native MLX Whisper model: %s", model_path)
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (WhisperMlxModel, ModelConfig),
        )
        logger.info(
            "Loaded native MLX Whisper model in %.2fs", time.perf_counter() - started
        )

    def _new_native_cache(self) -> list[Any]:
        """One self-attention cache plus one cross-attention cache per layer.

        The base implementation installs SGLang's pooled attention caches, which
        hold a single growing KV stream per layer and cannot represent the
        cross-attention half.
        """
        return self.model.make_cache()

    @staticmethod
    def _audio_item(req: Any) -> Any:
        mm_inputs = req.multimodal_inputs
        if mm_inputs is None:
            raise ValueError("Whisper MLX prefill requires multimodal inputs")
        if len(mm_inputs.mm_items) != 1:
            raise ValueError(
                "Whisper MLX prefill requires exactly one audio item, got "
                f"{len(mm_inputs.mm_items)}"
            )
        return mm_inputs.mm_items[0]

    @staticmethod
    def _to_numpy(tensor: Any) -> Any:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
            if str(tensor.dtype) == "torch.bfloat16":
                tensor = tensor.float()
            return tensor.numpy()
        return tensor

    def _decoder_prompt_ids(self, req: Any, token_ids: list[int]) -> list[int]:
        """Strip the encoder placeholder prefix from a request's input ids.

        The shared request builder emits ``[pad] * encoder_token_count`` ahead
        of the real decoder prompt so the CUDA path can reserve KV slots for the
        cross-attention entries. This path caches those keys and values itself,
        so the placeholders carry no state and must not be decoded.
        """
        item = self._audio_item(req)
        num_audio_tokens = getattr(item, "num_audio_tokens", None)
        if num_audio_tokens is None:
            num_audio_tokens = getattr(item, "num_image_tokens", None)
        if num_audio_tokens is None:
            raise ValueError("Whisper MLX prefill needs the encoder token count")
        num_audio_tokens = int(num_audio_tokens)
        if len(token_ids) <= num_audio_tokens:
            raise ValueError(
                f"Whisper MLX prefill got {len(token_ids)} input tokens, which "
                f"leaves no decoder prompt after {num_audio_tokens} encoder "
                "placeholders"
            )
        return list(token_ids[num_audio_tokens:])

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: Any = None,
    ):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingPrefill

        if req is None:
            raise ValueError("Whisper MLX prefill requires its scheduler request")
        if prefix_slot_ids:
            raise NotImplementedError(
                "Whisper MLX prefill does not support a radix prefix yet"
            )
        if not self.disable_radix_cache:
            raise RuntimeError("Whisper MLX requires disable_radix_cache=True")
        if logit_edit_row is not None or logprob_spec is not None:
            raise NotImplementedError(
                "Whisper MLX prefill supports greedy decoding only"
            )
        del new_slot_ids, needs_logits

        item = self._audio_item(req)
        if item.feature is None:
            raise ValueError("Whisper MLX prefill requires audio features")

        prompt_ids = self._decoder_prompt_ids(req, new_token_ids)
        encoder_hidden_states = self.model.encode(
            mx.array(self._to_numpy(item.feature))
        )

        cache = self._acquire_cache()
        # This call is what fills every layer's cross-attention cache; decode
        # steps after it reach cross-attention with tokens alone.
        logits = self.model.decode(
            mx.array([prompt_ids], dtype=mx.int32),
            encoder_hidden_states,
            cache=cache,
        )
        lazy_token = mx.argmax(logits[:, -1, :], axis=-1)
        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            full_token_ids=self._decoder_prompt_ids(req, full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
        )


def make_whisper_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class WhisperMlxRunner(WhisperMlxModelRunner, MlxModelRunner):
        pass

    return WhisperMlxRunner


__all__ = ["WhisperMlxModelRunner", "make_whisper_mlx_runner_class"]
