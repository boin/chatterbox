# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass
from types import MethodType


logger = logging.getLogger(__name__)


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j-i)
        # self.alignment_bin = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        # Using `output_attentions=True` is incompatible with optimized attention kernels, so
        # using it for all layers slows things down too much. We can apply it to just one layer
        # by intercepting the kwargs and adding a forward hook (credit: jrm)
        self.last_aligned_attn = None
        self.hook_handle = None
        self.original_forward = None
        self.target_layer = None
        self._add_attention_spy(tfmr, alignment_layer_idx)
        
        # Memory management settings
        self.max_alignment_length = 10000  # Prevent unlimited growth
        self.cleanup_frequency = 100  # Clean up every N frames

    def _add_attention_spy(self, tfmr, alignment_layer_idx):
        """
        Adds a forward hook to a specific attention layer to collect outputs.
        Using `output_attentions=True` is incompatible with optimized attention kernels, so
        using it for all layers slows things down too much.
        (credit: jrm)
        """

        def attention_forward_hook(module, input, output):
            """
            See `LlamaAttention.forward`; the output is a 3-tuple: `attn_output, attn_weights, past_key_value`.
            NOTE:
            - When `output_attentions=True`, `LlamaSdpaAttention.forward` calls `LlamaAttention.forward`.
            - `attn_output` has shape [B, H, T0, T0] for the 0th entry, and [B, H, 1, T0+i] for the rest i-th.
            """
            step_attention = output[1].cpu() # (B, 16, N, N)
            # Clear old attention data to prevent accumulation
            if self.last_aligned_attn is not None:
                del self.last_aligned_attn
            self.last_aligned_attn = step_attention[0].mean(0) # (N, N)

        self.target_layer = tfmr.layers[alignment_layer_idx].self_attn
        self.hook_handle = self.target_layer.register_forward_hook(attention_forward_hook)

        # Backup original forward
        self.original_forward = self.target_layer.forward
        def patched_forward(self, *args, **kwargs):
            kwargs['output_attentions'] = True
            return self.original_forward(*args, **kwargs)

        # Store reference to restore later
        self.target_layer.forward = MethodType(patched_forward, self.target_layer)

    def cleanup(self):
        """
        Clean up hooks and restore original forward method to prevent memory leaks.
        """
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        
        if self.target_layer is not None and self.original_forward is not None:
            self.target_layer.forward = self.original_forward
            self.original_forward = None
            self.target_layer = None
        
        # Clear accumulated data
        if self.last_aligned_attn is not None:
            del self.last_aligned_attn
            self.last_aligned_attn = None
        
        # Clear alignment history beyond reasonable bounds
        if self.alignment.size(0) > self.max_alignment_length:
            # Keep only the most recent data
            keep_length = self.max_alignment_length // 2
            self.alignment = self.alignment[-keep_length:].clone()
            if self.started_at is not None:
                self.started_at = max(0, self.started_at - (self.alignment.size(0) - keep_length))
            if self.completed_at is not None:
                self.completed_at = max(0, self.completed_at - (self.alignment.size(0) - keep_length))

    def _manage_memory(self):
        """
        Periodically clean up memory to prevent unlimited growth.
        """
        if self.curr_frame_pos % self.cleanup_frequency == 0:
            if self.alignment.size(0) > self.max_alignment_length:
                # Keep only the most recent data
                keep_length = self.max_alignment_length // 2
                self.alignment = self.alignment[-keep_length:].clone()
                # Adjust tracking variables
                if self.started_at is not None:
                    self.started_at = max(0, self.started_at - (self.alignment.size(0) - keep_length))
                if self.completed_at is not None:
                    self.completed_at = max(0, self.completed_at - (self.alignment.size(0) - keep_length))
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def step(self, logits):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # Manage memory periodically
        self._manage_memory()
        
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        aligned_attn = self.last_aligned_attn # (N, N)
        if aligned_attn is None:
            # If no attention data available, return logits unchanged
            self.curr_frame_pos += 1
            return logits
            
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone().cpu() # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone().cpu() # (1, S)

        # TODO: monotonic masking; could have issue b/c spaces are often skipped.
        A_chunk[:, self.curr_frame_pos + 1:] = 0


        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not(-4 < cur_text_posn - self.text_position < 7) # NOTE: very lenient!
        if not discontinuity:
            self.text_position = cur_text_posn

        # Hallucinations at the start of speech show up as activations at the bottom of the attention maps!
        # To mitigate this, we just wait until there are no activations far off-diagonal in the last 2 tokens,
        # and there are some strong activations in the first few tokens.
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Is generation likely complete?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # NOTE: EOS rarely assigned activations, and second-last token is often punctuation, so use last 3 tokens.
        # NOTE: due to the false-start behaviour, we need to make sure we skip activations for the first few tokens.
        last_text_token_duration = A[15:, -3:].sum()

        # Activations for the final token that last too long are likely hallucinations.
        long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 10) # 400ms

        # If there are activations in previous tokens after generation has completed, assume this is a repetition error.
        repetition = self.complete and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        # If a bad ending is detected, force emit EOS by modifying logits
        # NOTE: this means logits may be inconsistent with latents!
        if long_tail or repetition:
            logger.warn(f"forcing EOS token, {long_tail=}, {repetition=}")
            # (±2**15 is safe for all dtypes >= 16bit)
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        # Suppress EoS to prevent early termination
        if cur_text_posn < S - 3: # FIXME: arbitrary
            logits[..., self.eos_idx] = -2**15

        self.curr_frame_pos += 1
        return logits
