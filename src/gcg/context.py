from dataclasses import dataclass

import torch
import transformers

from src.gcg.record_set import RecordSet


@dataclass
class GCGRunContext:
    """
    This class is used to store the context of the GCG algorithm run.
    """

    """
    The model to be attacked, which should be a Hugging Face transformers model.
    """
    model: transformers.PreTrainedModel

    """
    The input tensor, which contains the input tokens to be attacked.
    This tensor should be of shape (input_len,).
    If None, the input of the model will be only the attack tokens to optimize.
    """
    x_fixed_input_ids: torch.Tensor | None

    """
    The input tokens, already embedded.
    This tensor should be of shape (input_len, embed_dim),
    and it follows the same logic as x_fixed_input_ids.
    """
    x_fixed_input_embed: torch.Tensor | None

    """
    The target tensor, which contains the target tokens to
    ideally have as the model output, given the attack tokens.
    This tensor should be of shape (output_len,).
    """
    y_target_output_ids: torch.Tensor

    """
    The target tokens, already embedded.
    This tensor should be of shape (output_len, embed_dim),
    and it follows the same logic as y_target_output_ids.
    """
    y_target_output_embed: torch.Tensor

    """
    The input tokens to be optimized, which are the adversarial tokens.
    This tensor should be of shape (attack_len,).
    """
    x_attack_token_ids: torch.Tensor

    """
    The historical record set of the Faster-GCG algorithm.
    It is necessary to avoid self loop in the optimization process.
    """
    record_set: RecordSet
