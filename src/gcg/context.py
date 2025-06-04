#  Copyright (c) 2025 Simone Sestito
#
#  This file is part of Faster-GGC-Lib.
#
#  Faster-GGC-Lib is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Faster-GGC-Lib is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Faster-GGC-Lib.  If not, see <http://www.gnu.org/licenses/>.

from dataclasses import dataclass

import torch
import transformers

from .tensor_set import TensorSet


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
    record_set: TensorSet | None
