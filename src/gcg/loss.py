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

from collections.abc import Callable

import torch
import torch.nn.functional as F

type LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def loss_ce(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Implement the batched Cross-Entropy loss function.

    :param input:
        The input tensor, which contains the output logits of the model,
        of shape (batch_size, seq_len, vocab_size).
    :param target:
        The target tensor, which is the true labels,
        of shape (batch_size, seq_len).
    :return:
        The loss value per sample in the batch, so the shape is (batch_size,).
    """
    assert input.ndim == 3, f"Expected input to be 3D, but got {input.ndim}D"
    assert target.ndim == 2, f"Expected target to be 2D, but got {target.ndim}D"

    batch_size = input.size(0)
    assert target.size(0) == batch_size, \
        f"Expected target to have the same batch size as input, but got {target.size(0)} vs {batch_size}"
    assert target.size(1) == input.size(1), \
        f"Expected target to have the same sequence length as input, but got {target.size(1)} vs {input.size(1)}"

    vocab_size = input.size(2)

    return F.cross_entropy(
        input=input.reshape(-1, vocab_size),
        target=target.view(-1),
        reduction='none',
    ).view(batch_size, -1).mean(dim=1)
