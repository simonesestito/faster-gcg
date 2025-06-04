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

import torch
import torch.nn.functional as F

from .base_gcg import GCGAlgorithm
from .context import GCGRunContext
from .tensor_set import TensorSet


class FasterGCG(GCGAlgorithm):
    """
    Implementation of the faster GCG algorithm.
    Authors: https://arxiv.org/pdf/2410.15362v1
    """

    def __init__(self,
                 num_iterations: int,
                 batch_size: int,
                 adversarial_tokens_length: int,
                 top_k_substitutions_length: int,
                 vocab_size: int,
                 lambda_reg_embeddings_distance: float = 4.0,
                 ):
        """
        Initialize the FasterGCG class, with the default hyperparameters for the GCG algorithm.

        :param num_iterations: Number of iterations for the optimization process.
        :param batch_size: Batch size for the optimization process, also called the search window size.
        :param adversarial_tokens_length: Length of the adversarial tokens to be generated and optimized.
        :param top_k_substitutions_length: Length of the top-k substitutions to consider in the optimization process.
        :param vocab_size: Size of the vocabulary, used to generate one-hot tokens.
        :param lambda_reg_embeddings_distance: Weight of the regularization term, which is the distance between token embeddings.
                                               The suggested value is 4.0, according to the paper.
        """
        super().__init__(
            num_iterations=num_iterations,
            batch_size=batch_size,
            adversarial_tokens_length=adversarial_tokens_length,
            top_k_substitutions_length=top_k_substitutions_length,
            vocab_size=vocab_size,
        )

        self.lambda_reg_embeddings_distance = lambda_reg_embeddings_distance

        # Make greedy-sampling always possible
        max_batch_size = self.adversarial_tokens_length * (self.top_k_substitutions_length - 1)
        if self.batch_size > max_batch_size:
            raise ValueError(f'Batch size {self.batch_size} is too large for the given adversarial tokens length '
                             f'({self.adversarial_tokens_length}) and top-k substitutions length '
                             f'({self.top_k_substitutions_length}). The maximum batch size is {max_batch_size}.')

    def _prepare_record_set(self, device: torch.device, dtype: torch.dtype) -> TensorSet | None:
        # Create an empty historical record set, to avoid self loop
        return TensorSet(hidden_size=self.adversarial_tokens_length,
                         device=device,
                         dtype=dtype)

    def _compute_top_k_substitutions(self, run_context: GCGRunContext) -> torch.Tensor:
        """
        First step of GCG: compute the top-k substitutions for each token in the attack input IDs.

        :param run_context:
            The context of the GCG algorithm run, which contains the model, input and target tensors,
            and the historical record set.

        :return:
            The top-k substitutions for each token in the attack input IDs,
            which is a tensor of shape (top_k_substitutions_length, vocab_size).
        """
        attack_one_hot_grads = self._compute_attack_one_hot_grads(run_context)

        # Add the regularization term to the gradients,
        # which is the distance between the token embeddings.
        # This means that attack_one_hot_grads[i, j] += lambda * || embedding_x[i] - embedding_j ||
        embeddings_j = run_context.model.get_input_embeddings().weight.data  # [vocab_size, embed_dim]
        embeddings_i = embeddings_j[run_context.x_attack_token_ids]  # [adversarial_tokens_length, embed_dim]
        embeddings_distance = torch.norm(embeddings_i[:, None] - embeddings_j[None, :], dim=-1)
        attack_one_hot_grads += self.lambda_reg_embeddings_distance * embeddings_distance

        # Now, for each token in the prefix, we need to find the top-k replacements with the lowest gradient
        best_replacements = torch.topk(attack_one_hot_grads, self.top_k_substitutions_length, dim=-1,
                                       largest=False).indices

        return best_replacements

    def _run_step(self, run_context: GCGRunContext) -> torch.Tensor:
        """
        Run a single step of the GCG algorithm.

        :param run_context:
            The context of the GCG algorithm run, which contains the model, input and target tensors
            and the historical record set.

        :return:
            The loss value for the new optimized attack prompt, which is a scalar tensor.
        """
        # Compute the top-k substitutions for every token in the attack tokens.
        top_k_substitutions = self._compute_top_k_substitutions(run_context)

        # To later perform the greedy sampling, keep a table to understand what's the next index j
        # to sample from the top-k substitutions, for each token i.
        next_top_k_substitution_index = torch.zeros((self.adversarial_tokens_length,), dtype=torch.int64)

        # Exploration part:
        # Populate the batch size with the substituted samples, according to the greedy sampling strategy.
        b = 0  # Still, keep the first sample in the batch, which is the original one (last one)
        x_batch = run_context.x_attack_token_ids.repeat(self.batch_size + 1, 1)
        while b < self.batch_size:
            # Choose the token to replace, from 0 to N - 1 inclusive
            i = b % self.adversarial_tokens_length

            # Greedy sample from the top-k substitutions
            next_top_k_index = next_top_k_substitution_index[i]
            next_top_k_substitution_index[i] += 1

            # Perform the token replacement
            # However, in some cases it may happen that next_top_k_index is out of bounds,
            # so we handle that by replacing it with the first one,
            # and ignoring the self loop check.
            is_valid_next_top_k_index = next_top_k_index < self.top_k_substitutions_length
            x_batch[b, i] = top_k_substitutions[i, next_top_k_index % self.top_k_substitutions_length]

            # Avoid the self loop
            if not is_valid_next_top_k_index or x_batch[b] not in run_context.record_set:
                run_context.record_set.add(x_batch[b])
                b += 1

        # Finally, keep only the best X in x_batch
        x_batch_one_hot = F.one_hot(x_batch, num_classes=self.vocab_size).float().detach()
        loss_batch = self._compute_gcg_loss(x_batch_one_hot, run_context)
        best_x_index = torch.argmin(loss_batch)
        run_context.x_attack_token_ids = x_batch[best_x_index]

        return loss_batch[best_x_index]
