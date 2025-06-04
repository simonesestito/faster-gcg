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


class StandardGCG(GCGAlgorithm):
    """
    Implementation of the standard original GCG algorithm.
    Authors: https://arxiv.org/pdf/2307.15043
    """

    def __init__(self,
                 num_iterations: int,
                 batch_size: int,
                 adversarial_tokens_length: int,
                 top_k_substitutions_length: int,
                 vocab_size: int,
                 ):
        """
        Initialize the FasterGCG class, with the default hyperparameters for the GCG algorithm.

        :param num_iterations: Number of iterations for the optimization process.
        :param batch_size: Batch size for the optimization process, also called the search window size.
        :param adversarial_tokens_length: Length of the adversarial tokens to be generated and optimized.
        :param top_k_substitutions_length: Length of the top-k substitutions to consider in the optimization process.
        :param vocab_size: Size of the vocabulary, used to generate one-hot tokens.
        """
        super().__init__(num_iterations, batch_size, adversarial_tokens_length, top_k_substitutions_length, vocab_size)

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

        # Now, for each token in the prefix, we need to find the top-k replacements with the lowest gradient
        best_replacements = torch.topk(attack_one_hot_grads, self.top_k_substitutions_length, dim=-1,
                                       largest=False).indices

        return best_replacements

    def _run_step(self, run_context: GCGRunContext) -> torch.Tensor:
        """
        Run a single step of the GCG algorithm.

        :param run_context:
            The context of the GCG algorithm run, which contains the model, input and target tensors.

        :return:
            The loss value for the new optimized attack prompt, which is a scalar tensor.
        """
        top_k_substitutions = self._compute_top_k_substitutions(run_context)
        assert top_k_substitutions.shape == (self.adversarial_tokens_length, self.top_k_substitutions_length), \
            f'Expected top_k_substitutions to be of shape ({self.adversarial_tokens_length}, ' \
            f'{self.top_k_substitutions_length}), but got {top_k_substitutions.shape}'

        attack_token_ids = run_context.x_attack_token_ids
        proposed_prefixes_one_hot = self._compute_random_replacements(attack_token_ids,
                                                                      top_k_substitutions).float().detach()

        with torch.no_grad():
            # Finally, keep only the best X in x_batch
            loss_batch = self._compute_gcg_loss(proposed_prefixes_one_hot, run_context)
            best_x_index = torch.argmin(loss_batch)
            best_x_one_hot = proposed_prefixes_one_hot[best_x_index]
            run_context.x_attack_token_ids = torch.argmax(best_x_one_hot, dim=-1) \
                .to(dtype=run_context.x_attack_token_ids.dtype)

        return loss_batch[best_x_index]

    @torch.no_grad()
    def _compute_random_replacements(self, attack_token_ids: torch.Tensor,
                                     top_k_substitutions: torch.Tensor) -> torch.Tensor:
        """
        Compute the random replacements for the attack input IDs.

        Args:
            attack_token_ids: Input IDs to be used for the attack.
            top_k_substitutions: Top-k substitutions for each token in the attack input IDs.

        Returns:
            torch.Tensor: The candidates prefixes with random replacements for each token.
        """
        # Convert the attack input IDs to one-hot encoding
        attack_one_hot = F.one_hot(attack_token_ids, num_classes=self.vocab_size).long()

        # Create the batch of candidates (the first one is supposed to stay the original attack input IDs)
        candidates = attack_one_hot.repeat(self.batch_size + 1, 1, 1)

        # For each candidate, pick a random token index to replace,
        # and then for every token, choose a random replacement from its top-k substitutions
        token_indexes = torch.randint(0, self.adversarial_tokens_length, (self.batch_size,),
                                      device=attack_token_ids.device)

        top_k_token_replacements = top_k_substitutions[token_indexes]
        top_k_random_replacement_indexes = torch.randint(0, self.top_k_substitutions_length, (self.batch_size,),
                                                         device=attack_token_ids.device)

        # Now, take replacement tokens from the top-k substitutions.
        # Replacement token for candidate i-th is the top-k substitution for token i, at index top_k_random_replacement[i]
        search_width_range = torch.arange(self.batch_size, device=attack_token_ids.device)
        top_k_random_replacement = top_k_token_replacements[search_width_range, top_k_random_replacement_indexes]
        one_hot_random_replacement = F.one_hot(top_k_random_replacement, self.vocab_size)
        candidates[search_width_range, token_indexes] = one_hot_random_replacement

        return candidates
