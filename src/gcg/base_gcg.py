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
import abc

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from .context import GCGRunContext
from .early_stop import LossEarlyStop
from .loss import loss_ce
from .tensor_set import TensorSet


class GCGAlgorithm(abc.ABC):
    """
    Abstract class to implement a GCG algorithm.
    """

    def __init__(self,
                 num_iterations: int,
                 batch_size: int,
                 adversarial_tokens_length: int,
                 top_k_substitutions_length: int,
                 vocab_size: int,
                 ):
        """
        Initialize the GCG class, with the default hyperparameters for the GCG algorithm.

        :param num_iterations: Number of iterations for the optimization process.
        :param batch_size: Batch size for the optimization process, also called the search window size.
        :param adversarial_tokens_length: Length of the adversarial tokens to be generated and optimized.
        :param top_k_substitutions_length: Length of the top-k substitutions to consider in the optimization process.
        :param vocab_size: Size of the vocabulary, used to generate one-hot tokens.
        """
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.adversarial_tokens_length = adversarial_tokens_length
        self.top_k_substitutions_length = top_k_substitutions_length
        self.vocab_size = vocab_size

    def tokenize_and_attack(self,
                            tokenizer: transformers.PreTrainedTokenizerBase,
                            model: transformers.PreTrainedModel,
                            x_fixed_input: str | None,
                            y_target_output: str,
                            show_progress: bool = False,
                            ) -> tuple[torch.Tensor, torch.Tensor, str, str, int]:
        """
        Tokenize the input and target output, and perform the attack on the model using the GCG algorithm.
        This function is the main entry point for the attack, when the input and target output are not already tokenized.

        :param tokenizer:
            The tokenizer to be used for the input and target output.
            This should be a Hugging Face transformers tokenizer.
        :param model:
            The model to be attacked, which should be a Hugging Face transformers model.
        :param x_fixed_input:
            The input string, which contains the input text to be used as a fixed prefix in the attack.
        :param y_target_output:
            The target string, which is what the model should output, ideally.
        :param show_progress:
            Whether to show the progress bar of the optimization process.
        :return:
            - The adversarial tokens, as tensor input IDs, which are the optimized tokens to be used for the attack.
              This tensor should be of shape (batch_size, seq_len).
            - The response of the model being attack, as tensor input IDs.
            - The adversarial suffix string, to be appended to the input string to perform the attack.
            - The response of the model being attacked, as string.
            - The number of iterations performed for the optimization process.
        """
        x_fixed_input_ids: torch.Tensor | None = None
        if x_fixed_input is not None:
            x_fixed_input_ids = tokenizer(x_fixed_input, return_tensors="pt").input_ids.to(model.device)

        y_target_output_ids = tokenizer(y_target_output, return_tensors="pt").input_ids.to(model.device)

        x_suffix_ids, steps = self.attack(
            model=model,
            x_fixed_input=x_fixed_input_ids,
            y_target_output=y_target_output_ids,
            show_progress=show_progress,
        )
        x_suffix = tokenizer.decode(x_suffix_ids.view(-1), skip_special_tokens=True)

        # Get the actual response of the model being attacked
        full_text = x_suffix_ids
        if x_fixed_input_ids is not None:
            full_text = torch.cat([x_fixed_input_ids, x_suffix_ids], dim=-1)

        with torch.no_grad():
            y_attack_response_ids = model.generate(
                input_ids=full_text.unsqueeze(0),
                max_new_tokens=y_target_output_ids.size(-1),
                do_sample=False,
            )[0][full_text.size(-1):]
            y_attack_response = tokenizer.decode(y_attack_response_ids, skip_special_tokens=True)

        return x_suffix_ids, y_attack_response_ids, x_suffix, y_attack_response, steps

    def _prepare_record_set(self, device: torch.device, dtype: torch.dtype) -> TensorSet | None:
        """
        Prepare the record set to keep track of the already seen attack tokens.
        This is used to avoid self-loops in the optimization process.

        :param device:
            The device to be used for the record set, usually the same as the model's device.
        :param dtype:
            The data type to be used for the record set, usually torch.int64.

        :return:
            A TensorSet object to keep track of the already seen attack tokens,
            or None if no record set is needed.
        """
        # By default, we do not use a record set, so we return None
        return None

    def attack(self,
               model: transformers.PreTrainedModel,
               x_fixed_input: torch.Tensor | None,
               y_target_output: torch.Tensor,
               show_progress: bool = False,
               ) -> tuple[torch.Tensor, int]:
        """
        Perform the attack on the model using the GCG algorithm.
        This function is the main entry point for the attack.

        :param model:
            The model to be attacked, which should be a Hugging Face transformers model.
        :param x_fixed_input:
            The input tensor, which contains the input tokens to be attacked.
            This tensor should be of shape (batch_size, seq_len).
            If None, the input of the model will be only the attack tokens to optimize.
        :param y_target_output:
            The target tensor, which contains the target tokens to be attacked.
        :param show_progress:
            Whether to show the progress bar of the optimization process.
        :return:
            - The adversarial tokens, as tensor input IDs, which are the optimized tokens to be used for the attack.
              This tensor should be of shape (batch_size, seq_len).
            - The number of iterations performed for the optimization process.
        """
        model_was_training = model.training
        model.eval()

        # Generate the attack input suffix to give to the model
        x_attack_token_ids = torch.randint(0, self.vocab_size, (self.adversarial_tokens_length,), device=model.device)

        run_context = GCGRunContext(
            model=model,
            x_fixed_input_ids=x_fixed_input,
            x_fixed_input_embed=model.get_input_embeddings()(x_fixed_input) if x_fixed_input is not None else None,
            y_target_output_ids=y_target_output,
            y_target_output_embed=model.get_input_embeddings()(y_target_output),
            x_attack_token_ids=x_attack_token_ids,
            record_set=self._prepare_record_set(model.device, x_attack_token_ids.dtype),
        )

        early_stop = LossEarlyStop(patience=10)

        step_i = 0
        iterations_range = range(self.num_iterations)
        if show_progress:
            iterations_range = tqdm(iterations_range, desc="Running GCG attack", unit="iteration")
        for step_i in iterations_range:
            # Generate the attack input suffix to give to the model
            loss = self._run_step(run_context)

            # Do early stopping
            early_stop.step(loss)
            if early_stop:
                break

        # Restore the model to its original state
        if model_was_training:
            model.train()

        # Return the optimized attack tokens
        return run_context.x_attack_token_ids, step_i + 1

    def _compute_attack_one_hot_grads(self, run_context: GCGRunContext) -> torch.Tensor:
        """
        Compute the gradients of the attack one-hot tensor with respect to the loss.

        :param run_context:
            The context of the GCG algorithm run, which contains the model, input and target tensors,
            and the historical record set.

        :return:
            The gradients of the attack one-hot tensor with respect to the loss,
            which is a tensor of shape (adversarial_tokens_length, vocab_size).
        """
        attack_one_hot = F.one_hot(run_context.x_attack_token_ids, num_classes=self.vocab_size).float().detach()
        attack_one_hot.requires_grad_(True)
        assert attack_one_hot.ndim == 2, \
            f'Expected attack_one_hot to be of shape ({self.adversarial_tokens_length}, {self.vocab_size}), but got {attack_one_hot.shape}'

        loss = self._compute_gcg_loss(attack_one_hot.unsqueeze(0), run_context)[0]

        attack_one_hot_grads = torch.autograd.grad(loss, attack_one_hot)[0]  # [adversarial_tokens_length, vocab_size]
        attack_one_hot_grads /= attack_one_hot_grads.norm(dim=-1, keepdim=True)
        return attack_one_hot_grads

    @abc.abstractmethod
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
        raise NotImplementedError(
            "This method should be implemented in the subclass to compute the top-k substitutions.")

    def _compute_gcg_loss(self, attack_one_hot: torch.Tensor, run_context: GCGRunContext) -> torch.Tensor:
        """
        Compute the GCG loss for the given attack one-hot tensor.

        :param attack_one_hot:
            The one-hot tensor of shape (batch_size, sequence_length, vocab_size).
            This tensor contains the attack tokens to be optimized.
        :param run_context:
            The context of the GCG algorithm run, which contains the model, input and target tensors,
            and the historical record set.
        :return:
            The loss value for the GCG algorithm, which is a scalar tensor.
        """
        assert attack_one_hot.ndim == 3, \
            f'Expected attack_one_hot to be batched ({attack_one_hot.ndim}D instead of 3D), got with actual shape {attack_one_hot.shape}'
        batch_size = attack_one_hot.size(0)
        assert attack_one_hot.shape == (batch_size, self.adversarial_tokens_length, self.vocab_size), \
            f'Expected attack_one_hot to be of shape (batch_size, {self.adversarial_tokens_length}, {self.vocab_size}), but got {attack_one_hot.shape}'

        attack_embed = F.linear(attack_one_hot, run_context.model.get_input_embeddings().weight.t())

        full_input_embed_list: list[torch.Tensor] = []
        if run_context.x_fixed_input_embed is not None:
            full_input_embed_list.append(run_context.x_fixed_input_embed)
        full_input_embed_list.append(attack_embed)
        full_input_embed_list.append(run_context.y_target_output_embed.repeat(batch_size, 1, 1))

        full_input_embed: torch.Tensor = torch.cat(full_input_embed_list, dim=1)

        # Compute the CE-loss, using teacher forcing.
        # The use of teacher forcing SHOULD be okay, since it is used in nanoGCG:
        # https://github.com/GraySwanAI/nanoGCG/blob/7d45952b0e75131025a44985a75306593e1bd69f/nanogcg/gcg.py#L478
        # From the logits, take only the ones that correspond to the attack target IDs
        output_logits = run_context.model(inputs_embeds=full_input_embed) \
                            .logits[:, -run_context.y_target_output_ids.size(-1) - 1:-1]

        per_sample_loss = loss_ce(output_logits, run_context.y_target_output_ids.repeat(batch_size, 1))
        assert per_sample_loss.shape == (batch_size,), \
            f'Expected per_sample_loss to be of shape (batch_size,), but got {per_sample_loss.shape}'

        return per_sample_loss

    @abc.abstractmethod
    def _run_step(self, run_context: GCGRunContext) -> torch.Tensor:
        """
        Run a single step of the GCG algorithm.

        :param run_context:
            The context of the GCG algorithm run, which contains the model, input and target tensors and more.

        :return:
            The loss value for the new optimized attack prompt, which is a scalar tensor.
        """
        raise NotImplementedError(
            "This method should be implemented in the subclass to run a single step of the GCG algorithm.")
