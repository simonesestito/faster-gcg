import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from .context import GCGRunContext
from .early_stop import LossEarlyStop
from .tensor_set import TensorSet
from .loss import loss_ce


class FasterGCG:
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
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.adversarial_tokens_length = adversarial_tokens_length
        self.top_k_substitutions_length = top_k_substitutions_length
        self.vocab_size = vocab_size
        self.lambda_reg_embeddings_distance = lambda_reg_embeddings_distance

        # Make greedy-sampling always possible
        max_batch_size = self.adversarial_tokens_length * (self.top_k_substitutions_length - 1)
        if self.batch_size > max_batch_size:
            raise ValueError(f'Batch size {self.batch_size} is too large for the given adversarial tokens length '
                             f'({self.adversarial_tokens_length}) and top-k substitutions length '
                             f'({self.top_k_substitutions_length}). The maximum batch size is {max_batch_size}.')


    def tokenize_and_attack(self,
                            tokenizer: transformers.PreTrainedTokenizerBase,
                            model: transformers.PreTrainedModel,
                            x_fixed_input: str | None,
                            y_target_output: str,
                            show_progress: bool = False,
                            ) -> tuple[str, str, int]:
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
            - The adversarial suffix string, to be appended to the input string to perform the attack.
            - The response of the model being attacked
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
            full_text = torch.cat([x_fixed_input_ids, x_suffix_ids], dim=1)

        with torch.no_grad():
            y_attack_response_ids = model.generate(
                input_ids=full_text,
                max_new_tokens=y_target_output_ids.size(-1),
                do_sample=False,
            )
            y_attack_response = tokenizer.decode(y_attack_response_ids[0], skip_special_tokens=True)

        return x_suffix, y_attack_response, steps


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

        # Create an empty historical record set, to avoid self loop
        record_set = TensorSet(hidden_size=self.adversarial_tokens_length,
                               device=x_attack_token_ids.device,
                               dtype=x_attack_token_ids.dtype)

        run_context = GCGRunContext(
            model=model,
            x_fixed_input_ids=x_fixed_input,
            x_fixed_input_embed=model.get_input_embeddings()(x_fixed_input) if x_fixed_input is not None else None,
            y_target_output_ids=y_target_output,
            y_target_output_embed=model.get_input_embeddings()(y_target_output),
            x_attack_token_ids=x_attack_token_ids,
            record_set=record_set,
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
        attack_one_hot = F.one_hot(run_context.x_attack_token_ids, num_classes=self.vocab_size).float().detach()
        attack_one_hot.requires_grad_(True)
        assert attack_one_hot.ndim == 2, \
            f'Expected attack_one_hot to be of shape ({self.adversarial_tokens_length}, {self.vocab_size}), but got {attack_one_hot.shape}'

        loss = self._compute_gcg_loss(attack_one_hot.unsqueeze(0), run_context)[0]

        attack_one_hot_grads = torch.autograd.grad(loss, attack_one_hot)[0]     # [adversarial_tokens_length, vocab_size]
        attack_one_hot_grads /= attack_one_hot_grads.norm(dim=-1, keepdim=True)

        # Add the regularization term to the gradients,
        # which is the distance between the token embeddings.
        # This means that attack_one_hot_grads[i, j] += lambda * || embedding_x[i] - embedding_j ||
        embeddings_j = run_context.model.get_input_embeddings().weight.data  # [vocab_size, embed_dim]
        embeddings_i = embeddings_j[run_context.x_attack_token_ids]          # [adversarial_tokens_length, embed_dim]
        embeddings_distance = torch.norm(embeddings_i[:, None] - embeddings_j[None, :], dim=-1)
        attack_one_hot_grads += self.lambda_reg_embeddings_distance * embeddings_distance

        # Now, for each token in the prefix, we need to find the top-k replacements with the lowest gradient
        best_replacements = torch.topk(attack_one_hot_grads, self.top_k_substitutions_length, dim=-1, largest=False).indices

        return best_replacements


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
        output_logits = run_context.model(inputs_embeds=full_input_embed)\
                            .logits[:, -run_context.y_target_output_ids.size(-1)-1:-1]

        per_sample_loss = loss_ce(output_logits, run_context.y_target_output_ids)
        assert per_sample_loss.shape == (batch_size,), \
            f'Expected per_sample_loss to be of shape (batch_size,), but got {per_sample_loss.shape}'

        return per_sample_loss


    def _run_step(self, run_context: GCGRunContext) -> torch.Tensor:
        """
        Run a single step of the GCG algorithm.

        :param run_context:
            The context of the GCG algorithm run, which contains the model, input and target tensors,
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
        b = 0   # Still, keep the first sample in the batch, which is the original one (last one)
        x_batch = run_context.x_attack_token_ids.repeat(self.batch_size + 1, 1)
        while b < self.batch_size:
            # Choose the token to replace, from 0 to N - 1 inclusive
            i = b % self.adversarial_tokens_length

            # Greedy sample from the top-k substitutions
            next_top_k_index = next_top_k_substitution_index[i]
            next_top_k_substitution_index[i] += 1

            # Perform the token replacement
            x_batch[b, i] = top_k_substitutions[i, next_top_k_index]

            # Avoid the self loop
            if x_batch[b] not in run_context.record_set:
                run_context.record_set.add(x_batch[b])
                b += 1

        # Finally, keep only the best X in x_batch
        x_batch_one_hot = F.one_hot(x_batch, num_classes=self.vocab_size).float().detach()
        loss_batch = self._compute_gcg_loss(x_batch_one_hot, run_context)
        best_x_index = torch.argmin(loss_batch)
        run_context.x_attack_token_ids = x_batch[best_x_index]

        return loss_batch[best_x_index]
