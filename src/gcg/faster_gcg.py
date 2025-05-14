import torch
import torch.nn.functional as F
import transformers

from .context import GCGRunContext
from .record_set import RecordSet
from .loss import LossFunction, loss_cw

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
                 loss_fn: LossFunction = loss_cw,
                 ):
        """
        Initialize the FasterGCG class, with the default hyperparameters for the GCG algorithm.

        :param num_iterations: Number of iterations for the optimization process.
        :param batch_size: Batch size for the optimization process, also called the search window size.
        :param adversarial_tokens_length: Length of the adversarial tokens to be generated and optimized.
        :param top_k_substitutions_length: Length of the top-k substitutions to consider in the optimization process.
        :param vocab_size: Size of the vocabulary, used to generate one-hot tokens.
        :param loss_fn: Loss function to be used for the optimization process.
        """
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.adversarial_tokens_length = adversarial_tokens_length
        self.top_k_substitutions_length = top_k_substitutions_length
        self.vocab_size = vocab_size


    def tokenize_and_attack(self,
                            tokenizer: transformers.PreTrainedTokenizerBase,
                            model: transformers.PreTrainedModel,
                            x_fixed_input: str,
                            y_target_output: str,
                            ) -> str:
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
        :return:
            The adversarial suffix string, to be appended to the input string to perform the attack.
        """
        x_fixed_input_ids = tokenizer(x_fixed_input, return_tensors="pt").input_ids.to(model.device)
        y_target_output_ids = tokenizer(y_target_output, return_tensors="pt").input_ids.to(model.device)

        x_suffix_ids = self.attack(
            model=model,
            x_fixed_input=x_fixed_input_ids,
            y_target_output=y_target_output_ids,
        )
        x_suffix = tokenizer.decode(x_suffix_ids.view(-1), skip_special_tokens=True)
        return x_suffix


    def attack(self,
               model: transformers.PreTrainedModel,
               x_fixed_input: torch.Tensor | None,
               y_target_output: torch.Tensor,
               ) -> torch.Tensor:
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
        :return:
            The adversarial tokens, which are the optimized tokens to be used for the attack.
            This tensor should be of shape (batch_size, seq_len).
        """
        # Generate the attack input suffix to give to the model
        x_attack_tokens = self._generate_one_hot_tokens(sequence_length=self.vocab_size)

        # Create an empty historical record set, to avoid self loop
        record_set = RecordSet()

        run_context = GCGRunContext(
            model=model,
            x_fixed_input_ids=x_fixed_input,
            x_fixed_input_embed=model.get_input_embeddings()(x_fixed_input),
            y_target_output_ids=y_target_output,
            y_target_output_embed=model.get_input_embeddings()(y_target_output),
            x_attack_token_ids=x_attack_tokens,
            record_set=record_set,
        )

        for step_i in range(self.num_iterations):
            # Generate the attack input suffix to give to the model
            loss = self._run_step(run_context)

            # TODO: implement early stopping on loss convergence



    @torch.no_grad()
    def _generate_one_hot_tokens(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate one-hot tokens for a random sample, of the given sequence length.
        The one-hot sample is not batched.

        :param sequence_length: Length of the sequence to be generated.
        :param device: The device to generate the one-hot sample on.
        :return: The one-hot sample, of shape (sequence_length, vocab_size).
        """
        input_ids = torch.randint(0, self.vocab_size, (sequence_length,), device=device)
        return F.one_hot(input_ids, num_classes=self.vocab_size)


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
        assert attack_one_hot.ndim == 2 or attack_one_hot.size(0) == 1, \
            f'Expected attack_one_hot to be of shape (1, {self.adversarial_tokens_length}, {self.vocab_size}), but got {attack_one_hot.shape}'

        loss = self._compute_gcg_loss(attack_one_hot, run_context)[0]

        attack_one_hot_grads = torch.autograd.grad(loss, attack_one_hot)[0][0]
        attack_one_hot_grads /= attack_one_hot_grads.norm(dim=-1, keepdim=True)

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
            f'Expected attack_one_hot to be batched'
        batch_size = attack_one_hot.size(0)
        assert attack_one_hot.shape == (batch_size, self.adversarial_tokens_length, self.vocab_size), \
            f'Expected attack_one_hot to be of shape (batch_size, {self.adversarial_tokens_length}, {self.vocab_size}), but got {attack_one_hot.shape}'

        attack_embed = F.linear(attack_one_hot, run_context.model.get_input_embeddings().weight)
        full_input_embed = torch.cat([
            run_context.x_fixed_input_embed,
            attack_embed,
            run_context.y_target_output_embed
        ], dim=1)

        # Compute the CE-loss, using teacher forcing.
        # The use of teacher forcing SHOULD be okay, since it is used in nanoGCG:
        # https://github.com/GraySwanAI/nanoGCG/blob/7d45952b0e75131025a44985a75306593e1bd69f/nanogcg/gcg.py#L478
        # From the logits, take only the ones that correspond to the attack target IDs
        output_logits = run_context.model(inputs_embeds=full_input_embed)\
                            .logits[:, -run_context.y_target_output_ids.size(-1)-1:-1]

        per_sample_loss = self.loss_fn(output_logits, run_context.y_target_output_ids)
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

        # Exploration part:
        # Populate the batch size with the substituted samples, according to the greedy sampling strategy.
        # TODO

        raise NotImplementedError("FasterGCG _run_step method not implemented yet")
