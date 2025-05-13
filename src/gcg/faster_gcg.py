import torch
import torch.nn.functional as F
import transformers

from .record_set import RecordSet
from .typedef import LossFunction

class FasterGCG:
    """
    Implementation of the faster GCG algorithm.
    Authors: https://arxiv.org/pdf/2410.15362v1
    """

    def __init__(self,
                 num_iterations: int,
                 batch_size: int,
                 loss_fn: LossFunction,
                 adversarial_tokens_length: int,
                 top_k_substitutions_length: int,
                 vocab_size: int,
                 ):
        """
        Initialize the FasterGCG class, with the default hyperparameters for the GCG algorithm.

        :param num_iterations: Number of iterations for the optimization process.
        :param batch_size: Batch size for the optimization process, also called the search window size.
        :param loss_fn: Loss function to be used for the optimization process.
        :param adversarial_tokens_length: Length of the adversarial tokens to be generated and optimized.
        :param top_k_substitutions_length: Length of the top-k substitutions to consider in the optimization process.
        :param vocab_size: Size of the vocabulary, used to generate one-hot tokens.
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

        for step_i in range(self.num_iterations):
            # Generate the attack input suffix to give to the model
            x_attack_tokens, loss = self._run_step(model, x_fixed_input, y_target_output, x_attack_tokens, record_set)

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


    def _run_step(self,
                  model: transformers.PreTrainedModel,
                  x_fixed_input: torch.Tensor | None,
                  y_target_output: torch.Tensor,
                  x_attack_tokens: torch.Tensor,
                  record_set: RecordSet,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single step of the GCG algorithm.

        :param model:
            The model to be attacked, which should be a Hugging Face transformers model.
        :param x_fixed_input:
            The input tensor, which contains the input tokens to be attacked.
            This tensor should be of shape (batch_size, seq_len).
            If None, the input of the model will be only the attack tokens to optimize.
        :param y_target_output:
            The target tokens to make the model give as output.
            This tensor should be of shape (batch_size, seq_len).
        :param x_attack_tokens:
            The attack suffix tokens to be optimized.
            This tensor should be of shape (batch_size, seq_len).
        :param record_set:
            The historical record set, to avoid self loop in the optimization process.

        :return:
            - The optimized attack suffix tokens, of shape (batch_size, seq_len).
            - The loss value, which is a scalar tensor.
        """
        # Compute the top-k substitutions for every token in the attack tokens.
        # TODO

        # Exploration part:
        # Populate the batch size with the substituted samples, according to the greedy sampling strategy.
        # TODO

        raise NotImplementedError("FasterGCG _run_step method not implemented yet")
