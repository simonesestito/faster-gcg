import torch


class LossEarlyStop:
    def __init__(self, patience: int):
        """
        Initialize the LossEarlyStop class.

        :param patience: The number of steps with no improvement after which the procedure will be stopped.
        """
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __bool__(self) -> bool:
        """
        Check if the early stopping condition is met.

        :return: True if the early stopping condition is met, False otherwise.
        """
        return self.early_stop

    def step(self, loss: torch.Tensor) -> None:
        """
        Do another step of the procedure,
        and determine if the early stopping condition is met.

        :param loss: The current loss value.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter > self.patience:
            self.early_stop = True
