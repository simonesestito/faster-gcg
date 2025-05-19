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
