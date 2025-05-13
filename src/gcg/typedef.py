from collections.abc import Callable

import torch

type LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
