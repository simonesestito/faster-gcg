import torch


class TensorSet:
    """
    A class to represent the historical record set of the Faster-GCG algorithm.
    It is necessary to avoid self loop in the optimization process.

    All tensors in the record set must be on the same device, with the same dtype and shape.
    """

    def __init__(self, hidden_size: int, device: torch.device, dtype: torch.dtype, initial_capacity: int = 10):
        """
        Initialize the RecordSet class.

        :param hidden_size: The size of the hidden dimension.
        :param device: The device on which the tensors will be stored.
        :param dtype: The data type of the tensors.
        :param initial_capacity: The initial capacity of the record set.
        """
        assert initial_capacity > 0, "Initial capacity must be greater than 0"
        self.container = torch.zeros((initial_capacity, hidden_size), device=device, dtype=dtype)
        self.length = 0
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __contains__(self, item: torch.Tensor) -> bool:
        """
        Check if the item is in the record set.

        :param item: The tensor to check.
        :return: True if the item is in the record set, False otherwise.
        """
        assert item.shape == self.container.shape[1:], \
            f"Item shape {item.shape} does not match record set shape {self.container.shape[1:]}"

        if self.length == 0:
            return False

        # Parallel computation on the tensor device
        return (self.container[:self.length] == item).all(dim=1).any().item()

    @torch.no_grad()
    def add(self, item: torch.Tensor) -> None:
        """
        Add an item to the record set.

        :param item: The tensor to add.
        """
        # Check if we have enough capacity left in the container
        if self.length == self.container.size(0):
            # No capacity left, we need to double the size of the container
            new_container_capacity = torch.zeros_like(self.container)
            self.container = torch.cat([self.container, new_container_capacity], dim=0)

        # Add the item to the container.
        # We assume that the item is not already in the container,
        # which is fine for our use-case.
        self.container[self.length] = item
        self.length += 1

    def __len__(self) -> int:
        """
        Get the length of the record set.

        :return: The length of the record set.
        """
        return self.length
