import torch


class RecordSet:
    """
    A class to represent the historical record set of the Faster-GCG algorithm.
    It is necessary to avoid self loop in the optimization process.

    All tensors in the record set must be on the same device, with the same dtype and shape.
    """

    def __init__(self):
        """
        Initialize the RecordSet class.
        """
        pass

    def __contains__(self, item: torch.Tensor) -> bool:
        """
        Check if the item is in the record set.

        :param item: The tensor to check.
        :return: True if the item is in the record set, False otherwise.
        """
        raise NotImplementedError("RecordSet __contains__ method not implemented yet")

    def add(self, item: torch.Tensor) -> None:
        """
        Add an item to the record set.

        :param item: The tensor to add.
        """
        raise NotImplementedError("RecordSet add method not implemented yet")

    @staticmethod
    def _hash_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Hash the tensor to a semi-unique integer value.
        The computation will happen in the same device as the tensor.

        :param tensor: The tensor to hash.
        :return: The resulting hash value, on the same device as the input tensor.
        """
        raise NotImplementedError("RecordSet _hash_tensor method not implemented yet")
