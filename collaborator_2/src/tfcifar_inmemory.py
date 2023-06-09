from openfl.federated import TensorFlowDataLoader
from .cifar10 import load_cifar10_shard


class TensorFlowCIFARInMemory(TensorFlowDataLoader):
    """TensorFlow Data Loader for CIFAR Dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # Then we have a way to automatically shard based on rank and size of
        # collaborator list.

        _, num_classes, X_train, y_train, X_valid, y_valid = load_cifar10_shard(
            shard_num=int(data_path), **kwargs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes
