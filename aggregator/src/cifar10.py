from logging import getLogger

import numpy as np
from keras.datasets import cifar10
from PIL import Image

logger = getLogger(__name__)


def one_hot(labels, classes):
    """
    One Hot encode a vector.

    Args:
        labels (list):  List of labels to onehot encode
        classes (int): Total number of categorical classes

    Returns:
        np.array: Matrix of one-hot encoded labels
    """
    return np.eye(classes)[labels]


def _load_raw_datashards(shard_num, collaborator_count):
    """
    Load the raw data by shard.

    Returns tuples of the dataset shard divided into training and validation.

    Args:
        shard_num (int): The shard number to use
        collaborator_count (int): The number of collaborators in the federation

    Returns:
        2 tuples: (image, label) of the training, validation dataset
    """
    (X_train_tot, y_train_tot), (X_valid_tot, y_valid_tot) = cifar10.load_data()

    gray_images = []
    for image in X_train_tot:
        # Convert the image to grayscale using PIL
        gray_image = Image.fromarray(image).convert('L')

        # Append the grayscale image to the list
        gray_images.append(np.array(gray_image))
    X_train_tot = np.array(gray_images)

    gray_images_v = []
    for image in X_valid_tot:
        # Convert the image to grayscale using PIL
        gray_image = Image.fromarray(image).convert('L')
        # Append the grayscale image to the list
        gray_images_v.append(np.array(gray_image))

    X_valid_tot = np.array(gray_images_v)

    # create the shards
    shard_num = int(shard_num)
    X_train = X_train_tot[shard_num::collaborator_count]
    y_train = y_train_tot[shard_num::collaborator_count]

    X_valid = X_valid_tot[shard_num::collaborator_count]
    y_valid = y_valid_tot[shard_num::collaborator_count]

    X_train = np.resize(X_train, (X_train.shape[0], 28, 28, 1))
    X_valid = np.resize(X_valid, (X_valid.shape[0], 28, 28, 1))

    return (X_train, y_train), (X_valid, y_valid)


def load_cifar10_shard(shard_num, collaborator_count, categorical=True,
                     channels_last=True, **kwargs):
    """
    Load the MNIST dataset.

    Args:
        shard_num (int): The shard to use from the dataset
        collaborator_count (int): The number of collaborators in the federation
        categorical (bool): True = convert the labels to one-hot encoded
         vectors (Default = True)
        channels_last (bool): True = The input images have the channels
         last (Default = True)
        **kwargs: Additional parameters to pass to the function

    Returns:
        list: The input shape
        int: The number of classes
        numpy.ndarray: The training data
        numpy.ndarray: The training labels
        numpy.ndarray: The validation data
        numpy.ndarray: The validation labels
    """
    img_rows, img_cols = 28, 28
    num_classes = 10


    (X_train, y_train), (X_valid, y_valid) = _load_raw_datashards(
        shard_num, collaborator_count
    )

    if channels_last:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    else:
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_valid /= 255

    logger.info(f'CIFAR10 > X_train Shape : {X_train.shape}')
    logger.info(f'CIFAR10 > y_train Shape : {y_train.shape}')
    logger.info(f'CIFAR10 > Train Samples : {X_train.shape[0]}')
    logger.info(f'CIFAR10 > Valid Samples : {X_valid.shape[0]}')

    if categorical:
        # convert class vectors to binary class matrices
        y_train = one_hot(y_train.reshape((y_train.shape[0],)), num_classes)
        y_valid = one_hot(y_valid.reshape((y_valid.shape[0],)), num_classes)

    return input_shape, num_classes, X_train, y_train, X_valid, y_valid
