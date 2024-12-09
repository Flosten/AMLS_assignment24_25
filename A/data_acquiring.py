import os

import numpy as np


def A_load_datasets(fder_name: str, file_name: str, data_type: str):
    """
    General function to load different parts of the dataset (train, val, test).

    Args:
        fder_name (str): Name of the dataset folder("BreastMNIST").
        file_name (str): Name of the dataset file("breastmnist.npz").
        data_type (str): Data type (e.g., 'train', 'val', 'test').

    Returns:
        Returns the corresponding images and labels.
    """
    # Construct the dataset path
    dataset_path = os.path.join("Datasets", fder_name, file_name)
    data = np.load(dataset_path)

    # Load the specified part of the dataset
    images = data[f"{data_type}_images"]
    labels = data[f"{data_type}_labels"]

    return images, labels


# Wrappers for specific data types
def A_train_image_acquire(fder_name: str, file_name: str):
    """
    Acquire training images and labels.

    Args:
        fder_name (str): Name of the dataset folder.
        file_name (str): Name of the dataset file.

    Returns:
        Training images and labels.
    """
    return load_datasets(fder_name, file_name, "train")


def A_val_image_acquire(fder_name: str, file_name: str):
    """
    Acquire validation images and labels.

    Args:
        fder_name (str): Name of the dataset folder.
        file_name (str): Name of the dataset file.

    Returns:
        Validation images and labels.
    """
    return load_datasets(fder_name, file_name, "val")


def A_test_image_acquire(fder_name: str, file_name: str):
    """
    Acquire test images and labels.

    Args:
        fder_name (str): Name of the dataset folder.
        file_name (str): Name of the dataset file.

    Returns:
        Test images and labels.
    """
    return load_datasets(fder_name, file_name, "test")
