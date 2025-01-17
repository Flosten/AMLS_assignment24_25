"""
Module for acquiring the BreastMNIST dataset

This file is used to acquire the BreastMNIST dataset from the medmnist library.

Functions:
    a_load_datasets(data_type: str) -> tuple: Load the BreastMNIST dataset
    a_train_image_acquire(): Acquire the training images and labels from the BreastMNIST dataset
    a_val_image_acquire(): Acquire the validation images and labels from the BreastMNIST dataset
    a_test_image_acquire(): Acquire the test images and labels from the BreastMNIST dataset
"""

import os

from medmnist import BreastMNIST


def a_load_datasets(data_type: str) -> tuple:
    """
    Load the BreastMNIST dataset

    Args:
        data_type (str): The type of the dataset to load (eg. train, val, test)

    Returns:
        tuple: A tuple containing the images and labels of the dataset
    """
    # Construct the dataset path
    dataset_path = os.path.join(".", "Datasets")
    # download the dataset
    trainset = BreastMNIST(root=dataset_path, split=data_type, download=True)

    # Load the specified part of the dataset
    images, labels = trainset.imgs, trainset.labels

    return images, labels


def a_train_image_acquire():
    """
    Acquire the training images and labels from the BreastMNIST dataset

    Returns:
        tuple: A tuple containing the training images and labels
    """
    return a_load_datasets("train")


def a_val_image_acquire():
    """
    Acquire the validation images and labels from the BreastMNIST dataset

    Returns:
        tuple: A tuple containing the validation images and labels
    """
    return a_load_datasets("val")


def a_test_image_acquire():
    """
    Acquire the test images and labels from the BreastMNIST dataset

    Returns:
        tuple: A tuple containing the test images and labels
    """
    return a_load_datasets("test")
