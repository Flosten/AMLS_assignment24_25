"""
Module for acquiring the BloodMNIST dataset

This file is used to acquire the BloodMNIST dataset from the medmnist library.

Functions:
    b_load_datasets(data_type: str) -> tuple: Load the BloodMNIST dataset
    b_train_image_acquire(): Acquire the training images and labels from the BloodMNIST dataset
    b_val_image_acquire(): Acquire the validation images and labels from the BloodMNIST dataset
    b_test_image_acquire(): Acquire the test images and labels from the BloodMNIST dataset
"""

import os

from medmnist import BloodMNIST


def b_load_datasets(data_type: str):
    """
    Load the BloodMNIST dataset

    Args:
        data_type (str): The type of the dataset to load (eg. train, val, test)

    Returns:
        tuple: A tuple containing the images and labels of the dataset
    """
    # Construct the dataset path
    dataset_path = os.path.join(".", "Datasets")
    # download the dataset
    trainset = BloodMNIST(root=dataset_path, split=data_type, download=True)

    # Load the specified part of the dataset
    images, labels = trainset.imgs, trainset.labels

    return images, labels


def b_train_image_acquire():
    """
    Acquire the training images and labels from the BloodMNIST dataset

    Returns:
        tuple: A tuple containing the training images and labels
    """
    return b_load_datasets("train")


def b_val_image_acquire():
    """
    Acquire the validation images and labels from the BloodMNIST dataset

    Returns:
        tuple: A tuple containing the validation images and labels
    """
    return b_load_datasets("val")


def b_test_image_acquire():
    """
    Acquire the test images and labels from the BloodMNIST dataset

    Returns:
        tuple: A tuple containing the test images and labels
    """
    return b_load_datasets("test")
