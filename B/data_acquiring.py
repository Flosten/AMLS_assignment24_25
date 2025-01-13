import os

import numpy as np
from medmnist import BloodMNIST


def B_load_datasets(data_type: str):

    # Construct the dataset path
    dataset_path = os.path.join(".", "Datasets")
    # download the dataset
    trainset = BloodMNIST(root=dataset_path, split=data_type, download=True)

    # Load the specified part of the dataset
    images, labels = trainset.imgs, trainset.labels

    return images, labels


# Wrappers for specific data types
def B_train_image_acquire():

    return B_load_datasets("train")


def B_val_image_acquire():

    return B_load_datasets("val")


def B_test_image_acquire():

    return B_load_datasets("test")
