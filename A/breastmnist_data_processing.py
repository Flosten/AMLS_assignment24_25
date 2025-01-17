"""
Module for processing the BreastMNIST dataset

This file is used for preprocessing images in the BreastMNIST dataset and 
building an SVM model for breast cancer image classification.

Functions:
    Feature extraction for breast cancer images
    Train the SVM model
    Train the SVM model using cross-validation
    Test the SVM model
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from A.breastmnist_result_visualising import model_evaluation


def image_preprocess(images: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Extract images features.

    Args:
        images (np.ndarray): Input images set.
        labels (np.ndarray): Input labels set.

    Returns:
        tuple: Image and label set after feature extraction.
    """
    images_dim_1 = images.reshape(images.shape[0], -1)
    labels_dim_1 = labels.flatten()

    return images_dim_1, labels_dim_1


def svm_train(
    images_train: np.ndarray,
    labels_train: np.ndarray,
    images_val: np.ndarray,
    labels_val: np.ndarray,
    kernel: str,
    regularization: float,
) -> tuple:
    """
    Train the SVM model and return the trained model along with the scaler.

    Args:
        images_train (np.ndarray): Input images training set.
        labels_train (np.ndarray): Corresponding labels for the training set.
        images_val (np.ndarray): Input images validation set.
        labels_val (np.ndarray): Corresponding labels for the validation set.
        kernel (str): The kernel type to be used in the SVM model.
        regularization (float): The regularization parameter for the SVM model.

    Returns:
        tuple:
            - sklearn.svm.SVC:
                A trained SVM.SVC object.
            - sklearn.preprocessing.StandardScaler:
                The StandardScaler used to standardize the image data.
    """
    # standardize the data
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(images_train)

    # train the model
    svm_model = svm.SVC(
        kernel=kernel,
        C=regularization,
        class_weight="balanced",  # deal with the imbalanced dataset
        tol=0.001,
        random_state=711,
    )
    svm_model.fit(images_train_std, labels_train)

    # evaluate the model
    images_val_std = scaler.transform(images_val)
    label_pred = svm_model.predict(images_val_std)
    accuracy = accuracy_score(labels_val, label_pred)

    # print the accuracy on the training set
    train_label_pred = svm_model.predict(images_train_std)
    train_accuracy = accuracy_score(labels_train, train_label_pred)
    print(f"the SVM's accuracy on the training set is: {train_accuracy}")

    # print the accuracy on the validation set
    print(f"the SVM's accuracy on the validation set is: {accuracy}")

    return svm_model, scaler


def svm_train_cross_val(
    images_train: np.ndarray,
    labels_train: np.ndarray,
    images_val: np.ndarray,
    labels_val: np.ndarray,
    kernel: str,
    regularization: float,
    other_params: float,
) -> tuple:
    """
    Create the SVM model and train it using cross-validation.

    Args:
        images_train (np.ndarray): Input images training set.
        labels_train (np.ndarray): Corresponding labels for the training set.
        images_val (np.ndarray): Input images validation set.
        labels_val (np.ndarray): Corresponding labels for the validation set.
        kernel (str): The kernel type to be used in the SVM model.
        regularization (float): The regularization parameter for the SVM model.
        other_params (float): The other parameter for the SVM model(eg. gamma for rbf kernel).

    Returns:
        tuple:
            - sklearn.svm.SVC:
                A trained SVM.SVC object.
            - sklearn.preprocessing.StandardScaler:
                The StandardScaler used to standardize the image data.
    """
    # concatenate the training and validation set
    images_train = np.concatenate((images_train, images_val), axis=0)
    labels_train = np.concatenate((labels_train, labels_val), axis=0)

    # standardize the data
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(images_train)

    svm_model = None
    # train the model
    if kernel == "linear":
        svm_model = svm.SVC(
            kernel=kernel,
            C=regularization,
            class_weight="balanced",
            tol=0.001,
            random_state=711,
        )
    elif kernel == "rbf":
        svm_model = svm.SVC(
            kernel=kernel,
            C=regularization,
            class_weight="balanced",
            tol=0.001,
            random_state=711,
            gamma=other_params,
        )
    elif kernel == "poly":
        svm_model = svm.SVC(
            kernel=kernel,
            C=regularization,
            class_weight="balanced",
            tol=0.001,
            random_state=711,
            degree=other_params,
        )
    svm_model.fit(images_train_std, labels_train)

    # print the accuracy on the training set
    train_label_pred = svm_model.predict(images_train_std)
    train_accuracy = accuracy_score(labels_train, train_label_pred)
    print(f"the SVM's accuracy on the training set is: {train_accuracy}")

    # cross validation
    scores = cross_val_score(
        svm_model, images_train_std, labels_train, cv=5, scoring="accuracy"
    )
    print(f"Cross validation scores: {scores}")
    print(f"Mean cross validation score: {scores.mean()}")

    return svm_model, scaler


def svm_test_cross_val(
    images_test: np.ndarray,
    labels_test: np.ndarray,
    model: svm.SVC,
    scaler: StandardScaler,
):
    """
    Test the SVM model.

    Args:
        images_test (np.ndarray): Input images test set.
        labels_test (np.ndarray): Corresponding labels for the test set.
        model (svm.SVC): The trained SVM model.
        scaler (StandardScaler): The StandardScaler used to standardize the test data.

    Returns:
        tuple:
            - matplotlib.figure.Figure:
                The figure object for the confusion matrix.
            - matplotlib.axes.Axes:
                The axes object for the confusion matrix.
            - float:
                The accuracy of the SVM model.
            - float:
                The AUC value of the SVM model.
    """
    images_test_std = scaler.transform(images_test)
    label_pred = model.predict(images_test_std)
    label_score = model.decision_function(images_test_std)

    # evaluate the model
    fig, ax, accuracy, auc = model_evaluation(labels_test, label_pred, label_score)

    return fig, ax, accuracy, auc
