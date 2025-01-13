import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from A.visualising import model_evaluation, train_val_loss_plot


def image_preprocess(images: np.ndarray, labels: np.ndarray):
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


# def wavelet_transform(images: np.ndarray, labels: np.ndarray):
#     """
#     Perform wavelet transform on the images.

#     Args:
#         images (np.ndarray): Input images set.
#         wavelet (str): The wavelet type to be used.

#     Returns:
#         tuple: Image and label set after feature extraction.
#     """
#     # get the images wavelet features and labels
#     images_wavelet = []
#     for image in images:
#         coeffs = pywt.dwt2(image, "haar")
#         ll, (lh, hl, hh) = coeffs
#         images_wavelet.append(
#             # np.concatenate([ll.flatten(), lh.flatten(), hl.flatten(), hh.flatten()])
#             np.concatenate([lh.flatten(), hl.flatten(), hh.flatten()])
#         )

#     images_wavelet = np.array(images_wavelet)
#     labels_wavelet = labels.flatten()

#     return images_wavelet, labels_wavelet


def PCA_preprocess(train_images: np.ndarray, rate: float):

    pca = PCA(rate)
    train_images_pca = pca.fit_transform(train_images)
    # print(f"PCA: {train_images_pca.shape[1]}")

    return train_images_pca, pca


def svm_train(
    images_train: np.ndarray,
    labels_train: np.ndarray,
    images_val: np.ndarray,
    labels_val: np.ndarray,
    kernal: str,
    regularization: float,
):

    # standardize the data
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(images_train)

    # train the model  C=0.005 0.002
    svm_model = svm.SVC(
        kernel=kernal,
        C=regularization,
        class_weight="balanced",
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
    kernal: str,
    regularization: float,
    other_params: float,
):
    """
    Train an SVM model and return the trained model along with the scaler.

    Args:
        images (np.ndarray): Input images training set.
        labels (np.ndarray): Corresponding labels for the training set.
        kernal (str): The kernel type to be used in the SVM model.

    Returns:
        tuple:
            - sklearn.svm.SVC:
                A trained Support Vector Classifier (SVC) object.
            - sklearn.preprocessing.StandardScaler:
                The StandardScaler used to standardize the training data.
    """
    # concatenate the training and validation set
    images_train = np.concatenate((images_train, images_val), axis=0)
    labels_train = np.concatenate((labels_train, labels_val), axis=0)

    # standardize the data
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(images_train)

    svm_model = None
    # train the model  C=0.005
    if kernal == "linear":
        svm_model = svm.SVC(
            kernel=kernal,
            C=regularization,
            class_weight="balanced",
            tol=0.001,
            random_state=711,
        )
    elif kernal == "rbf":
        svm_model = svm.SVC(
            kernel=kernal,
            C=regularization,
            class_weight="balanced",
            tol=0.001,
            random_state=711,
            gamma=other_params,
        )
    elif kernal == "poly":
        svm_model = svm.SVC(
            kernel=kernal,
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
        images (np.ndarray): Input images test set.
        labels (np.ndarray): Corresponding labels for the test set.
        model (svm.SVC): The trained SVM model.
        scaler (StandardScaler): The StandardScaler used to standardize the data.

    Returns:
        tuple:
            - matplotlib.figure.Figure:
                The figure object for the confusion matrix.
            - matplotlib.axes.Axes:
                The axes object for the confusion matrix.
            - float:
                The accuracy of the SVM model.
            - str:
                The classification report of the SVM model.
    """
    images_test_std = scaler.transform(images_test)
    label_pred = model.predict(images_test_std)
    label_score = model.decision_function(images_test_std)

    # evaluate the model
    fig, ax, accuracy, f1 = model_evaluation(labels_test, label_pred, label_score)

    return fig, ax, accuracy, f1


def svm_test(
    images: np.ndarray, labels: np.ndarray, model: svm.SVC, scaler: StandardScaler
):
    """
    Test the SVM model.

    Args:
        images (np.ndarray): Input images set.
        labels (np.ndarray): Corresponding labels for the images set.
        model (svm.SVC): The trained SVM model.
        scaler (StandardScaler): The StandardScaler used to standardize the data.

    Returns:
        tuple:
            - matplotlib.figure.Figure:
                The figure object for the confusion matrix.
            - matplotlib.axes.Axes:
                The axes object for the confusion matrix.
            - float:
                The accuracy of the SVM model.
            - str:
                The classification report of the SVM model.
    """
    images_std = scaler.transform(images)
    label_pred = model.predict(images_std)
    label_score = model.predict_proba(images_std)

    # evaluate the model
    fig, ax, accuracy, f1 = model_evaluation(labels, label_pred, label_score)

    return fig, ax, accuracy, f1


def compute_kernel(
    x1: np.ndarray, x2: np.ndarray, param_linear: float, param_rbf: float, gamma: float
):
    """
    Compute the kernel matrix.

    Args:
        input (np.ndarray): Input data.
        param_linear (float): The linear parameter.
        param_rbf (float): The rbf parameter.
        gamma (float): The gamma parameter.

    Returns:
        np.ndarray: The kernel matrix.
    """
    if x2 is None:
        x2 = x1
    linear_kernel_matrix = linear_kernel(x1, x2)
    rbf_kernel_matrix = rbf_kernel(x1, x2, gamma=gamma)
    kernel_matrix = linear_kernel_matrix * param_linear + rbf_kernel_matrix * param_rbf

    return kernel_matrix


def multiple_kernel_svm_train(
    images_train: np.ndarray,
    labels_train: np.ndarray,
    images_val: np.ndarray,
    labels_val: np.ndarray,
    param_linear: float,
    param_rbf: float,
    gamma: float,
):
    """
    Train a multiple kernel SVM model.

    Args:
        images_train (np.ndarray): Input images training set.
        labels_train (np.ndarray): Corresponding labels for the training set.
        param_linear (float): The linear parameter.
        param_rbf (float): The rbf parameter.
        gamma (float): The gamma parameter.

    Returns:
        sklearn.svm.SVC: A trained Support Vector Classifier (SVC) object.
    """
    # standardize the data
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(images_train)

    # compute the kernel matrix
    kernel_matrix = compute_kernel(
        images_train_std, None, param_linear, param_rbf, gamma
    )

    # train the model
    mksvm_model = svm.SVC(
        kernel="precomputed", C=1, class_weight="balanced", tol=0.001, random_state=711
    )
    mksvm_model.fit(kernel_matrix, labels_train)

    # evaluate the model
    images_val_std = scaler.transform(images_val)
    kernel_matrix_val = compute_kernel(
        images_val_std, images_train_std, param_linear, param_rbf, gamma
    )
    label_pred = mksvm_model.predict(kernel_matrix_val)
    accuracy = accuracy_score(labels_val, label_pred)

    # print the accuracy
    print(f"the Multiple Kernel SVM's accuracy on the validation set is: {accuracy}")

    return mksvm_model, scaler


def multiple_kernel_svm_test(
    images_test: np.ndarray,
    labels_test: np.ndarray,
    image_train: np.ndarray,
    model: svm.SVC,
    param_linear: float,
    param_rbf: float,
    gamma: float,
):
    """
    Test the multiple kernel SVM model.

    Args:
        images_test (np.ndarray): Input images test set.
        labels_test (np.ndarray): Corresponding labels for the test set.
        image_train (np.ndarray): Input images training set.
        model (svm.SVC): The trained SVM model.
        param_linear (float): The linear parameter.
        param_rbf (float): The rbf parameter.
        gamma (float): The gamma parameter.

    Returns:
        tuple:
            - matplotlib.figure.Figure:
                The figure object for the confusion matrix.
            - matplotlib.axes.Axes:
                The axes object for the confusion matrix.
            - float:
                The accuracy of the SVM model.
            - str:
                The classification report of the SVM model.
    """
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(image_train)
    images_test_std = scaler.transform(images_test)

    # compute the kernel matrix
    kernel_matrix_test = compute_kernel(
        images_test_std, images_train_std, param_linear, param_rbf, gamma
    )
    label_pred = model.predict(kernel_matrix_test)
    label_score = model.predict_proba(kernel_matrix_test)

    # evaluate the model
    fig, ax, accuracy, report = model_evaluation(labels_test, label_pred, label_score)

    return fig, ax, accuracy, report
