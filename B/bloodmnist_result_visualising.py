"""
Module for visualising the classification results based on the CNN model.

This module contains functions for evaluating the CNN model 
using the confusion matrix, accuracy, and AUC value.

Functions:
    b_train_val_loss_plot: Plot the learning curve of the training and validation loss.
    b_model_evaluation: Evaluate the model using the confusion matrix, accuracy, and AUC value.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def b_train_val_loss_plot(train_loss: list, val_loss: list):
    """
    Plot the learning curve of the training and validation loss.

    Args:
        train_loss (list): The training loss.
        val_loss (list): The validation loss.

    Returns:
        tuple:
            - matplotlib.figure.Figure:
                The figure object for the learning curve of the training and validation loss.
            - matplotlib.axes.Axes:
                The axes object for the learning curve of the training and validation loss.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_loss, label="Train")
    ax.plot(val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()

    return fig, ax


# evaluate the model
def b_model_evaluation(y_true, y_pred, y_score):
    """
    Evaluate the model using the confusion matrix, accuracy, and AUC value.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        y_score (np.ndarray): The score predicted by the CNN model.

    Returns:
        tuple:
            - matplotlib.figure.Figure:
                The figure object for the confusion matrix.
            - matplotlib.axes.Axes:
                The axes object for the confusion matrix.
            - float:
                The accuracy of the model.
            - float:
                The AUC value of the model
    """
    # plot the confusion matrix
    cm_matrix = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_matrix)
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix of BloodMNIST")

    # calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # calculate the AUC multi-class
    auc = roc_auc_score(y_true, y_score, multi_class="ovr")

    return fig, ax, accuracy, auc
