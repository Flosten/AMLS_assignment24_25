"""
Module for visualising the classification results based on the SVM model.

This file contains the analysis results of the dataset distribution characteristics and 
the evaluation of the classification results of the SVM model.

Functions:
    plot_label_distribution: Plot the distribution of the categories in the dataset.
    model_evaluation: Evaluate the model using the confusion matrix, accuracy, and AUC value.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def plot_label_distribution(labels: np.ndarray):
    """
    Explore the distribution characteristics of the dataset and plot the distribution graph.

    Args:
        labels (np.ndarray): The labels of the training dataset.

    Returns:
        tuple:
            - matplotlib.figure.Figure:
                The figure object for the distribution of different categories.
            - matplotlib.axes.Axes:
                The axes object for the distribution of different categories.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # flatten the label
    labels = labels.flatten()
    # count the number of each label
    unique, counts = np.unique(labels, return_counts=True)

    # plot the bar chart
    ax.bar(unique, counts, tick_label=["malignant", "normal and benign"], width=0.5)
    ax.set_xlabel("category")
    ax.set_ylabel("sample number")
    ax.set_title("Distribution of the training set")

    return fig, ax


# evaluate the model
def model_evaluation(y_true, y_pred, y_score):
    """
    Evaluate the model using the confusion matrix, accuracy, and AUC value.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        y_score (np.ndarray): The score predicted by the SVM model.

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
    label_name = ["malignant", "normal and benign"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(
        cm_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 20},
        xticklabels=label_name,
        yticklabels=label_name,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # calculate the AUC value
    auc = roc_auc_score(y_true, y_score)

    return fig, ax, accuracy, auc
