import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def train_val_loss_plot(train_loss: list, val_loss: list):

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_loss, label="Train")
    ax.plot(val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()

    return fig, ax


def plot_label_distribution(labels: np.ndarray):

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

    # print the f1 score
    auc = roc_auc_score(y_true, y_score)

    return fig, ax, accuracy, auc
