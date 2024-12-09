import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
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


# evaluate the model
def model_evaluation(y_true, y_pred):
    # plot the confusion matrix
    cm_matrix = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm_matrix, display_labels=["Benign", "Malignant"]
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix of BreastMNIST")

    # calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # print the classification report
    report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])

    return fig, ax, accuracy, report
