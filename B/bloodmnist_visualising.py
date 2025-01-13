import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def B_train_val_loss_plot(train_loss: list, val_loss: list):

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_loss, label="Train")
    ax.plot(val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()

    return fig, ax


# evaluate the model
def B_model_evaluation(y_true, y_pred, y_score):
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
