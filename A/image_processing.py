import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from A.data_acquiring import *
from A.visualising import *


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


def svm_train(
    images_train: np.ndarray,
    labels_train: np.ndarray,
    images_val: np.ndarray,
    labels_val: np.ndarray,
    kernal: str,
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
    # standardize the data
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(images_train)

    # train the model
    svm_model = svm.SVC(kernel=kernal, random_state=711)
    svm_model.fit(images_train_std, labels_train)

    # evaluate the model
    images_val_std = scaler.transform(images_val)
    label_pred = svm_model.predict(images_val_std)
    accuracy = accuracy_score(labels_val, label_pred)

    # print the accuracy
    print(
        f"Without hyperparameter selection, the SVM's accuracy on the validation set is: {accuracy}"
    )

    return svm_model, scaler


def svm_train_best_params(
    images_train: np.ndarray,
    images_val: np.ndarray,
    labels_train: np.ndarray,
    labels_val: np.ndarray,
    kernal: str,
):
    """
    Find the best hyperparameters for the SVM model.

    Args:
        images_train (np.ndarray): Input images training set.
        images_val (np.ndarray): Input images validation set.
        labels_train (np.ndarray): Corresponding labels for the training set.
        labels_val (np.ndarray): Corresponding labels for the validation set.
        kernal (str): The kernel type to be used in the SVM model.

    Returns:
        tuple:
            - sklearn.svm.SVC:
                The best trained Support Vector Classifier (SVC) object.
            - sklearn.preprocessing.StandardScaler:
                The StandardScaler used to standardize the training data.
            - dict:
                The best hyperparameters found.
    """
    best_score = 0
    best_params = {}
    best_model = None

    # standardize the data
    scaler = StandardScaler()
    images_train_std = scaler.fit_transform(images_train)
    images_val_std = scaler.transform(images_val)

    # set the hyperparamters
    hyperparams = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": [0.01, 0.1, 1, 10, 100, "scale", "auto"] if kernal == "rbf" else [],
    }

    # serach for the best hyperparameters
    for C1 in hyperparams["C"]:
        for gamma1 in hyperparams["gamma"]:
            # train the model
            model = svm.SVC(kernel=kernal, C=C1, gamma=gamma1, random_state=711)
            model.fit(images_train_std, labels_train)

            # evaluate the model
            label_pred = model.predict(images_val_std)
            score = accuracy_score(labels_val, label_pred)

            # update the best model
            if score > best_score:
                best_score = score
                best_params["C"] = C1
                best_params["gamma"] = gamma1
                best_model = model

    # print the best score
    print(
        f"With hyperparameter selection, the SVM's accuracy on the validation set is: {best_score}"
    )

    return best_model, scaler, best_params


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

    # evaluate the model
    fig, ax, accuracy, report = model_evaluation(labels, label_pred)

    return fig, ax, accuracy, report


class cnn(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 7, kernel_size=2, stride=1, padding=1),  # 1*28*28 -> 5*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 5*28*28 -> 5*14*14
            nn.Conv2d(7, 11, kernel_size=2, stride=1, padding=1),  # 5*14*14 -> 11*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 11*14*14 -> 11*7*7
        )

        self.classifier = nn.Sequential(
            nn.Linear(11 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# def cnn_params_init(model: nn.Module):
#     """
#     Initialize the CNN model parameters.

#     Args:
#         model (nn.Module): The CNN model to be initialized.
#     """
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.normal_(m.weight, mean=0, std=0.01)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, mean=0, std=0.01)
#             nn.init.constant_(m.bias, 0)


def cnn_params_init(model: nn.Module):
    """
    Initialize the CNN model parameters with best practices.

    Args:
        model (nn.Module): The CNN model to be initialized.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                # initialize the bias to 0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # He initialization
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def data_preprocess_for_cnn(
    images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool
):
    """
    Preprocess the data for the CNN model.

    Args:
        images (np.ndarray): Input images set.
        labels (np.ndarray): Input labels set.
        batch_size (int): The batch size for the data loader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        torch.utils.data.DataLoader: The DataLoader object for the CNN model.
    """
    # numpy -> tensor & normalize
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    labels = torch.tensor(labels, dtype=torch.long).squeeze(1)

    # print(labels.shape)

    # create dataset
    dataset = TensorDataset(images, labels)

    # load data
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    return data_loader


def cnn_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    num_epochs: int,
):
    train_loss = []
    val_loss = []

    # train set -> train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs_train, labels_train in tqdm(
            train_loader, desc=f"epoch:{epoch+1} / {num_epochs}", unit="batch"
        ):
            optimizer.zero_grad()
            outputs_train = model(inputs_train)

            loss = criterion(outputs_train, labels_train)
            loss.backward()

            optimizer.step()
            running_loss += loss.item() * inputs_train.size(0)

        epoch_loss_train = running_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss_train)
        print(f"epoch:{epoch+1} / {num_epochs}  Train Loss: {epoch_loss_train:.4f}")

        # validation set -> adjust the hyperparameters
        model.eval()
        running_loss = 0.0
        total = 0

        with torch.no_grad():
            for (
                inputs,
                labels,
            ) in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        epoch_loss_val = running_loss / len(val_loader.dataset)
        val_loss.append(epoch_loss_val)
    fig, ax = train_val_loss_plot(train_loss, val_loss)

    return fig, ax, model


def cnn_test(model: nn.Module, test_loader: DataLoader, criterion: nn.CrossEntropyLoss):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # get the predictions
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # evaluate the model
    test_fig, test_ax, accuracy, report = model_evaluation(all_labels, all_preds)
    test_loss = running_loss / len(test_loader.dataset)

    return test_loss, test_fig, test_ax, accuracy, report
