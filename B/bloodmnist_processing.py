import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from B.bloodmnist_visualising import B_model_evaluation, B_train_val_loss_plot


def B_image_preprocess(images: np.ndarray, labels: np.ndarray):

    images = images.reshape(images.shape[0], -1)
    labels = labels.flatten()

    return images, labels


def B_image_scaling(images: np.ndarray):

    scalar = StandardScaler()
    images = scalar.fit_transform(images)

    return images, scalar


def B_PCA_preprocess(train_images: np.ndarray, rate: float):

    pca = PCA(rate)
    train_images_pca = pca.fit_transform(train_images)
    # print(f"PCA: {train_images_pca.shape[1]}")

    return train_images_pca, pca


# CNN
class blood_cnn(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 7, kernel_size=2, stride=1, padding=1),  # 3*28*28 -> 7*28*28
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7*28*28 -> 7*14*14
            # nn.Dropout(0.2),
            nn.Conv2d(7, 11, kernel_size=2, stride=1, padding=1),  # 7*14*14 -> 11*14*14
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 11*14*14 -> 11*7*7
            # nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(11 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def blood_cnn_params_init(model: nn.Module):
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


def blood_image_preprocess_for_cnn(
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
    images = torch.tensor(images, dtype=torch.float32) / 255.0
    # num * 28 * 28 * 3 -> num * 3 * 28 * 28
    images = images.permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long).squeeze(1)

    # print(images.shape)
    # print(labels.shape)

    # create dataset
    dataset = TensorDataset(images, labels)
    # print(type(dataset))

    # load data
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    return data_loader


def blood_cnn_train(
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
    fig, ax = B_train_val_loss_plot(train_loss, val_loss)

    return fig, ax, model


def blood_cnn_train_steplr(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    num_epochs: int,
    scheduler: optim.lr_scheduler.StepLR,
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
    fig, ax = B_train_val_loss_plot(train_loss, val_loss)

    return fig, ax, model


def blood_image_cnn_test(
    model: nn.Module, test_loader: DataLoader, criterion: nn.CrossEntropyLoss
):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # get the predictions
            scores = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_scores.append(scores)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()

    # evaluate the model
    test_fig, test_ax, accuracy, auc = B_model_evaluation(
        all_labels, all_preds, all_scores
    )
    test_loss = running_loss / len(test_loader.dataset)

    return test_loss, test_fig, test_ax, accuracy, auc
