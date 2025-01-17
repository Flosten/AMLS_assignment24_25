"""
Main script for the project.

This script contains the main function to run this project. 
It includes two tasks:
    Task A: Breast cancer images classification using SVM.
    Task B: Blood cell images classification using CNN.
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import A.breastmnist_data_acquiring as a_data_acquiring
import A.breastmnist_data_processing as a_image_processing
import A.breastmnist_result_visualising as a_visualising
import B.bloodmnist_data_acquiring as b_data_acquiring
import B.bloodmnist_data_processing as b_bloodmnist_processing


def random_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed to be set.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def task_A():
    """
    Task A: Breast Cancer Detection

    This function contains the entire process of breast cancer images classification using SVM.

    The process includes the following steps:

    1. Data acquisition
    2. Data preprocessing
    3. Model training
    4. Model evaluation
    5. Result visualisation
    """
    # Load the data
    train_images, train_labels = a_data_acquiring.a_train_image_acquire()
    val_images, val_labels = a_data_acquiring.a_val_image_acquire()
    test_image, test_labels = a_data_acquiring.a_test_image_acquire()
    print("Check the data type: ")
    print(type(train_images), type(val_labels), type(test_image))
    print("Check the shape of the data: ")
    print(train_labels.shape, val_labels.shape, test_labels.shape)
    print(train_images.shape, val_images.shape, test_image.shape)

    # figure path
    figure_path = "figures"

    # SVM
    print("-----------------SVM-----------------")
    # check the distribution of the dataset
    fig_dis, _ = a_visualising.plot_label_distribution(train_labels)
    fig_dis.savefig(
        os.path.join(figure_path, "Task A - Distribution of the training set.png")
    )

    # Preprocess the images for SVM
    train_images_for_svm, train_labels_for_svm = a_image_processing.image_preprocess(
        train_images, train_labels
    )
    # print(train_images_for_svm.shape, train_labels_for_svm.shape)
    val_images_for_svm, val_labels_for_svm = a_image_processing.image_preprocess(
        val_images, val_labels
    )
    test_images_for_svm, test_labels_for_svm = a_image_processing.image_preprocess(
        test_image, test_labels
    )
    # print the shape of the images
    # print(train_images_for_svm.shape)

    # Train the SVM model
    _, _ = a_image_processing.svm_train(
        train_images_for_svm,
        train_labels_for_svm,
        val_images_for_svm,
        val_labels_for_svm,
        "linear",
        1,
    )

    # using cross-validation to select the hyperparameters
    _, _ = a_image_processing.svm_train_cross_val(
        train_images_for_svm,
        train_labels_for_svm,
        val_images_for_svm,
        val_labels_for_svm,
        "linear",
        1,
        None,
    )

    print("-----------------training-----------------")
    # linear kernel
    print("Linear kernel SVM model")
    svm_model_linear, scaler_linear = a_image_processing.svm_train_cross_val(
        train_images_for_svm,
        train_labels_for_svm,
        val_images_for_svm,
        val_labels_for_svm,
        "linear",
        0.0015,
        None,
    )

    # Test the SVM model
    svm_fig_linear, _, svm_acc_linear, svm_auc_linear = (
        a_image_processing.svm_test_cross_val(
            test_images_for_svm, test_labels_for_svm, svm_model_linear, scaler_linear
        )
    )
    print(f"Accuracy of SVM model: {svm_acc_linear}")
    print(f"The AUC value of SVM model: {svm_auc_linear}")

    # save the figure as svg
    svm_fig_linear.savefig(
        os.path.join(figure_path, "Confusion matrix of Linear kernel SVM model.png")
    )

    # rbf kernel
    print("RBF kernel SVM model")
    svm_model_rbf, scaler_rbf = a_image_processing.svm_train_cross_val(
        train_images_for_svm,
        train_labels_for_svm,
        val_images_for_svm,
        val_labels_for_svm,
        "rbf",
        0.1,
        0.002,
    )

    # Test the SVM model
    svm_fig_rbf, _, svm_acc_rbf, svm_auc_rbf = a_image_processing.svm_test_cross_val(
        test_images_for_svm,
        test_labels_for_svm,
        svm_model_rbf,
        scaler_rbf,
    )
    print(f"Accuracy of SVM model: {svm_acc_rbf}")
    print(f"The AUC value of SVM model: {svm_auc_rbf}")

    # save the figure as svg
    svm_fig_rbf.savefig(
        os.path.join(figure_path, "Confusion matrix of RBF kernel SVM model.png")
    )


def task_B():
    """
    Task B: Blood Cell Classification

    This function contains the entire process of blood cell images classification using CNN.

    The process includes the following steps:

    1. Data acquisition
    2. Data preprocessing
    3. Model training
    4. Model evaluation
    5. Result visualisation
    """
    # load the bloodmnist data: num * 28 * 28 * 3
    train_images, train_labels = b_data_acquiring.b_train_image_acquire()
    val_images, val_labels = b_data_acquiring.b_val_image_acquire()
    test_images, test_labels = b_data_acquiring.b_test_image_acquire()
    # print(len(train_images), len(val_images), len(test_image))
    print("Check the data type: ")
    print(type(train_labels))

    # figure path
    figure_path = "figures"

    # plot the image
    plt.figure()
    plt.imshow(train_images[0])
    plt.savefig(os.path.join(figure_path, "Blood cell image.png"))
    plt.close()

    # CNN
    print("-----------------CNN-----------------")
    # set hyperparameters
    num_epochs = 120
    num_classes = 8
    learning_rate = 0.001

    # Preprocess the images for CNN
    train_dataset = b_bloodmnist_processing.blood_image_preprocess_for_cnn(
        train_images, train_labels, 128, True
    )
    val_dataset = b_bloodmnist_processing.blood_image_preprocess_for_cnn(
        val_images, val_labels, 72, False
    )
    test_dataset = b_bloodmnist_processing.blood_image_preprocess_for_cnn(
        test_images, test_labels, len(test_labels), False
    )

    # create the CNN model
    cnn_model = b_bloodmnist_processing.BloodCNN(num_classes)
    # model initialization
    b_bloodmnist_processing.blood_cnn_params_init(cnn_model)

    # set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

    # train the model
    cnn_fig, _, cnn_model = b_bloodmnist_processing.blood_cnn_train(
        cnn_model, train_dataset, val_dataset, criterion, optimizer, num_epochs
    )

    # plot the training and validation loss
    print("Loss function image of CNN model: ")
    # cnn_ax.plot()
    # plt.show()
    cnn_fig.savefig(
        os.path.join(
            figure_path,
            "CNN training and validation loss with learning rate 0.001, "
            "epoch = 120 (with drop-out).png",
        )
    )

    # test the model
    cnn_test_loss, cnn_test_fig, _, cnn_test_acc, cnn_test_auc = (
        b_bloodmnist_processing.blood_image_cnn_test(cnn_model, test_dataset, criterion)
    )

    # print the test results
    print(f"Loss of CNN model: {cnn_test_loss}")
    print(f"Accuracy of CNN model: {cnn_test_acc}")
    print(f"The AUC value of CNN model: {cnn_test_auc}")

    # plot the confusion matrix
    print("Confusion matrix of CNN model: ")
    # cnn_test_ax.plot()
    # plt.show()

    cnn_test_fig.savefig(os.path.join(figure_path, "simple CNN confusion matrix.png"))


if __name__ == "__main__":
    # set the random seed for reproducibility
    random_seed(711)
    # create the figures folder
    os.makedirs("figures", exist_ok=True)
    # run task A
    task_A()

    # run task B
    task_B()
