import torch
import torch.optim as optim

from A.data_acquiring import *
from A.image_processing import *
from A.visualising import *
from B.data_acquiring import *


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

    # Load the data
    train_images, train_labels = A_train_image_acquire("BreastMNIST", "breastmnist.npz")
    val_images, val_labels = A_val_image_acquire("BreastMNIST", "breastmnist.npz")
    test_image, test_labels = A_test_image_acquire("BreastMNIST", "breastmnist.npz")
    # print(len(train_images), len(val_images), len(test_image))

    # # SVM
    # print("-----------------SVM-----------------")
    # # Preprocess the images for SVM
    # train_images_for_svm, train_labels_for_svm = image_preprocess(
    #     train_images, train_labels
    # )
    # val_images_for_svm, val_labels_for_svm = image_preprocess(val_images, val_labels)
    # test_images_for_svm, test_labels_for_svm = image_preprocess(test_image, test_labels)

    # # Train the SVM model (without selecting the hyperparameters)
    # svm_model, scaler = svm_train(
    #     train_images_for_svm,
    #     train_labels_for_svm,
    #     val_images_for_svm,
    #     val_labels_for_svm,
    #     "rbf",
    # )

    # # Test the SVM model
    # svm_fig1, svm_ax1, svm_acc1, svm_report1 = svm_test(
    #     test_images_for_svm, test_labels_for_svm, svm_model, scaler
    # )
    # print("SVM model without hyperparameter selection")
    # print(f"Accuracy of SVM model: {svm_acc1}")
    # print(f"Classification report of SVM model: {svm_report1}")
    # print(f"Confusion matrix of SVM model: ")
    # svm_ax1.plot()
    # plt.show()

    # # svm_fig1.savefig("SVM model without hyperparameter selection.png")

    # # Train the SVM model (with selecting the hyperparameters)
    # kernal = "rbf"
    # svm_best_model, svm_scaler_new, svm_best_params = svm_train_best_params(
    #     train_images_for_svm,
    #     val_images_for_svm,
    #     train_labels_for_svm,
    #     val_labels_for_svm,
    #     kernal,
    # )
    # print(f"Best hyperparameters for SVM: {svm_best_params}")

    # # Test the SVM model
    # svm_fig2, svm_ax2, svm_acc2, svm_report2 = svm_test(
    #     test_images_for_svm, test_labels_for_svm, svm_best_model, svm_scaler_new
    # )
    # print("SVM model with hyperparameter selection")
    # print(f"Accuracy of SVM model: {svm_acc2}")
    # print(f"Classification report of SVM model: {svm_report2}")
    # print(f"Confusion matrix of SVM model: ")
    # svm_ax2.plot()
    # plt.show()

    # svm_fig2.savefig("SVM model with hyperparameter selection.png")

    # CNN
    print("-----------------CNN-----------------")

    # set hyperparameters
    num_epochs = 200
    num_classes = 2
    learning_rate = 0.0001

    # Preprocess the images for CNN
    train_dataset = data_preprocess_for_cnn(train_images, train_labels, 128, True)
    val_dataset = data_preprocess_for_cnn(val_images, val_labels, 78, False)
    test_dataset = data_preprocess_for_cnn(
        test_image, test_labels, len(test_labels), False
    )

    # create the CNN model
    model = cnn(num_classes)
    # model initialization
    cnn_params_init(model)

    # set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # train the model
    cnn_fig, cnn_ax, cnn_model = cnn_train(
        model, train_dataset, val_dataset, criterion, optimizer, num_epochs
    )

    # plot the training and validation loss
    print(f"Loss function image of CNN model: ")
    cnn_ax.plot()
    plt.show()
    # cnn_fig.savefig("CNN training and validation loss.png")

    # test the model
    cnn_test_loss, cnn_test_fig, cnn_test_ax, cnn_test_acc, cnn_test_repo = cnn_test(
        cnn_model, test_dataset, criterion
    )

    # print the test results
    print(f"Loss of CNN model(test set): {cnn_test_loss}")
    print(f"Accuracy of CNN model(test set): {cnn_test_acc}")
    print(f"Classification report of CNN model(test set):")
    print(cnn_test_repo)

    # plot the confusion matrix
    print(f"Confusion matrix of CNN model: ")
    cnn_test_ax.plot()
    plt.show()
    # cnn_test_fig.savefig("CNN_confusion_matrix.png")


def task_B():
    # load the bloodmnist data
    train_images, train_labels = train_image_acquire("BloodMNIST", "bloodmnist.npz")


if __name__ == "__main__":
    random_seed(711)
    # run task A
    task_A()
