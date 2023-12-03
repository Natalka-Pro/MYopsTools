import matplotlib.pyplot as plt
import numpy as np
import torch


def get_accuracy(model, data_loader, device):
    """
    Function for computing the accuracy of the predictions
    over the entire data_loader
    """
    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def number_of_parameters(model):
    return f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} \
        parameters"


def plot_dataset(loader, title, N_ROWS=5, N_COLUMNS=10):
    """
    Function for plotting multiple images from dataset
    """
    fig = plt.figure(figsize=(N_COLUMNS * 1.3, N_ROWS * 1.3))
    fig.suptitle(title, fontsize=20)

    for index in range(1, N_COLUMNS * N_ROWS + 1):
        plt.subplot(N_ROWS, N_COLUMNS, index)
        plt.axis("off")
        image, label = loader.dataset.__getitem__(index)
        plt.imshow(image.permute(1, 2, 0).numpy())


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Function for plotting train/test losses and accuracies
    """
    plt.style.use("seaborn")

    train_losses = np.array(train_losses)
    test_loss = np.array(test_losses)
    train_acc = np.array(train_accuracies)
    test_acc = np.array(test_accuracies)

    plt.figure(figsize=(13, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color="blue", label="Training loss")
    plt.plot(test_loss, color="red", label="Test loss")
    plt.title("Loss over epochs", fontsize=13)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, color="green", label="Training accuracy")
    plt.plot(test_acc, color="orangered", label="Test accuracy")
    plt.title("Accuracy over epochs", fontsize=13)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.legend()
    plt.show()

    plt.style.use("default")
