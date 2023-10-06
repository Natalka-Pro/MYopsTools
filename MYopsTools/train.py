import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from datetime import datetime
from torchvision import models

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')


LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 5
N_CLASSES = 10
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


def number_of_parameters(model):
    return f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters"


def dataset_loaders(train):
    resize = transforms.Resize((32, 32))
    to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    if train:
        dataset_train = datasets.MNIST(root = '../data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = BATCH_SIZE, shuffle=True)
        return train_loader
    else:
        dataset_test  = datasets.MNIST(root = '../data', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = BATCH_SIZE, shuffle=False)
        return test_loader


def plot_dataset(loader, title, N_ROWS = 5, N_COLUMNS = 10):
    '''
    Function for plotting multiple images from dataset
    '''
    fig = plt.figure(figsize=(N_COLUMNS * 1.3, N_ROWS * 1.3))
    fig.suptitle(title, fontsize = 20)

    for index in range(1, N_COLUMNS * N_ROWS + 1):
        plt.subplot(N_ROWS, N_COLUMNS, index)
        plt.axis('off')
        image, label = loader.dataset.__getitem__(index)
        plt.imshow(image.permute(1, 2, 0).numpy())


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    '''
    Function for plotting train/test losses and accuracies
    '''
    plt.style.use('seaborn')

    train_loss = np.array(train_losses)
    test_loss = np.array(test_losses)
    train_acc = np.array(train_accuracies)
    test_acc = np.array(test_accuracies)

    plt.figure(figsize = (13, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='blue', label='Training loss')
    plt.plot(test_loss, color='red', label='Test loss')
    plt.title("Loss over epochs", fontsize=13)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, color='green', label='Training accuracy')
    plt.plot(test_acc, color='orangered', label='Test accuracy')
    plt.title("Accuracy over epochs", fontsize=13)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.legend()
    plt.show()

    plt.style.use('default')


def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
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


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''
    model.train()
    running_loss = 0

    for X, y_true in train_loader:

        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(test_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
    model.eval()
    running_loss = 0

    for X, y_true in test_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)

    return model, epoch_loss


def training_loop(model, name, criterion, optimizer, train_loader, num_epoch, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    # test_losses = []
    train_accuracies = []
    # test_accuracies = []

    # Train model
    for epoch in tqdm(range(num_epoch)):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        # with torch.no_grad():
        #     model, test_loss = validate(test_loader, model, criterion, device)
        #     test_losses.append(test_loss)

        if epoch % print_every == (print_every - 1):

            train_acc = get_accuracy(model, train_loader, device=device)
            # test_acc = get_accuracy(model, test_loader, device=device)

            train_accuracies.append(train_acc.item())
            # test_accuracies.append(test_acc.item())

            print(  f'Time: {datetime.now().time().replace(microsecond=0)}   ---   ',
                    f'Epoch: {epoch}\t',
                    f'Train loss: {train_loss:.4f}\t',
                    # f'Test loss: {test_loss:.4f}\t',
                    f'Train accuracy: {100 * train_acc:.2f}\t',
                    # f'Test accuracy: {100 * test_acc:.2f}',
                  )

    torch.save(model.state_dict(), name + '.pth')
    return model, optimizer #, (train_losses, test_losses, train_accuracies, test_accuracies)

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4 * 8 * 8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{DEVICE = }")

    train_loader  = dataset_loaders(train = True)
    model = AlexNet(num_classes = 10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model, optimizer = training_loop(model, "model", criterion, optimizer, 
                                    train_loader, N_EPOCHS, DEVICE)
