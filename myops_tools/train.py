from datetime import datetime

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .dataloader import MnistData
from .models import AlexNet
from .utils import get_accuracy

LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 5
N_CLASSES = 10
RANDOM_SEED = 42


def train(train_loader, model, criterion, optimizer, device):
    """
    Function for the training step of the training loop
    """
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
    """
    Function for the validation step of the training loop
    """
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


def training_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    num_epoch,
    device,
    print_every=1,
):
    """
    Function defining the entire training loop
    """
    # set objects for storing metrics
    train_losses = []
    # test_losses = []
    train_accuracies = []
    # test_accuracies = []

    # Train model
    for epoch in tqdm(range(num_epoch)):
        # training
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device
        )
        train_losses.append(train_loss)

        # validation
        # with torch.no_grad():
        #     model, test_loss = validate(test_loader, model,
        #                                 criterion, device)
        #     test_losses.append(test_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            # test_acc = get_accuracy(model, test_loader, device=device)

            train_accuracies.append(train_acc.item())
            # test_accuracies.append(test_acc.item())

            print(
                f"Time: {datetime.now().time().replace(microsecond=0)}",
                "   ---   ",
                f"Epoch: {epoch}\t",
                f"Train loss: {train_loss:.4f}\t",
                # f'Test loss: {test_loss:.4f}\t',
                f"Train accuracy: {100 * train_acc:.2f}\t",
                # f'Test accuracy: {100 * test_acc:.2f}',
            )

    return model, optimizer
    # , (train_losses, test_losses, train_accuracies, test_accuracies)


def main():
    torch.manual_seed(RANDOM_SEED)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{DEVICE = }")

    train_loader = MnistData(batch_size=BATCH_SIZE).train_loader()
    model = AlexNet(num_classes=10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model, optimizer = training_loop(
        model, criterion, optimizer, train_loader, N_EPOCHS, DEVICE
    )

    model_name = "model.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model is saved with the name \"{model_name}\"")
