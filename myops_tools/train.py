from datetime import datetime

# import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import MnistData
from .models import AlexNet
from .utils import get_accuracy


class TrainClass:
    def __init__(
        self, model, criterion, optimizer, train_loader, n_epochs, device
    ):
        # super().__init__()
        # self.save_hyperparameters()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.device = device

    def train(self):
        """
        Function for the training step of the training loop
        """
        self.model.train()
        running_loss = 0

        for X, y_true in self.train_loader:
            self.optimizer.zero_grad()

            X = X.to(self.device)
            y_true = y_true.to(self.device)

            # Forward pass
            y_hat = self.model(X)
            loss = self.criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)

            # Backward pass
            loss.backward()
            self.optimizer.step()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def training_loop(self, print_every=1):
        """
        Function defining the entire training loop
        """
        train_losses = []
        # test_losses = []
        train_accuracies = []
        # test_accuracies = []

        # Train model
        for epoch in tqdm(range(self.n_epochs)):
            # training
            train_loss = self.train()
            train_losses.append(train_loss)

            # validation
            # with torch.no_grad():
            #     model, test_loss = validate(test_loader, model,
            #                                 criterion, device)
            #     test_losses.append(test_loss)

            if epoch % print_every == (print_every - 1):
                train_acc = get_accuracy(
                    self.model, self.train_loader, self.device
                )
                # test_acc = get_accuracy(model, test_loader, device=device)

                train_accuracies.append(train_acc.item())
                # test_accuracies.append(test_acc.item())

                print(
                    f"Time: {datetime.now().time().replace(microsecond=0)}",
                    "   ---   ",
                    f"Epoch: {epoch}\t",
                    f"Train loss: {train_loss:.4f}\t",
                    # f'Test loss: {test_loss:.4f}\t',
                    f"Train accuracy: {100 * train_acc:.4f}\t",
                    # f'Test accuracy: {100 * test_acc:.4f}',
                )

        # return self.model, self.optimizer
        # , (train_losses, test_losses, train_accuracies, test_accuracies)


def main(config):
    torch.manual_seed(config.train.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device type: {device}")

    train_loader = MnistData(
        batch_size=config.train.batch_size
    ).train_dataloader()
    model = AlexNet(config.model.n_classes, config.model.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    Training = TrainClass(
        model,
        criterion,
        optimizer,
        train_loader,
        config.train.n_epochs,
        device,
    )
    Training.training_loop()
    model = Training.model

    torch.save(model.state_dict(), config.train.model_save)
    print(f"Model is saved with the name \"{config.train.model_save}\"")
