import pandas as pd
import torch
import torch.nn as nn

from .data import MnistDataModule
from .models import AlexNet


class TestClass:
    def __init__(self, model, criterion, test_loader, device):
        self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device

        self.pred_true = None  # size N x 2
        self.accuracy = None
        self.loss = None

    def validate(self):
        """
        Function for the validation step of the training loop
        """
        pred_true = torch.tensor([]).to(self.device)

        correct_pred = 0
        n = 0

        running_loss = 0

        with torch.no_grad():
            self.model.eval()

            for X, y_true in self.test_loader:
                X = X.to(self.device)
                y_true = y_true.to(self.device)

                y_prob = self.model(X)
                _, y_pred = torch.max(y_prob, 1)
                y_pred_true = torch.stack([y_pred, y_true], dim=1)
                pred_true = torch.cat([pred_true, y_pred_true])

                correct_pred += (y_pred == y_true).sum()
                n += y_true.size(0)

                # Forward pass and record loss
                loss = self.criterion(y_prob, y_true)
                running_loss += loss.item() * X.size(0)

        self.pred_true = pred_true.int()
        self.accuracy = (correct_pred.int() / n).item()
        self.loss = running_loss / len(self.test_loader.dataset)


def main(config):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.model.accelerator)
    print(f"Device type: {device}")

    test_loader = MnistDataModule(
        batch_size=config.infer.batch_size
    ).test_dataloader()
    model = AlexNet(config.model.n_classes, config.model.dropout).to(device)
    model.load_state_dict(torch.load(config.infer.model_load))
    print(f"Model \"{config.infer.model_load}\" is loaded")
    criterion = nn.CrossEntropyLoss()

    Testing = TestClass(model, criterion, test_loader, device)
    Testing.validate()
    pred_true = Testing.pred_true.to("cpu")
    test_acc = Testing.accuracy
    test_loss = Testing.loss

    print(f"Validation: accuracy = {test_acc:.4f}, loss = {test_loss:.4f}")

    df = pd.DataFrame(pred_true, columns=["Predicted labels", "True labels"])

    df.to_csv(config.infer.preds_file)
    print(
        f"A matrix of size {tuple(pred_true.size())} ",
        f"was written to the file \"{config.infer.preds_file}\"\n",
        "Column names - \"Predicted labels\" and \"True labels\"",
        sep="",
    )
