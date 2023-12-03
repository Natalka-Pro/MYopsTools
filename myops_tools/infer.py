import pandas as pd
import torch
import torch.nn as nn

from .dataloader import MnistData
from .models import AlexNet

BATCH_SIZE = 64


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

        self.pred_true = pred_true
        self.accuracy = (correct_pred.float() / n).item()
        self.loss = running_loss / len(self.test_loader.dataset)


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{DEVICE = }")

    test_loader = MnistData(batch_size=BATCH_SIZE).test_loader()
    model = AlexNet(num_classes=10).to(DEVICE)
    model_name = "model.pth"
    model.load_state_dict(torch.load(model_name))
    print(f"Model \"{model_name}\" is loaded")
    criterion = nn.CrossEntropyLoss()

    Testing = TestClass(model, criterion, test_loader, DEVICE)
    Testing.validate()
    pred_true = Testing.pred_true
    test_acc = Testing.accuracy
    test_loss = Testing.loss

    # x = (pred_true[:, 0] == pred_true[:, 1]
    #               ).sum().float() / pred_true.size(0)
    print(f"Validation: accuracy = {test_acc}, loss = {test_loss}")

    # ans = ans.to("cpu")
    df = pd.DataFrame(pred_true, columns=["Predicted labels", "True labels"])

    file_name = "labels.csv"
    df.to_csv(file_name)
    print(
        f"A matrix of size {tuple(pred_true.size())} ",
        f"was written to the file \"{file_name}\"\n",
        "Column names - \"Predicted labels\" and \"True labels\"",
        sep="",
    )
