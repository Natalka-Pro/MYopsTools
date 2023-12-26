import mlflow
import numpy as np
import onnx
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

        loss = 0

        with torch.no_grad():
            self.model.eval()

            for X, y_true in self.test_loader:
                X = X.to(self.device)
                y_true = y_true.to(self.device)

                y_prob = self.model(X)
                _, y_pred = torch.max(y_prob, axis=1)
                y_pred_true = torch.stack([y_pred, y_true], dim=1)
                pred_true = torch.cat([pred_true, y_pred_true])

                correct_pred += (y_pred == y_true).sum()
                n += y_true.size(0)

                # Forward pass and record loss
                lo = self.criterion(y_prob, y_true)
                loss += lo.item() * X.size(0)

        self.pred_true = pred_true.int()
        self.accuracy = (correct_pred.int() / n).item()
        self.loss = loss / len(self.test_loader.dataset)


def save_to_csv(matrix, file):
    print("<<< infer.save_to_csv:")
    df = pd.DataFrame(matrix, columns=["Predicted labels", "True labels"])
    df.to_csv(file)
    print(
        f"A matrix of size {tuple(matrix.shape)} ",
        f"was written to the file \"{file}\"\n",
        "Column names - \"Predicted labels\" and \"True labels\"",
        sep="",
    )
    print(">>>")


def infer(config):
    print("<<< infer.infer:")
    device = torch.device(config.model.accelerator)
    print(f"Device type: {device}")

    model = AlexNet(config.model.n_classes, config.model.dropout).to(device)
    model.load_state_dict(torch.load(config.infer.model_load))
    print(f"Model \"{config.infer.model_load}\" is loaded")
    criterion = nn.CrossEntropyLoss()

    test_loader = MnistDataModule(
        batch_size=config.infer.batch_size
    ).test_dataloader()

    Testing = TestClass(model, criterion, test_loader, device)
    Testing.validate()
    pred_true = Testing.pred_true.to("cpu")
    accuracy = Testing.accuracy
    loss = Testing.loss

    print(f"Validation accuracy = {accuracy:.6f}, loss = {loss:.6f}")
    save_to_csv(pred_true, config.infer.preds_file)
    print(">>>")


def infer_onnx(onnx_pyfunc, config):
    print("<<< infer.infer_onnx:")
    # https://mlflow.org/docs/2.9.2/getting-started/intro-quickstart/index.html
    test_loader = MnistDataModule(
        batch_size=config.infer.batch_size
    ).test_dataloader()

    pred_true = np.empty((0, 2), int)  # size N x 2
    criterion = nn.CrossEntropyLoss()
    loss = 0

    for X, y_true in test_loader:
        X = X.detach().numpy()

        y_prob = onnx_pyfunc.predict(X)['proba_distr']
        y_pred = np.argmax(y_prob, axis=1)
        y_pred_true = np.stack([y_pred, y_true], axis=1)
        pred_true = np.append(pred_true, y_pred_true, axis=0)
        y_prob = torch.from_numpy(y_prob)
        lo = criterion(y_prob, y_true)
        loss += lo.item() * X.shape[0]

    pred_true = pred_true.astype(int)
    accuracy = (pred_true[:, 0] == pred_true[:, 1]).sum() / len(pred_true)
    loss = loss / len(test_loader.dataset)
    print(f"Validation accuracy = {accuracy:.6f}, loss = {loss:.6f}")

    save_to_csv(pred_true, config.infer.preds_file)
    print(">>>")


def run_server(config):
    # https://mlflow.org/docs/latest/models.html#onnx-pyfunc-usage-example
    X = torch.randn(64, 3, 32, 32)

    mlflow.set_tracking_uri(
        f"http://{config.artifacts.host}:{config.artifacts.port}"
    )
    mlflow.set_experiment(experiment_name=config.artifacts.experiment_name)
    # log the model into a mlflow run
    model = AlexNet(config.model.n_classes, config.model.dropout).to(
        torch.device(config.model.accelerator)
    )

    onnx_model = onnx.load_model(config.train.model_save_onnx)
    print(f"Model \"{config.train.model_save_onnx}\" is loaded")

    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            X.numpy(), model(X).detach().numpy()
        )
        model_info = mlflow.onnx.log_model(
            onnx_model, "model", signature=signature
        )

    # load the logged model and make a prediction
    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

    infer_onnx(onnx_pyfunc, config)
    # predictions = onnx_pyfunc.predict(X.numpy())
    # print(predictions)
