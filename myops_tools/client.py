from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import np_to_triton_dtype

from .data import MnistDataModule
from .infer import save_to_csv


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton(input: np.ndarray):
    triton_client = get_client()

    infer_input = InferInput(
        name="images",
        shape=input.shape,
        datatype=np_to_triton_dtype(input.dtype),
    )

    infer_input.set_data_from_numpy(input, binary_data=True)

    infer_output = InferRequestedOutput("proba_distr", binary_data=True)
    query_response = triton_client.infer(
        "onnx-AlexNet", [infer_input], outputs=[infer_output]
    )
    probas = query_response.as_numpy("proba_distr")
    return probas


def main(config):
    print("<<< client.main:")
    test_loader = MnistDataModule(
        batch_size=config.infer.batch_size
    ).test_dataloader()

    pred_true = np.empty((0, 2), int)  # size N x 2
    criterion = nn.CrossEntropyLoss()
    loss = 0

    for X, y_true in test_loader:
        X = X.detach().numpy()

        y_prob = call_triton(X)
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
