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

from train import AlexNet, get_accuracy, validate, dataset_loaders

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')

criterion = nn.CrossEntropyLoss()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE = }")

test_loader   = dataset_loaders(train = False)

model = AlexNet(num_classes = 10).to(DEVICE)
model.load_state_dict(torch.load("model.pth"))

test_acc = get_accuracy(model, test_loader, DEVICE).to("cpu").item()
with torch.no_grad():
    _, test_loss = validate(test_loader, model, criterion, DEVICE)

print(f"{test_acc = }, ", f"{test_loss = }")

ans = torch.tensor([]).to(DEVICE)

for X, y_true in test_loader:
    X = X.to(DEVICE)
    y_true = y_true.to(DEVICE)
    y_prob = model(X)
    _, predicted_labels = torch.max(y_prob, 1)
    predicted_labels = torch.stack([predicted_labels, y_true], dim = 1)
    # print(predicted_labels.size())
    ans = torch.cat([ans, predicted_labels])
    # print(ans.size())

import pandas as pd
ans = ans.to("cpu")
df = pd.DataFrame(ans, columns=['Predicted labels', 'True labels'])

file_name = "labels.csv"
df.to_csv(file_name)
print(f"A matrix of size {ans.size()} was written to the file '{file_name}'",
      f"column names - 'Predicted labels' and 'True labels'")
