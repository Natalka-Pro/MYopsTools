import pandas as pd
import torch
import torch.nn as nn
from train import AlexNet, dataset_loaders, get_accuracy, validate


def prediction(test_loader, model, device):
    ans = torch.tensor([]).to(device)
    for X, y_true in test_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        y = model(X)
        _, predicted_labels = torch.max(y, dim=1)
        predicted_labels = torch.stack([predicted_labels, y_true], dim=1)
        ans = torch.cat([ans, predicted_labels])
    return ans


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{DEVICE = }")

    test_loader = dataset_loaders(train=False)
    model = AlexNet(num_classes=10).to(DEVICE)
    model_name = "model.pth"
    model.load_state_dict(torch.load(model_name))
    print(f"Model \"{model_name}\" is loaded")
    criterion = nn.CrossEntropyLoss()

    test_acc = get_accuracy(model, test_loader, DEVICE).to("cpu").item()
    with torch.no_grad():
        _, test_loss = validate(test_loader, model, criterion, DEVICE)

    print(f"Validation: accuracy = {test_acc}, loss = {test_loss}")

    # model prediction
    ans = prediction(test_loader, model, DEVICE)

    ans = ans.to("cpu")
    df = pd.DataFrame(ans, columns=["Predicted labels", "True labels"])

    file_name = "labels.csv"
    df.to_csv(file_name)
    print(
        f"A matrix of size {tuple(ans.size())} ",
        f"was written to the file \"{file_name}\"\n",
        "Column names - \"Predicted labels\" and \"True labels\"",
        sep="",
    )


if __name__ == "__main__":
    main()
