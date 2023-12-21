import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms


class MnistData(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        # self.save_hyperparameters()
        self.batch_size = batch_size

        resize = transforms.Resize((32, 32))
        to_rgb = transforms.Lambda(lambda image: image.convert("RGB"))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.transform = transforms.Compose(
            [resize, to_rgb, transforms.ToTensor(), normalize]
        )

        self.train_dataset = datasets.MNIST(
            root="./data", train=True, download=False, transform=self.transform
        )

        self.test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=False,
            transform=self.transform,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return train_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return test_loader
