import torch
from torchvision import datasets, transforms


class MnistData:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        resize = transforms.Resize((32, 32))
        to_rgb = transforms.Lambda(lambda image: image.convert("RGB"))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.transform = transforms.Compose(
            [resize, to_rgb, transforms.ToTensor(), normalize]
        )

    def train_loader(self):
        dataset_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=self.batch_size, shuffle=True
        )
        return train_loader

    def test_loader(self):
        dataset_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transform
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=dataset_test, batch_size=self.batch_size, shuffle=False
        )
        return test_loader
