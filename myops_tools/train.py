import pytorch_lightning as pl
import torch
import torch.nn as nn

from .data import MnistDataModule
from .models import AlexNet


class TrainModule(pl.LightningModule):
    def __init__(self, config, git_commit_id):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model = AlexNet(config.model.n_classes, config.model.dropout)
        self.criterion = nn.CrossEntropyLoss()

        # self.train_loader = train_loader
        # self.n_epochs = n_epochs
        # self.device = device

    def training_step(self, batch, batch_idx):
        X, y_true = batch

        y_prob = self.model(X)
        loss = self.criterion(y_prob, y_true)

        _, predicted_labels = torch.max(y_prob, 1)
        accuracy = (predicted_labels == y_true).sum() / predicted_labels.size(
            0
        )

        logs = {'train_loss': loss.detach(), 'train_accuracy': accuracy}

        self.log_dict(
            logs, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y_true = batch

        y_prob = self.model(X)
        loss = self.criterion(y_prob, y_true)

        _, predicted_labels = torch.max(y_prob, 1)
        accuracy = (predicted_labels == y_true).sum() / predicted_labels.size(
            0
        )

        logs = {'test_loss': loss.detach(), 'test_accuracy': accuracy}

        self.log_dict(
            logs, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.train.learning_rate
        )
        return optimizer


def main(config):
    pl.seed_everything(config.train.random_seed)
    # torch.set_float32_matmul_precision("medium")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Device type: {device}")

    dm = MnistDataModule(batch_size=config.train.batch_size)
    model = TrainModule(config, 1)

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config.artifacts.experiment_name,
        tracking_uri=config.artifacts.tracking_uri,
    )

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=config.callbacks.max_depth),
    ]

    trainer = pl.Trainer(
        accelerator=config.model.accelerator,
        devices=config.train.devices,
        max_epochs=config.train.n_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)

    # trainer.fit(model,
    #             train_dataloaders=dm.train_dataloader(),
    #             val_dataloaders=dm.val_dataloader())

    torch.save(model.model.state_dict(), config.train.model_save)
    print(f"Model is saved with the name \"{config.train.model_save}\"")
