import git
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

    def training_step(self, batch, batch_idx):
        X, y_true = batch

        y_prob = self.model(X)
        loss = self.criterion(y_prob, y_true)

        _, predicted_labels = torch.max(y_prob, axis=1)
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

        _, predicted_labels = torch.max(y_prob, axis=1)
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

    def save_model(self):
        torch.save(self.model.state_dict(), self.config.train.model_save)
        print(
            f"Model is saved with the name \"{self.config.train.model_save}\""
        )


def lightning_log(config):
    pl.seed_everything(config.train.random_seed)
    torch.set_float32_matmul_precision("medium")
    datamodule = MnistDataModule(batch_size=config.train.batch_size)

    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit_id = repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        git_commit_id = "git repository not found"

    trainmodule = TrainModule(config, git_commit_id)

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config.artifacts.experiment_name,
        # tracking_uri=config.artifacts.tracking_uri,
        tracking_uri=f"http://{config.artifacts.host}:{config.artifacts.port}",
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

    trainer.fit(trainmodule, datamodule=datamodule)
    # trainer.fit(model,
    #             train_dataloaders=dm.train_dataloader(),
    #             val_dataloaders=dm.val_dataloader())
    trainmodule.save_model()
    return trainmodule.model


def onnx_save(model, config):
    # (https://learn.microsoft.com/ru-ru/windows/ai/windows-ml/tutorials/pytorch-convert-model)
    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        config.train.model_save_onnx,  # where to save the model
        export_params=True,
        # store the trained parameter weights inside the model file
        # opset_version=10,
        # the ONNX version to export the model to
        # do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=['images'],  # the model's input names
        output_names=['proba_distr'],  # the model's output names
        dynamic_axes={
            'images': {0: 'batch_size'},  # variable length axes
            'proba_distr': {0: 'batch_size'},
        },
    )


def main(config):
    model = lightning_log(config).eval()
    onnx_save(model, config)
