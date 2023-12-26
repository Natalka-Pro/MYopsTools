import fire
from hydra import compose, initialize

from myops_tools.client import main as client
from myops_tools.download import load_dataset, load_onnx
from myops_tools.infer import infer, run_server
from myops_tools.train import main as train


def hydra_config():
    initialize(config_path="configs", version_base="1.3")
    config = compose(config_name="config.yaml")
    return config


def training():
    config = hydra_config()
    load_dataset(config.download)
    train(config)


def infering():
    config = hydra_config()
    load_dataset(config.download)
    infer(config)


def running_server():
    config = hydra_config()
    load_dataset(config.download)
    load_onnx(config)
    run_server(config)


def running_client():
    config = hydra_config()
    load_dataset(config.download)
    client(config)


if __name__ == "__main__":
    fire.Fire(
        {
            'train': training,
            'infer': infering,
            'run_server': running_server,
            'client': running_client,
        }
    )
