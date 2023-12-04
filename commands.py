import fire
from hydra import compose, initialize

from myops_tools.infer import main as infer
from myops_tools.train import main as train


def training():
    initialize(config_path="configs", version_base="1.3")
    config = compose(config_name="config.yaml")
    train(config["train"])


def infering():
    initialize(config_path="configs", version_base="1.3")
    config = compose(config_name="config.yaml")
    infer(config["infer"])


if __name__ == "__main__":
    fire.Fire(
        {
            'train': training,
            'infer': infering,
        }
    )
