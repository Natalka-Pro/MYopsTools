import fire

from myops_tools.infer import main as infer
from myops_tools.train import main as train

if __name__ == "__main__":
    fire.Fire(
        {
            'train': train,
            'infer': infer,
        }
    )
