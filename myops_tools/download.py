import os

from dvc.api import DVCFileSystem


def main(config):
    if os.path.isdir(f"./data/{config.dataset}"):
        print(f"The folder \"{config.dataset}\" exists")
    else:
        print("Dataset download begins")
        url = "https://github.com/Natalka-Pro/myops_tools.git"
        fs = DVCFileSystem(url, rev="main")
        fs.get("./data", "./", recursive=True)
        print("Dataset download completed")
