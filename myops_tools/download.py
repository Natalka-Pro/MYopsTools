import os

from dvc.api import DVCFileSystem


def load_dataset(config):
    print("<<< download.main:")
    if os.path.isdir(f"./data/{config.dataset}"):
        print(f"The folder \"{config.dataset}\" exists")
    else:
        print("Dataset download begins")
        url = "https://github.com/Natalka-Pro/myops_tools.git"
        fs = DVCFileSystem(url, rev="main")
        fs.get("./data", "./", recursive=True)
        print("Dataset download completed")
    print(">>>")


def load_onnx(config):
    print("<<< download.load_onnx:")
    if os.path.isfile(f"./{config.train.model_save_onnx}"):
        print(f"The file \"{config.train.model_save_onnx}\" exists")
    else:
        print("onnx-file download begins")
        url = "https://github.com/Natalka-Pro/myops_tools.git"
        fs = DVCFileSystem(url, rev="main")
        fs.get(
            "triton/model_repository/onnx-AlexNet/1",
            "triton/model_repository/onnx-AlexNet/",
            recursive=True,
        )
        print("onnx-file download completed")
    print(">>>")
