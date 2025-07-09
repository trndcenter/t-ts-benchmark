import os
import pickle
import nip

from pathlib import Path
from typing import Annotated
import typer
from typer import Option

import src.preprocessing
import src.models as models


def main(
    path_data_dir: Annotated[
        Path,
        Option(
            "--data",
            "-d",
            help="Path to data",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = os.path.join("data"),
    model_type: Annotated[
        models.ModelType,
        Option(
            "--model",
            "-m",
            help="Model to train",
        ),
    ] = models.ModelType.xgb,
    num_train_commits: Annotated[
        int,
        Option(
            "--num_train_commits",
            "-t",
            help="Number of commits to train on",
            min=1,
        ),
    ] = 1,
):
    print(
        f"""Working with
    model type: {model_type}
    data dir: {path_data_dir},
    number of commits to train on: {num_train_commits}
    """
    )

    cur_path = Path(__file__).parent.resolve()
    path_config = os.path.join(cur_path, "configs", model_type.value, "train.yaml")
    path_inference_dir = os.path.join(cur_path, "inference", model_type.value)

    config = nip.parse(path_config)
    config["creator"]["path_data"] = str(path_data_dir)
    config["creator"]["path_inference"] = str(path_inference_dir)
    os.makedirs(path_inference_dir, exist_ok=True)

    creator = nip.construct(config["creator"])
    transformer = nip.construct(config["transformer"])
    model = nip.construct(config["model"])

    creator.create_dataset(start_commit=-num_train_commits)

    print("Training model...")
    target = config["target"].to_python()
    X, y = (
        creator.train_dataset.drop(columns=[target]),
        creator.train_dataset[target],
    )
    X, y = transformer.fit_transform(X, y)
    model.fit(X, y)

    with open(os.path.join(path_inference_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Saving files to create a dataset for inference
    if config["transformer"]["preprocessing_files"].to_python():
        train_files = transformer.preprocessed_files
    else:
        train_files = creator.train_files
    with open(os.path.join(path_inference_dir, "train_files.pkl"), "wb") as f:
        pickle.dump(train_files, f)

    # Saving a dataset
    if config["save_dataset"].to_python():
        creator.train_dataset.to_pickle(
            os.path.join(path_data_dir, "interface_train_dataset.pkl")
        )

    print("Done!")


if __name__ == "__main__":
    typer.run(main)
