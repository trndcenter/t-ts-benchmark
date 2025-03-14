import os
import json
import pickle
import nip

from pathlib import Path
from typing import Annotated
import typer.core
from typer import Option

import src.preprocessing
import src.models as models


def main(
    input: Annotated[
        Path,
        Option(
            "--input",
            "-in",
            help="Path to inference data dir",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path("data_inference"),
    output: Annotated[
        Path,
        Option(
            "--output",
            "-out",
            help="Path to output prediction file",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = os.path.join("output", "output.json"),
    model_type: Annotated[
        models.ModelType,
        Option(
            "--model",
            "-m",
            help="Model type",
        ),
    ] = models.ModelType.xgb,
    predict_limit: Annotated[
        int,
        Option(
            "--topN",
            "-N",
            envvar="PTS_TOP_N_PREDICT",
            min=1,
            help="Number of highly probable tests in the output"
        )
    ] = None
):
    print(
        f"""Working with
    model type: {model_type}
    input data dir: {input},
    output prediction file: {output}
    """
    )

    cur_path = Path(__file__).parent.resolve()
    path_config = os.path.join(cur_path, "configs", model_type.value, "predict.yaml")
    path_inference_dir = os.path.join(cur_path, "inference", model_type.value)

    config = nip.parse(path_config)
    config["creator"]["path_data"] = str(input)
    config["creator"]["path_inference"] = str(path_inference_dir)

    creator = nip.construct(config["creator"])
    transformer = nip.construct(config["transformer"])
    with open(os.path.join(path_inference_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    print("Creating dataset...")
    dataset = creator.create_dataset(train_size=0)

    print("Predicting...")
    X_test = transformer.transform(dataset)
    proba = [p[1] for p in model.predict_proba(X_test)]

    # Saving predictions
    df_tests = dataset[["allure_id", "test_file_path", "test_method"]].copy()
    df_tests["proba"] = proba
    df_tests = df_tests.sort_values(by="proba", ascending=False)
    if predict_limit is not None:
        df_tests = df_tests.head(predict_limit)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    result = json.loads(df_tests.to_json(orient="records"))
    with open(output, "w") as f:
        json.dump(result, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    typer.core.rich = None
    typer.run(
        main,
    )
