import os
import pandas as pd
import numpy as np
import nip

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier


@nip.nip
class RedhModel:
    def __init__(
        self,
        path_dataset: str,
        split_date: str,
        drop_cols: list[str],
        cat_cols: list[str],
        params: dict,
        early_stopping_rounds: int = 10,
    ):
        assert os.path.isfile(path_dataset), "Data not found"
        self.raw_data = pd.read_csv(path_dataset)
        self.split_date = pd.to_datetime(split_date)
        self.drop_cols = drop_cols
        self.cat_cols = cat_cols
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds

    def train_test(self) -> list[float]:
        train_dataset, test_dataset = self._split_by_date()

        # clean
        train_dataset = train_dataset.drop(columns=self.drop_cols)
        test_dataset = test_dataset.drop(columns=self.drop_cols)

        # set categories
        train_dataset[self.cat_cols] = train_dataset[self.cat_cols].astype("category")
        test_dataset[self.cat_cols] = test_dataset[self.cat_cols].astype("category")

        y_train = train_dataset.pop("test_status")
        self.y_test = test_dataset.pop("test_status")

        # train
        model = XGBClassifier(
            **self.params,
            enable_categorical=True,
            early_stopping_rounds=self.early_stopping_rounds
        )
        model.fit(
            train_dataset, y_train, eval_set=[(test_dataset, self.y_test)], verbose=0
        )

        # get failed scores
        y_proba = np.array([p[1] for p in model.predict_proba(test_dataset)])

        return y_proba

    def find_metrics(self, threshold: float):
        y_proba = self.train_test()
        y_pred = np.where(y_proba > threshold, 1, 0)
        y_true = np.array(self.y_test)

        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
        }

    def _split_by_date(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw_data = self.raw_data.copy()
        raw_data["date"] = pd.to_datetime(self.raw_data["date"])
        train_dataset = raw_data[raw_data["date"] <= self.split_date]
        test_dataset = raw_data[raw_data["date"] > self.split_date]
        return train_dataset, test_dataset


