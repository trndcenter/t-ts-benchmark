import os
import pandas as pd
import numpy as np
import nip
from datetime import datetime

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def apfd(y_true: list[int], y_proba: list[float], normalize: bool = False, k: int = 50) -> float:
    y_true, y_proba = pd.Series(y_true), pd.Series(y_proba)
    y_proba_sorted = y_proba.sort_values(ascending=False)
    sorted_idx = y_proba_sorted.index.to_list()
    y_true_sorted = y_true.loc[sorted_idx].reset_index(drop=True)

    m = y_true[y_true == 1].shape[0]
    if m == 0:
        return 0
    n = y_true.shape[0]

    sum_tfi = 0
    p = 1
    if normalize:
        for i in range(m):
            tfi = y_true_sorted[y_true_sorted == 1].index[i]
            if tfi <= k:
                sum_tfi += tfi
        y_true_sorted_at_k = y_true_sorted.iloc[:k]
        p = y_true_sorted_at_k[y_true_sorted_at_k == 1].shape[0] / m
    else:
        for i in range(m):
            sum_tfi += y_true_sorted[y_true_sorted == 1].index[i]

    return p - (sum_tfi / (n * m)) + p / (2 * n)


@nip.nip
class IofrolGsdtsrModel:
    def __init__(
        self,
        path_dataset: str,
        drop_cols: list[str],
        test_size: float = 0.2,
        params: dict = None,
        periods: list[int] = [7, 14, 56],
        early_stopping_rounds: int = 10,
        verbose: int = 0,
    ):
        assert os.path.isfile(path_dataset), "Dataset not found"
        self.model = None
        self.y_test = None
        self.dataset = pd.read_csv(path_dataset, sep=";")

        self.drop_cols = drop_cols
        self.test_size = test_size
        self.periods = periods
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self.time_metrics = {}

    def train_test(self) -> None:
        start_time = datetime.now()

        cleaned = self._clean(self.dataset)
        with_features = self._create_features(cleaned)

        train_cycles, test_cycles = train_test_split(
            with_features["Cycle"], test_size=self.test_size, shuffle=False
        )
        train_dataset, test_dataset = (
            with_features[with_features["Cycle"].isin(train_cycles)].copy(),
            with_features[with_features["Cycle"].isin(test_cycles)].copy(),
        )

        self.test_cycles, self.test_duration = test_dataset['Cycle'].values, test_dataset['Duration'].values

        train_dataset.drop(["Cycle"], axis=1, inplace=True)
        test_dataset.drop(["Cycle"], axis=1, inplace=True)

        X_train, y_train = (
            train_dataset.drop(columns=["status"]),
            train_dataset["status"],
        )
        X_test, self.y_test = test_dataset.drop(columns=["status"]), test_dataset[
            "status"
        ].values

        self.preprocess_time = datetime.now() - start_time

        start_time = datetime.now()
        self.model = XGBClassifier(
            **self.params, early_stopping_rounds=self.early_stopping_rounds
        )
        self.model.fit(
            X_train, y_train, eval_set=[(X_test, self.y_test)], verbose=self.verbose
        )
        self.train_time = datetime.now() - start_time

        start_time = datetime.now()
        self.y_proba = np.array([p[1] for p in self.model.predict_proba(X_test)])
        self.inference_time = datetime.now() - start_time

    def find_apfd(self, ratio: float = 1.0):
        normalize = True if ratio != 1.0 else False

        return apfd(self.y_test, self.y_proba, k=int(self.y_test.shape[0] * ratio), normalize=normalize)
    
    def find_time(self):
        df_time = pd.DataFrame({'cycles': self.test_cycles, 'true': self.y_test, 'proba': self.y_proba, 'duration': self.test_duration})

        cycles = df_time['cycles'].unique().tolist()
        all_time = {
            "FT": [],
            "LT": [],
            "AT": [],
            "RT": []
        }
        for cycle in cycles:
            df_one_cycle = df_time[df_time['cycles'] == cycle]
            ft, lt = self._get_ft(df_one_cycle), self._get_lt(df_one_cycle)
            all_time["FT"].append(ft)
            all_time["LT"].append(lt)
            all_time["AT"].append((ft + lt) / 2)
            all_time["RT"].append(self._get_rt(df_one_cycle))

        mean_time = {}
        for key in all_time:
            mean_time[key] = (np.mean(all_time[key]) + self.inference_time).total_seconds()
        mean_time['PT'] = (self.preprocess_time + self.train_time + self.inference_time).total_seconds() + mean_time['LT']
        mean_time['TT'] = mean_time['PT'] + mean_time['LT']

        return mean_time
    
    def _clean(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        cleaned = raw_data.drop(self.drop_cols, axis=1).rename(
            columns={"Verdict": "status"}
        )
        cleaned["int_last_results"] = cleaned["LastResults"].apply(
            self._convert_str_to_lst
        )
        return cleaned

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.periods:
            df[f"failure_rate_{period}_cycles"] = df["int_last_results"].apply(
                self._failure_rate, args=(period,)
            )

        with_failure_rate = df.drop(columns=["LastResults", "int_last_results"])

        return with_failure_rate

    def _convert_str_to_lst(self, x: pd.Series) -> list[int]:
        x = x.strip("[]").replace(" ", "").split(",")
        return [int(el) if el else 0 for el in x]

    def _failure_rate(self, last_resuts: list[int], period: int) -> float:
        return sum(last_resuts[:period]) / period
    
    def _get_ft(self, df_time: pd.DataFrame) -> datetime:
        start = datetime.now()
        df = df_time.sort_values(by='proba', ascending=False)
        for i, row in df.iterrows():
            if row['true'] == 1:
                break
        return datetime.now() - start

    def _get_lt(self, df_time: pd.DataFrame) -> datetime:
        all_tests = df_time['true'].sum()
        found_failed = 0
        start = datetime.now()
        df = df_time.sort_values(by='proba', ascending=False)
        for i, row in df.iterrows():
            if found_failed == all_tests:
                break
        return datetime.now() - start

    def _get_rt(self, df_time: pd.DataFrame) -> datetime:
        start = datetime.now()
        df_time.sort_values(by='proba', ascending=False)
        return datetime.now() - start
