import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TestsDiffsDataset(Dataset):
    def __init__(
        self, 
        commits: list[str], 
        tests_embs: pd.DataFrame,
        diffs_embs: pd.DataFrame,
        test_results: list[pd.DataFrame] = None, 
    ):
        self.commits = commits
        self.tests_embs = tests_embs
        self.diffs_embs = diffs_embs
        self.test_results = test_results

        self.diffs_embs = self.diffs_embs.fillna(0)
        # normalization
        self.tests_embs = self._normalize(self.tests_embs)
        self.diffs_embs = self._normalize(self.diffs_embs)
    

    def __len__(self):
        return len(self.commits)
    

    def __getitem__(self, idx):
        diffs_embs = self.diffs_embs[
            self.diffs_embs["vcs_commit_sha"] == self.commits[idx]
        ].drop(columns=["vcs_commit_sha"])

        diffs_embs = torch.tensor(diffs_embs.values, dtype=torch.float)
        if self.test_results is not None:
            df_targets = self.test_results[idx]
            df_tests_embs = self.tests_embs[self.tests_embs.index.isin(df_targets["allure_id"])]
            df_tests_embs = df_targets.merge(
                df_tests_embs, how="inner", left_on="allure_id", right_index=True
            )
            tests_embs = torch.tensor(df_tests_embs.drop(["allure_id", "status"], axis=1).values, dtype=torch.float)
            targets = torch.tensor(df_tests_embs["status"].values, dtype=torch.float)
            return diffs_embs, tests_embs, targets
        
        else:
            tests_embs = torch.tensor(self.tests_embs.drop(["allure_id"], axis=1).values, dtype=torch.float)
            return diffs_embs, tests_embs
        
        
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        emb_cols = df.columns.drop(
            df.select_dtypes(exclude=np.number).columns
        )
        std_scaler = StandardScaler()
        df[emb_cols] = std_scaler.fit_transform(df[emb_cols].to_numpy())
        return df