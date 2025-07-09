import os
import nip

from typing import Union
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from .commit_dataset_creator import CommitDatasetCreator
from .cba.embedders import TestsEmbedder, DiffsEmbedder
from .cba.tests_diffs_dataset import TestsDiffsDataset
from ..models.csa_model import CSAModel

@nip.nip
class CSACommitDatasetCreator(CommitDatasetCreator):
    def __init__(
        self,
        path_data: Union[str, Path] = Path('data'),
        inference: bool = False,
        save_csa_model: bool = False,
        without_unstable: bool = False,
        csa_model_name: str = "bigcode/starencoder",
        csa_input_size: int = 768,
        csa_hidden_size: int = 700,
        csa_nonlinear: nn.Module = nn.ReLU(),
        csa_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        csa_lr: float = 1e-4,
        train_epochs: int = 15,
        path_inference: Union[str, Path] = os.path.join("inference", "csaxgb"),
    ):
        self.csa_model = CSAModel(
            input_size=csa_input_size, hidden_size=csa_hidden_size,
            nonlinear=csa_nonlinear, optimizer=csa_optimizer,
            lr=csa_lr
        )
        super().__init__(
            path_data=path_data,
            inference=inference,
            without_unstable=without_unstable,
            path_inference=path_inference,
        )
        self.train_epochs = train_epochs
        self.save_csa_model = save_csa_model
        self.path_save_csa_model = os.path.join(path_inference, "checkpoint.pt")
        self.tests_embedder = TestsEmbedder(csa_model_name)
        self.diffs_embedder = DiffsEmbedder(csa_model_name)

        if self.is_inference: 
            self.csa_model.load_model_optimizer(self.path_save_csa_model)

    def create_dataframes(
        self,
        start_commit: int = None,
        last_commit: int = None,
        num_cross_files: int = 3,
        train_size: float = 1.0,
        min_commit_freq: int = 0,
        max_commit_freq: float = 1.0
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if train_size == 0.0:
            train_files = self._get_train_files(self._path_inference)

            self.test_commits = self._get_commits(
                start_commit=start_commit, last_commit=last_commit
            )

            self.test_dataframes = self.create_commits_tests_crossfiles(
                self.test_commits,
                files=train_files,
                num_cross_files=num_cross_files,
            )
            return self.test_dataframes
        else:
            commits = self._get_commits(
                start_commit=start_commit, last_commit=last_commit
            )
            i_split = int(len(commits) * train_size)
            self.train_commits, self.test_commits = (
                commits[:i_split],
                commits[i_split:],
            )

            self.train_csa_model(self.train_commits)

            self.train_files = self._get_files(self.train_commits, min_cf=min_commit_freq, max_cf=max_commit_freq)
            print("Creating train dataset...")
            self.train_dataframes = self.create_commits_tests_crossfiles(
                self.train_commits,
                files=self.train_files,
                num_cross_files=num_cross_files,
            )

            if train_size < 1.0:
                print("Creating test dataset...")
                self.test_dataframes = self.create_commits_tests_crossfiles(
                    self.test_commits,
                    files=self.train_files,
                    num_cross_files=num_cross_files,
                )
                return self.train_dataframes, self.test_dataframes

            return self.train_dataframes
        

    def create_df_tests(self, commits: list[str]) -> pd.DataFrame:
        df_tests = super().create_df_tests(commits)
        df_tests = self.add_csa_score(df_tests)
        return df_tests


    def train_csa_model(self, commits: list[str]) -> None:
        """
        Train the CSA model using data collected from the provided commits.
        
        :param commits: A list of commits' shas used to gather data for training.
        """
        df_tests = super().create_df_tests(commits)
        dataset = self._create_csa_dataset(df_tests)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.csa_model.fit(max_epoch=self.train_epochs, dataloader=dataloader)
        
        if not self.is_inference and self.save_csa_model:
            self.csa_model.save_model_optimizer(self.path_save_csa_model)


    def predict_csa_model(self, df_tests_ids: pd.DataFrame):
        """Predict csa score with CSA model for every test"""
        cols = ["allure_id", "vcs_commit_sha"]
        commits = df_tests_ids["vcs_commit_sha"].unique().tolist()
        tests_embs = self._create_tests_embs(commits, df_tests_ids)
        diff_embs  = self._create_diffs_embs(commits)
        df_tests_csa = []
        for commit in commits:
            df_commit = df_tests_ids[df_tests_ids["vcs_commit_sha"] == commit]
            df_commit_tembs = df_commit[cols].merge(
                tests_embs, how="inner", left_on="allure_id", right_index=True
            )
            commit_tembs = df_commit_tembs.drop(["vcs_commit_sha"], axis=1)

            commit_dembs = diff_embs[diff_embs["vcs_commit_sha"] == commit]
            dataset = TestsDiffsDataset([commit], commit_tembs, commit_dembs)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
            csa_score = self.csa_model.predict_proba(dataloader)
            df_tests_csa.append(pd.concat(
                [df_commit.reset_index(drop=True), pd.DataFrame(csa_score, columns=["csa_score"])], 
                axis=1
            ))
        csa_score = pd.concat(df_tests_csa, axis=0, ignore_index=True)
        return csa_score


    def add_csa_score(self, df_tests: pd.DataFrame) -> pd.DataFrame:
        df_tests = self.predict_csa_model(df_tests)
        df_tests = df_tests.drop(columns=["test_content"])
        return df_tests


    def _create_csa_dataset(self, df_tests: pd.DataFrame) -> TestsDiffsDataset:
        """
        Create embeddings for tests and diffs, and load them into a `TestsDiffsDataset` for use with the CSA model.
    
        The method performs the following steps:
        1. Extracts unique commit SHAs from the provided :df_tests: DataFrame.
        2. Creates the test embeddings and diff embeddings.
        3. If `self.is_inference` 
            - is `True`, it initializes a `TestsDiffsDataset` with the commits, 
            test embeddings, and diff embeddings.
            - is `False`, it also collects test results using 
            the `_collect_test_results` method and initializes a `TestsDiffsDataset` 
            with the commits, test embeddings, diff embeddings, and test results.
            
        :return: A `TestsDiffsDataset` object containing the commits, test embeddings,
        diff embeddings, and optionally test results.
    """
        commits = df_tests["vcs_commit_sha"].unique().tolist()
        tests_embs = self._create_tests_embs(commits, df_tests)
        diffs_embs = self._create_diffs_embs(commits)
        if self.is_inference:
            dataset = TestsDiffsDataset(commits, tests_embs, diffs_embs)
        else:
            test_results = self._collect_test_results(commits, df_tests)
            dataset = TestsDiffsDataset(commits, tests_embs, diffs_embs, test_results)  
        return dataset

    
    def _create_tests_embs(self, commits: list[str], df_all_tests: pd.DataFrame) -> pd.DataFrame:
        """
        Build embeddings for tests using the :ivar:`self.csa_model_name` embedder.
        The embeddings are created from the earliest commits to the latest.
        If an embedding for a test has already been built, it is not rebuilt for later commits
        """
        built_allures = set()
        list_ctembs = []
        for commit in tqdm(commits, desc="Creating tests' embeddings"):
            df_tests = df_all_tests[df_all_tests["vcs_commit_sha"] == commit][["allure_id", "test_content"]]
            missed_allures = set(df_tests["allure_id"].tolist()) - built_allures
            if len(missed_allures) == 0:
                continue
            df_tests = df_tests[df_tests["allure_id"].isin(missed_allures)]
            df_tests = df_tests.drop_duplicates("allure_id")
            df_tembs = self.tests_embedder.create_vectors(df_tests=df_tests)
            list_ctembs.append(df_tembs)
            built_allures |= missed_allures
        df_commits_tembs = pd.concat(list_ctembs, axis=0)
        return df_commits_tembs

    def _create_diffs_embs(self, commits: list[str]) -> list[pd.DataFrame]:
        """Creating embeddings for files' diff using :ivar self.csa_model_name: embedder"""
        list_dembs = []
        for commit in tqdm(commits, desc="Creating diffs' embeddings"):
            df_dembs = self.diffs_embedder.create_vectors(
                os.path.join(self._path_file_stats, f"{commit}.csv"),
                granularity="files"
            )
            df_dembs["vcs_commit_sha"] = commit
            list_dembs.append(df_dembs)
        df_commits_dembs = pd.concat(list_dembs, axis=0, ignore_index=True)
        return df_commits_dembs
    
    def _collect_test_results(self, commits: list[str], df_tests: pd.DataFrame) -> list[pd.DataFrame]:
        """Collect test results (allure_id and status) for each commit"""
        test_results = []
        for commit in commits:
            results = df_tests[df_tests["vcs_commit_sha"] == commit][["allure_id", "status"]]
            test_results.append(results)
        return test_results
    

    def _get_file_features(self, commits: list[str]) -> list[str]:
        """Find the list of features for a file"""
        df_file_stats = pd.read_csv(
            os.path.join(self._path_file_stats, commits[0] + ".csv"), delimiter=";"
        )

        file_features = df_file_stats.columns.drop(
            [
                "last_commit_changed_files",
                "last_commit_sha",
                "last_authored_date",
                "file_extension",
                "file_name",
                "code_diff"
            ]
        ).to_list()

        file_features.append("changed")

        return file_features
