import os
import pickle
from pathlib import Path
from typing import Union
import pandas as pd
import nip

# Imports
from ..commit_dataset_creator import CommitDatasetCreator
from . import graph_clustering
from . import modular_clustering


@nip.nip
class ClusteringDatasetCreator(CommitDatasetCreator):
    """
    This class creates a dataset from commit and test data,
    inheriting most functionality from CommitDatasetCreator.
    It excludes cross-files features and adds options for graph-based and modular-based clustering.
    
    Parameters:
      - path_data: Path to the data (default: "data")
      - inference: Flag for inference mode; if True, saved settings are used
      - without_unstable: Flag to exclude unstable tests
      - path_inference: Path for saving/loading inference files (default: "inference/gbcxgb")
      - graph_based_clustering: If True, apply graph-based clustering
      - modular_based_clustering: If True, apply modular-based clustering
    """
    
    def __init__(
        self,
        path_data: Union[str, Path] = Path("data"),
        inference: bool = False,
        without_unstable: bool = False,
        path_inference: Union[str, Path] = os.path.join("inference", "xgb"),
        graph_based_clustering: bool = False,
        modular_based_clustering: bool = False,
    ):
        
        super().__init__(
            path_data=path_data,
            inference=inference,
            without_unstable=without_unstable,
            path_inference=path_inference,
        )
        self._path_data = path_data
        self._path_inference = path_inference
        self.graph_based_clustering = graph_based_clustering
        self.modular_based_clustering = modular_based_clustering

    def create_dataset(
        self,
        start_commit: int = None,
        last_commit: int = None,
        train_size: float = 1.0,
        min_commit_freq: int = 0,
        max_commit_freq: float = 1.0,
    ) -> pd.DataFrame:
        """
        Creates a dataset by merging commit and test data.
        This overridden method excludes cross-files features.
        """
        dataframes = self.create_dataframes(
            start_commit=start_commit,
            last_commit=last_commit,
            train_size=train_size,
            min_commit_freq=min_commit_freq,
            max_commit_freq=max_commit_freq,
        )

        if train_size == 0.0:
            self.test_dataset = self.merge_dataframes(*dataframes)
            return self.test_dataset
        elif train_size == 1.0:
            self.train_dataset = self.merge_dataframes(*dataframes)
            return self.train_dataset
        else:
            train_dataframes, test_dataframes = dataframes
            self.train_dataset = self.merge_dataframes(*train_dataframes)
            self.test_dataset = self.merge_dataframes(*test_dataframes)
            return self.train_dataset, self.test_dataset

    def create_dataframes(
        self,
        start_commit: int = None,
        last_commit: int = None,
        train_size: float = 1.0,
        min_commit_freq: int = 0,
        max_commit_freq: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Overridden method to create DataFrames containing commit and test data
        (without cross-files features).
        """
        if train_size == 0.0:
            # For inference mode: use the list of files from the training set.
            train_files = self._get_train_files(self._path_inference)
            self.test_commits = self._get_commits(start_commit=start_commit, last_commit=last_commit)
            self.test_dataframes = self.create_commits_tests(self.test_commits, files=train_files)
            return self.test_dataframes
        else:
            commits = self._get_commits(start_commit=start_commit, last_commit=last_commit)
            i_split = int(len(commits) * train_size)
            self.train_commits, self.test_commits = commits[:i_split], commits[i_split:]
            self.train_files = self._get_files(self.train_commits, min_cf=min_commit_freq, max_cf=max_commit_freq)
            print("Creating train dataset...")
            self.train_dataframes = self.create_commits_tests(self.train_commits, files=self.train_files)
            if train_size < 1.0:
                print("Creating test dataset...")
                self.test_dataframes = self.create_commits_tests(self.test_commits, files=self.train_files)
                return self.train_dataframes, self.test_dataframes
            return self.train_dataframes

    def create_commits_tests(self, commits: list[str], files: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gathers commit and test data without cross-files features.
        """
        df_commits = self.create_df_commit(commits, files=files)
        df_tests = self.create_df_tests(commits)
        return df_commits, df_tests

    def merge_dataframes(
        self,
        df_commits: pd.DataFrame,
        df_tests: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges the commit and test DataFrames.
        If graph-based or modular-based clustering is enabled, the commit DataFrame is transformed first.
        """
        if self.graph_based_clustering:
            if self.is_inference:
                df_commits = self._apply_graph_based_clustering_inference(df_commits)
            else:
                df_commits = self._apply_graph_based_clustering(df_commits)
        elif self.modular_based_clustering:
            if self.is_inference:
                df_commits = self._apply_modular_based_clustering_inference(df_commits)
            else:
                df_commits = self._apply_modular_based_clustering(df_commits)

        df_tests["allure_id"] = df_tests["allure_id"].astype("int64")
        df_merged = df_commits.merge(df_tests, how="left", left_index=True, right_on="vcs_commit_sha")
        return self._convert_to_sparse(df_merged)

    def _apply_graph_based_clustering(self, df_commits: pd.DataFrame) -> pd.DataFrame:
        """
        Applies graph-based clustering to the commit DataFrame during training.
        """
        df_modified, file_to_cluster = graph_clustering.perform_graph_clustering(df_commits)
        self.train_files = file_to_cluster
        with open(os.path.join(self._path_inference, "train_files.pkl"), "wb") as f:
            pickle.dump(file_to_cluster, f)
        expected_cols = df_modified.columns.tolist()
        with open(os.path.join(self._path_inference, "expected_columns.pkl"), "wb") as f:
            pickle.dump(expected_cols, f)
        return df_modified

    def _apply_graph_based_clustering_inference(self, df_commits: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the saved cluster mapping to new data in inference mode.
        Aggregates features based on clusters and reindexes the DataFrame to match training features.
        """
        with open(os.path.join(self._path_inference, "train_files.pkl"), "rb") as f:
            file_to_cluster = pickle.load(f)
        df_modified = graph_clustering.transform_df_by_cluster(
            df_commits, file_to_cluster, drop_original_file_cols=True, aggfunc="sum"
        )
        df_modified = graph_clustering.ensure_default_cluster_features(df_modified)
        expected_columns_file = os.path.join(self._path_inference, "expected_columns.pkl")
        with open(expected_columns_file, "rb") as f:
            expected_cols = pickle.load(f)
        df_modified = df_modified.reindex(columns=expected_cols, fill_value=0)
        return df_modified

    def _apply_modular_based_clustering(self, df_commits: pd.DataFrame) -> pd.DataFrame:
        """
        Applies modular-based clustering to the commit DataFrame during training.
        It uses a JSON file to map files to modules and aggregates features accordingly.
        """
        json_path = Path(self._path_data) / "modular_structure" / "project_structure.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Modular clustering JSON file not found at {json_path}")
        df_modified, file_to_module = modular_clustering.perform_module_clustering(df_commits, str(json_path))
        self.train_files = file_to_module
        with open(os.path.join(self._path_inference, "train_files.pkl"), "wb") as f:
            pickle.dump(file_to_module, f)
        expected_cols = df_modified.columns.tolist()
        with open(os.path.join(self._path_inference, "expected_columns.pkl"), "wb") as f:
            pickle.dump(expected_cols, f)
        return df_modified

    def _apply_modular_based_clustering_inference(self, df_commits: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the saved module mapping to new data in inference mode.
        Aggregates features based on modules and reindexes the DataFrame to match training features.
        """
        with open(os.path.join(self._path_inference, "train_files.pkl"), "rb") as f:
            file_to_module = pickle.load(f)
        df_modified = modular_clustering.transform_df_by_module(
            df_commits, file_to_module, drop_original_file_cols=True, aggfunc="sum"
        )
        df_modified = modular_clustering.ensure_default_module_features(df_modified)
        expected_columns_file = os.path.join(self._path_inference, "expected_columns.pkl")
        with open(expected_columns_file, "rb") as f:
            expected_cols = pickle.load(f)
        df_modified = df_modified.reindex(columns=expected_cols, fill_value=0)
        return df_modified