import os
import nip

from typing import Union
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import Counter
from .test_data_joiner import TestDataJoiner

@nip.nip
class CommitDatasetCreator:
    """Create a dataset from the data in :path_data"""

    def __init__(
        self,
        path_data: Union[str, Path] = Path('data'),
        inference: bool = False,
        without_unstable: bool = False,
        path_inference: Union[str, Path] = os.path.join("inference", "xgb"),
    ):
        self.is_inference = inference


        if not self.is_inference:
            self.train_commits = None
            self.train_dataset = None
            self.train_dataframes = None
            self.train_files = None

        self.test_commits = None
        self.test_dataset = None
        self.test_dataframes = None

        self._path_data = path_data
        self._path_sessions = os.path.join(path_data, "sessions.csv")
        self._path_file_stats = os.path.join(path_data, "file_stats")
        self._path_test_results = os.path.join(path_data, "test_results")

        self.commits = self._read_commits(self._path_sessions)
        self._file_features = self._get_file_features(self.commits)

        self.without_unstable = without_unstable
        self._path_inference = path_inference

    def create_dataset(
        self,
        start_commit: int = None,
        last_commit: int = None,
        num_cross_files: int = 3,
        train_size: float = 1.0,
        min_commit_freq: int = 0,
        max_commit_freq: float = 1.0
    ) -> pd.DataFrame:
        """Create a dataset with file, test and cross features \n
        
        :param start_commit: index of first commit\n
        :param last_commit: index of last commit\n
        :param num_cross_files: number of files to join with test\n
        :param train_size: size of train dataset\n"""
        dataframes = self.create_dataframes(
            start_commit=start_commit,
            last_commit=last_commit,
            num_cross_files=num_cross_files,
            train_size=train_size,
            min_commit_freq=min_commit_freq,
            max_commit_freq=max_commit_freq
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
        num_cross_files: int = 3,
        train_size: float = 1.0,
        min_commit_freq: int = 0,
        max_commit_freq: float = 1.0
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create dataframes with file, test and cross features \n
        
        :param start_commit: index of first commit\n
        :param last_commit: index of last commit\n
        :param num_cross_files: number of files to join with test\n
        :param train_size: size of train dataset\n
        
        :return: (commits, tests, crossfiles)"""
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

    def create_commits_tests_crossfiles(
        self, commits: list[str], files: set[str], num_cross_files: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create dataframes with file, test and cross features using ::commits"""
        df_commits = self.create_df_commit(commits, files=files)
        df_tests = self.create_df_tests(commits)
        df_cross_files = self.create_cross_files(
            df_tests, commits, num_cross_files=num_cross_files
        )

        if not self.is_inference:
            df_tests = df_tests.drop(columns=["test_file_path"])

        return df_commits, df_tests, df_cross_files

    def create_df_commit(self, commits: list[str], files: set[str]) -> pd.DataFrame:
        """Create a dataframe with file features"""
        df_commits = None
        files_features = ["unknown_files_count"]
        for file in files:
            for feature in self._file_features:
                files_features.append(f"{file}_{feature}")

        for commit in tqdm(commits, desc="Creating commits dataset"):
            df_file_stats = pd.read_csv(
                os.path.join(self._path_file_stats, commit + ".csv"), delimiter=";"
            )
            df_commit = pd.DataFrame(0, index=[commit], columns=files_features)

            for _, row in df_file_stats.iterrows():
                file_name = row["file_name"]
                if file_name in files:
                    df_commit.at[commit, f"{file_name}_changed"] = 1
                    for feature in self._file_features:
                        if feature != "changed":
                            if feature == "change_type":
                                df_commit.at[commit, f"{file_name}_{feature}"] = (
                                    self._get_change_type_code(row[feature])
                                )
                            else:
                                df_commit.at[commit, f"{file_name}_{feature}"] = row[
                                    feature
                                ]
                else:
                    df_commit.at[commit, "unknown_files_count"] += 1
            
            if len(df_file_stats) != 0:
                df_commit["unknown_files_count"] = df_commit["unknown_files_count"].astype('float')
                df_commit.at[commit, "unknown_files_count"] /= len(df_file_stats)

            df_commits = (
                pd.concat([df_commits, df_commit])
                if df_commits is not None
                else df_commit.copy()
            )

        return self._convert_to_sparse(df_commits)

    def create_df_tests(self, commits: list[str]) -> pd.DataFrame:
        """Create a dataframe with test features"""
        df_tests = self._join_test_data(commits)
        if not self.is_inference:
            df_tests = self._drop_flaky_tests(
                df_tests, columns=["vcs_commit_sha", "test_case_id", "allure_id"]
            )
            df_tests = self._drop_tests_columns(df_tests)
        if self.without_unstable:
            df_tests = df_tests[df_tests['flaky'] == 0]

        return self._convert_to_sparse(df_tests)

    def create_cross_files(
        self, df_tests: pd.DataFrame, commits: list[str], num_cross_files: int = 3
    ) -> pd.DataFrame:
        """Create a dataframe with cross features"""
        cols = ["vcs_commit_sha", "allure_id"]
        cross_file_features = self._file_features + ["file_extension"]
        cols.extend(
            [
                f"cross_file_{i}_{feat}"
                for i in range(num_cross_files)
                for feat in cross_file_features
            ]
        )
        df_cross_files = pd.DataFrame(columns=cols)

        if num_cross_files == 0:
            return df_cross_files

        for commit in tqdm(commits, desc="Creating cross-files dataset"):
            df_tests_paths = df_tests[df_tests["vcs_commit_sha"] == commit][
                ["allure_id", "test_file_path"]
            ].drop_duplicates()
            df_files = pd.read_csv(
                os.path.join(self._path_file_stats, commit + ".csv"),
                index_col=["file_name"],
                delimiter=";",
            )
            df_tests_closest_files = self._find_all_closest_files(
                df_tests_paths,
                df_files.index.to_series(),
                num_cross_files=num_cross_files,
            )
            for _, row in df_tests_closest_files.iterrows():
                dict_one_test = {
                    "vcs_commit_sha": commit,
                    "allure_id": row["allure_id"],
                }
                for i_cross_file in range(len(row) - 1):
                    if row.at[i_cross_file] is not np.nan:
                        file_name = row.at[i_cross_file]
                        file_features = df_files.loc[file_name]
                        dict_one_test[f"cross_file_{i_cross_file}_changed"] = 1
                        for feature in cross_file_features:
                            if feature != "changed":
                                if feature == "change_type":
                                    dict_one_test[
                                        f"cross_file_{i_cross_file}_{feature}"
                                    ] = self._get_change_type_code(
                                        file_features[feature]
                                    )
                                else:
                                    dict_one_test[
                                        f"cross_file_{i_cross_file}_{feature}"
                                    ] = file_features[feature]
                df_cross_files.loc[df_cross_files.shape[0]] = dict_one_test

        cross_file_columns = [
            f"cross_file_{i_cross_file}_{feature}"
            for i_cross_file in range(num_cross_files)
            for feature in cross_file_features
        ]
        df_cross_files[cross_file_columns] = df_cross_files[cross_file_columns].fillna(
            0
        )

        return df_cross_files

    def merge_dataframes(
        self,
        df_commits: pd.DataFrame,
        df_tests: pd.DataFrame,
        df_cross_files: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge dataframew with file, test, and cross features"""
        print("Merging datasets...")

        df_tests["allure_id"] = df_tests["allure_id"].astype("int64")

        df_tests_cross_files = df_tests.merge(
            df_cross_files, how="left", on=["vcs_commit_sha", "allure_id"]
        )
        df_merged = df_commits.merge(
            df_tests_cross_files, how="left", left_index=True, right_on="vcs_commit_sha"
        )

        return self._convert_to_sparse(df_merged)

    def set_path_data(self, path_data: Union[str, Path]):
        self._path_data = path_data
        self._path_sessions = os.path.join(path_data, "sessions.csv")
        self._path_file_stats = os.path.join(path_data, "file_stats")
        self._path_test_results = os.path.join(path_data, "test_results")

    # ------------------------------------------------
    # Private methods to work with commits
    def _read_commits(self, path_sessions: str):
        """Read and filter commints in sessions"""
        df_sessions = pd.read_csv(path_sessions, delimiter=";")

        if "created_at" in df_sessions.columns:
            df_sessions = df_sessions.sort_values("created_at")
        unique_commits = self._filter_commits(df_sessions["vcs_commit_sha"].unique())

        return unique_commits

    def _get_commits(self, start_commit: int, last_commit: int) -> list[str]:
        return self.commits[start_commit:last_commit]

    def _filter_commits(
        self,
        unique_commits: list[str],
    ) -> list[str]:
        """Delete all commits without test results or file stats"""
        bad_commits = self._get_file_stats_bad_commits(unique_commits)
        if not self.is_inference:
            bad_commits |= self._get_test_results_bad_commits()

        idx_remove = []
        for i in range(len(unique_commits)):
            if unique_commits[i] in bad_commits:
                idx_remove.append(i)

        if idx_remove:
            print(
                "Some commits were removed because of lack of test results or file stats"
            )
            print(
                f"Number of removed commits: {len(idx_remove)} out of {len(unique_commits)}"
            )

        unique_commits = np.delete(unique_commits, idx_remove)

        assert (
            len(unique_commits) >= 1
        ), "No commits left after filtering"

        return unique_commits

    def _get_file_stats_bad_commits(self, unique_commits: list[str]) -> list[str]:
        """Find commits without file stats"""
        file_stats_commits = set()
        for file in os.listdir(self._path_file_stats):
            commit = file[: file.find(".csv")]
            file_stats_commits.add(commit)

        bad_commits = set()
        for commit in unique_commits:
            if commit not in file_stats_commits:
                bad_commits.add(commit)

        return bad_commits

    def _get_test_results_bad_commits(self) -> list[str]:
        """Find commits without test results"""
        bad_commits = set()
        for file in os.listdir(self._path_test_results):
            commit = file[: file.find(".csv")]
            df = pd.read_csv(os.path.join(self._path_test_results, file), delimiter=";")
            if df.shape[0] == 0 or np.all(
                df["status"].apply(lambda x: x not in {0, 2})
            ):
                bad_commits.add(commit)

        return bad_commits

    # -------------------------------------------
    # Private methods to work with tests

    def _join_test_data(self, commits: list[str]) -> pd.DataFrame:
        """Create a dataframe with test data"""
        test_data_joiner = TestDataJoiner(self._path_data, inference=self.is_inference)
        df_tests_joined = pd.concat(
            [
                test_data_joiner.create_dataset(commit)
                for commit in tqdm(commits, desc="Creating test dataset")
            ],
            ignore_index=True,
        )

        return df_tests_joined

    def _drop_tests_columns(
        self, df_tests: pd.DataFrame, columns: list[str] = None
    ) -> pd.DataFrame:
        if columns is None:
            columns = ["launch_id", "session_id", "duration", "test_method"]
        df_dropped = df_tests.drop(columns, axis=1)

        return df_dropped

    def _drop_flaky_tests(
        self, df_tests: pd.DataFrame, columns=list[str]
    ) -> pd.DataFrame:
        """Mark flaky tests. The test is marked as flaky if it has more than 1
        status during one session"""
        df_tests_ = df_tests.copy()
        df_tests_["count"] = df_tests_.groupby(columns)["status"].transform("count")
        df_tests_["flaky"] = (df_tests_["count"] != 1).astype("int")

        df_tests_dropped = df_tests_.drop(
            df_tests_[(df_tests_["flaky"] == 1) & (df_tests_["status"] == 0)].index
        )
        df_tests_dropped = df_tests_dropped.drop(["count"], axis=1)

        return df_tests_dropped
    
    # ------------------------------------------------
    # Private methods to work with files

    def _get_files(self, commits: list[str], min_cf: int = 1, max_cf: float = 1.0) -> list[str]:
        """Go over the files in :commits and save the unique files"""
        files = Counter()
        for commit in commits:
            df_file_stats = pd.read_csv(
                os.path.join(self._path_file_stats, commit + ".csv"), delimiter=";"
            )
            files.update(set(df_file_stats["file_name"]))
        
        unique_files = set()
        for file in files:
            if files[file] >= min_cf and (files[file] / len(commits)) <= max_cf:
                unique_files.add(file)
        
        return unique_files

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
            ]
        ).to_list()

        file_features.append("changed")

        return file_features

    def _get_change_type_code(self, change_type: str) -> int:
        """Convert a letter of change type to a code
        Possible values:
            A - 1: addition of a file
            C - 2: copy of a file into a new one
            D - 3: deletion of a file
            M - 4: modification of the contents or mode of a file
            R - 5: renaming of a file
            T - 6: change in the type of the file (regular file, symbolic link or submodule)
            U - 7: file is unmerged (you must complete the merge before it can be committed)
            X - 8: "unknown" change type (most probably a bug, please report it)
        """
        dict_change_type = {
            "A": 1,
            "C": 2,
            "D": 3,
            "M": 4,
            "R": 5,
            "T": 6,
            "U": 7,
            "X": 8,
        }
        label_change_type = (
            dict_change_type[change_type] if change_type in dict_change_type else 9
        )
        return label_change_type

    def _get_train_files(self, path: Union[str, Path]):
        """Get a list of files which were in the training set"""
        if self.is_inference:
            with open(os.path.join(path, "train_files.pkl"), "rb") as f:
                files = pickle.load(f)
        else:
            assert (
                self.train_files is not None
            ), "Training set must be created before inference"
            files = self.train_files

        return files

    # ------------------------------------------------
    # Private methods to work with cross files

    def _find_all_closest_files(
        self,
        tests_paths: pd.DataFrame,
        files_paths: pd.DataFrame,
        num_cross_files: int = 3,
    ) -> pd.DataFrame:
        """Find :num_cross_file closest files to the tests"""
        files_tree = self._build_tree(files_paths)

        prepared_files = []
        for _, row in tests_paths.iterrows():
            allure_id, test_path = row["allure_id"], row["test_file_path"]
            closest_files = self._find_closest_files(
                test_path, files_tree, num_closest_files=num_cross_files
            )
            for file in closest_files:
                prepared_files.append([allure_id, test_path, file[1]])

        tests_closest_files = pd.DataFrame(
            prepared_files, columns=["allure_id", "test_file_path", "file_name"]
        )

        tests_closest_files = (
            tests_closest_files.groupby("allure_id")["file_name"]
            .apply(lambda df: df.reset_index(drop=True))
            .unstack()
            .reset_index()
        )
        return tests_closest_files

    def _build_tree(
        self, pathes: list[str], tree: dict = None, num_closest_files: int = 3
    ) -> dict:
        """Build a tree of pathes, keeping only :num_closest_files of the closest files in 
        each node"""
        if tree is None:
            tree = {}
        start_value = [[1000, np.nan] for _ in range(num_closest_files)]
        for path in pathes:
            parts = path.split("/")
            node = tree
            for i in range(len(parts)):
                part = parts[i]
                if part not in node:
                    node[part] = {}

                previous_files = node.get("closest_files", start_value.copy())
                new_file = [len(parts) - i - 1, path]
                node["closest_files"] = self._add_new_min(previous_files, new_file)

                node = node[part]
        return tree

    def _find_closest_files(
        self, path: str, tree: dict, num_closest_files: int = 3
    ) -> list:
        """Find :num_closest_files closest files to :path"""
        parts = path.split("/")
        node = tree
        closest_files = [[1000, np.nan] for _ in range(num_closest_files)]
        for i in range(len(parts)):
            rest = len(parts) - i + 1
            for file in node["closest_files"]:
                file_with_rest = [file[0] + rest, file[1]]
                closest_files = self._add_new_min(closest_files, file_with_rest)

            part = parts[i]
            if part not in node:
                break
            node = node[part]

        return closest_files

    def _add_new_min(self, current_files: list, new_file: list[int, str]) -> list:
        """Add :new_file to :current_files, if the path to it is less"""
        for i in range(len(current_files)):
            cur_dist, cur_file = current_files[i]
            new_dist, new_f = new_file
            if (
                cur_file is not np.nan
                and new_f is not np.nan
                and len(cur_file) == len(new_f)
                and cur_file == new_f
                and cur_dist > new_dist
            ):
                current_files[i][0] = new_dist
                return current_files

        old_min = current_files[-1][0]
        if old_min > new_file[0]:
            current_files[-1] = new_file
            return sorted(current_files, key=lambda x: x[0])

        return current_files

    # --------------------------------------
    # Private methods to convert to a sparse matrix

    def _convert_to_sparse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert int and float data to SparseDtype"""
        if self.is_inference:
            return df

        df_sparse = df.copy()

        unit16_columns, int64_columns = self._find_int_columns(df)

        df_sparse[unit16_columns] = df[unit16_columns].astype(
            pd.SparseDtype("int16", 0)
        )

        df_sparse[int64_columns] = df[int64_columns].astype(pd.SparseDtype("int64", 0))

        float_columns = df.select_dtypes("float").columns
        df_sparse[float_columns] = df[float_columns].astype(
            pd.SparseDtype("float32", 0)
        )

        return df_sparse

    def _find_int_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Find columns with int64 values, which can be converted to uint16"""
        int_columns = df.select_dtypes(include=["int64"]).columns.to_numpy()

        max_uint16 = 2**16 - 1

        def check_function(x):
            return (x > max_uint16) | (x < 0)

        idx_int64 = []
        for i in range(len(int_columns)):
            if np.any(check_function(df[int_columns[i]])):
                idx_int64.append(i)
        uint16_columns, int64_columns = np.delete(int_columns, idx_int64), np.take(
            int_columns, idx_int64
        )
        return uint16_columns, int64_columns
