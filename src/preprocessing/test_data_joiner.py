import os
import pandas as pd


class TestDataJoiner:
    """Create dataset for inference from data in :path_data"""

    def __init__(self, path_data="data_inference", inference: bool = False):
        self.path_diff = os.path.join(path_data, "diff.json")
        self.path_dir_test_history = os.path.join(path_data, "test_history")
        self.path_dir_project_data = os.path.join(path_data, "project_data")
        self.is_inference = inference

        if not self.is_inference:
            self.path_sessions = os.path.join(path_data, "sessions.csv")
            self.path_dir_test_results = os.path.join(path_data, "test_results")

    def create_dataset(self, commit: str = None) -> pd.DataFrame:
        """Create datase with test data"""
        df_history = self._get_tests_history(commit)
        df_tranformed = self._transform_counts(df_history)
        df_with_failures = self._find_failure_rates(df_tranformed)

        df_project_data = self._get_project_data(commit)
        
        if not self.is_inference:
            df_tests_results = self._get_tests_results(commit)
            df_merged = self._merge_data(df_with_failures, df_project_data, df_tests_results)
        else:
            df_merged = self._merge_data(df_with_failures, df_project_data)

        df_merged['vcs_commit_sha'] = commit

        return df_merged

    def _get_tests_history(self, commit: str) -> pd.DataFrame:
        """Get test history"""
        df_history = pd.read_csv(
            os.path.join(self.path_dir_test_history, commit + ".csv"), delimiter=";"
        )

        df_history["vcs_commit_sha"] = commit

        return df_history

    def _transform_counts(self, df_history: pd.DataFrame) -> pd.DataFrame:
        """Transform rows with failures to columns with failures"""
        df_transformed = df_history[["allure_id"]].copy()
        period_cols = df_history.columns[df_history.columns.str.startswith("count")]

        for period in period_cols:
            for status in [0, 2]:
                df_right = df_history[df_history["status"] == status][
                    ["allure_id", period]
                ]
                df_transformed = df_transformed.merge(
                    df_right, how="left", left_on="allure_id", right_on="allure_id"
                )
                df_transformed = df_transformed.rename(
                    columns={period: f"{period}_{ status }"}
                )
        df_transformed = df_transformed.drop_duplicates(['allure_id'])

        counts_col = df_transformed.columns[
            df_transformed.columns.str.startswith("count")
        ]
        df_transformed[counts_col] = df_transformed[counts_col].fillna(0).astype(int)

        return df_transformed

    def _find_failure_rates(self, df_tests: pd.DataFrame) -> pd.DataFrame:
        """Find failure rates for tests"""
        df_with_failures = pd.DataFrame(index=df_tests.index)
        periods = ["7d", "14d", "28d", "56d"]
        for period in periods:
            df_with_failures[f"failure_rate_{period}"] = df_tests[
                f"count_{period}_0"
            ] / (df_tests[f"count_{period}_0"] + df_tests[f"count_{period}_2"])

            df_with_failures[f"failure_rate_{period}"] = df_with_failures[
                f"failure_rate_{period}"
            ].fillna(0)
        df_tests_failures = pd.concat([df_tests, df_with_failures], axis=1)

        drop_columns = []
        for period in periods:
            drop_columns.extend([f"count_{ period }_0", f"count_{ period }_2"])
        df_tests_failures = df_tests_failures.drop(drop_columns, axis=1)

        return df_tests_failures

    def _get_tests_results(self, commit: str) -> pd.DataFrame:
        """Find test results in sessions with :commit and delete some rows"""
        unique_sessions = self._get_commit_sessions(self.path_sessions, commit)
        df_tests_results = pd.read_csv(
            os.path.join(self.path_dir_test_results, commit + ".csv"), delimiter=";"
        )
        df_tests_results = df_tests_results[
            df_tests_results["session_id"].isin(unique_sessions)
        ]

        df_tests_results = self._remove_redundant_statuses(df_tests_results)
        df_tests_results = self._drop_duplicated_tests(
            df_tests_results, columns=["allure_id", "status"]
        )

        return df_tests_results
    
    def _remove_redundant_statuses(self, df_tests: pd.DataFrame) -> pd.DataFrame:
        """Keep only failed (0) and passed (2) tests. Also replace 0 and 2 with 1 and 0"""
        df_cleaned = df_tests[
            (df_tests["status"] == 0) | (df_tests["status"] == 2)
        ].copy()
        df_cleaned.loc[df_cleaned["status"] == 0, "status"] = 1
        df_cleaned.loc[df_cleaned["status"] == 2, "status"] = 0

        return df_cleaned
    
    def _drop_duplicated_tests(
        self, df_tests: pd.DataFrame, columns=list[str]
    ) -> pd.DataFrame:
        """Drop duplicated tests in :columns"""
        original_order = df_tests.index
        df_tests_ = df_tests.copy()
        df_tests_["duplicated"] = df_tests_[columns].duplicated(keep="last")
        df_tests_ = df_tests_.loc[original_order]
        df_tests_ = df_tests_[~df_tests_["duplicated"].to_numpy()]
        df_tests_dropped = df_tests_.drop(["duplicated"], axis=1)

        return df_tests_dropped

    def _get_commit_sessions(self, path_sessions, vcs_commit):
        """Get sessions with :vcs_commit"""
        df_sessions = pd.read_csv(path_sessions, delimiter=";")
        unique_sessions = set(
            df_sessions[df_sessions["vcs_commit_sha"] == vcs_commit]["session_id"]
        )
        return unique_sessions

    def _get_project_data(self, commit: str) -> pd.DataFrame:
        """Get paths and methods of the tests"""
        df_project_data = pd.read_csv(
            os.path.join(self.path_dir_project_data, commit + ".csv"), delimiter=";"
        )

        return df_project_data

    def _merge_data(
        self,
        df_with_failures: pd.DataFrame,
        df_project_data: pd.DataFrame,
        df_results: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Merge dataframes with file, test and cross features"""
        df_merged = df_project_data.merge(
            df_with_failures, how="left", on='allure_id'
        )
        failure_rate_cols = df_merged.columns[
            df_merged.columns.str.startswith("failure_rate")
        ]
        df_merged[failure_rate_cols] = df_merged[failure_rate_cols].fillna(0)

        if df_results is not None:
            df_merged = df_merged.merge(
                df_results, how="inner", on='allure_id'
            )

        return df_merged
