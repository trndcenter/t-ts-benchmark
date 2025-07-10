import pandas as pd

import nip


@nip.nip
class XGBTransformer:
    def __init__(
        self,
        drop_patterns: list[str] = [],
        file_drop_patterns: list[str] = [],
        preprocessing_files: bool = True,
        frequent_files_limit: int = 6000,
        sparse: bool = False,
        file_features: list[str] = None,
    ):
        self.drop_patterns = drop_patterns
        self.file_drop_patterns = file_drop_patterns
        self.preprocessing_files = preprocessing_files
        self.frequent_files_limit = frequent_files_limit
        self.file_features = file_features
        self.sparse = sparse

        self.preprocessed_files = None

    def fit_transform(self, X: pd.DataFrame, y):
        if self.preprocessing_files:
            self.preprocessed_files = self._preprocess_files(X)
            X = self._drop_rare_files(X)

        X = self._drop_columns(X)
        X.drop(columns=['flaky'], inplace=True)
        if self.sparse:
            X = X.sparse.to_coo()

        return X, y

    def transform(self, X: pd.DataFrame):
        X = self._drop_columns(X)
        if self.sparse:
            X = X.sparse.to_coo()

        return X

    def _preprocess_files(self, train_data: pd.DataFrame) -> list[str]:
        """Find files that are in the dataset more than :frequent_files_limit times"""
        frequent_files = set()
        files_columns = [
            col
            for col in train_data.columns
            if col.endswith("changed") and not col.startswith("cross_file")
        ]
        for col in files_columns:
            file = self._get_file_name(col)
            if (
                file not in frequent_files
                and (train_data[col] != 0).sum() > self.frequent_files_limit
            ):
                frequent_files.add(file)

        return frequent_files

    def _drop_rare_files(
        self,
        train_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop file columns that are not in ::frequint_files"""
        columns = train_data.columns
        all_files = {
            self._get_file_name(col)
            for col in columns
            if col.endswith("changed") and not col.startswith("cross_file")
        }
        mask = [True] * len(columns)
        for i in range(len(columns)):
            file = self._get_file_name(columns[i])
            if file not in self.preprocessed_files and file in all_files:
                mask[i] = False

        return train_data[columns[mask]]

    def _get_file_name(self, col_name: str):
        """Get file name from column name"""
        for feature in self.file_features:
            if col_name.endswith(feature):
                return col_name[: col_name.rfind("_" + feature)]
        return col_name

    def _drop_columns(
        self,
        data: pd.DataFrame,
    ):
        """Drop columns"""
        file_cols = self._get_files_columns(data)
        columns_to_drop = []
        for file_col in file_cols:
            if (
                any([file_col.endswith(pattern) for pattern in self.file_drop_patterns])
            ):
                columns_to_drop.append(file_col)
        for pattern in self.drop_patterns:
            columns_to_drop.extend(
                data.columns[data.columns.str.contains(pattern)].to_list()
            )

        return data.drop(columns=columns_to_drop)

    def _get_files_columns(self, df: pd.DataFrame, file_mark="_changed"):
        """Get file columns"""
        columns = df.columns
        cross_files_mask = columns.str.startswith("cross_file_")
        no_cross_columns = columns[~cross_files_mask]
        all_files = (
            no_cross_columns[no_cross_columns.str.endswith(file_mark)]
            .to_series()
            .apply(lambda x: x[: x.rfind(file_mark)])
        )
        all_files_columns = []
        for el in all_files.values:
            for feature in self.file_features:
                if (el + "_" + feature) in columns:
                    all_files_columns.append(el + "_" + feature)
        return all_files_columns
