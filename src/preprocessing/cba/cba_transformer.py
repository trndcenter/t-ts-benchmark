import pandas as pd

import nip

@nip.nip
class CBATransformer:
    def __init__(self, preprocessing_files):
        self.preprocessed_files = False
        self.preprocessing_files = preprocessing_files

    def fit_transform(self, X: pd.DataFrame, y):
        return X.drop(columns=["test_file_path", "test_method"]).set_index(["vcs_commit_sha", "allure_id"]), y
    
    def transform(self, X: pd.DataFrame):
        return X.drop(columns=["test_file_path", "test_method"]).set_index(["vcs_commit_sha", "allure_id"])