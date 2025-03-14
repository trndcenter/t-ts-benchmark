import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from .preprocess_dataframes import prepare_tokens_lengths


class FileTestDataset(Dataset):
    def __init__(
        self,
        df_files: pd.DataFrame,
        df_tests: pd.DataFrame,
        commits: list[str],
        word_to_index: dict,
        max_len: int,
        shuffle=True,
    ):
        self.files = df_files
        self.tests = df_tests
        self.commits = commits
        self.word_to_index = word_to_index
        self.shuffle = shuffle
        self.max_len = max_len

    def __len__(self):
        return len(self.commits)

    def __getitem__(self, idx: int) -> dict:
        """Return a dict of the structure:

            - file_tokens_padding
            - file_lengths
            - pos_test_tokens_padding
            - pos_test_lengths
            - neg_test_tokens_padding
            - neg_test_lengths

            pos_test - failed tests, neg_test - passed tests
            """
        commit = self.commits[idx]
        files = self.files[self.files.commit == commit]
        tests = self.tests[self.tests.commit == commit]

        file_names = files["name"].to_numpy()
        pos_test_names = tests[tests.status == 1]["name"].to_numpy()
        neg_test_names = tests[tests.status == 0]["name"].to_numpy()

        if self.shuffle:
            np.random.shuffle(neg_test_names)

        item = dict()

        # max_len is needed to fix a length of padding
        item["file_tokens_padded"], item["file_lengths"] = prepare_tokens_lengths(
            file_names, self.word_to_index, max_len=self.max_len
        )
        item["pos_test_tokens_padded"], item["pos_test_lengths"] = (
            prepare_tokens_lengths(
                pos_test_names, self.word_to_index, max_len=self.max_len
            )
        )
        item["neg_test_tokens_padded"], item["neg_test_lengths"] = (
            prepare_tokens_lengths(
                neg_test_names, self.word_to_index, max_len=self.max_len
            )
        )

        return item
