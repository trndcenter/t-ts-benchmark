import os
import pandas as pd
import numpy as np
import pickle
import nip

import torch
import torch.nn as nn
from tqdm import tqdm

from ..commit_dataset_creator import CommitDatasetCreator
from .file_test_dataset import FileTestDataset
from .train_function import train
from ...models.code_test_relation_model import CodeTestRelationModel
from . import preprocess_dataframes as pred_data


@nip.nip
class AttentionCreator(CommitDatasetCreator):
    """Create a dataset with attention scores for each test"""

    def __init__(
        self,
        *args,
        embedding_dim: int = 32,
        hidden_dim: int = 8,
        output_dim: int = 1,
        heads: int = 8,
        epochs: int = 1,
        batch_size: int = 16,
        lr: float = 1e-4,
        margin: float = 0.5,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.margin = margin
        self.device = device

    def create_dataframes(self, train_size=1.0, **kwargs):
        super().create_dataframes(train_size=train_size, **kwargs)
        df_files, df_tests = self._split_dataframes(train_size=train_size)

        max_len = self._find_max_len(df_tests, df_files)
        model, word_to_index = self._prepare_model(
            df_files, df_tests, max_len, train_size
        )
        if self.is_inference:
            df_files, df_tests = self._filter_names(df_files, df_tests, word_to_index)

        tests_with_attention = self.create_tests_with_attention(
            df_files,
            df_tests,
            model,
            word_to_index,
            max_len,
            device=self.device,
        )
        if train_size == 0:
            test_tests = self._merge_tests(
                tests_with_attention, self.test_dataframes[1]
            )
            self.test_dataframes = (
                self.test_dataframes[0],
                test_tests,
                self.test_dataframes[2],
            )

            return self.test_dataframes
        elif train_size == 1:
            train_tests = self._merge_tests(
                tests_with_attention, self.train_dataframes[1]
            )
            self.train_dataframes = (
                self.train_dataframes[0],
                train_tests,
                self.train_dataframes[2],
            )

            return self.train_dataframes
        else:
            train_tests = self._merge_tests(
                tests_with_attention, self.train_dataframes[1]
            )
            test_tests = self._merge_tests(
                tests_with_attention, self.test_dataframes[1]
            )
            self.train_dataframes = (
                self.train_dataframes[0],
                train_tests,
                self.train_dataframes[2],
            )
            self.test_dataframes = (
                self.test_dataframes[0],
                test_tests,
                self.test_dataframes[2],
            )

            return self.train_dataframes, self.test_dataframes

    def create_commits_tests_crossfiles(
        self, commits: list[str], files: set[str], num_cross_files: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create dataframes with file, test and cross features using ::commits"""
        df_commits = self.create_df_commit(commits, files=files)
        df_tests = self.create_df_tests(commits)
        df_cross_files = self.create_cross_files(
            df_tests, commits, num_cross_files=num_cross_files
        )

        return df_commits, df_tests, df_cross_files

    def _split_dataframes(
        self,
        train_size: float = 1.0,
    ):
        """Split dataframes with names according to :train size"""
        # <type>_dataframes == (files, tests, crossfiles)
        if train_size == 1.0:
            df_files = pred_data.preprocess_df_files(self.train_dataframes[0])
            df_tests = pred_data.preprocess_df_tests(self.train_dataframes[1])
        elif train_size == 0.0:
            df_files = pred_data.preprocess_df_files(self.test_dataframes[0])
            df_tests = pred_data.preprocess_df_tests(self.test_dataframes[1])
        else:
            df_files = pd.concat(
                (
                    pred_data.preprocess_df_files(self.train_dataframes[0]),
                    pred_data.preprocess_df_files(self.test_dataframes[0]),
                ),
                ignore_index=True,
            )
            df_tests = pd.concat(
                (
                    pred_data.preprocess_df_tests(self.train_dataframes[1]),
                    pred_data.preprocess_df_tests(self.test_dataframes[1]),
                ),
                ignore_index=True,
            )
        return df_files, df_tests

    def _find_max_len(self, df_tests: pd.DataFrame, df_files: pd.DataFrame):
        """Find max length of test and file names"""
        max_len_tests = df_tests["name"].apply(lambda x: len(x.split(" "))).max()
        max_len_files = df_files["name"].apply(lambda x: len(x.split(" "))).max()
        return max(max_len_files, max_len_tests)

    def _prepare_model(
        self, df_files: pd.DataFrame, df_tests: pd.DataFrame, max_len: int, train_size: float
    ):
        """Train or read the attention model"""
        if not self.is_inference:
            word_to_index = self._create_word_to_index(df_tests, df_files)
            vocab_size = len(word_to_index)
            model = self._train_attention_model(
                df_tests, df_files, word_to_index, max_len, vocab_size, train_size
            )
            # Saving model
            with open(
                os.path.join(self._path_inference, "word_to_index.pkl"), "wb"
            ) as f:
                pickle.dump(word_to_index, f)
            torch.save(
                model.state_dict(),
                os.path.join(self._path_inference, "attention_model.pth"),
            )
        else:
            word_to_index = pickle.load(
                open(os.path.join(self._path_inference, "word_to_index.pkl"), "rb")
            )
            vocab_size = len(word_to_index)
            model = CodeTestRelationModel(
                vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.output_dim,
                self.heads,
                device=self.device,
            )
            model.load_state_dict(
                torch.load(
                    os.path.join(self._path_inference, "attention_model.pth"),
                    weights_only=True,
                )
            )

        return model, word_to_index

    def _create_word_to_index(self, df_tests, df_files):
        """Create a dictionary as <word>: <index>"""
        unique_tokens = ["<UKN>"] + list(
            set(" ".join(df_tests["name"]).split())
            | set(" ".join(df_files["name"]).split())
        )
        word_to_index = {word: i for i, word in enumerate(unique_tokens)}

        return word_to_index

    def _train_attention_model(
        self, df_tests, df_files, word_to_index, max_len, vocab_size, train_size
    ):
        dataset = FileTestDataset(
            df_files, df_tests, df_files.commit.unique(), word_to_index, max_len
        )

        model = CodeTestRelationModel(
            vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.heads,
            device=self.device,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        num_train_commits = int(len(dataset) * train_size)
        model = train(
            self.epochs,
            model,
            optimizer,
            num_train_commits,
            self.batch_size,
            dataset,
            self.margin,
            self.device,
        )

        return model

    def _filter_names(
        self, df_files: pd.DataFrame, df_tests: pd.DataFrame, word_to_index: dict
    ):
        """Filter unseen names on the inference stage"""
        
        def create_new_name(old_name):
            return " ".join(
                [
                    (token if token in word_to_index else "<UKN>")
                    for token in old_name.split()
                ]
            )

        df_files["name"] = df_files["name"].apply(create_new_name)
        df_tests["name"] = df_tests["name"].apply(create_new_name)

        return df_files, df_tests

    def _merge_tests(self, df_with_attention, df_tests):
        """Merge a general test dataframe with attention scores"""
        if not self.is_inference:
            df_tests["allure_id"] = df_tests["allure_id"].sparse.to_dense()
        df_merged = df_tests.merge(
            df_with_attention.drop_duplicates(subset=["commit", "allure_id"]),
            left_on=["vcs_commit_sha", "allure_id"],
            right_on=["commit", "allure_id"],
        ).drop(columns=["commit"])

        return df_merged

    def create_tests_with_attention(
        self,
        df_files: pd.DataFrame,
        df_tests: pd.DataFrame,
        model,
        word_to_index,
        max_len,
        device: str = "cpu",
    ):
        """Create a dataframe with attention scores based on the predictions"""
        df_with_attention = self._predict_attention(
            df_files, df_tests, model, word_to_index, max_len, device=device
        )
        test_columns = ["allure_id"]
        if not self.is_inference:
            test_columns.append("status")

        if not self.is_inference:
            df_with_attention[test_columns] = df_with_attention[
                test_columns
            ].sparse.to_dense()
        df_with_attention = df_with_attention[["commit", "allure_id", "attention"]]
        df_with_attention["attention"] = self._sigmoid(df_with_attention["attention"])

        return df_with_attention

    def _predict_attention(
        self,
        df_files: pd.DataFrame,
        df_tests: pd.DataFrame,
        model,
        word_to_index,
        max_len,
        device: str = "cpu",
    ):
        """Predict attention based on the tokens of the file and test names"""
        df_res = pd.DataFrame()
        for commit in tqdm(df_files["commit"].unique()):
            file_commit = df_files[df_files["commit"] == commit]
            test_commit = df_tests[df_tests["commit"] == commit]
            file_names = file_commit["name"].values
            test_names = test_commit["name"].values

            file_tokens_padded, file_lengths = self._prepare_tokens_lengths(
                file_names, word_to_index, max_len=max_len
            )
            test_tokens_padded, test_lengths = self._prepare_tokens_lengths(
                test_names, word_to_index, max_len=max_len
            )
            batch_size = file_tokens_padded.size()[0]

            for i_test in range(0, test_tokens_padded.size()[0], batch_size):
                test_tokens_padded_batch, test_lengths_batch = (
                    test_tokens_padded[i_test : i_test + batch_size],
                    test_lengths[i_test : i_test + batch_size],
                )
                if i_test + batch_size > test_tokens_padded.size()[0]:
                    while test_tokens_padded_batch.size()[0] < batch_size:
                        test_tokens_padded_batch = torch.concat(
                            (test_tokens_padded_batch, test_tokens_padded_batch)
                        )
                        test_lengths_batch = torch.concat(
                            (test_lengths_batch, test_lengths_batch)
                        )
                    test_tokens_padded_batch = test_tokens_padded_batch[:batch_size]
                    test_lengths_batch = test_lengths_batch[:batch_size]
                model.eval()
                with torch.no_grad():
                    preds = torch.squeeze(
                        model(
                            file_tokens_padded.to(device),
                            file_lengths,
                            test_tokens_padded_batch.to(device),
                            test_lengths_batch,
                        )
                        .cpu()
                        .detach(),
                        dim=1,
                    ).numpy()

                df_commit = pd.DataFrame(
                    {
                        "allure_id": test_commit["allure_id"].iloc[
                            i_test : i_test + batch_size
                        ],
                        "commit": commit,
                        "attention": preds[
                            : min(batch_size, test_tokens_padded.size()[0] - i_test)
                        ],
                    }
                )
                if not self.is_inference:
                    df_commit["status"] = test_commit["status"].iloc[
                        i_test : i_test + batch_size
                    ]
                df_res = pd.concat((df_res, df_commit))
        return df_res

    def _prepare_tokens_lengths(
        self, names: list[str], word_to_index: dict, max_len: int = 13
    ):
        """Padding the file and test names to have the same length"""
        name_tokens = [
            torch.tensor([word_to_index[token] for token in name.split()])
            for name in names
        ]
        name_lengths = torch.tensor([len(tokens) for tokens in name_tokens])

        name_tokens[0] = nn.ConstantPad1d((0, max_len - len(name_tokens[0])), 0)(
            name_tokens[0]
        )
        name_tokens = [tokens for tokens in name_tokens]

        name_tokens_padded = nn.utils.rnn.pad_sequence(name_tokens, batch_first=True)

        return name_tokens_padded, name_lengths

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _drop_tests_columns(
        self, df_tests: pd.DataFrame, columns: list[str] = None
    ) -> pd.DataFrame:
        if columns is None:
            columns = ["launch_id", "session_id", "duration"]
        df_dropped = df_tests.drop(columns, axis=1)

        return df_dropped
