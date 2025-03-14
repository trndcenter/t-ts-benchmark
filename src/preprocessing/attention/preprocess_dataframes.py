import re

import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm


def preprocess_df_tests(df_tests):
    """Preprocessing tests for the attention model"""
    test_columns = ["allure_id", "test_file_path", "test_method", "vcs_commit_sha"]
    if "status" in df_tests.columns:
        test_columns.append("status")

    df_preprocessed = df_tests[test_columns].rename(
        columns={"vcs_commit_sha": "commit"}
    )
    df_preprocessed["test_file_path"] = df_preprocessed["test_file_path"].apply(
        remove_extension
    )
    df_preprocessed["path"] = (
        df_preprocessed["test_file_path"] + "/" + df_preprocessed["test_method"]
    )
    df_preprocessed["name"] = df_preprocessed["path"].apply(preprocess_path)

    df_preprocessed = df_preprocessed.drop(columns=["test_file_path", "test_method"])

    return df_preprocessed


def preprocess_df_files(df_commits):
    """Preprocessing files for the attention model"""
    df_files = get_file_pathes(df_commits, num_commits=1000)
    df_files = df_files.drop_duplicates()
    df_files = df_files[~df_files["file_path"].str.startswith(".")]
    df_files["file_path"] = df_files["file_path"].apply(remove_extension)
    df_files["name"] = df_files["file_path"].apply(preprocess_path)

    return df_files


def remove_extension(path):
    return "".join([word.capitalize() for word in path.split(".")[:-1]])


def get_file_pathes(commit_dataset, num_commits: int = 2):
    """Get a file path from the column names"""
    file_pathes = pd.DataFrame()
    for commit in tqdm(
        commit_dataset.index[-num_commits:], desc="Collecting data for commits"
    ):
        one_commit = commit_dataset.loc[commit]
        only_changed = one_commit[one_commit.index.str.endswith("_changed")]
        changed_files = only_changed[only_changed > 0].index.str[:-8]
        file_pathes = pd.concat(
            (file_pathes, pd.DataFrame({"commit": commit, "file_path": changed_files}))
        )
    return file_pathes


def preprocess_path(path, use_stem=False):
    """Preprocess a path into a name"""
    path = fix_renamed_pathes(path)
    path = get_name_from_path(path)
    name = normalize(path)

    return lst_to_str(name)


def fix_renamed_pathes(path: str):
    """Fix a problem with naming of moved files"""
    if "{" in path:
        del_start, del_end = path.find("{"), path.find(" => ")
        path = path[:del_start] + path[del_end + 4 :]
        path = path.replace("}", "")
        path = path.replace("//", "/")
    return path


def get_name_from_path(path):
    """Get a name from the path"""
    return path.split("/")[-1]


def normalize(path: str):
    """Split a path into tokens"""
    delimiters = ["/", "_", "-", "."]
    regex_pattern = "|".join(map(re.escape, delimiters))
    path_tokens = re.split(regex_pattern, path)

    all_split_tokens = []
    for i in range(len(path_tokens)):
        token = path_tokens[i]
        split_token = re.findall(".[^A-Z]*", token)
        split_token = [w.lower() for w in split_token]
        all_split_tokens.extend(split_token)

    return all_split_tokens


def lst_to_str(lst, sep=" "):
    return sep.join([str(el) for el in lst])


def get_dataframe_with_attention(
    df_files: pd.DataFrame,
    df_tests: pd.DataFrame,
    model,
    word_to_index,
    max_len,
    device="cpu",
):
    """Get a dataframe with attention scores"""
    df_res = pd.DataFrame()
    for commit in tqdm(df_files["commit"].unique()):
        file_commit = df_files[df_files["commit"] == commit]
        test_commit = df_tests[df_tests["commit"] == commit]
        file_names = file_commit["name"].values
        test_names = test_commit["name"].values

        file_tokens_padded, file_lengths = prepare_tokens_lengths(
            file_names, word_to_index, max_len=max_len
        )
        test_tokens_padded, test_lengths = prepare_tokens_lengths(
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
            # print(preds)
            df_commit = pd.DataFrame(
                {
                    "allure_id": test_commit["allure_id"].iloc[
                        i_test : i_test + batch_size
                    ],
                    "commit": commit,
                    "attention": preds[
                        : min(batch_size, test_tokens_padded.size()[0] - i_test)
                    ],
                    "status": test_commit["status"].iloc[i_test : i_test + batch_size],
                }
            )
            df_res = pd.concat((df_res, df_commit))
    return df_res


def prepare_tokens_lengths(names: list[str], word_to_index, max_len: int = 20):
    """Prepare a list of tokens and lengths for the attention model by padding all of
    them to the :max_len lengths"""
    name_tokens = [
        torch.tensor([word_to_index[token] for token in name.split()]) for name in names
    ]
    name_lengths = torch.tensor([len(tokens) for tokens in name_tokens])

    # manually create a first sequence to make all of the sequence be the same lengths
    name_tokens[0] = nn.ConstantPad1d((0, max_len - len(name_tokens[0])), 0)(
        name_tokens[0]
    )
    name_tokens = [tokens for tokens in name_tokens]

    name_tokens_padded = nn.utils.rnn.pad_sequence(name_tokens, batch_first=True)

    return name_tokens_padded, name_lengths
