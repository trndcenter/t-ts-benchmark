import json
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple

"""
Expected JSON Structure for Modular Clustering:

The JSON file should be a dictionary where each key represents a module path, 
and its corresponding value is a list of file paths that belong to that module.
The path for the required project_structure.json file is 'data/modular_structure'.

Example:
{
    "path/to/module1": [
      "path/to/file1",
      "path/to/file2"
    ],
    "path/to/module2": [
      "path/to/file3",
      "path/to/file4",
      "path/to/file5"
    ]
  }

Notes:
- Each file path must match the file paths found in your commit data.
- Files not listed in the JSON will be assigned to a default module (e.g., "undefined_module").
- This is just a template. Replace the example paths with your own data if you want to experiment.
"""

# Global constants
DEBUG = False
DEFAULT_MODULE = "undefined_module"
FEATURES = [
    "number_changes_3d",
    "number_changes_14d",
    "number_changes_56d",
    "distinct_authors",
    "lines_added",
    "lines_deleted",
    "changed",
]
SPECIAL_COLS = {"vcs_commit_sha"}


def _parse_file_feature(col_name: str) -> Optional[Tuple[str, str]]:
    """
    Parses a column name of the format <file_part>_<feature>.
    Returns a tuple (file_part, feature) if successful, or None if parsing fails
    or if the column name contains 'change_type'.
    """
    if "change_type" in col_name:
        return None
    for feat in FEATURES:
        suffix = f"_{feat}"
        if col_name.endswith(suffix):
            file_part = col_name[:-len(suffix)]
            return (file_part, feat)
    return None


def load_module_mapping(json_path: str) -> dict:
    """
    Loads a JSON file containing the module structure and creates a mapping from file path to module name.
    The assumed JSON format is:
    
      {
        "path_to_1_module": ["file_path_1", "file_path_2"],
        "path_to_2_module": ["file_path_3", "file_path_4"]
      }
    
    :param json_path: Path to the JSON file.
    :return: A dictionary mapping file paths to module names.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = {}
    for module_name, files in data.items():
        for file_path in files:
            mapping[file_path] = module_name
    return mapping


def perform_module_clustering(df: pd.DataFrame, json_path: str) -> (pd.DataFrame, dict):
    """
    Final function analogous to perform_graph_clustering, but using a module structure from a JSON file.
      1) Drops columns containing 'change_type'.
      2) Loads the module structure and creates a mapping from file to module.
      3) Aggregates file features by modules, creating new columns in the format "module_<module_name>_<feature>".
    
    :param df: Input DataFrame.
    :param json_path: Path to the JSON file containing module information.
    :return: A tuple containing the transformed DataFrame and the file-to-module mapping.
    """
    drop_cols = [c for c in df.columns if "change_type" in c]
    df_clean = df.drop(columns=drop_cols, errors='ignore')

    file_to_module = load_module_mapping(json_path)

    modules = set(file_to_module.values())
    modules.add(DEFAULT_MODULE)

    feature_set = {
        parsed[1]
        for col in df_clean.columns if col not in SPECIAL_COLS
        for parsed in [_parse_file_feature(col)] if parsed is not None
    }

    special_cols_present = [col for col in df_clean.columns if col in SPECIAL_COLS]
    special_df = df_clean[special_cols_present] if special_cols_present else pd.DataFrame(index=df_clean.index)

    new_cols = {
        f"module_{module}_{feat}": pd.Series(0, index=df_clean.index)
        for module in modules for feat in feature_set
    }
    aggregated_df = pd.DataFrame(new_cols, index=df_clean.index)

    for col in tqdm(df_clean.columns, desc="Aggregating modules", leave=False):
        if col in SPECIAL_COLS:
            continue
        parsed = _parse_file_feature(col)
        if parsed is None:
            continue
        file_part, feat = parsed
        module = file_to_module.get(file_part, DEFAULT_MODULE)
        new_col = f"module_{module}_{feat}"
        aggregated_df[new_col] = df_clean[col]

    df_result = pd.concat([special_df, aggregated_df], axis=1)

    if DEBUG:
        out_path = "df_after_module_clustering.csv"
        df_result.to_csv(out_path, index=False)
        print(f"Result saved to {out_path} (shape={df_result.shape})")

    return df_result, file_to_module


def transform_df_by_module(
    df: pd.DataFrame, file_to_module: dict, drop_original_file_cols=True, aggfunc="sum"
) -> pd.DataFrame:
    """
    Transforms the DataFrame by aggregating file feature columns based on the module mapping.
    For each file feature column, the module is determined (using DEFAULT_MODULE if not found),
    and values are aggregated into a new column "module_<module>_<feature>".
    
    :param df: Input DataFrame with commit data.
    :param file_to_module: Mapping from file to module name.
    :param drop_original_file_cols: If True, original file columns are dropped.
    :param aggfunc: Aggregation function; currently only 'sum' is supported.
    :return: The transformed (aggregated) DataFrame.
    """
    special_cols_present = [col for col in df.columns if col in SPECIAL_COLS]
    special_df = df[special_cols_present] if special_cols_present else pd.DataFrame(index=df.index)

    feature_set = {
        parsed[1]
        for col in df.columns if col not in SPECIAL_COLS
        for parsed in [_parse_file_feature(col)] if parsed is not None
    }
    modules = set(file_to_module.values())
    modules.add(DEFAULT_MODULE)

    new_cols = {
        f"module_{module}_{feat}": pd.Series(0, index=df.index)
        for module in modules for feat in feature_set
    }
    agg_df = pd.DataFrame(new_cols, index=df.index)

    for col in df.columns:
        if col in SPECIAL_COLS:
            continue
        parsed = _parse_file_feature(col)
        if parsed is None:
            continue
        file_part, feat = parsed
        module = file_to_module.get(file_part, DEFAULT_MODULE)
        new_col = f"module_{module}_{feat}"
        if aggfunc == "sum":
            agg_df[new_col] = agg_df[new_col] + df[col]
        else:
            raise NotImplementedError("Only aggfunc='sum' is supported.")

    if drop_original_file_cols:
        cols_to_drop = [
            col for col in df.columns
            if col not in SPECIAL_COLS and _parse_file_feature(col) is not None
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")

    df_transformed = pd.concat([special_df, agg_df], axis=1)
    return df_transformed


def ensure_default_module_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that columns for the default module (undefined_module) exist for all primary features.
    If a column is missing, it is added with a default value of 0.
    
    :param df: Input DataFrame.
    :return: DataFrame with ensured default module feature columns.
    """
    for feat in FEATURES:
        col_name = f"module_{DEFAULT_MODULE}_{feat}"
        if col_name not in df.columns:
            df[col_name] = 0
    return df