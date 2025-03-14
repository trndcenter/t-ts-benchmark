import networkx as nx
import community.community_louvain as community
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple

# Global constants
DEBUG = False
DEFAULT_CLUSTER = -1
THRESHOLD = 3
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
    Parse a column name of the format <file_part>_<feature>, where <feature> is in FEATURES.
    Returns (file_part, feature) or None if parsing fails or if the column contains 'change_type'.
    """
    if "change_type" in col_name:
        return None
    for feat in FEATURES:
        suffix = f"_{feat}"
        if col_name.endswith(suffix):
            file_part = col_name[:-len(suffix)]
            return (file_part, feat)
    return None


def build_file_subgraph_from_changed(df: pd.DataFrame) -> nx.Graph:
    """
    Build a graph from columns ending with '_changed'. Only rows with non-zero values are considered.
    An edge between two files is added if they co-occur at least THRESHOLD times.
    """
    if "vcs_commit_sha" not in df.columns:
        df = df.reset_index().rename(columns={'index': 'vcs_commit_sha'})

    changed_cols = [c for c in df.columns if c.endswith("_changed") and "change_type" not in c]

    pair_weights = {}
    grouped = df.groupby("vcs_commit_sha")
    for commit_id, group in tqdm(grouped, desc="Grouping commits", leave=False):
        if group.empty:
            continue
        row = group.iloc[0]
        files_in_commit = [col[:-len("_changed")] for col in changed_cols if row[col] != 0]
        n = len(files_in_commit)
        for i in range(n):
            for j in range(i + 1, n):
                pair = tuple(sorted((files_in_commit[i], files_in_commit[j])))
                pair_weights[pair] = pair_weights.get(pair, 0) + 1

    G = nx.Graph()
    for (f1, f2), count in pair_weights.items():
        if count >= THRESHOLD:
            G.add_edge(f1, f2, weight=count)

    all_files = set()
    for col in changed_cols:
        if df[col].max() != 0:
            file_part = col[:-len("_changed")]
            all_files.add(file_part)
    for file_part in all_files:
        if file_part not in G:
            G.add_node(file_part)

    if DEBUG:
        print(f"[build_file_subgraph_from_changed] Result: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def cluster_files_in_graph(G: nx.Graph) -> dict[str, int]:
    """
    Apply the Louvain algorithm to graph G and return a mapping {file_part: cluster_id}.
    """
    if G.number_of_nodes() == 0:
        return {}
    partition = community.best_partition(G)
    return partition


def perform_graph_clustering(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Final function:
      1) Drops columns containing 'change_type'.
      2) Builds a graph from '_changed' columns.
      3) Applies Louvain clustering.
      4) Aggregates all features (except SPECIAL_COLS) into new columns named "cluster_<cid>_<feature>".
    Returns a tuple (df_result, file_to_cluster).
    """
    drop_cols = [c for c in df.columns if "change_type" in c]
    df_clean = df.drop(columns=drop_cols, errors='ignore')

    G = build_file_subgraph_from_changed(df_clean)
    file_to_cluster = cluster_files_in_graph(G)
    if DEBUG:
        print(f"[perform_graph_clustering] Clusters: {len(set(file_to_cluster.values()))} (including default)")

    special_cols_present = [col for col in df_clean.columns if col in SPECIAL_COLS]
    special_df = df_clean[special_cols_present] if special_cols_present else pd.DataFrame(index=df_clean.index)

    all_clusters = set(file_to_cluster.values())
    all_clusters.add(DEFAULT_CLUSTER)
    feature_set = {parsed[1] for col in df_clean.columns if col not in SPECIAL_COLS
                   for parsed in [_parse_file_feature(col)] if parsed is not None}


    new_cols = {
        f"cluster_{cid}_{feat}": pd.Series(0, index=df_clean.index)
        for cid in all_clusters for feat in feature_set
    }
    aggregated_df = pd.DataFrame(new_cols, index=df_clean.index)

    for col in tqdm(df_clean.columns, desc="Aggregating clusters", leave=False):
        if col in SPECIAL_COLS:
            continue
        parsed = _parse_file_feature(col)
        if parsed is None:
            continue
        file_part, feat = parsed
        cluster_id = file_to_cluster.get(file_part, DEFAULT_CLUSTER)
        new_col = f"cluster_{cluster_id}_{feat}"
        aggregated_df[new_col] = df_clean[col]

    df_result = pd.concat([special_df, aggregated_df], axis=1)

    if DEBUG:
        out_path = "df_after_clustering.csv"
        df_result.to_csv(out_path, index=False)
        print(f"Result saved to {out_path} (shape={df_result.shape})")

    return df_result, file_to_cluster


def transform_df_by_cluster(
    df: pd.DataFrame, file_to_cluster: dict, drop_original_file_cols=True, aggfunc="sum"
) -> pd.DataFrame:
    """
    Transforms the DataFrame by aggregating file columns based on cluster mapping.
    For each column of the format <file>_<feature>, determines the cluster from file_to_cluster
    (using DEFAULT_CLUSTER if not found) and aggregates values into new columns named "cluster_<cid>_<feature>".
    
    :param df: Input DataFrame with commit data.
    :param file_to_cluster: Mapping {file: cluster_id}.
    :param drop_original_file_cols: If True, original file columns are dropped.
    :param aggfunc: Aggregation function; only 'sum' is supported.
    :return: Transformed (aggregated) DataFrame.
    
    Optimized
    """
    special_cols_present = [col for col in df.columns if col in SPECIAL_COLS]
    transformed_special_df = df[special_cols_present] if special_cols_present else pd.DataFrame(index=df.index)

    feature_set = {parsed[1] for col in df.columns if col not in SPECIAL_COLS
                   for parsed in [_parse_file_feature(col)] if parsed is not None}
    all_possible_clusters = set(file_to_cluster.values())
    all_possible_clusters.add(DEFAULT_CLUSTER)

    new_cols = {
        f"cluster_{cid}_{feat}": pd.Series(0, index=df.index)
        for cid in all_possible_clusters for feat in feature_set
    }
    agg_df = pd.DataFrame(new_cols, index=df.index)

    for col in df.columns:
        if col in SPECIAL_COLS:
            continue
        parsed = _parse_file_feature(col)
        if parsed is None:
            continue
        file_part, feat = parsed
        cluster_id = file_to_cluster.get(file_part, DEFAULT_CLUSTER)
        new_col = f"cluster_{cluster_id}_{feat}"
        if aggfunc == "sum":
            agg_df[new_col] = agg_df[new_col] + df[col]
        else:
            raise NotImplementedError("Only aggfunc='sum' is supported.")

    if drop_original_file_cols:
        cols_to_drop = [col for col in df.columns if col not in SPECIAL_COLS and _parse_file_feature(col) is not None]
        df = df.drop(columns=cols_to_drop, errors="ignore")

    final_df = pd.concat([transformed_special_df, agg_df], axis=1)
    return final_df


def ensure_default_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns for the default cluster (cluster_-1) for the main features if they are absent.
    """
    for feat in FEATURES:
        col_name = f"cluster_{DEFAULT_CLUSTER}_{feat}"
        if col_name not in df.columns:
            df[col_name] = 0
    return df
