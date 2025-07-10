from . import embedders
import pandas as pd
import os

import nip

@nip.nip
class DatasetCBA:
    def __init__(self, path_data: str,  
                 train_files: list[str], 
                 model_name: str,
                 inference: bool = False,
                 path_inference: str = os.path.join("inference", "cba")):
        
        self.is_inference = inference
        self.path_data = path_data
        self.path_inference = path_inference
        self.train_files = train_files
        self.diffs_embedder = embedders.DiffsEmbedder(model_name)
        self.tests_embedder = embedders.TestsEmbedder(model_name)
        self.train_dataset = None

    
    def create_dataset(
            self,
            start_commit: int = 0,
            last_commit: int = None,
            train_size: float = 0.5
        ) -> pd.DataFrame:

        commits = self._read_commits()

        if train_size == 0 or train_size == 1:
            train_commits = commits
        else:
            train_commits = commits[start_commit:last_commit]
            i_split = int(len(commits)*train_size)
            train_commits = train_commits[:i_split]

        
        all_tests_embeds = self._create_tests_embeds()

        if not self.is_inference:
            # TODO: in case of using different embedders for files and tests adjust columns size to correct
            dataset = pd.DataFrame(columns=[i for i in range(self.tests_embedder.embedding_size*3 + 1 + 2 + 2)])

            for commit in train_commits:

                cur_path = os.path.join(self.path_data, "file_stats", f"{commit}.csv")
                commit_embed = self.diffs_embedder.create_vectors(cur_path)

                needed_tests_stats = self._tests_stats_in_commit(commit)
                needed_tests_embeds = all_tests_embeds.loc[needed_tests_stats.index.to_list()]
                
                tests_data = needed_tests_embeds.join(needed_tests_stats)

                commit_data = pd.DataFrame(commit_embed).T.merge(tests_data, how='cross')
                commit_data.index = pd.MultiIndex.from_product([[commit.split('.')[0]], tests_data.index.to_list()])
                commit_data.reset_index(inplace=True)

                dataset.columns = commit_data.columns
                dataset = pd.concat([dataset, commit_data], axis=0)
            
            dataset.rename(columns={'level_0': 'vcs_commit_sha', 'level_1': 'allure_id'}, inplace=True)
            dataset.set_index(['vcs_commit_sha', 'allure_id'], inplace=True)
            dataset.columns = pd.concat([pd.Series([f"{i}_fb" for i in range(self.diffs_embedder.embedding_size)]),
                                        pd.Series([f"{i}_fd" for i in range(self.diffs_embedder.embedding_size)]),
                                        pd.Series([f"{i}_t" for i in range(self.tests_embedder.embedding_size)]),
                                        pd.Series(["test_file_path", "test_method"]),
                                        pd.Series(['status'])], axis=0)
        
        else:

            dataset = pd.DataFrame(columns=[i for i in range(self.tests_embedder.embedding_size*3 + 2 + 2)])

            for commit in train_commits:

                cur_path = os.path.join(self.path_data, "file_stats", f"{commit}.csv")
                commit_embed = self.diffs_embedder.create_vectors(cur_path)

                commit_data = pd.DataFrame(commit_embed).T.merge(all_tests_embeds, how='cross')
                commit_data.index = pd.MultiIndex.from_product([[commit.split('.')[0]], all_tests_embeds.index.to_list()])
                commit_data.reset_index(inplace=True)

                dataset.columns = commit_data.columns
                dataset = pd.concat([dataset, commit_data], axis=0)

            dataset.rename(columns={'level_0': 'vcs_commit_sha', 'level_1': 'allure_id'}, inplace=True)
            dataset.set_index(['vcs_commit_sha', 'allure_id'], inplace=True)
            dataset.columns = pd.concat([pd.Series([f"{i}_fb" for i in range(self.diffs_embedder.embedding_size)]),
                                        pd.Series([f"{i}_fd" for i in range(self.diffs_embedder.embedding_size)]),
                                        pd.Series([f"{i}_t" for i in range(self.tests_embedder.embedding_size)]),
                                        pd.Series(["test_file_path", "test_method"])], axis=0)
            
        self.train_dataset = dataset.reset_index()

        return dataset.reset_index()


    def _create_tests_embeds(self):

        all_tests  = pd.DataFrame(columns=[i for i in range(self.tests_embedder.embedding_size)])

        path_tests = os.path.join(self.path_data, "project_data")

        for commit in os.listdir(path_tests):

            cur_path = os.path.join(path_tests, commit)
            cur_frame = pd.read_csv(cur_path, sep=';')[["allure_id", "test_file_path", "test_method"]].set_index(["allure_id"])
            cur_ids = set(cur_frame.index.to_frame()["allure_id"].values.tolist())
            not_yet_built_tests_id = cur_ids - set(all_tests.index.to_numpy().tolist())
            not_yet_built_tests_names = cur_frame.loc[list(not_yet_built_tests_id)]

            if len(not_yet_built_tests_id) > 0:
                tmp_embeds = self.tests_embedder.create_vectors(cur_path, loc=list(not_yet_built_tests_id))
                all_tests = pd.concat([all_tests, tmp_embeds.join(not_yet_built_tests_names, how='inner')], axis=0)
        return all_tests
    
    def _tests_stats_in_commit(self, commit):
        
        test_frame = pd.read_csv(os.path.join(self.path_data, "test_results", f"{commit}.csv"), sep=";")
        test_frame = test_frame[(test_frame["status"] == 0) | (test_frame["status"] == 2)]
        test_frame["status"] = 1 - test_frame["status"]/2
        test_frame.drop(columns=["test_case_id", "vcs_commit_sha", "duration", "session_id", "launch_id"], inplace=True)
        
        test_frame.drop_duplicates(subset=["allure_id"], inplace=True)
        test_frame.set_index(["allure_id"], inplace=True)

        return test_frame
    
    def _read_commits(self):
        sessions = pd.read_csv(os.path.join(self.path_data, "sessions.csv"), sep=';')
        
        if "created_at" in sessions.columns:
            sessions = sessions.sort_values("created_at")

        unique_commits = sessions["vcs_commit_sha"].unique()

        return unique_commits