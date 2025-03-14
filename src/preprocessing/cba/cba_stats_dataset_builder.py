from .cba_dataset_builder import DatasetCBA
from ..commit_dataset_creator import CommitDatasetCreator

import pandas as pd
import numpy as np
import os

from xgboost import XGBClassifier

import nip

@nip.nip
class DatasetCBAStats:
    def __init__(self, 
                 path_data: str,  
                 model_name: str,
                 num_important_coordinates: int = 100,
                 cba_params: dict = None,
                 inference: bool = False,
                 path_inference: str = os.path.join('inference', 'cba_stats')):
        
        self.path_data = path_data
        self.path_inference = path_inference
        self.train_files = None
        self.is_inference = inference

        self.model_name = model_name
        self.cba_params = cba_params
        self.num_important_coordinates = num_important_coordinates
        
        self.train_dataset = None


    
    def create_dataset(self, start_commit : int = 0, train_size : float = 0.5, last_commit : int = None) -> pd.DataFrame:

        embeddings_creator = DatasetCBA(self.path_data, self.train_files, self.model_name, self.is_inference, self.path_inference)
        dataset_embeddings = embeddings_creator.create_dataset(start_commit=start_commit, train_size=train_size, last_commit=last_commit)
        dataset_embeddings.set_index(['vcs_commit_sha', 'allure_id'], inplace=True)
        
        if not self.is_inference:

            X_embeds, y_embeds = dataset_embeddings.drop(columns=['status']), dataset_embeddings['status']
            X_embeds.drop(columns=["test_file_path", "test_method"], inplace=True)
            model_cba = XGBClassifier(**self.cba_params).fit(X_embeds, y_embeds)
            coordinates_importance = model_cba.feature_importances_
            most_important_columns = pd.Series(X_embeds.columns).loc[np.argsort(coordinates_importance)[-self.num_important_coordinates - 1 : -1].tolist()]
            most_important_columns.to_csv(os.path.join(self.path_inference, "important_coords.csv"), index=False)
            important_embeddings_data = X_embeds[most_important_columns].astype(pd.SparseDtype(dtype=float, fill_value=0))

        else:
            
            important_embeddings_coords = pd.read_csv(os.path.join(self.path_inference, "important_coords.csv"))
            important_embeddings_data = dataset_embeddings[important_embeddings_coords['0']].astype(pd.SparseDtype(dtype=float, fill_value=0))

        stats_dataset_creator = CommitDatasetCreator(path_data=self.path_data, path_inference=self.path_inference, inference=self.is_inference)
        
        if not self.is_inference:
            dataset_stats = stats_dataset_creator.create_dataset(start_commit=start_commit, train_size=train_size, last_commit=None)[0].set_index(["vcs_commit_sha", "allure_id"])
            self.train_files = stats_dataset_creator.train_files
        
        else:
            dataset_stats = stats_dataset_creator.create_dataset(start_commit=start_commit, train_size=train_size, last_commit=None).set_index(["vcs_commit_sha", "allure_id"])

        dataset = dataset_stats.join(important_embeddings_data, how='inner')

        self.train_dataset = dataset.reset_index()
        return dataset.reset_index()
