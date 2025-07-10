import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import nip

@nip.nip
class BaseEmbedder():

    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 1024)
        self._model = AutoModel.from_pretrained(model_name)
        self._emb_size = self._model.config.hidden_size
        self._model.eval()

    @property
    def embedding_size(self):
        return self._emb_size

    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer

    def mxp(self, x):
        return max(x.min(), x.max(), key=abs)
    
    def _util_tonumpy(self, x):
        return x.numpy()[0, :].reshape(x.shape[1]).tolist()
    
    def _get_embeds(self, code_sample):
        with torch.no_grad():
            inputs = self.tokenizer(code_sample, return_tensors="pt", truncation=True)
            if inputs.input_ids.size(dim=1) > 0:
                outputs = self.model(**inputs).last_hidden_state
                mean_embedding = outputs.mean(dim=1)
                return mean_embedding
            else:
                return torch.zeros(self.embedding_size, 1)
    
    def create_vectors(self, path_data: str):
        raise NotImplementedError("Please Implement this method")


class TestsEmbedder(BaseEmbedder):
    def create_vectors(self, path_data: str = None, loc: list[int] = None, df_tests: pd.DataFrame = None):
        if path_data is not None:
            df_test_codes = pd.read_csv(path_data, sep=';')[["allure_id", "test_content"]]
        elif df_tests is not None:
            df_test_codes = df_tests[["allure_id", "test_content"]]
        else:
             raise ValueError("Choose input format data.")
        df_test_codes = df_test_codes.drop_duplicates("allure_id").set_index("allure_id")
        
        if loc is None:
            code = df_test_codes["test_content"]
        else:
            code = df_test_codes.loc[loc, "test_content"]
        embeds = code.apply(self._get_embeds)

        vectors = pd.DataFrame(embeds.apply(self._util_tonumpy).tolist(), index=code.index)
        
        return vectors


class DiffsEmbedder(BaseEmbedder):

    def create_vectors(self, path_data: str, method="delta", granularity="maxpooling"):

        code_before_and_after = self.get_code_before_and_after(path_data)
        embeds_before = code_before_and_after["before"].apply(self._get_embeds)
        embeds_after = code_before_and_after["after"].apply(self._get_embeds)
        
        if method == "before_and_after":
            if granularity == "files":
                vectors = pd.concat(
                    [pd.DataFrame(embeds_before.apply(self._util_tonumpy).tolist()),
                     pd.DataFrame(embeds_after.apply(self._util_tonumpy).tolist())], 
                    axis=1
                )

            elif granularity =="avg":
                vectors = pd.DataFrame(
                    [(embeds_before.sum()/embeds_before.shape[0]).numpy()[0], 
                     (embeds_after.sum()/embeds_after.shape[0]).numpy()[0]]
                )

            elif granularity == "maxpooling":
                frame_before = pd.DataFrame(embeds_before.apply(self._util_tonumpy).tolist())
                frame_after = pd.DataFrame(embeds_after.apply(self._util_tonumpy).tolist(), 
                                           columns=range(frame_before.shape[1], 2*frame_before.shape[1]))

                vectors = pd.concat([frame_before, frame_after], axis=1)

                vectors = vectors.apply(self.mxp, axis=0)
                
        elif method == "delta":
            if granularity == "files":
                
                bf = pd.DataFrame(embeds_before.apply(self._util_tonumpy).tolist())
                af = pd.DataFrame(embeds_after.apply(self._util_tonumpy).tolist())
                delta = af - bf
                vectors = pd.concat([bf, delta], axis=1)

            elif granularity == "maxpooling":
                frame_before = pd.DataFrame(embeds_before.apply(self._util_tonumpy).tolist())
                frame_after = pd.DataFrame(
                    embeds_after.apply(self._util_tonumpy).tolist(), 
                    columns=range(frame_before.shape[1], 2*frame_before.shape[1]))
                delta = frame_after.rename(
                    columns={i : i - frame_before.shape[1] for i in range(frame_before.shape[1], 2*frame_before.shape[1])}) - frame_before

                vectors = pd.concat([frame_before, delta], axis=1)

                vectors = vectors.apply(self.mxp, axis=0)
            
        return vectors
        
    
    def get_code_before_and_after(self, path_data: str):

        df_diff = pd.read_csv(path_data, sep=';')["code_diff"]
        df_diff = df_diff.fillna("\n")
        unprocessed_code = self.get_code(df_diff).apply(self.process_added_and_deleted_lines)
        code_before_and_after = pd.DataFrame(columns=["before", "after"])
        
        for i in range(len(unprocessed_code)):
            code_before_and_after.loc[i] = pd.Series([unprocessed_code.iloc[i][0], unprocessed_code.iloc[i][1]]).values
        
        return code_before_and_after
        
        
    def process_added_and_deleted_lines(self, diff: str):

        added_pattern = r'\{\+(.*?)\+\}'
        deleted_pattern = r'\[-(.*?)-\]'

        before_changes = []
        after_changes = []

        for line in diff.strip().split('\n'):
            deleted_line = re.sub(deleted_pattern, r'\1', line)
            deleted_line = re.sub(added_pattern, '', deleted_line)
            before_changes.append(deleted_line)

            added_line = re.sub(added_pattern, r'\1', line)
            added_line = re.sub(deleted_pattern, '', added_line)
            after_changes.append(added_line)

        before_changes = '\n'.join(before_changes)
        after_changes = '\n'.join(after_changes)

        return [before_changes, after_changes]
        
        
    def get_code(self, df_diff: pd.Series):

        without_metadata = df_diff.str.split("@@", expand=True)
        cols = without_metadata.columns[2::2]
        without_metadata = without_metadata[cols]
        code = without_metadata.apply(self._join_code_blocks, axis=1)
        
        return code
    
        
    def _join_code_blocks(self, x: pd.Series) -> str:
        combined = ''
        
        for block_n in x.index:
            if x[block_n] != None:
                combined += '\n'.join(x[block_n].split("\n"))
        return combined