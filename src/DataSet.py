from typing import Optional

from embed import load_jsonl
from settings import settings


class DataSet:
    """ Basic Dataset class for downstream applications"""
    def __init__(self, embedding_path: Optional[str], annotation_path: Optional[str]):
        if not embedding_path:
            data_embedding = load_jsonl(f'{settings.data_dir}\\core_data_embedding.jsonl')
        else:
            data_embedding = load_jsonl(f'{embedding_path}')

        if not annotation_path:
            data_annotation = load_jsonl(f'{settings.data_dir}\\core_data_annotation.jsonl')
        else:
            data_annotation = load_jsonl(f'{annotation_path}')

        self.id_text_dict = {}
        for item in data_annotation:
            self.id_text_dict[item['nctId']] = item['title']
        # {nctId:index}
        self.name_index_dict = {k: i for i, k in enumerate(self.id_text_dict.keys())}
        # {index:nctId}
        self.index_name_dict = {v: k for k, v in self.name_index_dict.items()}
        # {index:embedding}
        index_embedding = {}
        for item in data_embedding:
            index_embedding[self.name_index_dict[item['nctId']]] = item['embedding']
        # {index:embedding}
        self.sorted_index_embedding = dict(sorted(index_embedding.items()))
