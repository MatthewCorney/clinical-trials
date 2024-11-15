from DataSet import DataSet
from settings import settings
from similarity import topn_similar_trials


def main():
    """
    Example Usage for getting similarity
    """
    data_set = DataSet(embedding_path=f'{settings.data_dir}\\core_data_embedding.jsonl',
                       annotation_path=f'{settings.data_dir}\\core_data_annotation.jsonl')
    # Retrieve example clinical Trial
    trial_id, trial_title = next(iter(data_set.id_text_dict.items()))
    print(f'{trial_id}')
    print(f'{trial_title}')
    # Get index
    query_index = data_set.name_index_dict[trial_id]
    # Run query
    similar_trials = topn_similar_trials(embeddings=list(data_set.sorted_index_embedding.values()),
                                         query_index=query_index)
    similar_trial_response = []
    for idx, score in similar_trials:
        response = {'score': score,
                    'text': data_set.id_text_dict[data_set.index_name_dict[idx]],
                    'id_val': data_set.index_name_dict[idx]
                    }
        similar_trial_response.append(response)
        print(response)
