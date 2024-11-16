import torch
from typing import Union, List, Tuple


def pairwise_similarity(embeddings: List[Union[float, torch.Tensor]]) -> torch.Tensor:
    """
    Compute pairwise cosine similarity for a given set of embeddings.

    :param embeddings: Tensor or array-like of shape (n_samples, n_features)
    :return: Tensor of shape (n_samples, n_samples) containing cosine similarities
    """
    # Convert embeddings to a PyTorch tensor if they aren't already
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    # Normalize embeddings to get cosine similarity as dot product
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute pairwise cosine similarity matrix
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix


def topn_similar_trials_from_vector(embeddings: List[List[Union[float, torch.Tensor]]],
                                    query_vector: List[Union[float, torch.Tensor]],
                                    top_x: int = 5) -> List[Tuple[int, float]]:
    """
    Retrieve top n similar vectors from an embedding list

    :param embeddings: Tensor or array-like of shape (n_samples, n_features)
    :param query_vector: Tensor or array-like of shape (1, n_features)
    :param top_x: Top n items to return
    :return:
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    if not isinstance(query_vector, torch.Tensor):
        query_vector = torch.tensor(query_vector).unsqueeze(0)
    trial_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
    similarities = torch.mm(query_vector, trial_embeddings.T).squeeze(0)
    top_similarities, top_indices = torch.topk(similarities, top_x)
    similar_trials = [(idx.item(), score.item()) for idx, score in zip(top_indices, top_similarities)]
    return similar_trials


def topn_similar_trials(embeddings: List[List[Union[float, torch.Tensor]]],
                        query_index: int,
                        top_x: int = 5) -> List[Tuple[int, float]]:
    """
    Retrieve top n similar vectors from an embedding list

    :param embeddings: Tensor or array-like of shape (n_samples, n_features)
    :param query_index: Integer of the embedding to query with
    :param top_x: Top n items to return
    :return:
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    trial_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    query_embedding = trial_embeddings[query_index].unsqueeze(0)
    similarities = torch.mm(query_embedding, trial_embeddings.T).squeeze(0)
    top_similarities, top_indices = torch.topk(similarities, top_x + 1)
    similar_trials = [(idx.item(), score.item()) for idx, score in zip(top_indices, top_similarities) if
                      idx != query_index]
    return similar_trials
