import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import faiss
from src.DataSet import DataSet
from src.settings import settings, logger
from typing import List, Union, Tuple
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from sklearn.metrics import silhouette_score


def perform_kmeans_faiss(data_scaled: np.ndarray,
                         n_clusters: int,
                         max_iter: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Kmeans clustering using the faiss implementation

    :param data_scaled: Embedding vectors
    :param n_clusters: Number of clusters
    :param max_iter: Iterations for FAISS
    :return:
    """
    logger.info(f"Starting faiss KMeans")
    start = time.time()
    verbose = True
    kmeans_faiss = faiss.Kmeans(d=data_scaled.shape[1], k=n_clusters, niter=max_iter, verbose=verbose)
    kmeans_faiss.train(data_scaled)
    faiss_time = time.time() - start
    logger.info(f"Time taken for faiss KMeans: {faiss_time:.4f} seconds")
    faiss_centroids = kmeans_faiss.centroids
    faiss_distances, faiss_labels = kmeans_faiss.index.search(data_scaled, kmeans_faiss.k)
    return faiss_centroids, faiss_distances, faiss_labels


def baysian_optimisation(data_scaled: np.ndarray, min_clusters: int, max_clusters: int, max_iter: int = 400,
                         n_calls: int = 10) -> int:
    """
    Optimisation Wrapper for faiss clustering

    :param n_calls: Number of calls for optimisation
    :param data_scaled: Embedding vectors
    :param min_clusters: Minimum number of clusters
    :param max_clusters: Maximum number of clusters
    :param max_iter: Iterations for FAISS
    :return:
    """
    search_space = [Integer(min_clusters, max_clusters, name='n_clusters')]

    @use_named_args(search_space)
    def objective(n_clusters):
        try:
            # Perform clustering with the given number of clusters
            faiss_centroids, faiss_distances, faiss_labels = perform_kmeans_faiss(
                data_scaled=data_scaled,
                n_clusters=n_clusters,
                max_iter=max_iter
            )

            # Calculate silhouette score
            faiss_silhouette = silhouette_score(data_scaled, faiss_labels[:, 0])
            logger.info(f"Score: {faiss_silhouette:.4f}")
            return -faiss_silhouette

        except Exception as e:
            logger.info(f"Error with n_clusters={n_clusters}: {e}")
            return float('inf')

    results = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    optimal_n_clusters = results.x[0]
    logger.info(f"Optimal n_clusters: {optimal_n_clusters}")
    return optimal_n_clusters
