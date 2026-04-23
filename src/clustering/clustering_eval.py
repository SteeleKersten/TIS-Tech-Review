"""
Document clustering evaluation using SBERT and TF-IDF baselines.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from utils.data_loader import load_clustering_dataset, get_sbert_embeddings, get_tfidf_embeddings
from evaluation.metrics import evaluate_clustering
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_sbert_clustering(df: pd.DataFrame, n_clusters: int = 4) -> dict[str, float | int | str]:
    """
    Evaluate SBERT on document clustering.
    """
    logger.info("Generating SBERT embeddings...")
    texts = df['text'].tolist()
    embeddings = get_sbert_embeddings(texts)

    logger.info("Performing K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)

    silhouette = evaluate_clustering(embeddings, predicted_labels)

    results = {
        'silhouette_score': silhouette,
        'n_clusters': n_clusters,
        'method': 'sbert'
    }
    logger.info(f"SBERT clustering results: {results}")
    return results

def evaluate_tfidf_clustering(df: pd.DataFrame, n_clusters: int = 4) -> dict[str, float | int | str]:
    """
    Evaluate TF-IDF baseline on document clustering.
    """
    logger.info("Generating TF-IDF embeddings...")
    texts = df['text'].tolist()
    embeddings, _ = get_tfidf_embeddings(texts)

    logger.info("Performing K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)

    silhouette = evaluate_clustering(embeddings, predicted_labels)

    results = {
        'silhouette_score': silhouette,
        'n_clusters': n_clusters,
        'method': 'tfidf'
    }
    logger.info(f"TF-IDF clustering results: {results}")
    return results

def run_clustering_evaluation():
    """
    Run clustering evaluation for both methods.
    """
    logger.info("Starting clustering evaluation...")
    results = {}

    # Load dataset once
    df = load_clustering_dataset()

    results['sbert_clustering'] = evaluate_sbert_clustering(df)
    results['tfidf_clustering'] = evaluate_tfidf_clustering(df)

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Clustering evaluation complete. Results saved to results/clustering_results.json")

if __name__ == "__main__":
    run_clustering_evaluation()