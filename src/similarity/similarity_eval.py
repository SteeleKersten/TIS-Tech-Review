"""
Semantic similarity evaluation using SBERT and TF-IDF baselines.
"""

import pandas as pd
import numpy as np
import logging
from utils.data_loader import load_stsb_dataset, get_sbert_embeddings, get_tfidf_embeddings
from evaluation.metrics import evaluate_similarity, compute_tfidf_similarity, cosine_sim
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_sbert_similarity(df: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate SBERT on semantic similarity task.
    """
    logger.info("Generating SBERT embeddings...")
    sentences1 = df['sentence1'].tolist()
    sentences2 = df['sentence2'].tolist()

    embeddings1 = get_sbert_embeddings(sentences1)
    embeddings2 = get_sbert_embeddings(sentences2)

    logger.info("Computing similarities...")
    predictions = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        sim = cosine_sim(emb1, emb2)
        predictions.append(sim)

    gold_scores = df['similarity_score'].values / 5.0  # Normalize to 0-1

    results = evaluate_similarity(predictions, gold_scores)
    logger.info(f"SBERT results: {results}")
    return results

def evaluate_tfidf_similarity(df: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate TF-IDF baseline on semantic similarity.
    """
    logger.info("Generating TF-IDF embeddings...")
    all_texts = df['sentence1'].tolist() + df['sentence2'].tolist()
    _, vectorizer = get_tfidf_embeddings(all_texts)

    logger.info("Computing similarities...")
    predictions = []
    for _, row in df.iterrows():
        sim = compute_tfidf_similarity(row['sentence1'], row['sentence2'], vectorizer)
        predictions.append(sim)

    gold_scores = df['similarity_score'].values / 5.0

    results = evaluate_similarity(predictions, gold_scores)
    logger.info(f"TF-IDF results: {results}")
    return results

def run_similarity_evaluation():
    """
    Run similarity evaluation for both methods and datasets.
    """
    logger.info("Starting similarity evaluation...")
    results = {}

    # Load datasets once
    stsb_train = load_stsb_dataset('train')
    stsb_test = load_stsb_dataset('test')

    results['sbert_stsb'] = evaluate_sbert_similarity(stsb_train)
    results['tfidf_stsb'] = evaluate_tfidf_similarity(stsb_train)

    results['sbert_stsbenchmark'] = evaluate_sbert_similarity(stsb_test)
    results['tfidf_stsbenchmark'] = evaluate_tfidf_similarity(stsb_test)

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/similarity_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Similarity evaluation complete. Results saved to results/similarity_results.json")

if __name__ == "__main__":
    run_similarity_evaluation()