# Text Information Systems - Tech Review
Tech Review for Text Information Retrieval @ UIUC

## Project: Evaluating Sentence-BERT for Semantic Similarity and Document Clustering

This project evaluates the effectiveness of Sentence-BERT (SBERT) embeddings for semantic text similarity and document clustering tasks, comparing against traditional baselines like TF-IDF.

### Tasks
- **Semantic Similarity**: Compute cosine similarity on SBERT embeddings for sentence pairs and compare to gold-standard annotations.
- **Document Clustering**: Apply K-means clustering on SBERT embeddings and evaluate cluster quality using silhouette score.

### Datasets
- Semantic Similarity: STSB, STSbenchmark
- Document Clustering: AG News (subset for short documents)

### Baselines
- TF-IDF with cosine similarity for similarity
- TF-IDF with K-means for clustering

### Structure
- `src/`: Source code
  - `similarity/`: Similarity evaluation code
  - `clustering/`: Clustering evaluation code
  - `utils/`: Data loading and utilities
  - `evaluation/`: Metrics and evaluation functions
- `scripts/`: Scripts to run experiments
- `notebooks/`: Jupyter notebooks for analysis
- `data/`: Raw and processed data
- `results/`: Experiment outputs

### Setup
1. Create virtual environment: `python3 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run similarity evaluation: `python scripts/run_similarity.py`
5. Run clustering evaluation: `python scripts/run_clustering.py`
6. Analyze results in `notebooks/analysis.ipynb`
