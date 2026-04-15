#!/usr/bin/env python3
"""
Script to run clustering evaluation.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering.clustering_eval import run_clustering_evaluation

if __name__ == "__main__":
    run_clustering_evaluation()