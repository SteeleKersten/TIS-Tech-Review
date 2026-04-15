#!/usr/bin/env python3
"""
Script to run similarity evaluation.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from similarity.similarity_eval import run_similarity_evaluation

if __name__ == "__main__":
    run_similarity_evaluation()