import torch
import numpy as np
from collections import Counter

def calculate_distinct_n(texts, n):
    """
    Calculate Distinct-N metric for a list of texts.
    
    Args:
        texts: List of strings (generated responses)
        n: The n-gram size (e.g., 1 for Distinct-1, 2 for Distinct-2)
        
    Returns:
        distinct_n score (float)
    """
    if not texts:
        return 0.0
    
    total_ngrams = 0
    unique_ngrams = set()
    
    for text in texts:
        # Simple whitespace tokenization
        tokens = text.split()
        if len(tokens) < n:
            continue
            
        # Generate n-grams
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total_ngrams += len(ngrams)
        for ngram in ngrams:
            unique_ngrams.add(ngram)
            
    if total_ngrams == 0:
        return 0.0
        
    return len(unique_ngrams) / total_ngrams

# Example usage for your models:
# 1. Collect all generated responses from a model evaluation run.
# 2. Pass the list of responses to this function.
# 
# results = ["I think A is the best choice.", "Option B seems correct.", ...]
# d1 = calculate_distinct_n(results, 1)
# d2 = calculate_distinct_n(results, 2)
