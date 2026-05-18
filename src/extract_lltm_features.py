#!/usr/bin/env python3
"""
LLTM Feature Extraction for PCR²
=================================
Computes 5 handcrafted text features for difficulty estimation.

Features (weights from Kang 2026):
  1. Sentence-length variance     (w=1.5)
  2. Long-word ratio (>8 chars)   (w=0.8)
  3. Normalized mean sentence len (w=0.6)
  4. Inverse type-token ratio     (w=1.2)
  5. Average word length           (w=0.3)

WARNING: These features achieve rho=0.068 on encyclopedic text (Wikipedia).
They only work when surface complexity correlates with conceptual difficulty
(e.g., children's textbook vs research paper). For same-register corpora,
use LLM-based difficulty rating or learned embeddings instead.
"""

import re
import numpy as np

LLTM_WEIGHTS = np.array([1.5, 0.8, 0.6, 1.2, 0.3])

def compute_lltm_features(text: str) -> np.ndarray:
    """
    Compute 5 LLTM text features from raw English text.
    
    Args:
        text: Raw text string.
    
    Returns:
        np.ndarray of shape (5,) with feature values.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 2:
        sentences = [text[:len(text)//2], text[len(text)//2:]]
    
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 5:
        return np.zeros(5)
    
    sent_lens = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    
    # Feature 1: Sentence length variance (normalized by mean)
    sent_var = np.std(sent_lens) / (np.mean(sent_lens) + 1e-6)
    
    # Feature 2: Long word ratio (words > 8 characters)
    long_ratio = sum(1 for w in words if len(w) > 8) / len(words)
    
    # Feature 3: Normalized mean sentence length
    mean_sent_len = np.mean(sent_lens) / 30.0
    
    # Feature 4: Inverse type-token ratio (1 - TTR)
    inv_ttr = 1.0 - (len(set(words)) / len(words))
    
    # Feature 5: Average word length (chars per word, normalized)
    avg_word_len = np.mean([len(w) for w in words]) / 10.0
    
    return np.array([sent_var, long_ratio, mean_sent_len, inv_ttr, avg_word_len])


def estimate_difficulty(text: str) -> float:
    """
    Estimate content difficulty using LLTM features.
    
    Returns raw LLTM score (not calibrated to IRT scale).
    Use linear calibration against known difficulty labels for routing.
    """
    features = compute_lltm_features(text)
    return float(features @ LLTM_WEIGHTS)


if __name__ == '__main__':
    # Demo
    easy = "Water is a liquid. It flows downhill. Fish live in water."
    hard = ("The renormalization group provides a systematic framework for "
            "analyzing the behavior of quantum field theories at different "
            "energy scales through the Callan-Symanzik equation.")
    
    print(f"Easy text: LLTM score = {estimate_difficulty(easy):.3f}")
    print(f"Hard text: LLTM score = {estimate_difficulty(hard):.3f}")
    print(f"Features (easy): {compute_lltm_features(easy)}")
    print(f"Features (hard): {compute_lltm_features(hard)}")
