"""
pythia/diagnostics.py

Trajectory diagnostic signals.
All functions are pure (no side effects, no LLM calls).
sentence-transformers is used for embedding — it runs locally.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy as scipy_entropy

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 is fast and good enough for drift detection.
        # For clinical tasks, consider: pritamdeka/S-PubMedBert-MS-MARCO
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def compute_prompt_embedding(prompt: str) -> list[float]:
    """
    Embeds a prompt string into a fixed-size float vector.
    Returns a plain Python list (JSON-serialisable).
    """
    model = _get_embed_model()
    vec   = model.encode(prompt, normalize_embeddings=True)
    return vec.tolist()


def compute_embedding_drift(embeddings: list[list[float]]) -> float:
    """
    Mean cosine distance between consecutive prompt embeddings.
    Range [0, 1]. Higher = prompt is changing more rapidly.
    Returns 0.0 if fewer than 2 embeddings provided.
    """
    if len(embeddings) < 2:
        return 0.0
    drifts = []
    for i in range(len(embeddings) - 1):
        a = np.array(embeddings[i])
        b = np.array(embeddings[i + 1])
        # cosine() returns distance (0 = identical, 1 = orthogonal)
        drifts.append(float(cosine(a, b)))
    return float(np.mean(drifts))


def compute_output_entropy(raw_outputs: list[str]) -> float:
    """
    Estimates entropy of the LLM output distribution across notes.
    A well-specified binary classifier outputs mostly 0s or 1s.
    Near-50/50 split => high entropy => prompt is underspecified.
    Range [0, 1] (bits, base-2, normalised to [0,1]).
    """
    if not raw_outputs:
        return 0.0
    positives = sum(
        1 for o in raw_outputs
        if '1' in o or 'yes' in o.lower() or 'true' in o.lower()
    )
    p = positives / len(raw_outputs)
    if p == 0.0 or p == 1.0:
        return 0.0
    # Binary entropy, normalised so max = 1.0
    return float(scipy_entropy([p, 1 - p], base=2))


def detect_oscillation(f1_series: list[float]) -> bool:
    """
    Returns True if F1 is oscillating (alternating high/low).
    Counts sign changes in first-differences.
    Three or more sign changes in the last 4+ values = oscillation.
    """
    if len(f1_series) < 4:
        return False
    diffs = [f1_series[i+1] - f1_series[i] for i in range(len(f1_series)-1)]
    sign_changes = sum(
        1 for i in range(len(diffs)-1)
        if diffs[i] * diffs[i+1] < 0
    )
    return sign_changes >= 3


def compute_inter_agent_disagreement(
    spec_critique:  str,
    sens_critique:  str,
) -> float:
    """
    Rough proxy for inter-agent disagreement: cosine distance between
    the embeddings of the sensitivity and specificity critiques.
    High distance => agents are pointing in very different directions
    => high task ambiguity (a Project 1 formal instability signal).
    """
    emb_spec = compute_prompt_embedding(spec_critique)
    emb_sens = compute_prompt_embedding(sens_critique)
    return float(cosine(np.array(emb_spec), np.array(emb_sens)))
