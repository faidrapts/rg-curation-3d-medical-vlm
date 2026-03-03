"""
rg_curation: Reference-guided data curation for medical vision-language pretraining.

Pipeline stages
---------------
1. Embedding generation  – DreamSim (vision) and Clinical Longformer (text), or CLIP.
2. Alignment / filtering – vision-only, text-only, early-fusion (kNN), or CLIP-score.
3. Rank aggregation      – Reciprocal Rank Fusion (RRF) over vision + text scores.
4. Pool selection        – keep the top-N samples by similarity / RRF / CLIP score.
"""

__version__ = "1.0.0"
