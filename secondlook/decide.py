"""Turn an anomaly score into a decision. Pure Python, no dependencies."""


def verdict(score, thr, band=0.0):
    """good / defective / uncertain. band = half-width of the uncertain zone around thr."""
    if score >= thr + band:
        return "defective"
    if score <= thr - band:
        return "good"
    return "uncertain"


def final_bowl(v):
    """Fail-safe: anything not clearly good goes to the defective bowl for a human to check."""
    return "good" if v == "good" else "defective"
