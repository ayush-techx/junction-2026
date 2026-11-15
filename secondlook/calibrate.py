"""Pick the anomaly threshold and uncertain band from scored good/defect images."""
from statistics import median


def rates(good, defect, thr):
    """Defect recall and false-reject rate at a threshold (score >= thr means defective)."""
    caught = sum(s >= thr for s in defect)
    rejected = sum(s >= thr for s in good)
    return (caught / len(defect) if defect else None,
            rejected / len(good) if good else None)


def suggest(good, defect=(), min_band=0.02):
    """Return (threshold, band, info).

    Separable sets: threshold in the middle of the gap, band = a third of the gap on each side,
    so borderline bricks get a second look instead of a guess.
    Overlapping sets: threshold that maximises recall - false rejects, band covers the overlap.
    Good images only: threshold just above the highest good score.
    """
    if not good:
        raise ValueError("need scores for good images")
    g_max = max(good)
    if not defect:
        thr = g_max + max(min_band, 0.1 * g_max)
        return thr, min_band, {"mode": "good-only", "good_max": g_max}

    d_min = min(defect)
    if d_min > g_max:
        gap = d_min - g_max
        thr = g_max + gap / 2
        band = max(min_band, gap / 3)
        mode = "separable"
    else:
        candidates = sorted(set(good) | set(defect))
        def gain(t):
            recall, false_rej = rates(good, defect, t)
            return recall - false_rej
        thr = max(candidates, key=gain)
        band = max(min_band, (g_max - d_min) / 2)
        mode = "overlap"
    recall, false_rej = rates(good, defect, thr)
    return thr, band, {"mode": mode, "good_max": g_max, "defect_min": d_min,
                       "good_median": median(good), "defect_median": median(defect),
                       "recall": recall, "false_reject": false_rej}
