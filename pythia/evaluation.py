"""
pythia/evaluation.py

Pure functions. No LLM calls, no state, no side effects.
Test these first — everything else depends on them being correct.
"""
from pythia.state import Metrics

def compute_metrics(predictions: list[int], labels: list[int]) -> Metrics:
    """
    predictions: list of 0/1 ints from the Specialist agent
    labels:      list of 0/1 ground-truth ints, same length
    Returns a Metrics object.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f'predictions ({len(predictions)}) and labels '
            f'({len(labels)}) must have the same length'
        )

    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) > 0 else 0.0)
    accuracy = (tp + tn) / len(labels) if labels else 0.0

    return Metrics(
        sensitivity = round(sensitivity, 4),
        specificity = round(specificity, 4),
        f1          = round(f1, 4),
        accuracy    = round(accuracy, 4),
        tp=tp, tn=tn, fp=fp, fn=fn
    )

def determine_optimization_target(
    metrics:         Metrics,
    sens_threshold:  float,
    spec_threshold:  float,
    priority:        str
) -> str:
    """
    Decides whether to optimise sensitivity or specificity next.
    If both are below threshold, returns the priority.
    If one is below threshold, returns that one.
    If both are above threshold, returns 'converged'.
    """
    sens_ok = metrics.sensitivity >= sens_threshold
    spec_ok = metrics.specificity >= spec_threshold

    if sens_ok and spec_ok:
        return 'converged'
    if sens_ok and not spec_ok:
        return 'specificity'
    if spec_ok and not sens_ok:
        return 'sensitivity'
    # Both below threshold: use priority
    return priority
