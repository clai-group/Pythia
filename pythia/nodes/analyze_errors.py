"""pythia/nodes/analyze_errors.py"""
from pythia.state import PythiaState
from pythia.agents.improver import ImproverAgent

def run(state: PythiaState, backend) -> dict:
    target      = state.optimization_target
    predictions = state.current_predictions

    if not predictions:
        raise ValueError(
            'current_predictions is empty. '
            'Ensure nodes/evaluate.py is populating it before analyze_errors runs.'
        )

    if target == 'sensitivity':
        error_indices = [
            i for i, (pred, label) in enumerate(zip(predictions, state.dev_labels))
            if pred == 0 and label == 1
        ]
    elif target == 'specificity':
        error_indices = [
            i for i, (pred, label) in enumerate(zip(predictions, state.dev_labels))
            if pred == 1 and label == 0
        ]
    else:
        return {'current_critiques': []}

    error_notes = [
        state.dev_records[i]['visit_text']
        for i in error_indices
    ]

    if not error_notes:
        return {'current_critiques': []}

    agent     = ImproverAgent(backend)
    critiques = agent.critique_errors(
        prompt      = state.current_prompt,
        error_notes = error_notes,
        target      = target,
    )
    return {'current_critiques': critiques}


# def _get_predictions(state: PythiaState) -> list[int]:
#     """
#     Re-derive predictions from the confusion matrix stored in metrics.
#     NOTE: This is a simplification. For full correctness, store
#     per-note predictions in state (add a 'current_predictions' field
#     to PythiaState and populate it in nodes/evaluate.py).
#     The implementation below re-runs the specialist, which is expensive.
#     Prefer adding current_predictions to state instead.
#     """
#     # TODO: Replace with state.current_predictions once that field is added.
#     # For now, raise an informative error to prompt the fix.
#     raise NotImplementedError(
#         'Add current_predictions: list[int] to PythiaState and populate it '
#         'in nodes/evaluate.py. Then read it here instead of re-running the agent.'
#     )
