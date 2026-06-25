"""pythia/nodes/record_iteration.py"""
from pythia.state import IterationSnapshot, PythiaState
from pythia.diagnostics import compute_prompt_embedding

def run(state: PythiaState) -> dict:
    snapshot = IterationSnapshot(
        iteration = state.current_iteration,
        prompt = state.current_prompt,
        tested_prompt = state.current_tested_prompt,
        improved_prompt = state.current_prompt,
        dev_metrics = state.current_dev_metrics,
        prompt_embedding = compute_prompt_embedding(state.current_prompt),
        output_entropy = state.current_output_entropy,
        critique_summary = _summarise_critiques(state.current_critiques),
        controller_action = (
            state.last_decision.action if state.last_decision else ''
        ),
    )
    return {
        'iteration_history': state.iteration_history + [snapshot],
        'current_iteration': state.current_iteration + 1,
        'current_critiques': [],   # clear for next iteration
    }


def _summarise_critiques(critiques: list[str]) -> str:
    """Join critiques into one string for storage. Truncate if very long."""
    joined = ' | '.join(critiques)
    return joined[:500] if len(joined) > 500 else joined
