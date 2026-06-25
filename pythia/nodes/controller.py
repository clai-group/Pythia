"""pythia/nodes/controller.py"""
from pythia.state import PythiaState, ControllerDecision, IterationSnapshot, TransitionSnapshot
from pythia.diagnostics import (
    compute_embedding_drift,
    detect_oscillation,
    compute_prompt_embedding,
)

# Tunable thresholds: These are starting values. Tune empirically using the data.
DRIFT_THRESHOLD      = 0.40   # cosine distance; 0 - 1
ENTROPY_THRESHOLD    = 0.90   # normalised entropy
OSCILLATION_WINDOW   = 4      # how many recent F1 values to inspect
REGRESSION_TOLERANCE = 0.05   # allow up to 10% F1 drop before backtracking


def _summarise_critiques(critiques: list[str]) -> str:
    """Join critiques into one string for storage. Truncate if very long."""
    joined = ' | '.join(critiques)
    return joined[:500] if len(joined) > 500 else joined


def run(state: PythiaState) -> dict:
    metrics   = state.current_dev_metrics
    history   = state.iteration_history
    iteration = state.current_iteration

    # Rule 1: Convergence
    if state.optimization_target == 'converged':
        return _halt('Both thresholds met: Converged.', state, halt_reason='threshold')

    # Rule 2: Max iterations
    if iteration >= state.max_iterations:
        return _halt(f'Max iterations ({state.max_iterations}) reached.', state, halt_reason='max_iterations')

    # Rule 3: Rejection limit — force halt if we have rejected too many times in a row
    if state.consecutive_rejections >= state.max_consecutive_rejections:
        decision = ControllerDecision(
          action='continue',
          reason=f'Infinity War.',
        )
        return {
        'last_decision': decision,
        'consecutive_rejections': state.consecutive_rejections,   # reset on successful continue
        }

    # Rule 4: F1 regression → backtrack
    if len(history) >= 1:
        prev_f1 = history[-1].dev_metrics.f1
        curr_f1 = metrics.f1
        if curr_f1 < prev_f1 * (1 - REGRESSION_TOLERANCE):
            best_idx = max(range(len(history)),
                          key=lambda i: history[i].dev_metrics.f1)
            return _backtrack(
                f'F1 regression: {prev_f1:.3f} → {curr_f1:.3f}.',
                target_iter=best_idx,
                state=state,
            )

    # Rule 5: Oscillation → reset
    if len(history) >= OSCILLATION_WINDOW:
        recent_f1 = [s.dev_metrics.f1 for s in history[-OSCILLATION_WINDOW:]]
        if detect_oscillation(recent_f1):
            return _reset(
                f'Oscillation detected over last {OSCILLATION_WINDOW} iters.',
                state,
            )

    # Drift warning (log only)
    drift_note = ''
    if len(history) >= 2:
        embeddings = [s.prompt_embedding for s in history]
        drift = compute_embedding_drift(embeddings)
        if drift > DRIFT_THRESHOLD:
            drift_note = f' [WARNING: high embedding drift = {drift:.3f}]'

    # Default: continue — reset the rejection counter
    decision = ControllerDecision(
        action='continue',
        reason=f'No instability signals detected.{drift_note}',
    )
    return {
        'last_decision':          decision,
        'consecutive_rejections': 0,   # reset on successful continue
    }
#Helpers
def _best_prompt(state: PythiaState) -> str:
    """Select the prompt with the highest dev F1 from history."""
    if not state.iteration_history:
        return state.current_prompt
    best = max(state.iteration_history, key=lambda s: s.dev_metrics.f1)
    return best.prompt


def _log_unstable_prompt(reason: str, action: str, restored_prompt: str, state: PythiaState) -> TransitionSnapshot:
    return TransitionSnapshot(
        iteration=state.current_iteration,
        unstable_prompt=state.current_prompt,
        unstable_dev_metrics=state.current_dev_metrics,
        prompt_embedding=compute_prompt_embedding(state.current_prompt),
        output_entropy=state.current_output_entropy,
        critique_summary=_summarise_critiques(state.current_critiques),
        action=action,
        restored_prompt=restored_prompt,
        reason=reason,
    )


def _halt(reason: str, state: PythiaState, halt_reason: str = 'threshold') -> dict:
    """
    Halt the optimization and select a prompt for validation.
    
    Args:
        reason: Human-readable reason for halting
        state: Current PythiaState
        halt_reason: Either 'threshold' (use current prompt) or 'max_iterations' (use best prompt)
    """
    # Select prompt based on halt reason
    if halt_reason == 'threshold':
        selected = state.current_prompt  # Use the prompt that met the threshold
    elif halt_reason == 'max_iterations':
        selected = _best_prompt(state)   # Use the best prompt from history
    else:
        selected = state.current_prompt  # Fallback
    
    snapshot = IterationSnapshot(
        iteration         = state.current_iteration,
        prompt            = selected,
        tested_prompt     = state.current_tested_prompt,
        improved_prompt   = selected,
        dev_metrics       = state.current_dev_metrics,
        prompt_embedding  = compute_prompt_embedding(selected),
        output_entropy    = state.current_output_entropy,
        critique_summary  = _summarise_critiques(state.current_critiques),
        controller_action = 'halt',
    )
    updated_history = state.iteration_history + [snapshot]
    return {
        'last_decision':    ControllerDecision(action='halt', reason=reason),
        'selected_prompt':  selected,
        'iteration_history': updated_history,
    }


def _backtrack(reason: str, target_iter: int, state: PythiaState) -> dict:
    restored   = state.iteration_history[target_iter].prompt
    transition = _log_unstable_prompt(
        reason=reason, action='backtrack',
        restored_prompt=restored, state=state,
    )
    return {
        'last_decision': ControllerDecision(
            action='backtrack', reason=reason, target_iter=target_iter
        ),
        'current_prompt':          restored,
        'consecutive_rejections':  state.consecutive_rejections + 1,
        'transition_history':      state.transition_history + [transition],
    }


def _reset(reason: str, state: PythiaState) -> dict:
    transition = _log_unstable_prompt(
        reason=reason, action='reset',
        restored_prompt=state.initial_prompt, state=state,
    )
    return {
        'last_decision': ControllerDecision(action='reset', reason=reason),
        'current_prompt':          state.initial_prompt,
        'consecutive_rejections':  state.consecutive_rejections + 1,
        'transition_history':      state.transition_history + [transition],
    }