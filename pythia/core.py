"""
pythia/core.py

Public entry point. The Pythia() function signature is backward-compatible
with the original codebase — existing call sites do not need to change.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from pythia.state import PythiaState, Metrics
from pythia.data  import load_dataset
from pythia.graph import build_graph
from pythia.llm.base import BaseLLMBackend


def set_global_determinism(seed: int = 0) -> None:
    """Apply global deterministic settings where possible.

    This sets environment variables and seeds for Python, numpy and (if
    available) PyTorch. It also disables some nondeterministic backend
    options. Note that some sources of nondeterminism (external APIs,
    GPU/driver differences, and some vendor libraries) cannot be fully
    eliminated from user-space.
    """
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    # Restrict thread counts to reduce race/scheduling variance
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    random.seed(seed)
    np.random.seed(seed)

    # Try configuring PyTorch determinism if available
    try:
        import torch
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        # Prefer deterministic algorithms where supported
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch not installed or failed to configure; continue gracefully
        pass


def Pythia(
    LLMbackend:     BaseLLMBackend,
    dev_data_path:  str,
    val_data_path:  str,
    output_dir:     str,
    SOP:            str,
    initial_prompt: str,
    iterations:     int   = 7,
    sens_threshold: float = 0.75,
    spec_threshold: float = 0.75,
    priority:       str   = 'sensitivity',
) -> PythiaState:
    """
    Run the Pythia autonomous prompt optimisation pipeline.

    Parameters
    ----------
    LLMbackend      Any BaseLLMBackend (OllamaBackend, OpenAICompatBackend, etc.)
    dev_data_path   Directory of CSVs for development/optimisation
    val_data_path   Directory of CSVs for held-out validation
    output_dir      Directory to write results JSON
    SOP             Standard operating procedure string (task description)
    initial_prompt  Starting prompt (can be just the symptom name)
    iterations      Maximum number of optimisation iterations
    sens_threshold  Stop when sensitivity >= this value
    spec_threshold  Stop when specificity >= this value
    priority        Which metric to optimise first when both are below threshold

    Returns
    -------
    PythiaState     The final state after optimisation + validation
    """
    # Apply global deterministic settings early. Seed can be set via
    # the PYTHIA_SEED environment variable (default 0).
    seed = int(os.getenv('PYTHIA_SEED', '0'))
    set_global_determinism(seed)

    print(f'[Pythia] Loading datasets...')
    dev_records, dev_labels = load_dataset(dev_data_path)
    val_records, val_labels = load_dataset(val_data_path)
    print(f'  Dev: {len(dev_records)} notes  |  Val: {len(val_records)} notes')

    #Initialize state
    initial_state = PythiaState(
        initial_prompt  = initial_prompt,
        current_prompt  = initial_prompt,
        sop             = SOP,
        max_iterations  = iterations,
        sens_threshold  = sens_threshold,
        spec_threshold  = spec_threshold,
        priority        = priority,
        output_dir      = output_dir,
        dev_records     = dev_records,
        dev_labels      = dev_labels,
        val_records     = val_records,
        val_labels      = val_labels,
    )

    #Build and run graph
    graph  = build_graph(LLMbackend)
    config = {'configurable': {'thread_id': f'pythia-{datetime.now():%Y%m%d-%H%M%S}'}}

    print(f'[Pythia] Starting optimisation ({iterations} max iterations)...')
    final_state_dict = graph.invoke(initial_state.model_dump(), config)
    final_state      = PythiaState(**final_state_dict)

    #Validate selected prompt
    print(f'\n[Pythia] Running validation on selected prompt...')
    final_state = _run_validation(final_state, LLMbackend)

    #Write outputs
    _write_outputs(final_state, output_dir)

    print(f'\n[Pythia] Done.')
    print(f'  Selected prompt: {final_state.selected_prompt}')
    if final_state.final_metrics:
        m = final_state.final_metrics
        print(f'  Validation — Sensitivity: {m.sensitivity:.3f} | '
              f'Specificity: {m.specificity:.3f} | F1: {m.f1:.3f}')
    return final_state


def _run_validation(state: PythiaState, backend: BaseLLMBackend) -> PythiaState:
    """Evaluate the selected prompt on the held-out validation set."""
    import csv
    from pathlib import Path
    from pythia.agents.specialist import SpecialistAgent
    from pythia.evaluation import compute_metrics

    if not state.selected_prompt:
        print('[Pythia] WARNING: no prompt was selected; skipping validation')
        return state

    agent = SpecialistAgent(backend)
    preds, raw_outputs = agent.classify_all(
        prompt = state.selected_prompt,
        sop = state.sop,
        records = state.val_records,
    )
    print(f"Sample raw outputs: {raw_outputs[:5]}")  # print first 5
    val_metrics = compute_metrics(preds, state.val_labels)

    # Log validation predictions to CSV
    Path(state.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(state.output_dir) / 'validation.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['note_text', 'prediction', 'ground_truth', 'raw_response']
        has_empi = state.val_records and 'empi' in state.val_records[0]
        if has_empi:
            header.insert(0, 'empi')
        writer.writerow(header)
        for record, pred, gt, raw in zip(state.val_records, preds, state.val_labels, raw_outputs):
            row = []
            if has_empi and 'empi' in record:
                row.append(record['empi'])
            row.extend([record.get('text', ''), pred, gt, raw])
            writer.writerow(row)

    return state.model_copy(update={'final_metrics': val_metrics})


def _write_outputs(state: PythiaState, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = Path(output_dir) / f'pythia_run_{timestamp}.json'

    summary = {
        'selected_prompt': state.selected_prompt,
        'final_metrics':   state.final_metrics.model_dump() if state.final_metrics else None,
        'total_iterations': state.current_iteration,
        'iterations': [
            {
                'iteration': s.iteration,
                'prompt': s.prompt,
                'tested_prompt': s.tested_prompt,
                'improved_prompt': s.improved_prompt,
                'dev_metrics': s.dev_metrics.model_dump(),
                'output_entropy': s.output_entropy,
                'controller_action': s.controller_action,
                'critique_summary': s.critique_summary,
            }
            for s in state.iteration_history
        ],
        'transition_history': [
            {
                'iteration': t.iteration,
                'unstable_prompt': t.unstable_prompt,
                'unstable_dev_metrics': t.unstable_dev_metrics.model_dump(),
                'output_entropy': t.output_entropy,
                'controller_action': t.action,
                'restored_prompt': t.restored_prompt,
                'reason': t.reason,
                'critique_summary': t.critique_summary,
            }
            for t in state.transition_history
        ],
        'embedding_drift_by_iteration': _compute_drift_series(state),
    }

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[Pythia] Results written to {out_path}')


def _compute_drift_series(state: PythiaState) -> list[float]:
    from pythia.diagnostics import compute_embedding_drift
    embeddings = [s.prompt_embedding for s in state.iteration_history
                  if s.prompt_embedding]
    if len(embeddings) < 2:
        return []
    return [
        compute_embedding_drift(embeddings[:i+2])
        for i in range(len(embeddings)-1)
    ]
