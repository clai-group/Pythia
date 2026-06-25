"""pythia/nodes/evaluate.py"""
import csv
from pathlib import Path
from pythia.state import PythiaState, Metrics
from pythia.agents.specialist import SpecialistAgent
from pythia.evaluation import compute_metrics, determine_optimization_target
from pythia.diagnostics import compute_output_entropy

def run(state: PythiaState, backend) -> dict:
    """
    Node function signature: takes state, returns partial state dict.
    'backend' is passed via LangGraph config (see graph.py for how).
    """
    agent = SpecialistAgent(backend)
    predictions, raw_outputs = agent.classify_all(
        prompt = state.current_prompt,
        sop = state.sop,
        records=state.dev_records
    )
    metrics = compute_metrics(predictions, state.dev_labels)
    entropy = compute_output_entropy(raw_outputs)
    target = determine_optimization_target(
        metrics = metrics,
        sens_threshold = state.sens_threshold,
        spec_threshold = state.spec_threshold,
        priority = state.priority
    )

    # Log predictions to CSV
    Path(state.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(state.output_dir) / f'iteration_{state.current_iteration}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['note_text', 'prediction', 'ground_truth', 'raw_response']
        has_empi = state.dev_records and 'empi' in state.dev_records[0]
        if has_empi:
            header.insert(0, 'empi')
        writer.writerow(header)
        for record, pred, gt, raw in zip(state.dev_records, predictions, state.dev_labels, raw_outputs):
            row = []
            if has_empi and 'empi' in record:
                row.append(record['empi'])
            row.extend([record.get('text', ''), pred, gt, raw])
            writer.writerow(row)

    return {
        'current_dev_metrics': metrics,
        'current_dev_output_entropy': entropy,
        'optimization_target': target,
        'current_predictions': predictions,
        'current_tested_prompt': state.current_prompt
        }