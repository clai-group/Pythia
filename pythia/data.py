
"""
pythia/data.py

Dataset loading and validation.
Expects: a directory of CSVs, each with columns 'visit' and 'Ground Truth'.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing_extensions import Literal
import pandas as pd
from pydantic import BaseModel, model_validator

class ClinicalNote(BaseModel):
    patient_id: str
    visit_text: str
    label: int #0 or 1

    @model_validator(mode='after')
    def label_must_be_binary(cls, v):
        if v.label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {v.label}")
        return v

def load_dataset(data_path: str | Path, mode: Literal['concat', 'individual'] = 'concat') -> tuple[list[dict], list[int]]:
    """
    Loads all CSVs from data_path directory.
    Returns (records, labels) where:
      records = list of dicts with keys 'patient_id', 'visit_text'
      labels  = list of int (0 or 1), same order as records

    Modes
    -----
    concat     : one record per CSV file; all visits concatenated with newline.
                 Ground Truth is taken from the first non-null value in the file.
    individual : one record per row; each visit is its own classification unit.
                 Every row must have its own Ground Truth value.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset directory not found: {path}')

    csv_files = sorted(path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {path}')

    records: list[dict] = []
    labels:  list[int]  = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Validate required columns
        missing = {'Visit', 'Ground Truth'} - set(df.columns)
        if missing:
            raise ValueError(
                f'{csv_file.name} is missing columns: {missing}. '
                f'Found: {list(df.columns)}'
            )

        if mode == 'concat':
            visit_text = '\n'.join(df['Visit'].astype(str).tolist())
            gt_values  = df['Ground Truth'].dropna()
            if gt_values.empty:
                raise ValueError(
                    f'{csv_file.name} has no non-null Ground Truth values.'
                )
            note = ClinicalNote(
                patient_id = csv_file.stem,
                visit_text = visit_text,
                label = _parse_label(gt_values.iloc[0])
            )
            records.append({
                'patient_id': note.patient_id,
                'visit_text': note.visit_text
            })
            labels.append(note.label)

        elif mode == 'individual':
            for row_idx, row in df.iterrows():
                if pd.isna(row['Ground Truth']):
                    raise ValueError(
                        f'{csv_file.name} row {row_idx} has a null Ground Truth. '
                        f'In individual mode every row must have a label.'
                    )
                if pd.isna(row['visit']) or str(row['visit']).strip() == '':
                    continue  # skip empty visit rows silently
                note = ClinicalNote(
                    patient_id = f'{csv_file.stem}_row{row_idx}',
                    visit_text = str(row['visit']).strip(),
                    label = _parse_label(row['Ground Truth'])
                )
                records.append({
                    'patient_id': note.patient_id,
                    'visit_text': note.visit_text,
                    'source_file': csv_file.stem,   # keeps patient grouping recoverable
                    'row_index':   int(row_idx)
                })
                labels.append(note.label)

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Must be 'concat' or 'individual'."
            )

    if not records:
        raise ValueError(
            f'No records loaded from {path}. '
            f'Check that CSVs have non-empty visit columns.'
        )

    return records, labels


def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert list[dict] from state back to a DataFrame."""
    return pd.DataFrame(records)


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list[dict] for storage in state."""
    return df.to_dict(orient='records')

def _parse_label(raw) -> int:
    """
    Converts ground truth values to 0 or 1.
    Handles: 1/0, '1'/'0', 'Yes'/'No', 'yes'/'no', 'TRUE'/'FALSE', etc.
    """
    if isinstance(raw, (int, float)):
        return int(raw)
    normalized = str(raw).strip().lower()
    if normalized in ('1', 'yes', 'true', 'y', 'positive'):
        return 1
    if normalized in ('0', 'no', 'false', 'n', 'negative'):
        return 0
    raise ValueError(
        f"Unrecognised ground truth value: '{raw}'. "
        f"Expected one of: 1/0, Yes/No, True/False, Positive/Negative."
    )