import os
import time
import glob
import logging
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from utility.utils import is_meaningful_paragraph, clean_prompt
from agents.specialist import evaluate_cognitive_concerns
from utility.evaluation import (
    preprocess_data,
    get_patient_data,
    calculate_metrics,
    full_save_fn_fp,
    read_patient_outputs,
    normalize_label,
)

def validation_workflow(
    Backend,
    input_data_path: str,
    SOP: str,
    BasePrompt: str,
    output_path: str,
  ) -> None:
    """
    Validation-only workflow:
    - Runs specialist once
    - Aggregates patient outputs
    - Computes metrics
    - No improvers, no prompt changes, no iterations
    """
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Validation data path {input_data_path} does not exist.")
    os.makedirs(output_path, exist_ok=True)

    # Create output folder for validation
    base_output_path = os.path.join(
        output_path,
        f"validation_output_{os.path.basename(os.path.normpath(input_data_path))}"
    )
    os.makedirs(base_output_path, exist_ok=True)

    logging.info("Starting validation workflow...")
    print("Starting validation workflow...")

    if os.path.isfile(BasePrompt):
        with open(BasePrompt, "r", encoding="utf-8") as f:
            current_prompt = f.read()
    else:
        current_prompt = BasePrompt

    parquet_files = glob.glob(os.path.join(input_data_path, "*.parquet"))
    csv_files = glob.glob(os.path.join(input_data_path, "*.csv"))
    patient_files = parquet_files + csv_files

    if not patient_files:
        raise ValueError(f"No .parquet or .csv files found in {input_data_path}")

    patient_files.sort()
    logging.info(f"Discovered {len(patient_files)} validation patient files")
    print(f"Found {len(patient_files)} validation patient files")

    pt_output_path = os.path.join(base_output_path, "patient_level")
    os.makedirs(pt_output_path, exist_ok=True)

    for fp in patient_files:
        patient_id = os.path.splitext(os.path.basename(fp))[0]
        file_ext = os.path.splitext(fp)[1].lower()

        if file_ext == ".parquet":
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp)

        df["empi"] = patient_id
        df["response"] = None

        for idx, row in df.iterrows():
            text = row.get("Visit", "")
            if pd.isna(text) or str(text).strip() == "":
                continue
            try:
                df.at[idx, "response"] = evaluate_cognitive_concerns(
                    Backend, current_prompt + "\n" + SOP, text
                )
            except Exception as e:
                logging.exception(f"Error on {patient_id}, row {idx}: {e}")

        df.to_csv(os.path.join(pt_output_path, f"{patient_id}.csv"), index=False)

    print("Specialist evaluation completed.")

    try:
        large_df = read_patient_outputs(pt_output_path)
        specialist_csv = os.path.join(base_output_path, "specialist_validation.csv")
        large_df.to_csv(specialist_csv, index=False)
    except Exception as e:
        logging.exception("Failed to combine validation outputs.")
        raise

    df = preprocess_data(large_df)
    df_pt = get_patient_data(df)

    label_file = os.path.join(base_output_path, "patient_level_label_validation.csv")
    df_pt.to_csv(label_file, index=False)

    df_pt["Ground Truth"] = df_pt["Ground Truth"].astype(str).str.strip().str.lower()
    df_pt["Ground Truth"] = df_pt["Ground Truth"].map({
            "yes": 1,
            "yes.": 1,
            "1": 1,
            "true": 1,
            "no": 0,
            "no.": 0,
            "0": 0,
            "false": 0
            })
    df_pt["final_answer"] = df_pt["final_answer"].apply(lambda x: 1 if x != 0 else 0)
    df_pt["final_answer"] = df_pt["final_answer"].astype(str).str.strip().str.lower()
    df_pt["final_answer"] = df_pt["final_answer"].map({
            "yes": 1,
            "yes.": 1,
            "1": 1,
            "true": 1,
            "no": 0,
            "no.": 0,
            "0": 0,
            "false": 0
            })

    metrics_file = os.path.join(base_output_path, "metrics_validation.csv")
    sensitivity, specificity = calculate_metrics(
        df_pt, y_true_col="Ground Truth", y_pred_col="final_answer", output_file=metrics_file
    )

    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print("Validation workflow completed.")
