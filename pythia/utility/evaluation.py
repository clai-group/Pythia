import pandas as pd
import os
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
   
def read_parquet_files(folder_path):
    dfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".parquet"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_parquet(full_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"ERROR reading {full_path}: {e}")
    if not dfs:
        raise ValueError("No parquet files found in folder or subfolders: " + folder_path)
    return pd.concat(dfs, ignore_index=True)

def read_csv_files(folder_path):
    dfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"ERROR reading {full_path}: {e}")
                    logging.error(f"ERROR reading CSV {full_path}: {e}")
    if not dfs:
        raise ValueError(f"No CSV files found in folder or subfolders: {folder_path}")
    return pd.concat(dfs, ignore_index=True)

def read_patient_outputs(folder_path):
    dfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv") and not file.startswith("._"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)
                    dfs.append(df)
                except Exception as e:
                    logging.error(f"Error reading {full_path}: {e}")
    if not dfs:
        raise ValueError(f"No CSV output files found in {folder_path}")
    return pd.concat(dfs, ignore_index=True)

def extract_yes_no(text):
    """
    Extract yes/no from LLM response using keyword matching.
    Handles variations like "Yes", "yes.", "YES,", "The patient shows signs: yes", etc.
    Returns: 1 for yes, 0 for no, None if ambiguous/unclear
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    text_clean = text.strip().lower()
    if not text_clean:
        return None
    
    # Check if response contains explicit yes
    if 'yes' in text_clean:
        return 1
    # Check if response contains explicit no
    elif 'no' in text_clean:
        return 0
    # If unclear, return None (will be handled separately)
    else:
        logging.warning(f"Ambiguous response, cannot extract yes/no: {text[:100]}")
        return None

def one_yes_nous(values):
    """
    Aggregate multiple visit-level responses to patient level.
    If ANY visit has 'yes' (value=1), patient is positive (return 1).
    Otherwise return 0.
    """
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    
    # If any visit says yes, patient is positive
    if any(v == 1 for v in valid_values):
        return 1
    else:
        return 0

def preprocess_data(df):
    # Convert response column to string, handling None/NaN values
    df['response'] = df['response'].astype(str).fillna('')
    # Remove rows with empty responses to avoid parsing errors
    df = df[df['response'].str.strip() != ''].copy()
    
    # Extract yes/no using keyword matching instead of regex
    df['yn'] = df['response'].apply(extract_yes_no)
    logging.info(f"YN Value Counts:\n{df.yn.value_counts(dropna=False)}\n")
    
    # Aggregate at patient level: if ANY visit for a patient is 'yes', patient is positive
    df['final_answer'] = df.groupby('empi')['yn'].transform(one_yes_nous)
    return df

def get_patient_data(df):
    df_pt = df[['empi', 'Ground Truth', 'final_answer']].drop_duplicates()
    logging.info(f"Patient-level Data Shape: {df_pt.shape}")
    logging.info(f"Answer Value Counts:\n{df_pt.final_answer.value_counts()}\n")
    return df_pt
    
def calculate_metrics(df, y_true_col="Ground Truth", y_pred_col="final_answer", output_file=None):
    # Ensure numeric labels and log composition for debugging
    y_true = pd.to_numeric(df[y_true_col], errors="coerce")
    y_pred = pd.to_numeric(df[y_pred_col], errors="coerce")

    # Report counts of classes seen in the inputs
    try:
        true_counts = y_true.value_counts(dropna=False).to_dict()
        pred_counts = y_pred.value_counts(dropna=False).to_dict()
        logging.info(f"y_true value counts: {true_counts}")
        logging.info(f"y_pred value counts: {pred_counts}")
    except Exception:
        logging.exception("Failed to log value counts for y_true/y_pred")

    # Coerce NaNs (unmapped/invalid values) to 0 and convert to int
    if y_true.isna().any() or y_pred.isna().any():
        logging.warning("NaN values detected in y_true/y_pred; coercing to 0 for metric calculation")
    y_true = y_true.fillna(0).astype(int)
    y_pred = y_pred.fillna(0).astype(int)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()

    n_pos = tp + fn
    n_neg = tn + fp

    # If there are no positive (or no negative) ground-truth cases,
    # returning 1.0 by default is misleading (it was causing sensitivity
    # to show 1.0 when no positives exist). Use 0.0 and log a warning so
    # downstream logic treats these edge-cases explicitly.
    if n_pos > 0:
        sensitivity = tp / n_pos
    else:
        logging.warning("No positive ground-truth cases (n_pos=0). Setting sensitivity=0.0")
        sensitivity = 0.0

    if n_neg > 0:
        specificity = tn / n_neg
    else:
        logging.warning("No negative ground-truth cases (n_neg=0). Setting specificity=0.0")
        specificity = 0.0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    results = pd.DataFrame([{
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "F1 Score": round(f1, 4),
        "PPV": round(ppv, 4),
        "NPV": round(npv, 4),
    }])

    if output_file:
        results.to_csv(output_file, index=False)
        logging.info(f"Metrics saved to {output_file}")

    return float(sensitivity), float(specificity)

def full_save_fn_fp(
    df_pt,       
    visit_df,      
    fn_file, fn_list,
    fp_file, fp_list,
):
    """
    Save FN/FP cases at visit level (only rows where the LLM made the error).
    df_pt: patient-level results with Ground Truth and final_answer
    visit_df: visit-level data with all visits and response
    
    FN: visits from FN patients where response predicted 0 (should have been 1)
    FP: visits from FP patients where response predicted 1 (should have been 0)
    """

    fn_patients = df_pt.loc[
        (df_pt["Ground Truth"] == 1) &
        (df_pt["final_answer"] == 0),
        "empi"
    ].unique()

    logging.info(f"FN patient count: {len(fn_patients)}")

    fn_df = None

    if len(fn_patients) > 0:
        fn_rows = visit_df[visit_df["empi"].isin(fn_patients)].copy()
        
        # Map response to 0/1 for filtering
        fn_rows['response_mapped'] = fn_rows['response'].astype(str).str.strip().str.lower()
        fn_rows['response_mapped'] = fn_rows['response_mapped'].map({
            "yes": 1, "yes.": 1, "1": 1, "true": 1,
            "no": 0, "no.": 0, "0": 0, "false": 0
        })
        
        # Keep only rows where response predicted 0 (the false negatives at visit level)
        fn_rows = fn_rows[fn_rows['response_mapped'] == 0].copy()
        fn_rows = fn_rows.drop(columns=['response_mapped'])

        logging.info(f"# FN Notes (where response=0): {fn_rows.shape[0]}, Patients: {fn_rows['empi'].nunique()}")

        if len(fn_rows) > 0:
            fn_rows.to_csv(fn_file, index=False)
            logging.info(f"FN notes saved to: {fn_file}")

            with open(fn_list, "w", encoding="utf-8") as f:
                for pid in fn_patients:
                    f.write(f"{pid}\n")

            logging.info(f"FN patient list saved to: {fn_list}")
            fn_df = fn_rows
        else:
            logging.info("No FN visit-level rows found after filtering")

    fp_patients = df_pt.loc[
        (df_pt["Ground Truth"] == 0) &
        (df_pt["final_answer"] == 1),
        "empi"
    ].unique()

    logging.info(f"FP patient count: {len(fp_patients)}")

    fp_df = None

    if len(fp_patients) > 0:
        fp_rows = visit_df[visit_df["empi"].isin(fp_patients)].copy()
        
        # Map response to 0/1 for filtering
        fp_rows['response_mapped'] = fp_rows['response'].astype(str).str.strip().str.lower()
        fp_rows['response_mapped'] = fp_rows['response_mapped'].map({
            "yes": 1, "yes.": 1, "1": 1, "true": 1,
            "no": 0, "no.": 0, "0": 0, "false": 0
        })
        
        # Keep only rows where response predicted 1 (the false positives at visit level)
        fp_rows = fp_rows[fp_rows['response_mapped'] == 1].copy()
        fp_rows = fp_rows.drop(columns=['response_mapped'])

        logging.info(f"# FP Notes (where response=1): {fp_rows.shape[0]}, Patients: {fp_rows['empi'].nunique()}")

        if len(fp_rows) > 0:
            fp_rows.to_csv(fp_file, index=False)
            logging.info(f"FP notes saved to: {fp_file}")

            with open(fp_list, "w", encoding="utf-8") as f:
                for pid in fp_patients:
                    f.write(f"{pid}\n")

            logging.info(f"FP patient list saved to: {fp_list}")
            fp_df = fp_rows
        else:
            logging.info("No FP visit-level rows found after filtering")

    return fn_df, fp_df

def normalize_label(val):
                """Convert various label formats to 0/1 with logging"""
                if pd.isna(val):
                    return None
                val_str = str(val).strip().lower()
                if 'yes' in val_str or val_str == '1' or val_str == 'true':
                    return 1
                elif 'no' in val_str or val_str == '0' or val_str == 'false':
                    return 0
                else:
                    logging.warning(f"Unexpected Ground Truth value: {val}")
                    return None
