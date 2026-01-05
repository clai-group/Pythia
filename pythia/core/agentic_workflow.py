import os
import time
import glob
import logging
import pandas as pd
from pythia.utility.utils import is_meaningful_paragraph, clean_prompt
from pythia.agents.specialist import evaluate_cognitive_concerns
from pythia.utility.evaluation import (
    preprocess_data,
    get_patient_data,
    calculate_metrics,
    full_save_fn_fp,
    read_patient_outputs,
    normalize_label,
)
from pythia.agents.specificity import specificity_agent, summarizer_specificity
from pythia.agents.sensitivity import sensitivity_agent, summarizer_sensitivity

def run_agentic_workflow(
    Backend,
    input_data_path: str,
    SOP: str,
    BasePrompt: str,
    output_path: str,
    Iterations: str,
    sensitivity_threshold: float,
    specificity_threshold: float,
    priority: str
) -> None:
    """
    Runs the full Development Dataset workflow.
    Args: 
        Backend: LLM backend instance.
        input_data_path (str): Path to input data folder.
        SOP (str): Standard Operating Procedure text.
        BasePrompt (str): Initial prompt text or path to prompt file.
        output_path (str): Path to output folder.
        Iterations (int): Maximum number of iterations.
        sensitivity_threshold (float): Sensitivity threshold to reach.
        specificity_threshold (float): Specificity threshold to reach.
        priority (str): "sensitivity" or "specificity" indicating which metric to prioritize.
    """
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input data path {input_data_path} does not exist.")
    os.makedirs(output_path, exist_ok=True)

    if priority is None:
        priority = "sensitivity"
    priority = priority.lower().strip()
    if priority not in ("sensitivity", "specificity"):
        priority = "sensitivity"

    if not isinstance(Iterations, int) or Iterations < 1:
        raise ValueError("Iterations must be positive integer")

    if os.path.isfile(BasePrompt):
        with open(BasePrompt, "r", encoding="utf-8") as f:
            current_prompt = f.read()
    else:
        current_prompt = BasePrompt

    base_output_path = os.path.join(output_path, f"output_{os.path.basename(os.path.normpath(input_data_path))}")
    os.makedirs(base_output_path, exist_ok=True)

    log_file = os.path.join(
        base_output_path,
        f"workflow_maxiter_{Iterations}_sens{int(sensitivity_threshold*100)}_spec{int(specificity_threshold*100)}.log",
    )
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parquet_files = glob.glob(os.path.join(input_data_path, "*.parquet"))
    csv_files     = glob.glob(os.path.join(input_data_path, "*.csv"))
    patient_files = parquet_files + csv_files

    if not patient_files:
        raise ValueError(f"No .parquet or .csv files found directly in {input_data_path}")

    patient_files.sort()
    logging.info(f"Auto-discovered {len(patient_files)} patient files (direct files, no subfolders)")
    print(f"Found {len(patient_files)} patient files in {input_data_path}")

    patient_id_list = []
    file_path_map = {}
    for fp in patient_files:
        patient_id = os.path.splitext(os.path.basename(fp))[0]
        patient_id_list.append(patient_id)
        file_path_map[patient_id] = fp

   
    iter = 1
    previous_specificity = None
    previous_sensitivity = None
    current_specificity = 0.0
    current_sensitivity = 0.0

    while iter <= Iterations and (
        current_specificity < specificity_threshold or current_sensitivity < sensitivity_threshold
    ):
        logging.info(f"--- Start Iteration {iter} ---")
        print(f"\n--- Start Iteration {iter} ---")
        print(f"\n --- Iterartion {iter} prompt: {current_prompt} ---")
        output_folder = os.path.join(base_output_path, f"iter_{iter}_{Backend.__class__.__name__}")
        os.makedirs(output_folder, exist_ok=True)
        output_pt_folder = os.path.join(output_folder, "patient_level")
        os.makedirs(output_pt_folder, exist_ok=True)

        start_iter_t0 = time.time()

        for pt_idx, patient_id in enumerate(patient_id_list):
            input_file_path = file_path_map[patient_id]
            file_ext = os.path.splitext(input_file_path)[1].lower()

            output_csv_path = os.path.join(output_pt_folder, f"{patient_id}.csv")
            if os.path.isfile(output_csv_path):
                logging.info(f"Patient-level file exists, skipping {patient_id}: {output_csv_path}")
                continue

            try:
                if file_ext == ".parquet":
                    subset_df = pd.read_parquet(input_file_path)
                else:  
                    subset_df = pd.read_csv(input_file_path)
                subset_df["empi"] = patient_id
                logging.info(f"Loaded {file_ext[1:]} for patient {patient_id}: shape={subset_df.shape}")
            except Exception as e:
                logging.exception(f"Failed to load {input_file_path}: {e}")
                continue
                
            if 'response' in subset_df.columns:
              subset_df = subset_df.drop(columns=['response'])
            subset_df['response'] = pd.NA

            logging.info(f"Running LLM specialist on {patient_id} ({len(subset_df)} reports)")
            #Specialist
            for row_idx, row in subset_df.iterrows():
                report_text = row.get("Visit", None)
                if pd.isna(report_text) or str(report_text).strip() == "":
                    subset_df.at[row_idx, "response"] = None
                    continue

                try:
                    t0 = time.time()
                    response = evaluate_cognitive_concerns(Backend, current_prompt + "\n" + SOP, report_text)
                    subset_df.at[row_idx, "response"] = response
                    logging.info(f"Specialist done for {patient_id}, row {row_idx}, t(s)={(time.time()-t0):.2f}")
                except Exception as e:
                    logging.exception(f"evaluate_cognitive_concerns failed for {patient_id}, row {row_idx}: {e}")
                    subset_df.at[row_idx, "response"] = None

            try:
                subset_df.to_csv(output_csv_path, index=False)
                logging.info(f"Saved specialist output for {patient_id} -> {output_csv_path}")
            except Exception as e:
                logging.exception(f"Failed to save {output_csv_path}: {e}")
            finally:
                del subset_df

        elapsed_minutes = (time.time() - start_iter_t0) / 60.0
        logging.info(f"Specialist iteration {iter} finished in {elapsed_minutes:.2f} min.")
        print(f"Specialist iteration {iter} finished in {elapsed_minutes:.2f} min.")

        try:
            large_df = read_patient_outputs(output_pt_folder)
            if large_df is None or len(large_df) == 0:
                logging.warning(f"No specialist outputs found in {output_pt_folder}; ending workflow.")
                print("No specialist outputs found; terminating.")
                break
            specialist_csv = os.path.join(output_folder, f"specialist_iter_{iter}.csv")
            large_df.to_csv(specialist_csv, index=False)
        except Exception as e:
            logging.exception(f"Failed to combine patient outputs: {e}")
            raise

        evaluation_output_folder = os.path.join(output_folder, f"evaluation_iter_{iter}")
        os.makedirs(evaluation_output_folder, exist_ok=True)

        try:
            df = preprocess_data(large_df)
            logging.info(f"After preprocess_data: Total rows={len(df)}, with yn extraction results")
            
            df_pt = get_patient_data(df)
            pt_label = os.path.join(evaluation_output_folder, f"patient_level_label_{iter}.csv")
            df_pt.to_csv(pt_label, index=False)
            logging.info(f"Patient-level labels saved to {pt_label}")
        except Exception as e:
            logging.exception(f"Error in preprocessing: {e}")
            raise

        try:
            df_pt["Ground Truth"] = df_pt["Ground Truth"].apply(normalize_label)
            logging.info(f"Normalized Ground Truth counts:\n{df_pt['Ground Truth'].value_counts(dropna=False)}")
            
            df_pt["final_answer"] = pd.to_numeric(df_pt["final_answer"], errors="coerce").fillna(0).astype(int)
            logging.info(f"Final answer counts:\n{df_pt['final_answer'].value_counts(dropna=False)}")
            
            # Map response column to 0/1 for visit-level filtering in FN/FP analysis
            # This is only used for FN/FP identification, not for final metrics
            df_pt["response"] = df_pt["response"].astype(str).str.strip().str.lower()
            response_map = {
                "yes": 1, "yes.": 1, "1": 1, "true": 1,
                "no": 0, "no.": 0, "0": 0, "false": 0
            }
            df_pt["response"] = df_pt["response"].map(response_map)
            logging.info(f"Response column mapped for FN/FP analysis:\n{df_pt['response'].value_counts(dropna=False)}")
        except Exception as e:
            logging.exception(f"Error normalizing label columns: {e}")
            raise

        try:
            metrics_output_file = os.path.join(evaluation_output_folder, "metrics_results.csv")
            current_sensitivity, current_specificity = calculate_metrics(
                df_pt, y_true_col="Ground Truth", y_pred_col="final_answer", output_file=metrics_output_file
            )
            logging.info(f"Metrics (iter {iter}) | Sens={current_sensitivity:.4f} | Spec={current_specificity:.4f}")
            print(f"Iteration {iter} metrics: Sens={current_sensitivity:.4f}, Spec={current_specificity:.4f}")
        except Exception as e:
            logging.exception(f"Error calculating metrics: {e}")
            raise

        try:
            fn_df, fp_df = full_save_fn_fp(
                df_pt,
                large_df,
                fn_file=os.path.join(evaluation_output_folder, "fn_result.csv"),
                fn_list=os.path.join(evaluation_output_folder, "fn_patients_id.txt"),
                fp_file=os.path.join(evaluation_output_folder, "fp_result.csv"),
                fp_list=os.path.join(evaluation_output_folder, "fp_patients_id.txt"),
            )
            fn_df = fn_df if fn_df is not None else pd.DataFrame()
            fp_df = fp_df if fp_df is not None else pd.DataFrame()
            
        except Exception as e:
            logging.exception(f"Error in save_fn_fp: {e}")
            fn_df = pd.DataFrame()
            fp_df = pd.DataFrame()

        #improver block
        debug_msg = (
            f"DEBUG before improver: priority={priority}, "
            f"current_sens={current_sensitivity:.4f}/{sensitivity_threshold}, "
            f"current_spec={current_specificity:.4f}/{specificity_threshold}, "
            f"FN={len(fn_df)}, FP={len(fp_df)}"
        )
        logging.info(debug_msg)
        print(debug_msg)

        improver_iter = iter

        #Sensitivity improver
        if (priority == "sensitivity" and current_sensitivity < sensitivity_threshold) or (priority == "specificity" and current_specificity > specificity_threshold and current_sensitivity < sensitivity_threshold):
            print(f"\nENTERING SENSITIVITY IMPROVER for iteration {improver_iter} | Current Sensitivity: {current_sensitivity:.4f}")
            logging.info(f"Entering sensitivity improver (iter {improver_iter}) | Sens={current_sensitivity:.4f} | FN notes={len(fn_df)}")

            sensitivity_output_folder = os.path.join(output_folder, f"sensitivity_iter_{improver_iter}")
            os.makedirs(sensitivity_output_folder, exist_ok=True)
            sst_fn_output_csv = os.path.join(sensitivity_output_folder, f"sensitivity_iter_{improver_iter}_result.csv")
            clean_output_csv = os.path.join(sensitivity_output_folder, f"sensitivity_iter_{improver_iter}_result_cleaned.csv")
            p_output_file_path = os.path.join(sensitivity_output_folder, f"ap{improver_iter}.txt")
            if fn_df.empty:
                logging.warning("FN dataframe empty; skipping sensitivity improver.")
            else:
                if not os.path.exists(sst_fn_output_csv) or os.path.getsize(sst_fn_output_csv) == 0:
                    results = []
                    for idx, row in fn_df.iterrows():
                        text = row.get("Visit", "")
                        empi = row.get("empi", None)
                        if pd.isna(text) or str(text).strip() == "":
                            continue
                        try:
                            evidence = sensitivity_agent(Backend, text, SOP)
                            results.append({"empi": empi, "evidence": evidence})
                        except Exception as e:
                            logging.exception(f"sensitivity_agent failed for fn row {idx}: {e}")
                            results.append({"empi": empi, "evidence": None})
                    result_df = pd.DataFrame(results)
                    result_df.to_csv(sst_fn_output_csv, index=False)
                    print(f"Evidence saved to {sst_fn_output_csv}")
                else:
                    result_df = pd.read_csv(sst_fn_output_csv)
                    print(f"Loaded existing sensitivity evidence CSV ({len(result_df)} rows)")

                if "evidence" in result_df.columns:
                    result_df["is_meaningful"] = result_df["evidence"].apply(
                        lambda x: is_meaningful_paragraph(x) if pd.notna(x) else False
                    )
                    filtered_df = result_df[result_df["is_meaningful"]].copy()
                    filtered_df.to_csv(clean_output_csv, index=False)
                    print(f"{len(filtered_df)} meaningful evidence paragraphs kept")
                else:
                    filtered_df = pd.DataFrame()
                    print("No evidence column found in sensitivity evidence CSV.")

                if not os.path.exists(p_output_file_path):
                    if filtered_df.empty:
                        print("No meaningful evidence produced; keeping current prompt.")
                        new_prompt = current_prompt
                    else:
                        try:
                            print(f"Calling summarizer_sensitivity with {len(filtered_df)} items")
                            new_prompt = summarizer_sensitivity(
                                Backend,
                                filtered_df["evidence"].tolist(),
                                current_prompt,
                                SOP,
                            )
                            print(f"Summarizer returned {len(new_prompt)} characters")
                        except Exception as e:
                            logging.exception(f"summarizer_sensitivity failed: {e}")
                            new_prompt = current_prompt + "\n# (summarizer failed; prompt unchanged)"

                    with open(os.path.join(sensitivity_output_folder, f"ap{improver_iter}_raw.txt"), "w", encoding="utf-8") as f:
                        f.write(new_prompt)
                    current_prompt = clean_prompt(new_prompt)
                    with open(p_output_file_path, "w", encoding="utf-8") as f:
                        f.write(current_prompt)
                    print(f"NEW SENSITIVITY PROMPT SAVED: {p_output_file_path}")
                    logging.info(f"new sensitivity prompt saved: {p_output_file_path}")
                else:
                    print(f"Prompt already exists: {p_output_file_path}")

        #Specificity improver
        if (priority == "specificity" and current_specificity < specificity_threshold) or (priority == "sensitivity" and current_sensitivity > sensitivity_threshold and current_specificity < specificity_threshold):
            print(f"\nENTERING SPECIFICITY IMPROVER for iteration {improver_iter} | Current Specificity: {current_specificity:.4f}")
            logging.info(f"Entering specificity improver (iter {improver_iter}) | Spec={current_specificity:.4f} | FP notes={len(fp_df)}")
            specificity_output_folder = os.path.join(output_folder, f"specificity_iter_{improver_iter}")
            os.makedirs(specificity_output_folder, exist_ok=True)
            spe_fp_output_csv = os.path.join(specificity_output_folder, f"specificity_iter_{improver_iter}_result.csv")
            clean_output_csv = os.path.join(specificity_output_folder, f"specificity_iter_{improver_iter}_result_cleaned.csv")
            p_output_file_path = os.path.join(specificity_output_folder, f"ap{improver_iter}.txt")
            
            if fp_df.empty:
                print("No false-positive rows to use for specificity improver.")
                logging.warning("FP dataframe empty; skipping specificity improver.")
            else:
                if not os.path.exists(spe_fp_output_csv) or os.path.getsize(spe_fp_output_csv) == 0:
                    results = []
                    for idx, row in fp_df.iterrows():
                        text = row.get("Visit", "")
                        empi = row.get("empi", None)

                        if pd.isna(text) or str(text).strip() == "":
                            continue

                        try:
                            evidence = specificity_agent(Backend, text, SOP)
                            results.append({"empi": empi, "evidence": evidence})
                        except Exception as e:
                            logging.exception(f"specificity_agent failed for fp row {idx}: {e}")
                            results.append({"empi": empi, "evidence": None})

                    result_df = pd.DataFrame(results)
                    result_df.to_csv(spe_fp_output_csv, index=False)
                    print(f"FP specificity evidence saved to {spe_fp_output_csv}")

                else:
                    result_df = pd.read_csv(spe_fp_output_csv)
                    print(f"Loaded existing specificity evidence CSV ({len(result_df)} rows)")

                if "evidence" in result_df.columns:
                    result_df["is_meaningful"] = result_df["evidence"].apply(
                        lambda x: is_meaningful_paragraph(x) if pd.notna(x) else False
                    )
                    filtered_df = result_df[result_df["is_meaningful"]].copy()
                    filtered_df.to_csv(clean_output_csv, index=False)
                    print(f"{len(filtered_df)} meaningful evidence paragraphs kept")
                else:
                    filtered_df = pd.DataFrame()
                    print("No evidence column found in specificity evidence CSV.")

                if not os.path.exists(p_output_file_path):
                    if filtered_df.empty:
                        print("No meaningful specificity evidence produced; keeping current prompt.")
                        new_prompt = current_prompt
                    else:
                        try:
                            print(f"Calling summarizer_specificity with {len(filtered_df)} items")
                            print(filtered_df["evidence"])
                            new_prompt = summarizer_specificity(
                                Backend,
                                filtered_df["evidence"].tolist(),
                                current_prompt,
                                SOP
                            )
                            print(f"Summarizer returned {len(new_prompt)} characters")
                        except Exception as e:
                            logging.exception(f"summarizer_specificity failed: {e}")
                            new_prompt = current_prompt + "\n# (specificity summarizer failed; prompt unchanged)"

                    with open(os.path.join(specificity_output_folder, f"ap{improver_iter}_raw.txt"), "w", encoding="utf-8") as f:
                        f.write(new_prompt)

                    current_prompt = clean_prompt(new_prompt)

                    with open(p_output_file_path, "w", encoding="utf-8") as f:
                        f.write(current_prompt)

                    print(f"NEW SPECIFICITY PROMPT SAVED: {p_output_file_path}")
                    logging.info(f"new specificity prompt saved: {p_output_file_path}")
                else:
                    print(f"Prompt already exists: {p_output_file_path}")

        previous_sensitivity = current_sensitivity
        previous_specificity = current_specificity
        iter += 1

        if current_sensitivity >= sensitivity_threshold and current_specificity >= specificity_threshold:
            logging.info("Both thresholds met. Ending early.")
            print("Both thresholds met. Ending early.")
            break

        logging.info(f"End Iteration {iter - 1}. Next iteration will be {iter}.\n")

    logging.info("Workflow completed.")
    print("\nWorkflow completed.")