from pythia.core.agentic_workflow import run_agentic_workflow
from pythia.core.validation_workflow import validation_workflow
import os
import glob
import re
import pandas as pd

def Pythia(LLMbackend, dev_data_path, val_data_path, output_dir, SOP, initial_prompt, iterations = None, sens_threshold = None, spec_threshold = None, priority = None):
    if iterations is None:
        iterations = 5
    if sens_threshold is None:
        sens_threshold = 0.75
    if spec_threshold is None:
        spec_threshold = 0.75
    if priority is None:
        priority = "sensitivity"
    
    if not hasattr(LLMbackend, 'invoke'):
        raise ValueError("LLMbackend must have an 'invoke' method.")
    
    if not os.path.isdir(dev_data_path):
        raise FileNotFoundError(f"Development data path '{dev_data_path}' does not exist or is not a directory.")
    
    if not os.path.isdir(val_data_path):
        raise FileNotFoundError(f"Validation data path '{val_data_path}' does not exist or is not a directory.")
    
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer.")
    
    if not isinstance(sens_threshold, (int, float)) or not (0 <= sens_threshold <= 1):
        raise ValueError("sens_threshold must be a number between 0 and 1.")
    
    if not isinstance(spec_threshold, (int, float)) or not (0 <= spec_threshold <= 1):
        raise ValueError("spec_threshold must be a number between 0 and 1.")
    
    if priority.lower() not in ["sensitivity", "specificity"]:
        raise ValueError("priority must be 'sensitivity' or 'specificity'.")
    
    if isinstance(initial_prompt, str) and os.path.isfile(initial_prompt) and not os.path.exists(initial_prompt):
        raise FileNotFoundError(f"Initial prompt file '{initial_prompt}' does not exist.")
    
    if not isinstance(SOP, str) or not SOP.strip():
        raise ValueError("SOP must be a non-empty string.")
    
    print("Beginning prompt development...")
    try:
        run_agentic_workflow(
        Backend = LLMbackend,
        input_data_path = dev_data_path,
        SOP = SOP,
        BasePrompt = initial_prompt,
        output_path = output_dir,
        Iterations = iterations,
        sensitivity_threshold = sens_threshold,
        specificity_threshold = spec_threshold,
        priority = priority,
        ) 
    except Exception as e:
        print(f"Error during development workflow: {e}")
        raise
    
    print("Completed development and evaluation on Development Data...")
    
    dev_output_base = os.path.join(
    output_dir,
    f"output_{os.path.basename(os.path.normpath(dev_data_path))}"
    )

    finalPrompt = None
    best_metrics = {"iteration": None, "metrics_file": None, "sensitivity": 0.0, "specificity": 0.0, "f1_score": 0.0}

    # Find all metrics_results.csv files and select the iteration with best metrics
    metrics_pattern = os.path.join(dev_output_base, "*", "evaluation_*", "metrics_results.csv")
    metrics_files = glob.glob(metrics_pattern)
    print(f"DEBUG: Looking for metrics in pattern: {metrics_pattern}")
    print(f"DEBUG: Found {len(metrics_files)} metrics files")
    
    if metrics_files:
        for metrics_file in metrics_files:
            try:
                df_metrics = pd.read_csv(metrics_file)
                if df_metrics.empty:
                    continue
                
                # Get metrics from the first row (or you can aggregate if multiple rows)
                sensitivity = float(df_metrics.iloc[0].get("Sensitivity", 0.0))
                specificity = float(df_metrics.iloc[0].get("Specificity", 0.0))
                f1_score = float(df_metrics.iloc[0].get("F1 Score", 0.0))
                
                # Extract iteration number from file path
                match = re.search(r'iter_(\d+)', metrics_file)
                iteration = int(match.group(1)) if match else 0
                
                # Select based on F1 score
                if f1_score > best_metrics["f1_score"] or best_metrics["iteration"] is None:
                    best_metrics["iteration"] = iteration
                    best_metrics["metrics_file"] = metrics_file
                    best_metrics["sensitivity"] = sensitivity
                    best_metrics["specificity"] = specificity
                    best_metrics["f1_score"] = f1_score
                    
            except Exception as e:
                print(f"Error reading metrics from {metrics_file}: {e}")
                continue
    
    # If best metrics found, load the corresponding prompt
    if best_metrics["iteration"] is not None:
        print(f"Best performing iteration: {best_metrics['iteration']} "
              f"(F1 Score: {best_metrics['f1_score']:.4f}, Sensitivity: {best_metrics['sensitivity']:.4f}, Specificity: {best_metrics['specificity']:.4f})")
        
        best_iter = best_metrics['iteration']
        
        # Try to find prompt starting from best iteration and going backwards
        for iter_to_check in range(best_iter, 0, -1):
            prompt_pattern = os.path.join(dev_output_base, f"iter_{iter_to_check}_*", "*", f"ap{iter_to_check}.txt")
            prompt_files = glob.glob(prompt_pattern)
            
            if prompt_files:
                try:
                    with open(prompt_files[0], "r", encoding="utf-8") as file:
                        finalPrompt = file.read()
                    if iter_to_check == best_iter:
                        print(f"Loaded prompt from best-performing iteration {best_iter}")
                    else:
                        print(f"Iteration {best_iter} had no prompt; loaded from earlier iteration {iter_to_check}")
                    break
                except IOError as e:
                    print(f"Error reading prompt file {prompt_files[0]}: {e}")
                    continue
    
    if not finalPrompt:
      print("Could not find final prompt in any iteration, using base prompt...")
      finalPrompt = initial_prompt
      
    print("Beginning evaluation on Validation Data...")
    try:
        validation_workflow(
        Backend = LLMbackend,
        input_data_path = val_data_path,
        SOP = SOP,
        BasePrompt = finalPrompt,
        output_path = output_dir,
        ) 
    except Exception as e:
        print(f"Error during validation workflow: {e}")
        raise
    print("Completed evaluation on Validation Data.")   
