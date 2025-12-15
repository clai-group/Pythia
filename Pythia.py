from core.agentic_workflow import run_agentic_workflow
from core.validation_workflow import validation_workflow
from core.split import split_csv_folder
import os

def Pythia(LLMbackend, dev_data_path, val_data_path, output_dir, SOP, initial_prompt, iterations = None, sens_threshold = None, spec_threshold = None, priority = None):
    if iterations is None:
        iterations = 5
    if sens_threshold is None:
        sens_threshold = 0.75
    if spec_threshold is None:
        spec_threshold = 0.75
    if priority is None:
        priority = "specificity"
    print("Beginning prompt development...")
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
    print("Completed development and evaluation on Development Data...")
    
    final_iter = iterations
    dev_output_base = os.path.join(
    output_dir,
    f"output_{os.path.basename(os.path.normpath(dev_data_path))}"
    )

    finalPrompt = None

    for iter in range(final_iter, 0, -1):
      iter_dir = os.path.join(dev_output_base, f"iter_{iter}_{LLMbackend.__class__.__name__}")

      for folder in [f"sensitivity_iter_{iter}", f"specificity_iter_{iter}"]:
        prompt_file = os.path.join(iter_dir, folder, f"ap{iter}.txt")

        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as file:
                finalPrompt = file.read()
            break

      if finalPrompt:
        break

    if not finalPrompt:
      print("Could not find final prompt file, using baseprompt...")
      finalPrompt = initial_prompt
      
    print("Beginning evaluation on Validation Data...")
    validation_workflow(
    Backend = LLMbackend,
    input_data_path = val_data_path,
    SOP = SOP,
    BasePrompt = finalPrompt,
    output_path = output_dir,
    ) 
    print("Completed evaluation on Validation Data.")