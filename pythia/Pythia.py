from pythia.core.agentic_workflow import run_agentic_workflow
from pythia.core.validation_workflow import validation_workflow
import os
import glob
import re

def Pythia(LLMbackend, dev_data_path, val_data_path, output_dir, SOP, initial_prompt, iterations = None, sens_threshold = None, spec_threshold = None, priority = None):
    if iterations is None:
        iterations = 5
    if sens_threshold is None:
        sens_threshold = 0.75
    if spec_threshold is None:
        spec_threshold = 0.75
    if priority is None:
        priority = "specificity"
    
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

    # Find all ap*.txt files in output
    pattern = os.path.join(dev_output_base, "*", "*", "ap*.txt")
    files = glob.glob(pattern)
    if files:
        def get_iter(file_path):
            match = re.search(r'ap(\d+)\.txt', file_path)
            return int(match.group(1)) if match else 0
        files.sort(key=get_iter, reverse=True)
        try:
            with open(files[0], "r", encoding="utf-8") as file:
                finalPrompt = file.read()
        except IOError as e:
            print(f"Error reading prompt file {files[0]}: {e}")

    if not finalPrompt:
      print("Could not find final prompt file, using baseprompt...")
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

if __name__ == "__main__":
    print("Starting script")
    # Example usage
    from pythia.llm.pythia_ollama import ollama_backend
    backend = ollama_backend(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=2048
        )
    Pythia(
            LLMbackend=backend,
            dev_data_path='dummy_data/dev',
            val_data_path='dummy_data/val',
            output_dir="retest",
            SOP="Is the patient having chest pains, etc?",
            initial_prompt="Is the patient showing signs of a cardiology concern?",
            iterations=5,
            sens_threshold=0.8,
            spec_threshold=0.8,
            priority="specificity"
        )