## Name:
Pythia

## Description:
Pythia is an iterative tool for allowing your LLM to improve upon it's own prompts. Instead of manually changing and perfecting your own prompt, Pythia can do it for you, by testing on a dataset you provide and determining where it needs to improve.

## Usage:
The main Pythia tool can be accessed through the Pythia() function

Pythia works by first calling the "specialist" agent to test your prompt, and evaluates the performance metrics compared to the provided ground truth answer. It will then route to the corresponding improvement agent (sensitivity, or specificity) based on your results and priority. For each false positive, or false negative individual, it will process the mistaken notes to figure out where the LLM went wrong, and provide evidence of the true classification of the note for the summarizer to build it's new prompt on. From there, the summarizer will be passed the evidence, as well as the original prompt and SOP to build the new prompt. Once a new prompt is created, it will repeat the process until it reaches the maximum iterations or meets the performance thresholds. From there, it will validate the new prompt on the development dataset.

A few things to note: Pythia is API call heavy, if you are worried about token costs, computation, or billing, you should consider running on a lower number of iterations, or with a smaller dataset.

## Parameters:
### LLMBackend
#### Type: 
String
#### Description:
Where a user would input their own backend, currently Pythia comes equipped with a backend for Gemini and Ollama, though it could be anything so long as the LLM is called with a .invoke() method.
#### Required:
True

### dev_data_path: 
#### Type:
Path
#### Description:
The filepath that leads to your development dataset, which is whatever dataset you would like to do your main development on. The best format for this is CSV's, where each CSV represents a single "person", and each row represents an individual visit/note. The main note text for this dataset should be in a column labled "visit", and the ground truth should be in a column labeled "Ground Truth".
#### Required:
True

### val_data_path: 
#### Type:
Path
#### Description:
The filepath that leads to your validation dataset, which is what the final "complete" prompt will be tested on to ensure it works.
#### Required:
True

### output_dir: 
#### Type:
Path
#### Description:
The path to the directory that you would like the results to be saved to, if it doesn't exist, it will be generated.
#### Required: 
True

### SOP:
#### Type:
String
#### Description: 
A string that is the standard operating procedure for the LLM as it tries to answer your prompt.
#### Required:
True

### initial_prompt: 
#### Type:
String
#### Description:
Your original prompt as a string or a filepath leading to a .txt containing your original prompt, this will be the foundation of the improvements.
#### Required:
True

### Iterations:
#### Type:
Int
#### Description:
An integer that controls how many times Pythia will iterate through the improvement cycle. If no input is provided, it will default to 5.
#### Required:
False


### sens_threshold: 
#### Type:
Float
#### Description: 
A float between 0 and 1, representing the threshold for when Pythia will be "satisfied" with the sensitivity score, defaults to 0.75.
#### Required:
False

### spec_threshold: 
#### Type:
Float
#### Description: 
A float between 0 and 1, representing the threshold for when Pythia will be "satisfied" with the specificity score, defaults to 0.75.
#### Required:
False

### priority:
#### Type:
 String
#### Description:
Either "specificity" or "sensitivity", representing which of the two metrics Pythia will prioritize improving. Defaults to specificity.
#### Required:
False

## Output
The output directory is organized to store results from both the development and validation phases of the Pythia workflow. For the development phase, a subdirectory named `output_{basename of dev_data_path}` (such as `output_dev` if the development data path ends with 'dev') is created within the specified output directory. Inside this subdirectory, results for each iteration are stored in folders labeled `iter_{iteration_number}_{BackendClassName}` (such as `iter_1_OllamaBackend`). Each iteration folder contains subfolders for whichever metric they improve, such as `sensitivity_iter_1` and `specificity_iter_1`, where the refined prompt for that iteration is saved as `ap{iteration}.txt`. A workflow log file summarizing the process is also generated in the base output subdirectory. For the validation phase, a separate subdirectory named `validation_output_{basename of val_data_path}` (e.g., `validation_output_val`) is created, containing the final evaluation metrics and outputs using the optimized prompt.


## Example usage
### Clinical Example
``` from pythia import Pythia
    from pythia.llm import ollama_backend
    backend = ollama_backend(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=2048
        )
    Pythia(
            LLMbackend=backend,
            dev_data_path='/path/to/dev/dataset/',
            val_data_path='/path/to/val/dataset/',
            output_dir="/path/to/output/",
            SOP="Look for if the patient having chest pains, shortness of breath, etc?",
            initial_prompt="Is the patient showing signs of a cardiology concern?"
        )
```
### General Example, pre split
``` from pythia import Pythia
    from pythia.llm import ollama_backend
    backend = ollama_backend(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=2048
        )
    
    Pythia(
            LLMbackend=backend,
            dev_data_path='/path/to/dev/dataset/',
            val_data_path='/path/to/val/dataset/',
            output_dir="/path/to/output/",
            SOP="Scan client notes for indicators such as explicit inquiries about property listings, mortgage options, or neighborhoods",
            initial_prompt="Does the client seem like they're going to buy a home?",
            iterations=3,
            sens_threshold=0.7,
            spec_threshold=0.7,
            priority="sensitivity"
        )
```