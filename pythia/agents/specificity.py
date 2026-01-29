from langchain_core.messages import SystemMessage, HumanMessage

def specificity_agent(backend, information, SOP):
    """
    Extract evidence from a FALSE POSITIVE case (where we said YES but should have said NO).
    Goal: Find signals/features that indicate this is NOT actually a positive case.
    
    Args:
        backend: LLM backend
        information: The visit/report text that was a false positive
        SOP: Standard Operating Procedure
    
    Returns:
        String containing extracted evidence of negative/exclusion indicators
    """
    system = SystemMessage(content=f"""You are a clinical expert analyzing patient cases.
Your task is to identify key differentiating factors that distinguish true from false positives.
Follow this Standard Operating Procedure:

{SOP}

Be specific and cite concrete evidence from the text.""")
    
    human = HumanMessage(content=f"""This patient note was INCORRECTLY flagged as positive (false alarm - we said YES but the answer should be NO).

Read this note carefully and extract the KEY EVIDENCE that indicates this is NOT actually a positive case:

MEDICAL NOTE:
{information}

Instructions:
1. Identify features, findings, or context that make this NOT a true positive case
2. Quote exact phrases that show absence of key indicators or presence of exclusion criteria
3. Explain why these findings suggest this is a false positive
4. Identify any confounding factors or ambiguous language that might have caused the misclassification
5. Be concise but thorough

Output only the extracted evidence explaining why this is NOT a positive case.""")
    
    messages = [system, human]
    llm = backend
    response = llm.invoke(messages)
    try:
        answer = response.content
    except:
        answer = response
    return answer

def summarizer_specificity(backend, evidence_list, prompt, sop, rejected_prompts=None):
    """
    Improve the prompt based on false positive evidence.
    Goal: Make the prompt more SPECIFIC (reduce false alarms).
    
    Args:
        backend: LLM backend
        evidence_list: List of evidence strings from false positives
        prompt: Current prompt
        sop: Standard Operating Procedure
        rejected_prompts: List of prompts that performed worse (optional)
    
    Returns:
        Improved prompt that is more specific (fewer false positives)
    """
    evidence_text = "\n---\n".join(evidence_list)
    
    rejected_context = ""
    if rejected_prompts and len(rejected_prompts) > 0:
        rejected_text = "\n---\n".join(rejected_prompts)
        rejected_context = f"\n\nPREVIOUSLY REJECTED PROMPTS (which performed worse):\n{rejected_text}\n\nLearn from these failures - improve upon them but avoid the same mistakes."
    
    system = SystemMessage(content="""You are an expert in clinical decision support and prompt engineering.
Your task is to improve a screening prompt based on evidence from false positive cases.

The goal is to increase SPECIFICITY (reduce false positives/false alarms) while maintaining sensitivity.
Be specific and actionable in your improvements. Make sure sensitivity does not drop.""")
    
    human = HumanMessage(content=f"""We have a screening prompt that is generating FALSE POSITIVES (high false alarm rate).

Here is evidence extracted from cases we INCORRECTLY FLAGGED (false positives):

EVIDENCE FROM FALSE POSITIVE CASES:
{evidence_text}

Current prompt:
{prompt}{rejected_context}

Standard Operating Procedure:
{sop}

Instructions:
1. Analyze the evidence to identify patterns in what's triggering false alarms
2. Identify which criteria in the current prompt are too broad/sensitive
3. Identify features that distinguish true positives from false positives in the evidence
4. Add exclusion criteria or negative indicators to the prompt
5. Make the prompt more restrictive/specific to reduce false alarms
6. Add clarity about required conditions and exclude borderline/ambiguous cases
7. Do NOT sacrifice too much sensitivity - try to maintain a balance
8. If rejected prompts are shown, avoid making the same mistakes they did

Output ONLY the improved prompt, with no explanations or commentary.""")
    
    messages = [system, human]
    llm = backend
    response = llm.invoke(messages)
    try:
        answer = response.content
    except:
        answer = response
    return answer