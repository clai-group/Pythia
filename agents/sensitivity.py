import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def sensitivity_agent(backend, information, SOP):
    """
    Extract evidence from a FALSE NEGATIVE case (where we should have said YES but said NO).
    Goal: Find signals/features that clearly indicate a positive case.
    
    Args:
        backend: LLM backend
        information: The visit/report text that was a false negative
        SOP: Standard Operating Procedure
    
    Returns:
        String containing extracted evidence of positive signals
    """
    system = SystemMessage(content=f"""You are a clinical expert analyzing patient cases.
Your task is to identify diagnostic evidence and clinical signals in medical notes.
Follow this Standard Operating Procedure:

{SOP}

Be specific and cite concrete evidence from the text.""")
    
    human = HumanMessage(content=f"""This patient note WAS MISSED by our screening process (we should have detected a positive case but didn't).

Read this note carefully and extract the KEY CLINICAL EVIDENCE that clearly indicates this IS a positive case:

MEDICAL NOTE:
{information}

Instructions:
1. Identify specific symptoms, findings, or clinical indicators that support a positive diagnosis
2. Quote exact phrases from the note that are diagnostic
3. Explain why these signals are important
4. Be concise but thorough

Output only the extracted evidence, organized by symptom/finding type.""")
    
    messages = [system, human]
    llm = backend
    response = llm.invoke(messages)
    try:
        answer = response.content
    except:
        answer = response
    return answer

def summarizer_sensitivity(backend, evidence_list, prompt, sop):
    """
    Improve the prompt based on false negative evidence.
    Goal: Make the prompt more SENSITIVE (catch more positives).
    
    Args:
        backend: LLM backend
        evidence_list: List of evidence strings from false negatives
        prompt: Current prompt
        sop: Standard Operating Procedure
    
    Returns:
        Improved prompt that is more sensitive to positive cases
    """
    evidence_text = "\n---\n".join(evidence_list)
    
    system = SystemMessage(content="""You are an expert in clinical decision support and prompt engineering.
Your task is to improve a screening prompt based on evidence from cases we missed.

The goal is to increase SENSITIVITY (catch more true positives) while maintaining specificity.
Be specific and actionable in your improvements.""")
    
    human = HumanMessage(content=f"""We have a screening prompt that is MISSING TRUE POSITIVE CASES (low sensitivity).

Here is evidence extracted from cases we MISSED (false negatives):

EVIDENCE FROM MISSED CASES:
{evidence_text}

Current prompt:
{prompt}

Standard Operating Procedure:
{sop}

Instructions:
1. Analyze the evidence to identify patterns in what we're missing
2. Identify which symptoms/findings in the evidence are NOT well-covered in the current prompt
3. Enhance the prompt to better detect these signals
4. Add specific keywords and clinical indicators from the evidence
5. Make the prompt more inclusive of relevant positive indicators
6. Do NOT make the prompt overly broad - stay clinically relevant

Output ONLY the improved prompt, with no explanations or commentary.""")
    
    messages = [system, human]
    llm = backend
    response = llm.invoke(messages)
    try:
        answer = response.content
    except:
        answer = response
    return answer

