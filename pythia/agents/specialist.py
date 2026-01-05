from langchain_core.messages import HumanMessage, SystemMessage


def evaluate_cognitive_concerns (backend, prompt, information):
    
    system= SystemMessage(content="You are a specialized medical agent. Answer the question with just a yes or no.")
    human= HumanMessage(content=f"{prompt}\n\n{information}")
    messages = [system, human]
    
    llm = backend 
    response = llm.invoke(messages)
    
    # Handle both string responses and response objects with .content attribute
    if isinstance(response, str):
        answer = response
    elif hasattr(response, 'content'):
        answer = response.content
    else:
        answer = str(response)
    
    return answer


