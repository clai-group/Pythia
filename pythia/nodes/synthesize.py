"""pythia/nodes/synthesize.py"""
from pythia.state import PythiaState
from pythia.agents.summarizer import SummarizerAgent

def run(state: PythiaState, backend) -> dict:
    if state.current_critiques is None:
        return {}
    
    #provide previous prompt as failed if we backtracked
    failed_prompt = None
    if (len(state.iteration_history) >= 2) and (state.iteration_history[-1].dev_metrics.f1 > state.current_dev_metrics.f1):
        failed_prompt = state.current_prompt
    agent = SummarizerAgent(backend)
    new_prompt = agent.synthesise(
        sop = state.sop,
        current_prompt = state.current_prompt,
        critiques = state.current_critiques,
        failed_prompt = failed_prompt,
    )
    return {'current_prompt': new_prompt}