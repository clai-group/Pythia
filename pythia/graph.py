"""
pythia/graph.py

Builds and compiles the Pythia LangGraph StateGraph.

LangGraph concepts used here:
  StateGraph  — a graph where every node shares a typed state object
  add_node    — registers a callable as a named node
  add_edge    — unconditional transition between nodes
  add_conditional_edges — route based on a function's return value
  compile     — validates the graph and returns a Runnable
  MemorySaver — checkpoints state after every node (enables backtracking)
"""
from functools import partial
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from pythia.state import PythiaState
from pythia.llm.base import BaseLLMBackend
import pythia.nodes.evaluate as _evaluate
import pythia.nodes.synthesize as _synthesize
import pythia.nodes.record_iteration as _record
import pythia.nodes.analyze_errors as _analyze
import pythia.nodes.controller as _controller

def build_graph(backend: BaseLLMBackend) -> StateGraph:
    """
    Build and compile the Pythia optimisation graph.

    backend is passed in here and bound to nodes that need it via
    functools.partial. This avoids putting the backend in state
    (it's not JSON-serialisable).
    """

    graph = StateGraph(PythiaState)

    #Register nodes, nodes needing llm backend get backend via partial. Non-LLM nodes do not
    graph.add_node('evaluate', partial(_evaluate.run, backend=backend))
    graph.add_node('controller', _controller.run) #no backend, deterministic for Project 1
    graph.add_node('analyze_errors',partial( _analyze.run, backend=backend)) #no backend, deterministic for Project 1
    graph.add_node('synthesize', partial(_synthesize.run, backend=backend))
    graph.add_node('record_iteration', _record.run)

    #Entry point
    graph.add_edge(START, 'evaluate')

    #Create fixed edges
    graph.add_edge('evaluate', 'controller')
    graph.add_edge('record_iteration', 'analyze_errors')
    graph.add_edge('analyze_errors', 'synthesize')
    graph.add_edge('synthesize', 'evaluate')

    #Conditional edges (from controller)
    graph.add_conditional_edges(
        'controller',
        _route,
        {
            'continue': 'record_iteration',
            'halt': END,
            'backtrack': 'analyze_errors',  # State already contains restored prompt
            'reset': 'evaluate',  # Start over with initial prompt
        }
    )

    #compile with checkpointing
    checkpointer = MemorySaver() #In-memory checkpointing for backtracking
    return graph.compile(checkpointer=checkpointer)

def _route(state: PythiaState) -> str:
    """Routing function for the conditional edge after controller."""
    if state.last_decision is None:
        return 'continue'
    return state.last_decision.action
