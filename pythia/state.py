"""
pythia/state.py

All shared state types for the Pythia LangGraph system.
Use Pydantic BaseModel (not TypedDict) so that field validation
runs automatically and JSON serialisation is free.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

#Iteration level information
class Metrics(BaseModel):
    """Metrics for a single Pythia iteration."""
    sensitivity: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

class IterationSnapshot(BaseModel):
    """Snapshot of the state of a single Pythia iteration."""
    iteration: int
    prompt: str
    tested_prompt: str = ''
    improved_prompt: str = ''
    dev_metrics: Metrics
    val_metrics: Optional[Metrics] = None #filled during val pass
    prompt_embedding: list[float] = Field(default_factory=list)
    output_entropy: float = 0.0
    critique_summary: str = ''
    controller_action: str = ''

class TransitionSnapshot(BaseModel):
    """Snapshot of a prompt transition event (backtrack/reset)."""
    iteration: int
    unstable_prompt: str
    unstable_dev_metrics: Metrics
    prompt_embedding: list[float] = Field(default_factory=list)
    output_entropy: float = 0.0
    critique_summary: str = ''
    action: Literal['backtrack', 'reset']
    restored_prompt: str
    reason: str

#Controller output
class ControllerDecision(BaseModel):
    "Represents decision made by Pythia contoller."
    action: Literal['continue', 'halt', 'backtrack', 'reset']
    reason: str
    target_iter: Optional[int] = None #for backtrack and reset, which iteration to go back to

#Main Graph
class PythiaState(BaseModel):
    #config: set up once during startup, never changes.
    initial_prompt: str
    sop: str
    max_iterations: int = 7
    sens_threshold: float = 0.75
    spec_threshold: float = 0.75
    priority: Literal['sensitivity', 'specificity'] = 'sensitivity'
    output_dir: str = 'outputs/'

    #dataset: Loaded at startup, carried thru graph. Stored as list of dicts so LangGraph can serialize.
    #Use data.py helpers to convert to DataFrames
    dev_records: list[dict] = Field(default_factory=list)
    dev_labels:   list[int]  = Field(default_factory=list)
    val_records:  list[dict] = Field(default_factory=list)
    val_labels:   list[int]  = Field(default_factory=list)
    
    #Active State (updates each iteration)
    current_prompt: str = ''
    current_tested_prompt: str = ''
    current_iteration: int = 0
    current_critiques: list[str] = Field(default_factory=list)
    current_predictions: list[int] = Field(default_factory=list)
    optimization_target: Literal['sensitivity', 'specificity', 'converged'] = 'sensitivity'
    current_dev_metrics: Optional[Metrics] = None
    current_output_entropy: float = 0.0

    #history
    iteration_history: list[IterationSnapshot] = Field(default_factory=list)
    transition_history: list[TransitionSnapshot] = Field(default_factory=list)
    last_decision: Optional[ControllerDecision] = None

    #Terminal output
    selected_prompt: Optional[str] = None
    selected_metrics: Optional[Metrics] = None

    #Pydantic config
    model_config = {'arbitrary_types_allowed': True}
    @field_validator('sens_threshold', 'spec_threshold')
    @classmethod
    def validate_thresholds(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Thresholds must be between 0 and 1')
        return v