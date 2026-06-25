"""pythia/llm/base.py"""
from abc import ABC, abstractmethod


class BaseLLMBackend(ABC):
    """
    All Pythia LLM backends implement this interface.
    invoke() takes a full prompt string and returns the model's
    text response as a string.
    """

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        ...

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
