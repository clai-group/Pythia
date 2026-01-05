from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Unified interface for all LLM backends.
    Every backend must implement .invoke(prompt).
    """

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Execute the model and return the generated text.
        """
        pass