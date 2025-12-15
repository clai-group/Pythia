import requests
from .base import BaseLLM
from ollama import chat, ChatResponse, Client

class ollama_backend(BaseLLM):
    def __init__(self, model="llama3.2",  base_url = "http://localhost:11434", temperature: float = 0.0, max_tokens: int = 2048):
        self.model = model
        self.base_url = base_url
        # default temperature to use when invoke() is called without an explicit temperature
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Client(host=base_url, headers={'x-some-header': 'some-value'})

    def invoke(self, prompt, **kwargs) -> str:
        if isinstance(prompt, list):
            # Assume it's a list of LangChain message objects
            prompt_text = ""
            for message in prompt:
                # Extract content from LangChain message objects
                if hasattr(message, 'content'):
                    content = message.content
                else:
                    content = str(message)
                prompt_text += f"{content}\n"
            prompt_text = prompt_text.strip()
        else:
            # Assume it's already a string
            prompt_text = prompt

        try:
            response = self.client.generate(
                model=self.model,  prompt = prompt_text, options = {"temperature": self.temperature, "num_predict": self.max_tokens}#temperature=self.temperature, max_tokens=self.max_tokens
            )
            #print(response)
            return response.response #.message.content #['message']['content']
        except Exception as e:
            print(f"[ERROR] Ollama API failed: {e}")
            raise