"""
pythia/llm/ollama_backend.py

Calls Ollama's local HTTP API directly (no LangChain dependency).
Start Ollama with: ollama serve
Pull a model with: ollama pull llama3.1
"""
import json
import requests
from pydantic import BaseModel
from pythia.llm.base import BaseLLMBackend


class OllamaBackend(BaseLLMBackend):
    def __init__(
        self,
        model:       str   = 'llama3.1',
        base_url:    str   = 'http://localhost:11434',
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
    ):
        self.model       = model
        self.base_url    = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def invoke(self, prompt: str) -> str:
        url = f'{self.base_url}/api/generate'
        payload = {
            'model':  self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
            }
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['response'].strip()
