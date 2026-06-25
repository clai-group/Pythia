"""
pythia/llm/openai_compat.py

OpenAI-compatible backend. Works with:
  - llama.cpp HTTP server  (./server -m model.gguf --port 8080)
  - LM Studio              (http://localhost:1234/v1)
  - vLLM                   (http://localhost:8000/v1)
  - Groq                   (https://api.groq.com/openai/v1)
  - Any server implementing /v1/chat/completions
"""
import requests
from pythia.llm.base import BaseLLMBackend


class OpenAICompatBackend(BaseLLMBackend):
    def __init__(
        self,
        model:       str   = 'llama3.1',
        base_url:    str   = 'http://localhost:8080/v1',
        api_key:     str   = 'not-needed',   # many local servers ignore this
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
    ):
        self.model       = model
        self.base_url    = base_url.rstrip('/')
        self.api_key     = api_key
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def invoke(self, prompt: str) -> str:
        url = f'{self.base_url}/chat/completions'
        headers = {
            'Content-Type':  'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model':       self.model,
            'messages':    [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
            'max_tokens':  self.max_tokens,
        }
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
