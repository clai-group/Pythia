# llm/gemini_backend.py
from google import genai
from google.genai import types
from .base import BaseLLM

class GeminiBackend(BaseLLM):
    def __init__(self, model="gemini-1.5-pro", api_key=None, temperature: float = 0.0, max_tokens: int = 2048):
        self.model = model
        self.client = genai.Client(api_key=api_key)
        # default temperature to use when invoke() is called without an explicit temperature
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt, **kwargs) -> str:
        """
        Sends the prompt to Google Gemini.
        Accepts either a string prompt or LangChain message objects.
        """
        #import logging
        #logger = logging.getLogger(__name__)
        
        # Convert LangChain messages to string format
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
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt_text,
                #generation_config=types.GenerationConfig(
                #    temperature=self.temperature,
               #     max_output_tokens=self.max_tokens
              #  )
            )
            
            result_text = response.text if response and hasattr(response, 'text') else ""
            print(f"[Gemini Response] {result_text}")
            #logger.debug(f"Gemini API returned: {len(result_text)} characters")
            return result_text
            
        except Exception as e:
            #logger.exception(f"Gemini API call failed: {type(e).__name__}: {e}")
            print(f"[ERROR] Gemini API failed: {e}")
            raise
