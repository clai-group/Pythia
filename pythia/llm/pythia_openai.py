from .base import BaseLLM
from openai import OpenAI

class OpenAILLM(BaseLLM):
    """
    OpenAI API implementation of the BaseLLM interface.
    Supports both OpenAI API and local OpenAI-compatible servers.
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "gpt-4o",
        base_url: str = None,
        **default_kwargs
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
                     For local servers, can be any string (e.g., "local")
            model: Model to use (default: gpt-4o)
            base_url: Base URL for API (e.g., "http://localhost:8000/v1" for local)
                      If None, uses OpenAI's default endpoint
            **default_kwargs: Default parameters for all API calls
        """
        client_kwargs = {}
        
        if base_url:
            client_kwargs["base_url"] = base_url
            # For local servers, use dummy key if none provided
            if api_key is None:
                api_key = "local"
        
        if api_key:
            client_kwargs["api_key"] = api_key
            
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.default_kwargs = default_kwargs
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Execute the model and return the generated text.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            The generated text response
        """
        # Merge default kwargs with call-specific kwargs
        params = {**self.default_kwargs, **kwargs}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        
        return response.choices[0].message.content