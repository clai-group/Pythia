from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolCall
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from openai import AsyncOpenAI
import json
import asyncio
from typing import List, Any, Optional
#from utils.logging_config import setup_logger
import asyncio
import httpx 

#logger = setup_logger()

class LocalLLM(BaseChatModel):
    """Custom LangChain chat model for a local LLM using OpenAI client."""
    client: AsyncOpenAI
    model: str = "local-model"
    temperature: float = 0.7
    max_tokens: int = 2048
    tools: List[BaseTool] = None  # ? Use BaseTool

    def __init__(self, base_url: str = "http://localhost:11484/v1", **kwargs: Any):
        client = AsyncOpenAI(base_url=base_url, api_key="not-needed", timeout=httpx.Timeout(900))
        super().__init__(client=client, **kwargs)
        self.client = client
    
    async def _agenerate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        formatted_messages = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages
        ]

        api_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
        }

        # === PASS TOOLS TO THE LLM ===
        if self.tools:
            api_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.args_schema.schema()  # ? FIXED: .schema() ? dict
                    }
                }
                for tool in self.tools
            ]
            api_kwargs["tool_choice"] = "auto"

        try:
            response = await self.client.chat.completions.create(**api_kwargs)
            choice = response.choices[0]
            message = choice.message

            # === PARSE TOOL CALLS ===
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            name=tc.function.name,
                            args=args,
                            id=tc.id
                        )
                    )

            # === BUILD AIMessage ===
            ai_message = AIMessage(
                content=message.content or "",
                tool_calls=tool_calls  # ? Always list, even if empty
            )

            generation = ChatGeneration(
                message=ai_message,
                generation_info={"finish_reason": choice.finish_reason}
            )

            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"Error in _agenerate: {e}")
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=f"Error: {e}")
            )])
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Use async version in running loop.")
            return loop.run_until_complete(self._agenerate(messages, stop, run_manager, **kwargs))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._agenerate(messages, stop, run_manager, **kwargs))
            finally:
                loop.close()
    
    async def aclose(self):
        await self.client.aclose()
    
    def bind_tools(self, tools: List[BaseTool], **kwargs: Any) -> "LocalLLM":
        return LocalLLM(
            base_url=self.client.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
            **kwargs
        )
        
    @property
    def _llm_type(self) -> str:
        return "local-llm"