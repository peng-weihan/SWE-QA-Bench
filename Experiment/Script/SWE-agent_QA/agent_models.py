"""Simplified model interface, based on SWE-agent's model system"""
from __future__ import annotations

import os
import litellm
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, Field, SecretStr

litellm.suppress_debug_info = True


class ModelConfig(BaseModel):
    """Model configuration"""
    name: str = Field(description="Model name, e.g., gpt-4o, claude-sonnet-4-20250514")
    api_key: SecretStr | None = Field(default=None, description="API key")
    api_base: str | None = Field(default=None, description="API base URL")
    temperature: float = 0.0
    max_tokens: int | None = None


class AbstractModel(ABC):
    """Abstract model interface"""
    
    @abstractmethod
    def forward(self, history: list[dict[str, str]]) -> tuple[str, Dict[str, Any]]:
        """Call model
        
        Args:
            history: Conversation history, format: [{"role": "user", "content": "..."}, ...]
            
        Returns:
            (response_text, stats): Response text and statistics (latency, input_tokens, output_tokens)
        """
        pass


class LiteLLMModel(AbstractModel):
    """LiteLLM-based model implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.name
        
        # Set API key
        if config.api_key:
            # Set corresponding environment variables based on model name
            if "gpt" in config.name.lower() or "openai" in config.name.lower():
                os.environ["OPENAI_API_KEY"] = config.api_key.get_secret_value()
            elif "claude" in config.name.lower() or "anthropic" in config.name.lower():
                os.environ["ANTHROPIC_API_KEY"] = config.api_key.get_secret_value()
    
    def forward(self, history: list[dict[str, str]]) -> tuple[str, Dict[str, Any]]:
        """Call model
        
        Returns:
            (response_text, stats): Response text and statistics
        """
        import time
        start_time = time.time()
        
        messages = []
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        
        response = litellm.completion(**kwargs)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Extract token statistics
        usage = getattr(response, "usage", None)
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        stats = {
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        
        return response.choices[0].message.content, stats


def get_model(config: ModelConfig) -> AbstractModel:
    """Get model instance based on configuration"""
    return LiteLLMModel(config)

