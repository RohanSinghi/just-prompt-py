"""
Shared data types used by Just-Prompt
"""
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Provider(str, Enum):
    """LLM provider enumeration"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    
    @classmethod
    def from_prefix(cls, prefix: str) -> "Provider":
        """Get provider from prefix"""
        prefix_map = {
            "o": cls.OPENAI,
            "openai": cls.OPENAI,
            "a": cls.ANTHROPIC,
            "anthropic": cls.ANTHROPIC,
            "g": cls.GEMINI,
            "gemini": cls.GEMINI,
            "q": cls.GROQ,
            "groq": cls.GROQ,
            "d": cls.DEEPSEEK,
            "deepseek": cls.DEEPSEEK,
            "l": cls.OLLAMA,
            "ollama": cls.OLLAMA,
        }
        
        if prefix.lower() not in prefix_map:
            raise ValueError(f"Unknown provider prefix: {prefix}")
        
        return prefix_map[prefix.lower()]


class PromptRequest(BaseModel):
    """Prompt request data model"""
    prompt: str
    models: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True


class PromptResponse(BaseModel):
    """Prompt response data model"""
    model: str
    content: str
    tokens: Optional[int] = None
    
    class Config:
        populate_by_name = True 