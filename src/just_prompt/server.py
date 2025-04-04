"""
Just-Prompt MCP server implementation
"""
import json
import os
import sys
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from just_prompt.atoms.shared.data_types import PromptRequest, PromptResponse

app = FastAPI(title="Just-Prompt", description="MCP server with unified interface for LLM providers")


@app.get("/")
async def root():
    """Root route, returns service information"""
    return {"status": "ok", "service": "Just-Prompt MCP Server"}


@app.get("/providers")
async def list_providers():
    """List all available providers"""
    # Provider detection logic will be added here
    return {"providers": ["To be implemented"]}


@app.get("/models")
async def list_models(provider: Optional[str] = None):
    """List all available models"""
    # Model listing logic will be added here
    return {"models": ["To be implemented"]}


@app.post("/prompt", response_model=List[PromptResponse])
async def prompt(request: PromptRequest):
    """Process prompt request"""
    # Prompt processing logic will be added here
    return [
        PromptResponse(
            model="example-model",
            content="This is an example response. Actual implementation will connect to LLM provider APIs.",
            tokens=0
        )
    ]


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port) 