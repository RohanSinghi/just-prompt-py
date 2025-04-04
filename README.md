# Just-Prompt

Just-Prompt is an MCP (Multi-Model Chat Platform) server that provides a unified interface to multiple top-tier LLM providers (OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, and Ollama).

## Features

* Unified API for multiple LLM providers
* Support for text prompts from strings or files
* Run multiple models in parallel
* Automatic model name correction using the first model in the `--default-models` list
* Ability to save responses to files
* Easy listing of available providers and models

## Provider Prefixes

> Every model must be prefixed with the provider name
> 
> Use the short name for faster referencing

* `o` or `openai`: OpenAI  
   * `o:gpt-4o-mini`  
   * `openai:gpt-4o-mini`
* `a` or `anthropic`: Anthropic  
   * `a:claude-3-5-haiku`  
   * `anthropic:claude-3-5-haiku`
* `g` or `gemini`: Google Gemini  
   * `g:gemini-2.5-pro-exp-03-25`  
   * `gemini:gemini-2.5-pro-exp-03-25`
* `q` or `groq`: Groq  
   * `q:llama-3.1-70b-versatile`  
   * `groq:llama-3.1-70b-versatile`
* `d` or `deepseek`: DeepSeek  
   * `d:deepseek-coder`  
   * `deepseek:deepseek-coder`
* `l` or `ollama`: Ollama  
   * `l:llama3.1`  
   * `ollama:llama3.1`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/just-prompt.git
cd just-prompt

# Install with pip
pip install -e .
# Or install with uv
uv sync
```

### Environment Variables

Create a `.env` file with your API keys (you can copy the `.env.sample` file):

```bash
cp .env.sample .env
```

Then edit the `.env` file to add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OLLAMA_HOST=http://localhost:11434
```

## Claude Thinking Tokens

The Anthropic Claude model `claude-3-7-sonnet-20250219` supports extended thinking capabilities using thinking tokens. This allows Claude to perform more thorough thought processes before answering.

You can enable thinking tokens by adding a suffix to the model name in this format:

* `anthropic:claude-3-7-sonnet-20250219:1k` - Use 1024 thinking tokens
* `anthropic:claude-3-7-sonnet-20250219:4k` - Use 4096 thinking tokens
* `anthropic:claude-3-7-sonnet-20250219:8000` - Use 8000 thinking tokens

## Running Tests

```bash
pytest
```

## Project Structure

```
.
├── ai_docs/                   # Documentation for AI model details
│   ├── llm_providers_details.xml
│   └── pocket-pick-mcp-server-example.xml
├── list_models.py             # Script to list available LLM models
├── pyproject.toml             # Python project configuration
├── specs/                     # Project specifications
│   └── init-just-prompt.md
├── src/                       # Source code directory
│   └── just_prompt/
│       ├── __init__.py
│       ├── __main__.py
│       ├── atoms/             # Core components
│       │   ├── llm_providers/ # Individual provider implementations
│       │   │   ├── anthropic.py
│       │   │   ├── deepseek.py
│       │   │   ├── gemini.py
│       │   │   ├── groq.py
│       │   │   ├── ollama.py
│       │   │   └── openai.py
│       │   └── shared/        # Shared utilities and data types
│       │       ├── data_types.py
│       │       ├── model_router.py
│       │       ├── utils.py
│       │       └── validator.py
│       ├── molecules/         # Higher-level functionality
│       │   ├── list_models.py
│       │   ├── list_providers.py
│       │   ├── prompt.py
│       │   ├── prompt_from_file.py
│       │   └── prompt_from_file_to_file.py
│       ├── server.py          # MCP server implementation
│       └── tests/             # Test directory
│           ├── atoms/         # Tests for atoms
│           │   ├── llm_providers/
│           │   └── shared/
│           └── molecules/     # Tests for molecules
```

## Resources

* [Anthropic API Documentation](https://docs.anthropic.com/en/api/models-list?q=list+models)
* [Google Generative AI](https://github.com/googleapis/python-genai)
* [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/models/list)
* [DeepSeek API Documentation](https://api-docs.deepseek.com/api/list-models)
* [Ollama Python Client](https://github.com/ollama/ollama-python)
* [OpenAI Python Client](https://github.com/openai/openai-python) 