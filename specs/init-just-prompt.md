# Just-Prompt Development Plan

## Project Phases

### Phase 1: Basic Framework Setup (Completed)

- [x] Create basic project structure
- [x] Set up configuration files (pyproject.toml, .env.sample, etc.)
- [x] Create core data types
- [x] Create basic server framework

### Phase 2: Implement LLM Provider Interfaces

- [ ] Implement OpenAI provider interface
  - [ ] Model list retrieval
  - [ ] Text completion/chat API
  - [ ] Error handling and retry mechanism

- [ ] Implement Anthropic provider interface
  - [ ] Model list retrieval
  - [ ] Messages API
  - [ ] Support for thinking tokens feature
  - [ ] Error handling and retry mechanism

- [ ] Implement Google Gemini provider interface
  - [ ] Model list retrieval
  - [ ] Text generation API
  - [ ] Error handling and retry mechanism

- [ ] Implement Groq provider interface
  - [ ] Model list retrieval
  - [ ] Chat completion API
  - [ ] Error handling and retry mechanism

- [ ] Implement DeepSeek provider interface
  - [ ] Model list retrieval
  - [ ] Text generation API
  - [ ] Error handling and retry mechanism

- [ ] Implement Ollama provider interface
  - [ ] Model list retrieval
  - [ ] Generation API
  - [ ] Error handling and retry mechanism

### Phase 3: Implement Routing and Dispatching

- [ ] Implement model router
  - [ ] Parse model names based on provider prefix
  - [ ] Support model name correction feature
  - [ ] Handle cases where models are not found

- [ ] Implement parallel request processing
  - [ ] Async parallel calls to multiple models
  - [ ] Handle partial model failures
  - [ ] Timeout handling

### Phase 4: Feature Extensions

- [ ] Implement prompt loading from files
  - [ ] Support multiple file formats
  - [ ] Handle cases where files don't exist

- [ ] Implement saving responses to files
  - [ ] Support multiple output formats
  - [ ] Handle file permission issues

- [ ] Implement provider and model listing commands
  - [ ] Display currently available providers
  - [ ] Display models supported by each provider

### Phase 5: Testing and Documentation

- [ ] Unit tests
  - [ ] Test each provider interface
  - [ ] Test model router
  - [ ] Test parallel processing

- [ ] Integration tests
  - [ ] Test end-to-end workflows
  - [ ] Test error scenarios

- [ ] Documentation
  - [ ] API usage documentation
  - [ ] Deployment guide
  - [ ] Developer documentation

### Phase 6: Optimization and Release

- [ ] Performance optimization
  - [ ] Reduce API call latency
  - [ ] Optimize memory usage

- [ ] Release preparation
  - [ ] Version definition
  - [ ] Package and publish to PyPI

## Technology Stack

- **Python**: 3.10+
- **Web Framework**: FastAPI
- **Async Processing**: asyncio
- **HTTP Client**: httpx
- **Dependency Management**: uv/pip
- **Testing Framework**: pytest
- **Code Quality**: black, isort, ruff

## External Library Dependencies

- **openai**: OpenAI API client
- **anthropic**: Anthropic API client
- **google-generativeai**: Google Gemini API client
- **groq**: Groq API client
- **ollama**: Ollama API client
- **pydantic**: Data validation
- **fastapi**: API server
- **uvicorn**: ASGI server

## Development Standards

1. **Code Style**: Follow PEP 8, use black and isort for formatting
2. **Type Hints**: Use type annotations for all functions and methods
3. **Error Handling**: Handle and report errors appropriately
4. **Documentation**: Provide docstrings for all public APIs
5. **Testing**: Write tests for new features, maintain test coverage

## Milestones

1. **Alpha Version** (0.1.0): Complete phases 1-3, support basic prompt functionality
2. **Beta Version** (0.2.0): Complete phase 4, support all planned features
3. **RC Version** (0.9.0): Complete phase 5, include comprehensive tests and documentation
4. **Final Release** (1.0.0): Complete phase 6, publish to PyPI 