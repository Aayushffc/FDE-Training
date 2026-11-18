# Day 1 — Daily Progress Log
**Date:** 17 November 2026  
**Track:** AI Engineering / Forward Deployed Engineer  
**Member:** Member B — AI Agents  
**Focus:** LLM Gateway Setup and Provider Integration

## 1. Work Completed

### Project Setup and Architecture
- Established the FastAPI project structure with separate layers for routers, services, schemas, and core utilities.
- Configured project dependencies using `pyproject.toml` and migrated dependency management to `uv`.
- Set up the project configuration and environment variable handling.
- Organized provider integrations into independent service modules.

### LLM Provider Integrations
- Implemented separate service integrations for OpenAI and Anthropic.
- Added Groq as an additional provider integration.
- Implemented a shared chat service to coordinate provider interactions.
- Worked towards a consistent response structure across providers, including response text, token usage, latency, and estimated cost.

### API Development
- Implemented the FastAPI chat router.
- Added request and response schemas using Pydantic.
- Implemented request validation and provider selection.
- Added a health-check endpoint.
- Implemented centralized custom exception handling for application and provider errors.

### Logging and Observability
- Implemented structured logging for LLM requests.
- Added token usage, latency, provider, model, request status, and estimated cost tracking.
- Implemented model-based cost calculation.
- Improved development logging with console output and file logging.

### Automated Testing
- Configured pytest.
- Organized tests into API, integration, smoke, and unit test categories.
- Implemented tests covering:
  - Health endpoint.
  - Chat API.
  - Request and response schema validation.
  - Custom exception handling.
  - OpenAI service.
  - Anthropic service.
  - Groq service.
  - Logging and cost calculation.
  - Live API smoke testing.

## 2. Technical Decisions

- Adopted a layered architecture to keep HTTP handling separate from business logic and provider integrations.
- Kept provider-specific API implementations isolated in individual service modules.
- Used Pydantic for request and response validation.
- Introduced a common exception hierarchy for consistent error handling.
- Adopted `uv` for dependency management, with `pyproject.toml` as the project configuration and `uv.lock` for dependency reproducibility.
- Separated automated tests into different categories to distinguish unit, API, integration, and live smoke tests.

## 3. Testing and Validation

- Implemented automated tests across the main application components.
- Added provider integration tests and a separate live smoke-test suite.
- Covered schema validation, error handling, and cost calculation.

**Additional verification:** Confirm successful live API calls to both OpenAI and Anthropic, check the spending limits, and capture the `/chat` endpoint demonstration evidence.

## 4. Challenges and Learnings

- Became more familiar with FastAPI's routing, Pydantic validation, exception handling, and application structure.
- Established a consistent architecture for integrating multiple LLM providers.
- Improved understanding of provider-specific response handling and token usage extraction.
- Set up a reproducible Python development environment using uv.
- Established a testing structure that can be extended as the project grows.

## 5. Day 1 Outcome

The foundational LLM gateway implementation is in place, with multiple provider integrations, a shared chat layer, API validation, centralized error handling, structured logging, cost tracking, and an organized automated test suite.

## 6. Next Steps — Day 2

Focus on prompt engineering, prompt management, and structured outputs.

Planned activities:
- Experiment with three different prompt versions using the same test inputs.
- Create a versioned prompt library.
- Implement structured output generation with JSON schema and Pydantic validation.
- Add streaming support.
- Demonstrate at least one LLM tool/function call.
- Record token usage and estimated cost.
- Document hallucination and direct/indirect prompt injection examples.
- Record relevant technical decisions in the decision log.