# Boo Documentation

Boo is a multi-provider AI assistant application built with Streamlit. It provides a single user
interface for provider-backed text, image, audio, embedding, file, document, vector-store, and local
data workflows.

## Purpose

This documentation explains how to install, configure, run, maintain, and extend the Boo
application. It also provides API reference pages generated from Google-style Python docstrings in
the application source modules.

## Application Scope

Boo is designed as a development and productivity workbench for multiple AI providers. The
application centralizes common assistant workflows while preserving provider-specific capabilities
where the underlying APIs differ.

The current documentation set covers:

| Section | Description |
| --- | --- |
| [Getting Started](getting-started.md) | Local setup, dependency installation, runtime configuration, and MkDocs commands. |
| [Architecture](architecture.md) | Application structure, provider routing, module responsibilities, and data flow. |
| [Application API](app.md) | Streamlit shell documentation and import-safety notes for `app.py`. |
| [GPT API](gpt.md) | OpenAI GPT provider wrapper reference. |
| [Gemini API](gemini.md) | Google Gemini provider wrapper reference. |
| [Grok API](grok.md) | xAI Grok provider wrapper reference. |

## Core Capabilities

Boo provides a common interface for the following workflows:

- conversational text generation;
- document question answering;
- semantic search;
- prompt engineering and prompt testing;
- image generation or image analysis where supported by the selected provider;
- audio generation, transcription, or translation where supported by the selected provider;
- embeddings and vector-store workflows;
- file upload, file retrieval, and file-based assistant operations;
- local data utility workflows used during development and testing.

## Provider Model

Boo uses provider wrapper modules to isolate provider-specific API syntax from the Streamlit user
interface. This keeps the application shell easier to maintain because the UI can call stable wrapper
methods rather than directly embedding provider SDK logic in the page rendering code.

The primary provider modules are:

| Module | Provider | Typical Responsibilities |
| --- | --- | --- |
| `gpt.py` | OpenAI | Text, images, audio, embeddings, files, vector stores, and assistant workflows. |
| `gemini.py` | Google Gemini | Text, multimodal generation, files, and Gemini-specific grounded workflows. |
| `grok.py` | xAI Grok | Text, image-capable workflows, embeddings where supported, files, and vector stores. |
| `app.py` | Streamlit | Page layout, session state, provider selection, workflow routing, and display logic. |

## Documentation Generation

The API reference pages use `mkdocstrings` directives such as:

```markdown
::: gpt
```

The documentation build expects the source modules to be importable from the repository root. If a
module executes UI code, network calls, or provider client initialization at import time, MkDocs may
fail during API generation. In that case, move import-safe reusable logic into separate modules and
document those modules instead.

## Recommended Review Order

Read the documentation in this order when onboarding to the project:

1. [Getting Started](getting-started.md)
2. [Architecture](architecture.md)
3. [Application API](app.md)
4. [GPT API](gpt.md)
5. [Gemini API](gemini.md)
6. [Grok API](grok.md)

This order starts with operational setup, then explains the design, and finally moves into the
source-level API reference.
