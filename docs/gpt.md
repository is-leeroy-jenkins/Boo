# GPT API

## Purpose

This page documents the OpenAI GPT provider wrapper module, `gpt.py`.

The GPT wrapper isolates OpenAI-specific SDK calls from the Streamlit application shell. The wrapper
should expose typed, documented methods for text, image, audio, embedding, file, and vector-store
workflows while keeping provider-specific request and response handling out of `app.py`.

## Provider Scope

The GPT module may support the following workflow families depending on the current source code:

| Workflow | Description |
| --- | --- |
| Chat/Text | Sends user prompts and optional conversation context to an OpenAI text-capable model. |
| Images | Generates, edits, or analyzes images where supported by the configured OpenAI model. |
| Audio | Handles text-to-speech, transcription, and translation workflows where implemented. |
| Embeddings | Converts text or document chunks into embedding vectors. |
| Files | Uploads, lists, retrieves, or deletes provider-managed files. |
| Vector Stores | Creates, lists, updates, searches, or deletes vector stores and related file batches. |
| Assistants or Responses | Executes higher-level assistant workflows where the wrapper exposes them. |

## Design Contract

The wrapper should follow these conventions:

| Contract | Expected Pattern |
| --- | --- |
| Configuration | Assign API keys and reusable configuration values from `config.py` or environment-derived config. |
| Validation | Validate mandatory method arguments before provider calls. |
| Client lifecycle | Create the OpenAI client inside the method that uses it, after credential validation. |
| Error handling | Capture exceptions using the project `Error` and `Logger` pattern. |
| Return values | Return provider responses, normalized dictionaries, files, IDs, or rendered-safe payloads. |
| Documentation | Use Google-style docstrings for every public class and method. |

## Common Configuration Values

| Value | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI credential used by GPT workflows. |
| GPT model constants | Default text or multimodal model names. |
| Image model constants | Default image-generation or image-editing model names. |
| Audio model constants | Default text-to-speech, transcription, or translation model names. |
| Embedding model constants | Default embedding model names. |
| Vector-store constants | Default vector-store names, IDs, or lookup values. |

## Usage Pattern

A Streamlit workflow should call the GPT wrapper through a narrow method boundary:

```python
wrapper = GPT()
response = wrapper.create_response( text=prompt_text, model=model_name )
```

The exact class and method names should match the source code. The important architectural rule is
that Streamlit should not need to know the low-level OpenAI request shape.

## Documentation Guidance

Every public method should document:

- the purpose of the operation;
- every input parameter and accepted value shape;
- the return type and meaning;
- whether the method creates remote resources;
- whether the method returns provider-managed IDs;
- any important limitations or provider-specific behavior.

## API Reference

The section below is generated from the source module when `mkdocs build` runs.

::: gpt
    options:
      show_source: true
      show_root_heading: true
      show_root_full_path: false
