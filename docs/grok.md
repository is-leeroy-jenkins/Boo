# Grok API

## Purpose

This page documents the xAI Grok provider wrapper module, `grok.py`.

The Grok wrapper isolates xAI-specific SDK calls from the Streamlit application shell. It should
provide typed, documented methods for Grok text, image-capable, embedding, file, and vector-store
workflows where those features are implemented in the source code.

## Provider Scope

The Grok module may support the following workflow families depending on the current source code:

| Workflow | Description |
| --- | --- |
| Chat/Text | Sends prompts and conversation context to a Grok model. |
| Image-Capable Workflows | Handles image input or image-related generation where supported by the selected Grok model. |
| Embeddings | Creates embeddings where supported by the xAI SDK or configured workflow. |
| Files | Uploads, retrieves, lists, or deletes provider-managed files where implemented. |
| Vector Stores | Creates, lists, searches, or manages vector stores and file batches where implemented. |
| Management Operations | Uses a management key for administrative operations where required. |

## Design Contract

The wrapper should follow these conventions:

| Contract | Expected Pattern |
| --- | --- |
| Configuration | Assign `XAI_API_KEY`, `XAI_MANAGEMENT_KEY`, model names, and collection mappings from config. |
| Validation | Validate mandatory method arguments before provider calls. |
| Client lifecycle | Create xAI clients inside the method that uses them, after credential validation. |
| Collections | Use a typed collection mapping such as `Dict[str, str]` for named vector-store collections. |
| Error handling | Capture exceptions using the project `Error` and `Logger` pattern. |
| Documentation | Use Google-style docstrings for every public class and method. |

## Common Configuration Values

| Value | Purpose |
| --- | --- |
| `XAI_API_KEY` | Primary xAI/Grok credential used for runtime calls. |
| `XAI_MANAGEMENT_KEY` | Management credential used for administrative or vector-store operations where required. |
| Grok model constants | Default model names for text, reasoning, or multimodal workflows. |
| `GROK_COLLECTIONS` | Mapping of logical collection names to provider collection/vector-store identifiers. |

## Usage Pattern

A Streamlit workflow should call the Grok wrapper through a narrow method boundary:

```python
wrapper = Grok()
response = wrapper.create_response( text=prompt_text, model=model_name )
```

The exact class and method names should match the source code. The important architectural rule is
that provider-specific xAI request construction should remain inside `grok.py`.

## Collection Handling

For vector-store and search workflows, prefer a strongly typed configuration mapping:

```python
self.collections = cfg.GROK_COLLECTIONS
```

The mapping should be validated in `config.py` so wrapper code can rely on a predictable
`Dict[str, str]` shape.

## Documentation Guidance

Every public method should document:

- the selected model or collection input;
- whether the method requires the runtime key or management key;
- whether provider-side files or vector stores are created;
- what identifier or response object is returned;
- how exceptions are logged and surfaced.

## API Reference

The section below is generated from the source module when `mkdocs build` runs.

::: grok
    options:
      show_source: true
      show_root_heading: true
      show_root_full_path: false
