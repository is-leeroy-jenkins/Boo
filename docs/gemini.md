# Gemini API

## Purpose

This page documents the Google Gemini provider wrapper module, `gemini.py`.

The Gemini wrapper isolates Google-specific SDK calls from the Streamlit application shell. It should
provide documented methods for Gemini text, multimodal, file, and grounded-generation workflows while
keeping Google SDK details outside the UI layer.

## Provider Scope

The Gemini module may support the following workflow families depending on the current source code:

| Workflow | Description |
| --- | --- |
| Chat/Text | Sends prompts to Gemini text-capable models. |
| Multimodal Generation | Sends text plus images or files to Gemini multimodal models. |
| Image Workflows | Handles image-related workflows where supported by the configured Gemini model. |
| Files | Uploads, retrieves, lists, or deletes Gemini-managed files. |
| Grounded Generation | Uses Google-supported grounding or search features where configured. |
| File Search Stores | Manages provider-side file-search or document-search resources where implemented. |

## Design Contract

The wrapper should follow these conventions:

| Contract | Expected Pattern |
| --- | --- |
| Configuration | Assign Gemini API keys and reusable model names from `config.py` or environment-derived config. |
| Validation | Validate mandatory method arguments before provider calls. |
| Client lifecycle | Create Gemini clients inside the method that uses them, not at import time. |
| File handling | Keep file upload, polling, and cleanup logic inside the wrapper. |
| Error handling | Capture exceptions using the project `Error` and `Logger` pattern. |
| Documentation | Use Google-style docstrings for every public class and method. |

## Common Configuration Values

| Value | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Primary Gemini credential. |
| `GOOGLE_API_KEY` | Related Google API credential where used. |
| Gemini model constants | Default text or multimodal model names. |
| File or store constants | Default Gemini file-search store names or identifiers where implemented. |

## Usage Pattern

A Streamlit workflow should call the Gemini wrapper through a narrow method boundary:

```python
wrapper = Gemini()
response = wrapper.generate_content( text=prompt_text, model=model_name )
```

The exact class and method names should match the source code. The important architectural rule is
that `app.py` should not directly embed Google SDK request construction.

## File Workflow Notes

Gemini file workflows often require additional steps beyond a simple prompt call:

1. upload the local file;
2. wait until the provider marks the file usable;
3. include the file reference in a model request;
4. optionally clean up provider-side files.

Those steps should remain encapsulated in `gemini.py` so the UI can remain simple.

## Documentation Guidance

Every public method should document:

- whether it accepts raw text, local file paths, uploaded file objects, or provider file IDs;
- whether it creates provider-managed resources;
- whether polling or waiting is performed;
- what response object or normalized payload is returned;
- what errors are logged or surfaced to the caller.

## API Reference

The section below is generated from the source module when `mkdocs build` runs.

::: gemini
    options:
      show_source: true
      show_root_heading: true
      show_root_full_path: false
