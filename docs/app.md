# Application API

## Purpose

This page documents the Streamlit application module, `app.py`.

`app.py` is the user-interface shell for Boo. It coordinates model selection, mode selection, user
inputs, uploaded files, provider routing, result rendering, and session-state persistence.

## Module Responsibilities

The application module normally owns the following responsibilities:

| Area | Responsibility |
| --- | --- |
| Page setup | Configure Streamlit page metadata, layout, title, branding, and top-level containers. |
| Navigation | Render tabs, expanders, sidebars, model selectors, mode selectors, and workflow controls. |
| Session state | Initialize, read, and update `st.session_state` keys used across reruns. |
| Provider routing | Dispatch user requests to `gpt.py`, `gemini.py`, or `grok.py` based on selected provider/model. |
| Input handling | Read text prompts, uploaded documents, URLs, audio files, images, and workflow options. |
| Output rendering | Display responses, errors, metrics, tables, images, files, and downloadable artifacts. |
| Error handling | Convert runtime exceptions into clear user-facing messages and developer-friendly logs. |

## Streamlit Import Warning

Streamlit application files often execute UI code at import time. That can make direct
`mkdocstrings` documentation fragile because MkDocs imports the module during the build.

If this page causes `mkdocs build` to fail, remove `::: app` temporarily and document import-safe
helper modules instead. A better long-term design is to move reusable application logic into modules
that do not render Streamlit widgets at import time.

Recommended import-safe module candidates include:

```text
src/app_services.py
src/session_state.py
src/provider_router.py
src/rendering.py
src/file_services.py
```

## Session-State Contract

The application should use explicit session-state keys for values that must survive Streamlit
reruns. The exact keys depend on the current source code, but the application should generally
separate these categories:

| State Category | Example Keys |
| --- | --- |
| Selection state | `selected_model_name`, `selected_mode_name`, `selected_provider_name` |
| Input state | `prompt_text`, `uploaded_files`, `uploaded_documents`, `allow_domains` |
| Intermediate state | `parsed_documents`, `chunked_documents`, `embedding_results` |
| Provider state | `provider_client_status`, `vector_store_id`, `file_ids` |
| Output state | `chat_response`, `image_response`, `audio_response`, `dataframe_response` |
| UI state | `active_tab`, `show_advanced_options`, `sidebar_expanded` |

Session-state keys should be written before they are read. Keys should not be reused for unrelated
data structures.

## Workflow Routing

A typical application workflow should follow this pattern:

1. Read the selected provider, model, and mode from Streamlit controls.
2. Validate required user inputs for the selected mode.
3. Instantiate or call the correct provider wrapper.
4. Execute the provider method.
5. Normalize the result for display.
6. Save the result to session state if it is needed after rerun.
7. Render the result in the appropriate UI panel.

## Documentation Guidance

Public helper functions in `app.py` should use Google-style docstrings so they can be rendered by
MkDocs when the module is import-safe.

A recommended docstring pattern is:

```python
def render_text_mode( selected_model_name: str ) -> None:
    """
    Purpose:
        Render the text-generation workflow for the selected model.

    Args:
        selected_model_name (str): Name of the model selected in the Streamlit UI.

    Returns:
        None: Writes Streamlit controls and output directly to the page.
    """
```

## API Reference

The section below is generated from the source module when `mkdocs build` runs.

::: app
    options:
      show_source: true
      show_root_heading: true
      show_root_full_path: false
