# Configuration

## Purpose

This page documents the configuration model for the Boo application. It explains how local settings,
provider credentials, runtime defaults, model options, and documentation settings should be managed
for development and demonstration deployments.

Boo is designed as a multi-provider Streamlit application. Configuration should therefore remain
explicit, predictable, and easy to audit. Provider keys, model names, file paths, feature flags, and
UI defaults should be centralized in `config.py`, environment variables, or Streamlit secrets rather
than scattered through the application code.

## Configuration Principles

Use these principles when adding or changing configuration:

1. Keep secrets out of source control.
2. Prefer environment variables or Streamlit secrets for API keys.
3. Keep provider-specific constants grouped by provider.
4. Use clear names that match the provider or feature they configure.
5. Do not hard-code local workstation paths into application modules.
6. Do not require every provider to be configured before the application starts.
7. Fail gracefully when a selected provider is missing its required key.

The application should be able to load even when only one provider is configured. Provider-specific
errors should occur only when the user selects a workflow that requires the missing credential.

## Common Configuration Files

A typical Boo repository uses the following configuration files:

| File | Purpose |
| --- | --- |
| `config.py` | Central Python configuration constants used by the application and provider wrappers. |
| `.env` | Optional local environment-variable file for development. Do not commit this file. |
| `.streamlit/secrets.toml` | Optional Streamlit secrets file for local or hosted Streamlit deployments. Do not commit real secrets. |
| `requirements.txt` | Runtime dependencies for the Streamlit application and provider wrappers. |
| `requirements-docs.txt` | Documentation dependencies for MkDocs and mkdocstrings. |
| `mkdocs.yml` | Documentation site navigation, theme, plugin, and mkdocstrings configuration. |

## Provider Credentials

Boo supports provider-backed workflows through OpenAI GPT, Google Gemini, and xAI Grok wrappers.
Each provider should have its own credential variable.

| Provider | Environment Variable | Typical Use |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | GPT chat, image, audio, embeddings, files, and vector stores. |
| Google Gemini | `GEMINI_API_KEY` | Gemini text, image, file, and grounded generation workflows. |
| xAI Grok | `XAI_API_KEY` | Grok chat, image, embeddings, files, and vector-store workflows. |
| xAI Management | `XAI_MANAGEMENT_KEY` | Optional management or administrative xAI workflows, if implemented. |

For PowerShell development, set keys before launching Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:GEMINI_API_KEY = "your-gemini-api-key"
$env:XAI_API_KEY = "your-xai-api-key"
$env:XAI_MANAGEMENT_KEY = "your-xai-management-key"
```

Do not paste real provider keys into Markdown documentation, screenshots, Git commits, issue
comments, or pull-request descriptions.

## Streamlit Secrets

For Streamlit-based local secrets, use `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
XAI_API_KEY = "your-xai-api-key"
XAI_MANAGEMENT_KEY = "your-xai-management-key"
```

The application may read from `st.secrets`, environment variables, or `config.py`, depending on how
provider wrappers are implemented. The preferred pattern is to keep credentials out of code and load
them into configuration constants at startup.

## Recommended `config.py` Pattern

Use `config.py` for centralized constants and safe defaults. A simplified pattern is shown below:

```python
from __future__ import annotations

import os
from typing import Dict, List

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
XAI_MANAGEMENT_KEY: str = os.getenv("XAI_MANAGEMENT_KEY", "")

DEFAULT_PROVIDER: str = "GPT"
DEFAULT_MODE: str = "Text"

GPT_MODELS: List[str] = [
    "gpt-4.1",
    "gpt-4.1-mini",
]

GEMINI_MODELS: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

GROK_MODELS: List[str] = [
    "grok-4",
    "grok-3-mini",
]

GROK_COLLECTIONS: Dict[str, str] = {
    "Default": "default",
}
```

The exact model names should match the supported models in the installed provider SDKs and the
application's provider wrapper classes.

## Provider Wrapper Configuration Contract

Provider wrappers should follow a consistent configuration contract:

| Wrapper | Required Configuration | Optional Configuration |
| --- | --- | --- |
| `gpt.py` | `OPENAI_API_KEY` | Default model, image size, embedding model, vector-store defaults. |
| `gemini.py` | `GEMINI_API_KEY` | Default model, grounding options, file-search defaults. |
| `grok.py` | `XAI_API_KEY` | `XAI_MANAGEMENT_KEY`, default model, collection mapping. |

The wrapper constructor should assign configuration values to instance members. Individual methods
should validate required user inputs before calling the provider SDK. Missing provider credentials
should produce clear, actionable messages.

## Model and Mode Configuration

Boo is organized around provider and workflow selections. Common workflow modes include:

| Mode | Description |
| --- | --- |
| Text | General chat, question answering, drafting, coding, and analysis. |
| Image | Image generation or image-related provider workflows where supported. |
| Audio | Text-to-speech, transcription, or translation workflows where supported. |
| Document Q&A | File upload, document parsing, retrieval, and document-grounded answering. |
| Semantic Search | Embedding and similarity-search workflows. |
| Prompt | Prompt construction, testing, and reusable prompt templates. |
| Data | Data analysis or structured-data utility workflows. |

Do not expose modes for a provider unless that provider wrapper implements the corresponding
workflow. This avoids presenting buttons or controls that cannot execute.

## File and Directory Settings

The application should keep generated or temporary files out of source directories unless the file is
an intentional repository artifact.

Recommended paths:

| Path | Purpose | Source Control |
| --- | --- | --- |
| `docs/` | MkDocs documentation source files. | Commit. |
| `site/` | Generated MkDocs static site. | Usually ignore. |
| `.venv/` | Local virtual environment. | Ignore. |
| `.streamlit/secrets.toml` | Local Streamlit secrets. | Ignore real secrets. |
| `data/` | Optional local test data or examples. | Commit only safe examples. |
| `outputs/` | Optional generated artifacts. | Usually ignore. |
| `logs/` | Optional local logs. | Usually ignore. |

## Documentation Configuration

MkDocs should include `mkdocstrings` so API pages can render from Google-style Python docstrings.
A typical `mkdocs.yml` plugin section is:

```yaml
plugins:
  - search
  - autorefs
  - mkdocstrings:
      handlers:
        python:
          paths:
            - .
          options:
            docstring_style: google
```

A typical navigation section is:

```yaml
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Configuration: configuration.md
  - User Guide: user-guide.md
  - Architecture: architecture.md
  - API Reference:
      - Application: app.md
      - GPT: gpt.md
      - Gemini: gemini.md
      - Grok: grok.md
```

If `::: app` causes MkDocs to execute Streamlit UI code during import, remove the application API
page from the navigation and document import-safe helper modules instead.

## Security Notes

Configuration should protect secrets and reduce accidental data exposure:

- Never commit real API keys.
- Never write provider keys to logs.
- Never print secrets in Streamlit status messages.
- Avoid storing uploaded documents unless the user explicitly downloads or saves an output.
- Prefer temporary in-memory handling for files used only during a session.
- Keep sample data synthetic, public, or explicitly approved for repository use.

## Validation Checklist

Before committing configuration changes, verify the following:

- `streamlit run app.py` starts without requiring every provider key.
- Missing provider keys produce clear messages only when that provider is used.
- `mkdocs build` succeeds.
- `requirements.txt` contains application dependencies.
- `requirements-docs.txt` contains documentation dependencies.
- Real secrets are not present in tracked files.
- Local path assumptions are not hard-coded into provider wrappers.

## Troubleshooting

### Provider key is missing

Confirm the key is available in the active PowerShell session:

```powershell
echo $env:OPENAI_API_KEY
echo $env:GEMINI_API_KEY
echo $env:XAI_API_KEY
```

If the value is blank, set the key and restart Streamlit.

### A provider mode appears but does not work

Confirm that the selected provider wrapper implements the selected mode. The UI should not advertise
provider workflows that are not implemented in the source module.

### MkDocs cannot import a module

Run the documentation build from the repository root:

```powershell
mkdocs build
```

If the error references `app.py`, the module may be executing Streamlit code during import. Document
provider modules first, then move application helper logic into import-safe modules.

### Local configuration works but deployment fails

Confirm that deployment secrets were configured in the hosting environment. Local PowerShell
environment variables do not automatically follow the application into Streamlit Cloud, Azure,
Docker, or other deployment targets.
