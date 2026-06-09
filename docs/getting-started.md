# Getting Started

## Purpose

This page explains how to install the Boo application dependencies, configure the local Python
environment, run the Streamlit application, install the documentation dependencies, and build or
serve the MkDocs documentation site.

## Prerequisites

Before running the application or documentation site, confirm that the workstation has the following
tools installed:

| Requirement | Purpose |
| --- | --- |
| Python 3.11 or later | Runs the Streamlit application and provider wrapper modules. |
| Git | Clones and updates the Boo repository. |
| PowerShell | Runs the Windows setup commands shown in this guide. |
| Local Boo repository clone | Provides the application source and `docs/` directory. |
| Provider API keys | Enables provider-backed workflows. Keys are only required for the providers being used. |

## Create the Virtual Environment

From the root of the Boo repository, create a local Python virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the PowerShell prompt should show the active environment name:

```text
(.venv) PS C:\path\to\Boo>
```

## Upgrade Packaging Tools

After activating the virtual environment, upgrade the packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

This helps avoid dependency installation failures caused by stale packaging packages.

## Install Application Dependencies

Install the application dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

This installs the packages needed to run the Streamlit application and provider wrapper modules.

## Install Documentation Dependencies

Install the documentation dependencies from `requirements-docs.txt`:

```powershell
pip install -r requirements-docs.txt
```

The documentation dependencies are intentionally separated from the application dependencies so the
Streamlit runtime does not need to carry MkDocs packages unless documentation generation is required.

## Configure Provider Credentials

Boo reads provider credentials from `config.py`, environment variables, Streamlit secrets, or
runtime state depending on the workflow and the local implementation.

Common provider configuration values include:

| Provider | Configuration Value | Purpose |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | Enables GPT text, image, audio, embedding, file, and vector-store workflows. |
| Google Gemini | `GEMINI_API_KEY` | Enables Gemini text, multimodal, file, and grounded generation workflows. |
| Google APIs | `GOOGLE_API_KEY` | Supports related Google API workflows where configured. |
| xAI Grok | `XAI_API_KEY` | Enables Grok text, image-capable, and native xAI SDK workflows. |
| xAI Management | `XAI_MANAGEMENT_KEY` | Enables management operations where the Grok wrapper requires a separate management credential. |

For local development, API keys may be set in PowerShell before launching the application:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:GEMINI_API_KEY = "your-gemini-api-key"
$env:XAI_API_KEY = "your-xai-api-key"
$env:XAI_MANAGEMENT_KEY = "your-xai-management-key"
```

Do not commit real API keys to the repository.

## Recommended Local Configuration Pattern

A clean local development setup normally follows this pattern:

1. Store stable non-secret defaults in `config.py`.
2. Store secrets in environment variables or `.streamlit/secrets.toml`.
3. Keep provider-specific model names, vector-store names, and feature flags centralized.
4. Avoid hard-coded local paths when the same path can be provided by an environment variable.
5. Validate required provider credentials inside provider wrapper methods before making SDK calls.

This pattern keeps the repository portable across development workstations, Streamlit Cloud,
containers, and server-hosted environments.

## Run the Streamlit Application

Start the Boo Streamlit application from the repository root:

```powershell
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually similar to:

```text
http://localhost:8501
```

Open that URL in a browser to use the application.

## Run the Documentation Site Locally

Start the MkDocs development server from the repository root:

```powershell
mkdocs serve
```

MkDocs will display a local documentation URL in the terminal, usually similar to:

```text
http://127.0.0.1:8000
```

Open that URL in a browser to review the documentation site.

## Build the Static Documentation Site

Build the static documentation site with:

```powershell
mkdocs build
```

The generated static site is written to:

```text
site/
```

The `site/` folder is generated output. It should normally be excluded from source control unless
the project intentionally commits static documentation artifacts.

## Recommended Local Development Flow

Use this sequence for normal development:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-docs.txt
streamlit run app.py
```

In a second PowerShell window, with the same environment activated, run:

```powershell
mkdocs serve
```

This allows the application and documentation site to run side by side during development.

## Verify Documentation Generation

After editing Python docstrings or Markdown documentation files, verify that MkDocs can build
successfully:

```powershell
mkdocs build
```

If the build succeeds, the terminal should report that the site was generated successfully.

## MkDocs Configuration

The `mkdocs.yml` configuration should include `mkdocstrings` with Google-style docstring parsing:

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
            show_source: true
            show_root_heading: true
            show_root_full_path: false
```

## Troubleshooting

### The virtual environment will not activate

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### MkDocs command is not recognized

Confirm the documentation dependencies were installed into the active virtual environment:

```powershell
pip install -r requirements-docs.txt
```

Then verify MkDocs is available:

```powershell
mkdocs --version
```

### API documentation does not render

Confirm that `mkdocs.yml` includes the `mkdocstrings` plugin and that the documented modules are
importable from the repository root.

Also confirm that the documented module names match the actual source filenames. For example:

```markdown
::: gpt
::: gemini
::: grok
```

These directives expect `gpt.py`, `gemini.py`, and `grok.py` to be importable from the repository
root or from a configured package path.

### The `app.py` API page fails during build

If `::: app` causes a build failure, the likely cause is Streamlit runtime code executing during
module import. Streamlit applications often run page setup, session-state initialization, and widget
creation at import time.

A safer long-term pattern is to move reusable application logic into import-safe modules, such as:

```text
src/app_services.py
src/session_state.py
src/rendering.py
src/provider_router.py
```

Then document those modules instead of directly documenting `app.py`.

### A provider wrapper fails during documentation build

Provider wrappers should not require live API credentials simply to import. If MkDocs fails because a
provider wrapper creates a client at import time, move client construction into the method that makes
the provider call. Importing the module should define classes and functions only.

## Completion Checklist

Before committing documentation changes, run:

```powershell
mkdocs build
```

Then confirm:

- the site builds without import errors;
- the API reference pages render the expected classes and methods;
- navigation links in `mkdocs.yml` point to existing files;
- no real API keys, local secrets, or generated `site/` files were committed.
