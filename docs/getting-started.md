# Getting Started

## Purpose

This page explains how to install the Boo application dependencies, configure the local Python
environment, run the Streamlit application, install the documentation dependencies, and build or
serve the MkDocs documentation site.

## Prerequisites

Before running the application or documentation site, confirm that the workstation has the following
tools installed:

* Python 3.11 or later
* Git
* PowerShell
* A local clone of the Boo repository
* Access to the required provider API keys, if provider-backed workflows will be used

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
Streamlit runtime does not need to carry MkDocs packages unless documentation generation is required
in that environment.

## Configure Provider Credentials

The application reads provider credentials from `config.py`, environment variables, or Streamlit
runtime state depending on the workflow.

Common provider configuration values include:

| Provider      | Configuration Value | Purpose                                                                      |
| ------------- | ------------------- | ---------------------------------------------------------------------------- |
| OpenAI        | `OPENAI_API_KEY`    | Enables GPT text, image, audio, embedding, file, and vector-store workflows. |
| Google Gemini | `GEMINI_API_KEY`    | Enables Gemini text, image, file, and grounded generation workflows.         |
| Google APIs   | `GOOGLE_API_KEY`    | Supports related Google API workflows where configured.                      |
| xAI Grok      | `XAI_API_KEY`       | Enables Grok text, image, and native xAI SDK workflows.                      |

For local development, API keys may be set in PowerShell before launching the application:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:GEMINI_API_KEY = "your-gemini-api-key"
$env:XAI_API_KEY = "your-xai-api-key"
```

Do not commit real API keys to the repository.

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

If the build fails while importing `app.py`, remove the `app.py` API reference page temporarily and
document only import-safe modules such as `gpt.py`, `gemini.py`, and `grok.py`. Streamlit
applications often execute UI code at import time, which can make direct API documentation
generation fragile.

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

The `mkdocs.yml` configuration should include:

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

### The `app.py` API page fails during build

If `::: app` causes a build failure, the likely cause is Streamlit runtime code executing during
module import. In that case, remove the `app.py` API page from `mkdocs.yml` and document import-safe
provider modules first.

A safer long-term pattern is to move reusable application logic into import-safe modules, such as:

```text
src/app_services.py
src/session_state.py
src/rendering.py
src/provider_router.py
```

Then document those modules instead of directly documenting `app.py`.

## Next Step

After this page is added, continue by adding:

```text
docs/configuration.md
docs/user-guide.md
docs/architecture.md
docs/api/gpt.md
docs/api/gemini.md
docs/api/grok.md
```
