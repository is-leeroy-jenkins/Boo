# User Guide

## Purpose

This guide explains how to use the Boo application from the Streamlit interface. It is written for
users who need to select a provider, choose a workflow, provide inputs, run the request, and review
or export results.

Boo is a multi-provider AI assistant application. The visible options may vary depending on which
provider credentials are configured and which workflows are implemented in the installed source
modules.

## Application Layout

The Boo interface is organized around a Streamlit application shell. A typical session includes:

| Area | Purpose |
| --- | --- |
| Header | Shows the application name, selected model branding, or high-level status. |
| Sidebar | Provides provider, model, mode, and workflow controls. |
| Main panel | Displays the active workflow, input controls, generated responses, and results. |
| Upload controls | Accept files for document, image, audio, or data workflows where supported. |
| Output controls | Show generated text, extracted results, tables, file references, or downloads. |

The application is intended to keep provider selection and workflow execution clear. Users should
not need to edit source code to run normal workflows.

## Start the Application

From the root of the repository, activate the virtual environment and run Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal. The default URL is commonly:

```text
http://localhost:8501
```

## Basic Workflow

Use this sequence for normal operation:

1. Select the provider or model.
2. Select the workflow mode.
3. Enter text or upload files required by the workflow.
4. Review provider-specific settings.
5. Run the workflow.
6. Review the result.
7. Copy, download, or continue refining the output.

The exact controls depend on the selected mode.

## Select a Provider

Boo commonly supports these provider wrappers:

| Provider | Wrapper Module | Typical Capabilities |
| --- | --- | --- |
| GPT | `gpt.py` | Text, images, audio, embeddings, files, vector stores. |
| Gemini | `gemini.py` | Text, image, file, and grounded generation workflows. |
| Grok | `grok.py` | Text, image, embeddings, files, and vector-store workflows. |

A provider may appear in the interface even if its API key is not configured. If a selected workflow
requires a missing key, the application should display an actionable message explaining which key is
needed.

## Text Workflow

Use Text mode for general assistant tasks such as writing, summarization, coding, analysis,
brainstorming, and question answering.

Recommended steps:

1. Select the provider and text-capable model.
2. Select `Text` mode.
3. Enter the prompt in the main input area.
4. Adjust optional settings such as temperature, response length, or system instructions if those
   controls are exposed.
5. Submit the request.
6. Review the generated response.

Good prompts usually include the task, context, constraints, and desired output format.

Example prompt:

```text
Review the following Python function for correctness, edge cases, and maintainability. Return a
brief issue list followed by a corrected drop-in replacement.
```

## Image Workflow

Use Image mode when the selected provider supports image generation or image-related operations.

Recommended steps:

1. Select an image-capable provider and model.
2. Select `Image` mode.
3. Enter a detailed visual prompt.
4. Choose optional image settings if exposed by the UI.
5. Generate the image.
6. Review or download the result.

A strong image prompt describes subject, environment, composition, style, lighting, and constraints.

Example prompt:

```text
Create a clean application architecture diagram showing a Streamlit UI connected to GPT, Gemini,
and Grok provider wrappers, with separate file, embedding, and vector-store workflows.
```

## Audio Workflow

Use Audio mode when the selected provider supports transcription, translation, or text-to-speech.

Typical audio workflows include:

| Workflow | Input | Output |
| --- | --- | --- |
| Transcription | Audio file | Text transcript. |
| Translation | Audio file or transcript | Translated text. |
| Text-to-speech | Text | Audio file. |

Recommended steps:

1. Select an audio-capable provider.
2. Select `Audio` mode.
3. Upload the audio file or enter the source text.
4. Choose the desired operation.
5. Run the workflow.
6. Review the transcript, translation, or generated audio.

For best transcription results, use clear audio with minimal background noise.

## Document Q&A Workflow

Use Document Q&A mode to ask questions against uploaded documents.

Recommended steps:

1. Select a provider that supports file or document workflows.
2. Select `Document Q&A` mode.
3. Upload one or more supported documents.
4. Wait for file processing or indexing to complete.
5. Ask a question about the uploaded content.
6. Review the answer and any cited or referenced source sections if the UI exposes them.

Good document questions are specific and grounded in the uploaded material.

Example questions:

```text
Summarize the document's main requirements.
```

```text
List every requirement that mentions accessibility, batch processing, or performance.
```

```text
Create a traceability matrix from the source document.
```

## Semantic Search Workflow

Use Semantic Search mode to locate conceptually similar content using embeddings or vector stores.

Recommended steps:

1. Select a provider that supports embeddings or vector stores.
2. Select `Semantic Search` mode.
3. Upload, paste, or select the source content.
4. Create or select the target collection if the UI supports collections.
5. Enter the search query.
6. Run the search.
7. Review ranked matches and similarity scores.

Semantic search is useful when exact keywords are unreliable or when the question uses different
wording from the source material.

## Prompt Workflow

Use Prompt mode to draft, test, refine, and reuse prompts.

Recommended steps:

1. Select `Prompt` mode.
2. Enter the draft prompt.
3. Add role, instructions, constraints, examples, and output requirements.
4. Test the prompt with a representative input.
5. Refine until the output is stable and clear.
6. Save or copy the final prompt.

A strong prompt usually contains:

- role or operating posture;
- task definition;
- input context;
- constraints;
- output format;
- quality checks.

## Data Workflow

Use Data mode when the application exposes structured-data utilities.

Possible tasks include:

| Task | Description |
| --- | --- |
| CSV inspection | Load and preview tabular data. |
| Data cleaning | Standardize fields, remove blanks, or normalize values. |
| Summarization | Describe rows, columns, counts, and patterns. |
| Export | Download transformed or analyzed outputs. |

Recommended steps:

1. Select `Data` mode.
2. Upload a supported data file.
3. Review the preview and inferred schema.
4. Select the analysis or transformation action.
5. Run the workflow.
6. Review and download results.

## Files and Uploads

When uploading files, use supported formats for the selected workflow. Common file categories include:

| Category | Examples |
| --- | --- |
| Documents | PDF, TXT, DOCX, Markdown. |
| Data | CSV, JSON, XLSX. |
| Images | PNG, JPG, JPEG, WEBP. |
| Audio | MP3, WAV, M4A. |

The application should not require long-term storage for routine user uploads. Files should be used
for the active session unless the application explicitly offers a save or export option.

## Reviewing Results

After a workflow runs, review the result before using it externally. Provider-generated content may
need validation, especially for:

- factual claims;
- code correctness;
- citations or source references;
- calculations;
- policy or legal interpretations;
- provider-specific limitations;
- outputs generated from incomplete source files.

For code outputs, run linting, tests, or compilation checks before merging changes.

## Exporting Results

Depending on the workflow, Boo may support copying text, downloading files, exporting tables, or
saving generated artifacts. Use exports for final review, documentation, issue tracking, or repository
updates.

Recommended export practices:

- Use descriptive filenames.
- Do not export secrets or private keys.
- Review generated files before committing them.
- Keep generated documentation in `docs/` and generated sites in `site/` unless the repository uses a
  different convention.

## Common Errors

### Provider credential is missing

The selected workflow requires an API key that is not configured. Set the provider key in the active
environment or Streamlit secrets and restart the application.

### The selected mode is unavailable

The selected provider may not implement the chosen mode. Select a different provider or workflow.
The UI should avoid exposing unsupported provider-mode combinations.

### Uploaded file is not accepted

Confirm that the file type is supported by the selected workflow. Some providers limit file size,
file type, or the number of files that can be processed in one request.

### The response is incomplete

Large prompts, long documents, or provider token limits can produce incomplete results. Narrow the
question, reduce the source material, or process the document in sections.

### Documentation build fails after source edits

Run:

```powershell
mkdocs build
```

If the failure references a module import, confirm that the module is import-safe and that the
mkdocstrings path in `mkdocs.yml` points to the repository root.

## Recommended Operating Practices

Use these practices for consistent results:

- Start with the provider best suited to the task.
- Keep prompts specific and structured.
- Upload only the files needed for the current workflow.
- Review generated outputs before relying on them.
- Keep API keys and secrets out of prompts, logs, and documentation.
- Use `mkdocs build` after documentation or docstring changes.
- Use source control to track intentional documentation and code changes.

## Support Checklist

When reporting an issue, include:

- selected provider;
- selected model;
- selected mode;
- a short description of the input;
- the error message;
- whether the issue occurs after restarting Streamlit;
- whether provider credentials are configured;
- whether `mkdocs build` or `streamlit run app.py` fails.

Do not include API keys, secrets, private tokens, or sensitive uploaded content in issue reports.
