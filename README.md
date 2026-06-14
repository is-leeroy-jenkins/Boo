###### Boo

<img src="https://github.com/is-leeroy-jenkins/Boo/blob/main/resources/images/boo_project.png" width="800" height="200"/>


<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-features">Features</a> |
  <a href="#-application-modes">Modes</a> |
  <a href="#-requirements">Requirements</a> |
  <a href="#-api-key-setup">Setup</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-running-the-streamlit-application">Run</a> |
  <a href="#-configuration">Configuration</a> |
  <a href="#-design-and-architecture">Architecture</a> |
  <a href="#-capabilities">Capabilities</a> |
</p>

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=github)](https://is-leeroy-jenkins.github.io/Boo/)

Boo is a Python and Streamlit application for building, running, and managing provider-aware
artificial intelligence workflows across OpenAI GPT, Google Gemini, and xAI Grok. It supports text
generation, image generation and analysis, image editing, audio transcription, audio translation,
text-to-speech, embeddings, document question answering, file operations, vector stores,
file-search stores, Google Cloud bucket workflows, prompt engineering, data export, and
SQLite-backed data management.

Boo is designed for federal data analysis, budget execution support, document review, knowledge
retrieval, prompt management, multimodal artificial intelligence experimentation, and controlled
local analytical data operations.

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Boo/blob/main/resources/images/boo-demo.gif)

## 🕸️ Streamlit (Web)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://boo-py.streamlit.app/)
![](https://github.com/is-leeroy-jenkins/Boo/blob/main/resources/images/Boo-streamlit.gif)
- A Python framework to build dynamic, interactive web applications.


## 🧱 Databricks

[![Boo](https://img.shields.io/badge/Databricks-Boo-FF3621?logo=databricks\&logoColor=white)](https://dbc-a0c21f80-7bb3.cloud.databricks.com/browse/folders/3169291152438882?o=7474645703081351)

* Databricks workspace repository for the Boo codebase.
* Supports collaborative development, analytics, notebook execution, and application deployment.

## 🧠 Custom LLM

[![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/boo)

Boo includes a custom compact, instruction-tuned model package intended for local or edge-oriented
experimentation. The Streamlit application itself is provider-aware and API-first, while the model
artifact remains decoupled from the repository.

The model is useful for:

* Lightweight reasoning.
* Concise instruction following.
* Summarization.
* Light code synthesis.
* RAG-agent experimentation.
* Local or edge deployments where latency and footprint matter.

## 🧰 Overview

Boo wraps provider-specific helper modules and exposes them through a single Streamlit interface.
The application uses a sidebar provider selector and a provider-filtered mode selector to present only
workflows supported by the selected provider.

Provider modules imported by the application include:

| Provider | Module      | Primary Role                                                                  |
| -------- | ----------- | ----------------------------------------------------------------------------- |
| GPT      | `gpt.py`    | OpenAI text, image, audio, embedding, file, and vector-store workflows        |
| Gemini   | `gemini.py` | Gemini text, image, audio, embedding, file-search, and Google Cloud workflows |
| Grok     | `grok.py`   | xAI Grok text, image, collection, and provider-supported workflows            |

The app initializes provider API keys, Google service keys, cloud settings, session-state defaults,
provider-wrapper aliases, chat history, prompt records, embedding tables, and imported data tables.
It then dispatches each mode through shared wrapper names such as `Chat`, `Images`, `Embeddings`,
`TTS`, `Transcription`, `Translation`, `Files`, `VectorStores`, `FileSearch`, and `CloudBuckets`.

## ✨ Features

* **Provider-aware interface** for GPT, Gemini, and Grok workflows.
* **Dynamic mode filtering** so unsupported provider/mode combinations are hidden or blocked.
* **Text generation controls** for model, reasoning, modalities, media resolution, response count,
  temperature, Top-P, Top-K, frequency penalty, presence penalty, tools, include fields, tool choice,
  Google grounding, parallel tools, URL context, allowed domains, vector stores, collections, JSON
  schema output, stop sequences, storage, streaming, background execution, and conversation state.
* **Image workflows** for generation, analysis, and editing with provider-specific model routing,
  uploaded image handling, mask support, MIME/output options, aspect ratio, quality, style,
  background, grounding, image search, tools, includes, and rendered output history.
* **Audio workflows** for text-to-speech, transcription, and translation with uploaded files,
  browser recordings, playback controls, voice options, output format, sample rate, and runtime
  inference settings.
* **Embedding mode** with provider embedding models, encoding format, dimensions, chunk size,
  overlap, chunk inspection, embedding metrics, usage tracking, and vector output display.
* **Document Q&A** with local document loading, PyMuPDF extraction, chunking, embeddings,
  `sqlite-vec` retrieval when available, and cosine-similarity fallback.
* **Files mode** for provider file upload, metadata retrieval, deletion, listing, and file-backed
  workflows where supported.
* **Vector Stores mode** for creating, retrieving, deleting, batching, uploading, and attaching files
  to provider-supported vector stores or collection-like storage.
* **File Search Stores mode** for Gemini file-search store management.
* **Google Cloud Buckets mode** for Gemini/Google Cloud bucket creation, retrieval, deletion, and upload workflows.
* **Prompt Engineering mode** backed by the local SQLite `Prompts` table.
* **Export mode** for exporting local application data and assets where configured.
* **Data Management mode** for SQLite import, browsing, CRUD operations, profiling, filtering,
  aggregation, visualization, administration, and guarded SQL queries.
* **Token usage tracking** for last call and accumulated session usage where provider responses expose usage metadata.
* **Fixed footer status bar** showing provider, mode, model, and active runtime settings.

## 🧩 Application Modes

| Mode                   | Description                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Text`                 | Provider-aware text generation with system prompts, tools, grounding, URL context, vector/collection retrieval, response schemas, and streaming. |
| `Images`               | Image generation, image analysis, and image editing through provider-specific image wrappers.                                                    |
| `Audio`                | Text-to-speech, audio transcription, audio translation, uploaded audio processing, recording, and playback.                                      |
| `Embedding`            | Text normalization, chunking, embedding generation, metrics, usage, and vector inspection.                                                       |
| `Document Q&A`         | Upload and process documents, retrieve relevant chunks, and ask document-grounded questions.                                                     |
| `Files`                | Manage provider file upload, retrieval, listing, metadata, and deletion workflows.                                                               |
| `Vector Stores`        | Create, retrieve, delete, batch, upload, and attach files to provider vector stores or collection-like storage.                                  |
| `File Search Stores`   | Manage Gemini file-search stores and upload supported files.                                                                                     |
| `Google Cloud Buckets` | Create, retrieve, delete, and upload files to Google Cloud bucket-backed workflows.                                                              |
| `Prompt Engineering`   | Manage reusable prompts in the local SQLite `Prompts` table.                                                                                     |
| `Export`               | Export local data, prompts, and application assets where configured.                                                                             |
| `Data Management`      | Import, browse, edit, profile, filter, aggregate, visualize, administer, and query SQLite data.                                                  |

## 🛠️ Requirements

| Requirement                            | Purpose                                                                            |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| Python 3.10+                           | Runtime environment                                                                |
| Streamlit                              | Web application framework                                                          |
| OpenAI Python SDK                      | GPT provider workflows                                                             |
| google-genai / Gemini SDK dependencies | Gemini provider workflows                                                          |
| xAI / Grok wrapper dependencies        | Grok provider workflows                                                            |
| pandas                                 | DataFrame operations, SQL import, table display, and export workflows              |
| numpy                                  | Vector math and cosine similarity                                                  |
| plotly.express / plotly.graph_objects  | Interactive visualizations                                                         |
| tiktoken                               | Token counting for text and embedding workflows                                    |
| sentence-transformers                  | Local document embedding model for retrieval workflows                             |
| sqlite-vec                             | Optional SQLite vector table support for Document Q&A                              |
| PyMuPDF / `fitz`                       | PDF text extraction and preview support                                            |
| openpyxl                               | Excel workbook import support through pandas                                       |
| boogr                                  | Application error handling                                                         |
| config.py                              | Provider lists, mode maps, model lists, paths, labels, help text, and API defaults |
| SQLite                                 | Local persistence for prompts, chat history, embeddings, and imported data         |


## 🔑 API Key Setup

Boo reads API and cloud configuration from `config.py`, environment variables, and Streamlit session
state. Sidebar-entered values override configuration defaults for the current session and mirror the
values into environment variables.

| Key / Setting             | Used For                                      |
| ------------------------- | --------------------------------------------- |
| `OPENAI_API_KEY`          | GPT/OpenAI API access                         |
| `GEMINI_API_KEY`          | Gemini API access                             |
| `GOOGLE_API_KEY`          | Google API access and Gemini-related services |
| `GOOGLE_CSE_ID`           | Google Custom Search integration              |
| `GOOGLEMAPS_API_KEY`      | Google Maps-related workflows                 |
| `GEOCODING_API_KEY`       | Geocoding workflows where configured          |
| `GEOAPIFY_API_KEY`        | Geoapify workflows where configured           |
| `GOOGLE_CLOUD_PROJECT_ID` | Google Cloud project routing                  |
| `GOOGLE_CLOUD_LOCATION`   | Google Cloud regional configuration           |
| `XAI_API_KEY`             | xAI Grok API access                           |

Helpful setup references:

* [OpenAI API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/openai.md)
* [Grok API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/xai.md)
* [Gemini API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/gemini.md)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/Boo.git
cd Boo
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## ⚙️ Configuration

Set the required values in `config.py`, environment variables, or the Streamlit sidebar.

Example environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CSE_ID="your-google-custom-search-id"
export GOOGLEMAPS_API_KEY="your-google-maps-api-key"
export GEOCODING_API_KEY="your-geocoding-api-key"
export GEOAPIFY_API_KEY="your-geoapify-api-key"
export GOOGLE_CLOUD_PROJECT_ID="your-google-cloud-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export XAI_API_KEY="your-xai-api-key"
```

Windows PowerShell:

```powershell
setx OPENAI_API_KEY "your-openai-api-key"
setx GEMINI_API_KEY "your-gemini-api-key"
setx GOOGLE_API_KEY "your-google-api-key"
setx GOOGLE_CSE_ID "your-google-custom-search-id"
setx GOOGLEMAPS_API_KEY "your-google-maps-api-key"
setx GEOCODING_API_KEY "your-geocoding-api-key"
setx GEOAPIFY_API_KEY "your-geoapify-api-key"
setx GOOGLE_CLOUD_PROJECT_ID "your-google-cloud-project-id"
setx GOOGLE_CLOUD_LOCATION "us-central1"
setx XAI_API_KEY "your-xai-api-key"
```

Important `config.py` objects include:

| Configuration Object                        | Purpose                                               |
| ------------------------------------------- | ----------------------------------------------------- |
| `PROVIDERS`                                 | Provider names and backing module labels              |
| `GPT_MODES` / `GEMINI_MODES` / `GROK_MODES` | Provider-specific mode lists                          |
| `MODE_CLASS_MAP` / `PROVIDER_CLASS_MAP`     | Mode-to-wrapper routing                               |
| `CLASS_MODE_MAP`                            | Provider-to-mode routing before runtime filtering     |
| `LOGO_MAP`                                  | Provider logo mapping for the sidebar                 |
| `DB_PATH`                                   | SQLite database path                                  |
| `BLUE_DIVIDER`                              | Shared divider markup                                 |
| `XML_BLOCK_PATTERN`                         | XML-like delimiter pattern used for prompt conversion |

## 🚀 Running the Streamlit Application

From the project root:

```bash
streamlit run app.py
```

Once running, the application is available at:

```text
http://localhost:8501
```

## 🧠 Provider Dispatch

Boo uses common wrapper names and provider-specific modules to keep the UI stable while switching
between GPT, Gemini, and Grok.

| Dispatch Function            | Wrapper Returned |
| ---------------------------- | ---------------- |
| `get_chat_module()`          | `Chat`           |
| `get_images_module()`        | `Images`         |
| `get_embeddings_module()`    | `Embeddings`     |
| `get_tts_module()`           | `TTS`            |
| `get_transcription_module()` | `Transcription`  |
| `get_translation_module()`   | `Translation`    |
| `get_files_module()`         | `Files`          |
| `get_vectorstores_module()`  | `VectorStores`   |
| `get_file_search_module()`   | `FileSearch`     |
| `get_cloud_buckets_module()` | `CloudBuckets`   |

The sidebar uses provider mode filtering so a provider only exposes modes that are configured and
supported by the available wrapper classes.

## 💬 Text Generation

The `Text` mode provides provider-aware chat and text generation through the selected provider's
`Chat` wrapper.

Supported control groups include:

| Control Group              | Options                                                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model Settings             | Model, reasoning, modalities, media resolution, response count                                                                                                           |
| Inference Settings         | Top-P, Top-K, temperature, frequency penalty, presence penalty                                                                                                           |
| Tools / Grounding Settings | Tools, include fields, tool choice, max tool calls, Google grounding, parallel tools, max URLs, input mode, URLs, allowed domains, vector store IDs, Grok collection IDs |
| Output / Response Settings | Max tokens, response format, store, stream, background, JSON schema name, JSON schema body, strict schema, stop sequences                                                |
| System Instructions        | Instruction editor, prompt-template loading, clear button, XML-to-Markdown conversion                                                                                    |

Text mode supports:

* GPT vector store IDs for file-search-enabled responses.
* Gemini Google Search grounding.
* Gemini file-search store names when exposed by the wrapper.
* Grok collection IDs and configured xAI collection labels.
* Conversation or single-turn input modes.
* Provider-safe JSON schema payload construction for GPT and Grok.
* Source rendering when a provider wrapper exposes grounding or file-search sources.

## 📷 Images

The `Images` mode supports provider-aware image generation, image analysis, and image editing.

Image controls include:

* Workflow mode: `Generation`, `Analysis`, or `Editing`.
* Provider-specific image model selection.
* Response count.
* Temperature, Top-P, Top-K, frequency penalty, presence penalty, and max tokens.
* Tools, include fields, tool choice, allowed domains, max tool calls, and max searches.
* Google grounding and image-search options where supported.
* Image size, quality, style, background, aspect ratio, detail, compression, MIME type, output type,
  media resolution, and response modality.
* Uploaded image support for analysis and editing.
* Optional mask upload for editing where supported.
* Rendered image output from bytes, paths, URLs, dictionaries, provider objects, or lists.

## 🎧 Audio

The `Audio` mode supports provider-aware audio workflows.

| Workflow       | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| Text-to-Speech | Generate audio output from text                                        |
| Transcribe     | Convert uploaded or recorded audio into text                           |
| Translate      | Translate uploaded or recorded audio into the selected output language |

Audio controls include:

* Task selection.
* Model selection.
* Voice selection.
* Output format.
* Language selection.
* Sample rate.
* Playback start and end time.
* Loop and autoplay.
* Temperature, Top-P, Top-K, frequency penalty, presence penalty, and max tokens.
* System prompt template support.
* Uploaded audio file or browser recording input.

## 🔢 Embeddings

The `Embedding` mode supports provider embedding workflows.

Embedding controls include:

* Embedding model selection.
* Encoding format selection.
* Dimension selection.
* Chunk-size control.
* Chunk-overlap control.
* Text input or file input where configured.
* Text normalization and chunking.
* Embedding vector generation.
* Metrics display.
* Chunk inspection.
* Usage metadata display.
* Data editor rendering for embedding vectors.

## 📓 Document Q&A

The `Document Q&A` mode supports retrieval-augmented document answering.

Supported document behavior includes:

* Uploading document bytes into Streamlit session state.
* Extracting PDF text with PyMuPDF where available.
* Defensive decoding for text-like files.
* Chunking document text.
* Generating local embeddings with `sentence-transformers`.
* Creating a `sqlite-vec` virtual table when available.
* Falling back to in-memory cosine similarity when vector-table retrieval is unavailable.
* Building document-grounded prompts from retrieved excerpts.
* Routing document prompts through the selected provider's `Chat` wrapper.

The document prompt instructs the model to answer from retrieved excerpts and state when the excerpts
do not contain enough information.

## 📚 Files

The `Files` mode exposes provider file workflows through the selected provider's `Files` wrapper.

Common workflows include:

* File upload.
* File listing.
* File metadata retrieval.
* File deletion.
* File ID tracking.
* Provider-compatible temporary file saving.
* File-backed prompt workflows where supported.

## 🏛️ Vector Stores

The `Vector Stores` mode exposes provider vector-store or collection-like storage workflows.

Supported workflows include:

| Workflow        | Description                                               |
| --------------- | --------------------------------------------------------- |
| Create          | Create a vector store or collection                       |
| Retrieve        | Retrieve store metadata                                   |
| Delete          | Delete a selected store                                   |
| Batch           | Attach multiple file IDs to a selected store              |
| Upload + Attach | Upload a supported file and attach it to a selected store |

Supported upload types include:

* `pdf`
* `txt`
* `md`
* `docx`
* `png`
* `jpg`
* `jpeg`
* `json`
* `csv`

## 📦 File Search Stores

The `File Search Stores` mode supports Gemini file-search store management through the `FileSearch`
wrapper. The mode is explicitly limited to Gemini and stops with a warning when another provider is
selected.

Supported workflows include:

| Workflow | Description                                  |
| -------- | -------------------------------------------- |
| Create   | Create a new file-search store               |
| Retrieve | Retrieve store metadata                      |
| Delete   | Delete a selected file-search store          |
| Upload   | Upload supported files to the selected store |

Supported upload types include:

* `pdf`
* `txt`
* `md`
* `docx`
* `png`
* `jpg`
* `jpeg`

## 🧊 Google Cloud Buckets

The `Google Cloud Buckets` mode supports Google Cloud bucket management through the `CloudBuckets`
wrapper.

Supported workflows include:

| Workflow | Description                                                        |
| -------- | ------------------------------------------------------------------ |
| Create   | Create a new cloud bucket                                          |
| Retrieve | Retrieve cloud bucket metadata                                     |
| Delete   | Delete a selected cloud bucket                                     |
| Upload   | Upload supported files through the available wrapper upload method |

## 📝 Prompt Engineering

The `Prompt Engineering` mode manages reusable prompts stored in the local SQLite `Prompts` table.

Prompt records include:

| Field       | Description                                |
| ----------- | ------------------------------------------ |
| `PromptsId` | Primary key                                |
| `Caption`   | Display caption used by template selectors |
| `Name`      | Prompt name                                |
| `Text`      | Prompt body                                |
| `Version`   | Prompt version                             |
| `ID`        | External or user-defined identifier        |

Prompt Engineering supports:

* Prompt search.
* Prompt sorting.
* Prompt pagination.
* Prompt selection.
* Prompt editing.
* Prompt insertion.
* Prompt update.
* Prompt deletion.
* Cascading selected prompts into system instructions where configured.

## 📤 Export

The `Export` mode supports local export workflows where configured, including prompt, chat, data,
or generated asset export paths.

## 🏛️ Data Management

The `Data Management` mode provides a SQLite administration and exploration interface.

Tabs include:

| Tab       | Purpose                                                                                 |
| --------- | --------------------------------------------------------------------------------------- |
| Import    | Import Excel workbook sheets into SQLite tables                                         |
| Browse    | Browse existing SQLite tables                                                           |
| CRUD      | Insert, update, and delete table rows                                                   |
| Explore   | Page through table records                                                              |
| Filter    | Filter rows by column text containment or advanced conditions                           |
| Aggregate | Run numeric aggregations                                                                |
| Visualize | Render charts from table data                                                           |
| Admin     | Profile data, drop tables, create indexes, create tables, view schema, and alter tables |
| SQL       | Execute guarded read-only SQL and download query results as CSV                         |

Visualization options include:

* Histogram.
* Bar chart.
* Line chart.
* Scatter plot.
* Box plot.
* Pie chart.
* Correlation heatmap.

SQL execution is guarded by a read-only validator that allows `SELECT`, `WITH`, `EXPLAIN`, and
read-oriented `PRAGMA` statements while blocking destructive operations such as `INSERT`, `UPDATE`,
`DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`, `VACUUM`, `REPLACE`, and `TRIGGER`.

## 🧩 Design and Architecture

Boo uses a provider-aware layered Streamlit architecture:

| Layer                       | Description                                                                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UI Layer                    | Streamlit sidebar, expanders, tabs, chat messages, uploaders, data editors, audio controls, and charts                                                                                |
| Provider Layer              | GPT, Gemini, and Grok provider selection and runtime mode filtering                                                                                                                   |
| Mode Layer                  | Text, Images, Audio, Embedding, Document Q&A, Files, Vector Stores, File Search Stores, Buckets, Prompt Engineering, Export, and Data Management blocks                               |
| Wrapper Layer               | Common wrapper names dispatched to provider modules: `Chat`, `Images`, `Embeddings`, `TTS`, `Transcription`, `Translation`, `Files`, `VectorStores`, `FileSearch`, and `CloudBuckets` |
| Runtime Configuration Layer | API-key/session-state configuration for OpenAI, Gemini, Google, Google Custom Search, Google Maps, Google Cloud, Geoapify, and xAI                                                    |
| Persistence Layer           | SQLite database under `stores/sqlite`                                                                                                                                                 |
| Retrieval Layer             | PyMuPDF extraction, `sentence-transformers`, `sqlite-vec`, chunking, and cosine similarity fallback                                                                                   |
| Utility Layer               | Token counting, file saving, storage-object normalization, markdown/XML conversion, usage tracking, error handling, and display-safe table rendering                                  |

Architecture diagram:

```text
┌──────────────────────────────────────────────┐
│                 Boo Streamlit App            │
│                                              │
│  Provider: GPT | Gemini | Grok               │
│                                              │
│  Modes: Text | Images | Audio | Embedding    │
│  Document Q&A | Files | Vector Stores        │
│  File Search Stores | Google Cloud Buckets   │
│  Prompt Engineering | Export | Data Mgmt     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│            Provider Dispatch Layer           │
│                                              │
│  gpt.py | gemini.py | grok.py                │
│  Chat | Images | Embeddings | TTS            │
│  Transcription | Translation | Files         │
│  VectorStores | FileSearch | CloudBuckets    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Configuration + State            │
│                                              │
│  config.py | environment variables           │
│  Streamlit session state                     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              SQLite Persistence              │
│                                              │
│  chat_history | embeddings | Prompts         │
│  imported data tables                        │
└──────────────────────────────────────────────┘
```

## 💻 Capabilities

| Capability           | Description                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| Provider Switching   | Select GPT, Gemini, or Grok from the sidebar                                                            |
| Mode Filtering       | Shows modes supported by the selected provider and configured wrappers                                  |
| Text Generation      | Provider-aware chat and prompt response generation                                                      |
| Google Grounding     | Optional Gemini Google Search grounding in Text mode                                                    |
| URL Context          | URL inputs can be added to Text mode context                                                            |
| Vector Retrieval     | GPT vector store IDs and Grok collection IDs can be routed through Text mode                            |
| System Prompts       | System-instruction text areas with template loading and XML/Markdown conversion                         |
| JSON Schema Output   | GPT/Grok schema payload construction from response-format controls                                      |
| Image Generation     | Prompt-to-image generation through provider image wrappers                                              |
| Image Analysis       | Uploaded image analysis using provider vision/image models                                              |
| Image Editing        | Uploaded image editing with optional masks where supported                                              |
| Audio Transcription  | Uploaded or recorded audio converted to text                                                            |
| Audio Translation    | Uploaded or recorded audio translated into the selected language                                        |
| Text-to-Speech       | Text converted into generated audio                                                                     |
| Embeddings           | Text chunking and vector generation                                                                     |
| Document Q&A         | Retrieval-augmented document question answering                                                         |
| Files API            | Provider file upload and metadata workflows                                                             |
| Vector Stores        | Store creation, retrieval, deletion, batch attachment, and upload workflows                             |
| File Search Stores   | Gemini file-search store creation, retrieval, deletion, and upload                                      |
| Google Cloud Buckets | Cloud bucket creation, retrieval, deletion, and upload                                                  |
| Prompt Engineering   | SQLite-backed reusable prompt management                                                                |
| Data Export          | Export workflows for local data and application assets                                                  |
| Data Management      | SQLite import, browse, CRUD, profile, filter, aggregate, visualize, administer, and SQL query workflows |
| Token Usage          | Last-call and accumulated token usage tracking where response metadata is available                     |

## 📁 File Organization

| File / Folder                                                                             | Description                                                                          |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [`app.py`](https://github.com/is-leeroy-jenkins/Boo/blob/main/app.py)                     | Main Streamlit application                                                           |
| [`gpt.py`](https://github.com/is-leeroy-jenkins/Boo/blob/main/gpt.py)                     | GPT/OpenAI wrapper classes                                                           |
| [`gemini.py`](https://github.com/is-leeroy-jenkins/Boo/blob/main/gemini.py)               | Gemini wrapper classes                                                               |
| [`grok.py`](https://github.com/is-leeroy-jenkins/Boo/blob/main/grok.py)                   | Grok/xAI wrapper classes                                                             |
| [`config.py`](https://github.com/is-leeroy-jenkins/Boo/blob/main/config.py)               | Constants, paths, provider maps, model lists, API defaults, UI labels, and help text |
| [`requirements.txt`](https://github.com/is-leeroy-jenkins/Boo/blob/main/requirements.txt) | Python package requirements                                                          |
| `stores/sqlite/Data.db`                                                                   | Local SQLite database for prompts, chat history, embeddings, and imported data       |
| `resources/images`                                                                        | Project images, logos, and README assets                                             |
| `resources/setup`                                                                         | API key and setup documentation                                                      |

## 🧪 Example Usage

### Text Generation

```python
from gpt import Chat

chat = Chat()
response = chat.generate_text(
    prompt="Explain how random forests reduce overfitting.",
    model="gpt-5-mini"
)

print(response)
```

### Gemini Text Generation

```python
from gemini import Chat

chat = Chat()
response = chat.generate_text(
    prompt="Summarize the purpose of retrieval augmented generation.",
    model="gemini-2.5-flash"
)

print(response)
```

### Grok Text Generation

```python
from grok import Chat

chat = Chat()
response = chat.generate_text(
    prompt="Create three concise bullets about semantic search.",
    model="grok-4"
)

print(response)
```

### Embeddings

```python
from gpt import Embeddings

embedding = Embeddings()
vectors = embedding.create(
    text=["Federal budget execution requires accurate obligations tracking."],
    model="text-embedding-3-small"
)

print(vectors)
```

### Image Generation

```python
from gpt import Images

images = Images()
result = images.generate(
    prompt="A clean technical diagram of a retrieval augmented generation pipeline.",
    model="gpt-image-1"
)

print(result)
```

### Audio Transcription

```python
from gpt import Transcription

transcriber = Transcription()
text = transcriber.transcribe("audio/meeting.m4a")

print(text)
```

### SQLite Prompt Query

```python
import sqlite3

with sqlite3.connect("stores/sqlite/Data.db") as conn:
    rows = conn.execute(
        "SELECT PromptsId, Caption, Name, Version FROM Prompts ORDER BY PromptsId DESC"
    ).fetchall()

print(rows)
```

## 🧮 Runtime Parameters

Boo exposes the following active runtime parameters across modes:

| Parameter         | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| Provider          | Selected provider: GPT, Gemini, or Grok                     |
| Mode              | Current workflow mode                                       |
| Model             | Provider-specific model selection                           |
| Temperature       | Sampling randomness                                         |
| Top-P             | Nucleus sampling probability                                |
| Top-K             | Token candidate limit where supported                       |
| Frequency Penalty | Penalizes repeated token frequency                          |
| Presence Penalty  | Penalizes already-present tokens                            |
| Max Tokens        | Maximum response length                                     |
| Tool Choice       | Provider tool-selection behavior                            |
| Tools             | Provider-supported tools such as file search or grounding   |
| Include           | Provider-supported response include fields                  |
| Store             | Whether supported providers should store responses          |
| Stream            | Whether supported providers should stream responses         |
| Background        | Whether supported providers should run background responses |
| Token Usage       | Last-call and accumulated prompt/completion/total tokens    |

## 🧰 Troubleshooting

| Issue                             | Resolution                                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Provider mode is missing          | Confirm the selected provider exposes the required wrapper class and that `config.py` maps the mode correctly. |
| API request fails                 | Confirm the provider API key is present in the sidebar, environment, or `config.py`.                           |
| GPT file search fails             | Confirm vector store IDs are valid and comma-delimited.                                                        |
| Gemini grounding is disabled      | Select the Gemini provider; Google grounding is only enabled for Gemini Text mode.                             |
| Grok retrieval does not run       | Confirm collection IDs or configured collection labels are available for Grok.                                 |
| Image generation fails            | Confirm the selected provider supports the selected image workflow and model.                                  |
| Audio task fails                  | Confirm the selected provider exposes `TTS`, `Transcription`, or `Translation` wrappers for the selected task. |
| Document Q&A returns weak answers | Confirm documents are loaded, extractable text exists, and document chunks are being retrieved.                |
| sqlite-vec unavailable            | Let the app fall back to cosine similarity or install/configure `sqlite-vec`.                                  |
| PDF extraction fails              | Confirm PyMuPDF is installed and the file is a valid PDF.                                                      |
| SQL query blocked                 | Use a read-only `SELECT`, `WITH`, `EXPLAIN`, or safe `PRAGMA` query.                                           |
| DataFrame rendering fails         | The app includes a display-safe fallback renderer for problematic SQLite/PyArrow values.                       |

## Documentation

The Boo documentation site is published with MkDocs Material and GitHub Pages.

[Boo Documentation](https://is-leeroy-jenkins.github.io/Boo/)

## 🚀 Application Badges

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](#-requirements)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](#-running-the-streamlit-application)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-412991?logo=openai\&logoColor=white)](#-api-key-setup)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google\&logoColor=white)](#-api-key-setup)
[![Grok](https://img.shields.io/badge/xAI-Grok-111111?logo=x\&logoColor=white)](#-api-key-setup)
[![SQLite](https://img.shields.io/badge/SQLite-Data%20Store-003B57?logo=sqlite\&logoColor=white)](#-data-management)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Buckets-4285F4?logo=googlecloud\&logoColor=white)](#-google-cloud-buckets)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20Host-FFD21E?logo=huggingface\&logoColor=black)](#-custom-llm)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)


## 📝 License

Boo is published under the [MIT License](https://github.com/is-leeroy-jenkins/Boo/blob/main/LICENSE).

