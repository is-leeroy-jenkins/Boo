###### Boo
<img src="https://github.com/is-leeroy-jenkins/Boo/blob/main/resources/img/github/Boo.gif" width="1400" height="250"/>



A modular Python framework for building, deploying, and managing AI-powered assistants
tailored for federal data analysis, budget execution, and data science. It integrates OpenAI's GPT
models with multimodal support for text, image, audio, and file analysis. Designed with
extensibility and federal applications in mind, it enables secure, scalable, and intelligent
automation of analytical tasks.

### ⚙️ Features

- Unified AI Framework: Integrates OpenAI APIs for text, image, audio, file analysis, transcription,
  and translation.
- Multimodal Capabilities: Supports text generation, image creation, image analysis, and document
  summarization.
- Assistant Classes:
- Chat: Generic multimodal interaction with GPT.
- Assistant: General-purpose assistant framework.
- Boo: Budget Execution Assistant.
- Bro: Data Science & Programming Assistant.
- Custom Fine-Tuned Models: Uses proprietary fine-tuned models for different domains (e.g., bro-gpt,
  bubba-gpt).
- Vector Store Integration: Embedded vector store lookups for domain-specific knowledge retrieval.
- Web & File Search: Built-in support for semantic document and web search.

### 🧭 Table of Contents

- 🧰 [Overview](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-overview)
- ✨ [Features](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-features)
- ⚡ [Quickstart](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-quickstart)
- 🔧 [Configuration](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-configuration)
- 🧩 [Design & Architecture](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-design--architecture)
- 🧪 Usage Examples
  - 📝 [Text generation](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-text-generation)
  - 🌐 [Web Search](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-web-search-responses) (Responses)
  - 📄 [Document Summarization](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-summarize-a-document-file-grounded) (file-grounded)
  - 🗂️ [File search](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#%EF%B8%8F-file-search-vector-stores) (vector stores)
  - 👀 [Vision](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-vision-analyze-an-image): analyze an image
  - 🖼️ Images: [generate](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-image-generation) / [edit](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#%EF%B8%8F-images-generate--edit)
  - 🧬 [Embeddings](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-embeddings)
  - 🔊 [Text-to-Speech](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#-text-to-speech-tts) (TTS)
  - 🎙️ [Transcription/Translation](https://github.com/is-leeroy-jenkins/Boo?tab=readme-ov-file#%EF%B8%8F-transcription--translation-whisper) (Whisper)
- 📄 License

### 📦 Installation

#### 1. Clone the Repository


```
bash
git clone https://github.com/your-username/Boo.git
cd Boo
```

#### 2. Create and Activate a Virtual Environment

```
bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```
bash
pip install -r requirements.txt
```




### ⚙️ Core Classes

- `AI`: Base class that provides shared API setup, keys, and model configurations.
- `Chat`, `Assistant`, `Bubba`, `Bro`: Extend `AI` to provide domain-specific implementations.
- `Models`, `Header`, `EndPoint`: Configuration utilities for model selection, headers, and
  endpoints.
- `Prompt`, `Message`, `Response`, `File`, `Reasoning`: Pydantic models for structured data
  exchange.



### 🧠 Capabilities

| Capability        | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| Text Generation   | GPT-powered completions, instructions, and prompts                          |
| Image Generation  | DALL·E 3-based prompt-to-image generation                                   |
| Image Analysis    | Multimodal image+text inference using vision models                         |
| Document Summary  | File upload + prompt-driven summarization via OpenAI file API               |
| Web Search        | Integrated API call to perform web-based lookups                            |
| File Search       | Vector store lookup with prompt-based semantic matching                     |
| Model Registry    | Fine-tuned and base model tracking for GPT-4, GPT-4o, and others            |
| Assistant List    | Query and list named assistant objects from the OpenAI API                  |



### 🛠️ Requirements

- Python 3.10+
- OpenAI Python SDK
- Pydantic
- Numpy, Pandas
- Tiktoken
- Requests
- Custom dependencies: `boogr`, `static`, `guro`

### 📁 File Organization

- [boo](https://github.com/is-leeroy-jenkins/Boo/blob/main/boo.py) – Main application framework
- [models](https://github.com/is-leeroy-jenkins/Boo/blob/main/models.py) – Models used for
  structured output
- [boogr](https://github.com/is-leeroy-jenkins/Boo/blob/main/boogr.py) – GUI and error dialogs
- [guro](https://github.com/is-leeroy-jenkins/Boo/blob/main/guro.py) – Prompt context utilities
- [foo](https://github.com/is-leeroy-jenkins/Boo/blob/main/foo.py) – ML/AI Layer
- [mathy](https://github.com/is-leeroy-jenkins/Boo/tree/main/mathy) - Machine Learning models
- [dbops](https://github.com/is-leeroy-jenkins/Boo/tree/main/dbops.py) - Machine Learning models

### 🔐 Environment Variables

Set the following in your environment or `.env` file:

```
bash
OPENAI_API_KEY=<your_api_key>
```



### 🧰 Overview

Boo wraps the latest **OpenAI Python SDK** with a thin class hierarchy:

- **GPT (base)** – holds the single `OpenAI` client, env config, and shared helpers.
- **Chat / Assistant / Bro / Bubba** – opinionated text assistants using the **Responses API**.
- **Image / LargeImage** – image generation and vision analysis.
- **Embedding** – small/large/legacy embeddings with consistent return types.
- **TTS** – text-to-speech helpers (streaming to file).
- **Transcription / Translation** – Whisper-powered speech-to-text (+ translate).
- **Vector Store helpers** – list files, search via `file_search` tool, merge results.



### ✨ Features

- **Responses-first**: consistent `input=[{role, content:[{type:...}]}]` builders.
- **One client**: a single `OpenAI` instance per process for reliability and testability.
- **Typed containers**: Pydantic models for prompts/messages with pass-through `__init__`.
- **Vector search**: easy `file_search` tools + helpers to fetch file IDs from vector stores.
- **Audio**: TTS (stream-to-file), ASR (transcribe), and translate via Whisper.
- **Vision & Images**: multimodal analysis (4o/4o-mini) and image generation (DALL·E 3).
- **Guardrails**: tiny helpers that prevent recurring mistakes (e.g., `inputs` vs `input`,
  content `type` keys, size strings, binary file handling).
- **Uniform errors**: `GptError` + `ErrorDialog` with `module/cause/method` metadata.



### ⚡ Quickstart

1) **Install**

```
pip install openai pydantic
```

2) **Configure**

```
# Power your client via environment
export OPENAI_API_KEY="sk-..."         # macOS/Linux
setx OPENAI_API_KEY "sk-..."           # Windows
```

3) **Hello Boo**

```
from boo import Chat

chat = Chat()
print(chat.generate_text("Say hello in one short sentence."))
```



### 🔧 Configuration

- **Environment**
  - `OPENAI_API_KEY` (required)
- **Models**
  - Text/Responses: e.g., `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`
  - Images: `dall-e-3`
  - Embeddings: `text-embedding-3-small`, `text-embedding-3-large`
  - TTS: `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`
  - ASR/Translate: `whisper-1`
- **Vector stores (optional)**
  - Configure your store IDs once; Boo converts to lists when calling tools.



### 🧩 Design & Architecture

- **Single client**: `OpenAI(api_key=...)` is created in `GPT.__init__()` and reused everywhere.
- **Schema helpers**: tiny, battle-tested builders ensure payloads are valid for the Responses API:
  - input text only
  - text + file
  - text + image
- **No duplicate methods**: each capability has one canonical implementation per class.
- **Type-safe Pydantic**: BaseModel subclasses do **not** override `__init__` except with
  pass-through `def __init__(self, **data): super().__init__(**data)`.
- **Consistent naming**: `vector_stores` (with underscore), `response_format`, `output_text`.



### 🔤 Text Generation

- Generate high-quality responses using OpenAI's GPT models.
- Supports parameter tuning (temperature, top_p, frequency penalties).
- Ideal for summarization, explanations, and knowledge retrieval.

```
python
from boo import Bro

bro = Bro( )
response = bro.generate_text( "Explain how random forests handle overfitting." )
print( response )
```

### 🎨 Image Generation

- Convert natural language prompts into images using DALL·E 3.
- Specify resolution and rendering quality options.
- Useful for creating visual illustrations and conceptual diagrams.

```
python

image_url = bro.generate_image("A conceptual illustration of quantum computing in federal AI")
print(f"Image URL: {image_url}")
```

### 🖼️ Image Analysis

- Analyze visual content by combining image and text prompts.
- Extract meaning, context, or structure from images.
- Leverages GPT-4o’s vision capabilities for advanced perception.

```
python

url = "https://example.com/sample-image.png"
response = bro.analyze_image("Describe the primary elements in this image", url)
print(response)
```

### 📄 Document Summarization

- Upload and process document files directly into the assistant.
- Use prompts to extract insights or summarize content.
- Supports PDFs, DOCX, and other file formats via OpenAI's file API.

```
python

file_path = "data/federal_strategy.pdf"
summary = bro.summarize_document(
  prompt = "Summarize key national cybersecurity strategies.",
  path = file_path
)
print( summary )
```

### 🔍 File Search with Vector Stores

- Embed and store documents in vector stores for semantic search.
- Retrieve contextually relevant content using natural language queries.
- Ideal for knowledge base querying and document Q&A systems.

```
python

result = bro.search_files("Legislation related to environmental impact funding")
print(result)
```

### 🔎 File & Web Search

- Performs semantic search over domain-specific document embeddings to retrieve relevant content.
- **File Search**: Query vector-embedded files using `vector_store_ids`.
- **Web Search**: Real-time information retrieval using GPT web search integration.

```
python

result = bro.search_files("Legislation related to environmental impact funding")
print(result)
```

### 🌐 Web Search (Real-Time Querying)

- Perform web lookups in real time via OpenAI’s web-enabled models.
- Extract current events, news, and regulatory updates.
- No scraping required—returns model-interpreted summaries.

```
python

insights = bro.search_web("Current status of the Federal AI Bill 2025")
print(insights)
```

### 🧾 Prompt & Message Structuring

- Build structured prompt schemas using Pydantic models.
- Define instructions, context, output goals, and data sources.
- Promotes reusable, interpretable prompt engineering.

```
python

from boo import Prompt
p = Prompt(
    instruction="Create a budget summary",
    context="Federal Defense Budget FY25",
    output_indicator="summary",
    input_data="defense_budget_raw.csv"
)
print(p.model_dump())

```

### ⚙️ API Endpoint Access

- Centralized access to OpenAI API endpoints.
- Includes endpoints for completions, images, speech, and files.
- Facilitates debugging and manual request construction.

```
python

from boo import EndPoint
api = EndPoint( )
print( api.get_data( ) ) 
```

### 🤖 Assistant Management

- Fetches and lists OpenAI assistants created or used within the system, enabling assistant
  lifecycle management.
- Chat: General multimodal chat
- Assistant: Generic AI assistant
- Bubba: Budget Execution Analyst
- Bro: Programming & Data Science Analyst

```
python

from boo import Assistant
assistant = Assistant()
assistants = assistant.get_list()
print("Available Assistants:", assistants)
```

> The snippets below show idiomatic Boo usage. They assume `chat = Chat()`, `img = Image()`,
> etc., and an `OPENAI_API_KEY` is present in your environment.

### 📝 Text generation

    ```
    from boo import Chat

    chat = Chat()
    out = chat.generate_text("Give me three bullet points on strict typing in Python.")
    print(out)
    ```

### 🌐 Web search (Responses)

    ```
    from boo import Chat

    chat = Chat()
    prompt = "Latest trends in Retrieval Augmented Generation. 3 bullets, 1 reference each."
    out = chat.search_web(prompt)  # internally uses web_search_options
    print(out)
    ```

### 📄 Summarize a document (file-grounded)

    ```
    from boo import Chat

    chat = Chat()
    out = chat.summarize_document(
        prompt="Summarize the document with a 5-bullet executive brief.",
        path="docs/paper.pdf"
    )
    print(out)
    ```

### 🗂️ File search (vector stores)

    ```
    from boo import Chat

    chat = Chat()
    # Assumes chat.vector_stores is configured with { "Appropriations": "...", "Guidance": "..." }
    out = chat.search_files("What are the major themes around FY2024 OCO funding?")
    print(out)
    ```

### 👀 Vision: analyze an image

    ```
    from boo import Image

    img = Image()
    out = img.analyze(
        text="Describe the chart and call out any anomalies in one paragraph.",
        path="https://example.com/plot.png"
    )
    print(out)
    ```

### 🖼️ Images: generate / edit

    ```
    from boo import Image

    img = Image()
    url = img.generate("A minimalist logo for 'Boo' in monochrome, vector style")
    print(url)

    # If your SDK supports edits, ensure the correct API path (images.edit vs images.edits)
    # url = img.edit("Add subtle grid background", "logo.png", size="1024x1024")
    ```

### 🧬 Embeddings

    ```
    from boo import Embedding

    emb = Embedding()
    vec = emb.create_small_embedding("Vectorize this sentence.")
    print(len(vec), "dims")
    ```

### 🔊 Text-to-Speech (TTS)

    ```
    from boo import TTS

    tts = TTS()
    outfile = tts.save_audio("Hello from Boo in a calm voice.", "out/hello.mp3")
    print("Saved:", outfile)
    ```

### 🎙️ Transcription / Translation (Whisper)

    ```
    from boo import Transcription, Translation

    asr = Transcription()
    text = asr.transcribe("audio/meeting.m4a")
    print(text)

    xlat = Translation()
    english = xlat.create("Translate this speech to English.", "audio/spanish.m4a")
    print(english)
    ```

---

## 📝 License

Boo is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Boo/blob/main/LICENSE).


