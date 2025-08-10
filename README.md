###### Boo
<img src="https://github.com/is-leeroy-jenkins/Boo/blob/main/resources/img/github/Boo.gif" width="1400" height="250"/>



A modular Python framework for building, deploying, and managing AI-powered assistants
tailored for federal data analysis, budget execution, and data science. It integrates OpenAI's GPT
models with multimodal support for text, image, audio, and file analysis. Designed with
extensibility and federal applications in mind, it enables secure, scalable, and intelligent
automation of analytical tasks.

## ⚙️ Features

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



## 📦 Installation

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




## ⚙️ Structure

#### Core Classes

- `AI`: Base class that provides shared API setup, keys, and model configurations.
- `Chat`, `Assistant`, `Bubba`, `Bro`: Extend `AI` to provide domain-specific implementations.
- `Models`, `Header`, `EndPoint`: Configuration utilities for model selection, headers, and
  endpoints.
- `Prompt`, `Message`, `Response`, `File`, `Reasoning`: Pydantic models for structured data
  exchange.



## 🧠 Capabilities

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



## 🛠️ Requirements

- Python 3.10+
- OpenAI Python SDK
- Pydantic
- Numpy, Pandas
- Tiktoken
- Requests
- Custom dependencies: `boogr`, `static`, `guro`



## 🔐 Environment Variables

Set the following in your environment or `.env` file:

```
bash
OPENAI_API_KEY=<your_api_key>
```


## 🔤 Text Generation

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

## 🎨 Image Generation

- Convert natural language prompts into images using DALL·E 3.
- Specify resolution and rendering quality options.
- Useful for creating visual illustrations and conceptual diagrams.

```
python

image_url = bro.generate_image("A conceptual illustration of quantum computing in federal AI")
print(f"Image URL: {image_url}")
```

## 🖼️ Image Analysis

- Analyze visual content by combining image and text prompts.
- Extract meaning, context, or structure from images.
- Leverages GPT-4o’s vision capabilities for advanced perception.

```
python

url = "https://example.com/sample-image.png"
response = bro.analyze_image("Describe the primary elements in this image", url)
print(response)
```

## 📄 Document Summarization

- Upload and process document files directly into the assistant.
- Use prompts to extract insights or summarize content.
- Supports PDFs, DOCX, and other file formats via OpenAI's file API.

```
python

file_path = "data/federal_strategy.pdf"
summary = bro.summmarize_document(
  prompt = "Summarize key national cybersecurity strategies.",
  path = file_path
)
print( summary )
```

## 🔍 File Search with Vector Stores

- Embed and store documents in vector stores for semantic search.
- Retrieve contextually relevant content using natural language queries.
- Ideal for knowledge base querying and document Q&A systems.

```
python

result = bro.search_files("Legislation related to environmental impact funding")
print(result)
```

## 🔎 File & Web Search
- Performs semantic search over domain-specific document embeddings to retrieve relevant content.
- **File Search**: Query vector-embedded files using `vector_store_ids`.
- **Web Search**: Real-time information retrieval using GPT web search integration.

```
python

result = bro.search_files("Legislation related to environmental impact funding")
print(result)
```

## 🌐 Web Search (Real-Time Querying)

- Perform web lookups in real time via OpenAI’s web-enabled models.
- Extract current events, news, and regulatory updates.
- No scraping required—returns model-interpreted summaries.

```
python

insights = bro.search_web("Current status of the Federal AI Bill 2025")
print(insights)
```

## 🧾 Prompt & Message Structuring

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

## ⚙️ API Endpoint Access

- Centralized access to OpenAI API endpoints.
- Includes endpoints for completions, images, speech, and files.
- Facilitates debugging and manual request construction.

```
python

from boo import EndPoint
api = EndPoint( )
print( api.get_data( ) ) 
```

## 🤖 Assistant Management
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

## 📁 File Organization
- boo.py – Main application framework
- [boogr](https://github.com/is-leeroy-jenkins/Boo/blob/main/src/boogr.py)/ – GUI and error dialogs
- [guro](https://github.com/is-leeroy-jenkins/Boo/blob/main/src/guro.py)/ – Prompt context utilities
- [boo](https://github.com/is-leeroy-jenkins/Boo/blob/main/src/boo.py) – ML/AI Layer
- [mathy](https://github.com/is-leeroy-jenkins/Boo/tree/main/src/mathy)/ - Machine Learning models



## 📝 License

Boo is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Boo/blob/main/LICENSE).


