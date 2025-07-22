###### Boo
<img src="https://github.com/is-leeroy-jenkins/Boo/blob/main/resources/img/github/Boo.gif" width="1400" height="250"/>



A modular Python framework for building, deploying, and managing AI-powered assistants
tailored for federal data analysis, budget execution, and data science. It integrates OpenAI's GPT
models with multimodal support for text, image, audio, and file analysis. Designed with
extensibility and federal applications in mind, it enables secure, scalable, and intelligent
automation of analytical tasks.



## 📌 Features

- **Unified AI Framework**: Integrates OpenAI APIs for text, image, audio, file analysis,
  transcription, and translation.
- **Multimodal Capabilities**: Supports text generation, image creation, image analysis, and
  document summarization.
- **Assistant Classes**:
   - `Chat`: Generic multimodal interaction with GPT.
   - `Assistant`: General-purpose assistant framework.
   - `Bubba`: Budget Execution Assistant.
   - `Bro`: Data Science & Programming Assistant.
- **Custom Fine-Tuned Models**: Uses proprietary fine-tuned models for different domains (e.g.,
  `bro-gpt`, `bubba-gpt`).
- **Vector Store Integration**: Embedded vector store lookups for domain-specific knowledge
  retrieval.
- **Web & File Search**: Built-in support for semantic document and web search.
- **Error Handling**: Custom exceptions with UI support via `ErrorDialog`.



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




## 🧱 Structure

#### Core Classes

- `AI`: Base class that provides shared API setup, keys, and model configurations.
- `Chat`, `Assistant`, `Bubba`, `Bro`: Extend `AI` to provide domain-specific implementations.
- `Models`, `Header`, `EndPoint`: Configuration utilities for model selection, headers, and
  endpoints.
- `Prompt`, `Message`, `Response`, `File`, `Reasoning`: Pydantic models for structured data
  exchange.



#### 🧠 Capabilities

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



#### 🛠️ Requirements

- Python 3.10+
- OpenAI Python SDK
- Pydantic
- Numpy, Pandas
- Tiktoken
- Requests
- Custom dependencies: `boogr`, `static`, `guro`



#### 🔐 Environment Variables

Set the following in your environment or `.env` file:

```
bash
OPENAI_API_KEY=<your_api_key>
```

#### 🧠 Assistants Included
- Chat: General multimodal chat
- Assistant: Generic AI assistant
- Bubba: Budget Execution Analyst
- Bro: Programming & Data Science Analyst

#### 🛠 Features
- OpenAI GPT model orchestration
- Document and image analysis
- Search via vector databases
- Assistant modularization
- Full multimodal AI stack  

#### 📁 File Organization
- boo.py – Main application framework
- boogr/ – GUI and error dialogs
- guro/ – Prompt context utilities
- static/ – Static config files (roles, languages, etc.)
- mathy/ - Machine Learning models


####  🔍  Natural Language DataFrame Querying
- Allow users to ask questions about a pandas DataFrame using plain English.

```
python
bro.query_dataframe(df, "What are the top 5 agencies by total spending?")
```

#### 📊  Chart Generation from Prompts
- Generate matplotlib or plotly charts from natural language prompts.

```
python
bro.visualize(prompt="Create a bar chart of spending by department", data=df)
```

#### 🧾 PDF Parsing and Table Extraction
- Automatically detect and extract structured tables from PDF files using pdfplumber or camelot.

```
python
tables = bro.extract_tables("appropriations.pdf")

```

#### 🧠 Embedded Agent Workflows
- Enable multi-step task execution (like agentic behavior).

```
python
bro.run_task( "Summarize the document, then find related guidance policies" )
```



#### 📝 License

Boo is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Boo/blob/main/LICENSE).


