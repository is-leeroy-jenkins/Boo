###### Boo
<img src="https://github.com/is-leeroy-jenkins/Boo/blob/main/resources/img/github/Boo.gif" width="1080" height="250"/>

___

- **Boo** is a modular Python framework for building, deploying, and managing AI-powered assistants
tailored for federal data analysis, budget execution, and data science. It integrates OpenAI's GPT
models with multimodal support for text, image, audio, and file analysis. Designed with
extensibility and federal applications in mind, it enables secure, scalable, and intelligent
automation of analytical tasks.

---

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

---

## 🧱 Structure

### Core Classes

- `AI`: Base class that provides shared API setup, keys, and model configurations.
- `Chat`, `Assistant`, `Bubba`, `Bro`: Extend `AI` to provide domain-specific implementations.
- `Models`, `Header`, `EndPoint`: Configuration utilities for model selection, headers, and
  endpoints.
- `Prompt`, `Message`, `Response`, `File`, `Reasoning`: Pydantic models for structured data
  exchange.

---

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

---

## 🛠️ Requirements

- Python 3.10+
- OpenAI Python SDK
- Pydantic
- Numpy, Pandas
- Tiktoken
- Requests
- Custom dependencies: `boogr`, `static`, `guro`

---

## 🔐 Environment Variables

Set the following in your environment or `.env` file:

```bash
OPENAI_API_KEY=<your_api_key>

  
## 📝 License

Boo is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Boo/blob/main/LICENSE).


