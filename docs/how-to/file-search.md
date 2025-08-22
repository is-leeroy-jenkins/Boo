
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