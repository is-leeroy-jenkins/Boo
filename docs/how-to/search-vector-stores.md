### `docs/how-to/search-vector-stores.md`
```markdown
# 🗂️ Search vector stores

```python
"""Search vector stores via file_search tool."""
from boo import Chat

chat = Chat()
# chat.vector_stores = {"Appropriations": "...", "Guidance": "..."}  # configured once
print(chat.search_files("Top FY2024 themes in OCO funding."))
