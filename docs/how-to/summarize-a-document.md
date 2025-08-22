# 📄 Summarize a document

This recipe uploads a local file and summarizes it using the Responses API.

```python
"""Summarize a PDF with Boo."""
from boo import Chat

chat = Chat()
summary = chat.summarize_document(
    prompt="Summarize with a 5-bullet executive brief.",
    path="docs/report.pdf",
)
print(summary)
