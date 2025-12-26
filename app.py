'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use, copy, modify, merge, publish,
     distribute, sublicense, and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to
'''
# Standard library
from __future__ import annotations
import base64
import sqlite3
import textwrap
from pathlib import Path
from typing import List

# External libraries
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# Local imports (gpt/boo etc)
# NOTE: gpt.py does not provide load_llm(); it does provide Chat/GPT classes.
# We import Chat and create a small adapter in this file to preserve the
# original app contract (llm.generate(prompt) and llm.tokenize(...)).
from gpt import Chat

# Constants / defaults
DB_PATH = "embeddings.db"
DEFAULT_CTX = 2048
CPU_CORES = 4

# ------------------------------------------------------------------------------

def load_llm():
    """
    Create an adapter around gpt.Chat so the rest of the app (which expects
    a `llm` with generate(prompt) and tokenize(...) methods) works without
    changing other code.
    """
    try:
        chat = Chat()  # instantiate Chat() from gpt.py (uses configured API client)
        # Provide an adapter that exposes `.generate(prompt)` and `.tokenize(...)`.
        class _Adapter:
            def __init__(self, chat_obj: Chat):
                self._chat = chat_obj

            def generate(self, prompt: str):
                """
                Call Chat.generate_text and return text. If Chat returns None or an
                unexpected shape, default to an empty string.
                """
                try:
                    # If Chat has model attribute set, use it; otherwise let Chat use default.
                    model = getattr(self._chat, "model", None)
                    # Chat.generate_text signature: (prompt, model=...)
                    resp = self._chat.generate_text(prompt, model=model) if model else self._chat.generate_text(prompt)
                    # Some client responses return the text directly; if dict, try to extract.
                    if resp is None:
                        return ""
                    return resp
                except Exception:
                    # If remote client fails, return safe fallback string.
                    return "Error: model invocation failed."

            def tokenize(self, data):
                """
                Tokenize data for crude token accounting. Prefer tiktoken (if available),
                otherwise fall back to simple whitespace split.
                Accepts bytes or str; returns a list-like token sequence.
                """
                try:
                    # Accept either bytes or str
                    if isinstance(data, bytes):
                        data = data.decode(errors="ignore")
                    # Prefer tiktoken if available
                    import tiktoken  # type: ignore
                    # Use common encoding (cl100k_base) if available; fall back if not.
                    try:
                        enc = tiktoken.get_encoding("cl100k_base")
                    except Exception:
                        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    return enc.encode(data)
                except Exception:
                    # Fallback: return list of whitespace tokens
                    if isinstance(data, bytes):
                        data = data.decode(errors="ignore")
                    return data.split()

        return _Adapter(chat)

    except Exception:
        # If creating Chat fails for any reason, return a dummy adapter so the app still runs.
        class _Dummy:
            def generate(self, prompt: str):
                return "Local llm adapter inactive — Chat() unavailable."

            def tokenize(self, data):
                if isinstance(data, bytes):
                    data = data.decode(errors="ignore")
                return data.split()

        return _Dummy()

# ------------------------------------------------------------------------------

# Utilities
# ------------------------------------------------------------------------------

def html(x: str, height: int = 0):
    """Small helper to place raw HTML via st.components.v1.html"""
    st.components.v1.html(x, height=height)

def image_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def chunk_text(text: str, size: int = 1200, overlap: int = 250) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ==============================================================================

# Sidebar (Branding + Parameters Only)
# ==============================================================================

with st.sidebar:
    logo_b64 = image_to_base64("resources/img/boo.png")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-bottom:10px;">
            <img src="data:image/png;base64,{logo_b64}" style="width:50px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("⚙️ Mind Controls")

    ctx = st.slider("Context Window", 2048, 8192, DEFAULT_CTX, 512)
    threads = st.slider("CPU Threads", 1, CPU_CORES, max(2, CPU_CORES // 2))
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 40, 10)
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.05)

    # Session controls (horizontal)
    sc1, sc2, sc3 = st.columns([1,1,1])
    with sc1:
        if st.button("Clear"):
            # Will call clear_conversation() (exists below)
            try:
                clear_conversation()
            except Exception:
                # fallback if helpers not yet defined during initial parse
                st.session_state.messages = []
    with sc2:
        if st.button("New"):
            try:
                start_new_session()
            except Exception:
                st.session_state.messages = []
                st.session_state.basic_docs = []
                st.session_state.token_usage = {"prompt": 0, "response": 0, "context_pct": 0.0}
    with sc3:
        if st.button("Clear Docs"):
            try:
                clear_document_context()
            except Exception:
                st.session_state.basic_docs = []

    # Token usage summary
    st.markdown("""**Token usage (latest)**""")
    _tu = st.session_state.get("token_usage", {"prompt":0, "response":0})
    st.write(f"Input: {_tu.get('prompt',0)}")
    st.write(f"Output: {_tu.get('response',0)}")
    st.write(f"Total: {_tu.get('prompt',0) + _tu.get('response',0)}")

# ==============================================================================

# Init
# ==============================================================================

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (chunk TEXT, vector BLOB)"
    )
    conn.commit()
    conn.close()

# Simple model embedder loader
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

ensure_db()

# >>> FIXED: use local load_llm adapter that wraps gpt.Chat <<<
llm = load_llm()
embedder = load_embedder()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are a helpful assistant who provides concise assistant optimized for instruction "
        "following, contextual comprehension, and structured reasoning."
    )

if "basic_docs" not in st.session_state:
    st.session_state.basic_docs = []

if "use_semantic" not in st.session_state:
    st.session_state.use_semantic = False

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "prompt": 0,
        "response": 0,
        "context_pct": 0.0
    }

# ==============================================================================

# Tabs
# ==============================================================================

(
    tab_system,
    tab_chat,
    tab_basic,
    tab_documents,
    tab_semantic,
    tab_export
) = st.tabs(
    [
        "System Instructions",
        "Text Generation",
        "Retrieval Augmentation",
        "Documents",
        "Semantic Search",
        "Export"
    ]
)

# ==============================================================================

# Prompt Builder
# ==============================================================================

def build_prompt(user_input: str) -> str:
    prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"

    if st.session_state.use_semantic:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT chunk, vector FROM embeddings"
            ).fetchall()

        if rows:
            q_vec = embedder.encode([user_input])[0]
            scored = [
                (chunk, cosine_sim(q_vec, np.frombuffer(vec)))
                for chunk, vec in rows
            ]
            top_chunks = [
                c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
            ]

            prompt += "<|system|>\nSemantic Context:\n"
            for c in top_chunks:
                prompt += f"- {c}\n"
            prompt += "</s>\n"

    if st.session_state.use_doc_context and st.session_state.basic_docs:
        prompt += "<|system|>\nDocument Context:\n"
        for d in st.session_state.basic_docs[:6]:
            prompt += f"- {d}\n"
        prompt += "</s>\n"

    for role, content in st.session_state.messages:
        prompt += f"<|{role}|>\n{content}\n</s>\n"

    prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
    return prompt

# ==============================================================================

# Session and document helpers
# ==============================================================================

def clear_conversation():
    """Clear the in-memory conversation messages."""
    st.session_state.messages = []

def start_new_session():
    """Reset the session: clear messages, docs, and token accounting."""
    st.session_state.messages = []
    st.session_state.basic_docs = []
    st.session_state.token_usage = {"prompt": 0, "response": 0, "context_pct": 0.0}

def clear_document_context():
    """Clear any loaded document chunks from the client-side context."""
    st.session_state.basic_docs = []

# Ensure the document-context toggle exists
if "use_doc_context" not in st.session_state:
    st.session_state.use_doc_context = True

# ==============================================================================

# System Instructions
# ==============================================================================

with tab_system:
    st.text_area(
        "System Instructions",
        value=st.session_state.system_prompt,
        key="system_prompt_area",
        height=220
    )

# ==============================================================================

# Text Generation
# ==============================================================================

with tab_chat:
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask Bro...")

    if user_input:
        # Show user message immediately
        st.session_state.messages.append(("user", user_input))

        prompt = build_prompt(user_input)

        # Basic LLM call (adapter exposes .generate)
        try:
            response = llm.generate(prompt)
            if isinstance(response, dict):
                response = response.get("text", "")
        except Exception:
            # fallback naive reply
            response = "Sorry — model error. Please check the server logs."

        st.session_state.messages.append(("assistant", response))

        # crude token accounting (tokenize if available)
        try:
            prompt_tokens = len(llm.tokenize(prompt))
            response_tokens = len(llm.tokenize(response))
        except Exception:
            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split())
        context_pct = (prompt_tokens + response_tokens) / max(1, ctx) * 100

        st.session_state.token_usage = {
            "prompt": prompt_tokens,
            "response": response_tokens,
            "context_pct": context_pct
        }

# ==============================================================================

# Retrieval Augmentation
# ==============================================================================

with tab_basic:
    uploads = st.file_uploader(
        "Upload TXT / MD / PDF",
        accept_multiple_files=True
    )

    if uploads:
        st.session_state.basic_docs.clear()
        for f in uploads:
            text = f.read().decode(errors="ignore")
            st.session_state.basic_docs.extend(chunk_text(text))
        st.success(f"{len(st.session_state.basic_docs)} chunks loaded.")

# ==============================================================================

# Documents (client-side)
# ==============================================================================

with tab_documents:
    st.header("📚 Documents (Client-side)")
    uploads = st.file_uploader(
        "Upload TXT / MD / PDF (additions append to client context)",
        accept_multiple_files=True
    )
    if uploads:
        for f in uploads:
            try:
                text = f.read().decode(errors="ignore")
            except Exception:
                # if already bytes or string
                text = f.read()
                if isinstance(text, bytes):
                    text = text.decode(errors="ignore")
            st.session_state.basic_docs.extend(chunk_text(text))
        st.success(f"{len(st.session_state.basic_docs)} chunks loaded into client context.")
    st.markdown(f"**Loaded chunks:** {len(st.session_state.basic_docs)}")
    if st.session_state.basic_docs:
        for i, d in enumerate(st.session_state.basic_docs[:10], start=1):
            st.markdown(f"- Chunk {i}: {d[:200]}...")
    if st.button("Clear document context (tab)"):
        clear_document_context()

# ==============================================================================

# Semantic Search
# ==============================================================================

with tab_semantic:
    st.session_state.use_semantic = st.checkbox(
        "Use Semantic Context in Text Generation",
        value=st.session_state.use_semantic
    )

    uploads = st.file_uploader(
        "Upload Documents for Semantic Index",
        accept_multiple_files=True
    )

    if uploads:
        chunks = []
        for f in uploads:
            chunks.extend(chunk_text(f.read().decode(errors="ignore")))

        # Store in sqlite embeddings table
        with sqlite3.connect(DB_PATH) as conn:
            for c in chunks:
                vec = embedder.encode([c])[0].astype(np.float32).tobytes()
                conn.execute(
                    "INSERT INTO embeddings (chunk, vector) VALUES (?, ?)",
                    (c, vec)
                )
            conn.commit()
        st.success(f"Indexed {len(chunks)} chunks into semantic store.")

# ==============================================================================

# Export
# ==============================================================================

with tab_export:
    st.download_button("Export conversation (txt)", "\n\n".join([c for _, c in st.session_state.messages]), file_name="conversation.txt")

# ==============================================================================

# Footer
# ==============================================================================

html(
    f"""
    <style>
        .bro-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 6px 12px;
            font-size: 0.8rem;
            background-color: rgba(20,20,20,1);
            color: #ddd;
            display: flex;
            justify-content: space-between;
            z-index: 9999;
        }}
    </style>
    <div class="bro-footer">
        <div>
            🧮 Tokens —
            Prompt: {st.session_state.token_usage.get("prompt", 0)} |
            Response: {st.session_state.token_usage.get("response", 0)} |
            Context Used: {st.session_state.token_usage.get("context_pct", 0.0):.1f}%
        </div>
        <div>
            ⚙️ ctx={ctx} · temp={temperature} · top_p={top_p} ·
            top_k={top_k} · repeat={repeat_penalty}
        </div>
    </div>
    """,
    height=40
)
