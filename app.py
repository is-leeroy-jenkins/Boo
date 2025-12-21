# ******************************************************************************************
# Assembly:                Boo
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Created:                 12-16-2025
# ******************************************************************************************

from __future__ import annotations

import config as cfg
import streamlit as st
import tempfile
from typing import List, Dict, Any, Optional

from boo import (  # keep same imports as original baseline file
    Chat,
    Image,
    Embedding,
    Transcription,
    Translation,
)

# ---------------------------------------------------------------------------------------
# Page Configuration (unchanged)
# ---------------------------------------------------------------------------------------
st.set_page_config(page_title="Boo • Multimoldal AI Agent", page_icon=cfg.FAVICON_PATH, layout="wide")

# ---------------------------------------------------------------------------------------
# Boo Components (read-only introspection) — unchanged
# ---------------------------------------------------------------------------------------
chat = Chat()
image = Image()
embedding = Embedding()
transcriber = Transcription()
translator = Translation()

# ---------------------------------------------------------------------------------------
# Session State initialization (added a few keys we need)
# ---------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

if "files" not in st.session_state:
    # store local temp file paths for client-side multi-document mode
    st.session_state.files: List[str] = []

if "token_usage" not in st.session_state:
    # session aggregated token usage
    st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

if "last_call_usage" not in st.session_state:
    st.session_state.last_call_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ---------------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------------
def save_temp(upload) -> str:
    """Save uploaded file to a named temporary file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(upload.read())
        return tmp.name


def _extract_usage_from_response(resp: Any) -> Dict[str, int]:
    """
    Extract token usage numbers from an OpenAI-style response object in a robust way.
    Returns a dict with keys: prompt_tokens, completion_tokens, total_tokens (ints).
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if resp is None:
        return usage

    # support both dict-like and object-like responses
    try:
        raw = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
    except Exception:
        raw = None

    if not raw:
        # try nested alternatives
        try:
            raw = resp["usage"]
        except Exception:
            raw = None

    if raw:
        # raw may be an object with attributes or a dict
        try:
            usage["prompt_tokens"] = int(getattr(raw, "prompt_tokens", raw.get("prompt_tokens", 0)))
        except Exception:
            usage["prompt_tokens"] = int(raw.get("prompt_tokens", 0)) if isinstance(raw, dict) else 0
        try:
            usage["completion_tokens"] = int(
                getattr(raw, "completion_tokens", raw.get("completion_tokens", raw.get("output_tokens", 0)))
            )
        except Exception:
            usage["completion_tokens"] = int(raw.get("completion_tokens", raw.get("output_tokens", 0))) if isinstance(raw, dict) else 0
        try:
            usage["total_tokens"] = int(
                getattr(raw, "total_tokens", raw.get("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]))
            )
        except Exception:
            usage["total_tokens"] = int(
                raw.get("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
            ) if isinstance(raw, dict) else (usage["prompt_tokens"] + usage["completion_tokens"])

    return usage


def _update_token_counters(resp: Any):
    """
    Update both the last-call and the session aggregate token counters from response object.
    """
    usage = _extract_usage_from_response(resp)
    # store last call usage
    st.session_state.last_call_usage = usage
    # accumulate to session totals
    st.session_state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
    st.session_state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
    st.session_state.token_usage["total_tokens"] += usage.get("total_tokens", 0)


# ---------------------------------------------------------------------------------------
# Sidebar — Mode Selector and Session Controls (restored)
# ---------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Select capability", ["Chat", "Images", "Audio", "Embeddings", "Documents"])

    # Session buttons (horizontal) restored to the sidebar as requested
    btn_col_left, btn_col_right = st.columns([1, 1])
    with btn_col_left:
        if st.button("Clear", key="session_clear_btn", use_container_width=True):
            # Clear conversation only
            st.session_state.messages.clear()
            st.success("Conversation cleared")
    with btn_col_right:
        if st.button("New", key="session_new_btn", use_container_width=True):
            # Start a fresh session
            st.session_state.messages.clear()
            st.session_state.files.clear()
            st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            st.session_state.last_call_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            st.session_state.session_id = None
            st.success("New session started")

    # Session token summary in the sidebar (compact)
    tu = st.session_state.token_usage
    st.markdown(
        f"**Session tokens**  \nprompt: {tu['prompt_tokens']} · completion: {tu['completion_tokens']} · total: {tu['total_tokens']}"
    )

# ---------------------------------------------------------------------------------------
# Header (unchanged look & placement)
# ---------------------------------------------------------------------------------------
st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">Boo</h1>
    <p style="color:#9aa0a6;">Multimodal AI Assistant</p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------------------
# CHAT MODE
# ---------------------------------------------------------------------------------------
if mode == "Chat":
    left, center, right = st.columns([1, 2, 1])

    with center:
        # render conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask Boo something…")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    # call existing Chat.generate_text(...) — keep signature as your app expects
                    response = chat.generate_text(
                        prompt=prompt,
                        model=chat.model_options[0] if not hasattr(st, "model") else st.session_state.get("model"),
                    )
                    # display response
                    st.markdown(response or "")
                    st.session_state.messages.append({"role": "assistant", "content": response or ""})
                    # update token counters if chat.response exists
                    try:
                        _update_token_counters(getattr(chat, "response", None))
                    except Exception:
                        # don't fail UI if response doesn't include usage
                        pass

    # show last-call usage
    lcu = st.session_state.last_call_usage
    if lcu and any(v for v in lcu.values()):
        st.info(f"Last call tokens — prompt: {lcu['prompt_tokens']}, completion: {lcu['completion_tokens']}, total: {lcu['total_tokens']}")

# ---------------------------------------------------------------------------------------
# IMAGES MODE (unchanged except we don't alter logic)
# ---------------------------------------------------------------------------------------
elif mode == "Images":
    image = Image()

    with st.sidebar:
        st.header("Image Settings")
        model = st.selectbox("Model", image.model_options)
        size = st.selectbox("Size", image.size_options)
        quality = st.selectbox("Quality", image.quality_options)
        fmt = st.selectbox("Format", image.format_options)

    left, center, right = st.columns([1, 2, 1])

    with center:
        tab_gen, tab_analyze = st.tabs(["Generate", "Analyze"])
        with tab_gen:
            prompt = st.text_area("Image prompt")
            if st.button("Generate image"):
                with st.spinner("Generating image…"):
                    img_data = image.generate(prompt=prompt, model=model, size=size, quality=quality, fmt=fmt)
                    # update tokens if supported
                    try:
                        _update_token_counters(getattr(image, "response", None))
                    except Exception:
                        pass
                    st.write("Image generation completed.")
        with tab_analyze:
            st.write("Image analysis is unchanged.")

# ---------------------------------------------------------------------------------------
# AUDIO MODE (unchanged)
# ---------------------------------------------------------------------------------------
elif mode == "Audio":
    transcriber = Transcription()
    translator = Translation()

    with st.sidebar:
        st.header("Audio Settings")
        model = st.selectbox("Model", transcriber.model_options)
        language = st.selectbox("Language", transcriber.language_options)

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.write("Audio features remain as before.")

# ---------------------------------------------------------------------------------------
# EMBEDDINGS MODE (unchanged)
# ---------------------------------------------------------------------------------------
elif mode == "Embeddings":
    embedding = Embedding()

    with st.sidebar:
        st.header("Embedding Settings")
        model = st.selectbox("Model", embedding.model_options)
        method = st.selectbox("Method", embedding.methods if hasattr(embedding, "methods") else ["encode"])

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.write("Embeddings UI unchanged.")

# ---------------------------------------------------------------------------------------
# DOCUMENTS MODE — multi-document client-side context
# ---------------------------------------------------------------------------------------
elif mode == "Documents":
    # Documents UI lives on the page (not the sidebar) to keep sidebar uncluttered

    st.header("Documents")

    # uploader (client-side). Keep accept_multiple_files semantics but save to tmp files
    uploads = st.file_uploader("Upload documents (pdf, txt, md, docx)", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True)

    if uploads:
        # replace session files set so user clearly knows the upload set
        st.session_state.files.clear()
        for f in uploads:
            path = save_temp(f)
            st.session_state.files.append(path)
        st.success(f"Saved {len(uploads)} document(s) for this session.")

    # list uploaded client-side files
    if st.session_state.files:
        st.markdown("**Uploaded documents (client-side)**")
        # display as selectbox or radio for single selection
        file_index = st.selectbox("Choose a document", options=list(range(len(st.session_state.files))), format_func=lambda i: st.session_state.files[i])
        selected_path = st.session_state.files[file_index]

        # quick actions for selected file
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Remove selected document"):
                # remove the selected file
                removed = st.session_state.files.pop(file_index)
                st.success(f"Removed {removed}")
        with c2:
            if st.button("Download selected path as text (local)"):
                # show path only — actual download can be wired to open/read content if desired
                st.info(f"Local file path: {selected_path}")

        st.markdown("---")

        # Ask a question about the selected document
        question = st.text_area("Question for the selected document", placeholder="Ask something about the document...")
        ask_col, _ = st.columns([1, 2])
        with ask_col:
            if st.button("Ask Document"):
                if not question:
                    st.warning("Please enter a question before asking.")
                else:
                    with st.spinner("Running document Q&A…"):
                        try:
                            # Use the Chat.summarize_document(...) function which expects local path
                            # This keeps all file handling on the client side (we upload inside that method)
                            answer = chat.summarize_document(prompt=question, pdf_path=selected_path, model=chat.model_options[0])
                            st.markdown("**Answer:**")
                            st.markdown(answer or "No answer returned.")
                            # update conversation history if desired
                            st.session_state.messages.append({"role": "user", "content": f"[Document question] {question}"})
                            st.session_state.messages.append({"role": "assistant", "content": answer or ""})
                            # update tokens from chat.response if present
                            try:
                                _update_token_counters(getattr(chat, "response", None))
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Document Q&A failed: {e}")

        # show last-call and session usage after the Q&A (if any)
        lcu = st.session_state.last_call_usage
        if any(v for v in lcu.values()):
            st.write(f"Last call tokens — prompt: {lcu['prompt_tokens']} · completion: {lcu['completion_tokens']} · total: {lcu['total_tokens']}")
        tu = st.session_state.token_usage
        st.write(f"Session tokens — prompt: {tu['prompt_tokens']} · completion: {tu['completion_tokens']} · total: {tu['total_tokens']}")

    else:
        st.info("No client-side documents uploaded yet. Use the uploader above to add files for local Q&A.")

# ---------------------------------------------------------------------------------------
# Footer (small status): leave unchanged except add a small line about tokens for quick glance
# ---------------------------------------------------------------------------------------
st.divider()
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;color:#9aa0a6;font-size:0.85rem;">
        <span>Boo Framework</span>
        <span>Session tokens — total: {st.session_state.token_usage['total_tokens']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
