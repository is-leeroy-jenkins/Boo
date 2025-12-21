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

# Compatibility import: prefer gpt.py if present, otherwise fallback to boo.py
try:
    from gpt import (
        Chat,
        Image,
        Embedding,
        Transcription,
        Translation,
    )
except Exception:
    from gpt import (
        Chat,
        Image,
        Embedding,
        Transcription,
        Translation,
    )

# ======================================================================================
# Page Configuration
# ======================================================================================

st.set_page_config(
    page_title="Boo • Multimoldal AI Agent",
    page_icon=cfg.FAVICON_PATH,
    layout="wide",
)

# ======================================================================================
# Session State
# ======================================================================================

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

# Token counters: last call and aggregated totals
if "last_call_usage" not in st.session_state:
    st.session_state.last_call_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Client-side uploaded files (temporary paths)
if "files" not in st.session_state:
    st.session_state.files: List[str] = []

# ======================================================================================
# Utilities
# ======================================================================================

def save_temp(upload) -> str:
    """Save uploaded file to a named temporary file and return path."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(upload.read())
        return tmp.name


def _extract_usage_from_response(resp: Any) -> Dict[str, int]:
    """
    Extract token usage from a response object/dict.
    Returns dict with prompt_tokens, completion_tokens, total_tokens.
    Defensive: returns zeros if not present.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not resp:
        return usage

    raw = None
    try:
        raw = getattr(resp, "usage", None)
    except Exception:
        raw = None

    if not raw and isinstance(resp, dict):
        raw = resp.get("usage")

    # Fallback: try typical nested places
    if not raw and isinstance(resp, dict) and resp.get("choices"):
        try:
            raw = resp["choices"][0].get("usage")
        except Exception:
            raw = None

    if not raw:
        return usage

    try:
        if isinstance(raw, dict):
            usage["prompt_tokens"] = int(raw.get("prompt_tokens", 0))
            usage["completion_tokens"] = int(raw.get("completion_tokens", raw.get("output_tokens", 0)))
            usage["total_tokens"] = int(raw.get("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]))
        else:
            usage["prompt_tokens"] = int(getattr(raw, "prompt_tokens", 0))
            usage["completion_tokens"] = int(getattr(raw, "completion_tokens", getattr(raw, "output_tokens", 0)))
            usage["total_tokens"] = int(getattr(raw, "total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]))
    except Exception:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    return usage


def _update_token_counters(resp: Any) -> None:
    """
    Update session_state.last_call_usage and accumulate into session_state.token_usage.
    """
    usage = _extract_usage_from_response(resp)
    st.session_state.last_call_usage = usage
    st.session_state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
    st.session_state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
    st.session_state.token_usage["total_tokens"] += usage.get("total_tokens", 0)


# ======================================================================================
# Sidebar — Mode Selector (baseline preserved)
# Add only the Files API radio option in the mode selector.
# ======================================================================================

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select capability",
        [ "Chat", "Images", "Audio", "Embeddings", "Documents", "Files" ],
    )

    # Horizontal session controls (only two short buttons requested)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Clear", key="session_clear_btn", use_container_width=True):
            st.session_state.messages.clear()
            st.success("Cleared!")
    with c2:
        if st.button("New", key="session_new_btn", use_container_width=True):
            st.session_state.messages.clear()
            st.session_state.files.clear()
            st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            st.session_state.last_call_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            st.success("Started!")

# ======================================================================================
# Header (baseline preserved)
# ======================================================================================

st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">Boo</h1>
    <p style="color:#9aa0a6;">Multimodal AI Assistant</p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ======================================================================================
# CHAT MODE
# - preserve baseline behavior, update token counters after generate_text calls
# ======================================================================================

if mode == "Chat":

    chat = Chat()

    with st.sidebar:
        st.header("Chat Settings")

        model = st.selectbox(
            "Model",
            chat.model_options,
        )

        include = st.multiselect(
            "Include in response",
            chat.include_options,
        )

        chat.include = include

    left, center, right = st.columns([1, 2, 1])

    with center:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask Boo something…")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        response = chat.generate_text(
                            prompt=prompt,
                            model=model,
                        )
                    except TypeError:
                        # defensive fallback: older signature
                        response = chat.generate_text(prompt=prompt)
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")
                        response = None

                    st.markdown(response or "")
                    st.session_state.messages.append({"role": "assistant", "content": response or ""})

                    # Update token counters defensively from chat.response or returned response
                    try:
                        _update_token_counters(getattr(chat, "response", None) or response)
                    except Exception:
                        pass

    # Token transparency in main area
    lcu = st.session_state.last_call_usage
    tu = st.session_state.token_usage
    if any(lcu.values()):
        st.info(f"Last call — prompt: {lcu['prompt_tokens']}, completion: {lcu['completion_tokens']}, total: {lcu['total_tokens']}")
    if tu["total_tokens"] > 0:
        st.write(f"Session totals — prompt: {tu['prompt_tokens']} · completion: {tu['completion_tokens']} · total: {tu['total_tokens']}")

# ======================================================================================
# IMAGE MODE (baseline preserved)
# ======================================================================================

elif mode == "Images":
    image = Image()
    with st.sidebar:
        st.header("Image Settings")
        model = st.selectbox("Model", image.model_options)
        size = st.selectbox("Size", image.size_options)
        quality = st.selectbox("Quality", image.quality_options)
        fmt = st.selectbox("Format", image.format_options)

    tab_gen, tab_analyze = st.tabs(["Generate", "Analyze"])
    with tab_gen:
        prompt = st.text_area("Prompt")
        if st.button("Generate Image"):
            with st.spinner("Generating…"):
                try:
                    img_url = image.generate(prompt=prompt, model=model, size=size, quality=quality, fmt=fmt)
                    st.image(img_url)
                    _update_token_counters(getattr(image, "response", None))
                except Exception as exc:
                    st.error(f"Image generation failed: {exc}")
    with tab_analyze:
        st.write("Image analysis — baseline behavior preserved.")

# ======================================================================================
# AUDIO MODE (baseline preserved)
# ======================================================================================

elif mode == "Audio":
    transcriber = Transcription()
    translator = Translation()

    with st.sidebar:
        st.header("Audio Settings")
        model = st.selectbox("Model", transcriber.model_options)
        language = st.selectbox("Language", transcriber.language_options)
        task = st.selectbox("Task", ["Transcribe", "Translate"])

    left, center, right = st.columns([1, 2, 1])
    with center:
        uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac"])
        if uploaded:
            tmp = save_temp(uploaded)
            if task == "Transcribe":
                with st.spinner("Transcribing…"):
                    text = transcriber.transcribe(tmp, model=model)
                    st.text_area("Transcript", value=text, height=300)
            else:
                with st.spinner("Translating…"):
                    text = translator.translate(tmp)
                    st.text_area("Translation", value=text, height=300)

# ======================================================================================
# EMBEDDINGS MODE (baseline preserved)
# ======================================================================================

elif mode == "Embeddings":
    embed = Embedding()
    with st.sidebar:
        st.header("Embedding Settings")
        model = st.selectbox("Model", embed.model_options)
        method = st.selectbox("Method", getattr(embed, "methods", ["encode"]))

    left, center, right = st.columns([1, 2, 1])
    with center:
        text = st.text_area("Text to embed")
        if st.button("Embed"):
            with st.spinner("Embedding…"):
                v = embed.create(text, model=model)
                st.write("Vector length:", len(v))

# ======================================================================================
# DOCUMENTS MODE — client-side multi-document context (purely session-local)
# ======================================================================================

if mode == "Documents":
    st.header("Documents")

    # Uploader (main page) — session-local files stored in st.session_state.files
    uploaded = st.file_uploader(
        "Upload documents (session only)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    if uploaded:
        for up in uploaded:
            st.session_state.files.append(save_temp(up))
        st.success(f"Saved {len(uploaded)} file(s) to session")

    if st.session_state.files:
        st.markdown("**Uploaded documents (session-only)**")
        idx = st.selectbox(
            "Choose a document",
            options=list(range(len(st.session_state.files))),
            format_func=lambda i: st.session_state.files[i],
        )
        selected_path = st.session_state.files[idx]

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Remove selected document"):
                removed = st.session_state.files.pop(idx)
                st.success(f"Removed {removed}")
        with c2:
            if st.button("Show selected path"):
                st.info(f"Local temp path: {selected_path}")

        st.markdown("---")
        question = st.text_area("Ask a question about the selected document")
        if st.button("Ask Document"):
            if not question:
                st.warning("Enter a question before asking.")
            else:
                with st.spinner("Running document Q&A…"):
                    try:
                        # Instantiate Chat defensively if not present
                        try:
                            chat  # type: ignore
                        except NameError:
                            chat = Chat()
                        answer = None
                        if hasattr(chat, "summarize_document"):
                            try:
                                answer = chat.summarize_document(prompt=question, pdf_path=selected_path)
                            except TypeError:
                                # fallback to positional signature
                                answer = chat.summarize_document(question, selected_path)
                        elif hasattr(chat, "ask_document"):
                            answer = chat.ask_document(selected_path, question)
                        elif hasattr(chat, "document_qa"):
                            answer = chat.document_qa(selected_path, question)
                        else:
                            raise RuntimeError("No document-QA method found on chat object.")

                        st.markdown("**Answer:**")
                        st.markdown(answer or "No answer returned.")
                        st.session_state.messages.append({"role": "user", "content": f"[Document question] {question}"})
                        st.session_state.messages.append({"role": "assistant", "content": answer or ""})

                        # update token counters defensively
                        try:
                            _update_token_counters(getattr(chat, "response", None) or answer)
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f"Document Q&A failed: {e}")

    else:
        st.info("No client-side documents uploaded this session. Use the uploader above to add files.")

# ======================================================================================
# FILES API MODE — minimal, non-invasive page that uses gpt.py file methods if present
# ======================================================================================
if mode == "Files":
    st.header("Files")

    # instantiate Chat (needed for file methods)
    try:
        chat  # type: ignore
    except NameError:
        chat = Chat()

    # Try to resolve a file-listing method from the chat object
    list_method = None
    for name in ("retrieve_files", "retreive_files", "list_files", "get_files"):
        if hasattr(chat, name):
            list_method = getattr(chat, name)
            list_method_name = name
            break

    uploaded_file = st.file_uploader("Upload file (server-side via Files API)", type=["pdf", "txt", "md", "docx", "png", "jpg", "jpeg"])
    if uploaded_file:
        tmp_path = save_temp(uploaded_file)
        upload_fn = None
        for name in ("upload_file", "upload", "files_upload"):
            if hasattr(chat, name):
                upload_fn = getattr(chat, name)
                upload_name = name
                break
        if not upload_fn:
            st.warning("No upload function found on chat object (upload_file).")
        else:
            with st.spinner("Uploading to Files API..."):
                try:
                    fid = upload_fn(tmp_path)
                    st.success(f"Uploaded; file id: {fid}")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    if st.button("List files"):
        if not list_method:
            st.warning("No file-listing method found on chat object (expected retrieve_files/list_files).")
        else:
            with st.spinner("Listing files..."):
                try:
                    files_resp = list_method()
                    # Normalize files_resp to a list of dict-like items
                    files_list = []
                    if files_resp is None:
                        files_list = []
                    elif isinstance(files_resp, dict):
                        # some clients return {'data': [...]}
                        files_list = files_resp.get("data") or files_resp.get("files") or []
                    elif isinstance(files_resp, list):
                        files_list = files_resp
                    else:
                        # try attribute access
                        try:
                            files_list = getattr(files_resp, "data", files_resp)
                        except Exception:
                            files_list = [files_resp]

                    # Render the files in a compact table where possible
                    rows = []
                    for f in files_list:
                        try:
                            # f may be dict-like or object-like
                            fid = f.get("id") if isinstance(f, dict) else getattr(f, "id", None)
                            name = f.get("filename") if isinstance(f, dict) else getattr(f, "filename", None)
                            purpose = f.get("purpose") if isinstance(f, dict) else getattr(f, "purpose", None)
                        except Exception:
                            fid = None
                            name = str(f)
                            purpose = None
                        rows.append({"id": fid, "filename": name, "purpose": purpose})
                    if rows:
                        st.table(rows)
                    else:
                        st.info("No files returned.")
                except Exception as exc:
                    st.error(f"List files failed: {exc}")

    # If we have a listing in the session, provide deletion capability
    # Allow user to choose a file id (re-list first is recommended)
    if "last_files_list" in st.session_state:
        ls = st.session_state.last_files_list
    else:
        ls = None

    # Attempt to load files for a local selectbox if earlier listed
    if 'files_list' in locals() and files_list:
        file_ids = [r.get("id") if isinstance(r, dict) else getattr(r, "id", None) for r in files_list]
        sel = st.selectbox("Select file id to delete", options=file_ids)
        if st.button("Delete selected file"):
            del_fn = None
            for name in ("delete_file", "delete", "files_delete"):
                if hasattr(chat, name):
                    del_fn = getattr(chat, name)
                    break
            if not del_fn:
                st.warning("No delete function found on chat object (expected delete_file).")
            else:
                with st.spinner("Deleting file..."):
                    try:
                        res = del_fn(sel)
                        st.success(f"Delete result: {res}")
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

# ======================================================================================
# Footer (baseline preserved; non-invasive token summary kept in main footer only if present)
# ======================================================================================

st.divider()
tu = st.session_state.token_usage
if tu["total_tokens"] > 0:
    footer_html = f"""
    <div style="display:flex;justify-content:space-between;color:#9aa0a6;font-size:0.85rem;">
        <span>Boo Framework</span>
        <span>Session tokens — total: {tu['total_tokens']}</span>
    </div>
    """
else:
    footer_html = """
    <div style="display:flex;justify-content:space-between;color:#9aa0a6;font-size:0.85rem;">
        <span>Boo Framework</span>
        <span>text • audio • images</span>
    </div>
    """

st.markdown(footer_html, unsafe_allow_html=True)
