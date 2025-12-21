# ******************************************************************************************
# Assembly:                Boo
# Filename:                app.py
# Author:                  Terry D. Eppler (integration) / ChatGPT (UI adjustments)
# Created:                 12-16-2025
# ******************************************************************************************

from __future__ import annotations

import config as cfg
import streamlit as st
import tempfile
from typing import List, Dict, Any
import uuid

from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

# Import the Boo framework classes
from gpt import (
    Chat,
    Image,
    Embedding,
    Transcription,
    Translation,
    TTS,
)

# ======================================================================================
# Page Configuration
# ======================================================================================

st.set_page_config(
    page_title="Boo • Multimodal AI",
    page_icon=cfg.FAVICON_PATH,
    layout="wide",
)

# ======================================================================================
# Session State Initialization
# ======================================================================================

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

if "doc_files" not in st.session_state:
    st.session_state.doc_files: Dict[str, Dict[str, Any]] = {}

if "token_usage" not in st.session_state:
    st.session_state.token_usage: List[Dict[str, Any]] = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Ensure a mode field exists so the dynamic header can read it
if "mode" not in st.session_state:
    st.session_state.mode = "Chat"

# ======================================================================================
# Utilities
# ======================================================================================

def save_temp(upload) -> str:
    """Save an uploaded file-like object to a temporary file and return path."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(upload.read())
        return tmp.name

def reset_conversation():
    """Clear chat messages and token accounting for the session."""
    st.session_state.messages.clear()
    st.session_state.token_usage.clear()

def start_new_session():
    """Reset conversation, documents, and token usage; roll new session_id."""
    reset_conversation()
    st.session_state.doc_files.clear()
    st.session_state.token_usage.clear()
    st.session_state.session_id = str(uuid.uuid4())

def clear_doc_context():
    """Clear only uploaded document context for the session."""
    st.session_state.doc_files.clear()

# ======================================================================================
# Dynamic header renderer
# ======================================================================================

def render_dynamic_header() -> None:
    """
    Renders a compact dynamic header showing the selected mode and
    a short summary of primary parameters. This is intentionally
    lightweight and avoids any network or heavy work.
    """
    current_mode = st.session_state.get("mode", "—")

    summary_items: List[str] = []

    # Chat summary
    if current_mode == "Chat":
        model_name = st.session_state.get("chat_model", "—")
        temperature = st.session_state.get("temperature", "—")
        top_p = st.session_state.get("top_p", "—")
        includes = st.session_state.get("chat_include", [])
        summary_items.append(f"Model: {model_name}")
        summary_items.append(f"T: {temperature} • Top-P: {top_p}")
        if includes:
            short = ", ".join(includes[:3]) + (",…" if len(includes) > 3 else "")
            summary_items.append(f"Include: {short}")

    # Images summary
    elif current_mode == "Images":
        model_name = st.session_state.get("image_model", "—")
        size = st.session_state.get("image_size", "—")
        summary_items.append(f"Model: {model_name}")
        summary_items.append(f"Size: {size}")

    # Documents summary
    elif current_mode == "Documents":
        doc_count = len(st.session_state.get("doc_files", {}))
        summary_items.append(f"Documents: {doc_count} uploaded")

    # Embeddings summary
    elif current_mode == "Embeddings":
        model_name = st.session_state.get("embedding_model", "—")
        summary_items.append(f"Model: {model_name}")

    # Audio summary (lightweight)
    elif current_mode == "Audio":
        summary_items.append("Audio tools")

    subtitle = "  |  ".join(summary_items) if summary_items else ""
    st.markdown(
        f"""
        <div style="margin-top:0.25rem;">
            <h2 style="margin:0;">{current_mode}</h2>
            <div style="color:#6b7280; font-size:0.9rem; margin-top:0.125rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================================================
# Sidebar — Primary Mode Selection and Session Controls
# ======================================================================================

with st.sidebar:
    colored_header(
        label="Boo",
        description="Generative AI Toolkit",
        color_name="violet-70",
    )

    # Mode radio — store in session_state so header can read it
    mode = st.radio(
        "Mode",
        ["Chat", "Images", "Audio", "Documents", "Embeddings"],
        index=["Chat", "Images", "Audio", "Documents", "Embeddings"].index(
            st.session_state.get("mode", "Chat")
        ),
        key="mode",
    )

    # Evenly distributed session buttons (horizontal)
    btn_col_left, btn_col_right = st.columns([1, 1])
    with btn_col_left:
        if st.button("Clear", key="session_clear_btn", use_container_width=True):
            if mode == "Documents":
                clear_doc_context()
            else:
                reset_conversation()
    with btn_col_right:
        if st.button("New", key="session_new_btn", use_container_width=True):
            start_new_session()
            st.rerun()

# ======================================================================================
# Dynamic subheader (no static top header)
# ======================================================================================

# Call the dynamic header so it updates on every rerun/widget change
render_dynamic_header()
st.divider()

# ======================================================================================
# CHAT MODE
# ======================================================================================

if st.session_state["mode"] == "Chat":

    chat = Chat()

    # Chat settings live in the sidebar (keys added so dynamic header can read)
    with st.sidebar:
        colored_header("Chat Settings", "", "violet-70")

        model = st.selectbox(
            "Model",
            chat.model_options,
            key="chat_model",
        )

        with st.expander("Parameters:", expanded=False):
            temperature = st.slider(
                "Temperature",
                0.0,
                2.0,
                0.7,
                0.05,
                key="temperature",
            )
            top_p = st.slider(
                "Top-P",
                0.0,
                1.0,
                1.0,
                0.05,
                key="top_p",
            )
            frequency_penalty = st.slider(
                "Frequency penalty",
                -2.0,
                2.0,
                0.0,
                0.1,
                key="frequency_penalty",
            )
            presence_penalty = st.slider(
                "Presence penalty",
                -2.0,
                2.0,
                0.0,
                0.1,
                key="presence_penalty",
            )
            max_tokens = st.slider(
                "Max tokens",
                128,
                4096,
                1024,
                128,
                key="max_tokens",
            )

        include = st.multiselect(
            "Include in response",
            chat.include_options,
            key="chat_include",
        )
        # keep Behavior consistent with Boo.Chat class expectations
        chat.include = include

    # Chat content area
    with stylable_container(
        key="chat_container",
        css_styles="""
            {
                max-width: 900px;
                margin: 0 auto;
            }
        """,
    ):
        total_input = 0
        total_output = 0
        total_tokens = 0

        # Render historical messages
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                usage = msg.get("token_usage")
                if usage:
                    st.caption(
                        f"Tokens: {usage.get('input','–')} in / {usage.get('output','–')} out / {usage.get('total','–')} total"
                        + (f" | Cost: ${usage.get('cost','–')}" if usage.get('cost') else "")
                    )
                    total_input += usage.get("input", 0) or 0
                    total_output += usage.get("output", 0) or 0
                    total_tokens += usage.get("total", 0) or 0

        # Chat input
        prompt = st.chat_input("Ask Boo something…")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    response_text = ""
                    usage = {}
                    try:
                        response_text = chat.generate_text(
                            prompt=prompt,
                            model=model,
                            temperature=temperature,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            max_tokens=max_tokens,
                        )
                        # Attempt to pull usage from chat.response if available
                        if hasattr(chat, "response") and getattr(chat, "response", None):
                            resp = getattr(chat, "response")
                            u = getattr(resp, "usage", {}) or {}
                            usage = {
                                "input": u.get("prompt_tokens"),
                                "output": u.get("completion_tokens"),
                                "total": u.get("total_tokens"),
                                "cost": getattr(resp, "cost", None),
                            }
                    except Exception as ex:
                        response_text = f"Error: {ex}"

                    st.markdown(response_text or "")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text or "", "token_usage": usage or {}}
                    )

                    if usage:
                        total_input += usage.get("input", 0) or 0
                        total_output += usage.get("output", 0) or 0
                        total_tokens += usage.get("total", 0) or 0

        # Show session totals (if any tokens consumed)
        if total_tokens > 0:
            # Note: adjust cost math to reflect real pricing when you have it
            cost = total_tokens * 0.002 / 1000
            st.info(
                f"**Session usage** — Input: {total_input} | Output: {total_output} | Total: {total_tokens} | Cost: ${cost:.4f}"
            )

# ======================================================================================
# IMAGES MODE
# ======================================================================================

elif st.session_state["mode"] == "Images":

    image = Image()

    with st.sidebar:
        colored_header("Image Settings", "", "violet-70")

        model = st.selectbox(
            "Model",
            image.model_options,
            key="image_model",
        )
        size = st.selectbox(
            "Size",
            image.size_options,
            key="image_size",
        )
        quality = st.selectbox("Quality", image.quality_options)
        style = st.selectbox("Style", image.style_options)
        detail = st.selectbox("Detail", image.detail_options)
        fmt = st.selectbox("Format", image.format_options)

    with stylable_container(
        key="images_container",
        css_styles="""
            {
                max-width: 900px;
                margin: 0 auto;
            }
        """,
    ):
        tab_gen, tab_analyze = st.tabs(["Generate", "Analyze"])

        with tab_gen:
            prompt = st.text_area("Prompt", height=120)

            if st.button("Generate Image"):
                with st.spinner("Generating…"):
                    url = image.generate(
                        prompt=prompt,
                        model=model,
                        size=size,
                        quality=quality,
                        style=style,
                        detail=detail,
                        format=fmt,
                    )
                    st.image(url)

        with tab_analyze:
            img = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg"],
            )

            prompt = st.text_area(
                "Analysis prompt",
                value="Describe this image in detail.",
            )

            if img and st.button("Analyze Image"):
                path = save_temp(img)
                with st.spinner("Analyzing…"):
                    result = image.analyze(
                        text=prompt,
                        path=path,
                        model=model,
                        detail=detail,
                    )
                    st.markdown(result)

# ======================================================================================
# AUDIO MODE
# ======================================================================================

elif st.session_state["mode"] == "Audio":
    with st.sidebar:
        colored_header("Audio Task", "", "violet-70")
        task = st.radio(
            "Select audio capability",
            ["Transcription", "Translation", "Text-to-Speech"],
            key="audio_task",
        )

    with stylable_container(
        key="audio_container",
        css_styles="""
            {
                max-width: 900px;
                margin: 0 auto;
            }
        """,
    ):
        if task == "Transcription":
            transcriber = Transcription()

            with st.sidebar:
                model = st.selectbox("Model", transcriber.model_options)
                fmt = st.selectbox("Output format", transcriber.format_options)

            audio = st.file_uploader("Record or upload audio", type=["wav", "mp3", "m4a", "ogg"])
            if audio and st.button("Transcribe"):
                path = save_temp(audio)
                with st.spinner("Transcribing…"):
                    text = transcriber.transcribe(path, model=model, format=fmt)
                    st.subheader("Transcription")
                    st.markdown(text)

        elif task == "Translation":
            translator = Translation()

            with st.sidebar:
                model = st.selectbox("Model", translator.model_options)
                st.caption("Speech → English (Whisper)")

            audio = st.file_uploader("Record or upload audio", type=["wav", "mp3", "m4a", "ogg"])
            if audio and st.button("Translate"):
                path = save_temp(audio)
                with st.spinner("Translating…"):
                    text = translator.translate(path, model=model)
                    st.subheader("Translation (to English)")
                    st.markdown(text)

        else:
            tts = TTS()

            with st.sidebar:
                model = st.selectbox("Model", tts.model_options)
                voice = st.selectbox("Voice", tts.voice_options)
                fmt = st.selectbox("Format", tts.format_options)
                speed = st.selectbox("Speed", tts.speed_options)

            text = st.text_area("Text to speak", height=120)

            if text and st.button("Generate Speech"):
                with st.spinner("Synthesizing…"):
                    audio_path = tts.speak(
                        text=text,
                        model=model,
                        voice=voice,
                        format=fmt,
                        speed=speed,
                    )
                    st.audio(audio_path)

# ======================================================================================
# DOCUMENTS MODE — Multi-document (client-side)
# ======================================================================================

elif st.session_state["mode"] == "Documents":

    chat = Chat()

    with st.sidebar:
        colored_header("Document Settings", "", "violet-70")
        model = st.selectbox("Model", chat.model_options)
        include = st.multiselect("Include in response", chat.include_options)
        chat.include = include

    with stylable_container(
        key="documents_container",
        css_styles="""
            {
                max-width: 900px;
                margin: 0 auto;
            }
        """,
    ):
        st.subheader("Uploaded Documents (session, client-side only):")
        uploaded = st.file_uploader(
            "Upload document(s)",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )
        if uploaded:
            for file in uploaded:
                file_id = str(uuid.uuid4())
                path = save_temp(file)
                st.session_state.doc_files[file_id] = {"name": file.name, "path": path}
            st.success(f"Uploaded {len(uploaded)} file(s)")

        # List and allow removal of uploaded docs
        if st.session_state.doc_files:
            cols = st.columns([3, 1])
            with cols[0]:
                for fid, doc in list(st.session_state.doc_files.items()):
                    st.markdown(f"- **{doc['name']}** (`{fid}`)")
            with cols[1]:
                for fid in list(st.session_state.doc_files):
                    if st.button(f"❌", key=f"del_{fid}", help="Remove this doc"):
                        st.session_state.doc_files.pop(fid)
                        st.rerun()

        tab_doc, tab_search = st.tabs(["Ask a Document", "Search Corpus"])

        with tab_doc:
            doc_id = st.selectbox(
                "Select document to ask",
                options=list(st.session_state.doc_files.keys()),
                format_func=lambda fid: st.session_state.doc_files[fid]["name"]
                if fid in st.session_state.doc_files
                else fid,
            ) if st.session_state.doc_files else None

            question = st.text_area("Ask a question about this document")
            if doc_id and question and st.button("Ask Document"):
                answer = chat.ask_document(file_id=doc_id, question=question, model=model)
                st.markdown(answer)

        with tab_search:
            # show available vector stores from Chat wrapper
            store = st.selectbox("Corpus", list(chat.vector_stores.keys()))
            query = st.text_area("Search query")
            if query and st.button("Search Corpus"):
                results = chat.search_file(
                    query=query, vector_store_id=chat.vector_stores[store], max_results=5, model=model
                )
                st.markdown(results)

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================

elif st.session_state["mode"] == "Embeddings":

    embedding = Embedding()

    with st.sidebar:
        colored_header("Embedding Settings", "", "violet-70")
        model = st.selectbox("Model", embedding.model_options, key="embedding_model")
        encoding = st.selectbox("Encoding", embedding.encoding_options)

    with stylable_container(
        key="embeddings_container",
        css_styles="""
            {
                max-width: 900px;
                margin: 0 auto;
            }
        """,
    ):
        text = st.text_area("Text to embed", height=150)

        if st.button("Create Embedding"):
            with st.spinner("Embedding…"):
                vector = embedding.create(text=text, model=model, format=encoding)
                st.success(f"Vector length: {len(vector)}")
                st.json(vector[:10])

# ======================================================================================
# Footer
# ======================================================================================

add_vertical_space(2)

st.markdown(
    """
    <hr/>
    <div style="display:flex; justify-content:space-between;
                color:#9aa0a6; font-size:0.85rem;">
        <span>Generative AI</span>
        <span>text • audio • images</span>
    </div>
    """,
    unsafe_allow_html=True,
)
