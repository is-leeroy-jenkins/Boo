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
from typing import List, Dict, Any

from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_extras.badges import badge

from boo import (
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
# Session State
# ======================================================================================

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

if "doc_file_id" not in st.session_state:
    st.session_state.doc_file_id: str | None = None

# ======================================================================================
# Utilities
# ======================================================================================

def save_temp(upload) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(upload.read())
        return tmp.name

# ======================================================================================
# Sidebar — Primary Mode Selection
# ======================================================================================

with st.sidebar:
    colored_header(
        label="Boo",
        description="Generative AI Toolkit",
        color_name="violet-70",
    )

    mode = st.radio(
        "Mode",
        ["Chat", "Images", "Audio", "Documents", "Embeddings"],
    )

# ======================================================================================
# Header
# ======================================================================================

add_vertical_space(1)

st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">Boo</h1>
    <p style="color:#9aa0a6;">Generative AI</p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ======================================================================================
# CHAT MODE
# ======================================================================================

if mode == "Chat":

    chat = Chat()
    st.badge("Chat", icon="💬")

    with st.sidebar:
        colored_header("Chat Settings", "", "violet-70")

        model = st.selectbox("Model", chat.model_options)

        with st.expander("Advanced generation settings", expanded=False):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
            top_p = st.slider("Top-P", 0.0, 1.0, 1.0, 0.05)
            frequency_penalty = st.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
            presence_penalty = st.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
            max_tokens = st.slider("Max tokens", 128, 4096, 1024, 128)

        include = st.multiselect("Include in response", chat.include_options)
        chat.include = include

    with stylable_container(
        key="chat_container",
        css_styles="""
            {
                max-width: 900px;
                margin: 0 auto;
            }
        """,
    ):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask Boo something…")

        if prompt:
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    response = chat.generate_text(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        max_tokens=max_tokens,
                    )
                    st.markdown(response or "")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response or ""}
                    )

# ======================================================================================
# IMAGES MODE
# ======================================================================================

elif mode == "Images":

    image = Image()
    st.badge("Images", icon="🖼️")

    with st.sidebar:
        colored_header("Image Settings", "", "violet-70")

        model = st.selectbox("Model", image.model_options)
        size = st.selectbox("Size", image.size_options)
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

elif mode == "Audio":

    st.badge("Audio", icon="🎙️")

    with st.sidebar:
        colored_header("Audio Task", "", "violet-70")
        task = st.radio(
            "Select audio capability",
            ["Transcription", "Translation", "Text-to-Speech"],
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

            audio = st.audio_input("Record or upload audio")

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

            audio = st.audio_input("Record or upload audio")

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
# DOCUMENTS MODE
# ======================================================================================

elif mode == "Documents":

    chat = Chat()
    st.badge("Documents", icon="📄")

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
        tab_doc, tab_search = st.tabs(["Ask a Document", "Search Corpus"])

        with tab_doc:
            uploaded = st.file_uploader(
                "Upload a document",
                type=["pdf", "txt", "md", "docx"],
            )

            if uploaded:
                path = save_temp(uploaded)
                with st.spinner("Uploading document…"):
                    file_id = chat.upload_document(path)
                    st.session_state.doc_file_id = file_id
                    st.toast("Document uploaded", icon="📄")

            question = st.text_area("Ask a question about this document")

            if st.session_state.doc_file_id and question:
                answer = chat.ask_document(
                    file_id=st.session_state.doc_file_id,
                    question=question,
                    model=model,
                )
                st.markdown(answer)

        with tab_search:
            store = st.selectbox("Corpus", list(chat.vector_stores.keys()))
            query = st.text_area("Search query")

            if query:
                results = chat.search_file(
                    query=query,
                    vector_store_id=chat.vector_stores[store],
                    max_results=5,
                    model=model,
                )
                st.markdown(results)

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================

elif mode == "Embeddings":

    embedding = Embedding()
    st.badge("Embeddings", icon="🧠")

    with st.sidebar:
        colored_header("Embedding Settings", "", "violet-70")
        model = st.selectbox("Model", embedding.model_options)
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
                vector = embedding.create(
                    text=text,
                    model=model,
                    format=encoding,
                )
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
