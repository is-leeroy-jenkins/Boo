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

st.set_page_config(  page_title="Boo • Multimoldal AI", page_icon=cfg.FAVICON_PATH, layout='wide' )

# ======================================================================================
# Session State
# ======================================================================================

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

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
    st.header("Mode")
    mode = st.radio(
        "Select capability",
        ["Chat", "Images", "Audio", "Embeddings"],
    )

# ======================================================================================
# Header
# ======================================================================================

st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">Boo</h1>
    <p style="color:#9aa0a6;">Multimodal Agent</p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ======================================================================================
# CHAT MODE
# ======================================================================================

if mode == "Chat":

    chat = Chat()

    with st.sidebar:
        st.header("Chat Settings")

        model = st.selectbox(
            "Model",
            chat.model_options,
        )

        # ---------------- Advanced Generation Controls ----------------

        with st.expander("Parameters", expanded=False):

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.05,
            )

            top_p = st.slider(
                "Top-P",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
            )

            frequency_penalty = st.slider(
                "Frequency penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
            )

            presence_penalty = st.slider(
                "Presence penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
            )

            max_tokens = st.slider(
                "Max tokens",
                min_value=128,
                max_value=4096,
                value=1024,
                step=128,
            )

        include = st.multiselect(
            "Includes:",
            chat.include_options,
        )

        chat.include = include

    _, center, _ = st.columns([1, 2, 1])

    with center:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask Boo…")

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
# IMAGE MODE
# ======================================================================================

elif mode == "Images":

    image = Image()

    with st.sidebar:
        st.header("Image Settings")

        model = st.selectbox("Model", image.model_options)
        size = st.selectbox("Size", image.size_options)
        quality = st.selectbox("Quality", image.quality_options)
        style = st.selectbox("Style", image.style_options)
        detail = st.selectbox("Detail", image.detail_options)
        fmt = st.selectbox("Format", image.format_options)

    _, center, _ = st.columns([1, 2, 1])

    with center:
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
# AUDIO MODE (SEMANTICALLY CORRECT)
# ======================================================================================

elif mode == "Audio":

    with st.sidebar:
        st.header("Audio Task")
        task = st.radio(
            "Select audio capability",
            ["Transcription", "Translation", "Text-to-Speech"],
        )

    # ---------------- Transcription ----------------

    if task == "Transcription":

        transcriber = Transcription()

        with st.sidebar:
            st.header("Transcription Settings")

            model = st.selectbox(
                "Model",
                transcriber.model_options,
            )

            fmt = st.selectbox(
                "Output format",
                transcriber.format_options,
            )

        _, center, _ = st.columns([1, 2, 1])

        with center:
            audio = st.audio_input("Record or upload audio")

            if audio and st.button("Transcribe"):
                path = save_temp(audio)

                with st.spinner("Transcribing…"):
                    text = transcriber.transcribe(
                        path,
                        model=model,
                        format=fmt,
                    )
                    st.subheader("Transcription")
                    st.markdown(text)

    # ---------------- Translation (Whisper → English) ----------------

    elif task == "Translation":

        translator = Translation()

        with st.sidebar:
            st.header("Translation Settings")

            model = st.selectbox(
                "Model",
                translator.model_options,
            )

            st.caption("Speech is translated to English (Whisper).")

        _, center, _ = st.columns([1, 2, 1])

        with center:
            audio = st.audio_input("Record or upload audio")

            if audio and st.button("Translate"):
                path = save_temp(audio)

                with st.spinner("Translating…"):
                    text = translator.translate(
                        path,
                        model=model,
                    )
                    st.subheader("Translation (to English)")
                    st.markdown(text)

    # ---------------- Text-to-Speech ----------------

    else:

        tts = TTS()

        with st.sidebar:
            st.header("Text-to-Speech Settings")

            model = st.selectbox("Model", tts.model_options)
            voice = st.selectbox("Voice", tts.voice_options)
            fmt = st.selectbox("Format", tts.format_options)
            speed = st.selectbox("Speed", tts.speed_options)

        _, center, _ = st.columns([1, 2, 1])

        with center:
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
# EMBEDDINGS MODE
# ======================================================================================

elif mode == "Embeddings":

    embedding = Embedding()

    with st.sidebar:
        st.header("Embedding Settings")

        model = st.selectbox("Model", embedding.model_options)
        encoding = st.selectbox("Encoding", embedding.encoding_options)

    _, center, _ = st.columns([1, 2, 1])

    with center:
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

st.markdown(
    """
    <hr/>
    <div style="display:flex; justify-content:space-between; color:#9aa0a6; font-size:0.85rem;">
        <span></span>
        <span>Generative AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)