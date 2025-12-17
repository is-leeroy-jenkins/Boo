# ******************************************************************************************
# Assembly:                Boo
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Created:                 12-16-2025
# ******************************************************************************************

from __future__ import annotations

import streamlit as st
from typing import Dict, List, Any
import tempfile

from boo import Boo
from config import DEFAULT_MODEL, MODELS


# =========================================================================================
# Streamlit Configuration
# =========================================================================================

st.set_page_config(
    page_title="Boo • AI Assistant",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================================
# Session State
# =========================================================================================

if "boo" not in st.session_state:
    st.session_state.boo = Boo(model=DEFAULT_MODEL)

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

if "files" not in st.session_state:
    st.session_state.files: List[str] = []


# =========================================================================================
# Sidebar — Control Panel
# =========================================================================================

with st.sidebar:
    st.markdown("## 👻 Boo")
    st.caption("Federal AI Assistant Framework")

    st.markdown("---")
    st.markdown("### 🧠 Model")

    model = st.selectbox(
        "Language Model",
        MODELS,
        index=MODELS.index(DEFAULT_MODEL),
    )

    if model != st.session_state.boo.model:
        st.session_state.boo = Boo(model=model)
        st.session_state.messages.clear()

    st.markdown("---")
    st.markdown("### 🧭 Mode")

    mode = st.radio(
        "Application Mode",
        [
            "Chat",
            "Image Generation",
            "Image Editing",
            "Image Analysis",
            "Audio (Transcription / Translation)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📄 Documents")

    uploaded_docs = st.file_uploader(
        "Upload documents for grounded Q&A",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_docs:
        for f in uploaded_docs:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(f.read())
                st.session_state.files.append(tmp.name)

    st.markdown("---")

    if st.button("Clear Conversation"):
        st.session_state.messages.clear()


# =========================================================================================
# Helpers
# =========================================================================================

def temp_file(upload) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(upload.read())
        return tmp.name


def run_chat(prompt: str) -> Dict[str, Any]:
    boo = st.session_state.boo
    kwargs = {}

    if st.session_state.files:
        kwargs["files"] = st.session_state.files

    result = boo.run(prompt, **kwargs)
    return result if isinstance(result, dict) else {"answer": result}


# =========================================================================================
# Hero Header
# =========================================================================================

st.markdown(
    """
    <div style="padding: 0.75rem 0;">
        <h1 style="margin-bottom: 0.25rem;">👻 Boo</h1>
        <p style="font-size:1.05rem; color:#9aa0a6;">
            Secure AI assistant for federal analytics, multimodal reasoning, and research
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =========================================================================================
# CHAT MODE — DOCUMENT-GROUNDED Q&A
# =========================================================================================

if mode == "Chat":

    with st.container(border=True):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask Boo a question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_chat(prompt)
                answer = response.get("answer", "")
                st.markdown(answer)

                if response.get("reasoning"):
                    with st.expander("🧠 Reasoning"):
                        st.markdown(response["reasoning"])

                if response.get("tools"):
                    with st.expander("🛠️ Tools Used"):
                        st.json(response["tools"])

                if response.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in response["sources"]:
                            st.markdown(f"- {src}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )


# =========================================================================================
# IMAGE GENERATION
# =========================================================================================

elif mode == "Image Generation":

    st.subheader("🖼️ Image Generation")

    prompt = st.text_area("Image prompt", height=120)
    size = st.selectbox("Image size", ["1024x1024", "512x512", "256x256"])

    if st.button("Generate Image"):
        if not hasattr(st.session_state.boo, "generate_image"):
            st.error("Image generation not available in this Boo build.")
        else:
            with st.spinner("Generating image..."):
                image = st.session_state.boo.generate_image(prompt, size=size)
                st.image(image, caption="Generated Image")


# =========================================================================================
# IMAGE EDITING
# =========================================================================================

elif mode == "Image Editing":

    st.subheader("✏️ Image Editing")

    base_img = st.file_uploader("Base image", type=["png", "jpg", "jpeg"])
    mask_img = st.file_uploader("Mask image (optional)", type=["png"])
    prompt = st.text_area("Edit instructions", height=120)

    if st.button("Edit Image"):
        if not hasattr(st.session_state.boo, "edit_image"):
            st.error("Image editing not available in this Boo build.")
        elif not base_img:
            st.warning("Please upload a base image.")
        else:
            base_path = temp_file(base_img)
            mask_path = temp_file(mask_img) if mask_img else None

            with st.spinner("Editing image..."):
                image = st.session_state.boo.edit_image(
                    image_path=base_path,
                    mask_path=mask_path,
                    prompt=prompt,
                )
                st.image(image, caption="Edited Image")


# =========================================================================================
# IMAGE ANALYSIS
# =========================================================================================

elif mode == "Image Analysis":

    st.subheader("🔍 Image Analysis")

    img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    prompt = st.text_area(
        "Analysis instructions",
        value="Describe the image in detail.",
        height=120,
    )

    if st.button("Analyze Image"):
        if not hasattr(st.session_state.boo, "analyze_image"):
            st.error("Image analysis not available in this Boo build.")
        elif not img:
            st.warning("Please upload an image.")
        else:
            img_path = temp_file(img)

            with st.spinner("Analyzing image..."):
                result = st.session_state.boo.analyze_image(
                    image_path=img_path,
                    prompt=prompt,
                )
                st.markdown(result)


# =========================================================================================
# AUDIO — TRANSCRIPTION & TRANSLATION
# =========================================================================================

elif mode == "Audio (Transcription / Translation)":

    st.subheader("🎙️ Audio Transcription & Translation")

    audio_file = st.audio_input("Record or upload audio")

    task = st.radio(
        "Task",
        ["Transcription", "Translation"],
        horizontal=True,
    )

    target_language = None
    if task == "Translation":
        target_language = st.text_input(
            "Target language (ISO code)",
            value="en",
        )

    if audio_file and st.button("Process Audio"):
        audio_path = temp_file(audio_file)
        boo = st.session_state.boo

        with st.spinner("Processing audio..."):
            if task == "Transcription":
                if not hasattr(boo, "transcribe"):
                    st.error("Audio transcription not available in this Boo build.")
                else:
                    text = boo.transcribe(audio_path)
                    st.markdown("### 📝 Transcription")
                    st.markdown(text)

            elif task == "Translation":
                if not hasattr(boo, "translate"):
                    st.error("Audio translation not available in this Boo build.")
                else:
                    text = boo.translate(
                        audio_path,
                        target_language=target_language,
                    )
                    st.markdown("### 🌍 Translation")
                    st.markdown(text)


# =========================================================================================
# Footer
# =========================================================================================

st.markdown(
    f"""
    <hr/>
    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#9aa0a6;">
        <span>Boo Framework</span>
        <span>Model: {st.session_state.boo.model}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
