# ******************************************************************************************
# Assembly:                Boo
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Created:                 12-16-2025
# Notes:                   Restored Parameters section for Text mode and wired parameters
#                          into generation call defensively. Minimal other edits.
# ******************************************************************************************

from __future__ import annotations

import config as cfg
import streamlit as st
import tempfile
import re
from typing import List, Dict, Any, Optional

# ADDITIVE ONLY (for Prompt Engineering)
import sqlite3
import os

from gpt import (
	Chat,
	Image,
	Embedding,
	Transcription,
	Translation,
)

def get_active_chat( ):
	provider = st.session_state.get( "provider", "GPT" )
	
	if provider == "Gemini":
		import gemini
		
		return gemini
	
	if provider == "Groq":
		import grok
		
		return grok
	
	# Default: GPT
	import gpt
	
	return gpt

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config(
	page_title="Boo",
	page_icon=cfg.FAVICON_PATH,
	layout="wide",
)

# ======================================================================================
# Sidebar — Provider Logo (Dynamic)
# ======================================================================================
with st.sidebar:
    st.markdown(
        """
        <style>
        .provider-logo img {
            max-height: 50px;
            width: auto;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    provider = st.session_state.get("provider")
    logo_path = {
        "GPT": "resource/images/gpt_logo.png",
        "Gemini": "resource/images/gemma_logo.png",
        "Groq": "resource/images/grok_logo.png",
    }.get(provider)

    if logo_path and os.path.exists(logo_path):
        st.markdown(
            f"""
            <div class="provider-logo">
                <img src="file://{logo_path}">
            </div>
            """,
            unsafe_allow_html=True,
        )


# ======================================================================================
# Session State — initialize per-mode model keys and token counters
# ======================================================================================
if "messages" not in st.session_state:
	st.session_state.messages: List[ Dict[ str, Any ] ] = [ ]

if "last_call_usage" not in st.session_state:
	st.session_state.last_call_usage = {
			"prompt_tokens": 0,
			"completion_tokens": 0,
			"total_tokens": 0,
	}

if "token_usage" not in st.session_state:
	st.session_state.token_usage = {
			"prompt_tokens": 0,
			"completion_tokens": 0,
			"total_tokens": 0,
	}

if "files" not in st.session_state:
	st.session_state.files: List[ str ] = [ ]

# Per-mode model keys (deterministic header behavior) - chat -> text
if "text_model" not in st.session_state:
	st.session_state[ "text_model" ] = None
if "image_model" not in st.session_state:
	st.session_state[ "image_model" ] = None
if "audio_model" not in st.session_state:
	st.session_state[ "audio_model" ] = None
if "embed_model" not in st.session_state:
	st.session_state[ "embed_model" ] = None

# Temperature / top_p / other generation params defaults (Text controls)
if "temperature" not in st.session_state:
	st.session_state[ "temperature" ] = 0.7
if "top_p" not in st.session_state:
	st.session_state[ "top_p" ] = 1.0
if "max_tokens" not in st.session_state:
	st.session_state[ "max_tokens" ] = 512
if "freq_penalty" not in st.session_state:
	st.session_state[ "freq_penalty" ] = 0.0
if "pres_penalty" not in st.session_state:
	st.session_state[ "pres_penalty" ] = 0.0
if "stop_sequences" not in st.session_state:
	st.session_state[ "stop_sequences" ] = [ ]

# Provider default
if "provider" not in st.session_state:
	st.session_state[ "provider" ] = "GPT"

if "api_keys" not in st.session_state:
	st.session_state.api_keys = {
			"GPT": None,
			"Groq": None,
			"Gemini": None,
	}

# ======================================================================================
# Utilities
# ======================================================================================

_TAG_OPEN = re.compile( r"<([A-Za-z0-9_\-:.]+)>" )
_TAG_CLOSE = re.compile( r"</([A-Za-z0-9_\-:.]+)>" )

def xml_converter( text: str ) -> str:
	"""
	Convert delimiter-based XML-like prompt text into Markdown.

	Rules:
	- Every tag becomes '## <tag>' (single depth only)
	- Nesting is supported, but depth is flattened
	- Content is preserved verbatim
	"""
	if not text or not text.strip( ):
		return ""
	
	i = 0
	n = len( text )
	out: List[ str ] = [ ]
	stack: List[ str ] = [ ]
	
	def emit_heading( tag: str ) -> None:
		out.append( f"## {tag}" )
	
	def emit_text( chunk: str ) -> None:
		s = chunk.strip( "\n" )
		if s.strip( ):
			out.append( s.strip( ) )
	
	while i < n:
		m_open = _TAG_OPEN.search( text, i )
		m_close = _TAG_CLOSE.search( text, i )
		
		if not m_open and not m_close:
			emit_text( text[ i: ] )
			break
		
		if m_open and (not m_close or m_open.start( ) <= m_close.start( )):
			start, end = m_open.span( )
			emit_text( text[ i:start ] )
			tag = m_open.group( 1 ).strip( )
			stack.append( tag )
			emit_heading( tag )
			i = end
		else:
			start, end = m_close.span( )
			emit_text( text[ i:start ] )
			tag = m_close.group( 1 ).strip( )
			if stack and tag in stack:
				while stack and stack[ -1 ] != tag:
					stack.pop( )
				stack.pop( )
			i = end
	
	# Normalize spacing
	cleaned: List[ str ] = [ ]
	for block in out:
		if cleaned:
			cleaned.append( "" )
		cleaned.append( block )
	
	return "\n".join( cleaned ).strip( )

def markdown_converter( text: str ) -> str:
	"""
	Convert Markdown (## headings only) into delimiter-based XML-like text.
	"""
	if not text or not text.strip( ):
		return ""
	
	lines = text.splitlines( )
	out: List[ str ] = [ ]
	
	current_tag = None
	buffer: List[ str ] = [ ]
	
	def flush( ) -> None:
		nonlocal buffer, current_tag
		if current_tag:
			body = "\n".join( buffer ).strip( )
			out.append( f"<{current_tag}>" )
			if body:
				out.append( body )
			out.append( f"</{current_tag}>" )
		buffer = [ ]
	
	for line in lines:
		if line.startswith( "## " ):
			flush( )
			current_tag = line[ 3: ].strip( )
		else:
			buffer.append( line )
	
	flush( )
	return "\n".join( out ).strip( )

def save_temp( upload ) -> str:
	"""Save uploaded file to a named temporary file and return path."""
	with tempfile.NamedTemporaryFile( delete=False ) as tmp:
		tmp.write( upload.read( ) )
		return tmp.name

def _extract_usage_from_response( resp: Any ) -> Dict[ str, int ]:
	"""
	Extract token usage from a response object/dict.
	Returns dict with prompt_tokens, completion_tokens, total_tokens.
	Defensive: returns zeros if not present.
	"""
	usage = {
			"prompt_tokens": 0,
			"completion_tokens": 0,
			"total_tokens": 0,
	}
	if not resp:
		return usage
	
	raw = None
	try:
		raw = getattr( resp, "usage", None )
	except Exception:
		raw = None
	
	if not raw and isinstance( resp, dict ):
		raw = resp.get( "usage" )
	
	if not raw:
		return usage
	
	try:
		if isinstance( raw, dict ):
			usage[ "prompt_tokens" ] = int( raw.get( "prompt_tokens", 0 ) )
			usage[ "completion_tokens" ] = int(
				raw.get( "completion_tokens", raw.get( "output_tokens", 0 ) )
			)
			usage[ "total_tokens" ] = int(
				raw.get(
					"total_tokens",
					usage[ "prompt_tokens" ] + usage[ "completion_tokens" ],
				)
			)
		else:
			usage[ "prompt_tokens" ] = int( getattr( raw, "prompt_tokens", 0 ) )
			usage[ "completion_tokens" ] = int(
				getattr( raw, "completion_tokens", getattr( raw, "output_tokens", 0 ) )
			)
			usage[ "total_tokens" ] = int(
				getattr(
					raw,
					"total_tokens",
					usage[ "prompt_tokens" ] + usage[ "completion_tokens" ],
				)
			)
	except Exception:
		usage[ "total_tokens" ] = (
				usage[ "prompt_tokens" ] + usage[ "completion_tokens" ]
		)
	
	return usage

def _update_token_counters( resp: Any ) -> None:
	"""
	Update session_state.last_call_usage and accumulate into session_state.token_usage.
	"""
	usage = _extract_usage_from_response( resp )
	st.session_state.last_call_usage = usage
	st.session_state.token_usage[ "prompt_tokens" ] += usage.get(
		"prompt_tokens", 0
	)
	st.session_state.token_usage[ "completion_tokens" ] += usage.get(
		"completion_tokens", 0
	)
	st.session_state.token_usage[ "total_tokens" ] += usage.get(
		"total_tokens", 0
	)

def _display_value( val: Any ) -> str:
	"""
	Render a friendly display string for header values.
	None -> em dash; otherwise str(value).
	"""
	if val is None:
		return "—"
	try:
		return str( val )
	except Exception:
		return "—"
# ======================================================================================
# SIDEBAR PROVIDER + MODE RESOLUTION (AUTHORITATIVE)
# ======================================================================================

PROVIDER_MODULES = {
    "GPT": "gpt",
    "Gemini": "gemini",
    "Groq": "grok",
}

MODE_CLASS_MAP = {
    "Text": ["Chat"],
    "Images": ["Image"],
    "Audio": ["TTS", "Translation", "Transcription"],
    "Embeddings": ["Embedding"],
}

def get_provider_module():
    provider = st.session_state.get("provider", "GPT")
    module_name = PROVIDER_MODULES.get(provider, "gpt")
    return __import__(module_name)

def instantiate_if_exists(module, class_name):
    return getattr(module, class_name, None)()

# ======================================================================================
# PROVIDER-AWARE OPTION SOURCING
# ======================================================================================

def _provider( ):
	return st.session_state.get( "provider", "GPT" )

def _safe( module, attr, fallback ):
	try:
		mod = __import__( module )
		return getattr( mod, attr, fallback )
	except Exception:
		return fallback

# ---------------- TEXT ----------------
def text_model_options( chat ):
	if _provider( ) == "Gemini":
		return _safe( "gemini", "model_options", chat.model_options )
	if _provider( ) == "Groq":
		return _safe( "grok", "model_options", chat.model_options )
	return chat.model_options

# ---------------- IMAGES ----------------
def image_model_options( image ):
	if _provider( ) == "Gemini":
		return _safe( "gemini", "image_model_options", image.model_options )
	if _provider( ) == "Groq":
		return _safe( "grok", "image_model_options", image.model_options )
	return image.model_options

def image_size_or_aspect_options( image ):
	if _provider( ) == "Gemini":
		return _safe( "gemini", "aspect_options", image.size_options )
	if _provider( ) == "Groq":
		return _safe( "grok", "aspect_options", image.size_options )
	return image.size_options

# ---------------- AUDIO ----------------
def audio_model_options( transcriber ):
	if _provider( ) == "Gemini":
		return _safe( "gemini", "audio_model_options", transcriber.model_options )
	if _provider( ) == "Groq":
		return _safe( "grok", "audio_model_options", transcriber.model_options )
	return transcriber.model_options

def audio_language_options( transcriber ):
	if _provider( ) == "Gemini":
		return _safe( "gemini", "language_options", transcriber.language_options )
	if _provider( ) == "Groq":
		return _safe( "grok", "language_options", transcriber.language_options )
	return transcriber.language_options

# ---------------- EMBEDDINGS ----------------
def embedding_model_options( embed ):
	if _provider( ) == "Gemini":
		return _safe( "gemini", "embedding_model_options", embed.model_options )
	if _provider( ) == "Groq":
		return _safe( "grok", "embedding_model_options", embed.model_options )
	return embed.model_options

# ======================================================================================
# Sidebar — Provider selector above Mode, then Mode selector.
# ======================================================================================
with st.sidebar:
    st.subheader("Provider")
    st.markdown(
        "<div style='height:2px;background:#0078FC;margin:6px 0 10px 0;'></div>",
        unsafe_allow_html=True,
    )

    provider = st.selectbox(
        "Choose provider",
        list(PROVIDER_MODULES.keys()),
        index=list(PROVIDER_MODULES.keys()).index(
            st.session_state.get("provider", "GPT")
        ),
    )
    st.session_state["provider"] = provider

    st.subheader("Mode")
    st.markdown(
        "<div style='height:2px;background:#0078FC;margin:6px 0 10px 0;'></div>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Select capability",
        ["Text", "Images", "Audio", "Embeddings"],
    )

# ======================================================================================
# Dynamic Header — show provider and mode, and model relevant to the active mode
# ======================================================================================
_mode_to_model_key = {
		"Text": "text_model",
		"Images": "image_model",
		"Audio": "audio_model",
		"Embeddings": "embed_model",
		"Documents": "text_model",
		"Files": "text_model",
		"Vector Store": "text_model",
}

model_key_for_header = _mode_to_model_key.get( mode, "text_model" )
model_val = st.session_state.get( model_key_for_header, None )
temperature_val = st.session_state.get( "temperature", None )
top_p_val = st.session_state.get( "top_p", None )
provider_val = st.session_state.get( "provider", None )

header_label = provider_val if provider_val else "Boo"

st.divider( )

# ======================================================================================
# TEXT MODE (Provider-correct, function-preserving rewrite)
# ======================================================================================
if mode == "Text":
    st.header("")

    # ------------------------------------------------------------------
    # Provider-aware Chat instantiation
    # ------------------------------------------------------------------
    provider_module = get_provider_module()
    chat = provider_module.Chat()

    # ------------------------------------------------------------------
    # Sidebar — Text Settings (NO functionality removed)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Text Settings")

        # ---------------- Model (provider-correct) ----------------
        text_model = st.selectbox(
            "Model",
            chat.model_options,
            index=(
                chat.model_options.index(st.session_state["text_model"])
                if st.session_state.get("text_model") in chat.model_options
                else 0
            ),
        )
        st.session_state["text_model"] = text_model

        # ---------------- Parameters (unchanged) ----------------
        with st.expander("Parameters:", expanded=True):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("temperature", 0.7)),
                step=0.01,
            )
            st.session_state["temperature"] = float(temperature)

            top_p = st.slider(
                "Top-P",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("top_p", 1.0)),
                step=0.01,
            )
            st.session_state["top_p"] = float(top_p)

            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=100000,
                value=int(st.session_state.get("max_tokens", 512)),
            )
            st.session_state["max_tokens"] = int(max_tokens)

            freq_penalty = st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=float(st.session_state.get("freq_penalty", 0.0)),
                step=0.01,
            )
            st.session_state["freq_penalty"] = float(freq_penalty)

            pres_penalty = st.slider(
                "Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=float(st.session_state.get("pres_penalty", 0.0)),
                step=0.01,
            )
            st.session_state["pres_penalty"] = float(pres_penalty)

            stop_text = st.text_area(
                "Stop Sequences (one per line)",
                value="\n".join(st.session_state.get("stop_sequences", [])),
                height=80,
            )
            st.session_state["stop_sequences"] = [
                s for s in stop_text.splitlines() if s.strip()
            ]

        # ---------------- Include options (unchanged) ----------------
        if mode == 'GPT':
	        include = st.multiselect("Include:", chat.include_options)
	        chat.include = include

    # ------------------------------------------------------------------
    # Main Chat UI (unchanged)
    # ------------------------------------------------------------------
    left, center, right = st.columns([1, 2, 1])

    with center:
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
                    gen_kwargs: Dict[str, Any] = {}

                    gen_kwargs["model"] = st.session_state["text_model"]
                    gen_kwargs["temperature"] = st.session_state["temperature"]
                    gen_kwargs["top_p"] = st.session_state["top_p"]
                    gen_kwargs["max_tokens"] = st.session_state["max_tokens"]
                    gen_kwargs["frequency_penalty"] = st.session_state["freq_penalty"]
                    gen_kwargs["presence_penalty"] = st.session_state["pres_penalty"]

                    if st.session_state["stop_sequences"]:
                        gen_kwargs["stop"] = st.session_state["stop_sequences"]

                    response = None
                    try:
                        response = chat.generate_text(
                            prompt=prompt, **gen_kwargs
                        )
                    except Exception as exc:
                        st.error(f"Generation Failed: {exc}")
                        response = None

                    st.markdown(response or "")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response or "",
                        }
                    )

                    try:
                        _update_token_counters(
                            getattr(chat, "response", None) or response
                        )
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Token usage display (unchanged)
    # ------------------------------------------------------------------
    lcu = st.session_state.last_call_usage
    tu = st.session_state.token_usage

    if any(lcu.values()):
        st.info(
            f"Last call — prompt: {lcu['prompt_tokens']}, "
            f"completion: {lcu['completion_tokens']}, "
            f"total: {lcu['total_tokens']}"
        )

    if tu["total_tokens"] > 0:
        st.write(
            f"Session totals — prompt: {tu['prompt_tokens']} · "
            f"completion: {tu['completion_tokens']} · "
            f"total: {tu['total_tokens']}"
        )

# ======================================================================================
# IMAGES MODE (Provider-correct, function-preserving rewrite)
# ======================================================================================
elif mode == "Images":

    # ------------------------------------------------------------------
    # Provider-aware Image instantiation
    # ------------------------------------------------------------------
    provider_module = get_provider_module()
    image = provider_module.Image()

    # ------------------------------------------------------------------
    # Sidebar — Image Settings (NO functionality removed)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Image Settings")

        # ---------------- Model (provider-correct) ----------------
        image_model = st.selectbox(
            "Model",
            image.model_options,
            index=(
                image.model_options.index(st.session_state["image_model"])
                if st.session_state.get("image_model") in image.model_options
                else 0
            ),
        )
        st.session_state["image_model"] = image_model

        # ---------------- Size / Aspect Ratio (provider-aware) ----------------
        if hasattr(image, "aspect_options"):
            size_or_aspect = st.selectbox(
                "Aspect Ratio",
                image.aspect_options,
            )
            size_arg = size_or_aspect
        else:
            size_or_aspect = st.selectbox(
                "Size",
                image.size_options,
            )
            size_arg = size_or_aspect

        # ---------------- Quality ----------------
        quality = None
        if hasattr(image, "quality_options"):
            quality = st.selectbox(
                "Quality",
                image.quality_options,
            )

        # ---------------- Format ----------------
        fmt = None
        if hasattr(image, "format_options"):
            fmt = st.selectbox(
                "Format",
                image.format_options,
            )

    # ------------------------------------------------------------------
    # Main UI — Tabs (Generate / Analyze) (unchanged)
    # ------------------------------------------------------------------
    tab_gen, tab_analyze = st.tabs(["Generate", "Analyze"])

    # ============================== GENERATE ===============================
    with tab_gen:
        prompt = st.text_area("Prompt")

        if st.button("Generate Image"):
            with st.spinner("Generating…"):
                try:
                    kwargs: Dict[str, Any] = {
                        "prompt": prompt,
                        "model": image_model,
                    }

                    # Provider-safe optional args
                    if size_arg is not None:
                        kwargs["size"] = size_arg
                    if quality is not None:
                        kwargs["quality"] = quality
                    if fmt is not None:
                        kwargs["fmt"] = fmt

                    img_url = image.generate(**kwargs)
                    st.image(img_url)

                    try:
                        _update_token_counters(
                            getattr(image, "response", None)
                        )
                    except Exception:
                        pass

                except Exception as exc:
                    st.error(f"Image generation failed: {exc}")

    # ============================== ANALYZE ===============================
    with tab_analyze:
        st.markdown("Image analysis — upload an image to analyze.")

        uploaded_img = st.file_uploader(
            "Upload an image for analysis",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="images_analyze_uploader",
        )

        if uploaded_img:
            tmp_path = save_temp(uploaded_img)

            st.image(
                uploaded_img,
                caption="Uploaded image preview",
                use_column_width=True,
            )

            # Discover available analysis methods on Image object
            available_methods = []
            for candidate in (
                "analyze",
                "describe_image",
                "describe",
                "classify",
                "detect_objects",
                "caption",
                "image_analysis",
            ):
                if hasattr(image, candidate):
                    available_methods.append(candidate)

            if available_methods:
                chosen_method = st.selectbox(
                    "Method",
                    available_methods,
                    index=0,
                )
            else:
                chosen_method = None
                st.info(
                    "No dedicated image analysis method found on Image object; "
                    "attempting generic handlers."
                )

            chosen_model = st.selectbox(
                "Model (analysis)",
                [image_model, None],
                index=0,
            )

            chosen_model_arg = (
                image_model if chosen_model is None else chosen_model
            )

            if st.button("Analyze Image"):
                with st.spinner("Analyzing image…"):
                    analysis_result = None
                    try:
                        if chosen_method:
                            func = getattr(image, chosen_method, None)
                            if func:
                                try:
                                    analysis_result = func(tmp_path)
                                except TypeError:
                                    analysis_result = func(
                                        tmp_path, model=chosen_model_arg
                                    )
                        else:
                            for fallback in (
                                "analyze",
                                "describe_image",
                                "describe",
                                "caption",
                            ):
                                if hasattr(image, fallback):
                                    func = getattr(image, fallback)
                                    try:
                                        analysis_result = func(tmp_path)
                                        break
                                    except Exception:
                                        continue

                        if analysis_result is None:
                            st.warning(
                                "No analysis output returned by the available methods."
                            )
                        else:
                            if isinstance(analysis_result, (dict, list)):
                                st.json(analysis_result)
                            else:
                                st.markdown("**Analysis result:**")
                                st.write(analysis_result)

                            try:
                                _update_token_counters(
                                    getattr(image, "response", None)
                                    or analysis_result
                                )
                            except Exception:
                                pass

                    except Exception as exc:
                        st.error(f"Analysis Failed: {exc}")

						
# ======================================================================================
# AUDIO MODE (Provider-correct, function-preserving rewrite)
# ======================================================================================
elif mode == "Audio":

    # ------------------------------------------------------------------
    # Provider-aware Audio instantiation
    # ------------------------------------------------------------------
    provider_module = get_provider_module()

    transcriber = None
    translator = None
    tts = None

    if hasattr(provider_module, "Transcription"):
        transcriber = provider_module.Transcription()
    if hasattr(provider_module, "Translation"):
        translator = provider_module.Translation()
    if hasattr(provider_module, "TTS"):
        tts = provider_module.TTS()

    # ------------------------------------------------------------------
    # Sidebar — Audio Settings (NO functionality removed)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Audio Settings")

        # ---------------- Task ----------------
        available_tasks = []
        if transcriber is not None:
            available_tasks.append("Transcribe")
        if translator is not None:
            available_tasks.append("Translate")
        if tts is not None:
            available_tasks.append("Text-to-Speech")

        if not available_tasks:
            st.info("Audio is not supported by the selected provider.")
            task = None
        else:
            task = st.selectbox("Task", available_tasks)

        # ---------------- Model (provider-correct) ----------------
        audio_model = None
        model_options = []

        if task == "Transcribe" and transcriber and hasattr(transcriber, "model_options"):
            model_options = transcriber.model_options
        elif task == "Translate" and translator and hasattr(translator, "model_options"):
            model_options = translator.model_options
        elif task == "Text-to-Speech" and tts and hasattr(tts, "model_options"):
            model_options = tts.model_options

        if model_options:
            audio_model = st.selectbox(
                "Model",
                model_options,
                index=(
                    model_options.index(st.session_state.get("audio_model"))
                    if st.session_state.get("audio_model") in model_options
                    else 0
                ),
            )
            st.session_state["audio_model"] = audio_model

        # ---------------- Language / Voice Options ----------------
        language = None
        voice = None

        if task in ("Transcribe", "Translate"):
            obj = transcriber if task == "Transcribe" else translator
            if obj and hasattr(obj, "language_options"):
                language = st.selectbox(
                    "Language",
                    obj.language_options,
                )

        if task == "Text-to-Speech" and tts:
            if hasattr(tts, "voice_options"):
                voice = st.selectbox(
                    "Voice",
                    tts.voice_options,
                )

    # ------------------------------------------------------------------
    # Main UI — Audio Input / Output (unchanged behavior)
    # ------------------------------------------------------------------
    left, center, right = st.columns([1, 2, 1])

    with center:
        if task in ("Transcribe", "Translate"):
            uploaded = st.file_uploader(
                "Upload audio file",
                type=["wav", "mp3", "m4a", "flac"],
            )

            if uploaded:
                tmp_path = save_temp(uploaded)

                if task == "Transcribe" and transcriber:
                    with st.spinner("Transcribing…"):
                        try:
                            text = transcriber.transcribe(
                                tmp_path,
                                model=audio_model,
                                language=language,
                            )
                            st.text_area("Transcript", value=text, height=300)

                            try:
                                _update_token_counters(
                                    getattr(transcriber, "response", None)
                                )
                            except Exception:
                                pass

                        except Exception as exc:
                            st.error(f"Transcription failed: {exc}")

                elif task == "Translate" and translator:
                    with st.spinner("Translating…"):
                        try:
                            text = translator.translate(
                                tmp_path,
                                model=audio_model,
                                language=language,
                            )
                            st.text_area("Translation", value=text, height=300)

                            try:
                                _update_token_counters(
                                    getattr(translator, "response", None)
                                )
                            except Exception:
                                pass

                        except Exception as exc:
                            st.error(f"Translation failed: {exc}")

        elif task == "Text-to-Speech" and tts:
            text = st.text_area("Text to synthesize")

            if text and st.button("Generate Audio"):
                with st.spinner("Synthesizing speech…"):
                    try:
                        audio_bytes = tts.speak(
                            text,
                            model=audio_model,
                            voice=voice,
                        )
                        st.audio(audio_bytes)

                        try:
                            _update_token_counters(
                                getattr(tts, "response", None)
                            )
                        except Exception:
                            pass

                    except Exception as exc:
                        st.error(f"Text-to-speech failed: {exc}")


# ======================================================================================
# EMBEDDINGS MODE (Provider-correct, function-preserving rewrite)
# ======================================================================================
elif mode == "Embeddings":

    # ------------------------------------------------------------------
    # Provider-aware Embedding instantiation
    # ------------------------------------------------------------------
    provider_module = get_provider_module()

    if not hasattr(provider_module, "Embedding"):
        st.info("Embeddings are not supported by the selected provider.")
    else:
        embed = provider_module.Embedding()

        # ------------------------------------------------------------------
        # Sidebar — Embedding Settings (NO functionality removed)
        # ------------------------------------------------------------------
        with st.sidebar:
            st.header("Embedding Settings")

            # ---------------- Model (provider-correct) ----------------
            embed_model = st.selectbox(
                "Model",
                embed.model_options,
                index=(
                    embed.model_options.index(st.session_state["embed_model"])
                    if st.session_state.get("embed_model") in embed.model_options
                    else 0
                ),
            )
            st.session_state["embed_model"] = embed_model

            # ---------------- Method (optional, provider-defined) ----------------
            method = None
            if hasattr(embed, "methods"):
                method = st.selectbox(
                    "Method",
                    embed.methods,
                )

        # ------------------------------------------------------------------
        # Main UI — Embedding execution (unchanged behavior)
        # ------------------------------------------------------------------
        left, center, right = st.columns([1, 2, 1])

        with center:
            text = st.text_area("Text to embed")

            if text and st.button("Embed"):
                with st.spinner("Embedding…"):
                    try:
                        if method:
                            vector = embed.create(
                                text,
                                model=embed_model,
                                method=method,
                            )
                        else:
                            vector = embed.create(
                                text,
                                model=embed_model,
                            )

                        st.write("Vector length:", len(vector))

                        try:
                            _update_token_counters(
                                getattr(embed, "response", None)
                            )
                        except Exception:
                            pass

                    except Exception as exc:
                        st.error(f"Embedding failed: {exc}")

# ======================================================================================
# Vector Store MODE
# ======================================================================================
elif mode == "Vector Store":
	try:
		chat  # type: ignore
	except NameError:
		chat = Chat( )
	
	vs_map = getattr( chat, "vector_stores", None )
	if vs_map and isinstance( vs_map, dict ):
		st.markdown( "**Known vector stores (local mapping)**" )
		for name, vid in vs_map.items( ):
			st.write( f"- **{name}** — `{vid}`" )
		st.markdown( "---" )
	
	with st.expander( "Create Vector Store", expanded=False ):
		new_store_name = st.text_input( "New store name" )
		if st.button( "Create store" ):
			if not new_store_name:
				st.warning( "Enter a store name." )
			else:
				try:
					if hasattr( chat, "create_store" ):
						res = chat.create_store( new_store_name )
						st.success(
							f"Create call submitted for '{new_store_name}'."
						)
					else:
						st.warning(
							"create_store method not found on chat object."
						)
				except Exception as exc:
					st.error( f"Create store failed: {exc}" )
	
	st.markdown( "**Manage Stores**" )
	options: List[ tuple ] = [ ]
	if vs_map and isinstance( vs_map, dict ):
		options = list( vs_map.items( ) )
	
	if not options:
		try:
			client = getattr( chat, "client", None )
			if (
					client
					and hasattr( client, "vector_stores" )
					and hasattr( client.vector_stores, "list" )
			):
				api_list = client.vector_stores.list( )
				temp: List[ tuple ] = [ ]
				for item in getattr( api_list, "data", [ ] ) or api_list:
					nm = getattr( item, "name", None ) or (
							item.get( "name" )
							if isinstance( item, dict )
							else None
					)
					vid = getattr( item, "id", None ) or (
							item.get( "id" )
							if isinstance( item, dict )
							else None
					)
					if nm and vid:
						temp.append( (nm, vid) )
				if temp:
					options = temp
		except Exception:
			options = [ ]
	
	if options:
		names = [ f"{n} — {i}" for n, i in options ]
		sel = st.selectbox( "Select a vector store", options=names )
		sel_id: Optional[ str ] = None
		sel_name: Optional[ str ] = None
		for n, i in options:
			label = f"{n} — {i}"
			if label == sel:
				sel_id = i
				sel_name = n
				break
		
		c1, c2 = st.columns( [ 1,
		                       1 ] )
		with c1:
			if st.button( "Retrieve store" ):
				try:
					if sel_id and hasattr( chat, "retrieve_store" ):
						vs = chat.retrieve_store( sel_id )
						st.json(
							vs.__dict__
							if hasattr( vs, "__dict__" )
							else vs
						)
					else:
						st.warning(
							"retrieve_store not available on chat object "
							"or no store selected."
						)
				except Exception as exc:
					st.error( f"Retrieve failed: {exc}" )
		
		with c2:
			if st.button( "Delete store" ):
				try:
					if sel_id and hasattr( chat, "delete_store" ):
						res = chat.delete_store( sel_id )
						st.success( f"Delete returned: {res}" )
					else:
						st.warning(
							"delete_store not available on chat object "
							"or no store selected."
						)
				except Exception as exc:
					st.error( f"Delete failed: {exc}" )
	else:
		st.info(
			"No vector stores discovered. Create one or confirm "
			"`chat.vector_stores` mapping exists."
		)

# ======================================================================================
# PROMPT ENGINEERING MODE — LEEROY-EQUIVALENT IMPLEMENTATION
# ======================================================================================
elif mode == "Prompt Engineering":
	import sqlite3
	import math
	
	DB_PATH = "stores/sqlite/datamodels/Data.db"
	TABLE = "Prompts"
	PAGE_SIZE = 10
	
	# ------------------------------------------------------------------
	# Session state (single source of truth)
	# ------------------------------------------------------------------
	st.session_state.setdefault( "pe_page", 1 )
	st.session_state.setdefault( "pe_search", "" )
	st.session_state.setdefault( "pe_sort_col", "PromptsId" )
	st.session_state.setdefault( "pe_sort_dir", "ASC" )
	st.session_state.setdefault( "pe_selected_id", None )
	
	st.session_state.setdefault( "pe_name", "" )
	st.session_state.setdefault( "pe_text", "" )
	st.session_state.setdefault( "pe_version", 1 )
	
	# ------------------------------------------------------------------
	# DB helpers
	# ------------------------------------------------------------------
	def get_conn( ):
		return sqlite3.connect( DB_PATH )
	
	def reset_selection( ):
		st.session_state.pe_selected_id = None
		st.session_state.pe_name = ""
		st.session_state.pe_text = ""
		st.session_state.pe_version = 1
	
	def load_prompt( pid: int ) -> None:
		with get_conn( ) as conn:
			cur = conn.execute(
				f"SELECT Name, Text, Version FROM {TABLE} WHERE PromptsId=?",
				(pid,),
			)
			row = cur.fetchone( )
			if row:
				st.session_state.pe_name = row[ 0 ]
				st.session_state.pe_text = row[ 1 ]
				st.session_state.pe_version = row[ 2 ]
	
	# ------------------------------------------------------------------
	# XML / Markdown converters (IDENTICAL BEHAVIOR TO LEEROY)
	# ------------------------------------------------------------------
	def xml_to_md( ):
		st.session_state.pe_text = xml_converter( st.session_state.pe_text )
	
	def md_to_xml( ):
		st.session_state.pe_text = markdown_converter( st.session_state.pe_text )
	
	# ------------------------------------------------------------------
	# Controls (table filters)
	# ------------------------------------------------------------------
	c1, c2, c3, c4 = st.columns( [ 4,
	                               2,
	                               2,
	                               3 ] )
	
	with c1:
		st.text_input( "Search (Name/Text contains)", key="pe_search" )
	
	with c2:
		st.selectbox(
			"Sort by",
			[ "PromptsId",
			  "Name",
			  "Version" ],
			key="pe_sort_col",
		)
	
	with c3:
		st.selectbox(
			"Direction",
			[ "ASC",
			  "DESC" ],
			key="pe_sort_dir",
		)
	
	with c4:
		st.markdown(
			"<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.25rem;'>Go to ID</div>",
			unsafe_allow_html=True,
		)
		a1, a2, a3 = st.columns( [ 2,
		                           1,
		                           1 ] )
		with a1:
			jump_id = st.number_input(
				"Go to ID",
				min_value=1,
				step=1,
				label_visibility="collapsed",
			)
		with a2:
			if st.button( "Go" ):
				st.session_state.pe_selected_id = int( jump_id )
				load_prompt( int( jump_id ) )
		with a3:
			if st.button( "Undo" ):
				reset_selection( )
	
	# ------------------------------------------------------------------
	# Load prompt table
	# ------------------------------------------------------------------
	where = ""
	params = [ ]
	
	if st.session_state.pe_search:
		where = "WHERE Name LIKE ? OR Text LIKE ?"
		s = f"%{st.session_state.pe_search}%"
		params.extend( [ s,
		                 s ] )
	
	offset = (st.session_state.pe_page - 1) * PAGE_SIZE
	
	query = f"""
        SELECT PromptsId, Name, Text, Version, ID
        FROM {TABLE}
        {where}
        ORDER BY {st.session_state.pe_sort_col} {st.session_state.pe_sort_dir}
        LIMIT {PAGE_SIZE} OFFSET {offset}
    """
	
	count_query = f"SELECT COUNT(*) FROM {TABLE} {where}"
	
	with get_conn( ) as conn:
		rows = conn.execute( query, params ).fetchall( )
		total_rows = conn.execute( count_query, params ).fetchone( )[ 0 ]
	
	total_pages = max( 1, math.ceil( total_rows / PAGE_SIZE ) )
	
	# ------------------------------------------------------------------
	# Prompt table (selection drives editor — NO DUPLICATE STATE)
	# ------------------------------------------------------------------
	table_rows = [ ]
	for r in rows:
		table_rows.append(
			{
					"Selected": r[ 0 ] == st.session_state.pe_selected_id,
					"PromptsId": r[ 0 ],
					"Name": r[ 1 ],
					"Version": r[ 3 ],
					"ID": r[ 4 ],
			}
		)
	
	edited = st.data_editor(
		table_rows,
		hide_index=True,
		use_container_width=True,
	)
	
	selected = [ r for r in edited if r.get( "Selected" ) ]
	if len( selected ) == 1:
		pid = selected[ 0 ][ "PromptsId" ]
		if pid != st.session_state.pe_selected_id:
			st.session_state.pe_selected_id = pid
			load_prompt( pid )
	
	# ------------------------------------------------------------------
	# Paging
	# ------------------------------------------------------------------
	p1, p2, p3 = st.columns( [ 1,
	                           2,
	                           1 ] )
	with p1:
		if st.button( "◀ Prev" ) and st.session_state.pe_page > 1:
			st.session_state.pe_page -= 1
	with p2:
		st.markdown( f"Page **{st.session_state.pe_page}** of **{total_pages}**" )
	with p3:
		if st.button( "Next ▶" ) and st.session_state.pe_page < total_pages:
			st.session_state.pe_page += 1
	
	st.divider( )
	
	# ------------------------------------------------------------------
	# Converter controls (SINGLE TEXT BUFFER — LEEROY STYLE)
	# ------------------------------------------------------------------
	with st.expander( "XML ↔ Markdown Converter", expanded=False ):
		b1, b2 = st.columns( 2 )
		with b1:
			st.button( "Convert XML → Markdown", on_click=xml_to_md )
		with b2:
			st.button( "Convert Markdown → XML", on_click=md_to_xml )
	
	# ------------------------------------------------------------------
	# Create / Edit Prompt (AUTHORITATIVE EDITOR)
	# ------------------------------------------------------------------
	with st.expander( "Create / Edit Prompt", expanded=True ):
		st.text_input(
			"PromptsId",
			value=st.session_state.pe_selected_id or "",
			disabled=True,
		)
		st.text_input( "Name", key="pe_name" )
		st.text_area( "Text", key="pe_text", height=260 )
		st.number_input( "Version", min_value=1, key="pe_version" )
		
		c1, c2, c3 = st.columns( 3 )
		
		with c1:
			if st.button( "Save Changes" if st.session_state.pe_selected_id else "Create Prompt" ):
				with get_conn( ) as conn:
					if st.session_state.pe_selected_id:
						conn.execute(
							f"""
                            UPDATE {TABLE}
                            SET Name=?, Text=?, Version=?
                            WHERE PromptsId=?
                            """,
							(
									st.session_state.pe_name,
									st.session_state.pe_text,
									st.session_state.pe_version,
									st.session_state.pe_selected_id,
							),
						)
					else:
						conn.execute(
							f"""
                            INSERT INTO {TABLE} (Name, Text, Version)
                            VALUES (?, ?, ?)
                            """,
							(
									st.session_state.pe_name,
									st.session_state.pe_text,
									st.session_state.pe_version,
							),
						)
					conn.commit( )
				st.success( "Saved." )
				reset_selection( )
		
		with c2:
			if st.session_state.pe_selected_id and st.button( "Delete" ):
				with get_conn( ) as conn:
					conn.execute(
						f"DELETE FROM {TABLE} WHERE PromptsId=?",
						(st.session_state.pe_selected_id,),
					)
					conn.commit( )
				reset_selection( )
				st.success( "Deleted." )
		
		with c3:
			if st.button( "Clear Selection" ):
				reset_selection( )

# ======================================================================================
# DOCUMENTS MODE
# ======================================================================================
if mode == "Documents":
	uploaded = st.file_uploader(
		"Upload documents (session only)",
		type=[ "pdf",
		       "txt",
		       "md",
		       "docx" ],
		accept_multiple_files=True,
	)
	
	if uploaded:
		for up in uploaded:
			st.session_state.files.append( save_temp( up ) )
		st.success( f"Saved {len( uploaded )} file(s) to session" )
	
	if st.session_state.files:
		st.markdown( "**Uploaded documents (session-only)**" )
		idx = st.selectbox(
			"Choose a document",
			options=list( range( len( st.session_state.files ) ) ),
			format_func=lambda i: st.session_state.files[ i ],
		)
		selected_path = st.session_state.files[ idx ]
		
		c1, c2 = st.columns( [ 1,
		                       1 ] )
		with c1:
			if st.button( "Remove selected document" ):
				removed = st.session_state.files.pop( idx )
				st.success( f"Removed {removed}" )
		with c2:
			if st.button( "Show selected path" ):
				st.info( f"Local temp path: {selected_path}" )
		
		st.markdown( "---" )
		question = st.text_area(
			"Ask a question about the selected document"
		)
		if st.button( "Ask Document" ):
			if not question:
				st.warning(
					"Enter a question before asking."
				)
			else:
				with st.spinner( "Running document Q&A…" ):
					try:
						try:
							chat  # type: ignore
						except NameError:
							chat = Chat( )
						answer = None
						if hasattr( chat, "summarize_document" ):
							try:
								answer = chat.summarize_document(
									prompt=question,
									pdf_path=selected_path,
								)
							except TypeError:
								answer = chat.summarize_document(
									question, selected_path
								)
						elif hasattr( chat, "ask_document" ):
							answer = chat.ask_document(
								selected_path, question
							)
						elif hasattr( chat, "document_qa" ):
							answer = chat.document_qa(
								selected_path, question
							)
						else:
							raise RuntimeError(
								"No document-QA method found on chat object."
							)
						
						st.markdown( "**Answer:**" )
						st.markdown( answer or "No answer returned." )
						
						st.session_state.messages.append(
							{
									"role": "user",
									"content": f"[Document question] {question}",
							}
						)
						st.session_state.messages.append(
							{
									"role": "assistant",
									"content": answer or "",
							}
						)
						
						try:
							_update_token_counters(
								getattr( chat, "response", None )
								or answer
							)
						except Exception:
							pass
					except Exception as e:
						st.error(
							f"Document Q&A failed: {e}"
						)
	else:
		st.info(
			"No client-side documents uploaded this session. "
			"Use the uploader in the sidebar to add files."
		)

# ======================================================================================
# FILES API MODE
# ======================================================================================
if mode == "Files":
	try:
		chat  # type: ignore
	except NameError:
		chat = Chat( )
	
	list_method = None
	for name in (
				"retrieve_files",
				"retreive_files",
				"list_files",
				"get_files",
	):
		if hasattr( chat, name ):
			list_method = getattr( chat, name )
			break
	
	uploaded_file = st.file_uploader(
		"Upload file (server-side via Files API)",
		type=[
				"pdf",
				"txt",
				"md",
				"docx",
				"png",
				"jpg",
				"jpeg",
		],
	)
	if uploaded_file:
		tmp_path = save_temp( uploaded_file )
		upload_fn = None
		for name in ("upload_file", "upload", "files_upload"):
			if hasattr( chat, name ):
				upload_fn = getattr( chat, name )
				break
		if not upload_fn:
			st.warning(
				"No upload function found on chat object (upload_file)."
			)
		else:
			with st.spinner( "Uploading to Files API..." ):
				try:
					fid = upload_fn( tmp_path )
					st.success( f"Uploaded; file id: {fid}" )
				except Exception as exc:
					st.error( f"Upload failed: {exc}" )
	
	if st.button( "List files" ):
		if not list_method:
			st.warning(
				"No file-listing method found on chat object."
			)
		else:
			with st.spinner( "Listing files..." ):
				try:
					files_resp = list_method( )
					files_list = [ ]
					if files_resp is None:
						files_list = [ ]
					elif isinstance( files_resp, dict ):
						files_list = (
								files_resp.get( "data" )
								or files_resp.get( "files" )
								or [ ]
						)
					elif isinstance( files_resp, list ):
						files_list = files_resp
					else:
						try:
							files_list = getattr(
								files_resp, "data", files_resp
							)
						except Exception:
							files_list = [ files_resp ]
					
					rows = [ ]
					for f in files_list:
						try:
							fid = (
									f.get( "id" )
									if isinstance( f, dict )
									else getattr( f, "id", None )
							)
							name = (
									f.get( "filename" )
									if isinstance( f, dict )
									else getattr(
										f, "filename", None
									)
							)
							purpose = (
									f.get( "purpose" )
									if isinstance( f, dict )
									else getattr(
										f, "purpose", None
									)
							)
						except Exception:
							fid = None
							name = str( f )
							purpose = None
						rows.append(
							{
									"id": fid,
									"filename": name,
									"purpose": purpose,
							}
						)
					if rows:
						st.table( rows )
					else:
						st.info( "No files returned." )
				except Exception as exc:
					st.error( f"List files failed: {exc}" )
	
	if "files_list" in locals( ) and files_list:
		file_ids = [
				r.get( "id" )
				if isinstance( r, dict )
				else getattr( r, "id", None )
				for r in files_list
		]
		sel = st.selectbox(
			"Select file id to delete", options=file_ids
		)
		if st.button( "Delete selected file" ):
			del_fn = None
			for name in ("delete_file", "delete", "files_delete"):
				if hasattr( chat, name ):
					del_fn = getattr( chat, name )
					break
			if not del_fn:
				st.warning(
					"No delete function found on chat object."
				)
			else:
				with st.spinner( "Deleting file..." ):
					try:
						res = del_fn( sel )
						st.success( f"Delete result: {res}" )
					except Exception as exc:
						st.error( f"Delete failed: {exc}" )

# ======================================================================================
# Footer — Fixed Bottom Status Bar (Desktop-style)
# ======================================================================================

# ---- Add bottom padding so content is not hidden behind footer
st.markdown(
    """
    <style>
    .block-container {
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Fixed footer container
st.markdown(
    """
    <style>
    .boo-status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(17, 17, 17, 0.95);
        border-top: 1px solid #2a2a2a;
        padding: 6px 16px;
        font-size: 0.85rem;
        color: #9aa0a6;
        z-index: 1000;
    }
    .boo-status-inner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Resolve active model by mode
_mode_to_model_key = {
    "Text": "text_model",
    "Images": "image_model",
    "Audio": "audio_model",
    "Embeddings": "embed_model",
}

provider_val = st.session_state.get("provider", "—")
mode_val = mode or "—"

active_model = st.session_state.get(
    _mode_to_model_key.get(mode, ""),
    None,
)

# ---- Build right-side (mode-gated)
right_parts = []

if active_model is not None:
    right_parts.append(f"Model: {active_model}")

if mode == "Text":
    temperature = st.session_state.get("temperature")
    top_p = st.session_state.get("top_p")

    if temperature is not None:
        right_parts.append(f"Temp: {temperature}")
    if top_p is not None:
        right_parts.append(f"Top-P: {top_p}")

elif mode == "Images":
    size = st.session_state.get("image_size")
    aspect = st.session_state.get("image_aspect")

    if aspect is not None:
        right_parts.append(f"Aspect: {aspect}")
    elif size is not None:
        right_parts.append(f"Size: {size}")

elif mode == "Audio":
    task = st.session_state.get("audio_task")
    if task is not None:
        right_parts.append(f"Task: {task}")

elif mode == "Embeddings":
    method = st.session_state.get("embed_method")
    if method is not None:
        right_parts.append(f"Method: {method}")

right_text = " · ".join(right_parts) if right_parts else "—"

# ---- Render footer
st.markdown(
    f"""
    <div class="boo-status-bar">
        <div class="boo-status-inner">
            <span>{provider_val} — {mode_val}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

