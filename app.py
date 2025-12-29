# ******************************************************************************************
# Assembly:                Boo
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Created:                 12-16-2025
# Notes:                   Gemini-only implementation (replaces gpt.py integration).
#                          Prompt Engineering mode retained (SQLite CRUD + paging).
# ******************************************************************************************

from __future__ import annotations

import config as cfg
import os
import sqlite3
import tempfile
import math

import streamlit as st
from typing import List, Dict, Any, Optional

from gpt import (
	Chat,
	Images,
	Embedding,
	VectorStore,
	Transcription,
	Translation,
	TTS,
)

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config(
	page_title="Boo",
	page_icon=cfg.FAVICON_PATH,
	layout="wide",
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
			"total_tokens": 0 }

if "token_usage" not in st.session_state:
	st.session_state.token_usage = {
			"prompt_tokens": 0,
			"completion_tokens": 0,
			"total_tokens": 0 }

if "files" not in st.session_state:
	st.session_state.files: List[ str ] = [ ]

# Per-mode model keys (deterministic header behavior)
if "text_model" not in st.session_state:
	st.session_state[ "text_model" ] = None
if "image_model" not in st.session_state:
	st.session_state[ "image_model" ] = None
if "audio_model" not in st.session_state:
	st.session_state[ "audio_model" ] = None
if "embed_model" not in st.session_state:
	st.session_state[ "embed_model" ] = None

# Gemini API version
if "gemini_version" not in st.session_state:
	st.session_state[ "gemini_version" ] = getattr( cfg, "GOOGLE_API_VERSION", "v1alpha" )

# Temperature / top_p / other generation params defaults (Text controls)
if "temperature" not in st.session_state:
	st.session_state[ "temperature" ] = 0.7
if "top_p" not in st.session_state:
	st.session_state[ "top_p" ] = 1.0
if "max_tokens" not in st.session_state:
	st.session_state[ "max_tokens" ] = 512

# Optional knobs (Gemini wrappers expose these as attributes)
if "candidate_count" not in st.session_state:
	st.session_state[ "candidate_count" ] = 1

# Session-only API Key override (Gemini only)
if "gemini_api_key" not in st.session_state:
	st.session_state[ "gemini_api_key" ] = ""

# ======================================================================================
# Utilities
# ======================================================================================
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
			"total_tokens": 0 }
	if not resp:
		return usage

	raw = None
	try:
		raw = getattr( resp, "usage", None )
	except Exception:
		raw = None

	if not raw and isinstance( resp, dict ):
		raw = resp.get( "usage" )

	# Gemini SDK commonly uses "usage_metadata"
	if not raw and isinstance( resp, dict ):
		raw = resp.get( "usage_metadata" )

	if not raw:
		try:
			raw = getattr( resp, "usage_metadata", None )
		except Exception:
			raw = None

	if not raw:
		return usage

	try:
		if isinstance( raw, dict ):
			usage[ "prompt_tokens" ] = int( raw.get( "prompt_tokens", raw.get( "input_tokens", 0 ) ) )
			usage[ "completion_tokens" ] = int(
				raw.get( "completion_tokens", raw.get( "output_tokens", 0 ) )
			)
			usage[ "total_tokens" ] = int(
				raw.get( "total_tokens", usage[ "prompt_tokens" ] + usage[ "completion_tokens" ] )
			)
		else:
			usage[ "prompt_tokens" ] = int( getattr( raw, "prompt_tokens", getattr( raw, "input_tokens", 0 ) ) )
			usage[ "completion_tokens" ] = int(
				getattr( raw, "completion_tokens", getattr( raw, "output_tokens", 0 ) )
			)
			usage[ "total_tokens" ] = int(
				getattr( raw, "total_tokens", usage[ "prompt_tokens" ] + usage[ "completion_tokens" ] )
			)
	except Exception:
		usage[ "total_tokens" ] = usage[ "prompt_tokens" ] + usage[ "completion_tokens" ]

	return usage

def _update_token_counters( resp: Any ) -> None:
	"""
	Update session_state.last_call_usage and accumulate into session_state.token_usage.
	"""
	usage = _extract_usage_from_response( resp )
	st.session_state.last_call_usage = usage
	st.session_state.token_usage[ "prompt_tokens" ] += usage.get( "prompt_tokens", 0 )
	st.session_state.token_usage[ "completion_tokens" ] += usage.get( "completion_tokens", 0 )
	st.session_state.token_usage[ "total_tokens" ] += usage.get( "total_tokens", 0 )

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

def resolve_gemini_api_key( ) -> Optional[ str ]:
	"""
	Resolve Gemini API key using the following precedence:
	1) Session override (user-entered)
	2) config.py default
	3) Environment variable (optional fallback)
	"""
	session_key = st.session_state.get( "gemini_api_key" )
	if session_key:
		return session_key

	cfg_key = getattr( cfg, "GOOGLE_API_KEY", None )
	if cfg_key:
		return cfg_key

	return os.environ.get( "GOOGLE_API_KEY" )

def _apply_gemini_runtime_config( ) -> None:
	"""
	Ensure Gemini client initializes in API-key mode (not Vertex AI).

	This avoids: "Project/location and API key are mutually exclusive in the client initializer."
	"""
	key = resolve_gemini_api_key( )
	if key:
		os.environ[ "GOOGLE_API_KEY" ] = key

	# Ensure project/location do not get passed when using API key mode.
	# gemini.py reads these from the shared config module at runtime.
	try:
		setattr( cfg, "GOOGLE_CLOUD_PROJECT", None )
	except Exception:
		pass
	try:
		setattr( cfg, "GOOGLE_CLOUD_LOCATION", None )
	except Exception:
		pass

# ======================================================================================
# Sidebar — Mode selector + Session controls + Gemini key/version controls
# ======================================================================================
with st.sidebar:
	st.header( "Mode" )

	# thin blue strip directly under "Mode"
	st.markdown(
		"""
		<div style="height:2px;border-radius:3px;background:#0078FC;margin:6px 0 10px 0;"></div>
		""",
		unsafe_allow_html=True,
	)

	mode = st.radio(
		"Select capability",
		[ "Text",
		  "Images",
		  "Audio",
		  "Embeddings",
		  "Documents",
		  "Files",
		  "Prompt Engineering" ],
	)

	# Horizontal session controls (short buttons)
	c1, c2 = st.columns( [ 1, 1 ] )
	with c1:
		if st.button( "Clear", key="session_clear_btn", use_container_width=True ):
			st.session_state.messages.clear( )
			st.success( "Cleared!" )
	with c2:
		if st.button( "New", key="session_new_btn", use_container_width=True ):
			st.session_state.messages.clear( )
			st.session_state.files.clear( )
			st.session_state.token_usage = {
					"prompt_tokens": 0,
					"completion_tokens": 0,
					"total_tokens": 0 }
			st.session_state.last_call_usage = {
					"prompt_tokens": 0,
					"completion_tokens": 0,
					"total_tokens": 0 }
			st.success( "Created!" )

	# second thin blue divider (restored)
	st.markdown(
		"""
		<div style="height:2px;border-radius:4px;background:#0078FC;margin:12px 0;"></div>
		""",
		unsafe_allow_html=True,
	)

	# Gemini API version + key (session override)
	st.subheader( "Gemini Settings" )

	# Version options from wrapper (fallback to config value)
	try:
		_tmp = Chat( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )
		version_options = getattr( _tmp, "version_options", [ st.session_state.get( "gemini_version", "v1alpha" ) ] )
	except Exception:
		version_options = [ st.session_state.get( "gemini_version", "v1alpha" ) ]

	st.selectbox(
		"API Version",
		options=version_options,
		key="gemini_version",
	)

	api_key_input = st.text_input(
		"Google API Key",
		type="password",
		value=st.session_state.get( "gemini_api_key", "" ) or "",
		help="Stored for this session only. Not written to disk.",
	)
	st.session_state[ "gemini_api_key" ] = api_key_input

	if resolve_gemini_api_key( ):
		source = "Session override" if st.session_state.get( "gemini_api_key" ) else "config.py / env"
		st.caption( f"Using API key from: {source}" )
	else:
		st.warning( "No API key found. Set GOOGLE_API_KEY in config.py or enter one above." )

# Always apply runtime Gemini config before any client instantiation
_apply_gemini_runtime_config( )

# ======================================================================================
# Dynamic Header — show mode and model relevant to the active mode
# ======================================================================================
_mode_to_model_key = {
		"Text": "text_model",
		"Images": "image_model",
		"Audio": "audio_model",
		"Embeddings": "embed_model",
		"Documents": "text_model",
		"Files": "text_model",
		"Prompt Engineering": "text_model",
}

model_key_for_header = _mode_to_model_key.get( mode, "text_model" )
model_val = st.session_state.get( model_key_for_header, None )
temperature_val = st.session_state.get( "temperature", None )
top_p_val = st.session_state.get( "top_p", None )

st.markdown(
	f"""
    <div style="margin-bottom:0.25rem;">
      <h3 style="margin:0;">Boo — {mode}</h3>
      <div style="color:#9aa0a6; margin-top:6px; font-size:0.95rem;">
        Model: {_display_value( model_val )} &nbsp;&nbsp;|&nbsp;&nbsp; Temp: {_display_value( temperature_val )} &nbsp;&nbsp;•&nbsp;&nbsp; Top-P: {_display_value( top_p_val )}
      </div>
    </div>
    """,
	unsafe_allow_html=True,
)
st.divider( )

# ======================================================================================
# TEXT MODE
# ======================================================================================
if mode == "Text":
	st.header( "" )

	chat = Chat( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )

	with st.sidebar:
		st.header( "Text Settings" )

		text_model = st.selectbox( "Model", chat.model_options )
		st.session_state[ "text_model" ] = text_model

		with st.expander( "Parameters:", expanded=True ):
			temperature = st.slider(
				"Temperature",
				min_value=0.0,
				max_value=2.0,
				value=float( st.session_state.get( "temperature", 0.7 ) ),
				step=0.01,
			)
			st.session_state[ "temperature" ] = float( temperature )

			top_p = st.slider(
				"Top-P",
				min_value=0.0,
				max_value=1.0,
				value=float( st.session_state.get( "top_p", 1.0 ) ),
				step=0.01,
			)
			st.session_state[ "top_p" ] = float( top_p )

			max_tokens = st.number_input(
				"Max Tokens",
				min_value=1,
				max_value=100000,
				value=int( st.session_state.get( "max_tokens", 512 ) ),
			)
			st.session_state[ "max_tokens" ] = int( max_tokens )

			candidate_count = st.number_input(
				"Candidates",
				min_value=1,
				max_value=8,
				value=int( st.session_state.get( "candidate_count", 1 ) ),
			)
			st.session_state[ "candidate_count" ] = int( candidate_count )

		include = st.multiselect( "Include:", getattr( chat, "include_options", [ ] ) )
		try:
			chat.include = include
		except Exception:
			pass

	left, center, right = st.columns( [ 1, 2, 1 ] )

	with center:
		for msg in st.session_state.messages:
			with st.chat_message( msg[ "role" ] ):
				st.markdown( msg[ "content" ] )

		prompt = st.chat_input( "Ask Boo something…" )

		if prompt:
			st.session_state.messages.append( {
					"role": "user",
					"content": prompt } )

			with st.chat_message( "assistant" ):
				with st.spinner( "Thinking…" ):
					# Map UI controls to Gemini wrapper public properties
					chat.model = text_model
					chat.temperature = st.session_state.get( "temperature", 0.7 )
					chat.top_p = st.session_state.get( "top_p", 1.0 )
					chat.max_tokens = st.session_state.get( "max_tokens", 512 )
					chat.number = st.session_state.get( "candidate_count", 1 )

					response = None
					try:
						response = chat.generate_text( prompt=prompt )
					except Exception as exc:
						st.error( f"Generation Failed: {exc}" )
						response = None

					st.markdown( response or "" )
					st.session_state.messages.append( {
							"role": "assistant",
							"content": response or "" } )

					try:
						_update_token_counters( getattr( chat, "usage", None ) or getattr( chat, "response", None ) or response )
					except Exception:
						pass

	lcu = st.session_state.last_call_usage
	tu = st.session_state.token_usage
	if any( lcu.values( ) ):
		st.info(
			f"Last call — prompt: {lcu[ 'prompt_tokens' ]}, completion: "
			f"{lcu[ 'completion_tokens' ]}, total: {lcu[ 'total_tokens' ]}"
		)
	if tu[ "total_tokens" ] > 0:
		st.write(
			f"Session totals — prompt: {tu[ 'prompt_tokens' ]} · completion: "
			f"{tu[ 'completion_tokens' ]} · total: {tu[ 'total_tokens' ]}"
		)

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == "Images":
	img = Images( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )

	with st.sidebar:
		st.header( "Image Settings" )
		image_model = st.selectbox( "Model", img.model_options )
		st.session_state[ "image_model" ] = image_model

		aspect = st.selectbox( "Aspect", img.aspect_options )
		n = st.number_input( "Number", min_value=1, max_value=8, value=int( getattr( img, "number", 1 ) ) )

	tab_gen, tab_analyze = st.tabs( [ "Generate", "Analyze" ] )

	with tab_gen:
		prompt = st.text_area( "Prompt" )
		if st.button( "Generate Image" ):
			with st.spinner( "Generating…" ):
				try:
					img.model = image_model
					img.aspect = aspect
					img.number = int( n )

					# Images.generate returns a list of PIL images (per wrapper)
					images = img.generate( prompt=prompt, aspect=aspect )
					if not images:
						st.warning( "No images returned." )
					else:
						for im in images:
							st.image( im, use_container_width=True )

					try:
						_update_token_counters( getattr( img, "usage", None ) or getattr( img, "response", None ) )
					except Exception:
						pass
				except Exception as exc:
					st.error( f"Image generation failed: {exc}" )

	with tab_analyze:
		st.markdown( "Image analysis — upload an image to analyze." )
		uploaded_img = st.file_uploader(
			"Upload an image for analysis",
			type=[ "png", "jpg", "jpeg", "webp" ],
			accept_multiple_files=False,
			key="images_analyze_uploader",
		)

		if uploaded_img:
			tmp_path = save_temp( uploaded_img )
			st.image( uploaded_img, caption="Uploaded image preview", use_container_width=True )
			st.info( "Gemini image analysis is not exposed by this wrapper. (Generate-only)" )

# ======================================================================================
# AUDIO MODE
# ======================================================================================
elif mode == "Audio":
	transcriber = Transcription( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )
	translator = Translation( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )
	tts = TTS( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )

	with st.sidebar:
		st.header( "Audio Settings" )

		task = st.selectbox( "Task", [ "Transcribe", "Translate", "Text-to-Speech" ] )

		if task in ( "Transcribe", "Translate" ):
			audio_model = st.selectbox( "Model", transcriber.model_options )
			st.session_state[ "audio_model" ] = audio_model

			language = st.selectbox( "Language", transcriber.language_options )
		else:
			tts_model = st.selectbox( "Model", tts.model_options )
			voice = st.selectbox( "Voice", tts.voice_options )
			speed = st.slider( "Speed", min_value=0.25, max_value=4.0, value=float( tts.speed ), step=0.05 )

	left, center, right = st.columns( [ 1, 2, 1 ] )

	with center:
		if task in ( "Transcribe", "Translate" ):
			uploaded = st.file_uploader( "Upload audio file", type=[ "wav", "mp3", "m4a", "flac" ] )
			if uploaded:
				tmp = save_temp( uploaded )
				if task == "Transcribe":
					with st.spinner( "Transcribing…" ):
						try:
							transcriber.model = st.session_state.get( "audio_model" )
							transcriber.language = language
							text = transcriber.transcribe( tmp )
							st.text_area( "Transcript", value=text, height=300 )
						except Exception as exc:
							st.error( f"Transcription failed: {exc}" )
				else:
					with st.spinner( "Translating…" ):
						try:
							translator.model = st.session_state.get( "audio_model" )
							translator.language = language
							text = translator.translate( tmp )
							st.text_area( "Translation", value=text, height=300 )
						except Exception as exc:
							st.error( f"Translation failed: {exc}" )
		else:
			text_in = st.text_area( "Text" )
			if st.button( "Generate Speech" ):
				with st.spinner( "Generating…" ):
					try:
						tts.model = tts_model
						tts.voice = voice
						tts.speed = float( speed )
						audio_bytes = tts.generate( text_in )
						if audio_bytes:
							st.audio( audio_bytes, format="audio/wav" )
						else:
							st.warning( "No audio returned." )
					except Exception as exc:
						st.error( f"TTS failed: {exc}" )

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == "Embeddings":
	embed = Embedding( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )

	with st.sidebar:
		st.header( "Embedding Settings" )
		embed_model = st.selectbox( "Model", embed.model_options )
		st.session_state[ "embed_model" ] = embed_model

	left, center, right = st.columns( [ 1, 2, 1 ] )

	with center:
		text = st.text_area( "Text to embed" )
		if st.button( "Embed" ):
			with st.spinner( "Embedding…" ):
				try:
					embed.model = embed_model
					v = embed.generate( text )
					st.write( "Vector length:", len( v ) )
				except Exception as exc:
					st.error( f"Embedding failed: {exc}" )

# ======================================================================================
# FILES MODE (Gemini FileStore)
# ======================================================================================
# ======================================================================================
# FILE STORE MODE (Gemini)
# ======================================================================================
elif mode == "Files":
	try:
		chat  # type: ignore
	except NameError:
		chat = Chat()

	st.subheader("Gemini File Store")

	# --------------------------------------------------
	# Upload File
	# --------------------------------------------------
	uploaded_file = st.file_uploader(
		"Upload file to Gemini",
		type=[
			"pdf", "txt", "md", "docx",
			"png", "jpg", "jpeg",
			"csv", "json"
		],
	)

	if uploaded_file:
		tmp_path = save_temp(uploaded_file)

		upload_fn = None
		for name in ("upload_file", "upload", "files_upload", "create_file"):
			if hasattr(chat, name):
				upload_fn = getattr(chat, name)
				break

		if not upload_fn:
			st.error("No file upload method found on Gemini client.")
		else:
			with st.spinner("Uploading to Gemini File Store…"):
				try:
					file_id = upload_fn(tmp_path)
					st.success(f"Uploaded successfully. File ID: {file_id}")
				except Exception as exc:
					st.error(f"Upload failed: {exc}")

	st.divider()

	# --------------------------------------------------
	# List Files
	# --------------------------------------------------
	list_fn = None
	for name in ("list_files", "retrieve_files", "files_list"):
		if hasattr(chat, name):
			list_fn = getattr(chat, name)
			break

	files_list = []

	if list_fn:
		if st.button("List Files"):
			with st.spinner("Retrieving files…"):
				try:
					resp = list_fn()

					if isinstance(resp, dict):
						files_list = resp.get("files") or resp.get("data") or []
					elif isinstance(resp, list):
						files_list = resp
					else:
						files_list = getattr(resp, "files", []) or getattr(resp, "data", [])

				except Exception as exc:
					st.error(f"List files failed: {exc}")
	else:
		st.warning("No file listing method found on Gemini client.")

	# --------------------------------------------------
	# Display Files
	# --------------------------------------------------
	if files_list:
		rows = []
		for f in files_list:
			try:
				rows.append({
					"id": f.get("id") if isinstance(f, dict) else getattr(f, "id", None),
					"name": f.get("name") if isinstance(f, dict) else getattr(f, "name", None),
					"mime_type": f.get("mime_type") if isinstance(f, dict) else getattr(f, "mime_type", None),
					"size": f.get("size") if isinstance(f, dict) else getattr(f, "size", None),
				})
			except Exception:
				rows.append({"id": str(f)})

		st.dataframe(rows, use_container_width=True)

		st.divider()

		# --------------------------------------------------
		# Delete File
		# --------------------------------------------------
		file_ids = [r["id"] for r in rows if r.get("id")]
		if file_ids:
			selected_id = st.selectbox("Select file to delete", file_ids)

			delete_fn = None
			for name in ("delete_file", "files_delete", "remove_file"):
				if hasattr(chat, name):
					delete_fn = getattr(chat, name)
					break

			if st.button("Delete Selected File"):
				if not delete_fn:
					st.error("No delete method found on Gemini client.")
				else:
					with st.spinner("Deleting file…"):
						try:
							delete_fn(selected_id)
							st.success("File deleted.")
							st.rerun()
						except Exception as exc:
							st.error(f"Delete failed: {exc}")
	else:
		st.info("No files found in Gemini File Store.")


# ======================================================================================
# PROMPT ENGINEERING MODE
# ======================================================================================
elif mode == "Prompt Engineering":
	DB_PATH = "stores/sqlite/datamodels/Data.db"
	TABLE = "Prompts"
	PAGE_SIZE = 10

	# ----------------------------
	# Session state initialization
	# ----------------------------
	st.session_state.setdefault( "pe_page", 1 )
	st.session_state.setdefault( "pe_search", "" )
	st.session_state.setdefault( "pe_sort_col", "PromptsId" )
	st.session_state.setdefault( "pe_sort_dir", "ASC" )
	st.session_state.setdefault( "pe_selected_id", None )
	st.session_state.setdefault( "pe_jump_id", 1 )

	# ----------------------------
	# Helpers
	# ----------------------------
	def get_conn( ):
		return sqlite3.connect( DB_PATH )

	def reset_controls( ):
		st.session_state.pe_page = 1
		st.session_state.pe_search = ""
		st.session_state.pe_sort_col = "PromptsId"
		st.session_state.pe_sort_dir = "ASC"
		st.session_state.pe_selected_id = None
		st.session_state.pe_jump_id = 1

	# ----------------------------
	# Control bar (ABOVE TABLE)
	# ----------------------------
	c1, c2, c3, c4 = st.columns( [ 4, 2, 2, 3 ] )

	with c1:
		st.text_input(
			"Search (Name/Text contains)",
			key="pe_search",
		)

	with c2:
		st.selectbox(
			"Sort by",
			[ "PromptsId", "Name", "Version" ],
			key="pe_sort_col",
		)

	with c3:
		st.selectbox(
			"Direction",
			[ "ASC", "DESC" ],
			key="pe_sort_dir",
		)

	with c4:
		st.markdown(
			"<div style='font-size:0.95rem; font-weight:600; margin-bottom:0.25rem;'>Go to ID</div>",
			unsafe_allow_html=True,
		)

		a1, a2, a3 = st.columns( [ 2, 1, 1 ] )

		with a1:
			jump_id = st.number_input(
				"Go to ID",
				min_value=1,
				step=1,
				key="pe_jump_id",
				label_visibility="collapsed",
			)

		with a2:
			if st.button( "Go", use_container_width=True ):
				st.session_state.pe_selected_id = int( jump_id )
				st.rerun( )

		with a3:
			if st.button( "Undo", use_container_width=True ):
				reset_controls( )
				st.rerun( )

	# ----------------------------
	# Load data
	# ----------------------------
	where = ""
	params: List[ Any ] = [ ]

	if st.session_state.pe_search:
		where = "WHERE Name LIKE ? OR Text LIKE ?"
		s = f"%{st.session_state.pe_search}%"
		params.extend( [ s, s ] )

	offset = (st.session_state.pe_page - 1) * PAGE_SIZE

	query = f"""
		SELECT PromptsId, Name, Text, Version, ID
		FROM {TABLE}
		{where}
		ORDER BY {st.session_state.pe_sort_col} {st.session_state.pe_sort_dir}
		LIMIT {PAGE_SIZE} OFFSET {offset}
	"""

	count_query = f"SELECT COUNT(*) FROM {TABLE} {where}"

	try:
		with get_conn( ) as conn:
			cur = conn.cursor( )
			cur.execute( query, params )
			rows = cur.fetchall( )

			cur.execute( count_query, params )
			total_rows = cur.fetchone( )[ 0 ]
	except Exception as exc:
		st.error( f"SQLite error: {exc}" )
		rows = [ ]
		total_rows = 0

	total_pages = max( 1, math.ceil( total_rows / PAGE_SIZE ) )

	# ----------------------------
	# Table display
	# ----------------------------
	table_data: List[ Dict[ str, Any ] ] = [ ]
	for r in rows:
		table_data.append(
			{
					"Select": r[ 0 ] == st.session_state.pe_selected_id,
					"PromptsId": r[ 0 ],
					"Name": r[ 1 ],
					"Text": r[ 2 ],
					"Version": r[ 3 ],
					"ID": r[ 4 ],
			}
		)

	edited = st.data_editor(
		table_data,
		use_container_width=True,
		hide_index=True,
		column_config={
				"Select": st.column_config.CheckboxColumn( ) },
	)

	for row in edited:
		if row.get( "Select" ):
			st.session_state.pe_selected_id = row.get( "PromptsId" )

	# ----------------------------
	# Paging
	# ----------------------------
	p1, p2, p3 = st.columns( [ 1, 2, 1 ] )

	with p1:
		if st.button( "◀ Prev" ) and st.session_state.pe_page > 1:
			st.session_state.pe_page -= 1
			st.rerun( )

	with p2:
		st.markdown( f"Page **{st.session_state.pe_page}** of **{total_pages}**" )

	with p3:
		if st.button( "Next ▶" ) and st.session_state.pe_page < total_pages:
			st.session_state.pe_page += 1
			st.rerun( )

	st.divider( )

	# ----------------------------
	# Create / Edit form
	# ----------------------------
	selected = None
	if st.session_state.pe_selected_id:
		try:
			with get_conn( ) as conn:
				cur = conn.cursor( )
				cur.execute(
					f"SELECT PromptsId, Name, Text, Version FROM {TABLE} WHERE PromptsId=?",
					(st.session_state.pe_selected_id,),
				)
				selected = cur.fetchone( )
		except Exception as exc:
			st.error( f"SQLite error: {exc}" )
			selected = None

	with st.expander( "Create / Edit Prompt", expanded=True ):
		pid = st.text_input( "PromptsId", value=selected[ 0 ] if selected else "", disabled=True )
		name = st.text_input( "Name", value=selected[ 1 ] if selected else "" )
		text = st.text_area( "Text", value=selected[ 2 ] if selected else "", height=200 )
		version = st.number_input( "Version", min_value=1, value=int( selected[ 3 ] ) if selected else 1 )

		b1, b2, b3 = st.columns( [ 1, 1, 1 ] )

		with b1:
			if st.button( "Save Changes" if selected else "Create Prompt" ):
				try:
					with get_conn( ) as conn:
						cur = conn.cursor( )
						if selected:
							cur.execute(
								f"""
								UPDATE {TABLE}
								SET Name=?, Text=?, Version=?
								WHERE PromptsId=?
								""",
								(name, text, int( version ), selected[ 0 ]),
							)
						else:
							cur.execute(
								f"""
								INSERT INTO {TABLE} (Name, Text, Version)
								VALUES (?, ?, ?)
								""",
								(name, text, int( version )),
							)
						conn.commit( )
					st.success( "Saved." )
					st.rerun( )
				except Exception as exc:
					st.error( f"SQLite error: {exc}" )

		with b2:
			if selected and st.button( "Delete" ):
				try:
					with get_conn( ) as conn:
						conn.execute(
							f"DELETE FROM {TABLE} WHERE PromptsId=?",
							(selected[ 0 ],),
						)
						conn.commit( )
					reset_controls( )
					st.success( "Deleted." )
					st.rerun( )
				except Exception as exc:
					st.error( f"SQLite error: {exc}" )

		with b3:
			if st.button( "Clear Selection" ):
				reset_controls( )
				st.rerun( )

# ======================================================================================
# DOCUMENTS MODE
# ======================================================================================
if mode == "Documents":
	uploaded = st.file_uploader(
		"Upload documents (session only)",
		type=[ "pdf", "txt", "md", "docx" ],
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

		c1, c2 = st.columns( [ 1, 1 ] )
		with c1:
			if st.button( "Remove selected document" ):
				removed = st.session_state.files.pop( idx )
				st.success( f"Removed {removed}" )
		with c2:
			if st.button( "Show selected path" ):
				st.info( f"Local temp path: {selected_path}" )

		st.markdown( "---" )
		question = st.text_area( "Ask a question about the selected document" )
		if st.button( "Ask Document" ):
			if not question:
				st.warning( "Enter a question before asking." )
			else:
				with st.spinner( "Running document Q&A…" ):
					try:
						chat = Chat( use_ai=True, version=st.session_state.get( "gemini_version", "v1alpha" ) )
						answer = chat.summarize_document( pdf_path=selected_path, prompt=question )

						st.markdown( "**Answer:**" )
						st.markdown( answer or "No answer returned." )
						st.session_state.messages.append( {
								"role": "user",
								"content": f"[Document question] {question}" } )
						st.session_state.messages.append( {
								"role": "assistant",
								"content": answer or "" } )

						try:
							_update_token_counters( getattr( chat, "usage", None ) or getattr( chat, "response", None ) or answer )
						except Exception:
							pass
					except Exception as e:
						st.error( f"Document Q&A failed: {e}" )
	else:
		st.info( "No client-side documents uploaded this session. Use the uploader to add files." )

# ======================================================================================
# Footer
# ======================================================================================
st.divider( )
tu = st.session_state.token_usage
if tu[ "total_tokens" ] > 0:
	footer_html = f"""
    <div style="display:flex;justify-content:space-between;color:#9aa0a6;font-size:0.85rem;">
        <span>Boo</span>
        <span>Session tokens — total: {tu[ 'total_tokens' ]}</span>
    </div>
    """
else:
	footer_html = """
    <div style="display:flex;justify-content:space-between;color:#9aa0a6;font-size:0.85rem;">
        <span>Boo</span>
        <span>Gemini</span>
    </div>
    """

st.markdown( footer_html, unsafe_allow_html=True )
