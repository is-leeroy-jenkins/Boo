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
from typing import List, Dict, Any, Optional

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
	
	# Fallback: try typical nested places
	if not raw and isinstance( resp, dict ) and resp.get( "choices" ):
		try:
			raw = resp[ "choices" ][ 0 ].get( "usage" )
		except Exception:
			raw = None
	
	if not raw:
		return usage
	
	try:
		if isinstance( raw, dict ):
			usage[ "prompt_tokens" ] = int( raw.get( "prompt_tokens", 0 ) )
			usage[
				"completion_tokens" ] = int( raw.get( "completion_tokens", raw.get( "output_tokens", 0 ) ) )
			usage[ "total_tokens" ] = int( raw.get( "total_tokens",
				usage[ "prompt_tokens" ] + usage[ "completion_tokens" ] ) )
		else:
			usage[ "prompt_tokens" ] = int( getattr( raw, "prompt_tokens", 0 ) )
			usage[
				"completion_tokens" ] = int( getattr( raw, "completion_tokens", getattr( raw, "output_tokens", 0 ) ) )
			usage[ "total_tokens" ] = int( getattr( raw, "total_tokens",
				usage[ "prompt_tokens" ] + usage[ "completion_tokens" ] ) )
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

# ======================================================================================
# Sidebar — Provider selector above Mode, then Mode selector.
# ======================================================================================
with st.sidebar:
	# Provider selector (visible, non-invasive UI only)
	st.subheader( "Provider" )
	
	# thin blue divider immediately under the "Provider" text (before selectbox)
	st.markdown(
		"""
		<div style="height:2px;border-radius:3px;background:#0078FC;margin:6px 0 10px 0;"></div>
		""",
		unsafe_allow_html=True,
	)
	
	provider = st.selectbox(
		"Choose provider",
		[ "GPT",
		  "Gemini",
		  "Groq" ],
		index=[ "GPT",
		        "Gemini",
		        "Groq" ].index( st.session_state.get( "provider", "GPT" ) ),
	)
	st.session_state[ "provider" ] = provider
	
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
		  "Vector Store" ],
	)
	
	# Horizontal session controls (short buttons)
	c1, c2 = st.columns( [ 1,
	                       1 ] )
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
	
	# Blue divider between session controls and mode-settings
	st.markdown(
		"""
		<div style="height:2px;border-radius:4px;background:#0078FC;margin:12px 0;"></div>
		""",
		unsafe_allow_html=True,
	)

# ======================================================================================
# Dynamic Header — show provider and mode, and model relevant to the active mode
# ======================================================================================
# map mode -> session_state key for model (updated: Text -> text_model)
_mode_to_model_key = {
		"Text": "text_model",
		"Image": "image_model",
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

st.markdown(
	f"""
    <div style="margin-bottom:0.25rem;">
      <h3 style="margin:0;">{header_label} — {mode}</h3>
      <div style="color:#9aa0a6; margin-top:6px; font-size:0.95rem;">
        Model: {_display_value( model_val )} &nbsp;&nbsp;|&nbsp;&nbsp; Temp: {_display_value( temperature_val )} &nbsp;&nbsp;•&nbsp;&nbsp; Top-P: {_display_value( top_p_val )}
      </div>
    </div>
    """,
	unsafe_allow_html=True,
)
st.divider( )

# ======================================================================================
# TEXT MODE (formerly "Chat")
# ======================================================================================
if mode == "Text":
	st.header( "" )
	chat = Chat( )
	
	with st.sidebar:
		st.header( "Text Settings" )  # renamed from "Chat Settings"
		
		# Text model -> store into text_model
		text_model = st.selectbox( "Model", chat.model_options )
		st.session_state[ "text_model" ] = text_model
		
		# Parameters expander (restores the Parameters section)
		with st.expander( "Parameters:", expanded=True ):
			# Temperature and Top-P
			temperature = st.slider(
				"Temperature",
				min_value=0.0,
				max_value=1.0,
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
			
			# Max tokens
			max_tokens = st.number_input(
				"Max Tokens",
				min_value=1,
				max_value=100000,
				value=int( st.session_state.get( "max_tokens", 512 ) ),
			)
			st.session_state[ "max_tokens" ] = int( max_tokens )
			
			# Frequency and presence penalties
			freq_penalty = st.slider(
				"Frequency Penalty",
				min_value=-2.0,
				max_value=2.0,
				value=float( st.session_state.get( "freq_penalty", 0.0 ) ),
				step=0.01,
			)
			st.session_state[ "freq_penalty" ] = float( freq_penalty )
			
			pres_penalty = st.slider(
				"Presence Penalty",
				min_value=-2.0,
				max_value=2.0,
				value=float( st.session_state.get( "pres_penalty", 0.0 ) ),
				step=0.01,
			)
			st.session_state[ "pres_penalty" ] = float( pres_penalty )
			
			# Stop sequences (one per line)
			stop_text = st.text_area(
				"Stop Sequences (one per line)",
				value="\n".join( st.session_state.get( "stop_sequences", [ ] ) ),
				height=80,
			)
			# normalize to list, stripping empty lines
			st.session_state[ "stop_sequences" ] = [ s for s in (stop_text.splitlines( )) if
			                                         s.strip( ) ]
		
		include = st.multiselect( "Include:", chat.include_options )
		chat.include = include
	
	left, center, right = st.columns( [ 1,  2,  1 ] )
	
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
					# Build kwargs for generation from session_state
					gen_kwargs: Dict[ str, Any ] = { }
					# always include the model param
					gen_kwargs[ "model" ] = text_model
					# sampling params
					if "temperature" in st.session_state:
						gen_kwargs[ "temperature" ] = st.session_state[ "temperature" ]
					if "top_p" in st.session_state:
						gen_kwargs[ "top_p" ] = st.session_state[ "top_p" ]
					# token limit
					if "max_tokens" in st.session_state:
						gen_kwargs[ "max_tokens" ] = st.session_state[ "max_tokens" ]
					# penalties
					if "freq_penalty" in st.session_state:
						gen_kwargs[ "frequency_penalty" ] = st.session_state[ "freq_penalty" ]
					if "pres_penalty" in st.session_state:
						gen_kwargs[ "presence_penalty" ] = st.session_state[ "pres_penalty" ]
					# stop sequences
					if "stop_sequences" in st.session_state and st.session_state[
						"stop_sequences" ]:
						gen_kwargs[ "stop" ] = st.session_state[ "stop_sequences" ]
					
					response = None
					try:
						# Many Chat wrappers accept named kwargs; call defensively
						try:
							response = chat.generate_text( prompt=prompt, **gen_kwargs )
						except TypeError:
							# maybe the wrapper expects explicit params; try a smaller set then fallbacks
							try:
								response = chat.generate_text(
									prompt=prompt,
									model=text_model,
									temperature=st.session_state.get( "temperature", None ),
									top_p=st.session_state.get( "top_p", None ),
									max_tokens=st.session_state.get( "max_tokens", None ),
									frequency_penalty=st.session_state.get( "freq_penalty", None ),
									presence_penalty=st.session_state.get( "pres_penalty", None ),
									stop=st.session_state.get( "stop_sequences", None ),
								)
							except TypeError:
								# final fallback: only model + prompt
								response = chat.generate_text( prompt=prompt, model=text_model )
						except Exception as exc:
							st.error( f"Generation Failed: {exc}" )
							response = None
					except Exception as exc:
						st.error( f"Generation Failed: {exc}" )
						response = None
					
					st.markdown( response or "" )
					st.session_state.messages.append( {
							"role": "assistant",
							"content": response or "" } )
					
					# Update token counters defensively
					try:
						_update_token_counters( getattr( chat, "response", None ) or response )
					except Exception:
						pass
	
	# Token transparency in main area
	lcu = st.session_state.last_call_usage
	tu = st.session_state.token_usage
	if any( lcu.values( ) ):
		st.info(
			f"Last call — prompt: {lcu[ 'prompt_tokens' ]}, completion: {lcu[ 'completion_tokens' ]}, total: {lcu[ 'total_tokens' ]}"
		)
	if tu[ "total_tokens" ] > 0:
		st.write(
			f"Session totals — prompt: {tu[ 'prompt_tokens' ]} · completion: {tu[ 'completion_tokens' ]} · total: {tu[ 'total_tokens' ]}"
		)

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == "Image":
	image = Image( )
	with st.sidebar:
		st.header( "Image Settings" )
		image_model = st.selectbox( "Model", image.model_options )
		st.session_state[ "image_model" ] = image_model
		size = st.selectbox( "Size", image.size_options )
		quality = st.selectbox( "Quality", image.quality_options )
		fmt = st.selectbox( "Format", image.format_options )
	
	tab_gen, tab_analyze = st.tabs( [ "Generate", "Analyze" ] )
	with tab_gen:
		prompt = st.text_area( "Prompt" )
		if st.button( "Generate Image" ):
			with st.spinner( "Generating…" ):
				try:
					img_url = image.generate( prompt=prompt, model=image_model, size=size, quality=quality, fmt=fmt )
					st.image( img_url )
					_update_token_counters( getattr( image, "response", None ) )
				except Exception as exc:
					st.error( f"Image generation failed: {exc}" )
	with tab_analyze:
		st.markdown( "Image analysis — upload an image to analyze." )
		uploaded_img = st.file_uploader(
			"Upload an image for analysis",
			type=[ "png",
			       "jpg",
			       "jpeg",
			       "webp" ],
			accept_multiple_files=False,
			key="images_analyze_uploader",
		)
		
		if uploaded_img:
			tmp_path = save_temp( uploaded_img )
			# show preview
			st.image( uploaded_img, caption="Uploaded image preview", use_column_width=True )
			
			# Choose available analysis methods (for UI transparency)
			available_methods = [ ]
			for candidate in (
						"analyze",
						"describe_image",
						"describe",
						"classify",
						"detect_objects",
						"caption",
						"image_analysis",
			):
				if hasattr( image, candidate ):
					available_methods.append( candidate )
			
			# Present the prioritized default selection if methods exist
			if available_methods:
				chosen_method = st.selectbox( "Method", available_methods, index=0 )
			else:
				chosen_method = None
				st.info( "No dedicated image analysis method found on Image object; attempting generic handlers." )
			
			# Allow optional model override for analysis if supported
			chosen_model = st.selectbox( "Model (analysis)", [ image_model,
			                                                   None ], index=0 )
			if chosen_model is None:
				chosen_model_arg = image_model
			else:
				chosen_model_arg = chosen_model
			
			if st.button( "Analyze Image" ):
				with st.spinner( "Analyzing image…" ):
					analysis_result = None
					try:
						# Call the chosen method if a specific method is available
						if chosen_method:
							func = getattr( image, chosen_method, None )
							if func:
								# attempt several common signatures:
								# func(path), func(image_path, model=...), func(file_obj)
								try:
									analysis_result = func( tmp_path )
								except TypeError:
									try:
										analysis_result = func( tmp_path, model=chosen_model_arg )
									except TypeError:
										try:
											# try file-like input
											analysis_result = func( uploaded_img )
										except Exception as inner_exc:
											raise inner_exc
						else:
							# Fallback: try generic 'analyze' then 'describe_image' then return
							for fallback in ("analyze", "describe_image", "describe", "caption"):
								if hasattr( image, fallback ):
									func = getattr( image, fallback )
									try:
										analysis_result = func( tmp_path )
										break
									except Exception:
										try:
											analysis_result = func( tmp_path, model=chosen_model_arg )
											break
										except Exception:
											continue
						
						# If nothing returned yet, attempt an attribute 'analyze_image' or 'image_inspect'
						if analysis_result is None:
							for opt in ("analyze_image", "image_inspect"):
								if hasattr( image, opt ):
									func = getattr( image, opt )
									try:
										analysis_result = func( tmp_path )
										break
									except Exception:
										continue
						
						# If still None, fallback message
						if analysis_result is None:
							st.warning( "No analysis output returned by the available methods." )
						else:
							# Display structured results if possible
							if isinstance( analysis_result, (dict, list) ):
								st.json( analysis_result )
							else:
								st.markdown( "**Analysis result:**" )
								st.write( analysis_result )
							
							# Update token counters from image.response or returned result if structured
							try:
								_update_token_counters( getattr( image, "response", None ) or analysis_result )
							except Exception:
								pass
					
					except Exception as exc:
						st.error( f"Analysis Failed: {exc}" )

# ======================================================================================
# AUDIO MODE
# ======================================================================================
elif mode == "Audio":
	transcriber = Transcription( )
	translator = Translation( )
	
	with st.sidebar:
		st.header( "Audio Settings" )
		audio_model = st.selectbox( "Model", transcriber.model_options )
		st.session_state[ "audio_model" ] = audio_model
		language = st.selectbox( "Language", transcriber.language_options )
		task = st.selectbox( "Task", [ "Transcribe",
		                               "Translate" ] )
	
	left, center, right = st.columns( [ 1,
	                                    2,
	                                    1 ] )
	with center:
		uploaded = st.file_uploader( "Upload audio file", type=[ "wav",
		                                                         "mp3",
		                                                         "m4a",
		                                                         "flac" ] )
		if uploaded:
			tmp = save_temp( uploaded )
			if task == "Transcribe":
				with st.spinner( "Transcribing…" ):
					text = transcriber.transcribe( tmp, model=audio_model )
					st.text_area( "Transcript", value=text, height=300 )
			else:
				with st.spinner( "Translating…" ):
					text = translator.translate( tmp )
					st.text_area( "Translation", value=text, height=300 )

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == "Embeddings":
	embed = Embedding( )
	with st.sidebar:
		st.header( "Embedding Settings" )
		embed_model = st.selectbox( "Model", embed.model_options )
		st.session_state[ "embed_model" ] = embed_model
		method = st.selectbox( "Method", getattr( embed, "methods", [ "encode" ] ) )
	
	left, center, right = st.columns( [ 1,
	                                    2,
	                                    1 ] )
	with center:
		text = st.text_area( "Text to embed" )
		if st.button( "Embed" ):
			with st.spinner( "Embedding…" ):
				v = embed.create( text, model=embed_model )
				st.write( "Vector length:", len( v ) )

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
						st.success( f"Create call submitted for '{new_store_name}'." )
					else:
						st.warning( "create_store method not found on chat object." )
				except Exception as exc:
					st.error( f"Create store failed: {exc}" )
	
	st.markdown( "**Manage Stores**" )
	options: List[ tuple ] = [ ]
	if vs_map and isinstance( vs_map, dict ):
		options = list( vs_map.items( ) )
	
	if not options:
		try:
			client = getattr( chat, "client", None )
			if client and hasattr( client, "vector_stores" ) and hasattr( client.vector_stores, "list" ):
				api_list = client.vector_stores.list( )
				temp: List[ tuple ] = [ ]
				for item in getattr( api_list, "data", [ ] ) or api_list:
					nm = getattr( item, "name", None ) or (
						item.get( "name" ) if isinstance( item, dict ) else None)
					vid = getattr( item, "id", None ) or (
						item.get( "id" ) if isinstance( item, dict ) else None)
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
						st.json( vs.__dict__ if hasattr( vs, "__dict__" ) else vs )
					else:
						st.warning( "retrieve_store not available on chat object or no store selected." )
				except Exception as exc:
					st.error( f"Retrieve failed: {exc}" )
		
		with c2:
			if st.button( "Delete store" ):
				try:
					if sel_id and hasattr( chat, "delete_store" ):
						res = chat.delete_store( sel_id )
						st.success( f"Delete returned: {res}" )
					else:
						st.warning( "delete_store not available on chat object or no store selected." )
				except Exception as exc:
					st.error( f"Delete failed: {exc}" )
	else:
		st.info( "No vector stores discovered. Create one or confirm `chat.vector_stores` mapping exists." )

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
		question = st.text_area( "Ask a question about the selected document" )
		if st.button( "Ask Document" ):
			if not question:
				st.warning( "Enter a question before asking." )
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
								answer = chat.summarize_document( prompt=question, pdf_path=selected_path )
							except TypeError:
								answer = chat.summarize_document( question, selected_path )
						elif hasattr( chat, "ask_document" ):
							answer = chat.ask_document( selected_path, question )
						elif hasattr( chat, "document_qa" ):
							answer = chat.document_qa( selected_path, question )
						else:
							raise RuntimeError( "No document-QA method found on chat object." )
						
						st.markdown( "**Answer:**" )
						st.markdown( answer or "No answer returned." )
						st.session_state.messages.append( {
								"role": "user",
								"content": f"[Document question] {question}" } )
						st.session_state.messages.append( {
								"role": "assistant",
								"content": answer or "" } )
						
						try:
							_update_token_counters( getattr( chat, "response", None ) or answer )
						except Exception:
							pass
					except Exception as e:
						st.error( f"Document Q&A failed: {e}" )
	else:
		st.info( "No client-side documents uploaded this session. Use the uploader in the sidebar to add files." )

# ======================================================================================
# FILES API MODE
# ======================================================================================
if mode == "Files":
	try:
		chat  # type: ignore
	except NameError:
		chat = Chat( )
	
	list_method = None
	for name in ("retrieve_files", "retreive_files", "list_files", "get_files"):
		if hasattr( chat, name ):
			list_method = getattr( chat, name )
			break
	
	uploaded_file = st.file_uploader(
		"Upload file (server-side via Files API)",
		type=[ "pdf",
		       "txt",
		       "md",
		       "docx",
		       "png",
		       "jpg",
		       "jpeg" ],
	)
	if uploaded_file:
		tmp_path = save_temp( uploaded_file )
		upload_fn = None
		for name in ("upload_file", "upload", "files_upload"):
			if hasattr( chat, name ):
				upload_fn = getattr( chat, name )
				break
		if not upload_fn:
			st.warning( "No upload function found on chat object (upload_file)." )
		else:
			with st.spinner( "Uploading to Files API..." ):
				try:
					fid = upload_fn( tmp_path )
					st.success( f"Uploaded; file id: {fid}" )
				except Exception as exc:
					st.error( f"Upload failed: {exc}" )
	
	if st.button( "List files" ):
		if not list_method:
			st.warning( "No file-listing method found on chat object (expected retrieve_files/list_files)." )
		else:
			with st.spinner( "Listing files..." ):
				try:
					files_resp = list_method( )
					files_list = [ ]
					if files_resp is None:
						files_list = [ ]
					elif isinstance( files_resp, dict ):
						files_list = files_resp.get( "data" ) or files_resp.get( "files" ) or [ ]
					elif isinstance( files_resp, list ):
						files_list = files_resp
					else:
						try:
							files_list = getattr( files_resp, "data", files_resp )
						except Exception:
							files_list = [ files_resp ]
					
					rows = [ ]
					for f in files_list:
						try:
							fid = f.get( "id" ) if isinstance( f, dict ) else getattr( f, "id", None )
							name = f.get( "filename" ) if isinstance( f, dict ) else getattr( f, "filename", None )
							purpose = f.get( "purpose" ) if isinstance( f, dict ) else getattr( f, "purpose", None )
						except Exception:
							fid = None
							name = str( f )
							purpose = None
						rows.append( {
								"id": fid,
								"filename": name,
								"purpose": purpose } )
					if rows:
						st.table( rows )
					else:
						st.info( "No files returned." )
				except Exception as exc:
					st.error( f"List files failed: {exc}" )
	
	if 'files_list' in locals( ) and files_list:
		file_ids = [ r.get( "id" ) if isinstance( r, dict ) else getattr( r, "id", None ) for r in
		             files_list ]
		sel = st.selectbox( "Select file id to delete", options=file_ids )
		if st.button( "Delete selected file" ):
			del_fn = None
			for name in ("delete_file", "delete", "files_delete"):
				if hasattr( chat, name ):
					del_fn = getattr( chat, name )
					break
			if not del_fn:
				st.warning( "No delete function found on chat object (expected delete_file)." )
			else:
				with st.spinner( "Deleting file..." ):
					try:
						res = del_fn( sel )
						st.success( f"Delete result: {res}" )
					except Exception as exc:
						st.error( f"Delete failed: {exc}" )

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
        <span>Groq • Gemini • GPT</span>
    </div>
    """

st.markdown( footer_html, unsafe_allow_html=True )