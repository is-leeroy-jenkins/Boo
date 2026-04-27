'''
	******************************************************************************************
	    Assembly:                Boo
	    Filename:                app.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2024
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="app.py" company="Terry D. Eppler">
	
	           Boo is a data analysis tool integrating various Generative GPT, Text-Processing, and
	           Machine-Learning algorithms for federal analysts.
	           Copyright ©  2022  Terry Eppler
	
	   Permission is hereby granted, free of charge, to any person obtaining a copy
	   of this software and associated documentation files (the “Software”),
	   to deal in the Software without restriction,
	   including without limitation the rights to use,
	   copy, modify, merge, publish, distribute, sublicense,
	   and/or sell copies of the Software,
	   and to permit persons to whom the Software is furnished to do so,
	   subject to the following conditions:
	
	   The above copyright notice and this permission notice shall be included in all
	   copies or substantial portions of the Software.
	
	   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	   FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
	   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
	   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	   DEALINGS IN THE SOFTWARE.
	
	   You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov
	
	</copyright>
	<summary>
	  app.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations

import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import tiktoken
import config as cfg
import streamlit as st
import tempfile
import re
from typing import List, Dict, Any, Optional, Tuple
from boogr import Error
from sentence_transformers import SentenceTransformer

import fitz  # pymupdf

import sqlite3
import os
import gpt
import gemini
import grok

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def init_state( key: str, value: Any ) -> None:
	"""
		
		Purpose:
		--------
		Initialize a Streamlit session-state key only when the key is not already present.
	
		Parameters:
		-----------
		key (str): Session-state key to initialize.
		value (Any): Default value assigned only when the key is absent.
	
		Returns:
		--------
		None
		
	"""
	if key not in st.session_state:
		st.session_state[ key ] = value

def init_env_state( key: str, config_name: str, env_name: str ) -> None:
	"""
		
		Purpose:
		--------
		Initialize a session-state API/configuration key from config.py and mirror the value to
		os.environ when a configured value exists.
	
		Parameters:
		-----------
		key (str): Session-state key to initialize.
		config_name (str): Attribute name to read from config.py.
		env_name (str): Environment variable name to assign.
	
		Returns:
		--------
		None
		
	"""
	init_state( key, '' )
	if st.session_state.get( key, '' ) == '':
		default = getattr( cfg, config_name, '' )
		if default:
			st.session_state[ key ] = default
			os.environ[ env_name ] = default

def copy_state_alias( source_key: str, target_key: str, default: Any ) -> None:
	"""
		
		Purpose:
		--------
		Create a non-destructive alias between legacy and corrected session-state keys.
	
		Parameters:
		-----------
		source_key (str): Existing source key that may contain a value.
		target_key (str): Target key that should receive the value when absent.
		default (Any): Default value used when neither key exists.
	
		Returns:
		--------
		None
		
	"""
	if target_key not in st.session_state:
		st.session_state[ target_key ] = st.session_state.get( source_key, default )
	
	if source_key not in st.session_state:
		st.session_state[ source_key ] = st.session_state.get( target_key, default )

# ---------- API / PROVIDER CONFIGURATION -------------------------------------

init_env_state( 'openai_api_key', 'OPENAI_API_KEY', 'OPENAI_API_KEY' )
init_env_state( 'gemini_api_key', 'GEMINI_API_KEY', 'GEMINI_API_KEY' )
init_env_state( 'groq_api_key', 'GROQ_API_KEY', 'GROQ_API_KEY' )
init_env_state( 'google_api_key', 'GOOGLE_API_KEY', 'GOOGLE_API_KEY' )
init_env_state( 'google_cse_id', 'GOOGLE_CSE_ID', 'GOOGLE_CSE_ID' )
init_env_state( 'googlemaps_api_key', 'GOOGLEMAPS_API_KEY', 'GOOGLEMAPS_API_KEY' )
init_env_state( 'geocoding_api_key', 'GEOCODING_API_KEY', 'GEOCODING_API_KEY' )
init_env_state( 'geoapify_api_key', 'GEOAPIFY_API_KEY', 'GEOAPIFY_API_KEY' )
init_env_state( 'google_cloud_project_id', 'GOOGLE_CLOUD_PROJECT_ID', 'GOOGLE_CLOUD_PROJECT_ID' )
init_env_state( 'google_cloud_location', 'GOOGLE_CLOUD_LOCATION', 'GOOGLE_CLOUD_LOCATION' )

init_state( 'api_keys', { 'GPT': None, 'Groq': None, 'Gemini': None } )
init_state( 'provider', 'GPT' )
init_state( 'mode', 'Text' )

if st.session_state[ 'provider' ] is None:
	st.session_state[ 'provider' ] = 'GPT'

if st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Text'

# ---------- SHARED APPLICATION STATE -----------------------------------------

init_state( 'messages', [ ] )
init_state( 'chat_history', [ ] )
init_state( 'files', [ ] )
init_state( 'last_sources', [ ] )
init_state( 'use_semantic', False )
init_state( 'is_grounded', False )
init_state( 'selected_prompt_id', '' )
init_state( 'pending_system_prompt_name', '' )

init_state(
	'last_call_usage',
	{
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0,
	}
)

init_state(
	'token_usage',
	{
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0,
	}
)

# ---------- SHARED MODEL PARAMETERS ------------------------------------------

init_state( 'chat_model', '' )
init_state( 'text_model', '' )
init_state( 'image_model', '' )
init_state( 'image_analysis_model', '' )
init_state( 'image_generation_model', '' )
init_state( 'image_editing_model', '' )
init_state( 'audio_model', '' )
init_state( 'embedding_model', '' )
init_state( 'docqna_model', '' )
init_state( 'files_model', '' )
init_state( 'stores_model', '' )
init_state( 'bucket_model', '' )
init_state( 'tts_model', '' )
init_state( 'transcription_model', '' )
init_state( 'translation_model', '' )

# ---------- SHARED INSTRUCTION STATE -----------------------------------------

init_state( 'instructions', '' )
init_state( 'chat_system_instructions', '' )
init_state( 'text_system_instructions', '' )
init_state( 'image_system_instructions', '' )
init_state( 'audio_system_instructions', '' )
init_state( 'docqna_system_instructions', '' )
init_state( 'docqna_systems_instructions', st.session_state[ 'docqna_system_instructions' ] )
init_state( 'files_system_instructions', '' )
init_state( 'stores_system_instructions', '' )
init_state( 'bucket_system_instructions', '' )

# ---------- SHARED GENERATION PARAMETERS -------------------------------------

init_state( 'max_tools', 0 )
init_state( 'max_tokens', 0 )
init_state( 'temperature', 0.0 )
init_state( 'top_p', 0.0 )
init_state( 'top_percent', 0.0 )
init_state( 'frequency_penalty', 0.0 )
init_state( 'presence_penalty', 0.0 )
init_state( 'presense_penalty', st.session_state[ 'presence_penalty' ] )
init_state( 'freq_penalty', 0.0 )
init_state( 'pres_penalty', 0.0 )
init_state( 'background', False )
init_state( 'parallel_tools', False )
init_state( 'store', False )
init_state( 'stream', False )
init_state( 'execution_mode', '' )
init_state( 'response_format', '' )
init_state( 'tool_choice', '' )
init_state( 'reasoning', '' )
init_state( 'stop_sequences', [ ] )
init_state( 'stops', [ ] )
init_state( 'include', [ ] )
init_state( 'input', [ ] )
init_state( 'tools', [ ] )

# ---------- TEXT MODE STATE ---------------------------------------------------

init_state( 'text_number', 0 )
init_state( 'text_max_calls', 0 )
init_state( 'text_max_tools', 0 )
init_state( 'text_max_searches', 0 )
init_state( 'text_max_urls', 0 )
init_state( 'text_top_k', 0 )
init_state( 'text_max_tokens', 0 )
init_state( 'text_temperature', 0.0 )
init_state( 'text_top_percent', 0.0 )
init_state( 'text_frequency_penalty', 0.0 )
init_state( 'text_presence_penalty', 0.0 )
init_state( 'text_presense_penalty', st.session_state[ 'text_presence_penalty' ] )
init_state( 'text_parallel_tools', False )
init_state( 'text_parallel_calls', st.session_state[ 'text_parallel_tools' ] )
init_state( 'text_background', False )
init_state( 'text_store', False )
init_state( 'text_stream', False )
init_state( 'text_google_grounding', False )
init_state( 'text_response_format', '' )
init_state( 'text_tool_choice', '' )
init_state( 'text_resolution', '' )
init_state( 'text_media_resolution', '' )
init_state( 'text_reasoning', '' )
init_state( 'text_input', '' )
init_state( 'text_content', '' )
init_state( 'text_previous_response_id', '' )
init_state( 'text_conversation_id', '' )
init_state( 'text_stops', [ ] )
init_state( 'text_modalities', [ ] )
init_state( 'text_include', [ ] )
init_state( 'text_domains', [ ] )
init_state( 'text_tools', [ ] )
init_state( 'text_context', [ ] )
init_state( 'text_messages', [ ] )
init_state( 'text_gemini_history', [ ] )
init_state( 'text_file_search_store_names', [ ] )
init_state( 'selected_filestore_id', '' )
init_state( 'selected_filestore_label', '' )

# ---------- IMAGE MODE STATE --------------------------------------------------

init_state( 'image_number', 0 )
init_state( 'image_max_calls', 0 )
init_state( 'image_max_tools', 0 )
init_state( 'image_max_searches', 0 )
init_state( 'image_top_k', 0 )
init_state( 'image_max_tokens', 0 )
init_state( 'image_temperature', 0.0 )
init_state( 'image_top_percent', 0.0 )
init_state( 'image_frequency_penalty', 0.0 )
init_state( 'image_presence_penalty', 0.0 )
init_state( 'image_presense_penalty', st.session_state[ 'image_presence_penalty' ] )
init_state( 'image_parallel_tools', False )
init_state( 'image_background', False )
init_state( 'image_store', False )
init_state( 'image_stream', False )
init_state( 'image_response_format', '' )
init_state( 'image_tool_choice', '' )
init_state( 'image_resolution', '' )
init_state( 'image_media_resolution', '' )
init_state( 'image_reasoning', '' )
init_state( 'image_input', '' )
init_state( 'image_content', '' )
init_state( 'image_size', '' )
init_state( 'image_quality', '' )
init_state( 'image_style', '' )
init_state( 'image_prompt', '' )
init_state( 'image_action', '' )
init_state( 'image_file', None )
init_state( 'image_uploaded_file', None )
init_state( 'image_mask_file', None )
init_state( 'image_stops', [ ] )
init_state( 'image_modalities', [ ] )
init_state( 'image_include', [ ] )
init_state( 'image_domains', [ ] )
init_state( 'image_tools', [ ] )
init_state( 'image_context', [ ] )
init_state( 'image_messages', [ ] )
init_state( 'generated_images', [ ] )
init_state( 'analyzed_images', [ ] )
init_state( 'edited_images', [ ] )

# ---------- AUDIO MODE STATE --------------------------------------------------

init_state( 'audio_number', 0 )
init_state( 'audio_max_calls', 0 )
init_state( 'audio_max_tools', 0 )
init_state( 'audio_max_searches', 0 )
init_state( 'audio_top_k', 0 )
init_state( 'audio_max_tokens', 0 )
init_state( 'audio_temperature', 0.0 )
init_state( 'audio_top_percent', 0.0 )
init_state( 'audio_frequency_penalty', 0.0 )
init_state( 'audio_presence_penalty', 0.0 )
init_state( 'audio_presense_penalty', st.session_state[ 'audio_presence_penalty' ] )
init_state( 'audio_parallel_tools', False )
init_state( 'audio_background', False )
init_state( 'audio_store', False )
init_state( 'audio_stream', False )
init_state( 'audio_loop', False )
init_state( 'audio_autoplay', False )
init_state( 'audio_response_format', '' )
init_state( 'audio_tool_choice', '' )
init_state( 'audio_resolution', '' )
init_state( 'audio_media_resolution', '' )
init_state( 'audio_reasoning', '' )
init_state( 'audio_input', '' )
init_state( 'audio_content', '' )
init_state( 'audio_task', '' )
init_state( 'audio_language', '' )
init_state( 'audio_format', '' )
init_state( 'audio_file', '' )
init_state( 'audio_voice', '' )
init_state( 'audio_rate', 1.0 )
init_state( 'audio_start_time', 0.0 )
init_state( 'audio_end_time', 0.0 )
init_state( 'audio_stops', [ ] )
init_state( 'audio_modalities', [ ] )
init_state( 'audio_include', [ ] )
init_state( 'audio_domains', [ ] )
init_state( 'audio_tools', [ ] )
init_state( 'audio_context', [ ] )
init_state( 'audio_messages', [ ] )
init_state( 'tts_input', '' )
init_state( 'tts_voice', '' )
init_state( 'tts_format', '' )
init_state( 'tts_output_path', '' )
init_state( 'transcription_file', None )
init_state( 'transcription_language', '' )
init_state( 'transcription_prompt', '' )
init_state( 'translation_file', None )
init_state( 'translation_prompt', '' )

# ---------- EMBEDDINGS MODE STATE --------------------------------------------

init_state( 'embedding_input', '' )
init_state( 'embedding_text', '' )
init_state( 'embedding_file', None )
init_state( 'embedding_dimensions', 0 )
init_state( 'embedding_encoding_format', '' )
init_state( 'embedding_chunk_size', 0 )
init_state( 'embedding_chunk_overlap', 0 )
init_state( 'embedding_chunks', [ ] )
init_state( 'embedding_vectors', [ ] )
init_state( 'embedding_results', None )
init_state( 'embedding_dataframe', None )
init_state( 'embedding_messages', [ ] )

# ---------- DOCUMENT Q&A MODE STATE ------------------------------------------

init_state( 'docqna_source', '' )
init_state( 'docqna_mode', '' )
init_state( 'docqna_file', None )
init_state( 'docqna_files', [ ] )
init_state( 'docqna_file_id', '' )
init_state( 'docqna_vector_store_id', '' )
init_state( 'docqna_question', '' )
init_state( 'docqna_context', '' )
init_state( 'docqna_answer', '' )
init_state( 'docqna_messages', [ ] )
init_state( 'docqna_history', [ ] )
init_state( 'docqna_chunks', [ ] )
init_state( 'docqna_sources', [ ] )
init_state( 'docqna_temperature', 0.0 )
init_state( 'docqna_top_percent', 0.0 )
init_state( 'docqna_max_tokens', 0 )
init_state( 'docqna_frequency_penalty', 0.0 )
init_state( 'docqna_presence_penalty', 0.0 )
init_state( 'docqna_response_format', '' )
init_state( 'docqna_tool_choice', '' )
init_state( 'docqna_reasoning', '' )

# ---------- FILES MODE STATE --------------------------------------------------

init_state( 'files_input', '' )
init_state( 'files_file', None )
init_state( 'files_uploaded', [ ] )
init_state( 'files_selected_id', '' )
init_state( 'files_selected_label', '' )
init_state( 'files_purpose', '' )
init_state( 'files_metadata', None )
init_state( 'files_results', None )
init_state( 'files_messages', [ ] )
init_state( 'files_temperature', 0.0 )
init_state( 'files_top_percent', 0.0 )
init_state( 'files_max_tokens', 0 )
init_state( 'files_frequency_penalty', 0.0 )
init_state( 'files_presence_penalty', 0.0 )
init_state( 'files_response_format', '' )
init_state( 'files_tool_choice', '' )
init_state( 'files_reasoning', '' )

# ---------- VECTOR STORES MODE STATE -----------------------------------------

init_state( 'stores_input', '' )
init_state( 'stores_query', '' )
init_state( 'stores_selected_id', '' )
init_state( 'stores_selected_label', '' )
init_state( 'stores_file_id', '' )
init_state( 'stores_file_ids', [ ] )
init_state( 'stores_results', None )
init_state( 'stores_messages', [ ] )
init_state( 'stores_temperature', 0.0 )
init_state( 'stores_top_percent', 0.0 )
init_state( 'stores_max_tokens', 0 )
init_state( 'stores_frequency_penalty', 0.0 )
init_state( 'stores_presence_penalty', 0.0 )
init_state( 'stores_response_format', '' )
init_state( 'stores_tool_choice', '' )
init_state( 'stores_reasoning', '' )
init_state( 'stores_background', False )
init_state( 'stores_store', False )
init_state( 'stores_stream', False )

# ---------- FILE SEARCH STORES MODE STATE ------------------------------------

init_state( 'filestore_model', '' )
init_state( 'filestore_input', '' )
init_state( 'filestore_query', '' )
init_state( 'filestore_selected_id', '' )
init_state( 'filestore_selected_label', '' )
init_state( 'filestore_file_id', '' )
init_state( 'filestore_file_ids', [ ] )
init_state( 'filestore_results', None )
init_state( 'filestore_messages', [ ] )
init_state( 'filestore_temperature', 0.0 )
init_state( 'filestore_top_percent', 0.0 )
init_state( 'filestore_max_tokens', 0 )
init_state( 'filestore_frequency_penalty', 0.0 )
init_state( 'filestore_presence_penalty', 0.0 )
init_state( 'filestore_response_format', '' )
init_state( 'filestore_tool_choice', '' )
init_state( 'filestore_reasoning', '' )
init_state( 'filestore_background', False )
init_state( 'filestore_store', False )
init_state( 'filestore_stream', False )

# ---------- GOOGLE CLOUD BUCKETS MODE STATE ----------------------------------

init_state( 'bucket_input', '' )
init_state( 'bucket_query', '' )
init_state( 'bucket_selected_id', '' )
init_state( 'bucket_selected_label', '' )
init_state( 'bucket_file_id', '' )
init_state( 'bucket_file_ids', [ ] )
init_state( 'bucket_results', None )
init_state( 'bucket_messages', [ ] )
init_state( 'bucket_number', 0 )
init_state( 'bucket_temperature', 0.0 )
init_state( 'bucket_top_percent', 0.0 )
init_state( 'bucket_max_tokens', 0 )
init_state( 'bucket_frequency_penalty', 0.0 )
init_state( 'bucket_presence_penalty', 0.0 )
init_state( 'bucket_response_format', '' )
init_state( 'bucket_tool_choice', '' )
init_state( 'bucket_reasoning', '' )
init_state( 'bucket_background', False )
init_state( 'bucket_store', False )
init_state( 'bucket_stream', False )

# ---------- PROMPT ENGINEERING STATE -----------------------------------------

init_state( 'prompt_id', getattr( cfg, 'PROMPT_ID', '' ) )
init_state( 'prompt_version', getattr( cfg, 'PROMPT_VERSION', '' ) )
init_state( 'prompt_name', '' )
init_state( 'prompt_text', '' )
init_state( 'prompt_rows', [ ] )
init_state( 'selected_prompt_name', '' )
init_state( 'selected_prompt_text', '' )

# ---------- DATA MANAGEMENT / EXPORT STATE -----------------------------------

init_state( 'df_original', None )
init_state( 'df_working', None )
init_state( 'df_processed', None )
init_state( 'df_results', None )
init_state( 'uploaded_data_file', None )
init_state( 'selected_table', '' )
init_state( 'selected_columns', [ ] )
init_state( 'target_column', '' )
init_state( 'export_format', '' )
init_state( 'export_path', '' )

# ---------- NON-DESTRUCTIVE LEGACY ALIASES -----------------------------------

copy_state_alias( 'text_presense_penalty', 'text_presence_penalty', 0.0 )
copy_state_alias( 'image_presense_penalty', 'image_presence_penalty', 0.0 )
copy_state_alias( 'audio_presense_penalty', 'audio_presence_penalty', 0.0 )
copy_state_alias( 'presense_penalty', 'presence_penalty', 0.0 )
copy_state_alias( 'docqna_systems_instructions', 'docqna_system_instructions', '' )
copy_state_alias( 'text_parallel_calls', 'text_parallel_tools', False )
copy_state_alias( 'text_max_tools', 'text_max_calls', 0 )

# ==============================================================================
# RESPONSE/CHAT UTILITIES
# ==============================================================================

def extract_response_text( response: object ) -> str:
	"""
		
		Purpose:
		--------
		Safely extract assistant text from a Responses API object.
	
		Parameters:
		-----------
		response (object): The response returned from the OpenAI client.
	
		Returns:
		--------
		str: Concatenated assistant text output. Empty string if none found.
		
	"""
	if response is None:
		return ""
	
	output = getattr( response, "output", None )
	if not output or not isinstance( output, list ):
		return ""
	
	text_chunks: list[ str ] = [ ]
	
	for item in output:
		if not hasattr( item, "type" ):
			continue
		
		if item.type == "message":
			content = getattr( item, "content", None )
			if not content or not isinstance( content, list ):
				continue
			
			for part in content:
				if getattr( part, "type", None ) == "output_text":
					text = getattr( part, "text", "" )
					if text:
						text_chunks.append( text )
	
	return "".join( text_chunks ).strip( )

def encode_image_base64( path: str ) -> str:
	"""
	
		Purpose:
		_________
		
		Parametes:
		----------
		
		
		Returns:
		--------
		
		
	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, max_tokens: int = 400 ) -> list[ str ]:
	"""
		
		Purpose
		-------
		Segment normalized text into chunks by:
			1. Sentence boundaries
			2. Fallback to token windowing if needed
	
		Parameters
		----------
		text: str
		max_tokens: int
	
		Returns
		-------
		list[str]
		
	"""
	if not text:
		return [ ]
	
	# Sentence-based segmentation
	sentences = re.split( r"(?<=[.!?])\s+", text )
	sentences = [ s.strip( ) for s in sentences if s.strip( ) ]
	
	if len( sentences ) > 1:
		return sentences
	
	# Fallback: token window segmentation
	words = text.split( )
	chunks = [ ]
	current_chunk = [ ]
	token_count = 0
	
	for word in words:
		current_chunk.append( word )
		token_count += 1
		
		if token_count >= max_tokens:
			chunks.append( " ".join( current_chunk ) )
			current_chunk = [ ]
			token_count = 0
	
	if current_chunk:
		chunks.append( " ".join( current_chunk ) )
	
	return chunks

def cosine_sim( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def sanitize_markdown( text: str ) -> str:
	"""
	
		Purpose:
		_________
		
		
	"""
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

def init_state( ) -> None:
	"""
	
		Purpose:
		_________
		Initializes all session state variables.
		
		
	"""
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = [ ]
	
	if 'chat_messages' not in st.session_state:
		st.session_state.chat_messages = [ ]
	
	if 'execution_mode' not in st.session_state:
		st.session_state.execution_mode = 'Standard'
	
	for k in ('audio_system_instructions',
	          'image_system_instructions',
	          'docqna_system_instructions',
	          'text_system_instructions'):
		st.session_state.setdefault( k, "" )

def reset_state( ) -> None:
	"""
	
		Purpose:
		_________
		Resets the session state to default values
		
	"""
	st.session_state.chat_history = [ ]
	st.session_state.last_answer = ""
	st.session_state.last_sources = [ ]
	st.session_state.last_analysis = {
			'tables': [ ],
			'files': [ ],
			'text': [ ],
	}

def normalize( obj ):
	if obj is None or isinstance( obj, (str, int, float, bool) ):
		return obj
	
	if isinstance( obj, dict ):
		return { k: normalize( v ) for k, v in obj.items( ) }
	
	if isinstance( obj, (list, tuple, set) ):
		return [ normalize( v ) for v in obj ]
	if hasattr( obj, "model_dump" ):
		try:
			return obj.model_dump( )
		except Exception:
			return str( obj )
	return str( obj )

def extract_answer( response: Any ) -> str:
	"""
	
		Purpose:
		_________
		Parses-out answer text from a structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response expected to contain an `output` attribute.
		
		Returns:
		---------
		str
			Concatenated assistant text or empty string.
	
	"""
	texts: List[ str ] = [ ]
	
	if response is None:
		return ''
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return ''
	
	for item in output:
		if item is None:
			continue
		
		item_type = getattr( item, 'type', None )
		
		# ---------------------------------------
		# Direct text items
		# ---------------------------------------
		if item_type in TEXT_TYPES:
			text = getattr( item, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				texts.append( text )
			continue
		
		# ---------------------------------------
		# Nested content blocks
		# ---------------------------------------
		content = getattr( item, 'content', None )
		if not isinstance( content, list ):
			continue
		
		for block in content:
			if block is None:
				continue
			
			block_type = getattr( block, 'type', None )
			if block_type in TEXT_TYPES:
				text = getattr( block, 'text', None )
				if isinstance( text, str ) and text.strip( ):
					texts.append( text )
	
	return '\n'.join( texts ).strip( )

def extract_sources( response: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		_________
		Parses-out sources from structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response.
		
		Returns:
		---------
		List[ Dict[ str, Any ] ]
			List of normalized source dictionaries.
	
	"""
	sources: List[ Dict[ str, Any ] ] = [ ]
	
	if response is None:
		return sources
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return sources
	
	for item in output:
		if item is None:
			continue
		
		t = getattr( item, 'type', None )
		
		# ------------------------------------------------
		# Web search
		# ------------------------------------------------
		if t == 'web_search_call':
			action = getattr( item, 'action', None )
			raw = getattr( action, 'sources', None ) if action else None
			
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for src in raw:
				s = normalize( src )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'title' ), 'snippet': s.get( 'snippet' ),
				                  'url': s.get( 'url' ), 'files_id': None, } )
		
		# ------------------------------------------------
		# File search (vector store)
		# ------------------------------------------------
		elif t == 'file_search_call':
			raw = getattr( item, 'results', None )
			
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for r in raw:
				s = normalize( r )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'file_name' ) or s.get( 'title' ),
				                  'snippet': s.get( 'text' ), 'url': None,
				                  'files_id': s.get( 'files_id' ), } )
	
	return sources

def extract_analysis( response: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		_________
		Parses-out code interpreter artifacts from structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response.
		
		Returns:
		---------
		Dict[ str, Any ]
			Dictionary containing tables, files, and text artifacts.
	
	"""
	artifacts: Dict[ str, Any ] = {
			'tables': [ ],
			'files': [ ],
			'text': [ ] }
	
	if response is None:
		return artifacts
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return artifacts
	
	for item in output:
		if item is None:
			continue
		
		if getattr( item, 'type', None ) != 'code_interpreter_call':
			continue
		
		outputs = getattr( item, 'outputs', None )
		if not isinstance( outputs, (list, tuple) ):
			continue
		
		for out in outputs:
			if out is None:
				continue
			
			out_type = getattr( out, 'type', None )
			
			if out_type == 'table':
				normalized = normalize( out )
				artifacts[ 'tables' ].append( normalized )
			
			elif out_type == 'file':
				normalized = normalize( out )
				artifacts[ 'files' ].append( normalized )
			
			elif out_type in TEXT_TYPES:
				text = getattr( out, 'text', None )
				if isinstance( text, str ) and text.strip( ):
					artifacts[ 'text' ].append( text )
	
	return artifacts

def save_temp( upload ) -> str | None:
	"""
		Purpose:
		--------
		Save a Streamlit UploadedFile object to a temporary file on disk
		and return the filesystem path.
	
		Parameters:
		-----------
		upload : streamlit.runtime.uploaded_file_manager.UploadedFile
			Uploaded file object from st.file_uploader.
	
		Returns:
		--------
		str | None
			Path to the temporary file, or None if invalid input.
	"""
	if upload is None:
		return None
	
	try:
		_, ext = os.path.splitext( upload.name )
		ext = ext or ""
		with tempfile.NamedTemporaryFile( delete=False, suffix=ext ) as tmp:
			tmp.write( upload.getbuffer( ) )
			tmp_path = tmp.name
		
		return tmp_path
	except Exception:
		return None

def _extract_usage_from_response( resp: Any ) -> Dict[ str, int ]:
	"""
	
		Purpose:
		_________
		Extract token usage from a response object/dict.
		Returns dict with prompt_tokens, completion_tokens, total_tokens.
		Defensive: returns zeros if not present.
		
	"""
	usage = { 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, }
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
				getattr( raw, "completion_tokens", getattr( raw, "output_tokens", 0 ) ) )
			usage[ "total_tokens" ] = int(
				getattr( raw, "total_tokens",
					usage[ "prompt_tokens" ] + usage[ "completion_tokens" ], ) )
	except Exception:
		usage[ "total_tokens" ] = (usage[ "prompt_tokens" ] + usage[ "completion_tokens" ])
	
	return usage

def _update_token_counters( resp: Any ) -> None:
	"""
	
		Purpose:
		_________
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

def build_intent_prefix( mode: str ) -> str:
	if mode == 'Guidance Only':
		return (
				'[ANALYST INTENT]\n'
				'Respond using authoritative policy and guidance only. '
				'Do not perform financial computation.\n\n'
		)
	if mode == 'Analysis Only':
		return (
				'[ANALYST INTENT]\n'
				'Respond using financial analysis and computation only. '
				'Minimize policy citation.\n\n'
		)
	return ''

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( "SELECT role, content FROM chat_history ORDER BY id" ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

def format_results( results ):
	formatted_results = ''
	for result in results.data:
		formatted_result = f"<li> '{result.name}'"
		formatted_results += formatted_result + "</li>"
	return f"<p>{formatted_results}</p>"

def count_tokens( text: str ) -> int:
	"""
		
		Purpose
		----------
		Returns the number of tokens in a text string.
		
		Parmeters
		-----------
		string : str
		encoding_name : str
		
		Return
		------------
		int
		
	"""
	encoding = tiktoken.get_encoding( 'cl100k_base' )
	num_tokens = len( encoding.encode( text ) )
	return num_tokens

# ==============================================================================
# TEXT UTILITIES
# ==============================================================================

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, size: int=1200, overlap: int=200 ) -> List[ str ]:
	chunks, i = [ ], 0
	while i < len( text ):
		chunks.append( text[ i:i + size ] )
		i += size - overlap
	return chunks

def convert_xml( text: str ) -> str:
	"""
		
			Purpose:
			_________
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
			Parameters:
			-----------
			text (str) - Prompt text containing XML-like opening and closing tags.
	
			Returns:
			---------
			Markdown-formatted text using level-2 headings (##).
	"""
	markdown_blocks: List[ str ] = [ ]
	for match in cfg.XML_BLOCK_PATTERN.finditer( text ):
		raw_tag: str = match.group( "tag" )
		body: str = match.group( "body" ).strip( )
		
		# Humanize tag name for Markdown heading
		heading: str = raw_tag.replace( "_", " " ).replace( "-", " " ).title( )
		markdown_blocks.append( f"## {heading}" )
		if body:
			markdown_blocks.append( body )
	return "\n\n".join( markdown_blocks )

def markdown_converter( text: Any ) -> str:
	"""
		Purpose:
		--------
		Convert between Markdown headings and simple XML-like heading tags.
	
		Behavior:
		---------
		Auto-detects direction:
		  - If <h1>...</h1> / <h2>...</h2> ... exist, converts to Markdown (# / ## / ###).
		  - Otherwise converts Markdown headings (# / ## / ###) to <hN>...</hN> tags.
	
		Parameters:
		-----------
		text : Any
			Source text. Non-string values return "".
	
		Returns:
		--------
		str
			Converted text.
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return ""
	
	# Normalize newlines
	src = text.replace( "\r\n", "\n" ).replace( "\r", "\n" )
	
	htag_pattern = re.compile( r"<h([1-6])>(.*?)</h\1>", flags=re.IGNORECASE | re.DOTALL )
	md_heading_pattern = re.compile( r"^(#{1,6})[ \t]+(.+?)[ \t]*$", flags=re.MULTILINE )
	
	# ------------------------------------------------------------------
	# Direction detection
	# ------------------------------------------------------------------
	contains_htags = bool( htag_pattern.search( src ) )
	
	# ------------------------------------------------------------------
	# XML-like heading tags -> Markdown headings
	# ------------------------------------------------------------------
	if contains_htags:
		def _htag_to_md( match: re.Match ) -> str:
			level = int( match.group( 1 ) )
			content = match.group( 2 ).strip( )
			
			# Preserve inner newlines safely by collapsing interior whitespace
			# while keeping content readable.
			content = re.sub( r"[ \t]+\n", "\n", content )
			content = re.sub( r"\n[ \t]+", "\n", content )
			
			return f"{'#' * level} {content}"
		
		out = htag_pattern.sub( _htag_to_md, src )
		return out.strip( )
	
	# ------------------------------------------------------------------
	# Markdown headings -> XML-like heading tags
	# ------------------------------------------------------------------
	def _md_to_htag( match: re.Match ) -> str:
		hashes = match.group( 1 )
		content = match.group( 2 ).strip( )
		level = len( hashes )
		return f"<h{level}>{content}</h{level}>"
	
	out = md_heading_pattern.sub( _md_to_htag, src )
	return out.strip( )

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True, )

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( 'SELECT role, content FROM chat_history ORDER BY id' ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

# ==============================================================================
# DOCQNA UTILITIES
# ==============================================================================

def extract_text_from_bytes( file_bytes: bytes ) -> str:
	"""
		Extracts text from PDF or text-based documents.
	"""
	try:
		import fitz  # PyMuPDF
		
		doc = fitz.open( stream=file_bytes, filetype="pdf" )
		text = ""
		for page in doc:
			text += page.get_text( )
		return text.strip( )
	
	except Exception:
		try:
			return file_bytes.decode( errors="ignore" )
		except Exception:
			return ""

def route_document_query( prompt: str ) -> str:
	"""
		Purpose:
		--------
		Route a document question through the unified chat pipeline and return a model-generated answer.

		Parameters:
		-----------
		prompt : str
			The user question to answer about active documents.

		Returns:
		--------
		str
			The assistant answer text.
	"""
	user_input = build_document_user_input( prompt )
	if not user_input:
		user_input = (prompt or '').strip( )
	
	return run_llm_turn(
		user_input=user_input,
		temperature=float( st.session_state.get( 'temperature', 0.0 ) ),
		top_p=float( st.session_state.get( 'top_percent', 0.95 ) ),
		repeat_penalty=float( st.session_state.get( 'repeat_penalty', 1.1 ) ),
		max_tokens=int( st.session_state.get( 'max_tokens', 1024 ) ) or 1024,
		stream=False,
		output=None
	)

def summarize_active_document( ) -> str:
	"""
		Uses the routing layer to summarize the currently active document.
	"""
	system_instructions = st.session_state.get( "system_instructions", "" )
	summary_prompt = """
		Provide a clear, structured summary of this document.
		Include:
		- Purpose
		- Key themes
		- Major conclusions
		- Important data points (if any)
		- Policy implications (if applicable)
		
		Be precise and concise.
		"""
	if system_instructions:
		summary_prompt = f"{system_instructions}\n\n{summary_prompt}"
	
	return route_document_query( summary_prompt.strip( ) )

def _docqna_compute_fingerprint( active_docs: List[ str ], doc_bytes: Dict[ str, bytes ] ) -> str:
	'''
		
		Purpose:
		--------
		Computes a stable fingerprint for the currently selected active documents and their byte contents.
	
		Parameters:
		-----------
		active_docs:
			A List[ str ] of active document names.
		doc_bytes:
			A Dict[ str, bytes ] mapping document name to file bytes.
	
		Returns:
		--------
		A str fingerprint suitable for cache invalidation.
	
	'''
	h = hashlib.sha256( )
	for name in sorted( active_docs ):
		b = doc_bytes.get( name, b'' )
		h.update( name.encode( 'utf-8', errors='ignore' ) )
		h.update( len( b ).to_bytes( 8, 'little', signed=False ) )
		h.update( hashlib.sha256( b ).digest( ) )
	return h.hexdigest( )

def _docqna_extract_text_from_pdf_bytes( file_bytes: bytes ) -> str:
	'''
	
		Purpose:
		--------
		Extracts text from a PDF byte stream using PyMuPDF.
	
		Parameters:
		-----------
		file_bytes:
			The PDF bytes.
	
		Returns:
		--------
		A str containing extracted text.
	
	'''
	if not file_bytes:
		return ''
	
	try:
		doc = fitz.open( stream=file_bytes, filetype='pdf' )
		parts: List[ str ] = [ ]
		for page in doc:
			parts.append( page.get_text( 'text' ) or '' )
		return '\n'.join( parts ).strip( )
	except Exception:
		return ''

def _docqna_safe_load_sqlite_vec( conn: sqlite3.Connection ) -> bool:
	'''
		
		Purpose:
		--------
		Attempts to load sqlite-vec into the provided SQLite connection.
	
		Parameters:
		-----------
		conn:
			The sqlite3.Connection.
	
		Returns:
		--------
		True if sqlite-vec loaded successfully; otherwise False.
		
	'''
	try:
		import sqlite_vec
		
		sqlite_vec.load( conn )
		return True
	except Exception:
		return False

def _docqna_ensure_vec_schema( dim: int ) -> bool:
	'''
	
		Purpose:
		--------
		Creates the sqlite-vec virtual table used for Document Q&A embeddings if possible.
	
		Parameters:
		-----------
		dim:
			The embedding dimension (e.g., 384 for all-MiniLM-L6-v2).
	
		Returns:
		--------
		True if the schema exists and is usable; otherwise False.
	
	'''
	conn = create_connection( )
	try:
		ok = _docqna_safe_load_sqlite_vec( conn )
		if not ok:
			return False
		
		cur = conn.cursor( )
		cur.execute(
			f'''
			CREATE VIRTUAL TABLE IF NOT EXISTS docqna_vec
			USING vec0(
				embedding float[{int( dim )}],
				doc_name TEXT,
				chunk TEXT
			);
			'''
		)
		conn.commit( )
		return True
	except Exception:
		return False
	finally:
		conn.close( )

def _docqna_rebuild_index_if_needed( embedder: SentenceTransformer ) -> None:
	'''
		
		Purpose:
		--------
		Builds or refreshes the Document Q&A vector index when active documents change.
	
		Parameters:
		-----------
		embedder:
			The SentenceTransformer used to generate embeddings.
	
		Returns:
		--------
		None
		
	'''
	active_docs: List[ str ] = st.session_state.get( 'active_docs', [ ] )
	doc_bytes: Dict[ str, bytes ] = st.session_state.get( 'doc_bytes', { } )
	
	fp = _docqna_compute_fingerprint( active_docs, doc_bytes )
	if fp and fp == st.session_state.get( 'docqna_fingerprint', '' ):
		return
	
	st.session_state[ 'docqna_fingerprint' ] = fp
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	
	dim_value = getattr( embedder, 'get_sentence_embedding_dimension', lambda: 384 )( )
	dim = int( dim_value ) if dim_value else 384
	
	vec_ready = _docqna_ensure_vec_schema( dim )
	st.session_state[ 'docqna_vec_ready' ] = bool( vec_ready )
	
	conn = create_connection( )
	try:
		cur = conn.cursor( )
		
		if vec_ready:
			try:
				cur.execute( 'DELETE FROM docqna_vec;' )
				conn.commit( )
			except Exception:
				st.session_state[ 'docqna_vec_ready' ] = False
				vec_ready = False
		
		total_chunks = 0
		fallback_rows: List[ Tuple[ str, str, bytes ] ] = [ ]
		
		for name in active_docs:
			b = doc_bytes.get( name )
			if not b:
				continue
			
			text = _docqna_extract_text_from_pdf_bytes( b )
			if not text:
				continue
			
			chunks = chunk_text( text )
			if not chunks:
				continue
			
			vecs = embedder.encode( chunks, show_progress_bar=False )
			vecs = np.asarray( vecs, dtype=np.float32 )
			
			if vec_ready:
				for chunk_text_value, v in zip( chunks, vecs ):
					cur.execute(
						'INSERT INTO docqna_vec ( embedding, doc_name, chunk ) VALUES ( ?, ?, ? );',
						(v.tobytes( ), name, chunk_text_value)
					)
			else:
				for chunk_text_value, v in zip( chunks, vecs ):
					fallback_rows.append( (name, chunk_text_value, v.tobytes( )) )
			
			total_chunks += int( len( chunks ) )
		
		conn.commit( )
		st.session_state[ 'docqna_chunk_count' ] = total_chunks
		
		if not vec_ready:
			st.session_state[ 'docqna_fallback_rows' ] = fallback_rows
	
	except Exception:
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_fallback_rows' ] = [ ]
		st.session_state[ 'docqna_chunk_count' ] = 0
	finally:
		conn.close( )

def retrieve_top_doc_chunks( query: str, k: int = 6 ) -> List[ Tuple[ str, str, float ] ]:
	'''
	
		Purpose:
		--------
		Retrieves top-k document chunks relevant to the query, using sqlite-vec when available, and falling
		back to in-memory cosine similarity when not.
	
		Parameters:
		-----------
		query:
			The user query string.
		k:
			The number of chunks to return.
	
		Returns:
		--------
		A List[ Tuple[ str, str, float ] ] of (doc_name, chunk, score_or_distance).
	
	'''
	if not query or not query.strip( ):
		return [ ]
	
	embedder: SentenceTransformer = load_embedder( )
	_docqna_rebuild_index_if_needed( embedder )
	
	qv = embedder.encode( [ query ], show_progress_bar=False )
	qv = np.asarray( qv, dtype=np.float32 )[ 0 ]
	
	if st.session_state.get( 'docqna_vec_ready', False ):
		conn = create_connection( )
		try:
			_docqna_safe_load_sqlite_vec( conn )
			cur = conn.cursor( )
			cur.execute(
				'''
                SELECT doc_name, chunk, distance
                FROM docqna_vec
                WHERE embedding MATCH ?
                ORDER BY distance ASC LIMIT ?;
				''',
				(qv.tobytes( ), int( k ))
			)
			rows = cur.fetchall( )
			return [ (r[ 0 ], r[ 1 ], float( r[ 2 ] )) for r in rows ]
		except Exception:
			st.session_state[ 'docqna_vec_ready' ] = False
		finally:
			conn.close( )
	
	fallback_rows: List[
		Tuple[ str, str, bytes ] ] = st.session_state.get( 'docqna_fallback_rows', [ ] )
	results: List[ Tuple[ str, str, float ] ] = [ ]
	
	for doc_name, chunk_text_value, vec_blob in fallback_rows:
		if not vec_blob:
			continue
		
		v = np.frombuffer( vec_blob, dtype=np.float32 )
		if v.size == 0:
			continue
		
		score = cosine_sim( qv, v )
		results.append( (doc_name, chunk_text_value, float( score )) )
	
	results.sort( key=lambda r: r[ 2 ], reverse=True )
	return results[ : int( k ) ]

def build_document_user_input( user_query: str, k: int = 6 ) -> str:
	'''
	
		Purpose:
		--------
		Builds a Document Q&A prompt that injects retrieved chunks (RAG) instead of stuffing full documents.
	
		Parameters:
		-----------
		user_query:
			The user question.
		k:
			The number of retrieved chunks to include.
	
		Returns:
		--------
		A str prompt suitable for llama.cpp completion.
	
	'''
	system = str( st.session_state.get( 'system_instructions', '' ) or '' ).strip( )
	hits = retrieve_top_doc_chunks( user_query, k=int( k ) )
	
	context_blocks: List[ str ] = [ ]
	for doc_name, chunk, score in hits:
		context_blocks.append( f'[Document: {doc_name}]\n{chunk}'.strip( ) )
	
	context = '\n\n'.join( context_blocks ).strip( )
	
	prompt_parts: List[ str ] = [ ]
	
	if system:
		prompt_parts.append( system )
	
	if context:
		prompt_parts.append(
			'Use the following document excerpts to answer the question. If the excerpts do not contain '
			'the answer, say you do not have enough information.\n\n'
			f'{context}'
		)
	
	prompt_parts.append( f'Question:\n{user_query}\n\nAnswer:' )
	
	return '\n\n'.join( prompt_parts ).strip( )

# ==============================================================================
# DATABASE UTILITIES
# ==============================================================================

def initialize_database( ) -> None:
	Path( "stores/sqlite/datamodels" ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS chat_history
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          role
                          TEXT,
                          content
                          TEXT
                      )
		              """ )
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS embeddings
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          chunk
                          TEXT,
                          vector
                          BLOB
                      )
		              """ )
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS Prompts
                      (
                          PromptsId
                          INTEGER
                          NOT
                          NULL
                          UNIQUE,
                          Name
                          TEXT
                      (
                          80
                      ),
                          Text TEXT,
                          Version TEXT
                      (
                          80
                      ),
                          ID TEXT
                      (
                          80
                      ),
                          PRIMARY KEY
                      (
                          PromptsId
                          AUTOINCREMENT
                      )
                          )
		              """ )

def create_connection( ) -> sqlite3.Connection:
	return sqlite3.connect( cfg.DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int=None, offset: int=0 ) -> pd.DataFrame:
	query = f'SELECT rowid, * FROM "{table}"'
	if limit:
		query += f" LIMIT {limit} OFFSET {offset}"
	with create_connection( ) as conn:
		return pd.read_sql_query( query, conn )

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( "Invalid table name." )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( "Invalid column name." )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ):
	st.subheader( 'Visualization Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	categorical_cols = df.select_dtypes( include=[ 'object' ] ).columns.tolist( )
	
	chart = st.selectbox( 'Chart Type', [ 'Histogram', 'Bar', 'Line',
	                                      'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram' and numeric_cols:
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.histogram( df, x=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.bar( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.line( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		x = st.selectbox( 'X', numeric_cols )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.scatter( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.box( df, y=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		col = st.selectbox( 'Category Column', categorical_cols )
		fig = px.pie( df, names=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation' and len( numeric_cols ) > 1:
		corr = df[ numeric_cols ].corr( )
		fig = px.imshow( corr, text_auto=True )
		st.plotly_chart( fig, use_container_width=True )

def dm_create_table_from_df( table_name: str, df: pd.DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if "int" in dtype_str:
		return "INTEGER"
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if "float" in dtype_str:
		return "REAL"
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if "bool" in dtype_str:
		return "INTEGER"
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if "datetime" in dtype_str:
		return "TEXT"
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if "category" in dtype_str:
		return "TEXT"
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return "TEXT"

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": bool,
					"primary_key": bool,
					"auto_increment": bool
				}
			]
	"""
	if not table_name:
		raise ValueError( "Table name required." )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( "Invalid table name." )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ "name" ]
		col_type = col[ "type" ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ "primary_key" ]:
			definition += " PRIMARY KEY"
			if col[ "auto_increment" ] and col_type == "INTEGER":
				definition += " AUTOINCREMENT"
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ',
	                    'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = \
			{
					'column': col, 'dtype': str( series.dtype ),
					'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
					'distinct_%': round( (
								                     distinct_count / total_rows) * 100, 2 ) if total_rows else 0,
			}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ "min" ] = series.min( )
			row[ "max" ] = series.max( )
			row[ "mean" ] = series.mean( )
		else:
			row[ "min" ] = None
			row[ "max" ] = None
			row[ "mean" ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( "Table and column required." )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		# ------------------------------------------------------------
		# Extract column definitions
		# ------------------------------------------------------------
		open_paren = create_sql.find( "(" )
		close_paren = create_sql.rfind( ")" )
		
		if open_paren == -1 or close_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		inner = create_sql[ open_paren + 1: close_paren ]
		
		column_defs = [ c.strip( ) for c in inner.split( "," ) ]
		
		# Remove target column
		new_defs = [ ]
		for col_def in column_defs:
			col_name = col_def.split( )[ 0 ].strip( '"' )
			if col_name != column:
				new_defs.append( col_def )
		
		if len( new_defs ) == len( column_defs ):
			raise ValueError( "Column not found." )
		
		# ------------------------------------------------------------
		# Build new CREATE TABLE statement
		# ------------------------------------------------------------
		temp_table = f"{table}_rebuild_temp"
		
		new_create_sql = (
				f'CREATE TABLE "{temp_table}" ('
				+ ", ".join( new_defs )
				+ ");"
		)
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [
				c.split( )[ 0 ].strip( '"' )
				for c in new_defs
		]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		# Preserve indexes
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute(
			f'ALTER TABLE "{temp_table}" RENAME TO "{table}";'
		)
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

# ==============================================================================
# PROMPT ENGINEERING UTILITIES
# ==============================================================================

def fetch_prompt_names( db_path: str ) -> list[ str ]:
	"""
		Purpose:
		--------
		Retrieve template names from Prompts table.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
	
		Returns:
		--------
		list[str]
			Sorted prompt names.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Caption FROM Prompts ORDER BY PromptsId;" )
		rows = cur.fetchall( )
		conn.close( )
		return [ r[ 0 ] for r in rows if r and r[ 0 ] is not None ]
	except Exception:
		return [ ]

def fetch_prompt_text( db_path: str, name: str ) -> str | None:
	"""
		Purpose:
		--------
		Retrieve template text by name.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
		name : str
			Template name.
	
		Returns:
		--------
		str | None
			Prompt text if found.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Text FROM Prompts WHERE Caption = ?;", (name,) )
		row = cur.fetchone( )
		conn.close( )
		return str( row[ 0 ] ) if row and row[ 0 ] is not None else None
	except Exception:
		return None

def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Caption,  Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn )
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE Caption=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO Prompts (Caption, Name, Text, Version, ID) VALUES (?, ?, ?, ?)',
			(data[ 'Caption' ], data[ 'Name' ], data[ 'Text' ], data[ 'Version' ], data[ 'ID' ]) )

def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Caption=?, Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ "Caption" ], data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ],
			 pid)
		)

def delete_prompt( pid: int ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM Prompts WHERE PromptsId=?", (pid,) )

def build_prompt( user_input: str ) -> str:
	prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"
	
	if st.session_state.use_semantic:
		with sqlite3.connect( DB_PATH ) as conn:
			rows = conn.execute( "SELECT chunk, vector FROM embeddings" ).fetchall( )
		if rows:
			q = embedder.encode( [ user_input ] )[ 0 ]
			scored = [ (c, cosine_sim( q, np.frombuffer( v ) )) for c, v in rows ]
			for c, _ in sorted( scored, key=lambda x: x[ 1 ], reverse=True )[ :top_k ]:
				prompt += f"<|system|>\n{c}\n</s>\n"
	
	for d in st.session_state.basic_docs[ :6 ]:
		prompt += f"<|system|>\n{d}\n</s>\n"
	
	for r, c in st.session_state.messages:
		prompt += f"<|{r}|>\n{c}\n</s>\n"
	
	prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
	return prompt

# ======================================================================================
#  PROVIDER UTILITIES
# ======================================================================================

def get_provider_name( provider: Optional[ str ] = None ) -> str:
	"""
		
		Purpose:
		--------
		Return a validated provider name for provider-aware wrapper dispatch.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		str: Valid provider name mapped in cfg.PROVIDERS.
		
	"""
	selected = provider or st.session_state.get( 'provider', 'GPT' )
	providers = getattr( cfg, 'PROVIDERS', { 'GPT': 'gpt', 'Gemini': 'gemini', 'Grok': 'grok' } )
	
	if selected not in providers:
		selected = 'GPT'
		st.session_state[ 'provider' ] = selected
	
	return selected

def get_provider_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return the imported provider module for the selected provider without using dynamic
		class-name imports that overwrite shared wrapper names.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Imported provider module, such as gpt, gemini, or grok.
		
	"""
	selected = get_provider_name( provider )
	provider_modules = {
			'GPT': gpt,
			'Gemini': gemini,
			'Grok': grok,
	}
	
	module = provider_modules.get( selected )
	if module is None:
		raise ValueError( f'Provider "{selected}" is not mapped to an imported module.' )
	
	return module

def provider_has_class( class_name: str, provider: Optional[ str ] = None ) -> bool:
	"""
		
		Purpose:
		--------
		Determine whether the selected provider module exposes a wrapper class.
	
		Parameters:
		-----------
		class_name (str): Public wrapper class name to inspect.
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		bool: True if the selected provider exposes the requested class; otherwise, False.
		
	"""
	if not class_name:
		return False
	
	provider_module = get_provider_module( provider )
	return hasattr( provider_module, class_name )

def get_provider_class( class_name: str, provider: Optional[ str ] = None ) -> type:
	"""
		
		Purpose:
		--------
		Return a provider wrapper class by common interface name.
	
		Parameters:
		-----------
		class_name (str): Public wrapper class name to retrieve.
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		type: Provider wrapper class.
		
	"""
	if not class_name:
		raise ValueError( 'class_name cannot be empty.' )
	
	selected = get_provider_name( provider )
	provider_module = get_provider_module( selected )
	
	if not hasattr( provider_module, class_name ):
		raise AttributeError(
			f'Provider "{selected}" does not expose a "{class_name}" wrapper class.'
		)
	
	return getattr( provider_module, class_name )

def get_provider_instance( class_name: str, provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Instantiate a provider wrapper class by common interface name.
	
		Parameters:
		-----------
		class_name (str): Public wrapper class name to instantiate.
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Provider wrapper instance.
		
	"""
	provider_class = get_provider_class( class_name, provider )
	return provider_class( )

def get_chat_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a Chat wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Chat wrapper instance.
		
	"""
	return get_provider_instance( 'Chat', provider )

def get_tts_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a TTS wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: TTS wrapper instance.
		
	"""
	return get_provider_instance( 'TTS', provider )

def get_images_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return an Images wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Images wrapper instance.
		
	"""
	return get_provider_instance( 'Images', provider )

def get_embeddings_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return an Embeddings wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Embeddings wrapper instance.
		
	"""
	return get_provider_instance( 'Embeddings', provider )

def get_translation_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a Translation wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Translation wrapper instance.
		
	"""
	return get_provider_instance( 'Translation', provider )

def get_transcription_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a Transcription wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Transcription wrapper instance.
		
	"""
	return get_provider_instance( 'Transcription', provider )

def get_files_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a Files wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: Files wrapper instance.
		
	"""
	return get_provider_instance( 'Files', provider )

def get_vectorstores_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a VectorStores wrapper instance for the selected provider.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: VectorStores wrapper instance.
		
	"""
	return get_provider_instance( 'VectorStores', provider )

def get_file_search_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a FileSearch wrapper instance for providers that expose file-search-store
		functionality.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: FileSearch wrapper instance.
		
	"""
	return get_provider_instance( 'FileSearch', provider )

def get_cloud_buckets_module( provider: Optional[ str ] = None ) -> Any:
	"""
		
		Purpose:
		--------
		Return a CloudBuckets wrapper instance for providers that expose cloud-bucket
		functionality.
	
		Parameters:
		-----------
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		Any: CloudBuckets wrapper instance.
		
	"""
	return get_provider_instance( 'CloudBuckets', provider )

def get_mode_classes( mode: Optional[ str ]=None, provider: Optional[ str ]=None ) -> List[ str ]:
	"""
		
		Purpose:
		--------
		Return the wrapper class names mapped to a Boo mode for the selected provider.
	
		Parameters:
		-----------
		mode (Optional[str]): Mode name override. If omitted, the mode is read from
			st.session_state['mode'].
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		List[str]: Wrapper class names associated with the selected mode.
		
	"""
	selected_mode = mode or st.session_state.get( 'mode', 'Text' )
	selected_provider = get_provider_name( provider )
	provider_class_map = getattr( cfg, 'PROVIDER_CLASS_MAP', None )
	
	if isinstance( provider_class_map, dict ):
		provider_modes = provider_class_map.get( selected_provider, { } )
		mapped = provider_modes.get( selected_mode, [ ] )
		
		if isinstance( mapped, str ):
			return [ mapped ]
		
		if isinstance( mapped, list ):
			return mapped
	
	mode_class_map = getattr( cfg, 'MODE_CLASS_MAP', { } )
	mapped = mode_class_map.get( selected_mode, [ ] )
	
	if isinstance( mapped, str ):
		return [ mapped ]
	
	if isinstance( mapped, list ):
		return mapped
	
	return [ ]

def provider_supports_mode( mode: Optional[ str ] = None,
		provider: Optional[ str ] = None ) -> bool:
	"""
		
		Purpose:
		--------
		Determine whether the selected provider exposes all wrapper classes required by a mode.
	
		Parameters:
		-----------
		mode (Optional[str]): Mode name override. If omitted, the mode is read from
			st.session_state['mode'].
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		bool: True if the selected provider supports all classes mapped to the selected mode.
		
	"""
	classes = get_mode_classes( mode, provider )
	if not classes:
		return True
	
	return all( provider_has_class( class_name, provider ) for class_name in classes )

def require_provider_mode( mode: Optional[ str ] = None, provider: Optional[ str ] = None ) -> bool:
	"""
		
		Purpose:
		--------
		Display a Streamlit warning when the selected provider does not expose all wrapper
		classes required by a mode.
	
		Parameters:
		-----------
		mode (Optional[str]): Mode name override. If omitted, the mode is read from
			st.session_state['mode'].
		provider (Optional[str]): Provider name override. If omitted, the provider is read from
			st.session_state['provider'].
	
		Returns:
		--------
		bool: True if the provider supports the mode; otherwise, False.
		
	"""
	selected_mode = mode or st.session_state.get( 'mode', 'Text' )
	selected_provider = get_provider_name( provider )
	classes = get_mode_classes( selected_mode, selected_provider )
	missing = [
			class_name for class_name in classes
			if not provider_has_class( class_name, selected_provider )
	]
	
	if missing:
		st.warning(
			f'{selected_provider} does not currently expose the required wrapper(s) for '
			f'{selected_mode}: {", ".join( missing )}.'
		)
		return False
	
	return True

def _provider( ) -> str:
	"""
		
		Purpose:
		--------
		Return the currently selected provider name.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		str: Current provider name.
		
	"""
	return get_provider_name( )

def _safe( module: str, attr: str, fallback: Any ) -> Any:
	"""
		
		Purpose:
		--------
		Safely retrieve an attribute from an imported module by module name.
	
		Parameters:
		-----------
		module (str): Module name to import.
		attr (str): Attribute name to retrieve.
		fallback (Any): Value returned if the module or attribute cannot be resolved.
	
		Returns:
		--------
		Any: Resolved module attribute or fallback value.
		
	"""
	try:
		mod = __import__( module )
		return getattr( mod, attr, fallback )
	except Exception:
		return fallback

# ==============================================================================
# Page Setup
# ==============================================================================
AVATARS = { 'user': cfg.ANALYST, 'assistant': cfg.BOO, }
st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON,
	initial_sidebar_state='collapsed', )

st.caption( cfg.APP_SUBTITLE )
inject_response_css( )
init_state( )

# ======================================================================================
# SIDEBAR
# ======================================================================================

def get_provider_options( ) -> List[ str ]:
	"""
		
		Purpose:
		--------
		Return configured provider names for the Boo sidebar provider selector.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		List[str]: Provider names from cfg.PROVIDERS.
		
	"""
	providers = getattr( cfg, 'PROVIDERS', { 'GPT': 'gpt', 'Gemini': 'gemini', 'Grok': 'grok' } )
	return list( providers.keys( ) )

def get_raw_provider_modes( provider: str ) -> List[ str ]:
	"""
		
		Purpose:
		--------
		Return the configured mode list for a provider before runtime wrapper filtering.
	
		Parameters:
		-----------
		provider (str): Provider name selected in the sidebar.
	
		Returns:
		--------
		List[str]: Configured mode names for the provider.
		
	"""
	class_mode_map = getattr( cfg, 'CLASS_MODE_MAP', None )
	
	if isinstance( class_mode_map, dict ) and provider in class_mode_map:
		modes = class_mode_map.get( provider, [ ] )
		return list( modes or [ ] )
	
	if provider == 'Gemini':
		return list( getattr( cfg, 'GEMINI_MODES', [ ] ) )
	
	if provider == 'Grok':
		return list( getattr( cfg, 'GROK_MODES', [ ] ) )
	
	return list( getattr( cfg, 'GPT_MODES', [ ] ) )

def normalize_mode_name( mode_name: Optional[ str ] ) -> str:
	"""
		
		Purpose:
		--------
		Normalize legacy mode labels to Boo's preferred canonical labels.
	
		Parameters:
		-----------
		mode_name (Optional[str]): Mode label to normalize.
	
		Returns:
		--------
		str: Canonical mode label.
		
	"""
	if not mode_name:
		return 'Text'
	
	mode_aliases = {
			'Embedding': 'Embeddings',
			'Documents': 'Document Q&A',
			'Data Export': 'Export',
			'Export Data': 'Export',
	}
	
	return mode_aliases.get( mode_name, mode_name )

def normalize_mode_list( modes: List[ str ] ) -> List[ str ]:
	"""
		
		Purpose:
		--------
		Normalize configured mode names and remove duplicates while preserving order.
	
		Parameters:
		-----------
		modes (List[str]): Raw configured mode labels.
	
		Returns:
		--------
		List[str]: Ordered canonical mode labels.
		
	"""
	normalized = [ ]
	
	for item in modes:
		mode_name = normalize_mode_name( item )
		if mode_name not in normalized:
			normalized.append( mode_name )
	
	return normalized

def mode_requires_runtime_wrapper( mode_name: str ) -> bool:
	"""
		
		Purpose:
		--------
		Determine whether a mode should be filtered by provider wrapper availability.
	
		Parameters:
		-----------
		mode_name (str): Canonical mode label.
	
		Returns:
		--------
		bool: True if mode requires wrapper classes; otherwise, False.
		
	"""
	non_wrapper_modes = [
			'Prompt Engineering',
			'Data Management',
			'Export',
	]
	
	return mode_name not in non_wrapper_modes

def get_supported_provider_modes( provider: str ) -> List[ str ]:
	"""
		
		Purpose:
		--------
		Return provider modes filtered by configured mode lists and runtime wrapper support.
	
		Parameters:
		-----------
		provider (str): Provider name selected in the sidebar.
	
		Returns:
		--------
		List[str]: Mode labels safe to present for the selected provider.
		
	"""
	raw_modes = get_raw_provider_modes( provider )
	modes = normalize_mode_list( raw_modes )
	supported = [ ]
	
	for mode_name in modes:
		if not mode_requires_runtime_wrapper( mode_name ):
			supported.append( mode_name )
			continue
		
		if provider_supports_mode( mode_name, provider ):
			supported.append( mode_name )
	
	if not supported:
		supported = [ 'Text' ]
	
	return supported

def get_mode_index( modes: List[ str ], current_mode: Optional[ str ] ) -> int:
	"""
		
		Purpose:
		--------
		Return a safe index for the current mode within a filtered mode list.
	
		Parameters:
		-----------
		modes (List[str]): Filtered provider mode labels.
		current_mode (Optional[str]): Current session-state mode.
	
		Returns:
		--------
		int: Safe selected index.
		
	"""
	mode_name = normalize_mode_name( current_mode )
	
	if mode_name in modes:
		return modes.index( mode_name )
	
	return 0

def render_provider_keys( ) -> None:
	"""
		
		Purpose:
		--------
		Render API key controls and update session/environment values from user input.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		None
		
	"""
	with st.expander( 'Keys:', expanded=False ):
		openai_key = st.text_input(
			'OpenAI API Key',
			type='password',
			value=st.session_state.get( 'openai_api_key', '' ) or '',
			help='Overrides OPENAI_API_KEY from config.py for this session only.',
			key='sidebar_openai_api_key'
		)
		
		gemini_key = st.text_input(
			'Gemini API Key',
			type='password',
			value=st.session_state.get( 'gemini_api_key', '' ) or '',
			help='Overrides GEMINI_API_KEY from config.py for this session only.',
			key='sidebar_gemini_api_key'
		)
		
		groq_key = st.text_input(
			'Groq API Key',
			type='password',
			value=st.session_state.get( 'groq_api_key', '' ) or '',
			help='Overrides GROQ_API_KEY from config.py for this session only.',
			key='sidebar_groq_api_key'
		)
		
		google_key = st.text_input(
			'Google API Key',
			type='password',
			value=st.session_state.get( 'google_api_key', '' ) or '',
			help='Overrides GOOGLE_API_KEY from config.py for this session only.',
			key='sidebar_google_api_key'
		)
		
		google_cse_id = st.text_input(
			'Google CSE ID',
			type='password',
			value=st.session_state.get( 'google_cse_id', '' ) or '',
			help='Overrides GOOGLE_CSE_ID from config.py for this session only.',
			key='sidebar_google_cse_id'
		)
		
		google_cloud_project_id = st.text_input(
			'Google Cloud Project ID',
			type='password',
			value=st.session_state.get( 'google_cloud_project_id', '' ) or '',
			help='Overrides GOOGLE_CLOUD_PROJECT_ID from config.py for this session only.',
			key='sidebar_google_cloud_project_id'
		)
		
		google_cloud_location = st.text_input(
			'Google Cloud Location',
			value=st.session_state.get( 'google_cloud_location', '' ) or '',
			help='Overrides GOOGLE_CLOUD_LOCATION from config.py for this session only.',
			key='sidebar_google_cloud_location'
		)
		
		if openai_key:
			st.session_state[ 'openai_api_key' ] = openai_key
			os.environ[ 'OPENAI_API_KEY' ] = openai_key
		
		if gemini_key:
			st.session_state[ 'gemini_api_key' ] = gemini_key
			os.environ[ 'GEMINI_API_KEY' ] = gemini_key
		
		if groq_key:
			st.session_state[ 'groq_api_key' ] = groq_key
			os.environ[ 'GROQ_API_KEY' ] = groq_key
		
		if google_key:
			st.session_state[ 'google_api_key' ] = google_key
			os.environ[ 'GOOGLE_API_KEY' ] = google_key
		
		if google_cse_id:
			st.session_state[ 'google_cse_id' ] = google_cse_id
			os.environ[ 'GOOGLE_CSE_ID' ] = google_cse_id
		
		if google_cloud_project_id:
			st.session_state[ 'google_cloud_project_id' ] = google_cloud_project_id
			os.environ[ 'GOOGLE_CLOUD_PROJECT_ID' ] = google_cloud_project_id
		
		if google_cloud_location:
			st.session_state[ 'google_cloud_location' ] = google_cloud_location
			os.environ[ 'GOOGLE_CLOUD_LOCATION' ] = google_cloud_location

with st.sidebar:
	provider_options = get_provider_options( )
	current_provider = st.session_state.get( 'provider', 'GPT' )
	
	if current_provider not in provider_options:
		current_provider = provider_options[ 0 ] if provider_options else 'GPT'
		st.session_state[ 'provider' ] = current_provider
	
	style_subheaders( )
	st.subheader( 'Provider' )
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	provider = st.selectbox(
		label='Choose provider',
		options=provider_options,
		index=provider_options.index( current_provider ),
		key='provider'
	)
	
	logo_path = getattr( cfg, 'LOGO_MAP', { } ).get( provider )
	if logo_path:
		st.logo( logo_path, size='large' )
	
	render_provider_keys( )
	
	st.subheader( 'Mode' )
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	mode_options = get_supported_provider_modes( provider )
	current_mode = normalize_mode_name( st.session_state.get( 'mode', 'Text' ) )
	
	if current_mode not in mode_options:
		current_mode = mode_options[ 0 ]
		st.session_state[ 'mode' ] = current_mode
	
	mode = st.radio(
		label='Select Mode',
		options=mode_options,
		index=get_mode_index( mode_options, current_mode ),
		key='mode'
	)
	
	st.caption( f'Provider: {provider} | Mode: {mode}' )

# ======================================================================================
# TEXT MODE
# ======================================================================================
if mode == 'Text':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	text = get_chat_module( provider_name )
	
	# ------------------------------------------------------------------
	# Text Mode Helpers
	# ------------------------------------------------------------------
	def get_text_options( instance: Any, attr_name: str,
			fallback: Optional[ List[ str ] ] = None ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return list-like option values from a provider wrapper property.
		
			Parameters:
			-----------
			instance (Any): Provider wrapper instance.
			attr_name (str): Property or attribute name to inspect.
			fallback (Optional[List[str]]): Fallback values returned when the attribute is absent.
		
			Returns:
			--------
			List[str]: Option values safe for Streamlit controls.
			
		"""
		values = getattr( instance, attr_name, None )
		if callable( values ):
			try:
				values = values( )
			except Exception:
				values = None
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if isinstance( values, list ):
			return [ str( value ) for value in values if str( value ).strip( ) ]
		
		return fallback or [ ]
	
	def get_text_help( name: str, fallback: str = '' ) -> str:
		"""
			
			Purpose:
			--------
			Return help text from config.py without failing when a constant is absent.
		
			Parameters:
			-----------
			name (str): Config attribute name.
			fallback (str): Fallback help text.
		
			Returns:
			--------
			str: Help text value.
			
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def parse_semicolon_list( value: Any ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Parse semicolon-delimited user input into an ordered list of strings.
		
			Parameters:
			-----------
			value (Any): User input value.
		
			Returns:
			--------
			List[str]: Parsed non-empty values.
			
		"""
		raw_value = str( value or '' )
		return [ item.strip( ) for item in raw_value.split( ';' ) if item.strip( ) ]
	
	def parse_comma_list( value: Any ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Parse comma-delimited user input into an ordered list of strings.
		
			Parameters:
			-----------
			value (Any): User input value.
		
			Returns:
			--------
			List[str]: Parsed non-empty values.
			
		"""
		raw_value = str( value or '' )
		return [ item.strip( ) for item in raw_value.split( ',' ) if item.strip( ) ]
	
	def normalize_bool_or_none( value: Any ) -> Optional[ bool ]:
		"""
			
			Purpose:
			--------
			Normalize optional API booleans so falsey UI toggles can be omitted when necessary.
		
			Parameters:
			-----------
			value (Any): Boolean-like value.
		
			Returns:
			--------
			Optional[bool]: True, False, or None.
			
		"""
		if value is None:
			return None
		
		return bool( value )
	
	def reset_text_model_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Text mode model-oriented controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'text_model',
				'text_reasoning',
				'text_modalities',
				'text_media_resolution',
				'text_number',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_text_inference_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Text mode inference controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'text_temperature',
				'text_top_percent',
				'text_top_k',
				'text_frequency_penalty',
				'text_presence_penalty',
				'text_presense_penalty',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_text_tool_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Text mode tool, include, vector-store, and grounding controls.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'text_google_grounding',
				'text_max_calls',
				'text_tool_choice',
				'text_include',
				'text_tools',
				'text_domains_input',
				'text_domains',
				'text_parallel_tools',
				'text_parallel_calls',
				'text_vector_store_ids',
				'text_urls_input',
				'text_urls',
				'text_max_urls',
				'selected_filestore_id',
				'selected_filestore_label',
				'text_file_search_store_names',
				'text_file_search_store_select',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_text_response_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Text mode response/output controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'text_stream',
				'text_store',
				'text_max_tokens',
				'text_background',
				'text_response_format',
				'text_response_schema',
				'text_json_schema',
				'text_json_schema_name',
				'text_json_schema_strict',
				'text_stops',
				'text_stops_input',
				'text_previous_response_id',
				'text_conversation_id',
				'text_input',
				'text_safety_profile',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def clear_text_instructions( ) -> None:
		"""
			
			Purpose:
			--------
			Clear Text mode system instructions and selected prompt template.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'text_system_instructions' ] = ''
		st.session_state[ 'instructions' ] = ''
	
	def convert_text_system_instructions( ) -> None:
		"""
			
			Purpose:
			--------
			Convert Text mode system instructions between XML-like blocks and Markdown headings.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		text_value = st.session_state.get( 'text_system_instructions', '' )
		if not isinstance( text_value, str ) or not text_value.strip( ):
			return
		
		source = text_value.strip( )
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'text_system_instructions' ] = converted
	
	def load_text_instruction_template( ) -> None:
		"""
			
			Purpose:
			--------
			Load the selected prompt template into Text mode system instructions.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		name = st.session_state.get( 'instructions' )
		if name and name != 'No Templates Found':
			prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
			if prompt_text is not None:
				st.session_state[ 'text_system_instructions' ] = prompt_text
	
	def build_text_response_format_payload( ) -> Any:
		"""
			
			Purpose:
			--------
			Build a provider-safe response format payload from Text mode response controls.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			Any: String format value, JSON schema payload, or None.
			
		"""
		response_format = st.session_state.get( 'text_response_format', '' )
		schema_text = st.session_state.get( 'text_json_schema', '' )
		schema_name = st.session_state.get( 'text_json_schema_name', 'response_schema' )
		strict = bool( st.session_state.get( 'text_json_schema_strict', True ) )
		
		if provider_name == 'GPT' and schema_text and str(
				response_format ).lower( ) == 'json_schema':
			try:
				import json
				
				schema = json.loads( schema_text )
				return {
						'type': 'json_schema',
						'json_schema': {
								'name': schema_name or 'response_schema',
								'strict': strict,
								'schema': schema,
						},
				}
			except Exception:
				st.warning( 'Text JSON schema is not valid JSON. Falling back to selected format.' )
		
		return response_format or None
	
	def build_text_context( include_last_message: bool = False ) -> List[ Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			Build provider-neutral chat context from Text mode message history.
		
			Parameters:
			-----------
			include_last_message (bool): Whether the most recent message should be included.
		
			Returns:
			--------
			List[Dict[str, Any]]: Chat context messages.
			
		"""
		messages = st.session_state.get( 'text_messages', [ ] )
		if not isinstance( messages, list ):
			return [ ]
		
		if include_last_message:
			return messages
		
		return messages[ :-1 ]
	
	def build_text_common_kwargs( prompt: str ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build provider-neutral Text mode keyword arguments for shared wrapper calls.
		
			Parameters:
			-----------
			prompt (str): User prompt submitted through the chat input.
		
			Returns:
			--------
			Dict[str, Any]: Keyword arguments used by provider-specific dispatch.
			
		"""
		raw_stops = st.session_state.get( 'text_stops_input', '' )
		raw_urls = st.session_state.get( 'text_urls_input', '' )
		raw_domains = st.session_state.get( 'text_domains_input', '' )
		
		derived_stops = parse_comma_list( raw_stops )
		derived_urls = parse_semicolon_list( raw_urls )
		derived_domains = parse_comma_list( raw_domains )
		if derived_domains:
			st.session_state[ 'text_domains' ] = derived_domains
		
		derived_modalities = [ str( modality ).strip( )
				for modality in st.session_state.get( 'text_modalities', [ ] )
				if str( modality ).strip( ) ]
		
		return {
				'prompt': prompt,
				'model': st.session_state.get( 'text_model' ),
				'number': st.session_state.get( 'text_number' ),
				'temperature': st.session_state.get( 'text_temperature' ),
				'top_p': st.session_state.get( 'text_top_percent' ),
				'top_k': st.session_state.get( 'text_top_k' ),
				'frequency': st.session_state.get( 'text_frequency_penalty' ),
				'presence': st.session_state.get( 'text_presence_penalty' ),
				'max_tokens': st.session_state.get( 'text_max_tokens' ),
				'stops': derived_stops,
				'instruct': st.session_state.get( 'text_system_instructions' ),
				'response_format': st.session_state.get( 'text_response_format' ) or None,
				'tools': st.session_state.get( 'text_tools', [ ] ),
				'tool_choice': st.session_state.get( 'text_tool_choice' ) or None,
				'reasoning': st.session_state.get( 'text_reasoning' ) or None,
				'modalities': derived_modalities,
				'media_resolution': st.session_state.get( 'text_media_resolution' ) or None,
				'context': build_text_context( include_last_message=False ),
				'content': st.session_state.get( 'text_content' ),
				'urls': derived_urls,
				'max_urls': st.session_state.get( 'text_max_urls' ),
				'response_schema': st.session_state.get( 'text_response_schema' ) or None,
				'safety_profile': st.session_state.get( 'text_safety_profile' ) or None,
		}
	
	def build_gpt_text_kwargs( prompt: str ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build GPT/OpenAI-oriented Text mode keyword arguments for the GPT wrapper.
		
			Parameters:
			-----------
			prompt (str): User prompt submitted through the chat input.
		
			Returns:
			--------
			Dict[str, Any]: GPT-compatible keyword arguments.
			
		"""
		vector_store_ids = parse_comma_list( st.session_state.get( 'text_vector_store_ids', '' ) )
		tools = st.session_state.get( 'text_tools', [ ] )
		include = st.session_state.get( 'text_include', [ ] )
		tool_choice = st.session_state.get( 'text_tool_choice' ) or None
		input_mode = st.session_state.get( 'text_input', '' )
		
		if 'file_search' in tools and vector_store_ids:
			text_tools = [
					{
							'type': 'file_search',
							'vector_store_ids': vector_store_ids,
					}
			]
		else:
			text_tools = tools
		
		if input_mode == 'single_turn':
			context = [ ]
		else:
			context = build_text_context( include_last_message=False )
		
		return {
				'prompt': prompt,
				'model': st.session_state.get( 'text_model' ),
				'temperature': st.session_state.get( 'text_temperature' ),
				'format': build_text_response_format_payload( ),
				'top_p': st.session_state.get( 'text_top_percent' ),
				'frequency': st.session_state.get( 'text_frequency_penalty' ),
				'presence': st.session_state.get( 'text_presence_penalty' ),
				'max_tools': st.session_state.get( 'text_max_calls' ),
				'max_tokens': st.session_state.get( 'text_max_tokens' ),
				'store': normalize_bool_or_none( st.session_state.get( 'text_store' ) ),
				'stream': normalize_bool_or_none( st.session_state.get( 'text_stream' ) ),
				'instruct': st.session_state.get( 'text_system_instructions' ),
				'background': normalize_bool_or_none( st.session_state.get( 'text_background' ) ),
				'reasoning': st.session_state.get( 'text_reasoning' ) or None,
				'include': include,
				'tools': text_tools,
				'allowed_domains': st.session_state.get( 'text_domains', [ ] ),
				'previous_id': st.session_state.get( 'text_previous_response_id' ) or None,
				'tool_choice': tool_choice,
				'is_parallel': bool( st.session_state.get( 'text_parallel_tools', False ) ),
				'context': context,
				'vector_store_ids': vector_store_ids,
				'conversation_id': st.session_state.get( 'text_conversation_id' ) or None,
		}
	
	def build_gemini_text_kwargs( prompt: str, stream_handler: Optional[ Any ]=None ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build Gemini-oriented Text mode keyword arguments for the Gemini wrapper.
		
			Parameters:
			-----------
			prompt (str): User prompt submitted through the chat input.
			stream_handler (Optional[Any]): Optional callback for streamed output chunks.
		
			Returns:
			--------
			Dict[str, Any]: Gemini-compatible keyword arguments.
			
		"""
		kwargs = build_text_common_kwargs( prompt )
		grounding_enabled = bool( st.session_state.get( 'text_google_grounding', False ) )
		kwargs[ 'tools' ] = [ 'google_search' ] if grounding_enabled else st.session_state.get(
			'text_tools', [ ] )
		kwargs[ 'file_search_store_names' ] = st.session_state.get( 'text_file_search_store_names',
			[ ] )
		kwargs[ 'stream' ] = bool( st.session_state.get( 'text_stream', False ) )
		kwargs[ 'stream_handler' ] = stream_handler
		return kwargs
	
	def build_grok_text_kwargs( prompt: str ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build Grok-oriented Text mode keyword arguments for the Grok wrapper.
		
			Parameters:
			-----------
			prompt (str): User prompt submitted through the chat input.
		
			Returns:
			--------
			Dict[str, Any]: Grok-compatible keyword arguments.
			
		"""
		kwargs = build_text_common_kwargs( prompt )
		kwargs[ 'include' ] = st.session_state.get( 'text_include', [ ] )
		kwargs[ 'allowed_domains' ] = st.session_state.get( 'text_domains', [ ] )
		kwargs[ 'background' ] = normalize_bool_or_none( st.session_state.get( 'text_background' ) )
		kwargs[ 'stream' ] = normalize_bool_or_none( st.session_state.get( 'text_stream' ) )
		kwargs[ 'store' ] = normalize_bool_or_none( st.session_state.get( 'text_store' ) )
		kwargs[ 'is_parallel' ] = bool( st.session_state.get( 'text_parallel_tools', False ) )
		return kwargs
	
	def call_generate_text( prompt: str, stream_handler: Optional[ Any ] = None ) -> Any:
		"""
			
			Purpose:
			--------
			Dispatch Text generation to the selected provider while preserving each provider's
			shared public wrapper interface.
		
			Parameters:
			-----------
			prompt (str): User prompt submitted through the chat input.
			stream_handler (Optional[Any]): Optional callback for streamed output chunks.
		
			Returns:
			--------
			Any: Provider response text or response object.
			
		"""
		if provider_name == 'GPT':
			kwargs = build_gpt_text_kwargs( prompt )
		elif provider_name == 'Gemini':
			kwargs = build_gemini_text_kwargs( prompt, stream_handler )
		else:
			kwargs = build_grok_text_kwargs( prompt )
		
		try:
			return text.generate_text( **kwargs )
		except TypeError:
			clean_kwargs = {
					key: value
					for key, value in kwargs.items( )
					if value is not None and value != '' and value != [ ]
			}
			return text.generate_text( **clean_kwargs )
	
	def extract_text_sources( instance: Any, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			Extract grounding or retrieval sources from a provider wrapper when available.
		
			Parameters:
			-----------
			instance (Any): Provider wrapper instance.
			response (Any): Provider response object or response text.
		
			Returns:
			--------
			List[Dict[str, Any]]: Source dictionaries for rendering.
			
		"""
		if hasattr( instance, 'get_grounding_sources' ):
			try:
				sources = instance.get_grounding_sources( )
				if isinstance( sources, list ):
					return sources
			except Exception:
				pass
		
		if 'extract_sources' in globals( ):
			try:
				sources = extract_sources( response )
				if isinstance( sources, list ):
					return sources
			except Exception:
				pass
		
		return [ ]
	
	def update_text_usage( response: Any ) -> None:
		"""
			
			Purpose:
			--------
			Update token counters using whichever counter helper exists in the current Boo file.
		
			Parameters:
			-----------
			response (Any): Provider response object.
		
			Returns:
			--------
			None
			
		"""
		try:
			if 'update_token_counters' in globals( ):
				update_token_counters( response )
			elif '_update_token_counters' in globals( ):
				_update_token_counters( response )
		except Exception:
			pass
	
	def get_text_avatar( role: str ) -> str:
		"""
			
			Purpose:
			--------
			Return the avatar used for Text mode chat messages.
		
			Parameters:
			-----------
			role (str): Message role.
		
			Returns:
			--------
			str: Avatar token or empty string.
			
		"""
		if role != 'assistant':
			return ''
		
		if provider_name == 'Gemini':
			return getattr( cfg, 'JENI', getattr( cfg, 'BOO', '🧠' ) )
		
		if provider_name == 'GPT':
			return getattr( cfg, 'GPT', getattr( cfg, 'BOO', '🧠' ) )
		
		if provider_name == 'Grok':
			return getattr( cfg, 'GROK', getattr( cfg, 'BOO', '🧠' ) )
		
		return getattr( cfg, 'BOO', '' )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '💬 Text Generation', help=get_text_help( 'TEXT_GENERATION' ) )
		st.divider( )
		
		if st.session_state.get( 'clear_instructions' ):
			st.session_state[ 'text_system_instructions' ] = ''
			st.session_state[ 'instructions_last_loaded' ] = ''
			st.session_state[ 'clear_instructions' ] = False
		
		# ------------------------------------------------------------------
		# Expander — Text Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			
			with st.expander( label='Model Settings', icon='🧊', expanded=False, width='stretch' ):
				model_c1, model_c2, model_c3, model_c4, model_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Model ------------
				with model_c1:
					model_options = get_text_options( text, 'model_options' )
					if not model_options:
						model_options = [
								st.session_state.get( 'text_model', '' ) or getattr( text, 'model',
									'' ) ]
						model_options = [ item for item in model_options if item ]
					
					st.selectbox( label='Model', options=model_options, key='text_model',
						placeholder='Options', index=None,
						help='Required. Text generation model used by the selected provider.' )
				
				# ---------- Reasoning ------------
				with model_c2:
					reasoning_options = get_text_options( text, 'reasoning_options' )
					st.selectbox( label='Reasoning', options=reasoning_options,
						key='text_reasoning',
						help=get_text_help( 'REASONING' ), index=None, placeholder='Options' )
				
				# ---------- Modalities ------------
				with model_c3:
					modality_options = get_text_options( text, 'modality_options',
						[ 'text' ] )
					st.multiselect( label='Modalities', options=modality_options,
						key='text_modalities',
						help='Optional. Provider-supported response modalities.',
						placeholder='Options' )
				
				# ---------- Media Resolution ------------
				with model_c4:
					media_options = get_text_options( text, 'media_options' )
					st.selectbox( label='Media Resolution', options=media_options,
						key='text_media_resolution',
						help='Optional. Provider-supported media resolution.',
						index=None, placeholder='Options' )
				
				# ---------- Candidate Count ------------
				with model_c5:
					st.slider( label='Responses', min_value=0, max_value=8, step=1,
						key='text_number',
						help='Optional. Number of candidate responses when supported.' )
				
				st.button( label='Reset', key='text_model_reset', width='stretch',
					on_click=reset_text_model_settings )
			
			with st.expander( label='Inference Settings', icon='🎚️', expanded=False,
					width='stretch' ):
				prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Top-P ------------
				with prm_c1:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0, step=0.01,
						help=get_text_help( 'TOP_P' ), key='text_top_percent' )
				
				# ---------- Top-K ------------
				with prm_c2:
					st.slider( label='Top-K', min_value=0, max_value=200, step=1,
						help=get_text_help( 'TOP_K' ), key='text_top_k' )
				
				# ---------- Temperature ------------
				with prm_c3:
					st.slider( label='Temperature', min_value=0.0, max_value=2.0,
						step=0.01, help=get_text_help( 'TEMPERATURE' ),
						key='text_temperature' )
				
				# ---------- Frequency Penalty ------------
				with prm_c4:
					st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
						step=0.01, help=get_text_help( 'FREQUENCY_PENALTY' ),
						key='text_frequency_penalty' )
				
				# ---------- Presence Penalty ------------
				with prm_c5:
					st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
						step=0.01, help=get_text_help( 'PRESENCE_PENALTY' ),
						key='text_presence_penalty' )
					st.session_state[ 'text_presense_penalty' ] = st.session_state.get(
						'text_presence_penalty', 0.0 )
				
				st.button( label='Reset', key='text_inference_reset', width='stretch',
					on_click=reset_text_inference_settings )
			
			with st.expander( label='Tools / Grounding Settings', icon='🔎',
					expanded=False, width='stretch' ):
				tool_c1, tool_c2, tool_c3, tool_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Tools ------------
				with tool_c1:
					tool_options = get_text_options( text, 'tool_options' )
					st.multiselect( label='Tools', options=tool_options, key='text_tools',
						help=get_text_help( 'TOOLS' ), placeholder='Options' )
				
				# ---------- Include ------------
				with tool_c2:
					include_options = get_text_options( text, 'include_options' )
					st.multiselect( label='Include', options=include_options,
						key='text_include', help=get_text_help( 'INCLUDE' ),
						placeholder='Options' )
				
				# ---------- Tool Choice ------------
				with tool_c3:
					choice_options = get_text_options( text, 'choice_options' )
					st.selectbox( label='Tool Choice', options=choice_options,
						key='text_tool_choice', help=get_text_help( 'CHOICE' ),
						index=None, placeholder='Options' )
				
				# ---------- Max Tool Calls ------------
				with tool_c4:
					st.slider( label='Max Tool Calls', min_value=0, max_value=100,
						step=1, key='text_max_calls',
						help=get_text_help( 'MAX_TOOL_CALLS' ) )
				
				ctx_c1, ctx_c2, ctx_c3, ctx_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Google Grounding ------------
				with ctx_c1:
					if provider_name == 'Gemini':
						st.toggle( label='Google Grounding', key='text_google_grounding',
							help='When enabled, Gemini grounds this Text response using Google Search.' )
					else:
						st.toggle( label='Google Grounding', key='text_google_grounding',
							disabled=True,
							help='Google grounding is available only for Gemini Text mode.' )
				
				# ---------- Parallel Tools ------------
				with ctx_c2:
					st.toggle( label='Parallel Tools', key='text_parallel_tools',
						help=get_text_help( 'PARALLEL_TOOL_CALLS' ) )
					st.session_state[ 'text_parallel_calls' ] = st.session_state.get(
						'text_parallel_tools', False )
				
				# ---------- Max URLs ------------
				with ctx_c3:
					st.slider( label='Max URLs', min_value=0, max_value=25, step=1,
						key='text_max_urls',
						help='Optional. Maximum number of URLs from the URL list to include.' )
				
				# ---------- Input Mode ------------
				with ctx_c4:
					st.selectbox( label='Input Mode',
						options=[ 'conversation', 'single_turn' ],
						key='text_input',
						help='Conversation uses prior Text messages as context; single_turn omits them.',
						index=None, placeholder='Options' )
				
				# ---------- URLs ------------
				st.text_input( label='URLs', key='text_urls_input',
					help='Optional. Enter URLs separated by semicolons for added prompt context.',
					width='stretch',
					placeholder='https://example.com/page-1;https://example.com/page-2' )
				
				# ---------- Allowed Domains ------------
				st.text_input( label='Allowed Domains', key='text_domains_input',
					help=get_text_help( 'ALLOWED_DOMAINS' ),
					width='stretch',
					placeholder='example.com,openai.com' )
				
				# ---------- Vector Store IDs ------------
				if provider_name == 'GPT':
					st.text_input( label='Vector Store IDs', key='text_vector_store_ids',
						help='Optional. Enter OpenAI vector store IDs separated by commas.',
						width='stretch',
						placeholder='vs_abc123,vs_def456' )
				
				st.button( label='Reset', key='reset_text_tools', width='stretch',
					on_click=reset_text_tool_settings )
			
			with st.expander( label='Output / Response Settings', icon='↔️',
					expanded=False, width='stretch' ):
				resp_c1, resp_c2, resp_c3, resp_c4, resp_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Max Tokens ------------
				with resp_c1:
					st.slider( label='Max Tokens', min_value=0, max_value=100000,
						step=500, help=get_text_help( 'MAX_OUTPUT_TOKENS' ),
						key='text_max_tokens' )
				
				# ---------- Response Format ------------
				with resp_c2:
					format_options = get_text_options( text, 'format_options' )
					st.selectbox( label='Response Format', options=format_options,
						key='text_response_format',
						help='Optional. Desired response format or MIME type.',
						index=None, placeholder='Options' )
				
				# ---------- Store ------------
				with resp_c3:
					st.toggle( label='Store', key='text_store',
						help=get_text_help( 'STORE' ) )
				
				# ---------- Stream ------------
				with resp_c4:
					st.toggle( label='Stream', key='text_stream',
						help=get_text_help( 'STREAM' ) )
				
				# ---------- Background ------------
				with resp_c5:
					st.toggle( label='Background', key='text_background',
						help=get_text_help( 'BACKGROUND_MODE' ) )
				
				schema_c1, schema_c2, schema_c3 = st.columns(
					[ 0.25, 0.50, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Schema Name ------------
				with schema_c1:
					st.text_input( label='Schema Name', key='text_json_schema_name',
						help='Optional. Name used for GPT JSON schema response format.',
						width='stretch', placeholder='response_schema' )
				
				# ---------- Response Schema ------------
				with schema_c2:
					st.text_area( label='Response Schema', key='text_json_schema',
						help='Optional. JSON schema used when Response Format is json_schema.',
						height=100, width='stretch',
						placeholder='{"type":"object","properties":{"answer":{"type":"string"}}}' )
					st.session_state[ 'text_response_schema' ] = st.session_state.get(
						'text_json_schema', '' )
				
				# ---------- Strict Schema ------------
				with schema_c3:
					st.toggle( label='Strict Schema', key='text_json_schema_strict',
						help='Optional. Enforce strict JSON schema when supported.' )
				
				# ---------- Stop Sequences ------------
				st.text_input( label='Stop Sequences', key='text_stops_input',
					help=get_text_help( 'STOP_SEQUENCE' ), width='stretch',
					placeholder='END,STOP,DONE' )
				
				st.button( label='Reset', key='text_response_reset', width='stretch',
					on_click=reset_text_response_settings )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			with in_left:
				st.text_area( label='Enter Text', height=120, width='stretch',
					help=get_text_help( 'SYSTEM_INSTRUCTIONS' ),
					key='text_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=load_text_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_text_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_text_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		if not isinstance( st.session_state.get( 'text_messages' ), list ):
			st.session_state[ 'text_messages' ] = [ ]
		
		for msg in st.session_state.get( 'text_messages', [ ] ):
			role = msg.get( 'role', 'assistant' )
			content = msg.get( 'content', '' )
			with st.chat_message( role, avatar=get_text_avatar( role ) ):
				st.markdown( content )
		
		if provider_name == 'GPT':
			prompt = st.chat_input( 'Ask ChatGPT…' )
		elif provider_name == 'Gemini':
			prompt = st.chat_input( 'Ask Gemini…' )
		elif provider_name == 'Grok':
			prompt = st.chat_input( 'Ask Grok…' )
		else:
			prompt = st.chat_input( 'Ask Boo…' )
		
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			st.session_state.text_messages.append(
				{
						'role': 'user',
						'content': prompt,
				}
			)
			
			with st.chat_message( 'assistant', avatar=get_text_avatar( 'assistant' ) ):
				with st.spinner( 'Thinking…' ):
					response = None
					response_obj = None
					stream_buffer: List[ str ] = [ ]
					stream_placeholder = st.empty( )
					
					def on_stream_chunk( chunk: str ) -> None:
						"""
							
							Purpose:
							--------
							Render streamed Text mode output chunks.
						
							Parameters:
							-----------
							chunk (str): Streamed text chunk.
						
							Returns:
							--------
							None
							
						"""
						if chunk is None:
							return
						
						stream_buffer.append( str( chunk ) )
						stream_placeholder.markdown( ''.join( stream_buffer ) + '▌' )
					
					try:
						response = call_generate_text(
							prompt=prompt,
							stream_handler=on_stream_chunk if st.session_state.get(
								'text_stream', False ) else None
						)
						response_obj = getattr( text, 'response', None ) or response
						
						if provider_name == 'GPT':
							st.session_state[ 'text_previous_response_id' ] = (
									getattr( text, 'previous_id', None ) or
									st.session_state.get( 'text_previous_response_id', '' ) or ''
							)
					except Exception as exc:
						err = Error( exc )
						st.error( f'Generation Failed: {err.info}' )
						response = None
						response_obj = getattr( text, 'response', None )
					
					if response is not None and str( response ).strip( ):
						response_text = str( response ).strip( )
						
						if st.session_state.get( 'text_stream', False ):
							stream_placeholder.markdown( response_text )
						else:
							st.markdown( response_text )
						
						st.session_state.text_messages.append(
							{
									'role': 'assistant',
									'content': response_text,
							}
						)
						
						st.session_state[ 'text_context' ] = build_text_context(
							include_last_message=True )
						st.session_state[ 'last_answer' ] = response_text
						
						sources = extract_text_sources( text, response_obj )
						st.session_state[ 'last_sources' ] = sources
						
						if sources:
							with st.expander( label='Sources', icon='🔎', expanded=False,
									width='stretch' ):
								for source in sources:
									title = source.get( 'title', '' ) or source.get( 'url', '' )
									url = source.get( 'url', '' )
									if url:
										st.markdown( f'- [{title}]({url})' )
									elif title:
										st.markdown( f'- {title}' )
					else:
						st.error( 'Generation Failed!.' )
					
					update_text_usage( response_obj )
		
		# ------------------------------------------------------------------
		# Message Reset
		# ------------------------------------------------------------------
		if st.button( 'Clear Messages', key='text_clear_messages' ):
			st.session_state[ 'text_messages' ] = [ ]
			st.session_state[ 'text_context' ] = [ ]
			st.session_state[ 'text_previous_response_id' ] = ''
			st.session_state[ 'text_conversation_id' ] = ''
			st.session_state[ 'last_answer' ] = ''
			st.session_state[ 'last_sources' ] = [ ]
			st.rerun( )

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == 'Images':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	image = get_images_module( provider_name )
	
	# ------------------------------------------------------------------
	# Images Mode Helpers
	# ------------------------------------------------------------------
	def get_image_options( instance: Any, attr_name: str,
			fallback: Optional[ List[ str ] ] = None ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return list-like option values from an Images wrapper property.
		
			Parameters:
			-----------
			instance (Any): Provider Images wrapper instance.
			attr_name (str): Property or attribute name to inspect.
			fallback (Optional[List[str]]): Fallback option values.
		
			Returns:
			--------
			List[str]: Option values safe for Streamlit controls.
			
		"""
		values = getattr( instance, attr_name, None )
		if callable( values ):
			try:
				values = values( )
			except Exception:
				values = None
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if isinstance( values, list ):
			return [ str( value ) for value in values if str( value ).strip( ) ]
		
		return fallback or [ ]
	
	def get_image_help( name: str, fallback: str = '' ) -> str:
		"""
			
			Purpose:
			--------
			Return image-mode help text from config.py without failing when absent.
		
			Parameters:
			-----------
			name (str): Config attribute name.
			fallback (str): Fallback help text.
		
			Returns:
			--------
			str: Help text value.
			
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_provider_image_models( selected_mode: Optional[ str ] ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return provider-specific image models for the selected image workflow.
		
			Parameters:
			-----------
			selected_mode (Optional[str]): Image workflow name.
		
			Returns:
			--------
			List[str]: Model names for the selected provider and image workflow.
			
		"""
		mode_name = selected_mode or ''
		
		if provider_name == 'Gemini':
			if mode_name == 'Generation':
				return list( getattr( cfg, 'GEMINI_GENERATION', [ ] ) )
			if mode_name == 'Analysis':
				return list( getattr( cfg, 'GEMINI_ANALYSIS', [ ] ) )
			if mode_name == 'Editing':
				return list( getattr( cfg, 'GEMINI_EDITING', [ ] ) )
		
		if provider_name in [ 'GPT', 'Grok' ]:
			if mode_name == 'Generation':
				return list( getattr( cfg, 'GPT_GENERATION', [ ] ) )
			if mode_name == 'Analysis':
				return list( getattr( cfg, 'GPT_ANALYSIS', [ ] ) )
			if mode_name == 'Editing':
				return list( getattr( cfg, 'GPT_EDITING', [ ] ) )
		
		models = get_image_options( image, 'model_options' )
		if not models:
			model_value = getattr( image, 'model', '' )
			models = [ model_value ] if model_value else [ ]
		
		return models
	
	def call_existing_image_method( instance: Any, method_names: List[ str ],
			kwargs: Dict[ str, Any ] ) -> Any:
		"""
			
			Purpose:
			--------
			Call the first available Images wrapper method from a provider-neutral method list.
		
			Parameters:
			-----------
			instance (Any): Provider Images wrapper instance.
			method_names (List[str]): Ordered method names to try.
			kwargs (Dict[str, Any]): Keyword arguments for the method call.
		
			Returns:
			--------
			Any: Provider method result.
			
		"""
		for method_name in method_names:
			method = getattr( instance, method_name, None )
			if callable( method ):
				try:
					return method( **kwargs )
				except TypeError:
					clean_kwargs = {
							key: value
							for key, value in kwargs.items( )
							if value is not None and value != '' and value != [ ]
					}
					return method( **clean_kwargs )
		
		raise AttributeError( f'Provider "{provider_name}" does not expose any image method from: '
			f'{", ".join( method_names )}.' )
	
	def save_uploaded_image( uploaded_file: Any ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Save an uploaded Streamlit image to a temporary file path for provider wrappers that
			expect a local path.
		
			Parameters:
			-----------
			uploaded_file (Any): Streamlit uploaded file object.
		
			Returns:
			--------
			Optional[str]: Temporary file path or None.
			
		"""
		if uploaded_file is None:
			return None
		
		if 'save_temp' in globals( ):
			try:
				return save_temp( uploaded_file )
			except Exception:
				pass
		
		try:
			import tempfile
			from pathlib import Path
			
			suffix = Path( uploaded_file.name ).suffix or '.png'
			with tempfile.NamedTemporaryFile( delete=False, suffix=suffix ) as tmp:
				tmp.write( uploaded_file.getvalue( ) )
				return tmp.name
		except Exception:
			return None
	
	def append_image_message( role: str, content: str ) -> None:
		"""
			
			Purpose:
			--------
			Append an Images-mode message without clearing other mode state.
		
			Parameters:
			-----------
			role (str): Message role.
			content (str): Message content.
		
			Returns:
			--------
			None
			
		"""
		if not isinstance( st.session_state.get( 'image_input' ), list ):
			st.session_state[ 'image_input' ] = [ ]
		
		st.session_state[ 'image_input' ].append(
			{
					'role': role,
					'content': content,
			}
		)
	
	def render_image_messages( ) -> None:
		"""
			
			Purpose:
			--------
			Render Images-mode message history.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		if not isinstance( st.session_state.get( 'image_input' ), list ):
			st.session_state[ 'image_input' ] = [ ]
		
		for msg in st.session_state.get( 'image_input', [ ] ):
			if isinstance( msg, dict ):
				with st.chat_message( msg.get( 'role', 'assistant' ), avatar='' ):
					st.markdown( msg.get( 'content', '' ) )
	
	def clear_image_messages( ) -> None:
		"""
			
			Purpose:
			--------
			Clear only Images-mode message/result state.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'image_input' ] = [ ]
		st.session_state[ 'generated_images' ] = [ ]
		st.session_state[ 'analyzed_images' ] = [ ]
		st.session_state[ 'edited_images' ] = [ ]
	
	def clear_image_instructions( ) -> None:
		"""
			
			Purpose:
			--------
			Clear Images-mode system instructions and selected prompt template.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'image_system_instructions' ] = ''
		st.session_state[ 'instructions' ] = ''
	
	def convert_image_system_instructions( ) -> None:
		"""
			
			Purpose:
			--------
			Convert Images-mode system instructions between XML-like blocks and Markdown
			headings.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		text_value = st.session_state.get( 'image_system_instructions', '' )
		if not isinstance( text_value, str ) or not text_value.strip( ):
			return
		
		source = text_value.strip( )
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'image_system_instructions' ] = converted
	
	def load_image_instruction_template( ) -> None:
		"""
			
			Purpose:
			--------
			Load the selected prompt template into Images-mode system instructions.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		name = st.session_state.get( 'instructions' )
		if name and name != 'No Templates Found':
			prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
			if prompt_text is not None:
				st.session_state[ 'image_system_instructions' ] = prompt_text
	
	def reset_image_model_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Images-mode model controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'image_mode',
				'image_model',
				'image_analysis_model',
				'image_generation_model',
				'image_editing_model',
				'image_number',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_image_inference_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Images-mode inference controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'image_temperature',
				'image_top_percent',
				'image_top_k',
				'image_frequency_penalty',
				'image_presence_penalty',
				'image_presense_penalty',
				'image_max_tokens',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_image_tool_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Images-mode tool, include, grounding, and domain controls.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'image_tools',
				'image_include',
				'image_tool_choice',
				'image_domains_input',
				'image_domains',
				'image_grounded',
				'image_image_search',
				'image_parallel_tools',
				'image_parallel_calls',
				'image_max_calls',
				'image_max_searches',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_image_visual_settings( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Images-mode visual output controls.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'image_resolution',
				'image_media_resolution',
				'image_mime_type',
				'image_output',
				'image_size',
				'image_quality',
				'image_style',
				'image_backcolor',
				'image_aspect_ratio',
				'image_detail',
				'image_analysis_detail',
				'image_compression',
				'image_modality',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def parse_image_domains( value: Any ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Parse comma-delimited allowed image domains.
		
			Parameters:
			-----------
			value (Any): Text input value.
		
			Returns:
			--------
			List[str]: Parsed domain values.
			
		"""
		raw = str( value or '' )
		return [ item.strip( ) for item in raw.split( ',' ) if item.strip( ) ]
	
	def render_image_output( result: Any, caption: str = 'Image output' ) -> bool:
		"""
			
			Purpose:
			--------
			Render an image result returned as bytes, path, URL, list, tuple, dictionary, or
			provider object.
		
			Parameters:
			-----------
			result (Any): Provider image result.
			caption (str): Caption displayed below rendered image output.
		
			Returns:
			--------
			bool: True if at least one image-like item was rendered.
			
		"""
		if result is None:
			return False
		
		if isinstance( result, list ) or isinstance( result, tuple ):
			rendered = False
			for idx, item in enumerate( result, start=1 ):
				rendered = render_image_output( item, f'{caption} {idx}' ) or rendered
			return rendered
		
		if isinstance( result, dict ):
			for key in [ 'image', 'images', 'data', 'bytes', 'content', 'url', 'path' ]:
				if key in result:
					return render_image_output( result.get( key ), caption )
			
			st.json( result )
			return False
		
		try:
			if isinstance( result, bytes ):
				st.image( result, caption=caption, use_container_width=False )
				return True
			
			if isinstance( result, str ):
				value = result.strip( )
				lower_value = value.lower( )
				if lower_value.startswith( 'http://' ) or lower_value.startswith( 'https://' ):
					st.image( value, caption=caption, use_container_width=False )
					return True
				
				if lower_value.endswith( ('.png', '.jpg', '.jpeg', '.webp', '.gif') ):
					st.image( value, caption=caption, use_container_width=False )
					return True
				
				st.markdown( value )
				return False
			
			if hasattr( result, 'read' ):
				st.image( result.read( ), caption=caption, use_container_width=False )
				return True
			
			if hasattr( result, 'content' ):
				return render_image_output( getattr( result, 'content' ), caption )
			
			if hasattr( result, 'url' ):
				return render_image_output( getattr( result, 'url' ), caption )
			
			if hasattr( result, 'path' ):
				return render_image_output( getattr( result, 'path' ), caption )
		except Exception:
			return False
		
		st.write( result )
		return False
	
	def update_image_usage( response: Any ) -> None:
		"""
			
			Purpose:
			--------
			Update token counters using whichever counter helper exists in the current Boo file.
		
			Parameters:
			-----------
			response (Any): Provider response object.
		
			Returns:
			--------
			None
			
		"""
		try:
			if 'update_token_counters' in globals( ):
				update_token_counters( response )
			elif '_update_token_counters' in globals( ):
				_update_token_counters( response )
			elif 'update_counters' in globals( ):
				update_counters( response )
		except Exception:
			pass
	
	def get_image_common_kwargs( prompt: str ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build provider-neutral keyword arguments for Images wrapper calls.
		
			Parameters:
			-----------
			prompt (str): User prompt.
		
			Returns:
			--------
			Dict[str, Any]: Common Images keyword arguments.
			
		"""
		domains = parse_image_domains( st.session_state.get( 'image_domains_input', '' ) )
		if domains:
			st.session_state[ 'image_domains' ] = domains
		
		return {
				'prompt': prompt,
				'model': st.session_state.get( 'image_model' ),
				'number': st.session_state.get( 'image_number' ),
				'temperature': st.session_state.get( 'image_temperature' ),
				'top_p': st.session_state.get( 'image_top_percent' ),
				'top_k': st.session_state.get( 'image_top_k' ),
				'frequency': st.session_state.get( 'image_frequency_penalty' ),
				'presence': st.session_state.get( 'image_presence_penalty' ),
				'max_tokens': st.session_state.get( 'image_max_tokens' ),
				'instruct': st.session_state.get( 'image_system_instructions', '' ),
				'tools': st.session_state.get( 'image_tools', [ ] ),
				'tool_choice': st.session_state.get( 'image_tool_choice' ) or None,
				'include': st.session_state.get( 'image_include', [ ] ),
				'allowed_domains': st.session_state.get( 'image_domains', [ ] ),
				'store': st.session_state.get( 'image_store' ),
				'stream': st.session_state.get( 'image_stream' ),
				'background': st.session_state.get( 'image_background' ),
				'is_parallel': st.session_state.get( 'image_parallel_tools',
					st.session_state.get( 'image_parallel_calls', False ) ),
				'max_tools': st.session_state.get( 'image_max_calls' ),
				'max_searches': st.session_state.get( 'image_max_searches' ),
		}
	
	def get_image_generation_kwargs( prompt: str ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build keyword arguments for image generation.
		
			Parameters:
			-----------
			prompt (str): Image generation prompt.
		
			Returns:
			--------
			Dict[str, Any]: Image generation keyword arguments.
			
		"""
		kwargs = get_image_common_kwargs( prompt )
		kwargs.update( {
					'size': st.session_state.get( 'image_size' ) or None,
					'quality': st.session_state.get( 'image_quality' ) or None,
					'style': st.session_state.get( 'image_style' ) or None,
					'fmt': st.session_state.get( 'image_mime_type' )
					       or st.session_state.get( 'image_output' )
					       or None,
					'mime_type': st.session_state.get( 'image_mime_type' ) or None,
					'compression': st.session_state.get( 'image_compression' ) or None,
					'background': st.session_state.get( 'image_backcolor' )
					              or st.session_state.get( 'image_background' )
					              or None,
					'aspect_ratio': st.session_state.get( 'image_aspect_ratio' ) or None,
					'response_modalities': st.session_state.get( 'image_modality' ) or None,
					'grounded': st.session_state.get( 'image_grounded', False ),
					'image_search': st.session_state.get( 'image_image_search', False ),
			} )
		
		return kwargs
	
	def get_image_analysis_kwargs( prompt: str, path: str ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build keyword arguments for image analysis.
		
			Parameters:
			-----------
			prompt (str): Image analysis prompt.
			path (str): Uploaded image temporary path.
		
			Returns:
			--------
			Dict[str, Any]: Image analysis keyword arguments.
			
		"""
		kwargs = get_image_common_kwargs( prompt )
		kwargs.update(
			{
					'path': path,
					'image_path': path,
					'detail': st.session_state.get( 'image_analysis_detail' )
					          or st.session_state.get( 'image_detail' )
					          or None,
					'response_modalities': st.session_state.get( 'image_modality' ) or None,
					'grounded': st.session_state.get( 'image_grounded', False ),
					'image_search': st.session_state.get( 'image_image_search', False ),
			}
		)
		
		return kwargs
	
	def get_image_edit_kwargs( prompt: str, path: str, mask_path: Optional[ str ] = None ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build keyword arguments for image editing.
		
			Parameters:
			-----------
			prompt (str): Image editing prompt.
			path (str): Uploaded image temporary path.
			mask_path (Optional[str]): Optional uploaded mask temporary path.
		
			Returns:
			--------
			Dict[str, Any]: Image editing keyword arguments.
			
		"""
		kwargs = get_image_generation_kwargs( prompt )
		kwargs.update( {
					'path': path,
					'image_path': path,
					'mask_path': mask_path,
					'mask': mask_path,
			} )
		
		return kwargs
	
	def run_image_generation( prompt: str ) -> Any:
		"""
			
			Purpose:
			--------
			Dispatch image generation to the selected provider.
		
			Parameters:
			-----------
			prompt (str): Image generation prompt.
		
			Returns:
			--------
			Any: Provider generation result.
			
		"""
		kwargs = get_image_generation_kwargs( prompt )
		return call_existing_image_method(
			instance=image,
			method_names=[ 'generate', 'generate_image', 'create', 'create_image' ],
			kwargs=kwargs
		)
	
	def run_image_analysis( prompt: str, path: str ) -> Any:
		"""
			
			Purpose:
			--------
			Dispatch image analysis to the selected provider.
		
			Parameters:
			-----------
			prompt (str): Image analysis prompt.
			path (str): Uploaded image temporary path.
		
			Returns:
			--------
			Any: Provider analysis result.
			
		"""
		kwargs = get_image_analysis_kwargs( prompt, path )
		return call_existing_image_method( instance=image,
			method_names=[ 'analyze', 'analyze_image', 'vision', 'describe' ], kwargs=kwargs )
	
	def run_image_editing( prompt: str, path: str, mask_path: Optional[ str ] = None ) -> Any:
		"""
			
			Purpose:
			--------
			Dispatch image editing to the selected provider.
		
			Parameters:
			-----------
			prompt (str): Image editing prompt.
			path (str): Uploaded image temporary path.
			mask_path (Optional[str]): Optional uploaded mask temporary path.
		
			Returns:
			--------
			Any: Provider editing result.
			
		"""
		kwargs = get_image_edit_kwargs( prompt, path, mask_path )
		return call_existing_image_method( instance=image,
			method_names=[ 'edit', 'edit_image', 'modify', 'generate_edit' ], kwargs=kwargs )
	
	# ------------------------------------------------------------------
	# Session Safety
	# ------------------------------------------------------------------
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'image_system_instructions' ] = ''
		st.session_state[ 'clear_image_instructions' ] = False
		st.session_state[ 'clear_instructions' ] = False
	
	if not isinstance( st.session_state.get( 'image_input' ), list ):
		st.session_state[ 'image_input' ] = [ ]
	
	if not isinstance( st.session_state.get( 'image_number' ), int ):
		st.session_state[ 'image_number' ] = 1
	
	if int( st.session_state.get( 'image_number', 1 ) or 1 ) < 1:
		st.session_state[ 'image_number' ] = 1
	
	if not isinstance( st.session_state.get( 'image_include' ), list ):
		st.session_state[ 'image_include' ] = [ ]
	
	if not isinstance( st.session_state.get( 'image_tools' ), list ):
		st.session_state[ 'image_tools' ] = [ ]
	
	if 'image_analysis_detail' not in st.session_state:
		st.session_state[ 'image_analysis_detail' ] = 'auto'
	
	if 'image_compression' not in st.session_state:
		st.session_state[ 'image_compression' ] = 0.0
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '📷 Images API', help=get_image_help( 'IMAGES_API' ) )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				llm_c1, llm_c2, llm_c3, llm_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Image Mode ------------
				with llm_c1:
					st.selectbox( label='Image Mode',
						options=[ 'Generation', 'Analysis', 'Editing' ],
						key='image_mode', help='Available provider image workflows.',
						index=None, placeholder='Options' )
					image_mode = st.session_state.get( 'image_mode', '' )
				
				# ---------- Model ------------
				with llm_c2:
					st.selectbox( label='Select Model',
						options=get_provider_image_models( image_mode ),
						key='image_model',
						help='Required. Model used by the selected image workflow.',
						index=None, placeholder='Options' )
				
				# ---------- Analysis Model ------------
				with llm_c3:
					analysis_models = get_provider_image_models( 'Analysis' )
					st.selectbox( label='Analysis Model',
						options=analysis_models,
						key='image_analysis_model',
						help='Optional. Separate model used for image analysis when supported.',
						index=None, placeholder='Options' )
				
				# ---------- Number ------------
				with llm_c4:
					st.slider( label='Number', min_value=1, max_value=10, step=1,
						key='image_number',
						help='Number of images or candidates requested when supported.' )
				
				st.button( label='Reset', key='image_model_reset', width='stretch',
					on_click=reset_image_model_settings )
			
			with st.expander( label='Inference Settings', icon='🎚️', expanded=False,
					width='stretch' ):
				prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Top-P ------------
				with prm_c1:
					st.slider( label='Top-P', key='image_top_percent',
						min_value=0.0, max_value=1.0, step=0.01,
						help=get_image_help( 'TOP_P' ) )
				
				# ---------- Top-K ------------
				with prm_c2:
					st.slider( label='Top-K', key='image_top_k',
						min_value=0, max_value=200, step=1,
						help=get_image_help( 'TOP_K' ) )
				
				# ---------- Temperature ------------
				with prm_c3:
					st.slider( label='Temperature', key='image_temperature',
						min_value=0.0, max_value=2.0, step=0.01,
						help=get_image_help( 'TEMPERATURE' ) )
				
				# ---------- Frequency Penalty ------------
				with prm_c4:
					st.slider( label='Frequency Penalty', key='image_frequency_penalty',
						min_value=-2.0, max_value=2.0, step=0.01,
						help=get_image_help( 'FREQUENCY_PENALTY' ) )
				
				# ---------- Presence Penalty ------------
				with prm_c5:
					st.slider( label='Presence Penalty', key='image_presence_penalty',
						min_value=-2.0, max_value=2.0, step=0.01,
						help=get_image_help( 'PRESENCE_PENALTY' ) )
					st.session_state[ 'image_presense_penalty' ] = st.session_state.get(
						'image_presence_penalty', 0.0 )
				
				resp_c1, resp_c2 = st.columns( [ 0.50, 0.50 ], border=True, gap='xxsmall' )
				
				# ---------- Max Tokens ------------
				with resp_c1:
					st.slider( label='Max Tokens', key='image_max_tokens',
						min_value=0, max_value=100000, step=500,
						help=get_image_help( 'MAX_OUTPUT_TOKENS' ) )
				
				# ---------- Compression ------------
				with resp_c2:
					st.slider( label='Compression', key='image_compression',
						min_value=0.0, max_value=1.0, step=0.01,
						help='Optional. Image compression when supported by the provider.' )
				
				st.button( label='Reset', key='image_inference_reset', width='stretch',
					on_click=reset_image_inference_settings )
			
			with st.expander( label='Tools / Grounding Settings', icon='🔎',
					expanded=False, width='stretch' ):
				tool_c1, tool_c2, tool_c3, tool_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Tools ------------
				with tool_c1:
					tool_options = get_image_options( image, 'tool_options' )
					st.multiselect( label='Tools', options=tool_options,
						key='image_tools', help=get_image_help( 'TOOLS' ),
						placeholder='Options' )
				
				# ---------- Include ------------
				with tool_c2:
					include_options = get_image_options( image, 'include_options' )
					st.multiselect( label='Include', options=include_options,
						key='image_include', help=get_image_help( 'INCLUDE' ),
						placeholder='Options' )
				
				# ---------- Tool Choice ------------
				with tool_c3:
					choice_options = get_image_options( image, 'choice_options' )
					st.selectbox( label='Tool Choice', options=choice_options,
						key='image_tool_choice', help=get_image_help( 'CHOICE' ),
						index=None, placeholder='Options' )
				
				# ---------- Max Calls ------------
				with tool_c4:
					st.slider( label='Max Calls', min_value=0, max_value=100,
						step=1, key='image_max_calls',
						help=get_image_help( 'MAX_TOOL_CALLS' ) )
				
				ctx_c1, ctx_c2, ctx_c3, ctx_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Gemini Grounding ------------
				with ctx_c1:
					grounding_supported = provider_name == 'Gemini'
					if hasattr( image, 'supports_search_grounding' ):
						try:
							grounding_supported = bool(
								image.supports_search_grounding(
									st.session_state.get( 'image_model', '' ) ) )
						except Exception:
							grounding_supported = provider_name == 'Gemini'
					
					st.toggle( label='Google Grounding', key='image_grounded',
						disabled=not grounding_supported,
						help='Ground image response through Google Search when supported.' )
				
				# ---------- Gemini Image Search ------------
				with ctx_c2:
					image_search_supported = provider_name == 'Gemini'
					if hasattr( image, 'supports_image_search' ):
						try:
							image_search_supported = bool(
								image.supports_image_search(
									st.session_state.get( 'image_model', '' ) ) )
						except Exception:
							image_search_supported = provider_name == 'Gemini'
					
					st.toggle( label='Image Search', key='image_image_search',
						disabled=not image_search_supported,
						help='Use image search when supported by Gemini image grounding.' )
				
				# ---------- Parallel Tools ------------
				with ctx_c3:
					st.toggle( label='Parallel Tools', key='image_parallel_tools',
						help=get_image_help( 'PARALLEL_TOOL_CALLS' ) )
					st.session_state[ 'image_parallel_calls' ] = st.session_state.get(
						'image_parallel_tools', False )
				
				# ---------- Max Searches ------------
				with ctx_c4:
					st.slider( label='Max Searches', min_value=0, max_value=100,
						step=1, key='image_max_searches',
						help='Optional. Maximum image/web searches when supported.' )
				
				st.text_input( label='Allowed Domains', key='image_domains_input',
					help=get_image_help( 'ALLOWED_DOMAINS' ),
					width='stretch', placeholder='example.com,openai.com' )
				
				st.button( label='Reset', key='image_tools_reset', width='stretch',
					on_click=reset_image_tool_settings )
			
			with st.expander( label='Visual Settings', icon='👁️', expanded=False,
					width='stretch' ):
				img_c1, img_c2, img_c3, img_c4, img_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Size ------------
				with img_c1:
					size_options = get_image_options( image, 'size_options',
						[ '1024x1024', '1024x1536', '1536x1024' ] )
					st.selectbox( label='Image Size', options=size_options,
						key='image_size', help='Optional. Generated or edited image size.',
						index=None, placeholder='Options' )
				
				# ---------- Quality ------------
				with img_c2:
					quality_options = get_image_options( image, 'quality_options',
						[ 'auto', 'standard', 'hd', 'low', 'medium', 'high' ] )
					st.selectbox( label='Image Quality', options=quality_options,
						key='image_quality', help='Optional. Image quality.',
						index=None, placeholder='Options' )
				
				# ---------- Style ------------
				with img_c3:
					style_options = get_image_options( image, 'style_options' )
					st.selectbox( label='Image Style', options=style_options,
						key='image_style', help='Optional. Image style when supported.',
						index=None, placeholder='Options' )
				
				# ---------- Background ------------
				with img_c4:
					background_options = get_image_options( image, 'backcolor_options',
						[ 'auto', 'transparent', 'opaque' ] )
					st.selectbox( label='Background', options=background_options,
						key='image_backcolor', help=get_image_help( 'IMAGE_BACKGROUND' ),
						index=None, placeholder='Options' )
				
				# ---------- MIME / Output Format ------------
				with img_c5:
					output_options = get_image_options( image, 'mime_options' )
					if not output_options:
						output_options = get_image_options( image, 'output_options',
							[ 'png', 'jpeg', 'webp' ] )
					
					st.selectbox( label='MIME Format', options=output_options,
						key='image_mime_type', help=get_image_help( 'IMAGE_RESPONSE' ),
						index=None, placeholder='Options' )
					st.session_state[ 'image_output' ] = st.session_state.get(
						'image_mime_type', '' )
				
				img2_c1, img2_c2, img2_c3 = st.columns(
					[ 0.34, 0.33, 0.33 ], border=True, gap='xxsmall' )
				
				# ---------- Aspect Ratio ------------
				with img2_c1:
					aspect_options = get_image_options( image, 'aspect_options',
						[ '1:1', '3:4', '4:3', '9:16', '16:9' ] )
					st.selectbox( label='Aspect Ratio', options=aspect_options,
						key='image_aspect_ratio',
						help='Optional. Output aspect ratio when supported.',
						index=None, placeholder='Options' )
				
				# ---------- Detail ------------
				with img2_c2:
					detail_options = get_image_options( image, 'detail_options',
						[ 'auto', 'low', 'high' ] )
					st.selectbox( label='Analysis Detail', options=detail_options,
						key='image_analysis_detail',
						help='Optional. Image analysis detail level.',
						index=None, placeholder='Options' )
					st.session_state[ 'image_detail' ] = st.session_state.get(
						'image_analysis_detail', '' )
				
				# ---------- Response Mode ------------
				with img2_c3:
					if st.session_state.get( 'image_mode' ) == 'Analysis':
						modality_options = [ 'TEXT' ]
					else:
						modality_options = [ 'IMAGE', 'TEXT_AND_IMAGE' ]
					
					st.selectbox( label='Response Mode', options=modality_options,
						key='image_modality',
						help='Provider response modality for image workflows.',
						index=None, placeholder='Options' )
				
				st.button( label='Reset', key='image_visual_reset', width='stretch',
					on_click=reset_image_visual_settings )
		
		# ------------------------------------------------------------------
		# Expander — Image System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			with in_left:
				st.text_area( label='Enter Text', height=80, width='stretch',
					help=get_image_help( 'SYSTEM_INSTRUCTIONS' ),
					key='image_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names,
					key='instructions', on_change=load_image_instruction_template,
					index=None )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_image_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_image_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Tab Section
		# ------------------------------------------------------------------
		tab_gen, tab_analyze, tab_edit = st.tabs( [ 'Generate', 'Analyze', 'Edit' ] )
		
		with tab_gen:
			render_image_messages( )
			generation_prompt = st.text_area( label='Image Generation Prompt',
				key='image_generate_prompt', height=120, width='stretch',
				placeholder='Describe the image to generate.' )
			
			gen_c1, gen_c2 = st.columns( [ 0.50, 0.50 ] )
			
			with gen_c1:
				if st.button( 'Generate Image', key='generate_image', width='stretch' ):
					with st.spinner( 'Generating…' ):
						try:
							if not isinstance( generation_prompt,
									str ) or not generation_prompt.strip( ):
								st.warning( 'Enter a prompt before generating an image.' )
							elif not st.session_state.get( 'image_model' ):
								st.warning( 'Select a model before generating an image.' )
							else:
								append_image_message( 'user', generation_prompt.strip( ) )
								result = run_image_generation( generation_prompt.strip( ) )
								
								if result is None:
									st.warning( 'No image output was returned.' )
								else:
									st.session_state[ 'generated_images' ].append( result )
									rendered = render_image_output( result, 'Generated image' )
									
									if rendered:
										append_image_message(
											'assistant',
											'Generated image returned successfully.' )
									else:
										append_image_message( 'assistant', str( result ) )
								
								update_image_usage( getattr( image, 'response', None ) )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Image generation failed: {err.info}' )
			
			with gen_c2:
				if st.button( 'Clear Messages', key='clear_image_generation',
						width='stretch', on_click=clear_image_messages ):
					st.rerun( )
		
		with tab_analyze:
			uploaded_img = st.file_uploader( 'Upload an image for analysis',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ], accept_multiple_files=False,
				key='images_analyze_uploader' )
			
			analysis_path = None
			if uploaded_img:
				analysis_path = save_uploaded_image( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview', width=250 )
			
			render_image_messages( )
			analysis_prompt = st.text_area( label='Image Analysis Prompt',
				key='image_analysis_prompt', height=120, width='stretch',
				placeholder='Ask a question about the uploaded image.' )
			
			ana_c1, ana_c2 = st.columns( [ 0.50, 0.50 ] )
			
			with ana_c1:
				if st.button( 'Analyze Image', key='analyze_image', width='stretch' ):
					with st.spinner( 'Analyzing image…' ):
						try:
							if not analysis_path:
								st.warning( 'Upload an image before analyzing.' )
							elif not isinstance( analysis_prompt,
									str ) or not analysis_prompt.strip( ):
								st.warning( 'Enter an analysis prompt before analyzing the image.' )
							elif not st.session_state.get(
									'image_model' ) and not st.session_state.get(
									'image_analysis_model' ):
								st.warning( 'Select a model before analyzing an image.' )
							else:
								if st.session_state.get( 'image_analysis_model' ):
									st.session_state[ 'image_model' ] = st.session_state.get(
										'image_analysis_model' )
								
								append_image_message( 'user', analysis_prompt.strip( ) )
								result = run_image_analysis( analysis_prompt.strip( ),
									analysis_path )
								
								if result is None:
									st.warning( 'No analysis output returned by the model.' )
								else:
									st.session_state[ 'analyzed_images' ].append( result )
									st.markdown( '**Analysis result:**' )
									st.write( result )
									append_image_message( 'assistant', str( result ) )
								
								update_image_usage( getattr( image, 'response', None ) )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Analysis Failed: {err.info}' )
			
			with ana_c2:
				if st.button( 'Clear Messages', key='clear_image_analysis',
						width='stretch', on_click=clear_image_messages ):
					st.rerun( )
		
		with tab_edit:
			uploaded_img = st.file_uploader( 'Upload Image for Edit',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ], accept_multiple_files=False,
				key='images_edit_uploader' )
			
			uploaded_mask = st.file_uploader( 'Upload Optional Mask',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ], accept_multiple_files=False,
				key='images_edit_mask_uploader' )
			
			edit_path = None
			mask_path = None
			
			if uploaded_img:
				edit_path = save_uploaded_image( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview', width=250 )
			
			if uploaded_mask:
				mask_path = save_uploaded_image( uploaded_mask )
				st.image( uploaded_mask, caption='Uploaded mask preview', width=250 )
			
			render_image_messages( )
			edit_prompt = st.text_area( label='Image Editing Prompt',
				key='image_edit_prompt', height=120, width='stretch',
				placeholder='Describe how the uploaded image should be edited.' )
			
			edit_c1, edit_c2 = st.columns( [ 0.50, 0.50 ] )
			
			with edit_c1:
				if st.button( 'Edit Image', key='edit_image', width='stretch' ):
					with st.spinner( 'Editing image…' ):
						try:
							if not edit_path:
								st.warning( 'Upload an image before editing.' )
							elif not isinstance( edit_prompt, str ) or not edit_prompt.strip( ):
								st.warning( 'Enter an edit prompt before editing the image.' )
							elif not st.session_state.get( 'image_model' ):
								st.warning( 'Select a model before editing an image.' )
							else:
								append_image_message( 'user', edit_prompt.strip( ) )
								result = run_image_editing( edit_prompt.strip( ), edit_path,
									mask_path )
								
								if result is None:
									st.warning( 'No edited image output was returned.' )
								else:
									st.session_state[ 'edited_images' ].append( result )
									rendered = render_image_output( result, 'Edited image' )
									
									if rendered:
										append_image_message(
											'assistant',
											'Edited image returned successfully.' )
									else:
										append_image_message( 'assistant', str( result ) )
								
								update_image_usage( getattr( image, 'response', None ) )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Image edit failed: {err.info}' )
			
			with edit_c2:
				if st.button( 'Clear Messages', key='clear_image_edit',
						width='stretch', on_click=clear_image_messages ):
					st.rerun( )

# ======================================================================================
# AUDIO MODE
# ======================================================================================
elif mode == 'Audio':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	transcriber = get_transcription_module( provider_name )
	translator = get_translation_module( provider_name )
	tts = get_tts_module( provider_name )
	
	# ------------------------------------------------------------------
	# Audio Mode Helpers
	# ------------------------------------------------------------------
	def get_audio_help( name: str, fallback: str = '' ) -> str:
		"""
			
			Purpose:
			--------
			Return Audio mode help text from config.py without failing when a constant is
			absent.
		
			Parameters:
			-----------
			name (str): Config attribute name.
			fallback (str): Fallback help text.
		
			Returns:
			--------
			str: Help text value.
			
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_audio_options( instance: Any, attr_name: str,
			fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""
			
			Purpose:
			--------
			Return list-like option values from an Audio wrapper property.
		
			Parameters:
			-----------
			instance (Any): Audio wrapper instance.
			attr_name (str): Property or attribute name to inspect.
			fallback (Optional[List[Any]]): Fallback option values.
		
			Returns:
			--------
			List[Any]: Option values safe for Streamlit controls.
			
		"""
		values = getattr( instance, attr_name, None )
		if callable( values ):
			try:
				values = values( )
			except Exception:
				values = None
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if isinstance( values, list ):
			return values
		
		return fallback or [ ]
	
	def get_audio_task_options( ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return provider-supported Audio mode task options.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]: Audio task labels.
			
		"""
		tasks = [ ]
		
		if transcriber is not None:
			tasks.append( 'Transcribe' )
		
		if translator is not None:
			tasks.append( 'Translate' )
		
		if tts is not None:
			tasks.append( 'Text-to-Speech' )
		
		return tasks
	
	def get_audio_model_options( task: Optional[ str ] ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return task-aware Audio model options.
		
			Parameters:
			-----------
			task (Optional[str]): Selected Audio task.
		
			Returns:
			--------
			List[str]: Model option labels.
			
		"""
		if task == 'Transcribe':
			options = get_audio_options( transcriber, 'model_options' )
		elif task == 'Translate':
			options = get_audio_options( translator, 'model_options' )
		elif task == 'Text-to-Speech':
			options = get_audio_options( tts, 'model_options' )
		else:
			options = [ ]
		
		return [ str( item ) for item in options if str( item ).strip( ) ]
	
	def get_audio_language_options( task: Optional[ str ] ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return task-aware Audio language options.
		
			Parameters:
			-----------
			task (Optional[str]): Selected Audio task.
		
			Returns:
			--------
			List[str]: Language option labels.
			
		"""
		if task == 'Transcribe':
			options = get_audio_options( transcriber, 'language_options' )
		elif task == 'Translate':
			options = get_audio_options( translator, 'language_options' )
		else:
			options = [ ]
		
		return [ str( item ) for item in options if str( item ).strip( ) ]
	
	def get_audio_voice_options( ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return text-to-speech voice options.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]: Voice option labels.
			
		"""
		options = get_audio_options( tts, 'voice_options' )
		return [ str( item ) for item in options if str( item ).strip( ) ]
	
	def get_audio_rate_options( ) -> List[ Any ]:
		"""
			
			Purpose:
			--------
			Return sample-rate options for audio playback and recording.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[Any]: Sample-rate option values.
			
		"""
		options = getattr( cfg, 'SAMPLE_RATES', [ 8000, 11025, 16000, 22050, 24000,
		                                          32000, 44100, 48000 ] )
		
		if isinstance( options, tuple ):
			options = list( options )
		
		if not isinstance( options, list ):
			return [ 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000 ]
		
		return options
	
	def get_audio_speed_options( ) -> List[ float ]:
		"""
			
			Purpose:
			--------
			Return text-to-speech speed options.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[float]: Speech speed values.
			
		"""
		options = get_audio_options( tts, 'speed_options' )
		if options:
			return [ float( item ) for item in options ]
		
		return [
				0.25,
				0.50,
				0.75,
				1.0,
				1.25,
				1.50,
				2.0,
				3.0,
				4.0,
		]
	
	def get_audio_format_options( task: Optional[ str ], model: Optional[ str ] ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return task-aware audio response/output format options.
		
			Parameters:
			-----------
			task (Optional[str]): Selected Audio task.
			model (Optional[str]): Selected Audio model.
		
			Returns:
			--------
			List[str]: Format option labels.
			
		"""
		if task == 'Transcribe':
			format_map = getattr( transcriber, 'response_format_options', None )
			if isinstance( format_map, dict ):
				options = format_map.get( model, [ 'json' ] )
			else:
				options = get_audio_options( transcriber, 'format_options',
					get_audio_options( transcriber, 'response_format_options', [ 'json' ] ) )
		
		elif task == 'Translate':
			options = get_audio_options( translator, 'format_options',
				get_audio_options( translator, 'response_format_options', [ 'json' ] ) )
		
		elif task == 'Text-to-Speech':
			options = get_audio_options( tts, 'format_options',
				get_audio_options( tts, 'response_format_options',
					get_audio_options( tts, 'output_format_options', [ 'mp3', 'wav' ] ) ) )
		
		else:
			options = [ ]
		
		return [ str( item ) for item in options if str( item ).strip( ) ]
	
	def get_audio_include_options( task: Optional[ str ], model: Optional[ str ] ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Return task-aware Audio include options.
		
			Parameters:
			-----------
			task (Optional[str]): Selected Audio task.
			model (Optional[str]): Selected Audio model.
		
			Returns:
			--------
			List[str]: Include option labels.
			
		"""
		if task != 'Transcribe':
			return [ ]
		
		include_map = getattr( transcriber, 'include_options', None )
		if isinstance( include_map, dict ):
			options = include_map.get( model, [ ] )
		else:
			options = get_audio_options( transcriber, 'include_options' )
		
		return [ str( item ) for item in options if str( item ).strip( ) ]
	
	def save_uploaded_audio( uploaded_file: Any ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Save a Streamlit uploaded or recorded audio file to a temporary file path.
		
			Parameters:
			-----------
			uploaded_file (Any): Streamlit uploaded file or audio input object.
		
			Returns:
			--------
			Optional[str]: Temporary file path or None.
			
		"""
		if uploaded_file is None:
			return None
		
		if 'save_temp' in globals( ):
			try:
				return save_temp( uploaded_file )
			except Exception:
				pass
		
		try:
			suffix = '.wav'
			if hasattr( uploaded_file, 'name' ):
				suffix = Path( uploaded_file.name ).suffix or suffix
			
			with tempfile.NamedTemporaryFile( delete=False, suffix=suffix ) as tmp:
				if hasattr( uploaded_file, 'getvalue' ):
					tmp.write( uploaded_file.getvalue( ) )
				elif hasattr( uploaded_file, 'read' ):
					tmp.write( uploaded_file.read( ) )
				else:
					tmp.write( bytes( uploaded_file ) )
				
				return tmp.name
		except Exception:
			return None
	
	def get_audio_prompt_value( task: Optional[ str ] ) -> str:
		"""
			
			Purpose:
			--------
			Return the prompt/instructions value used for transcription or translation.
		
			Parameters:
			-----------
			task (Optional[str]): Selected Audio task.
		
			Returns:
			--------
			str: Prompt text.
			
		"""
		if task == 'Transcribe':
			return st.session_state.get( 'transcription_prompt', '' ) \
				or st.session_state.get( 'audio_system_instructions', '' )
		
		if task == 'Translate':
			return st.session_state.get( 'translation_prompt', '' ) \
				or st.session_state.get( 'audio_system_instructions', '' )
		
		return st.session_state.get( 'audio_system_instructions', '' )
	
	def get_audio_format_value( task: Optional[ str ] ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Return the provider response/output format for the selected Audio task.
		
			Parameters:
			-----------
			task (Optional[str]): Selected Audio task.
		
			Returns:
			--------
			Optional[str]: Audio format value or None.
			
		"""
		value = st.session_state.get( 'audio_response_format', '' ) \
		        or st.session_state.get( 'audio_format', '' )
		
		if not value:
			return None
		
		return str( value )
	
	def call_existing_audio_method( instance: Any, method_names: List[ str ],
			kwargs: Dict[ str, Any ] ) -> Any:
		"""
			
			Purpose:
			--------
			Call the first available provider Audio wrapper method from an ordered list.
		
			Parameters:
			-----------
			instance (Any): Provider Audio wrapper instance.
			method_names (List[str]): Ordered provider method names to try.
			kwargs (Dict[str, Any]): Keyword arguments for the provider method call.
		
			Returns:
			--------
			Any: Provider method result.
			
		"""
		for method_name in method_names:
			method = getattr( instance, method_name, None )
			if callable( method ):
				try:
					return method( **kwargs )
				except TypeError:
					clean_kwargs = {
							key: value
							for key, value in kwargs.items( )
							if value is not None and value != '' and value != [ ]
					}
					return method( **clean_kwargs )
		
		raise AttributeError(
			f'Provider "{provider_name}" does not expose any audio method from: '
			f'{", ".join( method_names )}.' )
	
	def normalize_audio_text_result( result: Any ) -> str:
		"""
			
			Purpose:
			--------
			Normalize provider transcription or translation results to displayable text.
		
			Parameters:
			-----------
			result (Any): Provider result object, dictionary, or text.
		
			Returns:
			--------
			str: Displayable text result.
			
		"""
		if result is None:
			return ''
		
		if isinstance( result, str ):
			return result
		
		if isinstance( result, dict ):
			for key in [ 'text', 'transcript', 'translation', 'content', 'output' ]:
				if key in result and result.get( key ) is not None:
					return str( result.get( key ) )
			
			return str( result )
		
		for attr_name in [ 'text', 'transcript', 'translation', 'content', 'output' ]:
			if hasattr( result, attr_name ):
				value = getattr( result, attr_name )
				if value is not None:
					return str( value )
		
		return str( result )
	
	def normalize_audio_bytes_result( result: Any ) -> Optional[ bytes ]:
		"""
			
			Purpose:
			--------
			Normalize provider text-to-speech results to audio bytes when possible.
		
			Parameters:
			-----------
			result (Any): Provider TTS result.
		
			Returns:
			--------
			Optional[bytes]: Audio bytes or None.
			
		"""
		if result is None:
			return None
		
		if isinstance( result, bytes ):
			return result
		
		if isinstance( result, bytearray ):
			return bytes( result )
		
		if isinstance( result, dict ):
			for key in [ 'audio', 'bytes', 'data', 'content', 'output' ]:
				value = result.get( key )
				if isinstance( value, bytes ):
					return value
				
				if isinstance( value, bytearray ):
					return bytes( value )
		
		for attr_name in [ 'audio', 'bytes', 'data', 'content', 'output' ]:
			if hasattr( result, attr_name ):
				value = getattr( result, attr_name )
				if isinstance( value, bytes ):
					return value
				
				if isinstance( value, bytearray ):
					return bytes( value )
		
		if hasattr( result, 'read' ):
			try:
				value = result.read( )
				if isinstance( value, bytes ):
					return value
			except Exception:
				return None
		
		return None
	
	def extract_audio_usage( response: Any ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Extract token or usage metadata from an Audio provider response.
		
			Parameters:
			-----------
			response (Any): Provider response object.
		
			Returns:
			--------
			Dict[str, Any]: Usage metadata.
			
		"""
		if response is None:
			return { }
		
		usage = getattr( response, 'usage', None )
		if isinstance( usage, dict ):
			return usage
		
		if usage is not None:
			try:
				return dict( usage )
			except Exception:
				return { 'usage': str( usage ) }
		
		if isinstance( response, dict ) and isinstance( response.get( 'usage' ), dict ):
			return response.get( 'usage' )
		
		return { }
	
	def update_audio_usage( response: Any ) -> None:
		"""
			
			Purpose:
			--------
			Update Boo token counters from an Audio provider response when helper functions
			are available.
		
			Parameters:
			-----------
			response (Any): Provider response object.
		
			Returns:
			--------
			None
			
		"""
		try:
			if 'update_token_counters' in globals( ):
				update_token_counters( response )
			elif '_update_token_counters' in globals( ):
				_update_token_counters( response )
			elif 'update_counters' in globals( ):
				update_counters( response )
		except Exception:
			pass
	
	def append_audio_message( role: str, content: str ) -> None:
		"""
			
			Purpose:
			--------
			Append an Audio-mode message without clearing other mode state.
		
			Parameters:
			-----------
			role (str): Message role.
			content (str): Message content.
		
			Returns:
			--------
			None
			
		"""
		if not isinstance( st.session_state.get( 'audio_messages' ), list ):
			st.session_state[ 'audio_messages' ] = [ ]
		
		st.session_state[ 'audio_messages' ].append( {
					'role': role,
					'content': content,
			} )
	
	def render_audio_messages( ) -> None:
		"""
			
			Purpose:
			--------
			Render Audio-mode message history.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		if not isinstance( st.session_state.get( 'audio_messages' ), list ):
			st.session_state[ 'audio_messages' ] = [ ]
		
		for msg in st.session_state.get( 'audio_messages', [ ] ):
			if isinstance( msg, dict ):
				role = msg.get( 'role', 'assistant' )
				content = msg.get( 'content', '' )
				with st.chat_message( role ):
					st.markdown( content )
	
	def clear_audio_messages( ) -> None:
		"""
			
			Purpose:
			--------
			Clear Audio-mode message and result state.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'audio_messages' ] = [ ]
		st.session_state[ 'audio_output' ] = ''
		st.session_state[ 'audio_output_bytes' ] = None
		st.session_state[ 'audio_last_result' ] = { }
		st.session_state[ 'audio_last_usage' ] = { }
	
	def clear_audio_instructions( ) -> None:
		"""
			
			Purpose:
			--------
			Clear Audio-mode system instructions and selected prompt template.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'audio_system_instructions' ] = ''
		st.session_state[ 'instructions' ] = ''
	
	def convert_audio_system_instructions( ) -> None:
		"""
			
			Purpose:
			--------
			Convert Audio-mode system instructions between XML-like blocks and Markdown
			headings.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		text_value = st.session_state.get( 'audio_system_instructions', '' )
		if not isinstance( text_value, str ) or not text_value.strip( ):
			return
		
		source = text_value.strip( )
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'audio_system_instructions' ] = converted
	
	def load_audio_instruction_template( ) -> None:
		"""
			
			Purpose:
			--------
			Load the selected prompt template into Audio-mode system instructions.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		name = st.session_state.get( 'instructions' )
		if name and name != 'No Templates Found':
			prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
			if prompt_text is not None:
				st.session_state[ 'audio_system_instructions' ] = prompt_text
	
	def reset_audio_task_controls( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Audio task/model controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'audio_task',
				'audio_model',
				'audio_language',
				'audio_voice',
				'audio_rate',
				'audio_format',
				'audio_response_format',
				'audio_speed',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_audio_inference_controls( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Audio inference controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'audio_temperature',
				'audio_top_percent',
				'audio_frequency_penalty',
				'audio_presence_penalty',
				'audio_presense_penalty',
				'audio_include',
				'audio_stream',
				'audio_store',
				'audio_background',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_audio_playback_controls( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Audio playback controls through a widget-safe callback.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'audio_start_time',
				'audio_end_time',
				'audio_loop',
				'audio_autoplay',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def run_audio_transcription( file_path: str ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Run provider transcription and store normalized output.
		
			Parameters:
			-----------
			file_path (str): Temporary audio file path.
		
			Returns:
			--------
			Optional[str]: Transcript text.
			
		"""
		model = st.session_state.get( 'audio_model' )
		language = st.session_state.get( 'audio_language' )
		prompt = get_audio_prompt_value( 'Transcribe' )
		format_value = get_audio_format_value( 'Transcribe' )
		
		kwargs = {
				'path': file_path,
				'filepath': file_path,
				'file_path': file_path,
				'model': model or None,
				'language': language or None,
				'prompt': prompt or None,
				'format': format_value,
				'response_format': format_value,
				'temperature': st.session_state.get( 'audio_temperature' ),
				'include': st.session_state.get( 'audio_include', [ ] ),
				'instruct': st.session_state.get( 'audio_system_instructions', '' ),
		}
		
		result = call_existing_audio_method( instance=transcriber,
			method_names=[ 'transcribe', 'create_transcription', 'generate_transcription' ],
			kwargs=kwargs )
		
		text_result = normalize_audio_text_result( result )
		response = getattr( transcriber, 'response', None ) or result
		normalized = getattr( transcriber, 'normalized_result', None )
		
		st.session_state[ 'audio_output' ] = text_result
		st.session_state[ 'audio_last_result' ] = normalized if isinstance( normalized,
			dict ) else { }
		st.session_state[ 'audio_last_usage' ] = extract_audio_usage( response )
		update_audio_usage( response )
		
		return text_result
	
	def run_audio_translation( file_path: str ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Run provider translation and store normalized output.
		
			Parameters:
			-----------
			file_path (str): Temporary audio file path.
		
			Returns:
			--------
			Optional[str]: Translation text.
			
		"""
		model = st.session_state.get( 'audio_model' )
		language = st.session_state.get( 'audio_language' )
		prompt = get_audio_prompt_value( 'Translate' )
		format_value = get_audio_format_value( 'Translate' )
		
		kwargs = {
				'path': file_path,
				'filepath': file_path,
				'file_path': file_path,
				'model': model or None,
				'language': language or None,
				'prompt': prompt or None,
				'format': format_value,
				'response_format': format_value,
				'temperature': st.session_state.get( 'audio_temperature' ),
				'instruct': st.session_state.get( 'audio_system_instructions', '' ),
		}
		
		result = call_existing_audio_method( instance=translator,
			method_names=[ 'translate', 'create_translation', 'generate_translation' ], kwargs=kwargs )
		
		text_result = normalize_audio_text_result( result )
		response = getattr( translator, 'response', None ) or result
		normalized = getattr( translator, 'normalized_result', None )
		
		st.session_state[ 'audio_output' ] = text_result
		st.session_state[ 'audio_last_result' ] = normalized if isinstance( normalized,
			dict ) else { }
		st.session_state[ 'audio_last_usage' ] = extract_audio_usage( response )
		update_audio_usage( response )
		
		return text_result
	
	def run_audio_tts( text_value: str ) -> Optional[ bytes ]:
		"""
			
			Purpose:
			--------
			Run provider text-to-speech and store generated audio bytes.
		
			Parameters:
			-----------
			text_value (str): Text to synthesize.
		
			Returns:
			--------
			Optional[bytes]: Generated audio bytes.
			
		"""
		model = st.session_state.get( 'audio_model' )
		voice = st.session_state.get( 'audio_voice' )
		format_value = get_audio_format_value( 'Text-to-Speech' )
		speed = st.session_state.get( 'audio_speed', st.session_state.get( 'audio_rate', 1.0 ) )
		
		kwargs = {
				'text': text_value,
				'input': text_value,
				'model': model or None,
				'voice': voice or None,
				'format': format_value,
				'response_format': format_value,
				'output_format': format_value,
				'speed': speed,
				'rate': speed,
				'instruct': st.session_state.get( 'audio_system_instructions', '' ),
		}
		
		result = call_existing_audio_method( instance=tts,
			method_names=[ 'create_speech', 'synthesize', 'generate_speech', 'text_to_speech' ],
			kwargs=kwargs )
		
		audio_bytes = normalize_audio_bytes_result( result )
		response = getattr( tts, 'response', None ) or result
		normalized = getattr( tts, 'normalized_result', None )
		
		st.session_state[ 'audio_output_bytes' ] = audio_bytes
		st.session_state[ 'audio_last_result' ] = normalized if isinstance( normalized,
			dict ) else { }
		st.session_state[ 'audio_last_usage' ] = extract_audio_usage( response )
		update_audio_usage( response )
		
		return audio_bytes
	
	# ------------------------------------------------------------------
	# Session Safety
	# ------------------------------------------------------------------
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'audio_system_instructions' ] = ''
		st.session_state[ 'clear_audio_instructions' ] = False
		st.session_state[ 'clear_instructions' ] = False
	
	if not isinstance( st.session_state.get( 'audio_messages' ), list ):
		st.session_state[ 'audio_messages' ] = [ ]
	
	if not isinstance( st.session_state.get( 'audio_include' ), list ):
		st.session_state[ 'audio_include' ] = [ ]
	
	if not isinstance( st.session_state.get( 'audio_output' ), str ):
		st.session_state[ 'audio_output' ] = ''
	
	if 'audio_output_bytes' not in st.session_state:
		st.session_state[ 'audio_output_bytes' ] = None
	
	if not isinstance( st.session_state.get( 'audio_last_result' ), dict ):
		st.session_state[ 'audio_last_result' ] = { }
	
	if not isinstance( st.session_state.get( 'audio_last_usage' ), dict ):
		st.session_state[ 'audio_last_usage' ] = { }
	
	if not isinstance( st.session_state.get( 'audio_speed' ), float ):
		st.session_state[ 'audio_speed' ] = 1.0
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '🎧 Audio API', help=get_audio_help( 'AUDIO_API' ) )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				aud_c1, aud_c2, aud_c3, aud_c4, aud_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], gap='xxsmall', border=True )
				
				task_options = get_audio_task_options( )
				
				# ---------- Task ------------
				with aud_c1:
					if not task_options:
						st.info( 'Audio is not supported by the selected provider.' )
						audio_task = None
					else:
						st.selectbox( label='Mode', options=task_options, key='audio_task',
							placeholder='Options', index=None,
							help='Select the Audio API workflow to run.' )
						audio_task = st.session_state.get( 'audio_task' )
				
				model_options = get_audio_model_options( audio_task )
				language_options = get_audio_language_options( audio_task )
				voice_options = get_audio_voice_options( )
				format_options = get_audio_format_options(
					audio_task, st.session_state.get( 'audio_model' ) )
				
				if st.session_state.get( 'audio_model' ) not in model_options:
					st.session_state[ 'audio_model' ] = ''
				
				if st.session_state.get( 'audio_response_format' ) not in format_options:
					st.session_state[ 'audio_response_format' ] = ''
				
				include_options = get_audio_include_options(
					audio_task, st.session_state.get( 'audio_model' ) )
				st.session_state[ 'audio_include' ] = [
						item for item in st.session_state.get( 'audio_include', [ ] )
						if item in include_options
				]
				
				# ---------- Model ------------
				with aud_c2:
					st.selectbox( label='Model', options=model_options, key='audio_model',
						placeholder='Options', index=None,
						help='Task-aware Audio API model.' )
				
				# ---------- Language / Voice ------------
				with aud_c3:
					if audio_task in [ 'Transcribe', 'Translate' ]:
						st.selectbox( label='Language', options=language_options,
							key='audio_language', placeholder='Options', index=None,
							help='Optional source-language hint when supported.' )
					
					elif audio_task == 'Text-to-Speech':
						st.selectbox( label='Voice', options=voice_options,
							key='audio_voice', placeholder='Options', index=None,
							help='Text-to-speech voice when supported.' )
					
					else:
						st.caption( 'Language / Voice' )
						st.info( 'Select a task first.' )
				
				# ---------- Sample Rate ------------
				with aud_c4:
					st.selectbox( label='Sample Rate', options=get_audio_rate_options( ),
						key='audio_rate', placeholder='Options', index=None,
						help='Sample rate used for recording and playback controls.' )
				
				# ---------- Format ------------
				with aud_c5:
					st.selectbox( label='Format', options=format_options,
						key='audio_response_format', placeholder='Options', index=None,
						help='Task-aware response format or TTS audio output format.' )
					st.session_state[ 'audio_format' ] = st.session_state.get(
						'audio_response_format', '' )
				
				st.button( label='Reset', key='audio_model_reset', width='stretch',
					on_click=reset_audio_task_controls )
			
			with st.expander( label='Inference Options', icon='🎛️', expanded=False,
					width='stretch' ):
				prm_one, prm_two, prm_three, prm_four, prm_five = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ---------- Temperature ------------
				with prm_one:
					st.slider( label='Temperature', min_value=0.0, max_value=1.0,
						step=0.01, key='audio_temperature',
						help='Used by transcription/translation paths where supported.' )
				
				# ---------- Top-P ------------
				with prm_two:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						step=0.01, key='audio_top_percent',
						help=get_audio_help( 'TOP_P' ) )
				
				# ---------- Frequency Penalty ------------
				with prm_three:
					st.slider( label='Frequency Penalty', min_value=-2.0,
						max_value=2.0, step=0.01, key='audio_frequency_penalty',
						help=get_audio_help( 'FREQUENCY_PENALTY' ) )
				
				# ---------- Presence Penalty ------------
				with prm_four:
					st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
						step=0.01, key='audio_presence_penalty',
						help=get_audio_help( 'PRESENCE_PENALTY' ) )
					st.session_state[ 'audio_presense_penalty' ] = st.session_state.get(
						'audio_presence_penalty', 0.0 )
				
				# ---------- Speed ------------
				with prm_five:
					if audio_task == 'Text-to-Speech':
						st.selectbox( label='Speed', options=get_audio_speed_options( ),
							key='audio_speed', placeholder='Options', index=None,
							help='Text-to-speech speed when supported.' )
					else:
						st.caption( 'Speed' )
						st.info( 'Only used by Text-to-Speech.' )
				
				out_c1, out_c2, out_c3, out_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Include ------------
				with out_c1:
					if include_options:
						st.multiselect( label='Include', options=include_options,
							key='audio_include', placeholder='Options',
							help='Optional transcription include fields.' )
					else:
						st.caption( 'Include' )
						st.info( 'No include options for the selected task/model.' )
				
				# ---------- Stream ------------
				with out_c2:
					st.toggle( label='Stream', key='audio_stream',
						help=get_audio_help( 'STREAM' ) )
				
				# ---------- Store ------------
				with out_c3:
					st.toggle( label='Store', key='audio_store',
						help=get_audio_help( 'STORE' ) )
				
				# ---------- Background ------------
				with out_c4:
					st.toggle( label='Background', key='audio_background',
						help=get_audio_help( 'BACKGROUND_MODE' ) )
				
				st.button( label='Reset', key='audio_inference_reset', width='stretch',
					on_click=reset_audio_inference_controls )
			
			with st.expander( label='Playback Options', icon='▶️', expanded=False,
					width='stretch' ):
				play_c1, play_c2, play_c3, play_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Start Time ------------
				with play_c1:
					st.number_input( label='Start Time', min_value=0.0, step=1.0,
						key='audio_start_time',
						help='Audio playback start time in seconds.' )
				
				# ---------- End Time ------------
				with play_c2:
					st.number_input( label='End Time', min_value=0.0, step=1.0,
						key='audio_end_time',
						help='Audio playback end time in seconds. Zero means no end trim.' )
				
				# ---------- Loop ------------
				with play_c3:
					st.toggle( label='Loop', key='audio_loop',
						help='Loop local/test audio playback.' )
				
				# ---------- Autoplay ------------
				with play_c4:
					st.toggle( label='Autoplay', key='audio_autoplay',
						help='Autoplay local/test audio when supported by the browser.' )
				
				st.button( label='Reset', key='audio_playback_reset', width='stretch',
					on_click=reset_audio_playback_controls )
		
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			with in_left:
				st.text_area( label='Enter Text', height=80, width='stretch',
					help=get_audio_help( 'SYSTEM_INSTRUCTIONS' ),
					key='audio_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names,
					key='instructions', on_change=load_audio_instruction_template,
					index=None )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_audio_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_audio_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Audio Work Area
		# ------------------------------------------------------------------
		tab_process, tab_tts, tab_playback = st.tabs(
			[ 'Transcribe / Translate', 'Text-to-Speech', 'Playback' ] )
		
		with tab_process:
			render_audio_messages( )
			
			upload_c1, upload_c2 = st.columns( [ 0.50, 0.50 ], gap='medium' )
			
			with upload_c1:
				audio_file = st.file_uploader( label='Upload Audio',
					type=[ 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm', 'ogg' ],
					key='audio_file_uploader',
					accept_multiple_files=False )
			
			with upload_c2:
				sample_rate = st.session_state.get( 'audio_rate' )
				try:
					sample_rate = int( sample_rate ) if sample_rate else None
				except Exception:
					sample_rate = None
				
				if sample_rate:
					recording = st.audio_input( label='Record Audio', sample_rate=sample_rate,
						key='audio_recording_input' )
				else:
					recording = st.audio_input( label='Record Audio',
						key='audio_recording_input' )
			
			audio_source = audio_file or recording
			audio_path = save_uploaded_audio( audio_source ) if audio_source else None
			
			if audio_source is not None:
				st.audio( audio_source )
			
			transcription_prompt = st.text_area( label='Transcription Prompt',
				key='transcription_prompt', height=80, width='stretch',
				placeholder='Optional transcription prompt or vocabulary/context hints.' )
			
			translation_prompt = st.text_area( label='Translation Prompt',
				key='translation_prompt', height=80, width='stretch',
				placeholder='Optional translation prompt or instructions.' )
			
			process_c1, process_c2 = st.columns( [ 0.50, 0.50 ] )
			
			with process_c1:
				if st.button( 'Process Audio', key='process_audio', width='stretch' ):
					with st.spinner( 'Processing audio…' ):
						try:
							selected_task = st.session_state.get( 'audio_task' )
							
							if selected_task not in [ 'Transcribe', 'Translate' ]:
								st.warning(
									'Select Transcribe or Translate before processing audio.' )
							
							elif not audio_path:
								st.warning( 'Upload or record audio before processing.' )
							
							elif selected_task == 'Transcribe':
								result_text = run_audio_transcription( audio_path )
								if result_text:
									append_audio_message( 'user',
										'Transcribe uploaded/recorded audio.' )
									append_audio_message( 'assistant', result_text )
									st.text_area( 'Transcript', value=result_text, height=300 )
								else:
									st.warning( 'No transcript was returned.' )
							
							elif selected_task == 'Translate':
								result_text = run_audio_translation( audio_path )
								if result_text:
									append_audio_message( 'user',
										'Translate uploaded/recorded audio.' )
									append_audio_message( 'assistant', result_text )
									st.text_area( 'Translation', value=result_text, height=300 )
								else:
									st.warning( 'No translation was returned.' )
						
						except Exception as exc:
							err = Error( exc )
							st.error( f'Audio task failed: {err.info}' )
			
			with process_c2:
				if st.button( 'Clear Messages', key='audio_clear_process_messages',
						width='stretch', on_click=clear_audio_messages ):
					st.rerun( )
			
			if st.session_state.get( 'audio_output' ):
				st.download_button( label='Download Text Output',
					data=st.session_state.get( 'audio_output', '' ),
					file_name='audio_output.txt',
					mime='text/plain',
					width='stretch' )
		
		with tab_tts:
			render_audio_messages( )
			
			tts_input = st.text_area( label='Enter Text to Synthesize',
				key='audio_tts_input', height=160, width='stretch',
				placeholder='Enter text for speech synthesis.' )
			
			tts_c1, tts_c2 = st.columns( [ 0.50, 0.50 ] )
			
			with tts_c1:
				if st.button( 'Generate Audio', key='generate_tts_audio', width='stretch' ):
					with st.spinner( 'Synthesizing speech…' ):
						try:
							if st.session_state.get( 'audio_task' ) != 'Text-to-Speech':
								st.warning( 'Select Text-to-Speech as the Audio mode first.' )
							elif not isinstance( tts_input, str ) or not tts_input.strip( ):
								st.warning( 'Enter text before generating speech.' )
							else:
								audio_bytes = run_audio_tts( tts_input.strip( ) )
								
								if audio_bytes:
									append_audio_message( 'user', tts_input.strip( ) )
									append_audio_message( 'assistant',
										'Text-to-speech audio generated successfully.' )
									
									format_value = get_audio_format_value(
										'Text-to-Speech' ) or 'mp3'
									st.audio( audio_bytes, format=f'audio/{format_value}' )
									
									st.download_button( label='Download Audio',
										data=audio_bytes,
										file_name=f'tts_output.{format_value}',
										mime=f'audio/{format_value}',
										width='stretch' )
								else:
									st.warning( 'No audio bytes were returned.' )
						
						except Exception as exc:
							err = Error( exc )
							st.error( f'Text-to-speech failed: {err.info}' )
			
			with tts_c2:
				if st.button( 'Clear Messages', key='audio_clear_tts_messages',
						width='stretch', on_click=clear_audio_messages ):
					st.rerun( )
			
			if st.session_state.get( 'audio_output_bytes' ):
				format_value = get_audio_format_value( 'Text-to-Speech' ) or 'mp3'
				st.audio( st.session_state[ 'audio_output_bytes' ],
					format=f'audio/{format_value}' )
		
		with tab_playback:
			playback_c1, playback_c2 = st.columns( [ 0.50, 0.50 ], gap='medium' )
			
			with playback_c1:
				st.caption( 'Generated Audio' )
				if st.session_state.get( 'audio_output_bytes' ):
					format_value = get_audio_format_value( 'Text-to-Speech' ) or 'mp3'
					st.audio( st.session_state[ 'audio_output_bytes' ],
						format=f'audio/{format_value}' )
				else:
					st.info( 'No generated audio is available yet.' )
			
			with playback_c2:
				st.caption( 'Local Audio File' )
				local_audio = getattr( cfg, 'AUDIO_TEST_FILE', None )
				if local_audio:
					try:
						start_time = float( st.session_state.get( 'audio_start_time', 0.0 ) or 0.0 )
						end_time = float( st.session_state.get( 'audio_end_time', 0.0 ) or 0.0 )
						sample_rate = st.session_state.get( 'audio_rate' )
						
						st.audio( local_audio,
							sample_rate=int( sample_rate ) if sample_rate else None,
							start_time=start_time,
							end_time=end_time if end_time > 0 else None,
							format='audio/wav',
							loop=bool( st.session_state.get( 'audio_loop', False ) ),
							autoplay=bool( st.session_state.get( 'audio_autoplay', False ) ) )
					except TypeError:
						st.audio( local_audio )
					except Exception as exc:
						st.warning( f'Could not play local audio file: {exc}' )
				else:
					st.info( 'No local audio test file is configured.' )
		
		# ------------------------------------------------------------------
		# Result Metadata
		# ------------------------------------------------------------------
		if st.session_state.get( 'audio_last_usage' ) or st.session_state.get(
				'audio_last_result' ):
			with st.expander( label='Audio Result Metadata', icon='📊', expanded=False,
					width='stretch' ):
				if st.session_state.get( 'audio_last_usage' ):
					st.caption( 'Usage' )
					st.json( st.session_state.get( 'audio_last_usage', { } ) )
				
				if st.session_state.get( 'audio_last_result' ):
					st.caption( 'Normalized Result' )
					st.json( st.session_state.get( 'audio_last_result', { } ) )

# ======================================================================================
# DOCUMENTS MODE
# ======================================================================================
elif mode == 'Document Q&A':
	st.subheader( '📚 Document Q & A', help=cfg.DOCUMENT_Q_AND_A )
	st.divider( )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	docqna_number = st.session_state.get( 'docqna_number', 0 )
	docqna_max_calls = st.session_state.get( 'docqna_max_calls', 0 )
	docqna_max_searches = st.session_state.get( 'docqna_max_searches', 0 )
	docqna_max_tokens = st.session_state.get( 'docqna_max_tokens', 0 )
	docqna_top_percent = st.session_state.get( 'docqna_top_percent', 0.0 )
	docqna_top_k = st.session_state.get( 'docqna_top_k', 0 )
	docqna_freq = st.session_state.get( 'docqna_frequency_penalty', 0.0 )
	docqna_presense = st.session_state.get( 'docqna_presense_penalty', 0.0 )
	docqna_temperature = st.session_state.get( 'docqna_temperature', 0.0 )
	docqna_stream = st.session_state.get( 'docqna_stream', False )
	docqna_parallel_tools = st.session_state.get( 'docqna_parallel_tools', False )
	docqna_store = st.session_state.get( 'docqna_store', False )
	docqna_background = st.session_state.get( 'docqna_background', False )
	docqna_model = st.session_state.get( 'docqna_model', '' )
	docqna_reasoning = st.session_state.get( 'docqna_reasoning', '' )
	docqna_resolution = st.session_state.get( 'docqna_resolution', '' )
	docqna_media_resolution = st.session_state.get( 'docqna_media_resolution', '' )
	docqna_response_format = st.session_state.get( 'docqna_response_format', '' )
	docqna_tool_choice = st.session_state.get( 'docqna_tool_choice', '' )
	docqna_content = st.session_state.get( 'docqna_content', '' )
	docqna_input = st.session_state.get( 'docqna_input', '' )
	docqna_tools = st.session_state.get( 'docqna_tools', [ ] )
	docqna_modalities = st.session_state.get( 'docqna_modalities', [ ] )
	docqna_context = st.session_state.get( 'docqna_context', [ ] )
	docqna_include = st.session_state.get( 'docqna_include', [ ] )
	docqna_domains = st.session_state.get( 'docqna_domains', [ ] )
	docqna_stops = st.session_state.get( 'docqna_stops', [ ] )
	docqna_files = st.session_state.get( 'docqna_files' )
	docqna_uploaded = st.session_state.get( 'docqna_uploaded' )
	docqna_messages = st.session_state.get( 'docqna_messages' )
	docqna_active_docs = st.session_state.get( 'docqna_active_docs' )
	docqna_source = st.session_state.get( 'docqna_source' )
	docqna_multi_mode = st.session_state.get( 'docqna_multi_mode' )
	docqna = provider_module.Files( )
	
	for key in [ 'docqna_domains', 'docqna_stops', 'docqna_includes', 'docqna_input', ]:
		if key in st.session_state and isinstance( st.session_state[ key ], list ):
			del st.session_state[ key ]
	# ------------------------------------------------------------------
	#  DOCQNA SETTINGS
	# ------------------------------------------------------------------
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'docqna_system_instructions' ] = ''
		st.session_state[ 'clear_docqa_instructions' ] = False
		st.session_state[ 'clear_instructions' ] = False
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# EXPANDER — GROK DOCQNA LLM CONFIGURATION
		# ------------------------------------------------------------------
		if provider_name == 'Grok':
			with st.expander( label='LLM Configuration', icon='🧠', expanded=False, width='stretch' ):
				with st.expander( label='Model Settings', expanded=False, width='stretch' ):
					llm_c1, llm_c2, llm_c3, llm_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
					# ------------- Model Options ----------
					with llm_c1:
						model_options = list( docqna.model_options )
						set_docqna_model = st.selectbox( label='Select LLM', options=model_options,
							key='docqna_model', placeholder='Options', index=None,
							help='REQUIRED. Text Generation model used by the AI', )
						
						docqna_model = st.session_state[ 'docqna_model' ]
					
					# ------------- Include Options ----------
					with llm_c2:
						include_options = list( docqna.include_options )
						set_docqna_include = st.multiselect( label='Include:', options=include_options,
							key='docqna_include', help=cfg.INCLUDE, placeholder='Options' )
						
						docqna_include = [ d.strip( ) for d in set_docqna_include
						                   if d.strip( ) ]
						
						docqna_include = st.session_state[ 'docqna_include' ]
					
					# ------------- Reasoning Options ----------
					with llm_c3:
						reasoning_options = list( docqna.reasoning_options )
						set_docqna_reasoning = st.selectbox( label='Reasoning Effort:',
							options=reasoning_options, key='docqna_reasoning',
							help=cfg.REASONING, index=None, placeholder='Options' )
						
						docqna_reasoning = st.session_state[ 'docqna_reasoning' ]
					
					# ------------- Choice Options ----------
					with llm_c4:
						choice_options = list( docqna.choice_options )
						set_docqna_choice = st.multiselect( label='Tool Choice:', options=choice_options,
							key='docqna_tool_choice', help=cfg.INCLUDE, placeholder='Options' )
						
						docqna_tool_choice = st.session_state[ 'docqna_tool_choice' ]
					
					# ------------- Reset Settings ----------
					if st.button( label='Reset', key='docqna_model_reset', width='stretch' ):
						for key in [ 'docqna_model', 'docqna_include',
						             'docqna_reasoning', 'docqna_tool_choice' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Inference Settings', expanded=False, width='stretch' ):
					prm_c1, prm_c2, prm_c3, prm_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
					# ------------- Top P ----------
					with prm_c1:
						set_docqna_top_p = st.slider( label='Top-P', min_value=0.0, max_value=1.0,
							value=float( st.session_state.get( 'docqna_top_percent', 0.0 ) ),
							step=0.01, help=cfg.TOP_P, key='docqna_top_percent' )
						
						docqna_top_percent = st.session_state[ 'docqna_top_percent' ]
					
					# ------------- Temperature  ----------
					with prm_c2:
						set_docqna_temperature = st.slider( label='Temperature', min_value=0.0, max_value=1.0,
							value=float( st.session_state.get( 'docqna_temperature', 0.0 ) ), step=0.01,
							help=cfg.TEMPERATURE, key='docqna_temperature' )
						
						docqna_temperature = st.session_state[ 'docqna_temperature' ]
					
					# ------------- Number ----------
					with prm_c3:
						set_docqna_number = st.slider( label='Number', min_value=0, max_value=10,
							value=int( st.session_state.get( 'docqna_number', 0 ) ), step=1,
							help='Optional. Upper limit on the responses returned by the model',
							key='docqna_number' )
						
						docqna_number = st.session_state[ 'docqna_number' ]
					
					# ------------- Max tokens  ------------------
					with prm_c4:
						set_docqna_tokens = st.slider( label='Max Tokens',
							min_value=0, max_value=100000, step=500,
							value=int( st.session_state.get( 'docqna_max_tokens', 0 ) ),
							help=cfg.MAX_OUTPUT_TOKENS, key='docqna_max_tokens' )
						
						docqna_tokens = st.session_state[ 'docqna_max_tokens' ]
					
					# ------------- Reset Setting ----------
					if st.button( label='Reset', key='docqna_inference_reset', width='stretch' ):
						for key in [ 'docqna_top_percent', 'docqna_max_tokens',
						             'docqna_temperature', 'docqna_number', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
					tool_c1, tool_c2, tool_c3, tool_c4 = st.columns(
						[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='medium' )
					
					# ------------- Asynchronous  ------------------
					with tool_c1:
						set_docqna_parallel = st.toggle( label='Asynchronous Tool Calls', key='docqna_parallel_tools',
							help=cfg.PARALLEL_TOOL_CALLS )
						
						docqna_parallel_tools = st.session_state[ 'docqna_parallel_tools' ]
					
					# ------------- Max Tool Calls ------------------
					with tool_c2:
						set_docqna_calls = st.slider( label='Max Tool Calls', min_value=0, max_value=4,
							value=int( st.session_state.get( 'docqna_max_calls', 0 ) ), step=1,
							help=cfg.MAX_TOOL_CALLS, key='docqna_max_calls' )
						
						docqna_max_calls = st.session_state[ 'docqna_max_calls' ]
					
					# -------------  Max Web Searches ------------------
					with tool_c3:
						set_max_results = st.slider( label='Max Websearch Results', key='docqna_max_searches',
							value=int( st.session_state.get( 'docqna_max_searches', 0 ) ),
							min_value=0, max_value=30, step=1,
							help='Optional. Upper limit on the number web search results' )
						
						docqna_max_searches = st.session_state[ 'docqna_max_searches' ]
					
					# ------------- Tools ------------------
					with tool_c4:
						tool_options = list( docqna.tool_options )
						set_docqna_tools = st.multiselect( label='Tools:', options=tool_options,
							key='docqna_tools', help=cfg.TOOLS, placeholder='Options' )
						
						docqna_tools = [ d.strip( ) for d in set_docqna_tools
						                 if d.strip( ) ]
						
						docqna_tools = st.session_state[ 'docqna_tools' ]
					
					# ------------- Reset Settings -------------
					if st.button( label='Reset', key='docqna_tools_reset', width='stretch' ):
						for key in [ 'docqna_parallel_tools', 'docqna_max_searches',
						             'docqna_tools', 'docqna_max_calls' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Response Settings', expanded=False, width='stretch' ):
					resp_c1, resp_c2, resp_c3, resp_c4 = st.columns(
						[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='medium' )
					
					# ------------- Stream  ------------------
					with resp_c1:
						set_docqna_stream = st.toggle( label='Stream', key='docqna_stream',
							help=cfg.STREAM )
						
						docqna_stream = st.session_state[ 'docqna_stream' ]
					
					# ------------- Store  ------------------
					with resp_c2:
						set_docqna_store = st.toggle( label='Store', key='docqna_store',
							help=cfg.STORE )
						
						docqna_store = st.session_state[ 'docqna_store' ]
					
					# ------------- Background  ------------------
					with resp_c3:
						set_docqna_background = st.toggle( label='Background', key='docqna_background',
							help=cfg.BACKGROUND_MODE )
						
						docqna_background = st.session_state[ 'docqna_background' ]
					
					# ------------- Domains  ------------------
					with resp_c4:
						set_docqna_domains = st.text_input( label='Allowed Websites', key='docqna_domains',
							help=cfg.STOP_SEQUENCE, width='stretch', placeholder='Enter Web Domains' )
						
						docqna_domains = [ d.strip( ) for d in set_docqna_domains.split( ',' )
						                   if d.strip( ) ]
					
					# ------------- Reset Settings  ------------------
					if st.button( label='Reset', key='docqna_response_reset', width='stretch' ):
						for key in [ 'docqna_stream', 'docqna_store',
						             'docqna_background', 'docqna_domains' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						# If using separated UI key for stops
						if 'docqna_stops_input' in st.session_state:
							del st.session_state[ 'docqna_stops_input' ]
						
						st.rerun( )
		
		# ------------------------------------------------------------------
		# EXPANDER — GEMINI DOCQNA LLM CONFIGURATION
		# ------------------------------------------------------------------
		elif provider_name == 'Gemini':
			with st.expander( label='LLM Configuration', icon='🧠', expanded=False, width='stretch' ):
				with st.expander( label='Model Settings', expanded=False, width='stretch' ):
					llm_c1, llm_c2, llm_c3, llm_c4, llm_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Model ------------
					with llm_c1:
						model_options = list( docqna.model_options )
						set_docqna_model = st.selectbox( label='Select Model', options=model_options,
							key='docqna_model', placeholder='Options', index=None,
							help='REQUIRED. Text Generation model used by the AI', )
						
						docqna_model = st.session_state[ 'docqna_model' ]
					
					# ---------- Include ------------
					with llm_c2:
						include_options = list( docqna.include_options )
						set_docqna_include = st.multiselect( label='Include', options=include_options,
							key='docqna_include', help=cfg.INCLUDE, placeholder='Options' )
						
						docqna_include = [ d.strip( ) for d in set_docqna_include
						                   if d.strip( ) ]
						
						docqna_include = st.session_state[ 'docqna_include' ]
					
					# ---------- Allowed Domains ------------
					with llm_c3:
						set_docqna_domains = st.text_input( label='Allowed Domains', key='docqna_domains_input',
							value=','.join( st.session_state.get( 'docqna_domains', [ ] ) ),
							help=cfg.ALLOWED_DOMAINS, width='stretch', placeholder='Enter Domains' )
						
						docqna_domains = [ d.strip( ) for d in set_docqna_domains.split( ',' )
						                   if d.strip( ) ]
						
						st.session_state[ 'docqna_domains' ] = docqna_domains
					
					# ---------- Reasoning/Thinking Level ------------
					with llm_c4:
						reasoning_options = list( docqna.reasoning_options )
						set_docqna_reasoning = st.selectbox( label='Thinking Level:',
							options=reasoning_options, key='docqna_reasoning',
							help=cfg.REASONING, index=None, placeholder='Options' )
						
						docqna_reasoning = st.session_state[ 'docqna_reasoning' ]
					
					# ---------- Media Resolution ------------
					with llm_c5:
						media_options = list( docqna.media_options )
						set_media_resolution = st.selectbox( label='Media Resolution',
							options=media_options, key='docqna_media_resolution',
							help=cfg.REASONING, index=None, placeholder='Options' )
						
						media_resolution = st.session_state[ 'docqna_media_resolution' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_model_reset', width='stretch' ):
						for key in [ 'docqna_model', 'docqna_include', 'docqna_domains',
						             'docqna_reasoning', 'docqna_media_resolution' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Inference Settings', expanded=False, width='stretch' ):
					prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Top-P ------------
					with prm_c1:
						set_docqna_top_p = st.slider( label='Top-P', min_value=0.0, max_value=1.0,
							value=float( st.session_state.get( 'docqna_top_percent', 0.0 ) ),
							step=0.01, help=cfg.TOP_P, key='docqna_top_percent' )
						
						docqna_top_percent = st.session_state[ 'docqna_top_percent' ]
					
					# ---------- Frequency ------------
					with prm_c2:
						set_docqna_freq = st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
							value=float( st.session_state.get( 'docqna_frequency_penalty', 0.0 ) ),
							step=0.01, help=cfg.FREQUENCY_PENALTY, key='docqna_frequency_penalty' )
						
						docqna_fequency = st.session_state[ 'docqna_frequency_penalty' ]
					
					# ---------- Presense ------------
					with prm_c3:
						set_docqna_presense = st.slider( label='Presense Penalty', min_value=-2.0, max_value=2.0,
							value=float( st.session_state.get( 'docqna_presense_penalty', 0.0 ) ),
							step=0.01, help=cfg.PRESENCE_PENALTY, key='docqna_presense_penalty' )
						
						docqna_presense = st.session_state[ 'docqna_presense_penalty' ]
					
					# ---------- Temperature ------------
					with prm_c4:
						set_docqna_temperature = st.slider( label='Temperature', min_value=0.0, max_value=1.0,
							value=float( st.session_state.get( 'docqna_temperature', 0.0 ) ), step=0.01,
							help=cfg.TEMPERATURE, key='docqna_temperature' )
						
						docqna_temperature = st.session_state[ 'docqna_temperature' ]
					
					# ---------- Top-K ------------
					with prm_c5:
						set_docqna_topk = st.slider( label='Top K', min_value=0, max_value=20,
							value=int( st.session_state.get( 'docqna_top_k', 0 ) ), step=1,
							help=cfg.TOP_K,
							key='docqna_top_k' )
						
						docqna_top_k = st.session_state[ 'docqna_top_k' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_inference_reset', width='stretch' ):
						for key in [ 'docqna_top_percent', 'docqna_frequency_penalty',
						             'docqna_presense_penalty', 'docqna_temperature',
						             'docqna_top_k', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
					tool_c1, tool_c2, tool_c3, tool_c4, tool_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Number/Candidates ------------
					with tool_c1:
						set_docqna_number = st.slider( label='Candidates', min_value=0, max_value=50,
							value=int( st.session_state.get( 'docqna_number', 0 ) ), step=1,
							help='Optional. Upper limit on the responses returned by the model',
							key='docqna_number' )
						
						docqna_number = st.session_state[ 'docqna_number' ]
					
					# ---------- Max Calls ------------
					with tool_c2:
						set_docqna_calls = st.slider( label='Max Tool Calls', min_value=0, max_value=10,
							value=int( st.session_state.get( 'docqna_max_calls', 0 ) ), step=1,
							help=cfg.MAX_TOOL_CALLS, key='docqna_max_calls' )
						
						docqna_max_calls = st.session_state[ 'docqna_max_calls' ]
					
					# ---------- Choice/Calling Mode ------------
					with tool_c3:
						choice_options = list( docqna.choice_options )
						set_docqna_choice = st.selectbox( label='Calling Mode', options=choice_options,
							key='docqna_tool_choice', help=cfg.CHOICE, index=None, placeholder='Options' )
						
						docqna_tool_choice = st.session_state[ 'docqna_tool_choice' ]
					
					# ---------- Tools ------------
					with tool_c4:
						tool_options = list( docqna.tool_options )
						set_docqna_tools = st.multiselect( label='Available Tools', options=tool_options,
							key='docqna_tools', help=cfg.TOOLS, placeholder='Options' )
						
						docqna_tools = [ d.strip( ) for d in set_docqna_tools
						                 if d.strip( ) ]
						
						docqna_tools = st.session_state[ 'docqna_tools' ]
					
					# ---------- Modalities ------------
					with tool_c5:
						modality_options = list( docqna.modality_options )
						set_docqna_modalities = st.multiselect( label='Response Modalities', options=modality_options,
							key='docqna_modalities', help='Optional. Modality of the response',
							placeholder='Options' )
						
						docqna_modalities = [ d.strip( ) for d in set_docqna_modalities
						                      if d.strip( ) ]
						
						docqna_modalities = st.session_state[ 'docqna_modalities' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_tools_reset', width='stretch' ):
						for key in [ 'docqna_parallel_tools', 'docqna_tool_choice', 'docqna_number',
						             'docqna_tools', 'docqna_max_calls', 'docqna_modalities' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Response Settings', expanded=False, width='stretch' ):
					resp_c1, resp_c2, resp_c3, resp_c4, resp_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Stream ------------
					with resp_c1:
						set_docqna_stream = st.toggle( label='Stream', key='docqna_stream',
							help=cfg.STREAM )
						
						docqna_stream = st.session_state[ 'docqna_stream' ]
					
					# ---------- Store ------------
					with resp_c2:
						set_docqna_store = st.toggle( label='Store', key='docqna_store', help=cfg.STORE )
						
						docqna_store = st.session_state[ 'docqna_store' ]
					
					# ---------- Background ------------
					with resp_c3:
						set_docqna_background = st.toggle( label='Background', key='docqna_background',
							help=cfg.BACKGROUND_MODE )
						
						docqna_background = st.session_state[ 'docqna_background' ]
					
					# ---------- Stops ------------
					with resp_c4:
						set_docqna_stops = st.text_input( label='Stop Sequences', key='docqna_stops',
							help=cfg.STOP_SEQUENCE, width='stretch', placeholder='Enter Stops' )
						
						docqna_stops = [ d.strip( ) for d in set_docqna_stops.split( ',' )
						                 if d.strip( ) ]
					
					# ---------- Max Tokens ------------
					with resp_c5:
						set_docqna_tokens = st.slider( label='Max Tokens', min_value=0, max_value=100000,
							value=int( st.session_state.get( 'docqna_max_tokens', 0 ) ), step=500,
							help=cfg.MAX_OUTPUT_TOKENS, key='docqna_max_tokens' )
						
						docqna_tokens = st.session_state[ 'docqna_max_tokens' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_response_reset', width='stretch' ):
						for key in [ 'docqna_stream', 'docqna_store', 'docqna_background',
						             'docqna_stops',
						             'docqna_max_tokens' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						# If using separated UI key for stops
						if 'docqna_stops_input' in st.session_state:
							del st.session_state[ 'docqna_stops_input' ]
						
						st.rerun( )
		
		# ------------------------------------------------------------------
		# EXPANDER — GPT DOCQNA LLM CONFIGURATION
		# ------------------------------------------------------------------
		elif provider_name == 'GPT':
			with st.expander( label='LLM Configuration', icon='🧠', expanded=False, width='stretch' ):
				with st.expander( label='Model Settings', expanded=False, width='stretch' ):
					llm_c1, llm_c2, llm_c3, llm_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
					# ---------- Model ------------
					with llm_c1:
						model_options = list( docqna.model_options )
						set_docqna_model = st.selectbox( label='Select Model', options=model_options,
							key='docqna_model', placeholder='Options', index=None,
							help='REQUIRED. Text Generation model used by the AI', )
						
						docqna_model = st.session_state[ 'docqna_model' ]
					
					# ---------- Include ------------
					with llm_c2:
						include_options = list( docqna.include_options )
						set_docqna_include = st.multiselect( label='Include:', options=include_options,
							key='docqna_include', help=cfg.INCLUDE, placeholder='Options' )
						
						docqna_include = [ d.strip( ) for d in set_docqna_include
						                   if d.strip( ) ]
						
						docqna_include = st.session_state[ 'docqna_include' ]
					
					# ---------- Allowed Domains ------------
					with llm_c3:
						set_docqna_domains = st.text_input( label='Allowed Domains', key='docqna_domains_input',
							value=','.join( st.session_state.get( 'docqna_domains', [ ] ) ),
							help=cfg.ALLOWED_DOMAINS, width='stretch', placeholder='Enter Domains' )
						
						docqna_domains = [ d.strip( ) for d in set_docqna_domains.split( ',' )
						                   if d.strip( ) ]
						
						st.session_state[ 'docqna_domains' ] = docqna_domains
					
					# ---------- Reasoning ------------
					with llm_c4:
						reasoning_options = list( docqna.reasoning_options )
						set_docqna_reasoning = st.selectbox( label='Reasoning Effort:',
							options=reasoning_options, key='docqna_reasoning',
							help=cfg.REASONING, index=None, placeholder='Options' )
						
						docqna_reasoning = st.session_state[ 'docqna_reasoning' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_model_reset', width='stretch' ):
						for key in [ 'docqna_model', 'docqna_include', 'docqna_domains',
						             'docqna_reasoning' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Inference Settings', expanded=False, width='stretch' ):
					prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Top-P ------------
					with prm_c1:
						set_docqna_top_p = st.slider( label='Top-P', min_value=0.0, max_value=1.0,
							value=float( st.session_state.get( 'docqna_top_percent', 0.0 ) ),
							step=0.01, help=cfg.TOP_P, key='docqna_top_percent' )
						
						docqna_top_percent = st.session_state[ 'docqna_top_percent' ]
					
					# ---------- Frequency ------------
					with prm_c2:
						set_docqna_freq = st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
							value=float( st.session_state.get( 'docqna_frequency_penalty', 0.0 ) ),
							step=0.01, help=cfg.FREQUENCY_PENALTY, key='docqna_frequency_penalty' )
						
						docqna_fequency = st.session_state[ 'docqna_frequency_penalty' ]
					
					# ---------- Presense ------------
					with prm_c3:
						set_docqna_presense = st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
							value=float( st.session_state.get( 'docqna_presense_penalty', 0.0 ) ),
							step=0.01, help=cfg.PRESENCE_PENALTY, key='docqna_presense_penalty' )
						
						docqna_presense = st.session_state[ 'docqna_presense_penalty' ]
					
					# ---------- Temperature ------------
					with prm_c4:
						set_docqna_temperature = st.slider( label='Temperature', min_value=0.0, max_value=1.0,
							value=float( st.session_state.get( 'docqna_temperature', 0.0 ) ), step=0.01,
							help=cfg.TEMPERATURE, key='docqna_temperature' )
						
						docqna_temperature = st.session_state[ 'docqna_temperature' ]
					
					# ---------- Number ------------
					with prm_c5:
						set_docqna_number = st.slider( label='Number', min_value=0, max_value=10,
							value=int( st.session_state.get( 'docqna_number', 0 ) ), step=1,
							help='Optional. Upper limit on the responses returned by the model',
							key='docqna_number' )
						
						docqna_number = st.session_state[ 'docqna_number' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_inference_reset', width='stretch' ):
						for key in [ 'docqna_top_percent', 'docqna_frequency_penalty',
						             'docqna_presense_penalty', 'docqna_temperature',
						             'docqna_number', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
					tool_c1, tool_c2, tool_c3, tool_c4 = st.columns(
						[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='medium' )
					
					# ---------- Allow Parallel ------------
					with tool_c1:
						set_docqna_parallel = st.toggle( label='Asychronous Calls',
							key='docqna_parallel_tools',
							help=cfg.PARALLEL_TOOL_CALLS )
						
						docqna_parallel_tools = st.session_state[ 'docqna_parallel_tools' ]
					
					# ---------- Max Calls ------------
					with tool_c2:
						set_docqna_calls = st.slider( label='Max Tool Calls', min_value=0, max_value=5,
							value=int( st.session_state.get( 'docqna_max_calls', 0 ) ), step=1,
							help=cfg.MAX_TOOL_CALLS, key='docqna_max_calls' )
						
						docqna_max_calls = st.session_state[ 'docqna_max_calls' ]
					
					# ---------- Choice ------------
					with tool_c3:
						choice_options = list( docqna.choice_options )
						set_docqna_choice = st.selectbox( label='Tool Choice:', options=choice_options,
							key='docqna_tool_choice', help=cfg.CHOICE, index=None, placeholder='Options' )
						
						docqna_tool_choice = st.session_state[ 'docqna_tool_choice' ]
					
					# ---------- Tools ------------
					with tool_c4:
						tool_options = list( docqna.tool_options )
						set_docqna_tools = st.multiselect( label='Available Tools', options=tool_options,
							key='docqna_tools', help=cfg.TOOLS, placeholder='Options' )
						
						docqna_tools = [ d.strip( ) for d in set_docqna_tools
						                 if d.strip( ) ]
						
						docqna_tools = st.session_state[ 'docqna_tools' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_tools_reset', width='stretch' ):
						for key in [ 'docqna_parallel_tools', 'docqna_tool_choice',
						             'docqna_tools', 'docqna_max_calls' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				with st.expander( label='Response Settings', expanded=False, width='stretch' ):
					resp_c1, resp_c2, resp_c3, resp_c4, resp_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Stream ------------
					with resp_c1:
						set_docqna_stream = st.toggle( label='Stream', key='docqna_stream',
							help=cfg.STREAM )
						
						docqna_stream = st.session_state[ 'docqna_stream' ]
					
					# ---------- Store ------------
					with resp_c2:
						set_docqna_store = st.toggle( label='Store', key='docqna_store',
							help=cfg.STORE )
						
						docqna_store = st.session_state[ 'docqna_store' ]
					
					# ---------- Background ------------
					with resp_c3:
						set_docqna_background = st.toggle( label='Background', key='docqna_background',
							help=cfg.BACKGROUND_MODE )
						
						docqna_background = st.session_state[ 'docqna_background' ]
					
					# ---------- Stops ------------
					with resp_c4:
						set_docqna_stops = st.text_input( label='Stop Sequences', key='docqna_stops',
							help=cfg.STOP_SEQUENCE, width='stretch', placeholder='Enter Stops' )
						
						docqna_stops = [ d.strip( ) for d in set_docqna_stops.split( ',' )
						                 if d.strip( ) ]
					
					# ---------- Max Tokens ------------
					with resp_c5:
						set_docqna_tokens = st.slider( label='Max Output Tokens', min_value=0, max_value=100000,
							value=int( st.session_state.get( 'docqna_max_tokens', 0 ) ), step=500,
							help=cfg.MAX_OUTPUT_TOKENS, key='docqna_max_tokens' )
						
						docqna_tokens = st.session_state[ 'docqna_max_tokens' ]
					
					# ---------- Reset Settings ------------
					if st.button( label='Reset', key='docqna_response_reset', width='stretch' ):
						for key in [ 'docqna_stream', 'docqna_store', 'docqna_background',
						             'docqna_stops',
						             'docqna_max_tokens' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						# If using separated UI key for stops
						if 'docqna_stops_input' in st.session_state:
							del st.session_state[ 'docqna_stops_input' ]
						
						st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — DocQA System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			with in_left:
				st.text_area( 'Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='docqna_system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'docqna_system_instructions' ] = text
			
			with in_right:
				st.selectbox( 'Select Template', prompt_names,
					key='instructions', on_change=_on_template_change, index=None )
			
			def _on_clear( ) -> None:
				st.session_state[ 'docqna_system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( 'Clear Instructions', width='stretch', on_click=_on_clear )
		
		doc_left, doc_right = st.columns( [ 0.2, 0.8 ], border=True )
		with doc_left:
			docqna_uploaded = st.file_uploader( 'Upload', type=[ 'pdf', 'txt', 'md', 'docx' ],
				accept_multiple_files=False, label_visibility='visible' )
			
			if docqna_uploaded is not None:
				st.session_state.docqna_active_docs = [ docqna_uploaded.name ]
				st.session_state.doc_bytes = { docqna_uploaded.name: docqna_uploaded.getvalue( ) }
				st.success( f'{docqna_uploaded.name} has been loaded!' )
			else:
				st.info( 'Load a document.' )
			
			unload = st.button( label='Unload Document', width='stretch' )
			if unload:
				docqna_uploaded = None
				st.session_state.docqna_active_docs = None
		
		with doc_right:
			if st.session_state.get( 'docqna_active_docs' ):
				name = st.session_state.docqna_active_docs[ 0 ]
				file_bytes = st.session_state.doc_bytes.get( name )
				if file_bytes:
					st.pdf( file_bytes, height=420 )
		
		for msg in st.session_state.docqna_messages:
			with st.chat_message( msg[ 'role' ] ):
				st.markdown( msg[ 'content' ] )
		
		if prompt := st.chat_input( 'Ask a question about the document' ):
			st.session_state.docqna_messages.append( { 'role': 'user', 'content': prompt } )
			response = route_document_query( prompt )
			st.session_state.docqna_messages.append( { 'role': 'assistant', 'content': response } )
			st.rerun( )

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == 'Embeddings':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	embedding = get_embeddings_module( provider_name )
	
	# ------------------------------------------------------------------
	# Embeddings Mode Helpers
	# ------------------------------------------------------------------
	def get_embedding_help( name: str, fallback: str = '' ) -> str:
		"""
			
			Purpose:
			--------
			Return Embeddings mode help text from config.py without failing when a constant is
			absent.
		
			Parameters:
			-----------
			name (str): Config attribute name.
			fallback (str): Fallback help text.
		
			Returns:
			--------
			str: Help text value.
			
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_embedding_options( instance: Any, attr_name: str,
			fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""
			
			Purpose:
			--------
			Return list-like option values from an Embeddings wrapper property.
		
			Parameters:
			-----------
			instance (Any): Provider Embeddings wrapper instance.
			attr_name (str): Property or attribute name to inspect.
			fallback (Optional[List[Any]]): Fallback option values.
		
			Returns:
			--------
			List[Any]: Option values safe for Streamlit controls.
			
		"""
		values = getattr( instance, attr_name, None )
		if callable( values ):
			try:
				values = values( )
			except Exception:
				values = None
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if isinstance( values, list ):
			return values
		
		return fallback or [ ]
	
	def normalize_embedding_text( value: Any ) -> str:
		"""
			
			Purpose:
			--------
			Normalize user-provided embedding input text.
		
			Parameters:
			-----------
			value (Any): Raw text-like value.
		
			Returns:
			--------
			str: Normalized text value.
			
		"""
		if value is None:
			return ''
		
		return str( value ).replace( '\r\n', '\n' ).strip( )
	
	def chunk_embedding_text( text_value: str, chunk_size: int, overlap: int ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Chunk embedding text using token-aware helpers when available, with a safe word-based
			fallback.
		
			Parameters:
			-----------
			text_value (str): Text to split into embedding chunks.
			chunk_size (int): Maximum chunk size.
			overlap (int): Overlap between adjacent chunks.
		
			Returns:
			--------
			List[str]: Text chunks.
			
		"""
		source = normalize_embedding_text( text_value )
		if not source:
			return [ ]
		
		if chunk_size <= 0:
			return [ source ]
		
		for helper_name in [ 'chunk_text', 'chunk_by_tokens', 'split_text' ]:
			helper = globals( ).get( helper_name )
			if callable( helper ):
				try:
					return helper( source, chunk_size, overlap )
				except TypeError:
					try:
						return helper( source, chunk_size=chunk_size, overlap=overlap )
					except Exception:
						pass
				except Exception:
					pass
		
		words = source.split( )
		if not words:
			return [ ]
		
		step = max( 1, chunk_size - max( 0, overlap ) )
		chunks = [ ]
		
		for index in range( 0, len( words ), step ):
			chunk = ' '.join( words[ index:index + chunk_size ] ).strip( )
			if chunk:
				chunks.append( chunk )
		
		return chunks
	
	def normalize_embedding_vectors( vectors: Any ) -> List[ List[ float ] ]:
		"""
			
			Purpose:
			--------
			Normalize provider embedding vectors into a two-dimensional float matrix.
		
			Parameters:
			-----------
			vectors (Any): Provider vector result.
		
			Returns:
			--------
			List[List[float]]: Two-dimensional vector matrix.
			
		"""
		if vectors is None:
			return [ ]
		
		if isinstance( vectors, dict ):
			for key in [ 'data', 'embeddings', 'vectors', 'embedding' ]:
				if key in vectors:
					return normalize_embedding_vectors( vectors.get( key ) )
		
		if hasattr( vectors, 'data' ):
			return normalize_embedding_vectors( getattr( vectors, 'data' ) )
		
		if hasattr( vectors, 'embeddings' ):
			return normalize_embedding_vectors( getattr( vectors, 'embeddings' ) )
		
		if hasattr( vectors, 'embedding' ):
			return normalize_embedding_vectors( getattr( vectors, 'embedding' ) )
		
		if isinstance( vectors, list ) and vectors:
			first = vectors[ 0 ]
			
			if isinstance( first, float ) or isinstance( first, int ):
				return [ [ float( value ) for value in vectors ] ]
			
			if isinstance( first, dict ):
				rows = [ ]
				for item in vectors:
					if 'embedding' in item:
						rows.extend( normalize_embedding_vectors( item.get( 'embedding' ) ) )
					elif 'vector' in item:
						rows.extend( normalize_embedding_vectors( item.get( 'vector' ) ) )
				return rows
			
			if hasattr( first, 'embedding' ):
				return [
						[ float( value ) for value in getattr( item, 'embedding' ) ]
						for item in vectors
						if hasattr( item, 'embedding' )
				]
			
			if isinstance( first, list ):
				return [
						[ float( value ) for value in row ]
						for row in vectors
						if isinstance( row, list )
				]
		
		return [ ]
	
	def call_embeddings_create( chunks: List[ str ] ) -> Any:
		"""
			
			Purpose:
			--------
			Call the selected provider Embeddings wrapper using the first compatible method.
		
			Parameters:
			-----------
			chunks (List[str]): Text chunks to embed.
		
			Returns:
			--------
			Any: Provider embedding result.
			
		"""
		input_value = chunks if len( chunks ) != 1 else chunks[ 0 ]
		dimensions = st.session_state.get( 'embedding_dimensions',
			st.session_state.get( 'embeddings_dimensions', 0 ) )
		encoding_format = st.session_state.get( 'embedding_encoding_format',
			st.session_state.get( 'embeddings_encoding_format', '' ) )
		
		kwargs = {
				'input': input_value,
				'text': input_value,
				'texts': chunks,
				'model': st.session_state.get( 'embedding_model' ) or None,
				'dimensions': dimensions if int( dimensions or 0 ) > 0 else None,
				'encoding_format': encoding_format or None,
				'format': encoding_format or None,
		}
		
		for method_name in [ 'create', 'embed', 'embed_text', 'generate', 'create_embeddings' ]:
			method = getattr( embedding, method_name, None )
			if callable( method ):
				try:
					return method( **kwargs )
				except TypeError:
					clean_kwargs = {
							key: value
							for key, value in kwargs.items( )
							if value is not None and value != '' and value != [ ]
					}
					return method( **clean_kwargs )
		
		raise AttributeError(
			f'Provider "{provider_name}" does not expose a compatible Embeddings method.' )
	
	def extract_embedding_usage( result: Any ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Extract usage metadata from an Embeddings provider response.
		
			Parameters:
			-----------
			result (Any): Provider embedding result.
		
			Returns:
			--------
			Dict[str, Any]: Usage metadata.
			
		"""
		response = getattr( embedding, 'response', None ) or result
		
		if response is None:
			return { }
		
		usage = getattr( response, 'usage', None )
		if isinstance( usage, dict ):
			return usage
		
		if usage is not None:
			try:
				return dict( usage )
			except Exception:
				return { 'usage': str( usage ) }
		
		if isinstance( response, dict ) and isinstance( response.get( 'usage' ), dict ):
			return response.get( 'usage' )
		
		return { }
	
	def build_embedding_metrics( source_text: str, chunks: List[ str ],
			vectors: List[ List[ float ] ], usage: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Build display metrics for an Embeddings run.
		
			Parameters:
			-----------
			source_text (str): Original source text.
			chunks (List[str]): Text chunks embedded.
			vectors (List[List[float]]): Normalized embedding vectors.
			usage (Dict[str, Any]): Provider usage metadata.
		
			Returns:
			--------
			Dict[str, Any]: Embedding metrics.
			
		"""
		words = source_text.split( )
		total_words = len( words )
		unique_words = len( set( words ) )
		token_total = count_tokens( source_text ) if 'count_tokens' in globals( ) else total_words
		dimensions = len( vectors[ 0 ] ) if vectors else 0
		
		return {
				'tokens': token_total,
				'words': total_words,
				'unique_words': unique_words,
				'ttr': (unique_words / total_words) if total_words > 0 else 0.0,
				'characters': len( source_text ),
				'chunks': len( chunks ),
				'vectors': len( vectors ),
				'dimensions': dimensions,
				'usage': usage,
		}
	
	def build_embeddings_dataframe( chunks: List[ str ],
			vectors: List[ List[ float ] ] ) -> pd.DataFrame:
		"""
			
			Purpose:
			--------
			Build a dataframe from embedding vectors and chunk metadata.
		
			Parameters:
			-----------
			chunks (List[str]): Embedded text chunks.
			vectors (List[List[float]]): Normalized embedding vectors.
		
			Returns:
			--------
			pd.DataFrame: Embedding dataframe.
			
		"""
		if not vectors:
			return pd.DataFrame( )
		
		df_vectors = pd.DataFrame(
			vectors,
			columns=[ f'dim_{index}' for index in range( len( vectors[ 0 ] ) ) ]
		)
		
		df_vectors.insert( 0, 'ChunkIndex', range( 1, len( df_vectors ) + 1 ) )
		
		if chunks:
			df_vectors.insert( 1, 'Text', chunks[ :len( df_vectors ) ] )
		
		return df_vectors
	
	def render_embedding_metrics( metrics: Dict[ str, Any ] ) -> None:
		"""
			
			Purpose:
			--------
			Render Embeddings mode metric cards.
		
			Parameters:
			-----------
			metrics (Dict[str, Any]): Embedding metric values.
		
			Returns:
			--------
			None
			
		"""
		col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns( 5, border=True )
		col_m1.metric( 'Tokens', metrics.get( 'tokens', 0 ) )
		col_m2.metric( 'Chunks', metrics.get( 'chunks', 0 ) )
		col_m3.metric( 'Vectors', metrics.get( 'vectors', 0 ) )
		col_m4.metric( 'Dimensions', metrics.get( 'dimensions', 0 ) )
		col_m5.metric( 'TTR', f"{float( metrics.get( 'ttr', 0.0 ) ):.3f}" )
	
	def reset_embeddings_all( ) -> None:
		"""
			
			Purpose:
			--------
			Reset Embeddings mode input, output, and display state.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		for key in [
				'embedding_input',
				'embedding_text',
				'embeddings_input_text',
				'embedding_chunks',
				'embeddings_chunks',
				'embedding_vectors',
				'embeddings',
				'embedding_results',
				'embeddings_df',
				'embedding_dataframe',
				'embedding_metrics',
				'embedding_usage',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def update_embedding_usage( response: Any ) -> None:
		"""
			
			Purpose:
			--------
			Update Boo token counters from an Embeddings provider response when helpers exist.
		
			Parameters:
			-----------
			response (Any): Provider response object.
		
			Returns:
			--------
			None
			
		"""
		try:
			if 'update_token_counters' in globals( ):
				update_token_counters( response )
			elif '_update_token_counters' in globals( ):
				_update_token_counters( response )
			elif 'update_counters' in globals( ):
				update_counters( response )
		except Exception:
			pass
	
	# ------------------------------------------------------------------
	# Session Safety
	# ------------------------------------------------------------------
	if 'embeddings_dimensions' in st.session_state and 'embedding_dimensions' not in st.session_state:
		st.session_state[ 'embedding_dimensions' ] = st.session_state.get( 'embeddings_dimensions',
			0 )
	
	if 'embedding_dimensions' in st.session_state and 'embeddings_dimensions' not in st.session_state:
		st.session_state[ 'embeddings_dimensions' ] = st.session_state.get( 'embedding_dimensions',
			0 )
	
	if 'embeddings_encoding_format' in st.session_state and 'embedding_encoding_format' not in st.session_state:
		st.session_state[ 'embedding_encoding_format' ] = st.session_state.get(
			'embeddings_encoding_format', '' )
	
	if 'embedding_encoding_format' in st.session_state and 'embeddings_encoding_format' not in st.session_state:
		st.session_state[ 'embeddings_encoding_format' ] = st.session_state.get(
			'embedding_encoding_format', '' )
	
	if 'embeddings_chunk_size' in st.session_state and 'embedding_chunk_size' not in st.session_state:
		st.session_state[ 'embedding_chunk_size' ] = st.session_state.get( 'embeddings_chunk_size',
			0 )
	
	if 'embedding_chunk_size' in st.session_state and 'embeddings_chunk_size' not in st.session_state:
		st.session_state[ 'embeddings_chunk_size' ] = st.session_state.get( 'embedding_chunk_size',
			0 )
	
	if 'embeddings_overlap_amount' in st.session_state and 'embedding_chunk_overlap' not in st.session_state:
		st.session_state[ 'embedding_chunk_overlap' ] = st.session_state.get(
			'embeddings_overlap_amount', 0 )
	
	if 'embedding_chunk_overlap' in st.session_state and 'embeddings_overlap_amount' not in st.session_state:
		st.session_state[ 'embeddings_overlap_amount' ] = st.session_state.get(
			'embedding_chunk_overlap', 0 )
	
	if not isinstance( st.session_state.get( 'embedding_chunks' ), list ):
		st.session_state[ 'embedding_chunks' ] = [ ]
	
	if not isinstance( st.session_state.get( 'embedding_vectors' ), list ):
		st.session_state[ 'embedding_vectors' ] = [ ]
	
	if not isinstance( st.session_state.get( 'embedding_metrics' ), dict ):
		st.session_state[ 'embedding_metrics' ] = { }
	
	if not isinstance( st.session_state.get( 'embedding_usage' ), dict ):
		st.session_state[ 'embedding_usage' ] = { }
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	emb_left, emb_center, emb_right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with emb_center:
		st.subheader( '🔢 Embeddings', help=get_embedding_help( 'EMBEDDINGS_API' ) )
		st.divider( )
		
		with st.expander( label='Configuration', icon='🎚️', expanded=False, width='stretch' ):
			emb_c1, emb_c2, emb_c3, emb_c4, emb_c5 = st.columns(
				[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
			
			# --------- Model --------
			with emb_c1:
				model_options = get_embedding_options( embedding, 'model_options' )
				model_options = [ str( item ) for item in model_options if str( item ).strip( ) ]
				st.selectbox( label='Embedding Model', options=model_options,
					help='Required. Embedding model used by the selected provider.',
					key='embedding_model', index=None, placeholder='Options' )
			
			# --------- Encoding --------
			with emb_c2:
				encoding_options = get_embedding_options( embedding, 'encoding_options',
					[ 'float', 'base64' ] )
				encoding_options = [ str( item ) for item in encoding_options if
				                     str( item ).strip( ) ]
				st.selectbox( label='Encoding Format', options=encoding_options,
					key='embedding_encoding_format',
					help='Optional. Format returned by the embedding provider.',
					index=None, placeholder='Options' )
				st.session_state[ 'embeddings_encoding_format' ] = st.session_state.get(
					'embedding_encoding_format', '' )
			
			# --------- Dimensions --------
			with emb_c3:
				st.slider( label='Dimensions', min_value=0, max_value=4096,
					value=int( st.session_state.get( 'embedding_dimensions', 0 ) or 0 ),
					step=1, key='embedding_dimensions',
					help='Optional. Embedding output dimensions when supported.' )
				st.session_state[ 'embeddings_dimensions' ] = st.session_state.get(
					'embedding_dimensions', 0 )
			
			# --------- Chunk Size --------
			with emb_c4:
				st.slider( label='Chunk Size', min_value=0, max_value=8000,
					value=int( st.session_state.get( 'embedding_chunk_size', 0 ) or 0 ),
					step=50, key='embedding_chunk_size',
					help='Maximum words/tokens per chunk. Zero embeds the full input.' )
				st.session_state[ 'embeddings_chunk_size' ] = st.session_state.get(
					'embedding_chunk_size', 0 )
			
			# --------- Overlap --------
			with emb_c5:
				st.slider( label='Overlap', min_value=0, max_value=2000,
					value=int( st.session_state.get( 'embedding_chunk_overlap', 0 ) or 0 ),
					step=25, key='embedding_chunk_overlap',
					help='Overlap between adjacent chunks.' )
				st.session_state[ 'embeddings_overlap_amount' ] = st.session_state.get(
					'embedding_chunk_overlap', 0 )
		
		with st.expander( label='Source Text', icon='📝', expanded=True, width='stretch' ):
			source_text = st.text_area( label='Input Text', key='embeddings_input_text',
				height=240, width='stretch',
				placeholder='Paste text to embed, or upload a text-compatible file below.' )
			st.session_state[ 'embedding_input' ] = source_text
			st.session_state[ 'embedding_text' ] = source_text
			
			uploaded_embedding_file = st.file_uploader( label='Upload Text File',
				type=[ 'txt', 'md', 'csv', 'json', 'py', 'cs', 'sql', 'xml', 'html' ],
				accept_multiple_files=False, key='embedding_file_uploader' )
			
			if uploaded_embedding_file is not None:
				try:
					file_text = uploaded_embedding_file.getvalue( ).decode( 'utf-8',
						errors='ignore' )
					st.session_state[ 'embeddings_input_text' ] = file_text
					st.session_state[ 'embedding_input' ] = file_text
					st.session_state[ 'embedding_text' ] = file_text
					st.success( f'Loaded {uploaded_embedding_file.name}.' )
				except Exception as exc:
					st.error( f'Could not read uploaded file: {exc}' )
		
		action_c1, action_c2 = st.columns( [ 0.50, 0.50 ] )
		
		with action_c1:
			if st.button( 'Create Embeddings', key='create_embeddings', width='stretch' ):
				with st.spinner( 'Creating embeddings…' ):
					try:
						source_text = normalize_embedding_text(
							st.session_state.get( 'embeddings_input_text', '' ) )
						
						if not source_text:
							st.warning( 'Enter text before creating embeddings.' )
						
						elif not st.session_state.get( 'embedding_model' ):
							st.warning( 'Select an embedding model before creating embeddings.' )
						
						else:
							chunk_size = int(
								st.session_state.get( 'embedding_chunk_size', 0 ) or 0 )
							overlap = int(
								st.session_state.get( 'embedding_chunk_overlap', 0 ) or 0 )
							chunks = chunk_embedding_text( source_text, chunk_size, overlap )
							
							if not chunks:
								st.warning( 'No chunks were created from the source text.' )
							else:
								raw_result = call_embeddings_create( chunks )
								vectors = normalize_embedding_vectors( raw_result )
								usage = extract_embedding_usage( raw_result )
								df_embeddings = build_embeddings_dataframe( chunks, vectors )
								metrics = build_embedding_metrics( source_text, chunks, vectors,
									usage )
								
								st.session_state[ 'embedding_results' ] = raw_result
								st.session_state[ 'embedding_chunks' ] = chunks
								st.session_state[ 'embeddings_chunks' ] = chunks
								st.session_state[ 'embedding_vectors' ] = vectors
								st.session_state[ 'embeddings' ] = vectors
								st.session_state[ 'embedding_dataframe' ] = df_embeddings
								st.session_state[ 'embeddings_df' ] = df_embeddings
								st.session_state[ 'embedding_metrics' ] = metrics
								st.session_state[ 'embedding_usage' ] = usage
								
								update_embedding_usage(
									getattr( embedding, 'response', None ) or raw_result )
								st.success( 'Embeddings created successfully.' )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Embedding creation failed: {err.info}' )
		
		with action_c2:
			if st.button( 'Reset All', key='reset_embeddings_all', width='stretch',
					on_click=reset_embeddings_all ):
				st.rerun( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		metrics = st.session_state.get( 'embedding_metrics', { } )
		if isinstance( metrics, dict ) and len( metrics ) > 0:
			render_embedding_metrics( metrics )
		
		df_embeddings = st.session_state.get( 'embedding_dataframe', pd.DataFrame( ) )
		if isinstance( df_embeddings, pd.DataFrame ) and not df_embeddings.empty:
			st.subheader( 'Embedding Output' )
			st.data_editor( df_embeddings, use_container_width=True, hide_index=True,
				key='embedding_dataframe_view' )
		
		chunks = st.session_state.get( 'embedding_chunks', [ ] )
		if isinstance( chunks, list ) and len( chunks ) > 0:
			with st.expander( label='Chunks', icon='🧩', expanded=False, width='stretch' ):
				df_chunks = pd.DataFrame(
					[
							{
									'ChunkIndex': index + 1,
									'Text': chunk,
									'Tokens': count_tokens( chunk ) if 'count_tokens' in globals( )
									else len( chunk.split( ) ),
							}
							for index, chunk in enumerate( chunks )
					]
				)
				st.data_editor( df_chunks, use_container_width=True, hide_index=True,
					key='embedding_chunks_view' )
		
		usage = st.session_state.get( 'embedding_usage', { } )
		if isinstance( usage, dict ) and len( usage ) > 0:
			with st.expander( label='Embedding Usage', icon='📊', expanded=False,
					width='stretch' ):
				st.json( usage )

# ======================================================================================
# FILES MODE
# ======================================================================================
elif mode == 'Files':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	files = get_files_module( provider_name )
	
	# ------------------------------------------------------------------
	# Files Mode Helpers
	# ------------------------------------------------------------------
	def get_files_help( name: str, fallback: str = '' ) -> str:
		"""
			
			Purpose:
			--------
			Return Files mode help text from config.py without failing when a constant is absent.
		
			Parameters:
			-----------
			name (str): Config attribute name.
			fallback (str): Fallback help text.
		
			Returns:
			--------
			str: Help text value.
			
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_files_options( instance: Any, attr_name: str,
			fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""
			
			Purpose:
			--------
			Return list-like option values from a Files wrapper property.
		
			Parameters:
			-----------
			instance (Any): Provider Files wrapper instance.
			attr_name (str): Property or attribute name to inspect.
			fallback (Optional[List[Any]]): Fallback option values.
		
			Returns:
			--------
			List[Any]: Option values safe for Streamlit controls.
			
		"""
		values = getattr( instance, attr_name, None )
		if callable( values ):
			try:
				values = values( )
			except Exception:
				values = None
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if isinstance( values, list ):
			return values
		
		return fallback or [ ]
	
	def call_files_method( method_names: List[ str ],
			kwargs: Optional[ Dict[ str, Any ] ] = None ) -> Any:
		"""
			
			Purpose:
			--------
			Call the first compatible provider Files wrapper method from an ordered method list.
		
			Parameters:
			-----------
			method_names (List[str]): Ordered method names to try.
			kwargs (Optional[Dict[str, Any]]): Keyword arguments for the provider method.
		
			Returns:
			--------
			Any: Provider method result.
			
		"""
		kwargs = kwargs or { }
		
		for method_name in method_names:
			method = getattr( files, method_name, None )
			if callable( method ):
				try:
					return method( **kwargs )
				except TypeError:
					clean_kwargs = {
							key: value
							for key, value in kwargs.items( )
							if value is not None and value != '' and value != [ ]
					}
					try:
						return method( **clean_kwargs )
					except TypeError:
						if len( clean_kwargs ) == 1:
							return method( list( clean_kwargs.values( ) )[ 0 ] )
						raise
		
		raise AttributeError( f'Provider "{provider_name}" does not expose any Files method from: '
			f'{", ".join( method_names )}.' )
	
	def save_uploaded_file_for_api( uploaded_file: Any ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Save a Streamlit uploaded file to a temporary path for provider Files wrappers.
		
			Parameters:
			-----------
			uploaded_file (Any): Streamlit uploaded file object.
		
			Returns:
			--------
			Optional[str]: Temporary file path or None.
			
		"""
		if uploaded_file is None:
			return None
		
		if 'save_temp' in globals( ):
			try:
				return save_temp( uploaded_file )
			except Exception:
				pass
		
		try:
			suffix = Path( uploaded_file.name ).suffix or '.tmp'
			with tempfile.NamedTemporaryFile( delete=False, suffix=suffix ) as tmp:
				tmp.write( uploaded_file.getvalue( ) )
				return tmp.name
		except Exception:
			return None
	
	def normalize_file_id( file_object: Any ) -> str:
		"""
			
			Purpose:
			--------
			Extract a provider file identifier from a provider response object.
		
			Parameters:
			-----------
			file_object (Any): Provider file object, dictionary, or string.
		
			Returns:
			--------
			str: File identifier.
			
		"""
		if file_object is None:
			return ''
		
		if isinstance( file_object, str ):
			return file_object
		
		if isinstance( file_object, dict ):
			for key in [ 'id', 'name', 'file_id', 'uri' ]:
				if key in file_object and file_object.get( key ):
					return str( file_object.get( key ) )
		
		for attr_name in [ 'id', 'name', 'file_id', 'uri' ]:
			if hasattr( file_object, attr_name ):
				value = getattr( file_object, attr_name )
				if value:
					return str( value )
		
		return str( file_object )
	
	def normalize_files_list( result: Any ) -> List[ Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			Normalize provider file listing results into table rows.
		
			Parameters:
			-----------
			result (Any): Provider list response.
		
			Returns:
			--------
			List[Dict[str, Any]]: File table rows.
			
		"""
		if result is None:
			return [ ]
		
		if hasattr( result, 'data' ):
			items = getattr( result, 'data' )
		elif isinstance( result, dict ) and isinstance( result.get( 'data' ), list ):
			items = result.get( 'data' )
		elif isinstance( result, list ):
			items = result
		else:
			items = [ result ]
		
		rows = [ ]
		for item in items:
			if item is None:
				continue
			
			if isinstance( item, dict ):
				file_id = item.get( 'id' ) or item.get( 'name' ) or item.get( 'file_id' ) \
				          or item.get( 'uri' ) or ''
				filename = item.get( 'filename' ) or item.get( 'display_name' ) \
				           or item.get( 'name' ) or ''
				purpose = item.get( 'purpose' ) or item.get( 'files_purpose' ) \
				          or item.get( 'mime_type' ) or ''
				created = item.get( 'created_at' ) or item.get( 'create_time' ) or ''
				size = item.get( 'bytes' ) or item.get( 'size_bytes' ) or item.get( 'size' ) or ''
			
			else:
				file_id = getattr( item, 'id', None ) or getattr( item, 'name', None ) \
				          or getattr( item, 'file_id', None ) or getattr( item, 'uri', '' )
				filename = getattr( item, 'filename', None ) or getattr( item, 'display_name',
					None ) \
				           or getattr( item, 'name', '' )
				purpose = getattr( item, 'purpose', None ) or getattr( item, 'files_purpose', None ) \
				          or getattr( item, 'mime_type', '' )
				created = getattr( item, 'created_at', None ) or getattr( item, 'create_time', '' )
				size = getattr( item, 'bytes', None ) or getattr( item, 'size_bytes', None ) \
				       or getattr( item, 'size', '' )
			
			rows.append(
				{
						'id': str( file_id or '' ),
						'filename': str( filename or '' ),
						'purpose': str( purpose or '' ),
						'created': str( created or '' ),
						'size': str( size or '' ),
				}
			)
		
		return rows
	
	def refresh_files_table( ) -> List[ Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			List provider files and persist normalized rows to session state.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[Dict[str, Any]]: File table rows.
			
		"""
		result = call_files_method( [ 'list', 'list_files', 'files_list' ] )
		rows = normalize_files_list( result )
		st.session_state[ 'files_table' ] = rows
		return rows
	
	def upload_provider_file( uploaded_file: Any, purpose: Optional[ str ] = None ) -> Any:
		"""
			
			Purpose:
			--------
			Upload a file through the selected provider Files wrapper.
		
			Parameters:
			-----------
			uploaded_file (Any): Streamlit uploaded file object.
			purpose (Optional[str]): Provider file purpose.
		
			Returns:
			--------
			Any: Provider upload result.
			
		"""
		path = save_uploaded_file_for_api( uploaded_file )
		if not path:
			raise ValueError( 'Could not create a temporary file for upload.' )
		
		kwargs = {
				'path': path,
				'file_path': path,
				'filepath': path,
				'purpose': purpose,
				'mime_type': getattr( uploaded_file, 'type', None ),
				'display_name': getattr( uploaded_file, 'name', None ),
		}
		
		return call_files_method( [ 'upload_file', 'upload', 'files_upload', 'create' ], kwargs )
	
	def retrieve_provider_file( file_id: str ) -> Any:
		"""
			
			Purpose:
			--------
			Retrieve provider file metadata by identifier.
		
			Parameters:
			-----------
			file_id (str): Provider file identifier.
		
			Returns:
			--------
			Any: Provider retrieve result.
			
		"""
		kwargs = {
				'file_id': file_id,
				'id': file_id,
				'name': file_id,
		}
		
		return call_files_method( [ 'retrieve', 'retrieve_file', 'get', 'get_file', 'files_retrieve' ],
			kwargs )
	
	def delete_provider_file( file_id: str ) -> Any:
		"""
			
			Purpose:
			--------
			Delete provider file by identifier.
		
			Parameters:
			-----------
			file_id (str): Provider file identifier.
		
			Returns:
			--------
			Any: Provider delete result.
			
		"""
		kwargs = {
				'file_id': file_id,
				'id': file_id,
				'name': file_id,
		}
		
		return call_files_method( [ 'delete', 'delete_file', 'files_delete', 'remove' ], kwargs )
	
	def clear_files_outputs( ) -> None:
		"""
			
			Purpose:
			--------
			Clear Files mode output and transient selection state.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'files_metadata' ] = { }
		st.session_state[ 'files_results' ] = None
		st.session_state[ 'files_delete_result' ] = { }
		st.session_state[ 'files_last_answer' ] = ''
	
	# ------------------------------------------------------------------
	# Session Safety
	# ------------------------------------------------------------------
	if not isinstance( st.session_state.get( 'files_table' ), list ):
		st.session_state[ 'files_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'files_metadata' ), dict ):
		st.session_state[ 'files_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'files_delete_result' ), dict ):
		st.session_state[ 'files_delete_result' ] = { }
	
	if not isinstance( st.session_state.get( 'files_uploaded' ), list ):
		st.session_state[ 'files_uploaded' ] = [ ]
	
	if not isinstance( st.session_state.get( 'files_messages' ), list ):
		st.session_state[ 'files_messages' ] = [ ]
	
	if 'files_manual_id' not in st.session_state:
		st.session_state[ 'files_manual_id' ] = ''
	
	if 'files_type' not in st.session_state:
		st.session_state[ 'files_type' ] = ''
	
	if 'files_id' not in st.session_state:
		st.session_state[ 'files_id' ] = ''
	
	if 'files_url' not in st.session_state:
		st.session_state[ 'files_url' ] = ''
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'files_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📁 Files API', help=get_files_help( 'FILES_API' ) )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			
			with st.expander( label='File Management', icon='📂', expanded=False,
					width='stretch' ):
				mgmt_c1, mgmt_c2, mgmt_c3, mgmt_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				# ---------- Purpose ------------
				with mgmt_c1:
					purpose_options = get_files_options( files, 'purpose_options',
						[ 'assistants', 'batch', 'fine-tune', 'vision', 'user_data' ] )
					purpose_options = [ str( item ) for item in purpose_options if
					                    str( item ).strip( ) ]
					st.selectbox( label='Purpose', options=purpose_options,
						key='files_purpose', index=None, placeholder='Options',
						help='Optional provider file purpose.' )
				
				# ---------- File Type ------------
				with mgmt_c2:
					st.selectbox( label='File Type',
						options=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json',
						          'csv', 'xlsx', 'xls' ],
						key='files_type', index=None, placeholder='Options',
						help='Optional local filter for uploaded file types.' )
				
				# ---------- Manual ID ------------
				with mgmt_c3:
					st.text_input( label='Manual File ID', key='files_manual_id',
						help='Optional. Paste a provider file ID/name for retrieve or delete.',
						width='stretch' )
				
				# ---------- Selected ID ------------
				with mgmt_c4:
					table_rows = st.session_state.get( 'files_table', [ ] )
					file_options = [
							row.get( 'id', '' )
							for row in table_rows
							if isinstance( row, dict ) and row.get( 'id', '' )
					]
					
					st.selectbox( label='Selected File', options=file_options,
						key='files_selected_id', index=None, placeholder='Options',
						help='File selected from the latest provider list.' )
			
			with st.expander( label='Request Settings', icon='⚙️', expanded=False,
					width='stretch' ):
				req_c1, req_c2, req_c3, req_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				with req_c1:
					st.selectbox( label='Model',
						options=get_files_options( files, 'model_options', [ ] ),
						key='files_model', index=None, placeholder='Options',
						help='Optional provider model for file-aware operations.' )
				
				with req_c2:
					st.slider( label='Max Tokens', min_value=0, max_value=100000,
						step=500, key='files_max_tokens',
						help='Optional max tokens for file-aware model calls.' )
				
				with req_c3:
					st.slider( label='Temperature', min_value=0.0, max_value=2.0,
						step=0.01, key='files_temperature',
						help='Optional temperature for file-aware model calls.' )
				
				with req_c4:
					st.selectbox( label='Response Format',
						options=get_files_options( files, 'format_options', [ ] ),
						key='files_response_format', index=None, placeholder='Options',
						help='Optional response format.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		upload_tab, list_tab, retrieve_tab, delete_tab = st.tabs(
			[ 'Upload', 'List', 'Retrieve', 'Delete' ] )
		
		with upload_tab:
			allowed_types = [ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json',
			                  'csv', 'xlsx', 'xls' ]
			uploaded_file = st.file_uploader( label='Upload File',
				type=allowed_types, accept_multiple_files=False, key='files_uploader' )
			
			if uploaded_file is not None:
				st.caption( f'Selected: {uploaded_file.name}' )
			
			if st.button( 'Upload File', key='files_upload_button', width='stretch' ):
				with st.spinner( 'Uploading file…' ):
					try:
						if uploaded_file is None:
							st.warning( 'Select a file before uploading.' )
						else:
							result = upload_provider_file(
								uploaded_file=uploaded_file,
								purpose=st.session_state.get( 'files_purpose' ) or None )
							file_id = normalize_file_id( result )
							
							st.session_state[ 'files_results' ] = result
							st.session_state[ 'files_selected_id' ] = file_id
							st.session_state[ 'files_uploaded' ].append(
								{
										'id': file_id,
										'filename': uploaded_file.name,
										'provider': provider_name,
								}
							)
							
							st.success( f'Uploaded file: {file_id}' )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Upload failed: {err.info}' )
		
		with list_tab:
			list_c1, list_c2 = st.columns( [ 0.50, 0.50 ] )
			
			with list_c1:
				if st.button( 'List Files', key='files_list_button', width='stretch' ):
					with st.spinner( 'Listing files…' ):
						try:
							rows = refresh_files_table( )
							st.success( f'Loaded {len( rows )} file record(s).' )
						except Exception as exc:
							st.session_state[ 'files_table' ] = [ ]
							err = Error( exc )
							st.error( f'List files failed: {err.info}' )
			
			with list_c2:
				if st.button( 'Clear Outputs', key='files_clear_outputs', width='stretch',
						on_click=clear_files_outputs ):
					st.rerun( )
			
			df_files = pd.DataFrame( st.session_state.get( 'files_table', [ ] ) )
			if not df_files.empty:
				st.data_editor( df_files, use_container_width=True, hide_index=True,
					key='files_table_view' )
			else:
				st.info( 'No file records loaded yet.' )
		
		with retrieve_tab:
			retrieve_id = st.session_state.get( 'files_selected_id' ) \
			              or st.session_state.get( 'files_manual_id' )
			
			st.text_input( label='Retrieve File ID', key='files_retrieve_id',
				value=retrieve_id or '',
				help='Provider file ID/name to retrieve.',
				width='stretch' )
			
			if st.button( 'Retrieve File', key='files_retrieve_button', width='stretch' ):
				with st.spinner( 'Retrieving file metadata…' ):
					try:
						file_id = st.session_state.get( 'files_retrieve_id', '' ).strip( )
						
						if not file_id:
							st.warning( 'Select or enter a file ID before retrieving.' )
						else:
							result = retrieve_provider_file( file_id )
							st.session_state[ 'files_metadata' ] = result if isinstance( result,
								dict ) else {
									'result': str( result )
							}
							st.session_state[ 'files_results' ] = result
							st.success( 'File metadata retrieved.' )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Retrieve failed: {err.info}' )
			
			if st.session_state.get( 'files_metadata' ):
				st.json( st.session_state.get( 'files_metadata' ) )
		
		with delete_tab:
			delete_id = st.session_state.get( 'files_selected_id' ) \
			            or st.session_state.get( 'files_manual_id' )
			
			st.text_input( label='Delete File ID', key='files_delete_id',
				value=delete_id or '',
				help='Provider file ID/name to delete.',
				width='stretch' )
			
			confirm_delete = st.checkbox( 'Confirm Delete', key='files_confirm_delete' )
			
			if st.button( 'Delete File', key='files_delete_button', width='stretch',
					disabled=not confirm_delete ):
				with st.spinner( 'Deleting file…' ):
					try:
						file_id = st.session_state.get( 'files_delete_id', '' ).strip( )
						
						if not file_id:
							st.warning( 'Select or enter a file ID before deleting.' )
						else:
							result = delete_provider_file( file_id )
							st.session_state[ 'files_delete_result' ] = result if isinstance(
								result, dict ) else { 'result': str( result ) }
							st.success( f'Delete request completed for: {file_id}' )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Delete failed: {err.info}' )
			
			if st.session_state.get( 'files_delete_result' ):
				st.json( st.session_state.get( 'files_delete_result' ) )

# ======================================================================================
# VECTORSTORES MODE
# ======================================================================================
elif mode == 'Vector Stores':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	
	# ------------------------------------------------------------------
	# Vector Store Helpers
	# ------------------------------------------------------------------
	def get_storage_help( name: str, fallback: str = '' ) -> str:
		"""
			
			Purpose:
			--------
			Return storage mode help text from config.py without failing when a constant is
			absent.
		
			Parameters:
			-----------
			name (str): Config attribute name.
			fallback (str): Fallback help text.
		
			Returns:
			--------
			str: Help text value.
			
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_storage_options( instance: Any, attr_name: str,
			fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""
			
			Purpose:
			--------
			Return list-like option values from a storage wrapper property.
		
			Parameters:
			-----------
			instance (Any): Provider storage wrapper instance.
			attr_name (str): Property or attribute name to inspect.
			fallback (Optional[List[Any]]): Fallback option values.
		
			Returns:
			--------
			List[Any]: Option values safe for Streamlit controls.
			
		"""
		values = getattr( instance, attr_name, None )
		if callable( values ):
			try:
				values = values( )
			except Exception:
				values = None
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if isinstance( values, list ):
			return values
		
		return fallback or [ ]
	
	def parse_storage_json( value: Any, label: str = 'JSON' ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Parse optional JSON text into a dictionary for storage wrapper calls.
		
			Parameters:
			-----------
			value (Any): JSON text value.
			label (str): User-facing label for warning messages.
		
			Returns:
			--------
			Dict[str, Any]: Parsed JSON dictionary or an empty dictionary.
			
		"""
		raw = str( value or '' ).strip( )
		if not raw:
			return { }
		
		try:
			import json
			
			parsed = json.loads( raw )
			if isinstance( parsed, dict ):
				return parsed
			
			st.warning( f'{label} must be a JSON object.' )
			return { }
		except Exception as exc:
			st.warning( f'{label} is not valid JSON: {exc}' )
			return { }
	
	def parse_storage_ids( value: Any ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Parse comma-delimited file or store identifiers.
		
			Parameters:
			-----------
			value (Any): Raw identifier text.
		
			Returns:
			--------
			List[str]: Parsed identifiers.
			
		"""
		raw = str( value or '' )
		return [ item.strip( ) for item in raw.split( ',' ) if item.strip( ) ]
	
	def call_storage_method( instance: Any, method_names: List[ str ],
			kwargs: Optional[ Dict[ str, Any ] ] = None ) -> Any:
		"""
			
			Purpose:
			--------
			Call the first compatible storage wrapper method from an ordered method list.
		
			Parameters:
			-----------
			instance (Any): Storage wrapper instance.
			method_names (List[str]): Ordered method names to try.
			kwargs (Optional[Dict[str, Any]]): Keyword arguments for the method.
		
			Returns:
			--------
			Any: Provider storage method result.
			
		"""
		kwargs = kwargs or { }
		for method_name in method_names:
			method = getattr( instance, method_name, None )
			if callable( method ):
				try:
					return method( **kwargs )
				except TypeError:
					clean_kwargs = {
							key: value
							for key, value in kwargs.items( )
							if value is not None and value != '' and value != [ ]
					}
					try:
						return method( **clean_kwargs )
					except TypeError:
						if len( clean_kwargs ) == 1:
							return method( list( clean_kwargs.values( ) )[ 0 ] )
						raise
		
		raise AttributeError(
			f'Provider "{provider_name}" does not expose a compatible method from: '
			f'{", ".join( method_names )}.' )
	
	def normalize_storage_object( value: Any ) -> Dict[ str, Any ]:
		"""
			
			Purpose:
			--------
			Normalize provider storage objects to dictionaries for rendering and session state.
		
			Parameters:
			-----------
			value (Any): Provider result object.
		
			Returns:
			--------
			Dict[str, Any]: Normalized result dictionary.
			
		"""
		if value is None:
			return { }
		
		if isinstance( value, dict ):
			return value
		
		result = { }
		for attr_name in [
				'id',
				'name',
				'display_name',
				'description',
				'status',
				'file_counts',
				'usage_bytes',
				'created_at',
				'expires_at',
				'metadata',
				'deleted',
		]:
			if hasattr( value, attr_name ):
				attr_value = getattr( value, attr_name )
				try:
					if hasattr( attr_value, 'model_dump' ):
						attr_value = attr_value.model_dump( )
				except Exception:
					pass
				result[ attr_name ] = attr_value
		
		if result:
			return result
		
		return { 'result': str( value ) }
	
	def normalize_storage_rows( result: Any ) -> List[ Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			Normalize provider storage list results into table rows.
		
			Parameters:
			-----------
			result (Any): Provider list result.
		
			Returns:
			--------
			List[Dict[str, Any]]: Normalized table rows.
			
		"""
		if result is None:
			return [ ]
		
		if hasattr( result, 'data' ):
			items = getattr( result, 'data' )
		elif isinstance( result, dict ) and isinstance( result.get( 'data' ), list ):
			items = result.get( 'data' )
		elif isinstance( result, dict ) and isinstance( result.get( 'stores' ), list ):
			items = result.get( 'stores' )
		elif isinstance( result, dict ) and isinstance( result.get( 'items' ), list ):
			items = result.get( 'items' )
		elif isinstance( result, list ):
			items = result
		else:
			items = [ result ]
		
		rows = [ ]
		for item in items:
			obj = normalize_storage_object( item )
			if not obj:
				continue
			
			store_id = obj.get( 'id' ) or obj.get( 'name' ) or obj.get( 'display_name' ) or ''
			store_name = obj.get( 'name' ) or obj.get( 'display_name' ) or obj.get( 'id' ) or ''
			file_counts = obj.get( 'file_counts', '' )
			usage_bytes = obj.get( 'usage_bytes', '' )
			status = obj.get( 'status', '' )
			
			rows.append( {
						'id': str( store_id or '' ),
						'name': str( store_name or '' ),
						'status': str( status or '' ),
						'file_counts': str( file_counts or '' ),
						'usage_bytes': str( usage_bytes or '' ),
				} )
		
		return rows
	
	def normalize_search_results( result: Any ) -> List[ Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			Normalize storage search results into dictionaries for display.
		
			Parameters:
			-----------
			result (Any): Provider search result.
		
			Returns:
			--------
			List[Dict[str, Any]]: Normalized search result rows.
			
		"""
		if result is None:
			return [ ]
		
		if hasattr( result, 'data' ):
			items = getattr( result, 'data' )
		elif isinstance( result, dict ) and isinstance( result.get( 'data' ), list ):
			items = result.get( 'data' )
		elif isinstance( result, dict ) and isinstance( result.get( 'results' ), list ):
			items = result.get( 'results' )
		elif isinstance( result, list ):
			items = result
		else:
			items = [ result ]
		
		rows = [ ]
		for item in items:
			if isinstance( item, dict ):
				rows.append( item )
			else:
				rows.append( normalize_storage_object( item ) )
		
		return rows
	
	def save_uploaded_storage_file( uploaded_file: Any ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Save an uploaded file to a temporary path for storage upload methods.
		
			Parameters:
			-----------
			uploaded_file (Any): Streamlit uploaded file object.
		
			Returns:
			--------
			Optional[str]: Temporary file path or None.
			
		"""
		if uploaded_file is None:
			return None
		
		if 'save_temp' in globals( ):
			try:
				return save_temp( uploaded_file )
			except Exception:
				pass
		
		try:
			suffix = Path( uploaded_file.name ).suffix or '.tmp'
			with tempfile.NamedTemporaryFile( delete=False, suffix=suffix ) as tmp:
				tmp.write( uploaded_file.getvalue( ) )
				return tmp.name
		except Exception:
			return None
	
	def get_selected_store_id( table_key: str, manual_key: str, selected_key: str ) -> str:
		"""
			
			Purpose:
			--------
			Return selected or manually entered storage identifier.
		
			Parameters:
			-----------
			table_key (str): Session key containing normalized table rows.
			manual_key (str): Session key containing manually entered identifier.
			selected_key (str): Session key containing selected identifier.
		
			Returns:
			--------
			str: Selected storage identifier.
			
		"""
		selected = st.session_state.get( selected_key, '' )
		manual = st.session_state.get( manual_key, '' )
		
		if isinstance( selected, str ) and selected.strip( ):
			return selected.strip( )
		
		if isinstance( manual, str ) and manual.strip( ):
			return manual.strip( )
		
		rows = st.session_state.get( table_key, [ ] )
		if isinstance( rows, list ) and len( rows ) > 0:
			first = rows[ 0 ]
			if isinstance( first, dict ):
				return str( first.get( 'id', '' ) or '' ).strip( )
		
		return ''
	
	def render_storage_table( rows: List[ Dict[ str, Any ] ], key: str ) -> None:
		"""
			
			Purpose:
			--------
			Render normalized storage rows as a dataframe.
		
			Parameters:
			-----------
			rows (List[Dict[str, Any]]): Storage table rows.
			key (str): Streamlit dataframe key.
		
			Returns:
			--------
			None
			
		"""
		df_rows = pd.DataFrame( rows or [ ] )
		if df_rows.empty:
			st.info( 'No storage records loaded yet.' )
			return
		
		st.data_editor( df_rows, use_container_width=True, hide_index=True, key=key )
	
	def render_storage_metadata( metadata: Dict[ str, Any ] ) -> None:
		"""
			
			Purpose:
			--------
			Render selected storage metadata.
		
			Parameters:
			-----------
			metadata (Dict[str, Any]): Storage metadata.
		
			Returns:
			--------
			None
			
		"""
		if not isinstance( metadata, dict ) or len( metadata ) == 0:
			st.info( 'No metadata loaded yet.' )
			return
		
		st.json( metadata )
	
	def render_storage_search_results( rows: List[ Dict[ str, Any ] ] ) -> None:
		"""
			
			Purpose:
			--------
			Render normalized storage search results.
		
			Parameters:
			-----------
			rows (List[Dict[str, Any]]): Search result rows.
		
			Returns:
			--------
			None
			
		"""
		if not isinstance( rows, list ) or len( rows ) == 0:
			st.info( 'No search results loaded yet.' )
			return
		
		df_results = pd.DataFrame( rows )
		st.data_editor( df_results, use_container_width=True, hide_index=True,
			key='stores_search_results_view' )
	
	def clear_vector_store_outputs( ) -> None:
		"""
			
			Purpose:
			--------
			Clear Vector Stores output state without clearing upstream mode configuration.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'stores_store_metadata' ] = { }
		st.session_state[ 'stores_search_results' ] = [ ]
		st.session_state[ 'stores_files_table' ] = [ ]
		st.session_state[ 'stores_batch_result' ] = { }
	
	# ------------------------------------------------------------------
	# Vector Stores Provider Guard
	# ------------------------------------------------------------------
	if provider_name == 'Gemini':
		st.warning(
			'Gemini storage is available under File Search Stores and Google Cloud Buckets.' )
		st.stop( )
	
	vector = get_vectorstores_module( provider_name )
	
	# ------------------------------------------------------------------
	# Session Safety
	# ------------------------------------------------------------------
	for key, default_value in {
			'stores_table': [ ],
			'stores_files_table': [ ],
			'stores_store_metadata': { },
			'stores_batch_result': { },
			'stores_search_results': [ ],
			'stores_messages': [ ],
	}.items( ):
		if key not in st.session_state or not isinstance( st.session_state.get( key ),
				type( default_value ) ):
			st.session_state[ key ] = default_value
	
	if 'stores_name' not in st.session_state:
		st.session_state[ 'stores_name' ] = ''
	
	if 'stores_id' not in st.session_state:
		st.session_state[ 'stores_id' ] = ''
	
	if 'stores_manual_id' not in st.session_state:
		st.session_state[ 'stores_manual_id' ] = ''
	
	if 'stores_description' not in st.session_state:
		st.session_state[ 'stores_description' ] = ''
	
	if 'stores_metadata' not in st.session_state:
		st.session_state[ 'stores_metadata' ] = ''
	
	if 'stores_query' not in st.session_state:
		st.session_state[ 'stores_query' ] = ''
	
	if 'stores_file_id' not in st.session_state:
		st.session_state[ 'stores_file_id' ] = ''
	
	if 'stores_file_ids_text' not in st.session_state:
		st.session_state[ 'stores_file_ids_text' ] = ''
	
	if 'stores_selected_id' not in st.session_state:
		st.session_state[ 'stores_selected_id' ] = ''
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( '🧊 Vector Stores', help=get_storage_help( 'VECTORSTORES_API' ) )
		st.divider( )
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(
				[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
			
			with ctrl_c1:
				st.text_input( label='Store Name', key='stores_name',
					help='Name used when creating a vector store.',
					width='stretch', placeholder='Enter store name' )
			
			with ctrl_c2:
				st.text_input( label='Manual Store ID', key='stores_manual_id',
					help='Optional. Paste a vector store or collection ID.',
					width='stretch' )
			
			with ctrl_c3:
				st.selectbox( label='Answer Model',
					options=get_storage_options( vector, 'model_options', [ ] ),
					key='stores_model', index=None, placeholder='Options',
					help='Optional. Model used for store-backed answers when supported.' )
			
			with ctrl_c4:
				st.slider( label='Max Tokens', min_value=0, max_value=100000,
					step=500, key='stores_max_tokens',
					help='Optional. Max tokens for store-backed answers.' )
			
			desc_c1, desc_c2 = st.columns( [ 0.50, 0.50 ], border=True, gap='xxsmall' )
			with desc_c1:
				st.text_area( label='Description', key='stores_description',
					height=80, width='stretch',
					help='Optional. Vector store description when supported.' )
			
			with desc_c2:
				st.text_area( label='Metadata JSON', key='stores_metadata',
					height=80, width='stretch',
					help='Optional. JSON object metadata for create/update calls.' )
		
		store_col, detail_col = st.columns( [ 0.50, 0.50 ], border=True, gap='medium' )
		with store_col:
			st.subheader( 'Store Lifecycle' )
			
			create_c1, create_c2 = st.columns( [ 0.50, 0.50 ] )
			with create_c1:
				if st.button( 'Create Store', key='create_vector_store', width='stretch' ):
					with st.spinner( 'Creating vector store…' ):
						try:
							name = st.session_state.get( 'stores_name', '' ).strip( )
							if not name:
								st.warning( 'Enter a vector store name before creating.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'create', 'create_store', 'create_collection' ],
									kwargs={
											'name': name,
											'description': st.session_state.get(
												'stores_description', '' ) or None,
											'metadata': parse_storage_json(
												st.session_state.get( 'stores_metadata', '' ),
												'Vector store metadata' ),
											'file_ids': parse_storage_ids(
												st.session_state.get( 'stores_file_ids_text',
													'' ) ),
									}
								)
								metadata = normalize_storage_object( result )
								st.session_state[ 'stores_store_metadata' ] = metadata
								st.session_state[ 'stores_id' ] = metadata.get( 'id', name )
								st.success( f'Created store: {st.session_state[ "stores_id" ]}' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Create vector store failed: {err.info}' )
			
			with create_c2:
				if st.button( 'List Stores', key='list_vector_stores', width='stretch' ):
					with st.spinner( 'Listing vector stores…' ):
						try:
							result = call_storage_method(
								instance=vector,
								method_names=[ 'list_stores', 'list', 'list_collections' ],
								kwargs={ 'limit': 100, 'order': 'desc' }
							)
							rows = normalize_storage_rows( result )
							st.session_state[ 'stores_table' ] = rows
							st.success( f'Loaded {len( rows )} store record(s).' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'List vector stores failed: {err.info}' )
			
			rows = st.session_state.get( 'stores_table', [ ] )
			store_ids = [
					row.get( 'id', '' )
					for row in rows
					if isinstance( row, dict ) and row.get( 'id', '' ) ]
			
			st.selectbox( label='Selected Store', options=store_ids,
				key='stores_selected_id', index=None, placeholder='Options',
				help='Store selected from latest list.' )
			
			selected_store_id = get_selected_store_id( table_key='stores_table',
				manual_key='stores_manual_id', selected_key='stores_selected_id' )
			
			retrieve_c1, retrieve_c2, retrieve_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
			with retrieve_c1:
				if st.button( 'Retrieve Store', key='retrieve_vector_store', width='stretch' ):
					with st.spinner( 'Retrieving vector store…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID before retrieving.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'retrieve', 'retrieve_store', 'get_collection' ],
									kwargs={ 'store_id': selected_store_id,
									         'id': selected_store_id }
								)
								metadata = normalize_storage_object( result )
								st.session_state[ 'stores_store_metadata' ] = metadata
								st.session_state[ 'stores_id' ] = selected_store_id
								st.success( 'Store metadata retrieved.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Retrieve vector store failed: {err.info}' )
			
			with retrieve_c2:
				if st.button( 'Update Store', key='update_vector_store', width='stretch' ):
					with st.spinner( 'Updating vector store…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID before updating.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'update', 'update_store', 'update_collection' ],
									kwargs={
											'store_id': selected_store_id,
											'id': selected_store_id,
											'name': st.session_state.get( 'stores_name',
												'' ) or None,
											'description': st.session_state.get(
												'stores_description', '' ) or None,
											'metadata': parse_storage_json(
												st.session_state.get( 'stores_metadata', '' ),
												'Vector store metadata' ),
									}
								)
								st.session_state[
									'stores_store_metadata' ] = normalize_storage_object( result )
								st.success( 'Store update submitted.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Update vector store failed: {err.info}' )
			
			with retrieve_c3:
				if st.button( 'Delete Store', key='delete_vector_store', width='stretch' ):
					with st.spinner( 'Deleting vector store…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID before deleting.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'delete', 'delete_store', 'delete_collection' ],
									kwargs={ 'store_id': selected_store_id,
									         'id': selected_store_id }
								)
								st.session_state[
									'stores_store_metadata' ] = normalize_storage_object( result )
								st.success( 'Delete request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Delete vector store failed: {err.info}' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			render_storage_table( st.session_state.get( 'stores_table', [ ] ),
				'vector_stores_table_view' )
		
		with detail_col:
			st.subheader( 'Selected Store Details' )
			render_storage_metadata( st.session_state.get( 'stores_store_metadata', { } ) )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			
			st.text_area( label='Search Query', key='stores_query',
				height=90, width='stretch',
				placeholder='Search this vector store or collection.' )
			
			search_c1, search_c2 = st.columns( [ 0.50, 0.50 ] )
			with search_c1:
				if st.button( 'Search Store', key='search_vector_store', width='stretch' ):
					with st.spinner( 'Searching store…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID before searching.' )
							elif not st.session_state.get( 'stores_query', '' ).strip( ):
								st.warning( 'Enter a search query first.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'search', 'search_store', 'query',
									               'query_collection' ],
									kwargs={
											'store_id': selected_store_id,
											'id': selected_store_id,
											'query': st.session_state.get( 'stores_query',
												'' ).strip( ),
									}
								)
								rows = normalize_search_results( result )
								st.session_state[ 'stores_search_results' ] = rows
								st.success( f'Returned {len( rows )} result(s).' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Store search failed: {err.info}' )
			
			with search_c2:
				st.button( label='Clear Outputs', key='clear_vector_store_outputs',
					width='stretch', on_click=clear_vector_store_outputs )
			
			render_storage_search_results( st.session_state.get( 'stores_search_results', [ ] ) )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		file_col, batch_col = st.columns( [ 0.50, 0.50 ], border=True, gap='medium' )
		
		with file_col:
			st.subheader( 'Store Files' )
			st.text_input( label='File ID', key='stores_file_id',
				help='OpenAI/Grok file ID to attach, list, or delete.',
				width='stretch' )
			
			file_op_c1, file_op_c2, file_op_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
			with file_op_c1:
				if st.button( 'Attach File', key='attach_vector_store_file', width='stretch' ):
					with st.spinner( 'Attaching file…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							elif not st.session_state.get( 'stores_file_id', '' ).strip( ):
								st.warning( 'Enter a file ID first.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'create_file', 'attach_file', 'add_file' ],
									kwargs={
											'store_id': selected_store_id,
											'id': selected_store_id,
											'file_id': st.session_state.get( 'stores_file_id',
												'' ).strip( ),
									}
								)
								st.session_state[
									'stores_batch_result' ] = normalize_storage_object( result )
								st.success( 'File attach request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Attach file failed: {err.info}' )
			
			with file_op_c2:
				if st.button( 'List Files', key='list_vector_store_files', width='stretch' ):
					with st.spinner( 'Listing store files…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'list_files', 'files', 'list_store_files' ],
									kwargs={ 'store_id': selected_store_id, 'id': selected_store_id,
									         'limit': 100, 'order': 'desc' }
								)
								rows = normalize_storage_rows( result )
								st.session_state[ 'stores_files_table' ] = rows
								st.success( f'Loaded {len( rows )} store file record(s).' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'List store files failed: {err.info}' )
			
			with file_op_c3:
				if st.button( 'Delete File', key='delete_vector_store_file', width='stretch' ):
					with st.spinner( 'Deleting store file…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							elif not st.session_state.get( 'stores_file_id', '' ).strip( ):
								st.warning( 'Enter a file ID first.' )
							else:
								result = call_storage_method(
									instance=vector,
									method_names=[ 'delete_file', 'remove_file' ],
									kwargs={
											'store_id': selected_store_id,
											'id': selected_store_id,
											'file_id': st.session_state.get( 'stores_file_id',
												'' ).strip( ),
									}
								)
								st.session_state[
									'stores_batch_result' ] = normalize_storage_object( result )
								st.success( 'Store file delete request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Delete store file failed: {err.info}' )
			
			render_storage_table( st.session_state.get( 'stores_files_table', [ ] ),
				'vector_store_files_table_view' )
		
		with batch_col:
			st.subheader( 'Batch / Upload' )
			st.text_area( label='File IDs', key='stores_file_ids_text',
				height=80, width='stretch',
				placeholder='file_abc,file_def,file_xyz' )
			
			uploaded_store_file = st.file_uploader( label='Upload File to Store',
				type=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json', 'csv' ],
				key='stores_file_upload' )
			
			batch_c1, batch_c2 = st.columns( [ 0.50, 0.50 ] )
			with batch_c1:
				if st.button( 'Create Batch', key='create_vector_store_batch', width='stretch' ):
					with st.spinner( 'Creating file batch…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							else:
								file_ids = parse_storage_ids(
									st.session_state.get( 'stores_file_ids_text', '' ) )
								if not file_ids:
									st.warning( 'Enter one or more file IDs first.' )
								else:
									result = call_storage_method(
										instance=vector,
										method_names=[ 'create_file_batch', 'create_batch',
										               'batch' ],
										kwargs={ 'store_id': selected_store_id,
										         'id': selected_store_id,
										         'file_ids': file_ids }
									)
									st.session_state[
										'stores_batch_result' ] = normalize_storage_object( result )
									st.success( 'Batch request submitted.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Batch request failed: {err.info}' )
			
			with batch_c2:
				if st.button( 'Upload + Attach', key='upload_attach_vector_store_file',
						width='stretch' ):
					with st.spinner( 'Uploading and attaching file…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							elif uploaded_store_file is None:
								st.warning( 'Select a file first.' )
							else:
								path = save_uploaded_storage_file( uploaded_store_file )
								result = call_storage_method(
									instance=vector,
									method_names=[ 'upload_file', 'upload', 'files_upload' ],
									kwargs={ 'store_id': selected_store_id, 'id': selected_store_id,
									         'path': path, 'file_path': path }
								)
								st.session_state[
									'stores_batch_result' ] = normalize_storage_object( result )
								st.success( 'Upload request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Upload attach failed: {err.info}' )
			
			render_storage_metadata( st.session_state.get( 'stores_batch_result', { } ) )

# ======================================================================================
# FILE SEARCH STORES MODE
# ======================================================================================
elif mode == 'File Search Stores':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	
	if provider_name != 'Gemini':
		st.warning( 'File Search Stores are available for Gemini only.' )
		st.stop( )
	
	searcher = get_file_search_module( provider_name )
	
	def call_file_search_method( method_names: List[ str ], kwargs: Optional[ Dict[ str, Any ] ]=None ) -> Any:
		"""
			
			Purpose:
			--------
			Call the first compatible Gemini FileSearch wrapper method.
		
			Parameters:
			-----------
			method_names (List[str]): Ordered method names to try.
			kwargs (Optional[Dict[str, Any]]): Keyword arguments for the method.
		
			Returns:
			--------
			Any: Provider method result.
			
		"""
		kwargs = kwargs or { }
		
		for method_name in method_names:
			method = getattr( searcher, method_name, None )
			if callable( method ):
				try:
					return method( **kwargs )
				except TypeError:
					clean_kwargs = {
							key: value
							for key, value in kwargs.items( )
							if value is not None and value != '' and value != [ ]
					}
					try:
						return method( **clean_kwargs )
					except TypeError:
						if len( clean_kwargs ) == 1:
							return method( list( clean_kwargs.values( ) )[ 0 ] )
						raise
		
		raise AttributeError(
			f'Gemini FileSearch does not expose any method from: {", ".join( method_names )}.' )
	
	def clear_filestore_outputs( ) -> None:
		"""
			
			Purpose:
			--------
			Clear File Search Stores output state.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'filestore_results' ] = None
		st.session_state[ 'filestore_metadata' ] = { }
		st.session_state[ 'filestore_upload_result' ] = { }
	
	for key, default_value in {
			'filestore_table': [ ],
			'filestore_metadata': { },
			'filestore_upload_result': { },
	}.items( ):
		if key not in st.session_state or not isinstance( st.session_state.get( key ),
				type( default_value ) ):
			st.session_state[ key ] = default_value
	
	if 'filestore_name' not in st.session_state:
		st.session_state[ 'filestore_name' ] = ''
	
	if 'filestore_manual_id' not in st.session_state:
		st.session_state[ 'filestore_manual_id' ] = ''
	
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📦 File Search Stores', help=getattr( cfg, 'VECTORSTORES_API', '' ) )
		st.divider( )
		
		stores_left, stores_right = st.columns( [ 0.50, 0.50 ], border=True )
		
		with stores_left:
			
			with st.expander( label='Create', expanded=True ):
				st.text_input( label='New File Search Store Name', key='filestore_name',
					width='stretch' )
				
				if st.button( 'Create File Search Store', key='create_filestore',
						width='stretch' ):
					with st.spinner( 'Creating file search store…' ):
						try:
							name = st.session_state.get( 'filestore_name', '' ).strip( )
							if not name:
								st.warning( 'Enter a File Search Store name.' )
							else:
								result = call_file_search_method(
									[ 'create', 'create_store' ],
									{ 'name': name, 'display_name': name, 'store_id': name }
								)
								st.session_state[ 'filestore_metadata' ] = normalize_storage_object(
									result )
								st.success( f'Created File Search Store: {name}' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Create file search store failed: {err.info}' )
			
			with st.expander( label='Retrieve / Delete', expanded=True ):
				collections = getattr( searcher, 'collections', None )
				options = list( collections.items( ) ) if isinstance( collections, dict ) else [ ]
				option_labels = [ f'{name} — {store_id}' for name, store_id in options ]
				
				if option_labels:
					selected_label = st.selectbox( label='Select File Search Store',
						options=option_labels, key='filestore_select',
						index=None, placeholder='Options' )
					
					selected_id = ''
					for name, store_id in options:
						if f'{name} — {store_id}' == selected_label:
							selected_id = store_id
							break
					
					st.session_state[ 'filestore_selected_id' ] = selected_id
				else:
					st.info( 'No configured File Search Store collections found on the wrapper.' )
				
				st.text_input( label='Manual File Search Store ID', key='filestore_manual_id',
					width='stretch' )
				
				selected_store_id = st.session_state.get( 'filestore_selected_id', '' ) \
				                    or st.session_state.get( 'filestore_manual_id', '' )
				
				retr_c1, retr_c2 = st.columns( [ 0.50, 0.50 ] )
				with retr_c1:
					if st.button( 'Retrieve File Search Store', key='retrieve_filestore',
							width='stretch' ):
						with st.spinner( 'Retrieving file search store…' ):
							try:
								if not selected_store_id:
									st.warning( 'Select or enter a File Search Store ID.' )
								else:
									result = call_file_search_method(
										[ 'retrieve', 'retrieve_store', 'get' ],
										{ 'store_id': selected_store_id, 'id': selected_store_id,
										  'name': selected_store_id }
									)
									st.session_state[
										'filestore_metadata' ] = normalize_storage_object( result )
									st.success( 'File Search Store metadata retrieved.' )
							except Exception as exc:
								err = Error( exc )
								st.error( f'Retrieve failed: {err.info}' )
				
				with retr_c2:
					if st.button( 'Delete File Search Store', key='delete_filestore',
							width='stretch' ):
						with st.spinner( 'Deleting file search store…' ):
							try:
								if not selected_store_id:
									st.warning( 'Select or enter a File Search Store ID.' )
								else:
									result = call_file_search_method(
										[ 'delete', 'delete_store', 'remove' ],
										{ 'store_id': selected_store_id, 'id': selected_store_id,
										  'name': selected_store_id }
									)
									st.session_state[
										'filestore_metadata' ] = normalize_storage_object( result )
									st.success( 'Delete request completed.' )
							except Exception as exc:
								err = Error( exc )
								st.error( f'Delete failed: {err.info}' )
		
		with stores_right:
			st.subheader( 'Upload' )
			uploaded_file = st.file_uploader( label='Upload File to File Search Store',
				type=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json', 'csv' ],
				key='filestore_uploader' )
			
			target_store = st.session_state.get( 'filestore_selected_id', '' ) \
			               or st.session_state.get( 'filestore_manual_id', '' )
			
			if st.button( 'Upload File', key='upload_filestore_file', width='stretch' ):
				with st.spinner( 'Uploading file…' ):
					try:
						if uploaded_file is None:
							st.warning( 'Select a file first.' )
						else:
							path = save_uploaded_storage_file( uploaded_file )
							result = call_file_search_method(
								[ 'upload_file', 'upload', 'files_upload' ],
								{ 'path': path, 'file_path': path, 'store_id': target_store,
								  'id': target_store }
							)
							st.session_state[
								'filestore_upload_result' ] = normalize_storage_object( result )
							st.success( 'Upload request completed.' )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Upload failed: {err.info}' )
			
			if st.button( 'Clear Outputs', key='clear_filestore_outputs',
					width='stretch', on_click=clear_filestore_outputs ):
				st.rerun( )
			
			st.caption( 'Upload Result' )
			render_storage_metadata( st.session_state.get( 'filestore_upload_result', { } ) )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.subheader( 'File Search Store Metadata' )
		render_storage_metadata( st.session_state.get( 'filestore_metadata', { } ) )

# ======================================================================================
# GOOGLE CLOUD BUCKETS MODE
# ======================================================================================
elif mode == 'Google Cloud Buckets':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	
	if provider_name != 'Gemini':
		st.warning( 'Google Cloud Buckets are available for Gemini / Google Cloud only.' )
		st.stop( )
	
	buckets = get_cloud_buckets_module( provider_name )
	
	def call_bucket_method( method_names: List[ str ], kwargs: Optional[ Dict[ str, Any ] ]=None ) -> Any:
		"""
			
			Purpose:
			--------
			Call the first compatible Gemini CloudBuckets wrapper method.
		
			Parameters:
			-----------
			method_names (List[str]): Ordered method names to try.
			kwargs (Optional[Dict[str, Any]]): Keyword arguments for the method.
		
			Returns:
			--------
			Any: Provider method result.
			
		"""
		kwargs = kwargs or { }
		
		for method_name in method_names:
			method = getattr( buckets, method_name, None )
			if callable( method ):
				try:
					return method( **kwargs )
				except TypeError:
					clean_kwargs = {
							key: value
							for key, value in kwargs.items( )
							if value is not None and value != '' and value != [ ]
					}
					try:
						return method( **clean_kwargs )
					except TypeError:
						if len( clean_kwargs ) == 1:
							return method( list( clean_kwargs.values( ) )[ 0 ] )
						raise
		
		raise AttributeError(
			f'Gemini CloudBuckets does not expose any method from: {", ".join( method_names )}.'
		)
	
	def clear_bucket_outputs( ) -> None:
		"""
			
			Purpose:
			--------
			Clear Google Cloud Buckets output state.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
			
		"""
		st.session_state[ 'bucket_results' ] = None
		st.session_state[ 'bucket_metadata' ] = { }
		st.session_state[ 'bucket_upload_result' ] = { }
	
	for key, default_value in {
			'bucket_table': [ ],
			'bucket_metadata': { },
			'bucket_upload_result': { },
	}.items( ):
		if key not in st.session_state or not isinstance( st.session_state.get( key ),
				type( default_value ) ):
			st.session_state[ key ] = default_value
	
	if 'bucket_name' not in st.session_state:
		st.session_state[ 'bucket_name' ] = ''
	
	if 'bucket_manual_id' not in st.session_state:
		st.session_state[ 'bucket_manual_id' ] = ''
	
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '🧊 Google Cloud Buckets', help=getattr( cfg, 'VECTORSTORES_API', '' ) )
		st.divider( )
		
		project_id = st.session_state.get( 'google_cloud_project_id', '' ) \
		             or getattr( cfg, 'GOOGLE_CLOUD_PROJECT_ID', '' )
		location = st.session_state.get( 'google_cloud_location', '' ) \
		           or getattr( cfg, 'GOOGLE_CLOUD_LOCATION', '' )
		
		st.caption(
			f'Project: {project_id or "Not configured"} | Location: {location or "Not configured"}' )
		
		buckets_left, buckets_right = st.columns( [ 0.50, 0.50 ], border=True )
		with buckets_left:
			
			with st.expander( label='Create', expanded=True ):
				st.text_input( label='New Cloud Bucket Name', key='bucket_name',
					width='stretch' )
				
				if st.button( 'Create Cloud Bucket', key='create_bucket', width='stretch' ):
					with st.spinner( 'Creating cloud bucket…' ):
						try:
							name = st.session_state.get( 'bucket_name', '' ).strip( )
							if not name:
								st.warning( 'Enter a Cloud Bucket name.' )
							else:
								result = call_bucket_method(
									[ 'create', 'create_bucket' ],
									{ 'name': name, 'bucket_name': name,
									  'project_id': project_id, 'location': location }
								)
								st.session_state[ 'bucket_metadata' ] = normalize_storage_object(
									result )
								st.success( f'Created Cloud Bucket: {name}' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Create bucket failed: {err.info}' )
			
			with st.expander( label='Retrieve / Delete', expanded=True ):
				collections = getattr( buckets, 'collections', None )
				options = list( collections.items( ) ) if isinstance( collections, dict ) else [ ]
				option_labels = [ f'{name} — {bucket_id}' for name, bucket_id in options ]
				
				if option_labels:
					selected_label = st.selectbox( label='Select Cloud Bucket',
						options=option_labels, key='bucket_select',
						index=None, placeholder='Options' )
					
					selected_id = ''
					for name, bucket_id in options:
						if f'{name} — {bucket_id}' == selected_label:
							selected_id = bucket_id
							break
					
					st.session_state[ 'bucket_selected_id' ] = selected_id
				else:
					st.info( 'No configured Cloud Bucket collections found on the wrapper.' )
				
				st.text_input( label='Manual Cloud Bucket ID / Name', key='bucket_manual_id',
					width='stretch' )
				
				selected_bucket_id = st.session_state.get( 'bucket_selected_id', '' ) \
				                     or st.session_state.get( 'bucket_manual_id', '' )
				
				bucket_c1, bucket_c2 = st.columns( [ 0.50, 0.50 ] )
				
				with bucket_c1:
					if st.button( 'Retrieve Cloud Bucket', key='retrieve_bucket',
							width='stretch' ):
						with st.spinner( 'Retrieving cloud bucket…' ):
							try:
								if not selected_bucket_id:
									st.warning( 'Select or enter a Cloud Bucket ID.' )
								else:
									result = call_bucket_method(
										[ 'retrieve', 'retrieve_bucket', 'get' ],
										{ 'store_id': selected_bucket_id, 'id': selected_bucket_id,
										  'name': selected_bucket_id,
										  'bucket_name': selected_bucket_id,
										  'project_id': project_id, 'location': location }
									)
									st.session_state[
										'bucket_metadata' ] = normalize_storage_object( result )
									st.success( 'Cloud Bucket metadata retrieved.' )
							except Exception as exc:
								err = Error( exc )
								st.error( f'Retrieve bucket failed: {err.info}' )
				
				with bucket_c2:
					if st.button( 'Delete Cloud Bucket', key='delete_bucket',
							width='stretch' ):
						with st.spinner( 'Deleting cloud bucket…' ):
							try:
								if not selected_bucket_id:
									st.warning( 'Select or enter a Cloud Bucket ID.' )
								else:
									result = call_bucket_method(
										[ 'delete', 'delete_bucket', 'remove' ],
										{ 'store_id': selected_bucket_id, 'id': selected_bucket_id,
										  'name': selected_bucket_id,
										  'bucket_name': selected_bucket_id,
										  'project_id': project_id, 'location': location }
									)
									st.session_state[
										'bucket_metadata' ] = normalize_storage_object( result )
									st.success( 'Delete request completed.' )
							except Exception as exc:
								err = Error( exc )
								st.error( f'Delete bucket failed: {err.info}' )
		
		with buckets_right:
			st.subheader( 'Upload' )
			uploaded_file = st.file_uploader( label='Upload File to Cloud Bucket',
				type=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json', 'csv' ],
				key='bucket_uploader' )
			
			target_bucket = st.session_state.get( 'bucket_selected_id', '' ) \
			                or st.session_state.get( 'bucket_manual_id', '' )
			
			if st.button( 'Upload File', key='upload_bucket_file', width='stretch' ):
				with st.spinner( 'Uploading file…' ):
					try:
						if uploaded_file is None:
							st.warning( 'Select a file first.' )
						elif not target_bucket:
							st.warning( 'Select or enter a Cloud Bucket first.' )
						else:
							path = save_uploaded_storage_file( uploaded_file )
							result = call_bucket_method(
								[ 'upload_file', 'upload', 'files_upload' ],
								{ 'path': path, 'file_path': path,
								  'bucket_name': target_bucket, 'store_id': target_bucket,
								  'id': target_bucket, 'project_id': project_id,
								  'location': location }
							)
							st.session_state[ 'bucket_upload_result' ] = normalize_storage_object(
								result )
							st.success( 'Upload request completed.' )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Bucket upload failed: {err.info}' )
			
			if st.button( 'Clear Outputs', key='clear_bucket_outputs',
					width='stretch', on_click=clear_bucket_outputs ):
				st.rerun( )
			
			st.caption( 'Upload Result' )
			render_storage_metadata( st.session_state.get( 'bucket_upload_result', { } ) )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.subheader( 'Cloud Bucket Metadata' )
		render_storage_metadata( st.session_state.get( 'bucket_metadata', { } ) )
		
# ======================================================================================
# PROMPT ENGINEERING MODE
# ======================================================================================
elif mode == "Prompt Engineering":
	import sqlite3
	import math
	
	DB_PATH = 'stores/sqlite/datamodels/Data.db'
	TABLE = 'Prompts'
	PAGE_SIZE = 10
	
	# ------------------------------------------------------------------
	# Session state (single source of truth)
	# ------------------------------------------------------------------
	st.session_state.setdefault( 'pe_page', 1 )
	st.session_state.setdefault( 'pe_search', "" )
	st.session_state.setdefault( 'pe_sort_col', 'PromptsId' )
	st.session_state.setdefault( 'pe_sort_dir', 'ASC' )
	st.session_state.setdefault( 'pe_selected_id', None )
	
	st.session_state.setdefault( 'pe_name', "" )
	st.session_state.setdefault( 'pe_text', "" )
	st.session_state.setdefault( 'pe_version', 1 )
	
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
		st.session_state.pe_text = convert_xml( st.session_state.pe_text )
	
	def md_to_xml( ):
		st.session_state.pe_text = markdown_converter( st.session_state.pe_text )
	
	# ------------------------------------------------------------------
	# Controls (table filters)
	# ------------------------------------------------------------------
	c1, c2, c3, c4 = st.columns( [ 4, 2, 2, 3 ] )
	
	with c1:
		st.text_input( 'Search (Name/Text contains)', key='pe_search' )
	
	with c2:
		st.selectbox(
			'Sort by',
			[ 'PromptsId',
			  'Name',
			  'Version' ],
			key='pe_sort_col',
		)
	
	with c3:
		st.selectbox(
			'Direction',
			[ 'ASC',
			  'DESC' ],
			key='pe_sort_dir',
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
					'Selected': r[ 0 ] == st.session_state.pe_selected_id,
					'PromptsId': r[ 0 ],
					'Name': r[ 1 ],
					'Version': r[ 3 ],
					'ID': r[ 4 ],
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
	p1, p2, p3 = st.columns( [ 1, 2, 1 ] )
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
	with st.expander( 'XML ↔ Markdown Converter', expanded=False ):
		b1, b2 = st.columns( 2 )
		with b1:
			st.button( 'Convert XML → Markdown', on_click=xml_to_md )
		with b2:
			st.button( 'Convert Markdown → XML', on_click=md_to_xml )
	
	# ------------------------------------------------------------------
	# Create / Edit Prompt (AUTHORITATIVE EDITOR)
	# ------------------------------------------------------------------
	with st.expander( 'Create / Edit Prompt', expanded=True ):
		st.text_input(
			'PromptsId',
			value=st.session_state.pe_selected_id or "",
			disabled=True,
		)
		st.text_input( 'Name', key='pe_name' )
		st.text_area( 'Text', key='pe_text', height=260 )
		st.number_input( 'Version', min_value=1, key='pe_version' )
		
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
				st.success( 'Saved.' )
				reset_selection( )
		
		with c2:
			if st.session_state.pe_selected_id and st.button( 'Delete' ):
				with get_conn( ) as conn:
					conn.execute(
						f'DELETE FROM {TABLE} WHERE PromptsId=?',
						(st.session_state.pe_selected_id,),
					)
					conn.commit( )
				reset_selection( )
				st.success( 'Deleted.' )
		
		with c3:
			if st.button( 'Clear Selection' ):
				reset_selection( )

# ======================================================================================
# FOOTER — SECTION
# ======================================================================================
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

# ---- Fixed Container
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
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #35618c;
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

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================
_mode_to_model_key = \
{
		'Text': 'text_model',
		'Images': 'image_model',
		'TTS': 'tts_model',
		'Translation': 'translation_model',
		'Transcription': 'transcription_model',
		'Embeddings': 'embedding_model',
		'Document Q&A': 'docqna_model',
		'Files': 'files_model',
		'Vector Stores': 'stores_model'
}

provider_val = st.session_state.get( 'provider', '—' )
mode_val = mode or '—'
active_model = st.session_state.get( _mode_to_model_key.get( mode, "" ), None )
right_parts = [ ]
if active_model is not None:
	right_parts.append( f'Model: {active_model}' )

# ---- Rendered Variables
if mode == 'Text':
	temperature = st.session_state.get( 'text_temperature' )
	top_p = st.session_state.get( 'text_top_percent' )
	freq = st.session_state.get( 'text_frequency_penalty' )
	presence = st.session_state.get( 'text_presense_penalty' )
	number = st.session_state.get( 'text_number' )
	stream = st.session_state.get( 'text_stream' )
	parallel_tools = st.session_state.get( 'text_parallel_tools' )
	max_calls = st.session_state.get( 'text_max_tools' )
	store = st.session_state.get( 'text_store' )
	tools = st.session_state.get( 'text_tools' )
	include = st.session_state.get( 'text_include' )
	domains = st.session_state.get( 'text_domains' )
	input_mode = st.session_state.get( 'text_input' )
	tool_choice = st.session_state.get( 'text_tool_choice' )
	background = st.session_state.get( 'text_background' )
	messages = st.session_state.get( 'text_messages' )
	max_tokens = st.session_state.get( 'text_max_tokens' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature:.1%}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p:.1%}' )
	if freq is not None:
		right_parts.append( f'Freq: {freq:.2f}' )
	if presence is not None:
		right_parts.append( f'Presence: {presence:.2f}' )
	if number is not None:
		right_parts.append( f'N: {number}' )
	if max_tokens is not None:
		right_parts.append( f'Max Tokens: {max_tokens}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	if parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if max_calls is not None:
		right_parts.append( f'Max Calls: {max_calls}' )
	if store:
		right_parts.append( 'Store: On' )
	if tools:
		right_parts.append( f'Tools: {len( tools )}' )
	if include:
		right_parts.append( 'Include: On' )
	if domains:
		right_parts.append( 'Domains: Set' )
	if input_mode:
		right_parts.append( 'Input: Set' )
	if tool_choice:
		right_parts.append( f'Tool Choice: On' )
	if background:
		right_parts.append( 'Background: On' )
	if messages:
		right_parts.append( 'Messages: Set' )

elif mode == 'Images':
	image_mode = st.session_state.get( 'image_mode' )
	image_size = st.session_state.get( 'image_size' )
	image_aspect = st.session_state.get( 'image_aspect' )
	image_style = st.session_state.get( 'image_style' )
	image_backcolor = st.session_state.get( 'image_backcolor' )
	image_quality = st.session_state.get( 'image_quality' )
	image_fmt = st.session_state.get( 'image_format' )
	image_reasoning = st.session_state.get( 'image_reasoning' )
	image_detail = st.session_state.get( 'image_detail' )
	image_number = st.session_state.get( 'image_number' )
	image_stream = st.session_state.get( 'image_stream' )
	image_store = st.session_state.get( 'image_store' )
	image_background = st.session_state.get( 'image_background' )
	image_include = st.session_state.get( 'image_include' )
	image_parallel_tools = st.session_state.get( 'image_parallel_tools' )
	image_max_calls = st.session_state.get( 'text_max_tools' )
	image_tools = st.session_state.get( 'image_tools' )
	
	if image_aspect is not None:
		right_parts.append( f'Aspect: {image_aspect}' )
	elif image_size is not None:
		right_parts.append( f'Size: {image_size}' )
	
	if image_mode is not None:
		right_parts.append( f'Mode: {image_mode}' )
	if image_reasoning is not None:
		right_parts.append( f'Reasoning: {image_reasoning}' )
	if image_style is not None:
		right_parts.append( f'Style: {image_style}' )
	if image_quality is not None:
		right_parts.append( f'Quality: {image_quality}' )
	if image_backcolor is not None:
		right_parts.append( f'Backcolor: {image_backcolor}' )
	if image_fmt is not None:
		right_parts.append( f'Format: {image_fmt}' )
	if image_detail is not None:
		right_parts.append( f'Detail: {image_detail}' )
	
	if image_number is not None:
		right_parts.append( f'N: {image_number}' )
	if image_parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if image_max_calls is not None:
		right_parts.append( f'Max Calls: {image_max_calls}' )
	if image_tools:
		right_parts.append( f'Tools: {len( image_tools )}' )
	if image_include:
		right_parts.append( 'Include: On' )
	if image_stream:
		right_parts.append( 'Stream: On' )
	if image_store:
		right_parts.append( 'Store: On' )
	if image_background:
		right_parts.append( 'Background: On' )

elif mode == 'Audio':
	audio_task = st.session_state.get( 'audio_task' )
	audio_format = st.session_state.get( 'audio_response_format' )
	audio_top_p = st.session_state.get( 'audio_top_percent' )
	audio_freq = st.session_state.get( 'audio_frequency_penalty' )
	audio_presence = st.session_state.get( 'audio_presense_penalty' )
	audio_number = st.session_state.get( 'audio_number' )
	audio_temperature = st.session_state.get( 'audio_temperature' )
	audio_stream = st.session_state.get( 'audio_stream' )
	audio_store = st.session_state.get( 'audio_store' )
	audio_input_mode = st.session_state.get( 'audio_input' )
	audio_reasoning = st.session_state.get( 'audio_reasoning' )
	audio_tool_choice = st.session_state.get( 'audio_tool_choice' )
	audio_messages = st.session_state.get( 'audio_messages' )
	audio_background = st.session_state.get( 'audio_background' )
	audio_file = st.session_state.get( 'audio_file' )
	audio_rate = st.session_state.get( 'audio_rate' )
	audio_start = st.session_state.get( 'audio_start' )
	audio_end = st.session_state.get( 'audio_end' )
	audio_loop = st.session_state.get( 'audio_loop' )
	audio_play = st.session_state.get( 'auto_play' )
	audio_voice = st.session_state.get( 'voice', None )
	
	if audio_task is not None:
		right_parts.append( f'Task: {audio_task}' )
	if audio_format is not None:
		right_parts.append( f'Format: {audio_format}' )
	
	if audio_temperature is not None:
		right_parts.append( f'Temp: {audio_temperature:.1%}' )
	if audio_top_p is not None:
		right_parts.append( f'Top-P: {audio_top_p:.1%}' )
	if audio_freq is not None:
		right_parts.append( f'Freq: {audio_freq:.2f}' )
	if audio_presence is not None:
		right_parts.append( f'Presence: {audio_presence:.2f}' )
	if audio_number is not None:
		right_parts.append( f'N: {audio_number}' )
	
	if audio_stream:
		right_parts.append( 'Stream: On' )
	if audio_store:
		right_parts.append( 'Store: On' )
	if audio_reasoning:
		right_parts.append( 'Reasoning: On' )
	if audio_input:
		right_parts.append( 'Input: Set' )
	if audio_tool_choice:
		right_parts.append( f'Tool Choice: {audio_tool_choice}' )
	if audio_messages:
		right_parts.append( 'Messages: Set' )
	if audio_background:
		right_parts.append( 'Background: On' )
	
	if audio_voice:
		right_parts.append( f'Voice: {audio_voice}' )
	if audio_rate is not None:
		right_parts.append( f'Rate: {audio_rate}' )
	if (audio_start or audio_end) and audio_end >= audio_start:
		right_parts.append( f'Trim: {audio_start}s–{audio_end}s' )
	if audio_loop:
		right_parts.append( 'Loop: On' )
	if audio_play:
		right_parts.append( 'Autoplay: On' )
	if audio_file is not None:
		right_parts.append( 'File: Set' )

elif mode == 'Embeddings':
	model = st.session_state.get( 'embedding_model' )
	dimensions = st.session_state.get( 'embeddings_dimensions' )
	encoding = st.session_state.get( 'embeddings_encoding_format' )
	input_data = st.session_state.get( 'embedding_text_input' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if dimensions is not None:
		right_parts.append( f'Dim: {dimensions}' )
	
	if encoding is not None:
		right_parts.append( f'Format: {encoding}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )

elif mode == 'Files':
	files_purpose = st.session_state.get( 'files_purpose' )
	files_type = st.session_state.get( 'files_type' )
	files_id = st.session_state.get( 'files_id' )
	files_url = st.session_state.get( 'files_url' )
	
	if files_purpose is not None:
		right_parts.append( f'Purpose: {files_purpose}' )
	
	if files_type is not None:
		right_parts.append( f'Type: {files_type}' )
	
	if files_id is not None:
		right_parts.append( f'File ID: {files_id}' )
	
	if files_url is not None:
		right_parts.append( 'URL: Set' )

elif mode == 'VectorStores':
	model = st.session_state.get( 'stores_model' )
	fmt = st.session_state.get( 'stores_response_format' )
	temperature = st.session_state.get( 'stores_temperature' )
	top_p = st.session_state.get( 'stores_top_percent' )
	freq = st.session_state.get( 'stores_frequency_penalty' )
	presence = st.session_state.get( 'stores_presense_penalty' )
	number = st.session_state.get( 'stores_number' )
	stream = st.session_state.get( 'stores_stream' )
	store = st.session_state.get( 'stores_store' )
	input_data = st.session_state.get( 'stores_input' )
	reasoning = st.session_state.get( 'stores_reasoning' )
	tool_choice = st.session_state.get( 'stores_tool_choice' )
	messages = st.session_state.get( 'stores_messages' )
	background = st.session_state.get( 'stores_background' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if fmt is not None:
		right_parts.append( f'Format: {fmt}' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature}' )
	
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p}' )
	
	if freq is not None:
		right_parts.append( f'Freq: {freq}' )
	
	if presence is not None:
		right_parts.append( f'Presence: {presence}' )
	
	if number is not None:
		right_parts.append( f'N: {number}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	
	if store:
		right_parts.append( 'Store: On' )
	
	if reasoning is not None:
		right_parts.append( f'Reasoning: {reasoning}' )
	
	if tool_choice is not None:
		right_parts.append( f'Tool Choice: {tool_choice}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )
	
	if messages:
		right_parts.append( 'Messages: Set' )
	
	if background:
		right_parts.append( 'Background: On' )

right_text = ' ◽ '.join( right_parts ) if right_parts else '—'

# ---- Rendering Method
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
