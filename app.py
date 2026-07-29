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
import hashlib
import json
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tiktoken
import config as cfg
import streamlit as st
import tempfile
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from boogr import Error, Logger
from sentence_transformers import SentenceTransformer

try:
	import fitz
except Exception:
	fitz = None

import sqlite3
import os
import gpt
import gemini
import grok

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def throw_if( name: str, value: object ) -> None:
	"""Throw if.
	
	Purpose:
	    Validates that a required argument contains a usable value before the surrounding workflow
	    continues. This guard centralizes early validation so provider wrappers and UI routines
	    fail
	    with consistent, readable error messages.
	
	Args:
	    name (str): Name value used by the operation.
	    value (object): Value value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

def init_state( key: str, value: Any ) -> None:
	"""Init state.
	
	Purpose:
	    Performs the init_state workflow using the inputs supplied by the caller and the current
	    runtime
	    configuration. The function keeps this behavior isolated so related UI, provider, and
	    data-processing paths can call it consistently.
	
	Args:
	    key (str): Key value used by the operation.
	    value (Any): Value value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if key not in st.session_state:
		st.session_state[ key ] = value

def get_runtime_config_value( session_key: str, config_name: str, env_name: str ) -> str:
	"""Get runtime config value.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view
	    of provider capabilities, stored state, or response metadata so UI controls and downstream
	    logic
	    can consume it consistently.
	
	Args:
	    session_key (str): Session key value used by the operation.
	    config_name (str): Config name value used by the operation.
	    env_name (str): Env name value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	session_value = st.session_state.get( session_key, '' )
	config_value = getattr( cfg, config_name, None )
	env_value = os.environ.get( env_name, '' )
	
	if session_value:
		return str( session_value ).strip( )
	
	if config_value:
		return str( config_value ).strip( )
	
	if env_value:
		return str( env_value ).strip( )
	
	return ''

def sync_provider_config( session_key: str, config_name: str, env_name: str, value: Any,
	provider: Optional[ str ] = None ) -> None:
	"""Sync provider config.
	
	Purpose:
	    Performs the sync_provider_config workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    session_key (str): Session key value used by the operation.
	    config_name (str): Config name value used by the operation.
	    env_name (str): Env name value used by the operation.
	    value (Any): Value value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	text = str( value ).strip( ) if value is not None else ''
	st.session_state[ session_key ] = text
	
	if text:
		os.environ[ env_name ] = text
		setattr( cfg, config_name, text )
	else:
		os.environ.pop( env_name, None )
		setattr( cfg, config_name, None )
	
	if provider:
		if 'api_keys' not in st.session_state or not isinstance( st.session_state[ 'api_keys' ],
				dict ):
			st.session_state[ 'api_keys' ] = { 'GPT': None, 'Grok': None, 'Gemini': None }
		
		st.session_state[ 'api_keys' ][ provider ] = text if text else None

def init_env_state( key: str, config_name: str, env_name: str,
	provider: Optional[ str ] = None ) -> None:
	"""Init env state.
	
	Purpose:
	    Performs the init_env_state workflow using the inputs supplied by the caller and the
	    current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Args:
	    key (str): Key value used by the operation.
	    config_name (str): Config name value used by the operation.
	    env_name (str): Env name value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	init_state( key, '' )
	value = get_runtime_config_value( key, config_name, env_name )
	sync_provider_config( key, config_name, env_name, value, provider )

def copy_state_alias( source_key: str, target_key: str, default: Any ) -> None:
	"""Copy state alias.
	
	Purpose:
	    Performs the copy_state_alias workflow using the inputs supplied by the caller and the
	    current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Args:
	    source_key (str): Source key value used by the operation.
	    target_key (str): Target key value used by the operation.
	    default (Any): Default value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if target_key not in st.session_state:
		st.session_state[ target_key ] = st.session_state.get( source_key, default )
	
	if source_key not in st.session_state:
		st.session_state[ source_key ] = st.session_state.get( target_key, default )

# ---------- API / PROVIDER CONFIGURATION -------------------------------------

init_state( 'api_keys', { 'GPT': None, 'Grok': None, 'Gemini': None } )
init_env_state( 'openai_api_key', 'OPENAI_API_KEY', 'OPENAI_API_KEY', 'GPT' )
init_env_state( 'gemini_api_key', 'GEMINI_API_KEY', 'GEMINI_API_KEY', 'Gemini' )
init_env_state( 'google_api_key', 'GOOGLE_API_KEY', 'GOOGLE_API_KEY' )
init_env_state( 'google_cse_id', 'GOOGLE_CSE_ID', 'GOOGLE_CSE_ID' )
init_env_state( 'googlemaps_api_key', 'GOOGLEMAPS_API_KEY', 'GOOGLEMAPS_API_KEY' )
init_env_state( 'geocoding_api_key', 'GEOCODING_API_KEY', 'GEOCODING_API_KEY' )
init_env_state( 'geoapify_api_key', 'GEOAPIFY_API_KEY', 'GEOAPIFY_API_KEY' )
init_env_state( 'google_cloud_project_id', 'GOOGLE_CLOUD_PROJECT_ID', 'GOOGLE_CLOUD_PROJECT_ID' )
init_env_state( 'google_cloud_location', 'GOOGLE_CLOUD_LOCATION', 'GOOGLE_CLOUD_LOCATION' )
init_env_state( 'xai_api_key', 'XAI_API_KEY', 'XAI_API_KEY', 'Grok' )
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
init_state( 'last_call_usage', { 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, } )
init_state( 'token_usage', { 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, } )

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
init_state( 'filestore_system_instructions', '' )
init_state( 'bucket_system_instructions', '' )

# ---------- PROMPT TEMPLATE SELECTION STATE ----------------------------------

init_state( 'text_prompt_category', '' )
init_state( 'text_prompt_id', None )
init_state( 'image_prompt_category', '' )
init_state( 'image_prompt_id', None )
init_state( 'audio_prompt_category', '' )
init_state( 'audio_prompt_id', None )
init_state( 'docqna_prompt_category', '' )
init_state( 'docqna_prompt_id', None )
init_state( 'files_prompt_category', '' )
init_state( 'files_prompt_id', None )
init_state( 'stores_prompt_category', '' )
init_state( 'stores_prompt_id', None )
init_state( 'filestore_prompt_category', '' )
init_state( 'filestore_prompt_id', None )
init_state( 'bucket_prompt_category', '' )
init_state( 'bucket_prompt_id', None )

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
init_state( 'text_grok_collection_ids', [ ] )
init_state( 'text_grok_collection_ids_input', '' )
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

# ---------- NON-DESTRUCTIVE LIASES -----------------------------------

copy_state_alias( 'text_presense_penalty', 'text_presence_penalty', 0.0 )
copy_state_alias( 'image_presense_penalty', 'image_presence_penalty', 0.0 )
copy_state_alias( 'audio_presense_penalty', 'audio_presence_penalty', 0.0 )
copy_state_alias( 'presense_penalty', 'presence_penalty', 0.0 )
copy_state_alias( 'docqna_systems_instructions', 'docqna_system_instructions', '' )
copy_state_alias( 'text_parallel_calls', 'text_parallel_tools', False )
copy_state_alias( 'text_max_tools', 'text_max_calls', 0 )

# ------------ RESPONSE/CHAT UTILITIES

def extract_response_text( response: object ) -> str:
	"""Extract response text.
	
	Purpose:
	    Extracts structured information from a provider response, uploaded file, or application
	    data  object. The function normalizes provider-specific shapes into values that can be
	    rendered,
	    stored, or passed to later processing steps.
	
	Args:
	    response (object): Response value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
	"""Encode image base64.
	
	Purpose:
	    Performs the encode_image_base64 workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    path (str): Path value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def normalize_text( text: str ) -> str:
	"""Normalize text.
	
	Purpose:
	    Normalizes incoming values into a predictable representation for application processing.
	    The function reduces provider, user-input, or serialization differences before values are
	    stored or displayed.
	
	Args:
	    text (str): Text value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
	"""Chunk text.
	
	Purpose:
	    Performs the chunk_text workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Args:
	    text (str): Text value used by the operation.
	    max_tokens (int): Max tokens value used by the operation.
	
	Returns:
	    List[str]: Return value produced by the operation."""
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
	"""Sanitize markdown.
	
	Purpose:
	    Performs the sanitize_markdown workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    text (str): Text value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def inject_response_css( ) -> None:
	"""Inject response css.
	
	Purpose:
	    Performs the inject_response_css workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	st.markdown( """
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
	"""Style subheaders.
	
	Purpose:
	    Performs the style_subheaders workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider,
	    and data-processing paths can call it consistently.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	st.markdown( """
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stMarkdownContainer"] h4,
		div[data-testid="stMarkdownContainer"] h5,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h4,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h5,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""", unsafe_allow_html=True, )

def init_state( ) -> None:
	"""Init state.
	
	Purpose:
	    Performs the init_state workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = [ ]
	
	if 'chat_messages' not in st.session_state:
		st.session_state.chat_messages = [ ]
	
	if 'execution_mode' not in st.session_state:
		st.session_state.execution_mode = 'Standard'
	
	for k in ('audio_system_instructions', 'image_system_instructions',
		'docqna_system_instructions', 'text_system_instructions'):
		st.session_state.setdefault( k, "" )

def reset_state( ) -> None:
	"""Reset state.
	
	Purpose:
	    Removes or resets the requested application state or provider resource in a controlled
	    manner.  The function keeps cleanup behavior centralized so callers do not duplicate
	    lifecycle
	    logic.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	st.session_state.chat_history = [ ]
	st.session_state.last_answer = ""
	st.session_state.last_sources = [ ]
	st.session_state.last_analysis = { 'tables': [ ], 'files': [ ], 'text': [ ], }

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
	"""Extract answer.
	
	Purpose:
	    Extracts structured information from a provider response, uploaded file, or application
	    data object. The function normalizes provider-specific shapes into values that can be
	    rendered,
	    stored, or passed to later processing steps.
	
	Args:
	    response (Any): Response value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
		if item_type in cfg.TEXT_TYPES:
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
			if block_type in cfg.TEXT_TYPES:
				text = getattr( block, 'text', None )
				if isinstance( text, str ) and text.strip( ):
					texts.append( text )
	
	return '\n'.join( texts ).strip( )

def extract_sources( response: Any ) -> List[ Dict[ str, Any ] ]:
	"""Extract sources.
	
	Purpose:
	    Extracts structured information from a provider response, uploaded file, or application
	    data object. The function normalizes provider-specific shapes into values that can be
	    rendered,
	    stored, or passed to later processing steps.
	
	Args:
	    response (Any): Response value used by the operation.
	
	Returns:
	    List[Dict[str, Any]]: Return value produced by the operation."""
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
				
				sources.append(
					{ 'title': s.get( 'file_name' ) or s.get( 'title' ), 'snippet': s.get(
						'text' ),
						'url': None, 'files_id': s.get( 'files_id' ), } )
	
	return sources

def extract_analysis( response: Any ) -> Dict[ str, Any ]:
	"""Extract analysis.
	
	Purpose:
	    Extracts structured information from a provider response, uploaded file, or application
	    data object. The function normalizes provider-specific shapes into values that can be
	    rendered,
	    stored, or passed to later processing steps.
	
	Args:
	    response (Any): Response value used by the operation.
	
	Returns:
	    Dict[str, Any]: Return value produced by the operation."""
	artifacts: Dict[ str, Any ] = { 'tables': [ ], 'files': [ ], 'text': [ ] }
	
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
			
			elif out_type in cfg.TEXT_TYPES:
				text = getattr( out, 'text', None )
				if isinstance( text, str ) and text.strip( ):
					artifacts[ 'text' ].append( text )
	
	return artifacts

def save_temp( upload ) -> str | None:
	"""Save temp.
	
	Purpose:
	    Persists or stages input data so it can be used by later provider or application
	    workflows. The function standardizes file handling and returns a stable reference for
	    downstream
	    processing.
	
	Args:
	    upload (object): Upload value used by the operation.
	
	Returns:
	    Optional[str]: Return value produced by the operation."""
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
	"""Extract usage from response.
	
	Purpose:
	    Performs the _extract_usage_from_response workflow using the inputs supplied by the caller
	    and the current runtime configuration. The function keeps this behavior isolated so
	    related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    resp (Any): Resp value used by the operation.
	
	Returns:
	    Dict[str, int]: Return value produced by the operation."""
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
				raw.get( "completion_tokens", raw.get( "output_tokens", 0 ) ) )
			usage[ "total_tokens" ] = int( raw.get( "total_tokens",
				usage[ "prompt_tokens" ] + usage[ "completion_tokens" ], ) )
		else:
			usage[ "prompt_tokens" ] = int( getattr( raw, "prompt_tokens", 0 ) )
			usage[ "completion_tokens" ] = int(
				getattr( raw, "completion_tokens", getattr( raw, "output_tokens", 0 ) ) )
			usage[ "total_tokens" ] = int( getattr( raw, "total_tokens",
				usage[ "prompt_tokens" ] + usage[ "completion_tokens" ], ) )
	except Exception:
		usage[ "total_tokens" ] = (usage[ "prompt_tokens" ] + usage[ "completion_tokens" ])
	
	return usage

def update_token_counters( resp: Any ) -> None:
	"""Update token counters.
	
	Purpose:
	    Performs the update_token_counters workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    resp (Any): Resp value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	usage = _extract_usage_from_response( resp )
	st.session_state.last_call_usage = usage
	st.session_state.token_usage[ "prompt_tokens" ] += usage.get( "prompt_tokens", 0 )
	st.session_state.token_usage[ "completion_tokens" ] += usage.get( "completion_tokens", 0 )
	st.session_state.token_usage[ "total_tokens" ] += usage.get( "total_tokens", 0 )

def _display_value( val: Any ) -> str:
	"""Display value.
	
	Purpose:
	    Performs the _display_value workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    val (Any): Val value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	if val is None:
		return "—"
	try:
		return str( val )
	except Exception:
		return "—"

def build_intent_prefix( mode: str ) -> str:
	if mode == 'Guidance Only':
		return ('[ANALYST INTENT]\n'
		        'Respond using authoritative policy and guidance only. '
		        'Do not perform financial computation.\n\n')
	if mode == 'Analysis Only':
		return ('[ANALYST INTENT]\n'
		        'Respond using financial analysis and computation only. '
		        'Minimize policy citation.\n\n')
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
	"""Count tokens.
	
	Purpose:
	    Performs the count_tokens workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Args:
	    text (str): Text value used by the operation.
	
	Returns:
	    int: Return value produced by the operation."""
	encoding = tiktoken.get_encoding( 'cl100k_base' )
	num_tokens = len( encoding.encode( text ) )
	return num_tokens

# ------------- FILESTORE UTILITIES

def normalize_storage_object( value: Any ) -> Dict[ str, Any ]:
	"""Normalize storage object.
	
	Purpose:
	    Normalizes incoming values into a predictable representation for application processing.
	    The function reduces provider, user-input, or serialization differences before values are
	    stored or displayed.
	
	Args:
	    value (Any): Value value used by the operation.
	
	Returns:
	    Dict[str, Any]: Return value produced by the operation."""
	if value is None:
		return { }
	
	if isinstance( value, dict ):
		result = dict( value )
	elif hasattr( value, 'model_dump' ):
		try:
			dumped = value.model_dump( )
			result = dumped if isinstance( dumped, dict ) else { 'result': dumped }
		except Exception:
			result = { 'result': str( value ) }
	elif hasattr( value, 'dict' ):
		try:
			dumped = value.dict( )
			result = dumped if isinstance( dumped, dict ) else { 'result': dumped }
		except Exception:
			result = { 'result': str( value ) }
	else:
		result = { }
		for attr_name in [ 'id', 'name', 'display_name', 'description', 'status', 'state',
			'file_counts', 'usage_bytes', 'created_at', 'expires_at', 'metadata', 'deleted',
			'collection_id', 'collection_name', 'collection_description', 'documents_count',
			'document_count', 'file_id', 'filename', 'mime_type', 'size_bytes', 'bytes', ]:
			if hasattr( value, attr_name ):
				result[ attr_name ] = getattr( value, attr_name )
		
		if not result:
			result = { 'result': str( value ) }
	
	collection_id = result.get( 'collection_id' ) or result.get( 'id' ) or ''
	collection_name = result.get( 'collection_name' ) or result.get( 'display_name' )
	collection_name = collection_name or result.get( 'name' ) or collection_id or ''
	description = result.get( 'collection_description' ) or result.get( 'description' ) or ''
	status = result.get( 'status' ) or result.get( 'state' ) or ''
	file_counts = result.get( 'file_counts' )
	file_counts = file_counts if file_counts is not None else result.get( 'documents_count' )
	file_counts = file_counts if file_counts is not None else result.get( 'document_count' )
	usage_bytes = result.get( 'usage_bytes' )
	usage_bytes = usage_bytes if usage_bytes is not None else result.get( 'size_bytes' )
	usage_bytes = usage_bytes if usage_bytes is not None else result.get( 'bytes' )
	result[ 'id' ] = str( result.get( 'id' ) or collection_id or '' )
	result[ 'name' ] = str( result.get( 'name' ) or collection_name or '' )
	result[ 'display_name' ] = str( result.get( 'display_name' ) or collection_name or '' )
	result[ 'description' ] = str( result.get( 'description' ) or description or '' )
	result[ 'status' ] = str( status or '' )
	result[ 'file_counts' ] = file_counts if file_counts is not None else ''
	result[ 'usage_bytes' ] = usage_bytes if usage_bytes is not None else ''
	
	if collection_id:
		result[ 'collection_id' ] = str( collection_id )
	
	if collection_name:
		result[ 'collection_name' ] = str( collection_name )
	
	if description:
		result[ 'collection_description' ] = str( description )
	
	return result

def render_storage_metadata( metadata: Dict[ str, Any ] ) -> None:
	"""Render storage metadata.
	
	Purpose:
	    Renders the requested user interface element or result block in Streamlit using normalized
	    inputs. The function keeps presentation logic isolated from provider calls and
	    data-processing
	    steps so the screen output remains predictable.
	
	Args:
	    metadata (Dict[str, Any]): Metadata value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if not isinstance( metadata, dict ) or len( metadata ) == 0:
		st.info( 'No metadata loaded yet.' )
		return
	
	st.json( metadata )

def save_uploaded_storage_file( uploaded_file: Any ) -> Optional[ str ]:
	"""Save uploaded storage file.
	
	Purpose:
	    Persists or stages input data so it can be used by later provider or application
	    workflows. The function standardizes file handling and returns a stable reference for
	    downstream
	    processing.
	
	Args:
	    uploaded_file (Any): Uploaded file value used by the operation.
	
	Returns:
	    Optional[str]: Return value produced by the operation."""
	if uploaded_file is None:
		return None
	
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

# ------------ TEXT UTILITIES

def normalize_text( text: str ) -> str:
	"""Normalize text.
	
	Purpose:
	    Normalizes incoming values into a predictable representation for application processing.
	    The function reduces provider, user-input, or serialization differences before values are
	    stored or displayed.
	
	Args:
	    text (str): Text value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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

def chunk_text( text: str, size: int = 1200, overlap: int = 200 ) -> List[ str ]:
	chunks, i = [ ], 0
	while i < len( text ):
		chunks.append( text[ i:i + size ] )
		i += size - overlap
	return chunks

def convert_xml( text: str ) -> str:
	"""Convert xml.
	
	Purpose:
	    Performs the convert_xml workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    text (str): Text value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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

def convert_markdown( text: Any ) -> str:
	"""Convert markdown.
	
	Purpose:
	    Performs the convert_markdown workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    text (Any): Text value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
	"""Inject response css.
	
	Purpose:
	    Performs the inject_response_css workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	st.markdown( """
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
	"""Style subheaders.
	
	Purpose:
	    Performs the style_subheaders workflow using the inputs supplied by the caller and the
	    current  runtime configuration. The function keeps this behavior isolated so related UI,
	    provider,
	    and data-processing paths can call it consistently.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	st.markdown( """
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""", unsafe_allow_html=True, )

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( 'SELECT role, content FROM chat_history ORDER BY id' ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

# ------------ DOCQNA UTILITIES

def extract_text_from_bytes( file_bytes: bytes ) -> str:
	"""Extract text from bytes.
	
	Purpose:
	    Extracts structured information from a provider response, uploaded file, or application
	    data object. The function normalizes provider-specific shapes into values that can be
	    rendered,
	    stored, or passed to later processing steps.
	
	Args:
	    file_bytes (bytes): File bytes value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
	"""Route document query.
	
	Purpose:
	    Performs the route_document_query workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    prompt (str): Prompt value used by the operation.
	
	Returns:
	    str: Return value produced by the operation.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger."""
	try:
		throw_if( 'prompt', prompt )
		provider_name = st.session_state.get( 'provider', 'GPT' )
		docqna = get_chat_module( provider_name )
		user_input = build_document_user_input( prompt )
		
		if not user_input:
			user_input = (prompt or '').strip( )
		
		model = st.session_state.get( 'docqna_model' )
		if not model:
			model_options = list( getattr( docqna, 'model_options', [ ] ) or [ ] )
			model = model_options[ 0 ] if model_options else None
		
		if not model:
			raise ValueError(
				f'No Document Q&A model is configured for provider "{provider_name}".' )
		
		answer = docqna.generate_text( model=model, prompt=user_input,
			temperature=float( st.session_state.get( 'docqna_temperature', 0.0 ) ),
			top_p=float( st.session_state.get( 'docqna_top_percent', 0.95 ) ),
			frequency=float( st.session_state.get( 'docqna_frequency_penalty', 0.0 ) ),
			presence=float( st.session_state.get( 'docqna_presence_penalty', 0.0 ) ),
			max_tokens=int( st.session_state.get( 'docqna_max_tokens', 4096 ) ) or 4096,
			store=bool( st.session_state.get( 'docqna_store', False ) ), stream=False,
			instruct=st.session_state.get( 'docqna_system_instructions', '' ),
			tools=st.session_state.get( 'docqna_tools', [ ] ),
			include=st.session_state.get( 'docqna_include', [ ] ),
			tool_choice=st.session_state.get( 'docqna_tool_choice' ) or None,
			reasoning=st.session_state.get( 'docqna_reasoning' ) or None, )
		
		if isinstance( answer, str ):
			return answer
		
		output_text = getattr( docqna, 'output_text', None )
		if isinstance( output_text, str ) and output_text.strip( ):
			return output_text.strip( )
		
		output_text = getattr( answer, 'output_text', None )
		if isinstance( output_text, str ) and output_text.strip( ):
			return output_text.strip( )
		
		return str( answer or '' )
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Document Q&A'
		ex.method = 'route_document_query( prompt: str ) -> str'
		Logger( ).write( ex )
		raise ex

def summarize_active_document( ) -> str:
	"""Summarize active document.
	
	Purpose:
	    Performs the summarize_active_document workflow using the inputs supplied by the caller
	    and the current runtime configuration. The function keeps this behavior isolated so
	    related UI,
	    provider, and data-processing paths can call it consistently.
	
	Returns:
	    str: Return value produced by the operation."""
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
	"""Docqna compute fingerprint.
	
	Purpose:
	    Performs the _docqna_compute_fingerprint workflow using the inputs supplied by the caller
	    and the current runtime configuration. The function keeps this behavior isolated so
	    related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    active_docs (List[str]): Active docs value used by the operation.
	    doc_bytes (Dict[str, bytes]): Doc bytes value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	h = hashlib.sha256( )
	for name in sorted( active_docs ):
		b = doc_bytes.get( name, b'' )
		h.update( name.encode( 'utf-8', errors='ignore' ) )
		h.update( len( b ).to_bytes( 8, 'little', signed=False ) )
		h.update( hashlib.sha256( b ).digest( ) )
	return h.hexdigest( )

def _docqna_extract_text_from_pdf_bytes( file_bytes: bytes ) -> str:
	"""Docqna extract text from pdf bytes.
	
	Purpose:
	    Performs the _docqna_extract_text_from_pdf_bytes workflow using the inputs supplied by the
	    caller and the current runtime configuration. The function keeps this behavior isolated so
	    related UI, provider, and data-processing paths can call it consistently.
	
	Args:
	    file_bytes (bytes): File bytes value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
	"""Docqna safe load sqlite vec.
	
	Purpose:
	    Performs the _docqna_safe_load_sqlite_vec workflow using the inputs supplied by the caller
	    and the current runtime configuration. The function keeps this behavior isolated so
	    related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    conn (sqlite3.Connection): Conn value used by the operation.
	
	Returns:
	    bool: Return value produced by the operation."""
	try:
		import sqlite_vec
		
		sqlite_vec.load( conn )
		return True
	except Exception:
		return False

def _docqna_ensure_vec_schema( dim: int ) -> bool:
	"""Docqna ensure vec schema.
	
	Purpose:
	    Performs the _docqna_ensure_vec_schema workflow using the inputs supplied by the caller
	    and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    dim (int): Dim value used by the operation.
	
	Returns:
	    bool: Return value produced by the operation."""
	conn = create_connection( )
	try:
		ok = _docqna_safe_load_sqlite_vec( conn )
		if not ok:
			return False
		
		cur = conn.cursor( )
		cur.execute( f'''
			CREATE VIRTUAL TABLE IF NOT EXISTS docqna_vec
			USING vec0(
				embedding float[{int( dim )}],
				doc_name TEXT,
				chunk TEXT
			);
			''' )
		conn.commit( )
		return True
	except Exception:
		return False
	finally:
		conn.close( )

def _docqna_rebuild_index_if_needed( embedder: SentenceTransformer ) -> None:
	"""Docqna rebuild index if needed.
	
	Purpose:
	    Performs the _docqna_rebuild_index_if_needed workflow using the inputs supplied by the
	    caller
	    and the current runtime configuration. The function keeps this behavior isolated so
	    related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    embedder (SentenceTransformer): Embedder value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger."""
	try:
		throw_if( 'embedder', embedder )
		
		active_docs: List[ str ] = st.session_state.get( 'docqna_active_docs', [ ] )
		if active_docs is None:
			active_docs = [ ]
		
		if not active_docs:
			active_docs = st.session_state.get( 'active_docs', [ ] )
		
		if active_docs is None:
			active_docs = [ ]
		
		doc_bytes: Dict[ str, bytes ] = st.session_state.get( 'doc_bytes', { } )
		if doc_bytes is None:
			doc_bytes = { }
		
		st.session_state[ 'active_docs' ] = active_docs
		
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
				file_bytes = doc_bytes.get( name )
				if not file_bytes:
					continue
				
				suffix = Path( name ).suffix.lower( )
				if suffix == '.pdf':
					text = _docqna_extract_text_from_pdf_bytes( file_bytes )
				else:
					text = extract_text_from_bytes( file_bytes )
				
				if not text:
					continue
				
				chunks = chunk_text( text )
				if not chunks:
					continue
				
				vecs = embedder.encode( chunks, show_progress_bar=False )
				vecs = np.asarray( vecs, dtype=np.float32 )
				
				if vec_ready:
					for chunk_value, vector in zip( chunks, vecs ):
						cur.execute(
							'INSERT INTO docqna_vec ( embedding, doc_name, chunk ) VALUES ( ?, ?, '
							'? );', (vector.tobytes( ), name, chunk_value) )
				else:
					for chunk_value, vector in zip( chunks, vecs ):
						fallback_rows.append( (name, chunk_value, vector.tobytes( )) )
				
				total_chunks += int( len( chunks ) )
			
			conn.commit( )
			st.session_state[ 'docqna_chunk_count' ] = total_chunks
			
			if not vec_ready:
				st.session_state[ 'docqna_fallback_rows' ] = fallback_rows
		
		except Exception:
			st.session_state[ 'docqna_vec_ready' ] = False
			st.session_state[
				'docqna_fallback_rows' ] = fallback_rows if 'fallback_rows' in locals( ) else [ ]
			st.session_state[ 'docqna_chunk_count' ] = 0
		finally:
			conn.close( )
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Document Q&A'
		ex.method = '_docqna_rebuild_index_if_needed( embedder: SentenceTransformer ) -> None'
		Logger( ).write( ex )
		raise ex

@st.cache_resource( show_spinner=False )
def load_embedder( ) -> SentenceTransformer:
	"""Load embedder.
	
	Purpose:
	    Performs the load_embedder workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Returns:
	    SentenceTransformer: Return value produced by the operation.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger."""
	try:
		model_name = 'sentence-transformers/all-MiniLM-L6-v2'
		embedder = SentenceTransformer( model_name )
		if embedder is None:
			raise ValueError( 'The Document Q&A embedder could not be loaded.' )
		
		return embedder
	except Exception as e:
		exception = Error( e )
		exception.module = 'app'
		exception.cause = 'Document Q&A'
		exception.method = 'load_embedder( ) -> SentenceTransformer'
		Logger( ).write( exception )
		raise exception

def retrieve_top_doc_chunks( query: str, k: int = 6 ) -> List[ Tuple[ str, str, float ] ]:
	"""Retrieve top doc chunks.
	
	Purpose:
	    Performs the retrieve_top_doc_chunks workflow using the inputs supplied by the caller and
	    the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    query (str): Query value used by the operation.
	    k (int): K value used by the operation.
	
	Returns:
	    List[Tuple[str, str, float]]: Return value produced by the operation."""
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
			cur.execute( '''
                         SELECT doc_name, chunk, distance
                         FROM docqna_vec
                         WHERE embedding MATCH ?
                         ORDER BY distance ASC LIMIT ?;
			             ''', (qv.tobytes( ), int( k )) )
			rows = cur.fetchall( )
			return [ (r[ 0 ], r[ 1 ], float( r[ 2 ] )) for r in rows ]
		except Exception:
			st.session_state[ 'docqna_vec_ready' ] = False
		finally:
			conn.close( )
	
	fallback_rows: List[ Tuple[ str, str, bytes ] ] = st.session_state.get( 'docqna_fallback_rows',
		[ ] )
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
	"""Build document user input.
	
	Purpose:
	    Builds the normalized data structure required by the application workflow. The function
	    converts
	    caller input, session state, or provider-specific options into a stable shape that
	    downstream
	    API calls and rendering code can consume safely.
	
	Args:
	    user_query (str): User query value used by the operation.
	    k (int): K value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
			'Use the following document excerpts to answer the question. If the excerpts do not '
			'contain '
			'the answer, say you do not have enough information.\n\n'
			f'{context}' )
	
	prompt_parts.append( f'Question:\n{user_query}\n\nAnswer:' )
	
	return '\n\n'.join( prompt_parts ).strip( )

# ------------ DATABASE UTILITIES

def initialize_database( ) -> None:
	"""Initialize database.
	
	Purpose:
	    Performs the initialize_database workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		prompt_table_exists = conn.execute( """
                                            SELECT 1
                                            FROM sqlite_master
                                            WHERE type = 'table'
                                              AND name = 'Prompts';
		                                    """ ).fetchone( ) is not None
		
		if not prompt_table_exists:
			conn.execute( """
                          CREATE TABLE Prompts
                          (
                              ID       INTEGER NOT NULL PRIMARY KEY,
                              Caption  TEXT    NOT NULL,
                              Name     TEXT    NOT NULL,
                              Category TEXT    NOT NULL,
                              Prompt   TEXT    NOT NULL
                          );
			              """ )
		else:
			prompt_columns = { str( row[ 1 ] ) for row in
				conn.execute( 'PRAGMA table_info("Prompts");' ).fetchall( ) }
			
			required_columns = { 'ID', 'Caption', 'Name', 'Category', 'Prompt', }
			if prompt_columns != required_columns:
				conn.execute( """
                              CREATE TABLE Prompts_New
                              (
                                  ID       INTEGER NOT NULL PRIMARY KEY,
                                  Caption  TEXT    NOT NULL,
                                  Name     TEXT    NOT NULL,
                                  Category TEXT    NOT NULL,
                                  Prompt   TEXT    NOT NULL
                              );
				              """ )
				
				source_text_column = (
					'Prompt' if 'Prompt' in prompt_columns else 'Text' if 'Text' in prompt_columns
					else None)
				
				if source_text_column is not None:
					category_expression = (
						'COALESCE(NULLIF(TRIM(Category), \'\'), \'Uncategorized\')' if 'Category'
						                                                               in
						                                                               prompt_columns else '\'Uncategorized\'')
					
					conn.execute( f"""
						INSERT INTO Prompts_New
						(
							ID,
							Caption,
							Name,
							Category,
							Prompt
						)
						SELECT
							ID,
							COALESCE(NULLIF(TRIM(Caption), ''), 'Prompt ' || ID),
							COALESCE(NULLIF(TRIM(Name), ''), 'Prompt' || ID),
							{category_expression},
							COALESCE({source_text_column}, '')
						FROM Prompts
						WHERE ID IS NOT NULL;
						""" )
				
				conn.execute( 'DROP TABLE Prompts;' )
				conn.execute( 'ALTER TABLE Prompts_New RENAME TO Prompts;' )
		
		conn.execute( """
                      CREATE INDEX IF NOT EXISTS IX_Prompts_Category
                          ON Prompts ( Category );
		              """ )
		
		conn.execute( """
                      CREATE INDEX IF NOT EXISTS IX_Prompts_Caption
                          ON Prompts ( Caption );
		              """ )
		
		conn.execute( """
                      CREATE INDEX IF NOT EXISTS IX_Prompts_Name
                          ON Prompts ( Name );
		              """ )
		
		conn.commit( )

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

def read_table( table: str, limit: int = None, offset: int = 0 ) -> pd.DataFrame:
	"""Read table.
	
	Purpose:
	    Performs the read_table workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Args:
	    table (str): Table value used by the operation.
	    limit (int): Limit value used by the operation.
	    offset (int): Offset value used by the operation.
	
	Returns:
	    pd.DataFrame: Return value produced by the operation."""
	if not table:
		return pd.DataFrame( )
	
	query = f'SELECT * FROM "{table}"'
	if limit:
		query += f' LIMIT {int( limit )} OFFSET {int( offset )}'
	
	with create_connection( ) as conn:
		cur = conn.cursor( )
		cur.execute( query )
		
		raw_columns = [ d[ 0 ] for d in (cur.description or [ ]) ]
		rows = cur.fetchall( )
	
	seen: Dict[ str, int ] = { }
	columns: List[ str ] = [ ]
	
	for col in raw_columns:
		name = str( col )
		if name not in seen:
			seen[ name ] = 0
			columns.append( name )
		else:
			seen[ name ] += 1
			columns.append( f'{name}_{seen[ name ]}' )
	
	def _scalarize( value: Any ) -> Any:
		if value is None or isinstance( value, (str, int, float, bool) ):
			return value
		
		if isinstance( value, bytes ):
			try:
				return value.decode( 'utf-8' )
			except Exception:
				return value.hex( )
		
		if isinstance( value, (list, tuple, set, dict) ):
			try:
				return str( normalize( value ) )
			except Exception:
				return str( value )
		
		if hasattr( value, 'model_dump' ):
			try:
				return str( value.model_dump( ) )
			except Exception:
				return str( value )
		
		return str( value )
	
	normalized_rows: List[ Dict[ str, Any ] ] = [ ]
	for row in rows:
		record: Dict[ str, Any ] = { }
		for idx, col in enumerate( columns ):
			record[ col ] = _scalarize( row[ idx ] )
		normalized_rows.append( record )
	
	return pd.DataFrame( normalized_rows, columns=columns )

def render_table( df: pd.DataFrame ) -> None:
	"""Render table.
	
	Purpose:
	    Renders the requested user interface element or result block in Streamlit using normalized
	    inputs. The function keeps presentation logic isolated from provider calls and
	    data-processing
	    steps so the screen output remains predictable.
	
	Args:
	    df (pd.DataFrame): Df value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if df is None:
		st.info( 'No data available.' )
		return
	
	try:
		st.data_editor( df, use_container_width=True )
		return
	except Exception:
		pass
	
	fallback_df = df.copy( )
	fallback_df = fallback_df.where( pd.notnull( fallback_df ), '' )
	
	for col in fallback_df.columns:
		fallback_df[ col ] = fallback_df[ col ].map(
			lambda x: x if isinstance( x, (str, int, float, bool) ) or x == '' else str( x ) )
	
	st.markdown( fallback_df.to_html( index=False, escape=True ), unsafe_allow_html=True )

def make_display_safe( df: pd.DataFrame ) -> pd.DataFrame:
	display_df = df.copy( )
	
	for col in display_df.columns:
		display_df[ col ] = display_df[ col ].map( lambda x: '' if x is None else str( x ) )
	
	return display_df

def drop_table( table: str ) -> None:
	"""Drop table.
	
	Purpose:
	    Performs the drop_table workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Args:
	    table (str): Table value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""Create index.
	
	Purpose:
	    Creates the requested resource, connection, schema object, or user interface artifact using
	    validated inputs. The function encapsulates setup details so callers can rely on a
	    consistent resource lifecycle.
	
	Args:
	    table (str): Table value used by the operation.
	    column (str): Column value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( 'Invalid table name.' )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( 'Invalid column name.' )
	
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

def create_visualization( df: pd.DataFrame ) -> None:
	"""Create visualization.
	
	Purpose:
	    Creates the requested resource, connection, schema object, or user interface artifact using
	    validated inputs. The function encapsulates setup details so callers can rely on a
	    consistent
	    resource lifecycle.
	
	Args:
	    df (pd.DataFrame): Df value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	st.subheader( 'Visualization Engine' )
	
	if df is None or df.empty:
		st.info( 'No data available.' )
		return
	
	df_plot = df.copy( )
	
	for col in df_plot.columns:
		if df_plot[ col ].dtype == object:
			df_plot[ col ] = df_plot[ col ].map( lambda x: '' if x is None else str( x ) )
	
	numeric_cols: List[ str ] = [ ]
	for col in df_plot.columns:
		series_num = pd.to_numeric( df_plot[ col ], errors='coerce' )
		if series_num.notna( ).any( ):
			numeric_cols.append( col )
	
	categorical_cols: List[ str ] = [ col for col in df_plot.columns if col not in numeric_cols ]
	
	chart = st.selectbox( 'Chart Type',
		[ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Histogram( x=values ) ] )
		fig.update_layout( xaxis_title=col, yaxis_title='Count' )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Bar( x=x_values, y=y_values ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='lines' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		x = st.selectbox( 'X', numeric_cols, key='viz_scatter_x' )
		y = st.selectbox( 'Y', numeric_cols, key='viz_scatter_y' )
		
		x_series = pd.to_numeric( df_plot[ x ], errors='coerce' )
		y_series = pd.to_numeric( df_plot[ y ], errors='coerce' )
		mask = x_series.notna( ) & y_series.notna( )
		
		x_values = x_series[ mask ].tolist( )
		y_values = y_series[ mask ].tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='markers' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols, key='viz_box_col' )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Box( y=values, name=col ) ] )
		fig.update_layout( yaxis_title=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		if not categorical_cols:
			st.info( 'No categorical columns available.' )
			return
		
		col = st.selectbox( 'Category Column', categorical_cols )
		counts = df_plot[ col ].astype( str ).value_counts( )
		
		fig = go.Figure(
			data=[ go.Pie( labels=counts.index.tolist( ), values=counts.values.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		corr_df = pd.DataFrame( )
		for col in numeric_cols:
			corr_df[ col ] = pd.to_numeric( df_plot[ col ], errors='coerce' )
		
		corr = corr_df.corr( )
		
		fig = go.Figure( data=[ go.Heatmap( z=corr.values.tolist( ), x=corr.columns.tolist( ),
			y=corr.index.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )

def convert_dataframe( table_name: str, df: pd.DataFrame ):
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
	"""Get sqlite type.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    dtype (object): Dtype value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if 'int' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if 'float' in dtype_str:
		return 'REAL'
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if 'bool' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if 'datetime' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if 'category' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return 'TEXT'

def create_custom_table( table_name: str, columns: list ) -> None:
	"""Create custom table.
	
	Purpose:
	    Creates the requested resource, connection, schema object, or user interface artifact using
	    validated inputs. The function encapsulates setup details so callers can rely on a
	    consistent resource lifecycle.
	
	Args:
	    table_name (str): Table name value used by the operation.
	    columns (list): Columns value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if not table_name:
		raise ValueError( 'Table name required.' )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( 'Invalid table name.' )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ 'name' ]
		col_type = col[ 'type' ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ 'primary_key' ]:
			definition += ' PRIMARY KEY'
			if col[ 'auto_increment' ] and col_type == 'INTEGER':
				definition += ' AUTOINCREMENT'
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""Is safe query.
	
	Purpose:
	    Performs the is_safe_query workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    query (str): Query value used by the operation.
	
	Returns:
	    bool: Return value produced by the operation."""
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
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ', 'create ', 'attach ',
		'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""Create identifier.
	
	Purpose:
	    Creates the requested resource, connection, schema object, or user interface artifact using
	    validated inputs. The function encapsulates setup details so callers can rely on a
	    consistent resource lifecycle.
	
	Args:
	    name (str): Name value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
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
		conn.execute( f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""Rename column.
	
	Purpose:
	    Performs the rename_column workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    table_name (str): Table name value used by the operation.
	    old_name (str): Old name value used by the operation.
	    new_name (str): New name value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (table_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (table_name,) ).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";' )
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = { 'column': col, 'dtype': str( series.dtype ),
			'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
			'distinct_%': round( (distinct_count / total_rows) * 100, 2 ) if total_rows else 0, }
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ 'min' ] = series.min( )
			row[ 'max' ] = series.max( )
			row[ 'mean' ] = series.mean( )
		else:
			row[ 'min' ] = None
			row[ 'max' ] = None
			row[ 'mean' ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( 'Table and column required.' )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (table,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )
		
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
		
		new_create_sql = (f'CREATE TABLE "{temp_table}" (' + ", ".join( new_defs ) + ");")
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [ c.split( )[ 0 ].strip( '"' ) for c in new_defs ]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute( f'INSERT INTO "{temp_table}" ({col_list}) '
		              f'SELECT {col_list} FROM "{table}";' )
		
		# Preserve indexes
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (table,) ).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	"""Rename table.
	
	Purpose:
	    Performs the rename_table workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    old_name (str): Old name value used by the operation.
	    new_name (str): New name value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (old_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (old_name,) ).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";' )
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

# ------------ PROMPT ENGINEERING UTILITIES

PROMPT_CATEGORY_MODE_MAP: Dict[ str, List[ str ] ] = {
	'Text': [ 'Writing / Administrative', 'Research / Academic', 'Data Analytics & Governance',
		'Software Engineering', 'Business / Finance / Marketing', 'Compliance / Legal / Budget',
		'Prompt Engineering', 'Instruction/ Training / Planning', ],
	'Images': [ 'Image Generation', 'Image Analysis', 'Image Editing', ],
	'Audio': [ 'Transcription API', 'Translation API', 'Speech API', ],
	'Document Q&A': [ 'Research / Academic', 'Data Analytics & Governance',
		'Business / Finance / Marketing', 'Compliance / Legal / Budget',
		'Instruction/ Training / Planning', 'Writing / Administrative', ],
	'Files': [ 'Writing / Administrative', 'Research / Academic', 'Data Analytics & Governance',
		'Software Engineering', 'Business / Finance / Marketing', 'Compliance / Legal / Budget',
		'Instruction/ Training / Planning', ],
	'Vector Stores': [ 'Research / Academic', 'Data Analytics & Governance', 'Software '
	                                                                         'Engineering',
		'Compliance / Legal / Budget', 'Instruction/ Training / Planning', ],
	'File Search Stores': [ 'Research / Academic', 'Data Analytics & Governance',
		'Software Engineering', 'Compliance / Legal / Budget',
		'Instruction/ Training / Planning', ],
	'Cloud Buckets': [ 'Data Analytics & Governance', 'Software Engineering', 'Research / '
	                                                                          'Academic',
		'Compliance / Legal / Budget', ], }

def fetch_prompt_categories( mode_name: str ) -> List[ str ]:
	"""Fetch prompt categories.
	
	Purpose:
	    Returns populated prompt categories authorized for the selected application mode.
	    Categories retain their configured display order and categories without corresponding
	    database
	    records are excluded.
	
	Args:
	    mode_name (str): Application mode used to determine the permitted prompt categories.
	
	Returns:
	    List[str]: Ordered prompt categories available to the selected mode.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		throw_if( 'mode_name', mode_name )
		permitted_categories = PROMPT_CATEGORY_MODE_MAP.get( mode_name, [ ] )
		
		if not permitted_categories:
			return [ ]
		
		placeholders = ', '.join( [ '?' ] * len( permitted_categories ) )
		
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			rows = conn.execute( f"""
				SELECT DISTINCT Category
				FROM Prompts
				WHERE Category IN ({placeholders})
					AND TRIM(Category) <> '';
				""", tuple( permitted_categories ), ).fetchall( )
		
		available_categories = { str( row[ 0 ] ).strip( ) for row in rows if
			row and row[ 0 ] is not None and str( row[ 0 ] ).strip( ) }
		
		return [ category for category in permitted_categories if category in
		                                                          available_categories ]
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = 'fetch_prompt_categories( mode_name: str ) -> List[ str ]'
		Logger( ).write( ex )
		raise ex

def fetch_prompt_options( category: str ) -> List[ Dict[ str, Any ] ]:
	"""Fetch prompt options.
	
	Purpose:
	    Returns prompt-template identifiers and display metadata for the selected category. The
	    result provides stable numeric identifiers for widget state while preserving captions for
	    presentation.
	
	Args:
	    category (str): Prompt category used to filter the available templates.
	
	Returns:
	    List[Dict[str, Any]]: Prompt identifiers and display metadata ordered by caption and
	    identifier.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		if not category or not str( category ).strip( ):
			return [ ]
		
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			rows = conn.execute( """
                                 SELECT ID,
                                        Caption,
                                        Name,
                                        Category
                                 FROM Prompts
                                 WHERE Category = ?
                                 ORDER BY Caption, ID;
			                     """, (str( category ).strip( ),), ).fetchall( )
		
		return [ { 'ID': int( row[ 0 ] ), 'Caption': str( row[ 1 ] or '' ),
			'Name': str( row[ 2 ] or '' ), 'Category': str( row[ 3 ] or '' ), } for row in rows ]
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = 'fetch_prompt_options( category: str ) -> List[ Dict[ str, Any ] ]'
		Logger( ).write( ex )
		raise ex

def fetch_prompt_by_id( prompt_id: int ) -> Optional[ Dict[ str, Any ] ]:
	"""Fetch prompt by identifier.
	
	Purpose:
	    Returns the complete prompt-template record associated with a stable numeric identifier.
	    The identifier-based lookup prevents ambiguous template selection when captions or names
	    are
	    duplicated.
	
	Args:
	    prompt_id (int): Numeric primary key of the prompt-template record.
	
	Returns:
	    Optional[Dict[str, Any]]: Complete prompt-template record when found; otherwise None.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		if prompt_id is None:
			return None
		
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			cur = conn.execute( """
                                SELECT ID,
                                       Caption,
                                       Name,
                                       Category,
                                       Prompt
                                FROM Prompts
                                WHERE ID = ?;
			                    """, (int( prompt_id ),), )
			
			row = cur.fetchone( )
			
			if row is None:
				return None
			
			return { 'ID': int( row[ 0 ] ), 'Caption': str( row[ 1 ] or '' ),
				'Name': str( row[ 2 ] or '' ), 'Category': str( row[ 3 ] or '' ),
				'Prompt': str( row[ 4 ] or '' ), }
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = 'fetch_prompt_by_id( prompt_id: int ) -> Optional[ Dict[ str, Any ] ]'
		Logger( ).write( ex )
		raise ex

def reset_prompt_template_selection( prompt_id_key: str ) -> None:
	"""Reset prompt template selection.
	
	Purpose:
	    Clears a mode-specific prompt-template selection when its category changes without
	    modifying the current system-instruction text.
	
	Args:
	    prompt_id_key (str): Session-state key storing the selected prompt identifier.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		throw_if( 'prompt_id_key', prompt_id_key )
		st.session_state[ prompt_id_key ] = None
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = 'reset_prompt_template_selection( prompt_id_key: str ) -> None'
		Logger( ).write( ex )
		raise ex

def load_prompt_template( prompt_id_key: str, instructions_key: str, ) -> None:
	"""Load prompt template.
	
	Purpose:
	    Loads the selected prompt body into a mode-specific system-instruction field while
	    preserving independent template state across application modes.
	
	Args:
	    prompt_id_key (str): Session-state key storing the selected prompt identifier.
	    instructions_key (str): Session-state key receiving the selected prompt body.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		throw_if( 'prompt_id_key', prompt_id_key )
		throw_if( 'instructions_key', instructions_key )
		
		prompt_id = st.session_state.get( prompt_id_key )
		
		if prompt_id is None:
			return
		
		record = fetch_prompt_by_id( int( prompt_id ) )
		
		if record is None:
			return
		
		st.session_state[ instructions_key ] = record[ 'Prompt' ]
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = ('load_prompt_template( prompt_id_key: str, '
		             'instructions_key: str ) -> None')
		Logger( ).write( ex )
		raise ex

def format_prompt_option( prompt_id: int, prompt_options: List[ Dict[ str, Any ] ], ) -> str:
	"""Format prompt option.
	
	Purpose:
	    Resolves a prompt identifier to its human-readable caption for presentation in a Streamlit
	    selection control.
	
	Args:
	    prompt_id (int): Numeric prompt identifier rendered by the selection control.
	    prompt_options (List[Dict[str, Any]]): Available prompt records used to resolve the
	    caption.
	
	Returns:
	    str: Prompt caption when found; otherwise the numeric identifier as text.
	"""
	for option in prompt_options:
		if int( option.get( 'ID', -1 ) ) == int( prompt_id ):
			return str( option.get( 'Caption', prompt_id ) )
	
	return str( prompt_id )

def fetch_prompts_df( ) -> pd.DataFrame:
	"""Fetch prompts dataframe.
	
	Purpose:
	    Returns prompt-template metadata for management and review without rendering large prompt
	    bodies directly in the primary data grid.
	
	Returns:
	    pd.DataFrame: Prompt-template metadata with a selection column.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			df_prompts = pd.read_sql_query( """
                                            SELECT ID,
                                                   Caption,
                                                   Name,
                                                   Category
                                            FROM Prompts
                                            ORDER BY ID DESC;
			                                """, conn, )
		
		df_prompts.insert( 0, 'Selected', False )
		return df_prompts
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = 'fetch_prompts_df( ) -> pd.DataFrame'
		Logger( ).write( ex )
		raise ex

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	"""Insert prompt.
	
	Purpose:
	    Creates a prompt-template record using the canonical category-aware prompt schema.
	
	Args:
	    data (Dict[str, Any]): Prompt-template values containing Caption, Name, Category,
	    and Prompt.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		throw_if( 'data', data )
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			conn.execute( """
                          INSERT INTO Prompts
                          (Caption,
                           Name,
                           Category,
                           Prompt)
                          VALUES (?,
                                  ?,
                                  ?,
                                  ?);
			              """, (str( data[ 'Caption' ] ).strip( ), str( data[ 'Name' ] ).strip( ),
				str( data[ 'Category' ] ).strip( ), str( data[ 'Prompt' ] ),), )
			conn.commit( )
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = 'insert_prompt( data: Dict[ str, Any ] ) -> None'
		Logger( ).write( ex )
		raise ex

def update_prompt( prompt_id: int, data: Dict[ str, Any ] ) -> None:
	"""Update prompt.
	
	Purpose:
	    Updates an existing prompt-template record using the canonical category-aware prompt
	    schema.
	
	Args:
	    prompt_id (int): Numeric primary key of the prompt-template record.
	    data (Dict[str, Any]): Replacement Caption, Name, Category, and Prompt values.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		throw_if( 'data', data )
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			conn.execute( """
                          UPDATE Prompts
                          SET Caption  = ?,
                              Name     = ?,
                              Category = ?,
                              Prompt   = ?
                          WHERE ID = ?;
			              """, (str( data[ 'Caption' ] ).strip( ), str( data[ 'Name' ] ).strip( ),
				str( data[ 'Category' ] ).strip( ), str( data[ 'Prompt' ] ), int( prompt_id ),), )
			conn.commit( )
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = ('update_prompt( prompt_id: int, '
		             'data: Dict[ str, Any ] ) -> None')
		Logger( ).write( ex )
		raise ex

def delete_prompt( prompt_id: int ) -> None:
	"""Delete prompt.
	
	Purpose:
	    Removes the prompt-template record associated with the supplied numeric identifier.
	
	Args:
	    prompt_id (int): Numeric primary key of the prompt-template record.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger.
	"""
	try:
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			conn.execute( 'DELETE FROM Prompts WHERE ID = ?;', (int( prompt_id ),), )
			conn.commit( )
	except Exception as e:
		ex = Error( e )
		ex.module = 'app'
		ex.cause = 'Prompt Templates'
		ex.method = 'delete_prompt( prompt_id: int ) -> None'
		Logger( ).write( ex )
		raise ex

def build_prompt( user_input: str ) -> str:
	"""Build prompt.
	
	Purpose:
	    Builds the normalized data structure required by the application workflow. The function
	    converts caller input, session state, or provider-specific options into a stable shape that
	    downstream API calls and rendering code can consume safely.
	
	Args:
	    user_input (str): User input value used by the operation.
	
	Returns:
	    str: Return value produced by the operation.
	
	Raises:
	    Exception: Re-raises exceptions after recording them with the application logger."""
	try:
		throw_if( 'user_input', user_input )
		top_k = int( st.session_state.get( 'text_top_k', 6 ) or 6 )
		system_prompt = str( st.session_state.get( 'system_prompt',
			st.session_state.get( 'instructions', '' ) ) or '' ).strip( )
		basic_docs = st.session_state.get( 'basic_docs', [ ] )
		messages = st.session_state.get( 'messages', [ ] )
		use_semantic = bool( st.session_state.get( 'use_semantic', False ) )
		prompt = ''
		if system_prompt:
			prompt += f'<|system|>\n{system_prompt}\n</s>\n'
		if use_semantic:
			try:
				with sqlite3.connect( cfg.DB_PATH ) as conn:
					rows = conn.execute( 'SELECT chunk, vector FROM embeddings' ).fetchall( )
				if rows:
					embedder = load_embedder( )
					query_vector = embedder.encode( [ user_input ] )[ 0 ]
					query_vector = np.asarray( query_vector, dtype=np.float32 )
					scored = [ ]
					for chunk, vector_blob in rows:
						if chunk is None or vector_blob is None:
							continue
						
						vector = np.frombuffer( vector_blob, dtype=np.float32 )
						if vector.size != query_vector.size:
							alternate_vector = np.frombuffer( vector_blob, dtype=np.float64 )
							if alternate_vector.size != query_vector.size:
								continue
							
							vector = alternate_vector.astype( np.float32 )
						score = cosine_sim( query_vector, vector )
						scored.append( (chunk, score) )
					
					for chunk, _ in sorted( scored, key=lambda item: item[ 1 ], reverse=True )[
						:top_k ]:
						prompt += f'<|system|>\n{chunk}\n</s>\n'
			except Exception:
				pass
		if isinstance( basic_docs, list ):
			for document in basic_docs[ :6 ]:
				if document:
					prompt += f'<|system|>\n{document}\n</s>\n'
		
		if isinstance( messages, list ):
			for message in messages:
				if isinstance( message, dict ):
					role = message.get( 'role', 'user' )
					content = message.get( 'content', '' )
				elif isinstance( message, (list, tuple) ) and len( message ) >= 2:
					role, content = message[ 0 ], message[ 1 ]
				else:
					continue
				
				if role and content:
					prompt += f'<|{role}|>\n{content}\n</s>\n'
		
		prompt += f'<|user|>\n{user_input}\n</s>\n<|assistant|>\n'
		return prompt
	except Exception as e:
		exception = Error( e )
		exception.module = 'app'
		exception.cause = 'Prompt Builder'
		exception.method = 'build_prompt( user_input: str ) -> str'
		Logger( ).write( exception )
		raise exception

# ------------ PROVIDER UTILITIES

def get_provider_name( provider: Optional[ str ] = None ) -> str:
	"""Get provider name.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	selected = provider or st.session_state.get( 'provider', 'GPT' )
	providers = getattr( cfg, 'PROVIDERS', { 'GPT': 'gpt', 'Gemini': 'gemini', 'Grok': 'grok' } )
	
	if selected not in providers:
		selected = 'GPT'
		st.session_state[ 'provider' ] = selected
	
	return selected

def get_provider_module( provider: Optional[ str ] = None ) -> Any:
	"""Get provider module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	selected = get_provider_name( provider )
	provider_modules = { 'GPT': gpt, 'Gemini': gemini, 'Grok': grok, }
	
	module = provider_modules.get( selected )
	if module is None:
		raise ValueError( f'Provider "{selected}" is not mapped to an imported module.' )
	
	return module

def provider_has_class( class_name: str, provider: Optional[ str ] = None ) -> bool:
	"""Provider has class.
	
	Purpose:
	    Performs the provider_has_class workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider,
	    and data-processing paths can call it consistently.
	
	Args:
	    class_name (str): Class name value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    bool: Return value produced by the operation."""
	if not class_name:
		return False
	
	provider_module = get_provider_module( provider )
	return hasattr( provider_module, class_name )

def get_provider_class( class_name: str, provider: Optional[ str ] = None ) -> type:
	"""Get provider class.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    class_name (str): Class name value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    type: Return value produced by the operation."""
	if not class_name:
		raise ValueError( 'class_name cannot be empty.' )
	
	selected = get_provider_name( provider )
	provider_module = get_provider_module( selected )
	
	if not hasattr( provider_module, class_name ):
		raise AttributeError(
			f'Provider "{selected}" does not expose a "{class_name}" wrapper class.' )
	
	return getattr( provider_module, class_name )

def get_provider_instance( class_name: str, provider: Optional[ str ] = None ) -> Any:
	"""Get provider instance.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    class_name (str): Class name value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	provider_class = get_provider_class( class_name, provider )
	return provider_class( )

def get_chat_module( provider: Optional[ str ] = None ) -> Any:
	"""Get chat module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'Chat', provider )

def get_tts_module( provider: Optional[ str ] = None ) -> Any:
	"""Get tts module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'TTS', provider )

def get_images_module( provider: Optional[ str ] = None ) -> Any:
	"""Get images module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'Images', provider )

def get_embeddings_module( provider: Optional[ str ] = None ) -> Any:
	"""Get embeddings module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'Embeddings', provider )

def get_translation_module( provider: Optional[ str ] = None ) -> Any:
	"""Get translation module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'Translation', provider )

def get_transcription_module( provider: Optional[ str ] = None ) -> Any:
	"""Get transcription module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'Transcription', provider )

def get_files_module( provider: Optional[ str ] = None ) -> Any:
	"""Get files module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'Files', provider )

def get_vectorstores_module( provider: Optional[ str ] = None ) -> Any:
	"""Get vectorstores module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'VectorStores', provider )

def get_file_search_module( provider: Optional[ str ] = None ) -> Any:
	"""Get file search module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'FileSearch', provider )

def get_cloud_buckets_module( provider: Optional[ str ] = None ) -> Any:
	"""Get cloud buckets module.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	return get_provider_instance( 'CloudBuckets', provider )

def get_mode_classes( mode: Optional[ str ] = None,
	provider: Optional[ str ] = None ) -> List[ str ]:
	"""Get mode classes.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    mode (Optional[str]): Mode value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    List[str]: Return value produced by the operation."""
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
	"""Provider supports mode.
	
	Purpose:
	    Performs the provider_supports_mode workflow using the inputs supplied by the caller and
	    the current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    mode (Optional[str]): Mode value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    bool: Return value produced by the operation."""
	classes = get_mode_classes( mode, provider )
	if not classes:
		return True
	
	return all( provider_has_class( class_name, provider ) for class_name in classes )

def require_provider_mode( mode: Optional[ str ] = None, provider: Optional[ str ] = None ) -> bool:
	"""Require provider mode.
	
	Purpose:
	    Performs the require_provider_mode workflow using the inputs supplied by the caller and the
	    current runtime configuration. The function keeps this behavior isolated so related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    mode (Optional[str]): Mode value used by the operation.
	    provider (Optional[str]): Provider value used by the operation.
	
	Returns:
	    bool: Return value produced by the operation."""
	selected_mode = mode or st.session_state.get( 'mode', 'Text' )
	selected_provider = get_provider_name( provider )
	classes = get_mode_classes( selected_mode, selected_provider )
	missing = [ class_name for class_name in classes if
		not provider_has_class( class_name, selected_provider ) ]
	
	if missing:
		st.warning( f'{selected_provider} does not currently expose the required wrapper(s) for '
		            f'{selected_mode}: {", ".join( missing )}.' )
		return False
	
	return True

def _provider( ) -> str:
	"""Provider.
	
	Purpose:
	    Performs the _provider workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Returns:
	    str: Return value produced by the operation."""
	return get_provider_name( )

def _safe( module: str, attr: str, fallback: Any ) -> Any:
	"""Safe.
	
	Purpose:
	    Performs the _safe workflow using the inputs supplied by the caller and the current runtime
	    configuration. The function keeps this behavior isolated so related UI, provider, and
	    data-processing paths can call it consistently.
	
	Args:
	    module (str): Module value used by the operation.
	    attr (str): Attr value used by the operation.
	    fallback (Any): Fallback value used by the operation.
	
	Returns:
	    Any: Return value produced by the operation."""
	try:
		mod = __import__( module )
		return getattr( mod, attr, fallback )
	except Exception:
		return fallback

# ------------ SIDEBAR UTILITIES

def get_provider_options( ) -> List[ str ]:
	"""Get provider options.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
		logic can consume it consistently.
	
	Returns:
	    List[str]: Return value produced by the operation."""
	providers = getattr( cfg, 'PROVIDERS', { 'GPT': 'gpt', 'Gemini': 'gemini', 'Grok': 'grok' } )
	return list( providers.keys( ) )

def get_raw_provider_modes( provider: str ) -> List[ str ]:
	"""Get raw provider modes.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    provider (str): Provider value used by the operation.
	
	Returns:
	    List[str]: Return value produced by the operation."""
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
	"""Normalize mode name.
	
	Purpose:
	    Normalizes incoming values into a predictable representation for application processing.
	    The function reduces provider, user-input, or serialization differences before values are
	    stored or displayed.
	
	Args:
	    mode_name (Optional[str]): Mode name value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	if not mode_name:
		return 'Text'
	
	mode_aliases = { 'Embedding': 'Embeddings', 'Documents': 'Document Q&A',
		'Data Export': 'Export', 'Export Data': 'Export', }
	
	return mode_aliases.get( mode_name, mode_name )

def normalize_mode_list( modes: List[ str ] ) -> List[ str ]:
	"""Normalize mode list.
	
	Purpose:
	    Normalizes incoming values into a predictable representation for application processing.
	    The function reduces provider, user-input, or serialization differences before values are
	    stored or displayed.
	
	Args:
	    modes (List[str]): Modes value used by the operation.
	
	Returns:
	    List[str]: Return value produced by the operation."""
	normalized = [ ]
	
	for item in modes:
		mode_name = normalize_mode_name( item )
		if mode_name not in normalized:
			normalized.append( mode_name )
	
	return normalized

def mode_requires_runtime_wrapper( mode_name: str ) -> bool:
	"""Mode requires runtime wrapper.
	
	Purpose:
	    Performs the mode_requires_runtime_wrapper workflow using the inputs supplied by the
	    caller and the current runtime configuration. The function keeps this behavior isolated so
	    related UI,
	    provider, and data-processing paths can call it consistently.
	
	Args:
	    mode_name (str): Mode name value used by the operation.
	
	Returns:
	    bool: Return value produced by the operation."""
	non_wrapper_modes = [ 'Prompt Engineering', 'Data Management', 'Export', ]
	
	return mode_name not in non_wrapper_modes

def get_supported_provider_modes( provider: str ) -> List[ str ]:
	"""Get supported provider modes.
	
	Purpose:
	    Returns the configured application modes available to the selected provider. Mode
	    availability is determined by the provider-specific configuration and is not altered by
	    runtime wrapper introspection.
	
	Args:
	    provider (str): Selected provider name.
	
	Returns:
	    List[str]: Normalized provider modes in configured display order.
	"""
	raw_modes = get_raw_provider_modes( provider )
	modes = normalize_mode_list( raw_modes )
	
	if not modes:
		return [ 'Text' ]
	
	return modes

def get_mode_index( modes: List[ str ], current_mode: Optional[ str ] ) -> int:
	"""Get mode index.
	
	Purpose:
	    Returns normalized information for the application component. The method provides a stable
	    view of provider capabilities, stored state, or response metadata so UI controls and
	    downstream
	    logic can consume it consistently.
	
	Args:
	    modes (List[str]): Modes value used by the operation.
	    current_mode (Optional[str]): Current mode value used by the operation.
	
	Returns:
	    int: Return value produced by the operation."""
	mode_name = normalize_mode_name( current_mode )
	
	if mode_name in modes:
		return modes.index( mode_name )
	
	return 0

def render_provider_keys( ) -> None:
	"""Render provider keys.
	
	Purpose:
	    Renders the requested user interface element or result block in Streamlit using normalized
	    inputs. The function keeps presentation logic isolated from provider calls and
	    data-processing steps so the screen output remains predictable.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	with st.expander( 'Keys:', expanded=False ):
		openai_key = st.text_input( 'OpenAI API Key', type='password',
			value=get_runtime_config_value( 'openai_api_key', 'OPENAI_API_KEY', 'OPENAI_API_KEY' ),
			help='Overrides OPENAI_API_KEY from config.py for this session only.',
			key='sidebar_openai_api_key' )
		
		gemini_key = st.text_input( 'Gemini API Key', type='password',
			value=get_runtime_config_value( 'gemini_api_key', 'GEMINI_API_KEY', 'GEMINI_API_KEY' ),
			help='Overrides GEMINI_API_KEY from config.py for this session only.',
			key='sidebar_gemini_api_key' )
		
		xai_key = st.text_input( 'xAI API Key', type='password',
			value=get_runtime_config_value( 'xai_api_key', 'XAI_API_KEY', 'XAI_API_KEY' ),
			help='Overrides XAI_API_KEY from config.py for this session only.',
			key='sidebar_xai_api_key' )
		
		google_key = st.text_input( 'Google API Key', type='password',
			value=get_runtime_config_value( 'google_api_key', 'GOOGLE_API_KEY', 'GOOGLE_API_KEY' ),
			help='Overrides GOOGLE_API_KEY from config.py for this session only.',
			key='sidebar_google_api_key' )
		
		google_cse_id = st.text_input( 'Google CSE ID', type='password',
			value=get_runtime_config_value( 'google_cse_id', 'GOOGLE_CSE_ID', 'GOOGLE_CSE_ID' ),
			help='Overrides GOOGLE_CSE_ID from config.py for this session only.',
			key='sidebar_google_cse_id' )
		
		google_cloud_project_id = st.text_input( 'Google Cloud Project ID', type='password',
			value=get_runtime_config_value( 'google_cloud_project_id', 'GOOGLE_CLOUD_PROJECT_ID',
				'GOOGLE_CLOUD_PROJECT_ID' ),
			help='Overrides GOOGLE_CLOUD_PROJECT_ID from config.py for this session only.',
			key='sidebar_google_cloud_project_id' )
		
		google_cloud_location = st.text_input( 'Google Cloud Location', type='password',
			value=get_runtime_config_value( 'google_cloud_location', 'GOOGLE_CLOUD_LOCATION',
				'GOOGLE_CLOUD_LOCATION' ),
			help='Overrides GOOGLE_CLOUD_LOCATION from config.py for this session only.',
			key='sidebar_google_cloud_location' )
		
		sync_provider_config( 'openai_api_key', 'OPENAI_API_KEY', 'OPENAI_API_KEY', openai_key,
			'GPT' )
		sync_provider_config( 'gemini_api_key', 'GEMINI_API_KEY', 'GEMINI_API_KEY', gemini_key,
			'Gemini' )
		sync_provider_config( 'xai_api_key', 'XAI_API_KEY', 'XAI_API_KEY', xai_key, 'Grok' )
		sync_provider_config( 'google_api_key', 'GOOGLE_API_KEY', 'GOOGLE_API_KEY', google_key )
		sync_provider_config( 'google_cse_id', 'GOOGLE_CSE_ID', 'GOOGLE_CSE_ID', google_cse_id )
		sync_provider_config( 'google_cloud_project_id', 'GOOGLE_CLOUD_PROJECT_ID',
			'GOOGLE_CLOUD_PROJECT_ID', google_cloud_project_id )
		sync_provider_config( 'google_cloud_location', 'GOOGLE_CLOUD_LOCATION',
			'GOOGLE_CLOUD_LOCATION', google_cloud_location )

# ==============================================================================
# Page Setup
# ==============================================================================
AVATARS = { 'user': cfg.ANALYST, 'assistant': cfg.BOO, }
st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON,
	initial_sidebar_state='collapsed', )

style_subheaders( )
st.caption( cfg.APP_SUBTITLE )
inject_response_css( )
init_state( )

# ======================================================================================
# SIDEBAR
# ======================================================================================
with st.sidebar:
	provider_options = get_provider_options( )
	current_provider = st.session_state.get( 'provider', 'GPT' )
	if current_provider not in provider_options:
		current_provider = provider_options[ 0 ] if provider_options else 'GPT'
		st.session_state[ 'provider' ] = current_provider
	
	st.divider( )
	
	# ------------------------------------------------------------------
	# Provider Selection
	# ------------------------------------------------------------------
	with st.expander( 'Providers:', expanded=True ):
		provider = st.radio( label='Select Provider', options=provider_options,
			index=provider_options.index( current_provider ), key='provider' )
	
	logo_path = getattr( cfg, 'LOGO_MAP', { } ).get( provider )
	if logo_path:
		st.logo( logo_path, size='large' )
	
	st.divider( )
	
	mode_options = get_supported_provider_modes( provider )
	current_mode = normalize_mode_name( st.session_state.get( 'mode', 'Text' ) )
	if current_mode not in mode_options:
		current_mode = mode_options[ 0 ]
		st.session_state[ 'mode' ] = current_mode
	
	with st.expander( 'Modes:', expanded=False ):
		mode = st.radio( label='Select Mode', options=mode_options,
			index=get_mode_index( mode_options, current_mode ), key='mode' )
	
	st.caption( f'Provider: {provider} | Mode: {mode}' )
	st.divider( )
	render_provider_keys( )

# ======================================================================================
# TEXT MODE
# ======================================================================================
if mode == 'Text':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	text = get_chat_module( provider_name )
	
	# ------------------------------------------------------------------
	# Text Mode State Safety
	# ------------------------------------------------------------------
	text_defaults: Dict[ str, Any ] = { 'text_messages': [ ], 'text_model': '',
		'text_reasoning': '', 'text_temperature': 0.0, 'text_top_percent': 0.0, 'text_top_k': 0,
		'text_frequency_penalty': 0.0, 'text_presence_penalty': 0.0, 'text_max_tokens': 0,
		'text_tools': [ ], 'text_include': [ ], 'text_tool_choice': '', 'text_max_calls': 0,
		'text_parallel_tools': False, 'text_google_grounding': False, 'text_urls_input': '',
		'text_max_urls': 0, 'text_domains_input': '', 'text_vector_store_ids': '',
		'text_grok_collection_labels': [ ], 'text_grok_collection_ids': [ ],
		'text_grok_collection_ids_input': '', 'text_response_format': '',
		'text_response_schema': '', 'text_json_schema': '',
		'text_json_schema_name': 'response_schema', 'text_json_schema_strict': True,
		'text_stops_input': '', 'text_store': False, 'text_stream': False, 'text_background': False,
		'text_continuation_mode': 'None', 'text_previous_response_id': '',
		'text_conversation_id': '', 'text_input': 'conversation', 'text_system_instructions': '',
		'text_safety_profile': '', 'text_prompt_category': None, 'text_prompt_id': None,
		'text_context': [ ], 'last_answer': '', 'last_sources': [ ], }
	
	for state_key, state_default in text_defaults.items( ):
		if state_key not in st.session_state:
			st.session_state[ state_key ] = state_default
	
	if not isinstance( st.session_state.get( 'text_messages' ), list ):
		st.session_state[ 'text_messages' ] = [ ]
	
	if not isinstance( st.session_state.get( 'text_context' ), list ):
		st.session_state[ 'text_context' ] = [ ]
	
	if not isinstance( st.session_state.get( 'text_tools' ), list ):
		st.session_state[ 'text_tools' ] = [ ]
	
	if not isinstance( st.session_state.get( 'text_include' ), list ):
		st.session_state[ 'text_include' ] = [ ]
	
	if not isinstance( st.session_state.get( 'text_grok_collection_labels' ), list ):
		st.session_state[ 'text_grok_collection_labels' ] = [ ]
	
	if not isinstance( st.session_state.get( 'text_grok_collection_ids' ), list ):
		st.session_state[ 'text_grok_collection_ids' ] = [ ]
	
	# ----- Text Generation Utilities -------
	def get_text_options( instance: Any, attr_name: str,
		fallback: Optional[ List[ str ] ] = None ) -> List[ str ]:
		"""Get text options.
		
		Purpose:
			Returns normalized option values exposed by the selected provider wrapper.
		
		Args:
			instance (Any): Provider wrapper instance.
			attr_name (str): Option-property name.
			fallback (Optional[List[str]]): Values returned when the property is unavailable.
		
		Returns:
			List[str]: Normalized provider option values.
		"""
		values = getattr( instance, attr_name, None )
		
		if callable( values ):
			values = values( )
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if not isinstance( values, list ):
			return fallback or [ ]
		
		return [ str( value ) for value in values if str( value ).strip( ) ]
	
	def parse_semicolon_list( value: Any ) -> List[ str ]:
		"""Parse semicolon list.
		
		Purpose:
			Converts a semicolon-delimited value into normalized non-empty entries.
		
		Args:
			value (Any): Delimited source value.
		
		Returns:
			List[str]: Parsed values.
		"""
		return [ item.strip( ) for item in str( value or '' ).split( ';' ) if item.strip( ) ]
	
	def parse_comma_list( value: Any ) -> List[ str ]:
		"""Parse comma list.
		
		Purpose:
			Converts a comma-delimited value into normalized non-empty entries.
		
		Args:
			value (Any): Delimited source value.
		
		Returns:
			List[str]: Parsed values.
		"""
		return [ item.strip( ) for item in str( value or '' ).split( ',' ) if item.strip( ) ]
	
	def get_grok_collection_options( ) -> Dict[ str, str ]:
		"""Get Grok collection options.
		
		Purpose:
			Returns configured xAI Collection labels mapped to provider collection identifiers.
		
		Returns:
			Dict[str, str]: Collection labels mapped to identifiers.
		"""
		configured_collections = getattr( cfg, 'GROK_COLLECTIONS', { }, )
		collections: Dict[ str, str ] = { }
		
		if isinstance( configured_collections, dict ):
			for label, collection_id in configured_collections.items( ):
				if str( label ).strip( ) and str( collection_id ).strip( ):
					collections[ str( label ) ] = str( collection_id )
			
			return collections
		
		if isinstance( configured_collections, list ):
			for row in configured_collections:
				if not isinstance( row, dict ):
					continue
				
				for label, collection_id in row.items( ):
					if str( label ).strip( ) and str( collection_id ).strip( ):
						collections[ str( label ) ] = str( collection_id )
		
		return collections
	
	def get_selected_grok_collection_ids( ) -> List[ str ]:
		"""Get selected Grok collection IDs.
		
		Purpose:
			Resolves configured collection labels and manually entered collection identifiers.
		
		Returns:
			List[str]: Unique xAI Collection identifiers.
		"""
		collection_map = get_grok_collection_options( )
		selected_labels = st.session_state.get( 'text_grok_collection_labels', [ ], )
		manual_ids = parse_comma_list(
			st.session_state.get( 'text_grok_collection_ids_input', '', ) )
		resolved_ids: List[ str ] = [ ]
		
		for label in selected_labels:
			collection_id = collection_map.get( str( label ), '', )
			
			if collection_id and collection_id not in resolved_ids:
				resolved_ids.append( collection_id )
		
		for collection_id in manual_ids:
			if collection_id not in resolved_ids:
				resolved_ids.append( collection_id )
		
		st.session_state[ 'text_grok_collection_ids' ] = resolved_ids
		return resolved_ids
	
	def sanitize_text_selection( key: str, valid_options: List[ str ], default: Any = '' ) -> None:
		"""Sanitize text selection.
		
		Purpose:
			Removes a stale single-select value when it is not supported by the selected
			provider.
		
		Args:
			key (str): Session-state key.
			valid_options (List[str]): Supported option values.
			default (Any): Replacement value.
		
		Returns:
			None: This function updates session state.
		"""
		current_value = st.session_state.get( key, default, )
		
		if current_value in [ None, '', ]:
			return
		
		if valid_options and current_value not in valid_options:
			st.session_state[ key ] = default
	
	def sanitize_text_multiselect( key: str, valid_options: List[ str ] ) -> None:
		"""Sanitize text multiselect.
		
		Purpose:
			Removes stale multi-select values that are unsupported by the selected provider.
		
		Args:
			key (str): Session-state key.
			valid_options (List[str]): Supported option values.
		
		Returns:
			None: This function updates session state.
		"""
		current_values = st.session_state.get( key, [ ], )
		
		if not isinstance( current_values, list ):
			st.session_state[ key ] = [ ]
			return
		
		st.session_state[ key ] = [ value for value in current_values if value in valid_options ]
	
	def build_text_context( ) -> List[ Dict[ str, str ] ]:
		"""Build text context.
		
		Purpose:
			Returns prior Text Mode user and assistant messages without including the current
			prompt.
		
		Returns:
			List[Dict[str, str]]: Prior conversation messages.
		"""
		if st.session_state.get( 'text_input', 'conversation' ) != 'conversation':
			return [ ]
		
		messages = st.session_state.get( 'text_messages', [ ], )
		
		if not isinstance( messages, list ):
			return [ ]
		
		return [ { 'role': str( message.get( 'role', '', ) ),
			'content': str( message.get( 'content', '', ) ), } for message in messages if
			isinstance( message, dict ) and message.get( 'role' ) in [ 'user',
				'assistant', ] and str( message.get( 'content', '', ) ).strip( ) ]
	
	def normalize_text_domains( ) -> List[ str ]:
		"""Normalize allowed domains.
		
		Purpose:
			Removes URL schemes and paths from Web Search domain restrictions.
		
		Returns:
			List[str]: Provider-compatible domain names.
		"""
		domains: List[ str ] = [ ]
		
		for entered_domain in parse_comma_list( st.session_state.get( 'text_domains_input', '',
		) ):
			domain = entered_domain.strip( )
			
			if domain.startswith( 'https://' ):
				domain = domain[ len( 'https://' ): ]
			elif domain.startswith( 'http://' ):
				domain = domain[ len( 'http://' ): ]
			
			domain = domain.split( '/', 1, )[ 0 ].strip( )
			
			if domain and domain not in domains:
				domains.append( domain )
		
		return domains
	
	def get_text_response_schema( required: bool = False ) -> Optional[ Dict[ str, Any ] ]:
		"""Get text response schema.
		
		Purpose:
			Parses the configured JSON schema only when the selected response mode requires it.
		
		Args:
			required (bool): Indicates whether an empty schema is invalid.
		
		Returns:
			Optional[Dict[str, Any]]: Parsed JSON schema or None.
		
		Raises:
			ValueError: Raised when required schema text is missing or invalid.
		"""
		schema_text = str( st.session_state.get( 'text_json_schema', '', ) or '' ).strip( )
		
		if not schema_text:
			if required:
				raise ValueError( 'Response Schema is required when JSON Schema is selected.' )
			
			return None
		
		response_schema = json.loads( schema_text )
		if not isinstance( response_schema, dict ):
			raise ValueError( 'Response Schema must contain a JSON object.' )
		
		return response_schema
	
	def get_gpt_text_format( ) -> Any:
		"""Get GPT text format.
		
		Purpose:
			Builds the response-format value accepted by the GPT Chat replacement.
		
		Returns:
			Any: GPT response-format configuration.
		"""
		selected_format = str( st.session_state.get( 'text_response_format', '', ) or '' ).strip( )
		
		if not selected_format or selected_format == 'text':
			return None
		
		if selected_format == 'json_object':
			return { 'type': 'json_object', }
		
		if selected_format == 'json_schema':
			return { 'type': 'json_schema', 'name': str(
				st.session_state.get( 'text_json_schema_name',
					'response_schema', ) or 'response_schema' ).strip( ),
				'schema': get_text_response_schema( required=True ),
				'strict': bool( st.session_state.get( 'text_json_schema_strict', True, ) ), }
		
		return selected_format
	
	def get_gemini_text_schema( ) -> Optional[ Dict[ str, Any ] ]:
		"""Get Gemini text schema.
		
		Purpose:
			Returns a Gemini response schema only when JSON output is selected and schema text
			has been entered.
		
		Returns:
			Optional[Dict[str, Any]]: Parsed schema or None.
		"""
		selected_format = str( st.session_state.get( 'text_response_format', '', ) or '' ).strip( )
		
		if selected_format != 'application/json':
			return None
		
		return get_text_response_schema( required=False )
	
	def get_grok_text_schema( ) -> Any:
		"""Get Grok text schema.
		
		Purpose:
			Returns the required xAI schema only when JSON Schema output is selected.
		
		Returns:
			Any: Parsed schema or None.
		"""
		selected_format = str( st.session_state.get( 'text_response_format', '', ) or '' ).strip( )
		
		if selected_format != 'json_schema':
			return None
		
		return get_text_response_schema( required=True )
	
	def validate_text_request( ) -> None:
		"""Validate text request.
		
		Purpose:
			Prevents submission when the selected provider operation is missing required
			provider-specific controls.
		
		Returns:
			None: This function validates state.
		
		Raises:
			ValueError: Raised when required request values are missing.
		"""
		if not st.session_state.get( 'text_model' ):
			raise ValueError( 'Select a Text model before sending a prompt.' )
		
		selected_tools = st.session_state.get( 'text_tools', [ ], )
		
		if provider_name == 'GPT':
			continuation_mode = st.session_state.get( 'text_continuation_mode', 'None', )
			
			if (continuation_mode == 'Previous Response' and not str(
				st.session_state.get( 'text_previous_response_id', '', ) or '' ).strip( )):
				raise ValueError(
					'Enter a Previous Response ID or select another continuation mode.' )
			
			if (continuation_mode == 'Conversation' and not str(
				st.session_state.get( 'text_conversation_id', '', ) or '' ).strip( )):
				raise ValueError( 'Enter a Conversation ID or select another continuation mode.' )
			
			if ('file_search' in selected_tools and not parse_comma_list(
				st.session_state.get( 'text_vector_store_ids', '', ) )):
				raise ValueError(
					'Enter at least one GPT Vector Store ID when File Search is selected.' )
		
		if provider_name == 'Gemini':
			if ('file_search' in selected_tools and not parse_comma_list(
				st.session_state.get( 'text_vector_store_ids', '', ) )):
				raise ValueError( 'Enter at least one Gemini File Search Store resource name.' )
			
			if ('url_context' in selected_tools and not parse_semicolon_list(
				st.session_state.get( 'text_urls_input', '', ) )):
				raise ValueError( 'Enter at least one URL when URL Context is selected.' )
		
		if (
				provider_name == 'Grok' and 'collections_search' in selected_tools and not
		get_selected_grok_collection_ids( )):
			raise ValueError( 'Select or enter at least one xAI Collection.' )
		
		selected_format = str( st.session_state.get( 'text_response_format', '', ) or '' )
		
		if (provider_name in [ 'GPT', 'Grok', ] and selected_format == 'json_schema'):
			get_text_response_schema( required=True )
	
	def reset_text_model_settings( ) -> None:
		"""Reset text model settings.
		
		Purpose:
			Resets Text Mode model and provider-capability selections.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'text_model' ] = ''
		st.session_state[ 'text_reasoning' ] = ''
		st.session_state[ 'text_safety_profile' ] = ''
	
	def reset_text_inference_settings( ) -> None:
		"""Reset text inference settings.
		
		Purpose:
			Resets Text Mode sampling and penalty controls.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'text_temperature' ] = 0.0
		st.session_state[ 'text_top_percent' ] = 0.0
		st.session_state[ 'text_top_k' ] = 0
		st.session_state[ 'text_frequency_penalty' ] = 0.0
		st.session_state[ 'text_presence_penalty' ] = 0.0
	
	def reset_text_tool_settings( ) -> None:
		"""Reset text tool settings.
		
		Purpose:
			Resets Text Mode tools, grounding, domains, URLs, and retrieval resources.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'text_tools' ] = [ ]
		st.session_state[ 'text_include' ] = [ ]
		st.session_state[ 'text_tool_choice' ] = ''
		st.session_state[ 'text_max_calls' ] = 0
		st.session_state[ 'text_parallel_tools' ] = False
		st.session_state[ 'text_google_grounding' ] = False
		st.session_state[ 'text_urls_input' ] = ''
		st.session_state[ 'text_max_urls' ] = 0
		st.session_state[ 'text_domains_input' ] = ''
		st.session_state[ 'text_vector_store_ids' ] = ''
		st.session_state[ 'text_grok_collection_labels' ] = [ ]
		st.session_state[ 'text_grok_collection_ids' ] = [ ]
		st.session_state[ 'text_grok_collection_ids_input' ] = ''
	
	def reset_text_response_settings( ) -> None:
		"""Reset text response settings.
		
		Purpose:
			Resets Text Mode output, structured-response, streaming, storage, and continuation
			controls.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'text_max_tokens' ] = 0
		st.session_state[ 'text_response_format' ] = ''
		st.session_state[ 'text_response_schema' ] = ''
		st.session_state[ 'text_json_schema' ] = ''
		st.session_state[ 'text_json_schema_name' ] = 'response_schema'
		st.session_state[ 'text_json_schema_strict' ] = True
		st.session_state[ 'text_stops_input' ] = ''
		st.session_state[ 'text_store' ] = False
		st.session_state[ 'text_stream' ] = False
		st.session_state[ 'text_background' ] = False
		st.session_state[ 'text_continuation_mode' ] = 'None'
		st.session_state[ 'text_previous_response_id' ] = ''
		st.session_state[ 'text_conversation_id' ] = ''
	
	def clear_text_messages( ) -> None:
		"""Clear text messages.
		
		Purpose:
			Clears Text Mode conversation, continuation, answer, and source state.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'text_messages' ] = [ ]
		st.session_state[ 'text_context' ] = [ ]
		st.session_state[ 'text_previous_response_id' ] = ''
		st.session_state[ 'text_conversation_id' ] = ''
		st.session_state[ 'last_answer' ] = ''
		st.session_state[ 'last_sources' ] = [ ]
	
	def build_gpt_text_kwargs( prompt: str, prior_context: List[ Dict[ str, str ] ],
		stream_handler: Any = None ) -> Dict[ str, Any ]:
		"""Build GPT text arguments.
		
		Purpose:
			Builds only arguments accepted by the GPT Chat replacement.
		
		Args:
			prompt (str): Current user prompt.
			prior_context (List[Dict[str, str]]): Prior conversation messages.
			stream_handler (Any): Optional streaming callback.
		
		Returns:
			Dict[str, Any]: GPT text-generation arguments.
		"""
		selected_tools = list( st.session_state.get( 'text_tools', [ ], ) )
		continuation_mode = st.session_state.get( 'text_continuation_mode', 'None', )
		previous_id = ''
		conversation_id = ''
		
		if continuation_mode == 'Previous Response':
			previous_id = str(
				st.session_state.get( 'text_previous_response_id', '', ) or '' ).strip( )
		elif continuation_mode == 'Conversation':
			conversation_id = str(
				st.session_state.get( 'text_conversation_id', '', ) or '' ).strip( )
		
		return { 'prompt': prompt, 'model': st.session_state.get( 'text_model', '', ),
			'temperature': float( st.session_state.get( 'text_temperature', 0.0, ) ),
			'format': get_gpt_text_format( ),
			'top_p': float( st.session_state.get( 'text_top_percent', 0.0, ) ),
			'frequency': float( st.session_state.get( 'text_frequency_penalty', 0.0, ) ),
			'max_tools': int( st.session_state.get( 'text_max_calls', 0, ) ),
			'presence': float( st.session_state.get( 'text_presence_penalty', 0.0, ) ),
			'max_tokens': int( st.session_state.get( 'text_max_tokens', 0, ) ),
			'store': bool( st.session_state.get( 'text_store', False, ) ),
			'stream': bool( st.session_state.get( 'text_stream', False, ) ),
			'instruct': str( st.session_state.get( 'text_system_instructions', '', ) or '' ),
			'background': bool( st.session_state.get( 'text_background', False, ) ),
			'reasoning': str( st.session_state.get( 'text_reasoning', '', ) or '' ),
			'include': list( st.session_state.get( 'text_include', [ ], ) ),
			'tools': selected_tools, 'allowed_domains': (
				normalize_text_domains( ) if 'web_search' in selected_tools else [ ]),
			'previous_id': previous_id,
			'tool_choice': str( st.session_state.get( 'text_tool_choice', '', ) or '' ),
			'is_parallel': bool( st.session_state.get( 'text_parallel_tools', False, ) ),
			'context': prior_context, 'input_data': None, 'vector_store_ids': (parse_comma_list(
				st.session_state.get( 'text_vector_store_ids',
					'', ) ) if 'file_search' in selected_tools else [ ]),
			'conversation_id': conversation_id, }
	
	def build_gemini_text_kwargs( prompt: str, prior_context: List[ Dict[ str, str ] ],
		stream_handler: Any = None ) -> Dict[ str, Any ]:
		"""Build Gemini text arguments.
		
		Purpose:
			Builds only arguments accepted by the Gemini Chat replacement.
		
		Args:
			prompt (str): Current user prompt.
			prior_context (List[Dict[str, str]]): Prior conversation messages.
			stream_handler (Any): Optional streaming callback.
		
		Returns:
			Dict[str, Any]: Gemini text-generation arguments.
		"""
		selected_tools = list( st.session_state.get( 'text_tools', [ ], ) )
		
		if (st.session_state.get( 'text_google_grounding',
			False, ) and 'google_search' not in selected_tools):
			selected_tools.append( 'google_search' )
		
		return { 'prompt': prompt, 'model': st.session_state.get( 'text_model', '', ), 'number': 1,
			'temperature': float( st.session_state.get( 'text_temperature', 0.0, ) ),
			'top_p': float( st.session_state.get( 'text_top_percent', 0.0, ) ),
			'top_k': int( st.session_state.get( 'text_top_k', 0, ) ),
			'frequency': float( st.session_state.get( 'text_frequency_penalty', 0.0, ) ),
			'presence': float( st.session_state.get( 'text_presence_penalty', 0.0, ) ),
			'max_tokens': int( st.session_state.get( 'text_max_tokens', 0, ) ),
			'stops': parse_comma_list( st.session_state.get( 'text_stops_input', '', ) ),
			'instruct': str( st.session_state.get( 'text_system_instructions', '', ) or '' ),
			'response_format': str( st.session_state.get( 'text_response_format', '', ) or '' ),
			'tools': selected_tools, 'tool_choice': '',
			'reasoning': str( st.session_state.get( 'text_reasoning', '', ) or '' ),
			'modalities': [ ], 'media_resolution': '', 'context': prior_context, 'content': '',
			'urls': (parse_semicolon_list( st.session_state.get( 'text_urls_input',
				'', ) ) if 'url_context' in selected_tools else [ ]),
			'max_urls': int( st.session_state.get( 'text_max_urls', 0, ) ),
			'response_schema': get_gemini_text_schema( ),
			'safety_profile': str( st.session_state.get( 'text_safety_profile', '', ) or '' ),
			'file_search_store_names': (parse_comma_list(
				st.session_state.get( 'text_vector_store_ids',
					'', ) ) if 'file_search' in selected_tools else [ ]),
			'stream': bool( st.session_state.get( 'text_stream', False, ) ),
			'stream_handler': stream_handler, }
	
	def build_grok_text_kwargs( prompt: str, prior_context: List[ Dict[ str, str ] ],
		stream_handler: Any = None ) -> Dict[ str, Any ]:
		"""Build Grok text arguments.
		
		Purpose:
			Builds only arguments accepted by the Grok Chat replacement.
		
		Args:
			prompt (str): Current user prompt.
			prior_context (List[Dict[str, str]]): Prior conversation messages.
			stream_handler (Any): Optional streaming callback.
		
		Returns:
			Dict[str, Any]: Grok text-generation arguments.
		"""
		selected_tools = list( st.session_state.get( 'text_tools', [ ], ) )
		
		return { 'prompt': prompt, 'model': st.session_state.get( 'text_model', '', ),
			'temperature': float( st.session_state.get( 'text_temperature', 0.0, ) ),
			'format': str( st.session_state.get( 'text_response_format', '', ) or '' ),
			'top_p': float( st.session_state.get( 'text_top_percent', 0.0, ) ),
			'frequency': float( st.session_state.get( 'text_frequency_penalty', 0.0, ) ),
			'presence': float( st.session_state.get( 'text_presence_penalty', 0.0, ) ),
			'max_tokens': int( st.session_state.get( 'text_max_tokens', 0, ) ),
			'stops': parse_comma_list( st.session_state.get( 'text_stops_input', '', ) ),
			'store': bool( st.session_state.get( 'text_store', False, ) ),
			'stream': bool( st.session_state.get( 'text_stream', False, ) ),
			'instruct': str( st.session_state.get( 'text_system_instructions', '', ) or '' ),
			'reasoning': str( st.session_state.get( 'text_reasoning', '', ) or '' ),
			'include': list( st.session_state.get( 'text_include', [ ], ) ),
			'tools': selected_tools, 'allowed_domains': (
				normalize_text_domains( ) if 'web_search' in selected_tools else [ ]),
			'previous_id': str(
				st.session_state.get( 'text_previous_response_id', '', ) or '' ).strip( ),
			'tool_choice': str( st.session_state.get( 'text_tool_choice', '', ) or '' ),
			'is_parallel': bool( st.session_state.get( 'text_parallel_tools', False, ) ),
			'context': prior_context, 'vector_store_ids': (
				get_selected_grok_collection_ids( ) if 'collections_search' in selected_tools else
				[ ]),
			'max_tools': int( st.session_state.get( 'text_max_calls', 0, ) ),
			'response_schema': get_grok_text_schema( ), 'stream_handler': stream_handler, }
	
	def call_generate_text( prompt: str, prior_context: List[ Dict[ str, str ] ],
		stream_handler: Any = None ) -> Any:
		"""Call generate text.
		
		Purpose:
			Dispatches Text Mode through the exact replacement interface for the selected
			provider.
		
		Args:
			prompt (str): Current user prompt.
			prior_context (List[Dict[str, str]]): Prior conversation messages.
			stream_handler (Any): Optional streaming callback.
		
		Returns:
			Any: Provider-generated text.
		"""
		if provider_name == 'GPT':
			return text.generate_text(
				**build_gpt_text_kwargs( prompt, prior_context, stream_handler, ) )
		
		if provider_name == 'Gemini':
			return text.generate_text(
				**build_gemini_text_kwargs( prompt, prior_context, stream_handler, ) )
		
		if provider_name == 'Grok':
			return text.generate_text(
				**build_grok_text_kwargs( prompt, prior_context, stream_handler, ) )
		
		raise ValueError( f'Unsupported Text provider: {provider_name}' )
	
	def get_text_avatar( role: str ) -> str:
		"""Get text avatar.
		
		Purpose:
			Returns the configured assistant avatar for the selected provider.
		
		Args:
			role (str): Message role.
		
		Returns:
			str: Avatar value.
		"""
		if role != 'assistant':
			return ''
		
		if provider_name == 'GPT':
			return getattr( cfg, 'GPT', getattr( cfg, 'BOO', '🧠' ), )
		
		if provider_name == 'Gemini':
			return getattr( cfg, 'JENI', getattr( cfg, 'BOO', '🧠' ), )
		
		if provider_name == 'Grok':
			return getattr( cfg, 'GROK', getattr( cfg, 'BOO', '🧠' ), )
		
		return getattr( cfg, 'BOO', '🧠', )
	
	def extract_text_sources( instance: Any, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract text sources.
		
		Purpose:
			Extracts provider-specific grounding and search sources from the latest response.
		
		Args:
			instance (Any): Selected provider wrapper.
			response (Any): Provider response.
		
		Returns:
			List[Dict[str, Any]]: Normalized source records.
		"""
		sources: List[ Dict[ str, Any ] ] = [ ]
		
		if (provider_name == 'Gemini' and hasattr( instance, 'get_grounding_sources', )):
			result = instance.get_grounding_sources( )
			
			if isinstance( result, list ):
				sources = result
		
		elif (provider_name in [ 'GPT', 'Grok', ] and hasattr( instance, 'get_sources', )):
			result = instance.get_sources( )
			
			if isinstance( result, list ):
				sources = result
		
		elif 'extract_sources' in globals( ):
			result = extract_sources( response )
			
			if isinstance( result, list ):
				sources = result
		
		return sources
	
	def update_text_usage( response: Any ) -> None:
		"""Update text usage.
		
		Purpose:
			Updates application token counters from the latest provider response when the
			application usage helper is available.
		
		Args:
			response (Any): Provider response.
		
		Returns:
			None: This function updates application state.
		"""
		if 'update_token_counters' in globals( ):
			update_token_counters( response )
	
	def load_text_instruction_template( ) -> None:
		"""Load Text instruction template.
		
		Purpose:
		    Loads the selected prompt template into the Text Mode system-instruction field.
		
		Returns:
		    None: This function updates Text Mode session state.
		"""
		load_prompt_template( prompt_id_key='text_prompt_id',
			instructions_key='text_system_instructions', )
	
	def clear_text_instructions( ) -> None:
		"""Clear Text instructions.
		
		Purpose:
		    Clears the Text Mode system instructions and its selected prompt template.
		
		Returns:
		    None: This function updates Text Mode session state.
		"""
		st.session_state[ 'text_system_instructions' ] = ''
		st.session_state[ 'text_prompt_id' ] = None
	
	def convert_text_system_instructions( ) -> None:
		"""Convert Text system instructions.
		
		Purpose:
		    Converts the Text Mode system instructions between Markdown headings and XML-style
		    heading elements.
		
		Returns:
		    None: This function updates Text Mode session state.
		"""
		instructions = str( st.session_state.get( 'text_system_instructions', '' ) or '' )
		
		if not instructions.strip( ):
			return
		
		st.session_state[ 'text_system_instructions' ] = convert_markdown( instructions )
		
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05, ] )
	
	with center:
		st.subheader( '💬 Text Generation', help=cfg.TEXT_GENERATION, )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Expander — Text Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch', ):
			# ------------------------------------------------------------------
			# Expander — Model Settings
			# ------------------------------------------------------------------
			with st.expander( label='Model Settings', icon='🧊', expanded=False,
					width='stretch', ):
				model_options = get_text_options( text, 'model_options', )
				reasoning_options = get_text_options( text, 'reasoning_options', )
				
				if not model_options:
					default_model = str( getattr( text, 'model', '', ) or '' )
					
					if default_model:
						model_options = [ default_model, ]
				
				sanitize_text_selection( 'text_model', model_options, '', )
				sanitize_text_selection( 'text_reasoning', reasoning_options, '', )
				model_c1, model_c2, model_c3 = st.columns( [ 0.34, 0.33, 0.33, ], border=True,
					gap='xxsmall', )
				
				# ---------- Model ------------
				with model_c1:
					st.selectbox( label='Model', options=model_options, key='text_model',
						index=None, placeholder='Select Model',
						help='Required. Select the provider model used for Text generation.', )
				
				# ---------- Reasoning ------------
				with model_c2:
					st.selectbox( label='Reasoning', options=reasoning_options,
						key='text_reasoning', index=None, placeholder='Options',
						disabled=not reasoning_options, help=cfg.REASONING, )
				
				# ---------- Input Mode ------------
				with model_c3:
					st.selectbox( label='Input Mode', options=[ 'conversation', 'single_turn', ],
						key='text_input', help=('Conversation includes prior Text Mode messages. '
						                        'Single turn sends only the current prompt.'), )
				
				if provider_name == 'Gemini':
					safety_options = get_text_options( text, 'safety_options', )
					
					if safety_options:
						sanitize_text_selection( 'text_safety_profile', safety_options, '', )
						
						# ---------- Safety Profile ------------
						st.selectbox( label='Safety Profile', options=safety_options,
							key='text_safety_profile', index=None, placeholder='Options',
							help='Optional. Gemini safety configuration profile.', )
				
				st.button( label='Reset', key='text_model_reset', width='stretch',
					on_click=reset_text_model_settings, icon='🔄', )
			
			# ------------------------------------------------------------------
			# Expander — Inference Settings
			# ------------------------------------------------------------------
			with st.expander( label='Inference Settings', icon='🎚️', expanded=False,
					width='stretch', ):
				if provider_name == 'Gemini':
					inf_c1, inf_c2, inf_c3, inf_c4, inf_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20, ], border=True, gap='xxsmall', )
				else:
					inf_c1, inf_c2, inf_c3, inf_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25, ],
						border=True, gap='xxsmall', )
				
				# ---------- Top-P ------------
				with inf_c1:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0, step=0.01,
						help=cfg.TOP_P, key='text_top_percent', )
				
				# ---------- Temperature ------------
				with inf_c2:
					st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.01,
						help=cfg.TEMPERATURE, key='text_temperature', )
				
				if provider_name == 'Gemini':
					# ---------- Top-K ------------
					with inf_c3:
						st.slider( label='Top-K', min_value=0, max_value=200, step=1,
							help=cfg.TOP_K, key='text_top_k', )
					
					# ---------- Frequency Penalty ------------
					with inf_c4:
						st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
							step=0.01, help=cfg.FREQUENCY_PENALTY, key='text_frequency_penalty', )
					
					# ---------- Presence Penalty ------------
					with inf_c5:
						st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
							step=0.01, help=cfg.PRESENCE_PENALTY, key='text_presence_penalty', )
				else:
					# ---------- Frequency Penalty ------------
					with inf_c3:
						st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
							step=0.01, help=cfg.FREQUENCY_PENALTY, key='text_frequency_penalty', )
					
					# ---------- Presence Penalty ------------
					with inf_c4:
						st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
							step=0.01, help=cfg.PRESENCE_PENALTY, key='text_presence_penalty', )
				
				st.button( label='Reset', key='text_inference_reset', width='stretch',
					on_click=reset_text_inference_settings, icon='🔄', )
			
			# ------------------------------------------------------------------
			# Expander — Grounding Settings
			# ------------------------------------------------------------------
			with st.expander( label='Tools / Grounding Settings', icon='🔎', expanded=False,
					width='stretch', ):
				tool_options = get_text_options( text, 'tool_options', )
				include_options = get_text_options( text, 'include_options', )
				choice_options = get_text_options( text, 'choice_options', )
				
				sanitize_text_multiselect( 'text_tools', tool_options, )
				sanitize_text_multiselect( 'text_include', include_options, )
				sanitize_text_selection( 'text_tool_choice', choice_options, '', )
				
				selected_tools = st.session_state.get( 'text_tools', [ ], )
				gemini_builtin_tools = { 'google_search', 'url_context', 'file_search',
					'code_execution', }
				gemini_function_tool_selected = (provider_name == 'Gemini' and any(
					selected_tool not in gemini_builtin_tools for selected_tool in
						selected_tools ))
				
				tool_c1, tool_c2, tool_c3 = st.columns( [ 0.34, 0.33, 0.33, ], border=True,
					gap='xxsmall', )
				
				# ---------- Tools ------------
				with tool_c1:
					st.multiselect( label='Tools', options=tool_options, key='text_tools',
						placeholder='Options', help=cfg.TOOLS, )
				
				# ---------- Include ------------
				with tool_c2:
					st.multiselect( label='Include', options=include_options, key='text_include',
						placeholder='Options', disabled=not include_options, help=cfg.INCLUDE, )
				
				# ---------- Tool Choice ------------
				with tool_c3:
					st.selectbox( label='Tool Choice', options=choice_options,
						key='text_tool_choice', index=None, placeholder='Options', disabled=(
								not choice_options or (
								provider_name == 'Gemini' and not gemini_function_tool_selected)),
						help=cfg.CHOICE, )
				
				selected_tools = st.session_state.get( 'text_tools', [ ], )
				
				if provider_name in [ 'GPT', 'Grok', ]:
					tool_c4, tool_c5 = st.columns( [ 0.50, 0.50, ], border=True, gap='xxsmall', )
					
					# ---------- Max Tool Calls ------------
					with tool_c4:
						st.slider( label='Max Tool Calls', min_value=0, max_value=100, step=1,
							key='text_max_calls', help=cfg.MAX_TOOL_CALLS, )
					
					# ---------- Parallel Tools ------------
					with tool_c5:
						st.toggle( label='Parallel Tools', key='text_parallel_tools',
							help=cfg.PARALLEL_TOOL_CALLS, )
				else:
					st.session_state[ 'text_max_calls' ] = 0
					st.session_state[ 'text_parallel_tools' ] = False
				
				if provider_name == 'Gemini':
					# ---------- Google Grounding ------------
					st.toggle( label='Google Grounding', key='text_google_grounding',
						help=('Adds Google Search grounding without replacing '
						      'other selected Gemini tools.'), )
				else:
					st.session_state[ 'text_google_grounding' ] = False
				
				if (provider_name == 'Gemini' and 'url_context' in selected_tools):
					url_c1, url_c2 = st.columns( [ 0.80, 0.20, ], border=True, gap='xxsmall', )
					
					# ---------- URLs ------------
					with url_c1:
						st.text_input( label='URLs', key='text_urls_input',
							help='Required for Gemini URL Context. Separate URLs with semicolons.',
							width='stretch', placeholder=('https://example.com/page-1;'
							                              'https://example.com/page-2'), )
					
					# ---------- Max URLs ------------
					with url_c2:
						st.slider( label='Max URLs', min_value=0, max_value=25, step=1,
							key='text_max_urls',
							help='Optional maximum number of URL Context sources.', )
				else:
					st.session_state[ 'text_urls_input' ] = ''
					st.session_state[ 'text_max_urls' ] = 0
				
				# ---------- Allowed Domains ------------
				if (provider_name in [ 'GPT', 'Grok', ] and 'web_search' in selected_tools):
					st.text_input( label='Allowed Domains', key='text_domains_input',
						help=cfg.ALLOWED_DOMAINS, width='stretch',
						placeholder='example.com,openai.com', )
				else:
					st.session_state[ 'text_domains_input' ] = ''
				
				# ---------- Vector Store / Collection IDs ------------
				if (provider_name == 'GPT' and 'file_search' in selected_tools):
					st.text_input( label='Vector Store IDs', key='text_vector_store_ids',
						help='Required for GPT File Search. Separate IDs with commas.',
						width='stretch', placeholder='vs_abc123,vs_def456', )
				
				elif (provider_name == 'Gemini' and 'file_search' in selected_tools):
					st.text_input( label='File Search Store Names', key='text_vector_store_ids',
						help=('Required for Gemini File Search. Enter complete resource names '
						      'separated with commas.'), width='stretch',
						placeholder=('fileSearchStores/store-a,'
						             'fileSearchStores/store-b'), )
				
				elif (provider_name == 'Grok' and 'collections_search' in selected_tools):
					collection_options = get_grok_collection_options( )
					collection_labels = list( collection_options.keys( ) )
					
					if collection_labels:
						st.multiselect( label='Collections', options=collection_labels,
							key='text_grok_collection_labels', placeholder='Options',
							help='Configured xAI Collections used by Collections Search.', )
					
					st.text_input( label='Collection IDs', key='text_grok_collection_ids_input',
						help='Enter additional xAI Collection IDs separated with commas.',
						width='stretch', placeholder='collection_abc123,collection_def456', )
				else:
					st.session_state[ 'text_vector_store_ids' ] = ''
					
					if provider_name != 'Grok':
						st.session_state[ 'text_grok_collection_labels' ] = [ ]
						st.session_state[ 'text_grok_collection_ids' ] = [ ]
						st.session_state[ 'text_grok_collection_ids_input' ] = ''
				
				st.button( label='Reset', key='reset_text_tools', width='stretch',
					on_click=reset_text_tool_settings, icon='🔄', )
			
			# ------------------------------------------------------------------
			# Expander — Response Settings
			# ------------------------------------------------------------------
			with st.expander( label='Output / Response Settings', icon='↔️', expanded=False,
					width='stretch', ):
				format_options = get_text_options( text, 'format_options', )
				sanitize_text_selection( 'text_response_format', format_options, '', )
				
				resp_c1, resp_c2, resp_c3, resp_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25, ],
					border=True, gap='xxsmall', )
				
				# ---------- Max Tokens ------------
				with resp_c1:
					st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
						help=cfg.MAX_OUTPUT_TOKENS, key='text_max_tokens', )
				
				# ---------- Response Format ------------
				with resp_c2:
					st.selectbox( label='Response Format', options=format_options,
						key='text_response_format',
						help='Optional. Desired provider-supported response format.', index=None,
						placeholder='Options', disabled=not format_options, )
				
				# ---------- Store ------------
				with resp_c3:
					if provider_name in [ 'GPT', 'Grok', ]:
						st.toggle( label='Store', key='text_store', help=cfg.STORE, )
					else:
						st.session_state[ 'text_store' ] = False
						st.toggle( label='Store', key='text_store', help=cfg.STORE,
							disabled=True, )
				
				# ---------- Stream ------------
				with resp_c4:
					st.toggle( label='Stream', key='text_stream', help=cfg.STREAM, )
				
				if provider_name == 'GPT':
					gpt_resp_c1, gpt_resp_c2 = st.columns( [ 0.50, 0.50, ], border=True,
						gap='xxsmall', )
					
					# ---------- Background ------------
					with gpt_resp_c1:
						st.toggle( label='Background', key='text_background',
							help=cfg.BACKGROUND_MODE, )
					
					# ---------- Continuation ------------
					with gpt_resp_c2:
						st.selectbox( label='Continuation',
							options=[ 'None', 'Previous Response', 'Conversation', ],
							key='text_continuation_mode',
							help=('Select one OpenAI continuation mechanism. '
							      'Previous Response and Conversation are mutually exclusive.'), )
					
					if (st.session_state.get( 'text_continuation_mode' ) == 'Previous Response'):
						# ---------- Previous Response ID ------------
						st.text_input( label='Previous Response ID',
							key='text_previous_response_id',
							help='Required when Previous Response continuation is selected.',
							width='stretch', placeholder='resp_...', )
						st.session_state[ 'text_conversation_id' ] = ''
					
					elif (st.session_state.get( 'text_continuation_mode' ) == 'Conversation'):
						# ---------- Conversation ID ------------
						st.text_input( label='Conversation ID', key='text_conversation_id',
							help='Required when Conversation continuation is selected.',
							width='stretch', placeholder='conv_...', )
						st.session_state[ 'text_previous_response_id' ] = ''
					
					else:
						st.session_state[ 'text_previous_response_id' ] = ''
						st.session_state[ 'text_conversation_id' ] = ''
				
				elif provider_name == 'Grok':
					st.session_state[ 'text_background' ] = False
					st.session_state[ 'text_continuation_mode' ] = 'Previous Response'
					st.session_state[ 'text_conversation_id' ] = ''
					
					# ---------- Previous Response ID ------------
					st.text_input( label='Previous Response ID', key='text_previous_response_id',
						help='Optional xAI stored-response continuation identifier.',
						width='stretch', placeholder='response_...', )
				
				else:
					st.session_state[ 'text_background' ] = False
					st.session_state[ 'text_continuation_mode' ] = 'None'
					st.session_state[ 'text_previous_response_id' ] = ''
					st.session_state[ 'text_conversation_id' ] = ''
				
				selected_format = str( st.session_state.get( 'text_response_format', '', ) or '' )
				show_json_schema = ((provider_name in [ 'GPT',
					'Grok', ] and selected_format == 'json_schema') or (
						                    provider_name == 'Gemini' and selected_format ==
						                    'application/json'))
				
				if show_json_schema:
					schema_c1, schema_c2, schema_c3 = st.columns( [ 0.25, 0.50, 0.25, ],
						border=True, gap='xxsmall', )
					
					# ---------- Schema Name ------------
					with schema_c1:
						st.text_input( label='Schema Name', key='text_json_schema_name',
							help='Name used for GPT or Grok JSON Schema output.', width='stretch',
							placeholder='response_schema', disabled=provider_name == 'Gemini', )
					
					# ---------- Response Schema ------------
					with schema_c2:
						st.text_area( label='Response Schema', key='text_json_schema',
							help=('Optional for Gemini JSON output. Required for GPT or '
							      'Grok json_schema output.'), height=100, width='stretch',
							placeholder=('{"type":"object","properties":'
							             '{"answer":{"type":"string"}}}'), )
						st.session_state[ 'text_response_schema' ] = (
							st.session_state.get( 'text_json_schema', '', ))
					
					# ---------- Strict Schema ------------
					with schema_c3:
						st.toggle( label='Strict Schema', key='text_json_schema_strict',
							help='Enforce strict JSON Schema when supported.',
							disabled=provider_name == 'Gemini', )
				
				# ---------- Stop Sequences ------------
				st.text_input( label='Stop Sequences', key='text_stops_input',
					help=cfg.STOP_SEQUENCE, width='stretch', placeholder='END,STOP,DONE',
					disabled=provider_name == 'GPT', )
				
				st.button( label='Reset', key='text_response_reset', width='stretch',
					on_click=reset_text_response_settings, icon='🔄', )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch', ):
			in_left, in_right = st.columns( [ 0.80, 0.20, ] )
			
			# ------------------------------------------------------------------
			# Text Prompt Categories
			# ------------------------------------------------------------------
			text_prompt_categories = fetch_prompt_categories( 'Text' )
			current_text_category = st.session_state.get( 'text_prompt_category' )
			
			if current_text_category not in text_prompt_categories:
				st.session_state[ 'text_prompt_category' ] = None
			
			selected_text_category = st.session_state.get( 'text_prompt_category' )
			text_prompt_options = (
				fetch_prompt_options( selected_text_category ) if selected_text_category else [ ])
			text_prompt_ids = [ int( option[ 'ID' ] ) for option in text_prompt_options ]
			
			if st.session_state.get( 'text_prompt_id' ) not in text_prompt_ids:
				st.session_state[ 'text_prompt_id' ] = None
			
			# ------------------------------------------------------------------
			# Instruction Text
			# ------------------------------------------------------------------
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='text_system_instructions', )
			
			# ------------------------------------------------------------------
			# Prompt Template Selection
			# ------------------------------------------------------------------
			with in_right:
				st.selectbox( label='Category', options=text_prompt_categories, index=None,
					key='text_prompt_category', placeholder='Select Category',
					help=('Limits prompt templates to categories associated '
					      'with Text generation.'), on_change=reset_prompt_template_selection,
					args=('text_prompt_id',), )
				
				st.selectbox( label='Use Template', options=text_prompt_ids, index=None,
					key='text_prompt_id', placeholder='Select Template',
					disabled=not text_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						text_prompt_options, ), help=('Loads the selected prompt into the Text '
					                                  'system-instruction field.'),
					on_change=load_text_instruction_template, )
			
			# ------------------------------------------------------------------
			# Instruction Actions
			# ------------------------------------------------------------------
			btn_c1, btn_c2 = st.columns( [ 0.80, 0.20, ] )
			
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_text_instructions, icon='🧹', )
			
			with btn_c2:
				st.button( label='XML ↔️ Markdown', width='stretch',
					on_click=convert_text_system_instructions, )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		for msg in st.session_state.get( 'text_messages', [ ], ):
			if not isinstance( msg, dict ):
				continue
			
			role = str( msg.get( 'role', 'assistant', ) )
			content = str( msg.get( 'content', '', ) )
			
			with st.chat_message( role, avatar=get_text_avatar( role ), ):
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
			
			# Capture context before appending the current prompt so the prompt is sent once.
			prior_context = build_text_context( )
			st.session_state[ 'text_messages' ].append( { 'role': 'user', 'content': prompt, } )
			with st.chat_message( 'assistant', avatar=get_text_avatar( 'assistant' ), ):
				with st.spinner( 'Thinking…' ):
					response = None
					response_obj = None
					stream_buffer: List[ str ] = [ ]
					stream_placeholder = st.empty( )
					
					def on_stream_chunk( chunk: str ) -> None:
						"""On stream chunk.
						
						Purpose:
							Renders the accumulated provider streaming response.
						
						Args:
							chunk (str): Provider text delta.
						
						Returns:
							None: This function updates the streaming placeholder.
						"""
						if chunk is None:
							return
						
						chunk_text = str( chunk )
						
						if not chunk_text:
							return
						
						stream_buffer.append( chunk_text )
						stream_placeholder.markdown( ''.join( stream_buffer ) + '▌' )
					
					try:
						validate_text_request( )
						
						response = call_generate_text( prompt=prompt, prior_context=prior_context,
							stream_handler=(on_stream_chunk if st.session_state.get( 'text_stream',
								False, ) else None), )
						response_obj = (getattr( text, 'response', None, ) or response)
						response_text = str( response or '' ).strip( )
						streamed_text = ''.join( stream_buffer ).strip( )
						
						if streamed_text:
							response_text = streamed_text
						
						if not response_text:
							raise ValueError( 'The provider returned no text.' )
						
						if st.session_state.get( 'text_stream', False, ):
							stream_placeholder.markdown( response_text )
						else:
							st.markdown( response_text )
						
						st.session_state[ 'text_messages' ].append(
							{ 'role': 'assistant', 'content': response_text, } )
						st.session_state[ 'text_context' ] = build_text_context( )
						st.session_state[ 'last_answer' ] = response_text
						
						if provider_name in [ 'GPT', 'Grok', ]:
							previous_response_id = str(
								getattr( text, 'previous_response_id', '', ) or getattr( text,
									'previous_id', '', ) or '' )
							
							if previous_response_id:
								st.session_state[
									'text_previous_response_id' ] = previous_response_id
						
						if provider_name == 'GPT':
							conversation_id = str( getattr( text, 'conversation_id', '', ) or '' )
							
							if conversation_id:
								st.session_state[ 'text_conversation_id' ] = conversation_id
						
						sources = extract_text_sources( text, response_obj, )
						st.session_state[ 'last_sources' ] = sources
						
						if sources:
							with st.expander( label='Sources', icon='🔎', expanded=False,
									width='stretch', ):
								for source in sources:
									if not isinstance( source, dict ):
										continue
									
									title = str(
										source.get( 'title', '', ) or source.get( 'url', '', ) )
									url = str( source.get( 'url', '', ) or '' )
									
									if url:
										st.markdown( f'- [{title}]({url})' )
									elif title:
										st.markdown( f'- {title}' )
						
						update_text_usage( response_obj )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Generation Failed: {err.info}' )
		
		# ------------------------------------------------------------------
		# Message Reset
		# ------------------------------------------------------------------
		if st.button( 'Clear Messages', key='text_clear_messages', icon='🧹', width='stretch', ):
			clear_text_messages( )
			st.rerun( )

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == 'Images':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	image = get_images_module( provider_name )
	
	# ------------------------------------------------------------------
	# Images Mode State Safety
	# ------------------------------------------------------------------
	image_defaults: Dict[ str, Any ] = { 'image_mode': 'Generation', 'image_model': '',
		'image_analysis_model': '', 'image_generation_model': '', 'image_editing_model': '',
		'image_number': 1, 'image_size': '', 'image_quality': '', 'image_style': '',
		'image_mime_type': '', 'image_compression': 0.0, 'image_backcolor': '',
		'image_aspect_ratio': '', 'image_analysis_detail': 'auto', 'image_media_resolution': '',
		'image_modality': '', 'image_temperature': 0.0, 'image_top_percent': 0.0, 'image_top_k': 0,
		'image_frequency_penalty': 0.0, 'image_presence_penalty': 0.0, 'image_max_tokens': 0,
		'image_include': [ ], 'image_store': False, 'image_stream': False, 'image_grounded': False,
		'image_image_search': False, 'image_generate_prompt': '', 'image_analysis_prompt': '',
		'image_edit_prompt': '', 'image_system_instructions': '', 'image_prompt_category': None,
		'image_prompt_id': None, 'image_input': [ ], 'generated_images': [ ],
		'analyzed_images': [ ], 'edited_images': [ ], }
	
	for state_key, state_default in image_defaults.items( ):
		if state_key not in st.session_state:
			st.session_state[ state_key ] = state_default
	
	if not isinstance( st.session_state.get( 'image_input' ), list, ):
		st.session_state[ 'image_input' ] = [ ]
	
	if not isinstance( st.session_state.get( 'generated_images' ), list, ):
		st.session_state[ 'generated_images' ] = [ ]
	
	if not isinstance( st.session_state.get( 'analyzed_images' ), list, ):
		st.session_state[ 'analyzed_images' ] = [ ]
	
	if not isinstance( st.session_state.get( 'edited_images' ), list, ):
		st.session_state[ 'edited_images' ] = [ ]
	
	if not isinstance( st.session_state.get( 'image_include' ), list, ):
		st.session_state[ 'image_include' ] = [ ]
	
	image_number_value = st.session_state.get( 'image_number', 1, )
	
	if (not isinstance( image_number_value, int ) or isinstance( image_number_value,
		bool ) or image_number_value < 1 or image_number_value > 10):
		st.session_state[ 'image_number' ] = 1
	
	# ----- Image Helpers -----
	def get_image_options( instance: Any, attr_name: str,
		fallback: Optional[ List[ str ] ] = None ) -> List[ str ]:
		"""Get image options.
		
		Purpose:
			Returns normalized image options exposed by the selected provider wrapper.
		
		Args:
			instance (Any): Provider image-wrapper instance.
			attr_name (str): Wrapper option-property name.
			fallback (Optional[List[str]]): Values used when the property is unavailable.
		
		Returns:
			List[str]: Normalized provider option values.
		"""
		values = getattr( instance, attr_name, None, )
		
		if callable( values ):
			values = values( )
		
		if values is None:
			values = fallback or [ ]
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if not isinstance( values, list ):
			return fallback or [ ]
		
		return [ str( value ) for value in values if str( value ).strip( ) ]
	
	def sanitize_image_selection( key: str, valid_options: List[ str ], default: Any = '' ) -> \
			None:
		"""Sanitize image selection.
		
		Purpose:
			Removes stale single-select values that are unsupported by the selected provider
			and image operation.
		
		Args:
			key (str): Session-state key.
			valid_options (List[str]): Supported option values.
			default (Any): Replacement value.
		
		Returns:
			None: This function updates session state.
		"""
		current_value = st.session_state.get( key, default, )
		
		if current_value in [ None, '', ]:
			return
		
		if valid_options and current_value not in valid_options:
			st.session_state[ key ] = default
	
	def sanitize_image_multiselect( key: str, valid_options: List[ str ] ) -> None:
		"""Sanitize image multiselect.
		
		Purpose:
			Removes stale multi-select values that are unsupported by the selected provider.
		
		Args:
			key (str): Session-state key.
			valid_options (List[str]): Supported option values.
		
		Returns:
			None: This function updates session state.
		"""
		current_values = st.session_state.get( key, [ ], )
		
		if not isinstance( current_values, list ):
			st.session_state[ key ] = [ ]
			return
		
		st.session_state[ key ] = [ value for value in current_values if value in valid_options ]
	
	def get_provider_image_models( selected_mode: Optional[ str ] ) -> List[ str ]:
		"""Get provider image models.
		
		Purpose:
			Returns models configured for the selected provider and image operation.
		
		Args:
			selected_mode (Optional[str]): Generation, Analysis, or Editing.
		
		Returns:
			List[str]: Available model identifiers.
		"""
		mode_name = selected_mode or ''
		
		if provider_name == 'GPT':
			if mode_name == 'Generation':
				models = list( getattr( cfg, 'GPT_GENERATION', [ ], ) )
				
				if models:
					return models
			
			if mode_name == 'Analysis':
				models = list( getattr( cfg, 'GPT_ANALYSIS', [ ], ) )
				
				if models:
					return models
			
			if mode_name == 'Editing':
				models = list( getattr( cfg, 'GPT_EDITING', [ ], ) )
				
				if models:
					return models
		
		if provider_name == 'Gemini':
			if mode_name == 'Generation':
				models = list( getattr( cfg, 'GEMINI_GENERATION', [ ], ) )
				
				if models:
					return models
			
			if mode_name == 'Analysis':
				models = list( getattr( cfg, 'GEMINI_ANALYSIS', [ ], ) )
				
				if models:
					return models
			
			if mode_name == 'Editing':
				models = list( getattr( cfg, 'GEMINI_EDITING', [ ], ) )
				
				if models:
					return models
		
		if provider_name == 'Grok':
			if mode_name == 'Analysis':
				models = get_image_options( image, 'analysis_model_options', )
				
				if models:
					return models
				
				models = list( getattr( cfg, 'GROK_ANALYSIS', [ ], ) )
				
				if models:
					return models
			
			if mode_name == 'Generation':
				models = list( getattr( cfg, 'GROK_GENERATION', [ ], ) )
				
				if models:
					return models
			
			if mode_name == 'Editing':
				models = list( getattr( cfg, 'GROK_EDITING', [ ], ) )
				
				if models:
					return models
		
		models = get_image_options( image, 'model_options', )
		
		if not models:
			model_value = str( getattr( image, 'model', '', ) or '' )
			
			if model_value:
				models = [ model_value, ]
		
		return models
	
	def get_selected_image_model( operation: str ) -> str:
		"""Get selected image model.
		
		Purpose:
			Returns the model selected for the requested image operation.
		
		Args:
			operation (str): Generation, Analysis, or Editing.
		
		Returns:
			str: Selected model identifier.
		"""
		if operation == 'Generation':
			return str(
				st.session_state.get( 'image_generation_model', '', ) or st.session_state.get(
					'image_model', '', ) or '' )
		
		if operation == 'Analysis':
			return str( st.session_state.get( 'image_analysis_model', '',
			) or st.session_state.get(
				'image_model', '', ) or '' )
		
		if operation == 'Editing':
			return str( st.session_state.get( 'image_editing_model', '', ) or st.session_state.get(
				'image_model', '', ) or '' )
		
		return ''
	
	def save_uploaded_image( uploaded_file: Any ) -> Optional[ str ]:
		"""Save uploaded image.
		
		Purpose:
			Persists an uploaded image to a temporary local file for provider requests.
		
		Args:
			uploaded_file (Any): Streamlit uploaded-file object.
		
		Returns:
			Optional[str]: Temporary local path or None.
		"""
		if uploaded_file is None:
			return None
		
		if 'save_temp' in globals( ):
			return save_temp( uploaded_file )
		
		import tempfile
		from pathlib import Path
		
		suffix = Path( uploaded_file.name ).suffix or '.png'
		
		with tempfile.NamedTemporaryFile( delete=False, suffix=suffix, ) as temporary_file:
			temporary_file.write( uploaded_file.getvalue( ) )
			return temporary_file.name
	
	def append_image_message( role: str, content: str ) -> None:
		"""Append image message.
		
		Purpose:
			Appends an Images Mode conversation message.
		
		Args:
			role (str): Message role.
			content (str): Message content.
		
		Returns:
			None: This function updates session state.
		"""
		if not isinstance( st.session_state.get( 'image_input' ), list, ):
			st.session_state[ 'image_input' ] = [ ]
		
		st.session_state[ 'image_input' ].append( { 'role': role, 'content': content, } )
	
	def render_image_messages( ) -> None:
		"""Render image messages.
		
		Purpose:
			Renders Images Mode conversation messages.
		
		Returns:
			None: This function renders Streamlit content.
		"""
		if not isinstance( st.session_state.get( 'image_input' ), list, ):
			st.session_state[ 'image_input' ] = [ ]
		
		for message in st.session_state.get( 'image_input', [ ], ):
			if not isinstance( message, dict, ):
				continue
			
			with st.chat_message( message.get( 'role', 'assistant', ), avatar='', ):
				st.markdown( message.get( 'content', '', ) )
	
	def clear_image_messages( ) -> None:
		"""Clear image messages.
		
		Purpose:
			Clears Images Mode message and result collections.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'image_input' ] = [ ]
		st.session_state[ 'generated_images' ] = [ ]
		st.session_state[ 'analyzed_images' ] = [ ]
		st.session_state[ 'edited_images' ] = [ ]
	
	def clear_image_instructions( ) -> None:
		"""Clear image instructions.
		
		Purpose:
			Clears Images Mode system instructions.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'image_system_instructions' ] = ''
	
	def convert_image_system_instructions( ) -> None:
		"""Convert image system instructions.
		
		Purpose:
			Converts Images Mode instructions between XML and Markdown.
		
		Returns:
			None: This function updates session state.
		"""
		text_value = str( st.session_state.get( 'image_system_instructions', '', ) or '' ).strip( )
		
		if not text_value:
			return
		
		if cfg.XML_BLOCK_PATTERN.search( text_value ):
			converted = convert_xml( text_value )
		else:
			converted = convert_markdown( text_value )
		
		st.session_state[ 'image_system_instructions' ] = converted
	
	def load_image_instruction_template( ) -> None:
		"""Load image instruction template.
		
		Purpose:
			Loads the selected Images Mode prompt template into system instructions.
		
		Returns:
			None: This function updates session state.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			load_prompt_template( prompt_id_key='image_prompt_id',
				instructions_key='image_system_instructions', )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Images Mode'
			ex.method = ('load_image_instruction_template( ) -> None')
			Logger( ).write( ex )
			raise ex
	
	def reset_image_model_settings( ) -> None:
		"""Reset image model settings.
		
		Purpose:
			Resets Images Mode operation and model selections.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'image_mode' ] = 'Generation'
		st.session_state[ 'image_model' ] = ''
		st.session_state[ 'image_generation_model' ] = ''
		st.session_state[ 'image_analysis_model' ] = ''
		st.session_state[ 'image_editing_model' ] = ''
		st.session_state[ 'image_number' ] = 1
	
	def reset_image_inference_settings( ) -> None:
		"""Reset image inference settings.
		
		Purpose:
			Resets provider-supported image inference controls.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'image_temperature' ] = 0.0
		st.session_state[ 'image_top_percent' ] = 0.0
		st.session_state[ 'image_top_k' ] = 0
		st.session_state[ 'image_frequency_penalty' ] = 0.0
		st.session_state[ 'image_presence_penalty' ] = 0.0
		st.session_state[ 'image_max_tokens' ] = 0
	
	def reset_image_response_settings( ) -> None:
		"""Reset image response settings.
		
		Purpose:
			Resets provider-supported image analysis response controls.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'image_include' ] = [ ]
		st.session_state[ 'image_store' ] = False
		st.session_state[ 'image_stream' ] = False
		st.session_state[ 'image_analysis_detail' ] = 'auto'
		st.session_state[ 'image_media_resolution' ] = ''
	
	def reset_image_visual_settings( ) -> None:
		"""Reset image visual settings.
		
		Purpose:
			Resets provider-supported generation and editing output controls.
		
		Returns:
			None: This function updates session state.
		"""
		st.session_state[ 'image_size' ] = ''
		st.session_state[ 'image_quality' ] = ''
		st.session_state[ 'image_style' ] = ''
		st.session_state[ 'image_mime_type' ] = ''
		st.session_state[ 'image_compression' ] = 0.0
		st.session_state[ 'image_backcolor' ] = ''
		st.session_state[ 'image_aspect_ratio' ] = ''
		st.session_state[ 'image_modality' ] = ''
		st.session_state[ 'image_grounded' ] = False
		st.session_state[ 'image_image_search' ] = False
	
	def render_image_output( result: Any, caption: str ) -> bool:
		"""Render image output.
		
		Purpose:
			Renders provider image URLs, byte content, lists, or compatible image objects.
		
		Args:
			result (Any): Provider image output.
			caption (str): Image caption.
		
		Returns:
			bool: True when at least one image is rendered.
		"""
		if result is None:
			return False
		
		if isinstance( result, list ):
			rendered = False
			
			for index, item in enumerate( result, start=1, ):
				if render_image_output( item, f'{caption} {index}', ):
					rendered = True
			
			return rendered
		
		if isinstance( result, (str, bytes, bytearray,), ):
			st.image( result, caption=caption, use_container_width=True, )
			return True
		
		result_url = getattr( result, 'url', '', )
		
		if result_url:
			st.image( result_url, caption=caption, use_container_width=True, )
			return True
		
		result_bytes = getattr( result, 'image_bytes', None, )
		
		if result_bytes:
			st.image( result_bytes, caption=caption, use_container_width=True, )
			return True
		
		try:
			st.image( result, caption=caption, use_container_width=True, )
			return True
		except Exception:
			return False
	
	def update_image_usage( response: Any ) -> None:
		"""Update image usage.
		
		Purpose:
			Updates application usage counters when a compatible usage helper is available.
		
		Args:
			response (Any): Provider response.
		
		Returns:
			None: This function updates application state.
		"""
		if 'update_token_counters' in globals( ):
			update_token_counters( response )
	
	def validate_image_request( operation: str, prompt: str, path: str = '' ) -> str:
		"""Validate image request.
		
		Purpose:
			Validates required Images Mode controls for the selected provider operation.
		
		Args:
			operation (str): Generation, Analysis, or Editing.
			prompt (str): Operation prompt.
			path (str): Optional required local image path.
		
		Returns:
			str: Selected provider model.
		
		Raises:
			ValueError: Raised when a required input is missing.
		"""
		if not isinstance( prompt, str, ) or not prompt.strip( ):
			raise ValueError( f'Enter an {operation.lower( )} prompt.' )
		
		model = get_selected_image_model( operation )
		
		if not model:
			raise ValueError( f'Select a model for Image {operation}.' )
		
		if operation in [ 'Analysis', 'Editing', ] and not path:
			raise ValueError( f'Upload an image before {operation.lower( )}.' )
		
		if (provider_name == 'Gemini' and operation in [ 'Generation',
			'Editing', ] and st.session_state.get( 'image_image_search', False, ) and hasattr(
			image, 'supports_image_search', ) and not image.supports_image_search( model )):
			raise ValueError( 'The selected Gemini model does not support Image Search.' )
		
		if (provider_name == 'Gemini' and operation in [ 'Generation',
			'Editing', ] and st.session_state.get( 'image_grounded', False, ) and hasattr( image,
			'supports_grounding', ) and not image.supports_grounding( model )):
			raise ValueError( 'The selected Gemini model does not support Google grounding.' )
		
		return model
	
	def run_image_generation( prompt: str ) -> Any:
		"""Run image generation.
		
		Purpose:
			Calls the exact generation interface implemented by the selected provider wrapper.
		
		Args:
			prompt (str): Image-generation prompt.
		
		Returns:
			Any: Provider image output.
		"""
		model = validate_image_request( 'Generation', prompt, )
		
		if provider_name == 'GPT':
			return image.generate( prompt=prompt,
				number=int( st.session_state.get( 'image_number', 1, ) ), model=model,
				size=str( st.session_state.get( 'image_size', '', ) or '1024x1024' ),
				quality=str( st.session_state.get( 'image_quality', '', ) or 'auto' ),
				fmt=str( st.session_state.get( 'image_mime_type', '', ) or 'png' ).replace(
					'image/', '', ),
				compression=float( st.session_state.get( 'image_compression', 0.0, ) ),
				background=str( st.session_state.get( 'image_backcolor', '', ) or '' ), )
		
		if provider_name == 'Gemini':
			return image.generate( prompt=prompt, model=model,
				number=int( st.session_state.get( 'image_number', 1, ) ),
				aspect_ratio=str( st.session_state.get( 'image_aspect_ratio', '', ) or '' ),
				image_size=str( st.session_state.get( 'image_size', '', ) or '' ),
				output_mime_type=str( st.session_state.get( 'image_mime_type', '', ) or '' ),
				response_modalities=str( st.session_state.get( 'image_modality', '', ) or '' ),
				temperature=float( st.session_state.get( 'image_temperature', 0.0, ) ),
				top_p=float( st.session_state.get( 'image_top_percent', 0.0, ) ),
				frequency=float( st.session_state.get( 'image_frequency_penalty', 0.0, ) ),
				presence=float( st.session_state.get( 'image_presence_penalty', 0.0, ) ),
				max_tokens=int( st.session_state.get( 'image_max_tokens', 0, ) ),
				media_resolution=str( st.session_state.get( 'image_media_resolution', '', ) or '' ),
				instruct=str( st.session_state.get( 'image_system_instructions', '', ) or '' ),
				grounded=bool( st.session_state.get( 'image_grounded', False, ) ),
				image_search=bool( st.session_state.get( 'image_image_search', False, ) ), )
		
		if provider_name == 'Grok':
			return image.generate( prompt=prompt, model=model,
				number=int( st.session_state.get( 'image_number', 1, ) ), aspect_ratio=str(
					st.session_state.get( 'image_aspect_ratio', 'auto', ) or 'auto' ), )
		
		raise ValueError( f'Unsupported Images provider: {provider_name}' )
	
	def run_image_analysis( prompt: str, path: str ) -> Any:
		"""Run image analysis.
		
		Purpose:
			Calls the exact image-analysis interface implemented by the selected provider
			wrapper.
		
		Args:
			prompt (str): Image-analysis prompt.
			path (str): Local image path.
		
		Returns:
			Any: Provider analysis text.
		"""
		model = validate_image_request( 'Analysis', prompt, path, )
		
		if provider_name == 'GPT':
			return image.analyze( text=prompt, path=path,
				instruct=str( st.session_state.get( 'image_system_instructions', '', ) or '' ),
				model=model, max_tokens=int( st.session_state.get( 'image_max_tokens', 0, ) ),
				temperature=float( st.session_state.get( 'image_temperature', 0.0, ) ),
				include=list( st.session_state.get( 'image_include', [ ], ) ),
				store=bool( st.session_state.get( 'image_store', False, ) ),
				stream=bool( st.session_state.get( 'image_stream', False, ) ),
				detail=str( st.session_state.get( 'image_analysis_detail', 'auto', ) or 'auto' ), )
		
		if provider_name == 'Gemini':
			return image.analyze( prompt=prompt, path=path, model=model,
				temperature=float( st.session_state.get( 'image_temperature', 0.0, ) ),
				top_p=float( st.session_state.get( 'image_top_percent', 0.0, ) ),
				frequency=float( st.session_state.get( 'image_frequency_penalty', 0.0, ) ),
				presence=float( st.session_state.get( 'image_presence_penalty', 0.0, ) ),
				max_tokens=int( st.session_state.get( 'image_max_tokens', 0, ) ),
				media_resolution=str( st.session_state.get( 'image_media_resolution', '',
				) or '' ),
				instruct=str( st.session_state.get( 'image_system_instructions', '', ) or '' ), )
		
		if provider_name == 'Grok':
			return image.analyze( prompt=prompt, path=path, model=model,
				detail=str( st.session_state.get( 'image_analysis_detail', 'auto', ) or 'auto' ), )
		
		raise ValueError( f'Unsupported Images provider: {provider_name}' )
	
	def run_image_editing( prompt: str, path: str, mask_path: str = '' ) -> Any:
		"""Run image editing.
		
		Purpose:
			Calls the exact image-editing interface implemented by the selected provider
			wrapper.
		
		Args:
			prompt (str): Image-editing prompt.
			path (str): Local source-image path.
			mask_path (str): Optional GPT mask path.
		
		Returns:
			Any: Provider edited-image output.
		"""
		model = validate_image_request( 'Editing', prompt, path, )
		
		if provider_name == 'GPT':
			edit_method = getattr( image, 'edit', None, )
			
			if not callable( edit_method ):
				raise AttributeError( 'GPT Images does not expose edit().' )
			
			edit_parameters = inspect.signature( edit_method ).parameters
			edit_kwargs: Dict[ str, Any ] = { 'prompt': prompt, 'path': path, 'model': model,
				'size': str( st.session_state.get( 'image_size', '', ) or '1024x1024' ),
				'quality': str( st.session_state.get( 'image_quality', '', ) or 'auto' ),
				'fmt': str( st.session_state.get( 'image_mime_type', '', ) or 'png' ).replace(
					'image/', '', ),
				'compression': float( st.session_state.get( 'image_compression', 0.0, ) ),
				'background': str( st.session_state.get( 'image_backcolor', '', ) or '' ),
				'number': int( st.session_state.get( 'image_number', 1, ) ), }
			
			if mask_path:
				if 'mask_path' in edit_parameters:
					edit_kwargs[ 'mask_path' ] = mask_path
				elif 'mask' in edit_parameters:
					edit_kwargs[ 'mask' ] = mask_path
			
			return edit_method( **edit_kwargs )
		
		if provider_name == 'Gemini':
			return image.edit( prompt=prompt, path=path, model=model,
				number=int( st.session_state.get( 'image_number', 1, ) ),
				aspect_ratio=str( st.session_state.get( 'image_aspect_ratio', '', ) or '' ),
				image_size=str( st.session_state.get( 'image_size', '', ) or '' ),
				output_mime_type=str( st.session_state.get( 'image_mime_type', '', ) or '' ),
				response_modalities=str( st.session_state.get( 'image_modality', '', ) or '' ),
				temperature=float( st.session_state.get( 'image_temperature', 0.0, ) ),
				top_p=float( st.session_state.get( 'image_top_percent', 0.0, ) ),
				frequency=float( st.session_state.get( 'image_frequency_penalty', 0.0, ) ),
				presence=float( st.session_state.get( 'image_presence_penalty', 0.0, ) ),
				max_tokens=int( st.session_state.get( 'image_max_tokens', 0, ) ),
				media_resolution=str( st.session_state.get( 'image_media_resolution', '',
				) or '' ),
				instruct=str( st.session_state.get( 'image_system_instructions', '', ) or '' ),
				grounded=bool( st.session_state.get( 'image_grounded', False, ) ),
				image_search=bool( st.session_state.get( 'image_image_search', False, ) ), )
		
		if provider_name == 'Grok':
			return image.edit( prompt=prompt, model=model, path=path, image_url='',
				aspect_ratio=str( st.session_state.get( 'image_aspect_ratio', 'auto',
				) or 'auto' ),
				number=int( st.session_state.get( 'image_number', 1, ) ), )
		
		raise ValueError( f'Unsupported Images provider: {provider_name}' )
	
	# ------------------------------------------------------------------
	# Main Image UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05, ] )
	
	with center:
		st.subheader( '🖼️ Images',
			help=getattr( cfg, 'IMAGE_GENERATION', 'Generate, analyze, and edit images.', ), )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Expander — Image Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch', ):
			# ------------------------------------------------------------------
			# Expander — Model Settings
			# ------------------------------------------------------------------
			with st.expander( label='Model Settings', icon='🧊', expanded=False,
					width='stretch', ):
				mode_options = [ 'Generation', 'Analysis', 'Editing', ]
				sanitize_image_selection( 'image_mode', mode_options, 'Generation', )
				
				# ----- Image Mode -----
				st.selectbox( label='Image Mode', options=mode_options, key='image_mode',
					help='Select the image operation used to configure the controls below.', )
				
				selected_image_mode = st.session_state.get( 'image_mode', 'Generation', )
				model_options = get_provider_image_models( selected_image_mode )
				
				if selected_image_mode == 'Generation':
					sanitize_image_selection( 'image_generation_model', model_options, '', )
					model_key = 'image_generation_model'
				
				elif selected_image_mode == 'Analysis':
					sanitize_image_selection( 'image_analysis_model', model_options, '', )
					model_key = 'image_analysis_model'
				
				else:
					sanitize_image_selection( 'image_editing_model', model_options, '', )
					model_key = 'image_editing_model'
				
				model_c1, model_c2 = st.columns( [ 0.75, 0.25, ], border=True, gap='xxsmall', )
				
				# ----- Model -----
				with model_c1:
					st.selectbox( label='Model', options=model_options, key=model_key, index=None,
						placeholder='Select Model',
						help='Required. Select the provider model for the chosen image '
						     'operation.', )
				
				# ----- Number of Images -----
				with model_c2:
					st.slider( label='Images', min_value=1, max_value=10, step=1,
						key='image_number', disabled=selected_image_mode == 'Analysis',
						help='Number of generated or edited images requested.', )
				
				st.session_state[ 'image_model' ] = str(
					st.session_state.get( model_key, '', ) or '' )
				
				st.button( label='Reset', key='image_model_reset', width='stretch',
					on_click=reset_image_model_settings, icon='🔄', )
			
			# ------------------------------------------------------------------
			# Expander — Inference Settings
			# ------------------------------------------------------------------
			with st.expander( label='Inference Settings', icon='🎚️', expanded=False,
					width='stretch', ):
				selected_image_mode = st.session_state.get( 'image_mode', 'Generation', )
				
				if provider_name == 'Gemini':
					inf_c1, inf_c2, inf_c3, inf_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25, ],
						border=True, gap='xxsmall', )
					
					# ----- Temperature -----
					with inf_c1:
						st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.01,
							key='image_temperature', help=cfg.TEMPERATURE, )
					
					# ----- Top-P -----
					with inf_c2:
						st.slider( label='Top-P', min_value=0.0, max_value=1.0, step=0.01,
							key='image_top_percent', help=cfg.TOP_P, )
					
					# ----- Frequency Penalty -----
					with inf_c3:
						st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
							step=0.01, key='image_frequency_penalty', help=cfg.FREQUENCY_PENALTY, )
					
					# ----- Presence Penalty -----
					with inf_c4:
						st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
							step=0.01, key='image_presence_penalty', help=cfg.PRESENCE_PENALTY, )
					
					# ----- Max Tokens -----
					st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
						key='image_max_tokens', help=cfg.MAX_OUTPUT_TOKENS, )
				
				elif (provider_name == 'GPT' and selected_image_mode == 'Analysis'):
					inf_c1, inf_c2 = st.columns( [ 0.50, 0.50, ], border=True, gap='xxsmall', )
					
					# ----- Temperature -----
					with inf_c1:
						st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.01,
							key='image_temperature', help=cfg.TEMPERATURE, )
					
					# ----- Max Tokens -----
					with inf_c2:
						st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
							key='image_max_tokens', help=cfg.MAX_OUTPUT_TOKENS, )
				
				else:
					st.info( 'The selected provider operation does not expose sampling controls.' )
				
				st.button( label='Reset', key='image_inference_reset', width='stretch',
					on_click=reset_image_inference_settings, icon='🔄', )
			
			# ------------------------------------------------------------------
			# Expander — Response Settings
			# ------------------------------------------------------------------
			with st.expander( label='Output / Response Settings', icon='↔️', expanded=False,
					width='stretch', ):
				selected_image_mode = st.session_state.get( 'image_mode', 'Generation', )
				
				if selected_image_mode == 'Analysis':
					if provider_name == 'GPT':
						detail_options = get_image_options( image, 'detail_options',
							[ 'auto', 'low', 'high', ], )
						include_options = get_image_options( image, 'include_options', )
						
						sanitize_image_selection( 'image_analysis_detail', detail_options,
							'auto', )
						sanitize_image_multiselect( 'image_include', include_options, )
						
						resp_c1, resp_c2, resp_c3, resp_c4 = st.columns(
							[ 0.25, 0.25, 0.25, 0.25, ], border=True, gap='xxsmall', )
						
						# ----- Detail -----
						with resp_c1:
							st.selectbox( label='Detail', options=detail_options,
								key='image_analysis_detail',
								help='GPT image-analysis detail level.', )
						
						# ----- Include -----
						with resp_c2:
							st.multiselect( label='Include', options=include_options,
								key='image_include', placeholder='Options',
								disabled=not include_options, help=cfg.INCLUDE, )
						
						# ----- Store -----
						with resp_c3:
							st.toggle( label='Store', key='image_store', help=cfg.STORE, )
						
						# ----- Stream -----
						with resp_c4:
							st.toggle( label='Stream', key='image_stream', help=cfg.STREAM, )
					
					elif provider_name == 'Gemini':
						media_options = get_image_options( image, 'media_options', )
						sanitize_image_selection( 'image_media_resolution', media_options, '', )
						
						# ----- Media Resolution -----
						st.selectbox( label='Media Resolution', options=media_options,
							key='image_media_resolution', index=None, placeholder='Options',
							disabled=not media_options,
							help='Gemini media resolution used for image analysis.', )
						
						st.session_state[ 'image_include' ] = [ ]
						st.session_state[ 'image_store' ] = False
						st.session_state[ 'image_stream' ] = False
						st.session_state[ 'image_analysis_detail' ] = 'auto'
					
					elif provider_name == 'Grok':
						detail_options = get_image_options( image, 'detail_options',
							[ 'auto', 'low', 'high', ], )
						sanitize_image_selection( 'image_analysis_detail', detail_options,
							'auto', )
						
						# ----- Detail -----
						st.selectbox( label='Detail', options=detail_options,
							key='image_analysis_detail',
							help='Grok multimodal image-analysis detail level.', )
						
						st.session_state[ 'image_include' ] = [ ]
						st.session_state[ 'image_store' ] = False
						st.session_state[ 'image_stream' ] = False
						st.session_state[ 'image_media_resolution' ] = ''
				
				else:
					st.info( 'Response controls in this section apply to image analysis only.' )
				
				st.button( label='Reset', key='image_response_reset', width='stretch',
					on_click=reset_image_response_settings, icon='🔄', )
			
			# ------------------------------------------------------------------
			# Expander — Visual Settings
			# ------------------------------------------------------------------
			with st.expander( label='Visual Settings', icon='🎨', expanded=False,
					width='stretch', ):
				selected_image_mode = st.session_state.get( 'image_mode', 'Generation', )
				
				if selected_image_mode in [ 'Generation', 'Editing', ]:
					if provider_name == 'GPT':
						size_options = get_image_options( image, 'size_options', )
						quality_options = get_image_options( image, 'quality_options', )
						format_options = get_image_options( image, 'format_options',
							[ 'png', 'jpeg', 'webp', ], )
						background_options = get_image_options( image, 'background_options',
							[ 'auto', 'transparent', 'opaque', ], )
						
						sanitize_image_selection( 'image_size', size_options, '', )
						sanitize_image_selection( 'image_quality', quality_options, '', )
						sanitize_image_selection( 'image_mime_type', format_options, '', )
						sanitize_image_selection( 'image_backcolor', background_options, '', )
						
						visual_c1, visual_c2, visual_c3, visual_c4 = st.columns(
							[ 0.25, 0.25, 0.25, 0.25, ], border=True, gap='xxsmall', )
						
						# ----- Size -----
						with visual_c1:
							st.selectbox( label='Size', options=size_options, key='image_size',
								index=None, placeholder='Options', disabled=not size_options,
								help='GPT output pixel dimensions.', )
						
						# ----- Quality -----
						with visual_c2:
							st.selectbox( label='Quality', options=quality_options,
								key='image_quality', index=None, placeholder='Options',
								disabled=not quality_options, help='GPT output image quality.', )
						
						# ----- Format -----
						with visual_c3:
							st.selectbox( label='Format', options=format_options,
								key='image_mime_type', index=None, placeholder='Options',
								disabled=not format_options, help='GPT output image encoding.', )
						
						# ----- Background -----
						with visual_c4:
							st.selectbox( label='Background', options=background_options,
								key='image_backcolor', index=None, placeholder='Options',
								disabled=not background_options,
								help='GPT output background behavior.', )
						
						# ----- Compression -----
						st.slider( label='Compression', min_value=0.0, max_value=100.0, step=1.0,
							key='image_compression',
							help='GPT JPEG or WebP compression level when supported.', )
						
						st.session_state[ 'image_aspect_ratio' ] = ''
						st.session_state[ 'image_modality' ] = ''
						st.session_state[ 'image_grounded' ] = False
						st.session_state[ 'image_image_search' ] = False
					
					elif provider_name == 'Gemini':
						aspect_options = get_image_options( image, 'aspect_options', )
						size_options = get_image_options( image, 'size_options', )
						mime_options = get_image_options( image, 'mime_options',
							get_image_options( image, 'format_options', ), )
						modality_options = get_image_options( image, 'modality_options',
							[ 'IMAGE', 'TEXT_AND_IMAGE', ], )
						media_options = get_image_options( image, 'media_options', )
						
						sanitize_image_selection( 'image_aspect_ratio', aspect_options, '', )
						sanitize_image_selection( 'image_size', size_options, '', )
						sanitize_image_selection( 'image_mime_type', mime_options, '', )
						sanitize_image_selection( 'image_modality', modality_options, '', )
						sanitize_image_selection( 'image_media_resolution', media_options, '', )
						
						visual_c1, visual_c2, visual_c3, visual_c4 = st.columns(
							[ 0.25, 0.25, 0.25, 0.25, ], border=True, gap='xxsmall', )
						
						# ----- Aspect Ratio -----
						with visual_c1:
							st.selectbox( label='Aspect Ratio', options=aspect_options,
								key='image_aspect_ratio', index=None, placeholder='Options',
								disabled=not aspect_options,
								help='Gemini output image aspect ratio.', )
						
						# ----- Image Size -----
						with visual_c2:
							st.selectbox( label='Image Size', options=size_options,
								key='image_size', index=None, placeholder='Options',
								disabled=not size_options, help='Gemini output image size.', )
						
						# ----- Output MIME Type -----
						with visual_c3:
							st.selectbox( label='Output MIME Type', options=mime_options,
								key='image_mime_type', index=None, placeholder='Options',
								disabled=not mime_options, help='Gemini output image MIME type.', )
						
						# ----- Response Mode -----
						with visual_c4:
							st.selectbox( label='Response Mode', options=modality_options,
								key='image_modality', index=None, placeholder='Options',
								disabled=not modality_options,
								help='Gemini response modality for image generation or editing.', )
						
						# ----- Media Resolution -----
						st.selectbox( label='Media Resolution', options=media_options,
							key='image_media_resolution', index=None, placeholder='Options',
							disabled=not media_options,
							help='Gemini media-resolution configuration.', )
						
						ground_c1, ground_c2 = st.columns( [ 0.50, 0.50, ], border=True,
							gap='xxsmall', )
						
						# ----- Google Grounding -----
						with ground_c1:
							st.toggle( label='Google Grounding', key='image_grounded',
								help='Enable Gemini Google Search grounding when supported.', )
						
						# ----- Image Search -----
						with ground_c2:
							st.toggle( label='Image Search', key='image_image_search',
								help='Enable Gemini Image Search when supported.', )
						
						st.session_state[ 'image_quality' ] = ''
						st.session_state[ 'image_style' ] = ''
						st.session_state[ 'image_compression' ] = 0.0
						st.session_state[ 'image_backcolor' ] = ''
					
					elif provider_name == 'Grok':
						aspect_options = get_image_options( image, 'aspect_options', )
						sanitize_image_selection( 'image_aspect_ratio', aspect_options, 'auto', )
						
						# ----- Aspect Ratio -----
						st.selectbox( label='Aspect Ratio', options=aspect_options,
							key='image_aspect_ratio', index=None, placeholder='Options',
							disabled=not aspect_options, help='Grok output image aspect ratio.', )
						
						st.session_state[ 'image_size' ] = ''
						st.session_state[ 'image_quality' ] = ''
						st.session_state[ 'image_style' ] = ''
						st.session_state[ 'image_mime_type' ] = ''
						st.session_state[ 'image_compression' ] = 0.0
						st.session_state[ 'image_backcolor' ] = ''
						st.session_state[ 'image_modality' ] = ''
						st.session_state[ 'image_grounded' ] = False
						st.session_state[ 'image_image_search' ] = False
				
				else:
					st.info( 'Visual output controls apply to image generation and editing.' )
				
				# ----- Reset Button -----
				st.button( label='Reset', key='image_visual_reset', width='stretch',
					on_click=reset_image_visual_settings, icon='🔄', )
		
		# ------------------------------------------------------------------
		# Expander — Image System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch', ):
			in_left, in_right = st.columns( [ 0.80, 0.20, ] )
			
			# ------ Image Prompt Categories ------
			image_prompt_categories = fetch_prompt_categories( 'Images' )
			current_image_category = st.session_state.get( 'image_prompt_category' )
			
			if current_image_category not in image_prompt_categories:
				st.session_state[ 'image_prompt_category' ] = None
			
			selected_image_category = st.session_state.get( 'image_prompt_category' )
			image_prompt_options = (
				fetch_prompt_options( selected_image_category ) if selected_image_category else
				[ ])
			image_prompt_ids = [ int( option[ 'ID' ] ) for option in image_prompt_options ]
			
			if (st.session_state.get( 'image_prompt_id' ) not in image_prompt_ids):
				st.session_state[ 'image_prompt_id' ] = None
			
			# ----- Instruction Text ------
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='image_system_instructions', )
			
			# ----- Template Selection ------
			with in_right:
				st.selectbox( label='Category', options=image_prompt_categories, index=None,
					key='image_prompt_category', placeholder='Select Category',
					help=('Limits prompt templates to categories associated '
					      'with image workflows.'), on_change=reset_prompt_template_selection,
					args=('image_prompt_id',), )
				
				st.selectbox( label='Use Template', options=image_prompt_ids, index=None,
					key='image_prompt_id', placeholder='Select Template',
					disabled=not image_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						image_prompt_options, ), help=('Loads the selected prompt into the Images '
					                                   'system-instruction field.'),
					on_change=load_image_instruction_template, )
			
			btn_c1, btn_c2 = st.columns( [ 0.80, 0.20, ] )
			
			# ------ Clear Button -----
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_image_instructions, icon='🧹', )
			
			# ----- Convert Button ------
			with btn_c2:
				st.button( label='XML ↔️ Markdown', width='stretch',
					on_click=convert_image_system_instructions, )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
		
		# ----- Tab Section ------
		tab_gen, tab_analyze, tab_edit = st.tabs( [ 'Generate', 'Analyze', 'Edit', ] )
		
		with tab_gen:
			render_image_messages( )
			generation_prompt = st.text_area( label='Image Generation Prompt',
				key='image_generate_prompt', height=120, width='stretch',
				placeholder='Describe the image to generate.', )
			
			# ----- Generate Image -----
			gen_c1, gen_c2 = st.columns( [ 0.50, 0.50, ] )
			
			with gen_c1:
				if st.button( 'Generate Image', key='generate_image', width='stretch',
						icon='🎨', ):
					with st.spinner( 'Generating…' ):
						try:
							result = run_image_generation( str( generation_prompt or '' ).strip(
							
							) )
							append_image_message( 'user', str( generation_prompt or '' ).strip(
							
							), )
							
							if result is None:
								st.warning( 'No image output was returned.' )
							else:
								st.session_state[ 'generated_images' ].append( result )
								rendered = render_image_output( result, 'Generated image', )
								
								if rendered:
									append_image_message( 'assistant',
										'Generated image returned successfully.', )
								else:
									append_image_message( 'assistant', str( result ), )
							
							update_image_usage( getattr( image, 'response', None, ) )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Image generation failed: {err.info}' )
			
			# ---- Clear Button -----
			with gen_c2:
				if st.button( 'Clear Messages', key='clear_image_generation', width='stretch',
						on_click=clear_image_messages, icon='🧹', ):
					st.rerun( )
		
		# ----- Analyze Image -----
		with tab_analyze:
			uploaded_img = st.file_uploader( 'Upload an image for analysis',
				type=[ 'png', 'jpg', 'jpeg', 'webp', ], accept_multiple_files=False,
				key='images_analyze_uploader', )
			
			analysis_path = None
			
			if uploaded_img:
				analysis_path = save_uploaded_image( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview', width=250, )
			
			render_image_messages( )
			analysis_prompt = st.text_area( label='Image Analysis Prompt',
				key='image_analysis_prompt', height=120, width='stretch',
				placeholder='Ask a question about the uploaded image.', )
			
			ana_c1, ana_c2 = st.columns( [ 0.50, 0.50, ] )
			
			# ----- Analyze Button -----
			with ana_c1:
				if st.button( 'Analyze Image', key='analyze_image', width='stretch', icon='🔬', ):
					with st.spinner( 'Analyzing image…' ):
						try:
							result = run_image_analysis( str( analysis_prompt or '' ).strip( ),
								str( analysis_path or '' ), )
							append_image_message( 'user', str( analysis_prompt or '' ).strip( ), )
							
							if result is None:
								st.warning( 'No analysis output returned by the model.' )
							else:
								st.session_state[ 'analyzed_images' ].append( result )
								st.markdown( '**Analysis result:**' )
								st.write( result )
								append_image_message( 'assistant', str( result ), )
							
							update_image_usage( getattr( image, 'response', None, ) )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Analysis Failed: {err.info}' )
			
			# ----- Clear Button ------
			with ana_c2:
				if st.button( 'Clear Messages', key='clear_image_analysis', width='stretch',
						on_click=clear_image_messages, icon='🧹', ):
					st.rerun( )
		
		# ----- Edit Image -----
		with tab_edit:
			uploaded_img = st.file_uploader( 'Upload Image for Edit',
				type=[ 'png', 'jpg', 'jpeg', 'webp', ], accept_multiple_files=False,
				key='images_edit_uploader', )
			
			uploaded_mask = None
			
			if provider_name == 'GPT':
				uploaded_mask = st.file_uploader( 'Upload Optional Mask',
					type=[ 'png', 'jpg', 'jpeg', 'webp', ], accept_multiple_files=False,
					key='images_edit_mask_uploader', help='Optional GPT image-edit mask.', )
			
			edit_path = None
			mask_path = None
			
			if uploaded_img:
				edit_path = save_uploaded_image( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview', width=250, )
			
			if uploaded_mask:
				mask_path = save_uploaded_image( uploaded_mask )
				st.image( uploaded_mask, caption='Uploaded mask preview', width=250, )
			
			render_image_messages( )
			edit_prompt = st.text_area( label='Image Editing Prompt', key='image_edit_prompt',
				height=120, width='stretch',
				placeholder='Describe how the image should be edited.', )
			
			edit_c1, edit_c2 = st.columns( [ 0.50, 0.50, ] )
			
			# ----- Edit Button -----
			with edit_c1:
				if st.button( 'Edit Image', key='edit_image', width='stretch', icon='✏️', ):
					with st.spinner( 'Editing image…' ):
						try:
							result = run_image_editing( str( edit_prompt or '' ).strip( ),
								str( edit_path or '' ), str( mask_path or '' ), )
							append_image_message( 'user', str( edit_prompt or '' ).strip( ), )
							
							if result is None:
								st.warning( 'No edited image output was returned.' )
							else:
								st.session_state[ 'edited_images' ].append( result )
								rendered = render_image_output( result, 'Edited image', )
								
								if rendered:
									append_image_message( 'assistant',
										'Edited image returned successfully.', )
								else:
									append_image_message( 'assistant', str( result ), )
							
							update_image_usage( getattr( image, 'response', None, ) )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Image edit failed: {err.info}' )
			
			# ----- Clear Button -----
			with edit_c2:
				if st.button( 'Clear Messages', key='clear_image_edit', width='stretch',
						on_click=clear_image_messages, icon='🧹', ):
					st.rerun( )

# ======================================================================================
# AUDIO MODE
# ======================================================================================
elif mode == 'Audio':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	transcriber = get_transcription_module( provider_name )
	translator = get_translation_module( provider_name )
	tts = get_tts_module( provider_name )
	
	if not isinstance( st.session_state.get( 'audio_messages' ), list ):
		st.session_state[ 'audio_messages' ] = [ ]
	
	if not isinstance( st.session_state.get( 'audio_include' ), list ):
		st.session_state[ 'audio_include' ] = [ ]
	
	if not isinstance( st.session_state.get( 'audio_domains' ), list ):
		st.session_state[ 'audio_domains' ] = [ ]
	
	if not isinstance( st.session_state.get( 'audio_tools' ), list ):
		st.session_state[ 'audio_tools' ] = [ ]
	
	if not isinstance( st.session_state.get( 'audio_last_result' ), dict ):
		st.session_state[ 'audio_last_result' ] = { }
	
	if not isinstance( st.session_state.get( 'audio_last_usage' ), dict ):
		st.session_state[ 'audio_last_usage' ] = { }
	
	if 'audio_output' not in st.session_state:
		st.session_state[ 'audio_output' ] = ''
	
	if 'audio_output_bytes' not in st.session_state:
		st.session_state[ 'audio_output_bytes' ] = None
	
	if 'audio_output_path' not in st.session_state:
		st.session_state[ 'audio_output_path' ] = ''
	
	if 'audio_upload_path' not in st.session_state:
		st.session_state[ 'audio_upload_path' ] = ''
	
	if 'audio_recorded_path' not in st.session_state:
		st.session_state[ 'audio_recorded_path' ] = ''
	
	if 'audio_domains_input' not in st.session_state:
		st.session_state[ 'audio_domains_input' ] = ''
	
	if 'audio_tts_input' not in st.session_state:
		st.session_state[ 'audio_tts_input' ] = ''
	
	if 'audio_speed' not in st.session_state:
		st.session_state[ 'audio_speed' ] = 1.0
	
	if 'audio_sample_rate' not in st.session_state:
		st.session_state[ 'audio_sample_rate' ] = 0
	
	if 'audio_bit_rate' not in st.session_state:
		st.session_state[ 'audio_bit_rate' ] = 0
	
	# ------ Audio Mode Utilities ------
	def get_audio_help( name: str, fallback: str = '' ) -> str:
		"""Get audio help.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and
		    downstream logic can consume it consistently.
		
		Args:
		    name (str): Name value used by the operation.
		    fallback (str): Fallback value used by the operation.
		
		Returns:
		    str: Return value produced by the operation."""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_audio_options( instance: Any, attr_name: str,
		fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""Get audio options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and
		    downstream logic can consume it consistently.
		
		Args:
		    instance (Any): Instance value used by the operation.
		    attr_name (str): Attr name value used by the operation.
		    fallback (Optional[List[Any]]): Fallback value used by the operation.
		
		Returns:
		    List[Any]: Return value produced by the operation."""
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
	
	def audio_has_method( instance: Any, method_names: List[ str ] ) -> bool:
		"""Audio has method.
		
		Purpose:
		    Performs the audio_has_method workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    instance (Any): Instance value used by the operation.
		    method_names (List[str]): Method names value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation."""
		for method_name in method_names:
			method = getattr( instance, method_name, None )
			if callable( method ):
				return True
		
		return False
	
	def get_audio_task_options( ) -> List[ str ]:
		"""Get audio task options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and
		    downstream logic can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		tasks: List[ str ] = [ ]
		
		if audio_has_method( transcriber, [ 'transcribe', 'create_transcription', 'create' ] ):
			tasks.append( 'Transcribe' )
		
		if audio_has_method( translator, [ 'translate', 'create_translation', 'create' ] ):
			tasks.append( 'Translate' )
		
		if audio_has_method( tts, [ 'create_speech', 'synthesize', 'generate', 'create' ] ):
			tasks.append( 'Text-to-Speech' )
		
		return tasks
	
	def get_audio_task_instance( task: Optional[ str ] ) -> Any:
		"""Get audio task instance.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and
		    downstream logic can consume it consistently.
		
		Args:
		    task (Optional[str]): Task value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation."""
		if task == 'Translate':
			return translator
		
		if task == 'Text-to-Speech':
			return tts
		
		return transcriber
	
	def get_audio_model_options( task: Optional[ str ] ) -> List[ str ]:
		"""Get audio model options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI controls
		    and downstream logic can consume it consistently.
		
		Args:
		    task (Optional[str]): Task value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		instance = get_audio_task_instance( task )
		options = get_audio_options( instance, 'model_options' )
		
		if not options:
			model_value = getattr( instance, 'model', '' )
			options = [ model_value ] if model_value else [ ]
		
		return [ str( option ) for option in options if str( option ).strip( ) ]
	
	def get_audio_language_options( task: Optional[ str ] ) -> List[ str ]:
		"""Get audio language options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and
		    downstream logic can consume it consistently.
		
		Args:
		    task (Optional[str]): Task value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation.
		"""
		instance = get_audio_task_instance( task )
		options = get_audio_options( instance, 'language_options' )
		
		if not options:
			options = [ 'auto', 'en', 'Spanish', 'French', 'German', 'Italian', 'Japanese' ]
		
		return [ str( option ) for option in options if str( option ).strip( ) ]
	
	def get_audio_voice_options( ) -> List[ str ]:
		"""Get audio voice options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation.
		"""
		options = get_audio_options( tts, 'voice_options' )
		if not options:
			options = [ getattr( tts, 'voice', '' ) ]
		
		return [ str( option ) for option in options if str( option ).strip( ) ]
	
	def get_audio_format_options( task: Optional[ str ] ) -> List[ Any ]:
		"""Get audio format options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI controls
		    and downstream logic can consume it consistently.
		
		Args:
		    task (Optional[str]): Task value used by the operation.
		
		Returns:
		    List[Any]: Return value produced by the operation.
		"""
		instance = get_audio_task_instance( task )
		
		if task == 'Text-to-Speech':
			options = get_audio_options( instance, 'format_options' )
			if not options:
				options = get_audio_options( instance, 'response_format_options' )
			if not options:
				options = get_audio_options( instance, 'output_format_options' )
			if not options:
				options = [ 'mp3', 'wav' ]
			
			return options
		
		options = get_audio_options( instance, 'response_format_options' )
		if not options:
			options = get_audio_options( instance, 'format_options' )
		if not options:
			options = [ 'text', 'json' ]
		
		return options
	
	def get_audio_include_options( task: Optional[ str ] ) -> List[ str ]:
		"""Get audio include options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI controls
		    and down stream logic can consume it consistently.
		
		Args:
		    task (Optional[str]): Task value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation.
		"""
		instance = get_audio_task_instance( task )
		options = get_audio_options( instance, 'include_options' )
		return [ str( option ) for option in options if str( option ).strip( ) ]
	
	def get_audio_sample_rate_options( ) -> List[ int ]:
		"""Get audio sample rate options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and  downstream logic can consume it consistently.
		
		Returns:
		    List[int]: Return value produced by the operation.
		"""
		options = get_audio_options( tts, 'sample_rate_options' )
		if not options:
			options = [ 0, 8000, 16000, 22050, 24000, 44100, 48000 ]
		
		values: List[ int ] = [ 0 ]
		for option in options:
			try:
				value = int( option )
				if value not in values:
					values.append( value )
			except Exception:
				continue
		
		return values
	
	def get_audio_bit_rate_options( ) -> List[ int ]:
		"""Get audio bit rate options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and downstream logic can consume it consistently.
		
		Returns:
		    List[int]: Return value produced by the operation.
		"""
		options = get_audio_options( tts, 'bit_rate_options' )
		if not options:
			options = [ 0, 32000, 64000, 96000, 128000, 192000 ]
		
		values: List[ int ] = [ 0 ]
		for option in options:
			try:
				value = int( option )
				if value not in values:
					values.append( value )
			except Exception:
				continue
		
		return values
	
	def sanitize_audio_selection( key: str, valid_options: List[ Any ], default: Any='' ) -> None:
		"""Sanitize audio selection.
		
		Purpose:
		    Performs the sanitize_audio_selection workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so
		    related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    key (str): Key value used by the operation.
		    valid_options (List[Any]): Valid options value used by the operation.
		    default (Any): Default value used by the operation.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value."""
		current_value = st.session_state.get( key, default )
		
		if current_value in [ None, '' ]:
			return
		
		if valid_options and current_value not in valid_options:
			st.session_state[ key ] = default
	
	def sanitize_audio_multiselect( key: str, valid_options: List[ str ] ) -> None:
		"""Sanitize audio multiselect.
		
		Purpose:
		    Performs the sanitize_audio_multiselect workflow using the inputs supplied by the
		    caller and the current runtime configuration. The function keeps this behavior
		    isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    key (str): Key value used by the operation.
		    valid_options (List[str]): Valid options value used by the operation.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value.
		"""
		current_values = st.session_state.get( key, [ ] )
		
		if not isinstance( current_values, list ):
			st.session_state[ key ] = [ ]
			return
		
		st.session_state[ key ] = [ item for item in current_values if item in valid_options ]
	
	def parse_audio_domains( value: Any ) -> List[ str ]:
		"""Parse audio domains.
		
		Purpose:
		    Performs the parse_audio_domains workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so
		    related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    value (Any): Value value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		raw = str( value or '' )
		return [ item.strip( ) for item in raw.split( ',' ) if item.strip( ) ]
	
	def save_audio_upload( uploaded_file: Any ) -> Optional[ str ]:
		"""Save audio upload.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application
		    workflows. The function standardizes file handling and returns a stable reference
		    for downstream processing.
		
		Args:
		    uploaded_file (Any): Uploaded file value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation."""
		if uploaded_file is None:
			return None
		
		if 'save_temp' in globals( ):
			try:
				return save_temp( uploaded_file )
			except Exception:
				pass
		
		try:
			name = getattr( uploaded_file, 'name', 'audio.wav' )
			_, ext = os.path.splitext( name )
			ext = ext or '.wav'
			
			with tempfile.NamedTemporaryFile( delete=False, suffix=ext ) as tmp:
				if hasattr( uploaded_file, 'getbuffer' ):
					tmp.write( uploaded_file.getbuffer( ) )
				elif hasattr( uploaded_file, 'getvalue' ):
					tmp.write( uploaded_file.getvalue( ) )
				elif hasattr( uploaded_file, 'read' ):
					tmp.write( uploaded_file.read( ) )
				else:
					return None
				
				return tmp.name
		except Exception:
			return None
	
	def append_audio_message( role: str, content: str ) -> None:
		"""Append audio message.
		
		Purpose:
		    Performs the append_audio_message workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so
		    related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    role (str): Role value used by the operation.
		    content (str): Content value used by the operation.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value.
		"""
		if not isinstance( st.session_state.get( 'audio_messages' ), list ):
			st.session_state[ 'audio_messages' ] = [ ]
		
		st.session_state[ 'audio_messages' ].append( { 'role': role, 'content': content, } )
	
	def render_audio_messages( ) -> None:
		"""Render audio messages.
		
		Purpose:
		    Renders the requested user interface element or result block in Streamlit using
		    normalized inputs. The function keeps presentation logic isolated from provider calls
		    and
		    data-processing steps so the screen output remains predictable.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
				value.
		"""
		if not isinstance( st.session_state.get( 'audio_messages' ), list ):
			st.session_state[ 'audio_messages' ] = [ ]
		
		for msg in st.session_state.get( 'audio_messages', [ ] ):
			if not isinstance( msg, dict ):
				continue
			
			with st.chat_message( msg.get( 'role', 'assistant' ), avatar='' ):
				st.markdown( msg.get( 'content', '' ) )
	
	def clear_audio_messages( ) -> None:
		"""Clear audio messages.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled
		    manner. The function keeps cleanup behavior centralized so callers do not duplicate
		    lifecycle
		    logic.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value.
		"""
		st.session_state[ 'audio_messages' ] = [ ]
		st.session_state[ 'audio_output' ] = ''
		st.session_state[ 'audio_output_bytes' ] = None
		st.session_state[ 'audio_output_path' ] = ''
		st.session_state[ 'audio_last_result' ] = { }
		st.session_state[ 'audio_last_usage' ] = { }
	
	def clear_audio_instructions( ) -> None:
		"""Clear audio instructions.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled
		    manner. The function keeps cleanup behavior centralized so callers do not duplicate
		    lifecycle
		    logic.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value.
		"""
		st.session_state[ 'audio_system_instructions' ] = ''
		st.session_state[ 'instructions' ] = ''
	
	def convert_audio_system_instructions( ) -> None:
		"""Convert audio system instructions.
		
		Purpose:
		    Performs the convert_audio_system_instructions workflow using the inputs supplied by
		    the caller and the current runtime configuration. The function keeps this behavior
		    isolated so
		    related UI,  provider, and data-processing paths can call it consistently.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value.
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
		"""Load audio instruction template.
		
		Purpose:
		    Loads the selected Audio-mode prompt template into the Audio-mode system-instruction
		    field using the stable prompt identifier stored in session state.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			load_prompt_template( prompt_id_key='audio_prompt_id',
				instructions_key='audio_system_instructions', )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Audio Mode'
			ex.method = 'load_audio_instruction_template( ) -> None'
			Logger( ).write( ex )
			raise ex
	
	def reset_audio_task_controls( ) -> None:
		"""Reset audio task controls.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled
		    manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle
		    logic.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
			value.
		"""
		for key in [ 'audio_task', 'audio_model', 'audio_language', 'audio_voice', 'audio_format',
			'audio_response_format', 'audio_speed', 'audio_sample_rate', 'audio_bit_rate', ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_audio_inference_controls( ) -> None:
		"""Reset audio inference controls.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled
		    manner. The function keeps cleanup behavior centralized so callers do not duplicate
		    lifecycle
		    logic.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value.
		"""
		for key in [ 'audio_temperature', 'audio_top_percent', 'audio_top_k',
			'audio_frequency_penalty', 'audio_presence_penalty', 'audio_presense_penalty',
			'audio_max_tokens', 'audio_include', 'audio_stream', 'audio_store',
			'audio_background', ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_audio_playback_controls( ) -> None:
		"""Reset audio playback controls.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled
		    manner. The function keeps cleanup behavior centralized so callers do not duplicate
		    lifecycle logic.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		        value."""
		for key in [ 'audio_start_time', 'audio_end_time', 'audio_loop', 'audio_autoplay',
			'audio_output_bytes', 'audio_output_path', 'audio_upload_path',
			'audio_recorded_path', ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def update_audio_usage( instance: Any ) -> None:
		"""Update audio usage.
		
		Purpose:
		    Performs the update_audio_usage workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so
		    related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    instance (Any): Instance value used by the operation.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		try:
			response = getattr( instance, 'response', None )
			usage = getattr( response, 'usage', None )
			
			if usage is None and hasattr( instance, 'get_usage' ):
				usage = instance.get_usage( )
			
			if usage is None:
				st.session_state[ 'audio_last_usage' ] = { }
				return
			
			if hasattr( usage, 'model_dump' ):
				st.session_state[ 'audio_last_usage' ] = usage.model_dump( )
			elif isinstance( usage, dict ):
				st.session_state[ 'audio_last_usage' ] = usage
			else:
				st.session_state[ 'audio_last_usage' ] = { 'usage': str( usage ) }
			
			if 'update_token_counters' in globals( ):
				update_token_counters( response )
		except Exception:
			st.session_state[ 'audio_last_usage' ] = { }
	
	def normalize_audio_text_result( result: Any ) -> str:
		"""Normalize audio text result.

		Purpose:
		    Normalizes provider transcription and translation results into displayable text.

		Args:
		    result (Any): Provider result returned by the active audio wrapper.

		Returns:
		    str: Extracted text or an empty string.
		"""
		if result is None:
			return ''
		
		if isinstance( result, str ):
			return result.strip( )
		
		if isinstance( result, dict ):
			for key in [ 'text', 'transcript', 'translation', 'content', 'output_text' ]:
				value = result.get( key )
				if isinstance( value, str ) and value.strip( ):
					return value.strip( )
			
			return str( result )
		
		for attr_name in [ 'text', 'transcript', 'translation', 'content', 'output_text' ]:
			value = getattr( result, attr_name, None )
			if isinstance( value, str ) and value.strip( ):
				return value.strip( )
		
		return str( result ).strip( )
	
	def normalize_audio_bytes_result( result: Any ) -> Optional[ bytes ]:
		"""Normalize audio bytes result.

		Purpose:
		    Normalizes provider text-to-speech results into audio bytes.

		Args:
		    result (Any): Provider result returned by the active text-to-speech wrapper.

		Returns:
		    Optional[bytes]: Generated audio bytes when available; otherwise None.
		"""
		if result is None:
			return None
		
		if isinstance( result, bytes ):
			return result
		
		if isinstance( result, bytearray ):
			return bytes( result )
		
		if isinstance( result, dict ):
			for key in [ 'audio_bytes', 'bytes', 'content', 'data', 'audio' ]:
				value = result.get( key )
				if isinstance( value, bytes ):
					return value
				
				if isinstance( value, bytearray ):
					return bytes( value )
		
		for attr_name in [ 'audio_bytes', 'bytes', 'content', 'data', 'audio' ]:
			value = getattr( result, attr_name, None )
			if isinstance( value, bytes ):
				return value
			
			if isinstance( value, bytearray ):
				return bytes( value )
		
		return None
	
	def get_audio_mime_type( format_value: Any ) -> str:
		"""Get audio MIME type.

		Purpose:
		    Converts the selected provider audio format into a valid MIME type for Streamlit
		    playback and download controls.

		Args:
		    format_value (Any): Provider audio format or MIME-type value.

		Returns:
		    str: Valid audio MIME type.
		"""
		format_text = str( format_value or '' ).strip( ).lower( )
		
		if not format_text:
			return 'audio/mpeg'
		
		if format_text.startswith( 'audio/' ):
			return format_text
		
		mime_map = { 'mp3': 'audio/mpeg', 'mpeg': 'audio/mpeg', 'mpga': 'audio/mpeg',
			'wav': 'audio/wav', 'pcm': 'audio/pcm', 'opus': 'audio/opus', 'ogg': 'audio/ogg',
			'flac': 'audio/flac', 'aac': 'audio/aac', 'm4a': 'audio/mp4', 'mp4': 'audio/mp4',
			'webm': 'audio/webm', 'mulaw': 'audio/basic', 'alaw': 'audio/basic', }
		
		return mime_map.get( format_text, f'audio/{format_text}' )
	
	def get_audio_file_extension( format_value: Any ) -> str:
		"""Get audio file extension.

		Purpose:
		    Converts the selected provider audio format into a safe download-file extension.

		Args:
		    format_value (Any): Provider audio format or MIME-type value.

		Returns:
		    str: Audio file extension without a leading period.
		"""
		format_text = str( format_value or '' ).strip( ).lower( )
		
		if '/' in format_text:
			format_text = format_text.rsplit( '/', 1 )[ -1 ]
		
		extension_map = { 'mpeg': 'mp3', 'x-wav': 'wav', 'wave': 'wav', 'basic': 'au', }
		
		return extension_map.get( format_text, format_text or 'mp3' )
	
	def get_audio_source_mime_type( path: str, selected_format: Any ) -> str:
		"""Get audio source MIME type.

		Purpose:
		    Resolves the source-audio MIME type required by Gemini and Grok upload workflows.

		Args:
		    path (str): Local source-audio path.
		    selected_format (Any): Current format selection from Audio Mode.

		Returns:
		    str: Source-audio MIME type.
		"""
		selected_text = str( selected_format or '' ).strip( ).lower( )
		
		if selected_text.startswith( 'audio/' ):
			return selected_text
		
		suffix = Path( path ).suffix.lower( )
		mime_map = { '.mp3': 'audio/mpeg', '.mpeg': 'audio/mpeg', '.mpga': 'audio/mpeg',
			'.wav': 'audio/wav', '.flac': 'audio/flac', '.ogg': 'audio/ogg', '.webm': 'audio/webm',
			'.mp4': 'audio/mp4', '.m4a': 'audio/m4a', '.aac': 'audio/aac', '.aiff': 'audio/aiff', }
		
		return mime_map.get( suffix, get_audio_mime_type( selected_text ) )
	
	def get_audio_target_language( ) -> str:
		"""Get audio target language.

		Purpose:
		    Returns the selected target language required by provider translation wrappers.

		Returns:
		    str: Selected target language.
		"""
		return str( st.session_state.get( 'audio_language', '' ) or '' ).strip( )
	
	def run_audio_transcription( path: str, prompt: Optional[ str ] = None ) -> str:
		"""Run audio transcription.

		Purpose:
		    Executes the selected provider transcription wrapper using only arguments implemented
		    by that provider contract.

		Args:
		    path (str): Required local source-audio path.
		    prompt (Optional[str]): Optional transcription guidance.

		Returns:
		    str: Generated transcript.

		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			throw_if( 'path', path )
			model = str( st.session_state.get( 'audio_model', '' ) or '' ).strip( )
			throw_if( 'model', model )
			language = str( st.session_state.get( 'audio_language', '' ) or '' ).strip( )
			response_format = st.session_state.get( 'audio_response_format' )
			prompt_text = str( prompt or '' )
			
			if provider_name == 'GPT':
				result = transcriber.transcribe( path=path, model=model, language=language,
					prompt=prompt_text, format=str( response_format or 'json' ),
					temperature=float( st.session_state.get( 'audio_temperature', 0.0 ) or 0.0 ),
					include=list( st.session_state.get( 'audio_include', [ ] ) or [ ] ), )
			
			elif provider_name == 'Gemini':
				result = transcriber.transcribe( path=path, model=model,
					language=language or 'Auto',
					mime_type=get_audio_source_mime_type( path, response_format ),
					temperature=float( st.session_state.get( 'audio_temperature', 0.0 ) or 0.0 ),
					top_p=float( st.session_state.get( 'audio_top_percent', 0.0 ) or 0.0 ),
					top_k=int( st.session_state.get( 'audio_top_k', 0 ) or 0 ), frequency=float(
						st.session_state.get( 'audio_frequency_penalty', 0.0 ) or 0.0 ),
					presence=float( st.session_state.get( 'audio_presence_penalty', 0.0 ) or 0.0 ),
					max_tokens=int( st.session_state.get( 'audio_max_tokens', 0 ) or 0 ),
					start_time=float( st.session_state.get( 'audio_start_time', 0.0 ) or 0.0 ),
					end_time=float( st.session_state.get( 'audio_end_time', 0.0 ) or 0.0 ),
					instruct=str( st.session_state.get( 'audio_system_instructions', '' ) or '' ),
					prompt=prompt_text, )
			
			elif provider_name == 'Grok':
				result = transcriber.transcribe( path=path, language=language,
					format=bool( response_format ),
					mime_type=get_audio_source_mime_type( path, '' ), keyterm=prompt_text, )
			
			else:
				raise ValueError( f'Unsupported Audio provider: {provider_name}' )
			
			text_result = normalize_audio_text_result( result )
			st.session_state[ 'audio_output' ] = text_result
			st.session_state[ 'audio_last_result' ] = { 'task': 'Transcribe',
				'provider': provider_name, 'model': model, 'text': text_result, }
			update_audio_usage( transcriber )
			return text_result
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Audio'
			ex.method = ('run_audio_transcription( path: str, '
			             'prompt: Optional[ str ] = None ) -> str')
			Logger( ).write( ex )
			raise ex
	
	def run_audio_translation( path: str, prompt: Optional[ str ] = None ) -> str:
		"""Run audio translation.

		Purpose:
		    Executes the selected provider audio-translation wrapper using only arguments
		    implemented by that provider contract.

		Args:
		    path (str): Required local source-audio path.
		    prompt (Optional[str]): Optional translation guidance.

		Returns:
		    str: Generated translated text.

		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			throw_if( 'path', path )
			model = str( st.session_state.get( 'audio_model', '' ) or '' ).strip( )
			throw_if( 'model', model )
			target_language = get_audio_target_language( )
			response_format = st.session_state.get( 'audio_response_format' )
			prompt_text = str( prompt or '' )
			
			if provider_name == 'GPT':
				result = translator.translate( path=path, model=model, prompt=prompt_text,
					format=str( response_format or 'json' ),
					temperature=float( st.session_state.get( 'audio_temperature', 0.0 ) or 0.0 ), )
			
			elif provider_name == 'Gemini':
				throw_if( 'audio_language', target_language )
				result = translator.translate( path=path, model=model, language=target_language,
					source='Auto', mime_type=get_audio_source_mime_type( path, response_format ),
					temperature=float( st.session_state.get( 'audio_temperature', 0.0 ) or 0.0 ),
					top_p=float( st.session_state.get( 'audio_top_percent', 0.0 ) or 0.0 ),
					top_k=int( st.session_state.get( 'audio_top_k', 0 ) or 0 ), frequency=float(
						st.session_state.get( 'audio_frequency_penalty', 0.0 ) or 0.0 ),
					presence=float( st.session_state.get( 'audio_presence_penalty', 0.0 ) or 0.0 ),
					max_tokens=int( st.session_state.get( 'audio_max_tokens', 0 ) or 0 ),
					start_time=float( st.session_state.get( 'audio_start_time', 0.0 ) or 0.0 ),
					end_time=float( st.session_state.get( 'audio_end_time', 0.0 ) or 0.0 ),
					instruct=str( st.session_state.get( 'audio_system_instructions', '' ) or '' ),
					prompt=prompt_text, )
			
			elif provider_name == 'Grok':
				throw_if( 'audio_language', target_language )
				result = translator.translate( path=path, target_language=target_language,
					model=model, source_language='', text_format=bool( response_format ),
					mime_type=get_audio_source_mime_type( path, '' ), keyterm=prompt_text,
					instruct=str( st.session_state.get( 'audio_system_instructions', '' ) or ''
					), )
			
			else:
				raise ValueError( f'Unsupported Audio provider: {provider_name}' )
			
			text_result = normalize_audio_text_result( result )
			st.session_state[ 'audio_output' ] = text_result
			st.session_state[ 'audio_last_result' ] = { 'task': 'Translate',
				'provider': provider_name, 'model': model,
				'target_language': target_language or 'English', 'text': text_result, }
			update_audio_usage( translator )
			return text_result
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Audio'
			ex.method = ('run_audio_translation( path: str, '
			             'prompt: Optional[ str ] = None ) -> str')
			Logger( ).write( ex )
			raise ex
	
	def run_audio_tts( text: str ) -> Optional[ bytes ]:
		"""Run audio text-to-speech.

		Purpose:
		    Executes the selected provider text-to-speech wrapper using only arguments
		    implemented by that provider contract.

		Args:
		    text (str): Required text converted to speech.

		Returns:
		    Optional[bytes]: Generated audio bytes when available.

		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			throw_if( 'text', text )
			model = str( st.session_state.get( 'audio_model', '' ) or '' ).strip( )
			voice = str( st.session_state.get( 'audio_voice', '' ) or '' ).strip( )
			response_format = st.session_state.get( 'audio_response_format' )
			speed = float( st.session_state.get( 'audio_speed', 1.0 ) or 1.0 )
			output_path = str( st.session_state.get( 'audio_output_path', '' ) or '' )
			throw_if( 'voice', voice )
			
			if provider_name == 'GPT':
				throw_if( 'model', model )
				result = tts.create_speech( text=text, model=model,
					format=str( response_format or 'mp3' ), speed=speed, voice=voice,
					instruct=str( st.session_state.get( 'audio_system_instructions', '' ) or '' ),
					file_path=output_path, )
			
			elif provider_name == 'Gemini':
				throw_if( 'model', model )
				result = tts.create_speech( text=text, model=model,
					format=str( response_format or 'audio/wav' ), voice=voice, speed=speed,
					instruct=str( st.session_state.get( 'audio_system_instructions', '' ) or '' ),
					file_path=output_path,
					temperature=float( st.session_state.get( 'audio_temperature', 0.0 ) or 0.0 ),
					top_p=float( st.session_state.get( 'audio_top_percent', 0.0 ) or 0.0 ),
					max_tokens=int( st.session_state.get( 'audio_max_tokens', 0 ) or 0 ),
					sample_rate=int(
						st.session_state.get( 'audio_sample_rate', 24000 ) or 24000 ), )
			
			elif provider_name == 'Grok':
				result = tts.create_speech( text=text,
					language=str( st.session_state.get( 'audio_language', '' ) or 'auto' ),
					voice_id=voice, output_format=str( response_format or 'mp3' ), speed=speed,
					sample_rate=int( st.session_state.get( 'audio_sample_rate', 24000 ) or 24000 ),
					bit_rate=int( st.session_state.get( 'audio_bit_rate', 128000 ) or 128000 ),
					filepath=output_path, )
			
			else:
				raise ValueError( f'Unsupported Audio provider: {provider_name}' )
			
			audio_bytes = normalize_audio_bytes_result( result )
			st.session_state[ 'audio_output_bytes' ] = audio_bytes
			st.session_state[ 'audio_last_result' ] = { 'task': 'Text-to-Speech',
				'provider': provider_name, 'model': model, 'format': str( response_format or '' ),
				'bytes': len( audio_bytes ) if audio_bytes else 0, }
			update_audio_usage( tts )
			return audio_bytes
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Audio'
			ex.method = 'run_audio_tts( text: str ) -> Optional[ bytes ]'
			Logger( ).write( ex )
			raise ex
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '🎧 Audio API', help=get_audio_help( 'AUDIO_API' ) )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Expander - Mind Controls (Audio)
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				aud_c1, aud_c2, aud_c3, aud_c4, aud_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], gap='xxsmall', border=True )
				
				task_options = get_audio_task_options( )
				sanitize_audio_selection( 'audio_task', task_options, '' )
				
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
				format_options = get_audio_format_options( audio_task )
				include_options = get_audio_include_options( audio_task )
				sample_rate_options = get_audio_sample_rate_options( )
				bit_rate_options = get_audio_bit_rate_options( )
				sanitize_audio_selection( 'audio_model', model_options, '' )
				sanitize_audio_selection( 'audio_language', language_options, '' )
				sanitize_audio_selection( 'audio_voice', voice_options, '' )
				sanitize_audio_selection( 'audio_response_format', format_options, '' )
				sanitize_audio_multiselect( 'audio_include', include_options )
				
				# ---------- Model ------------
				with aud_c2:
					st.selectbox( label='Model', options=model_options, key='audio_model',
						placeholder='Options', index=None, help='Task-aware Audio API model.' )
				
				# ---------- Language / Voice ------------
				with aud_c3:
					if audio_task == 'Text-to-Speech':
						st.selectbox( label='Voice', options=voice_options, key='audio_voice',
							placeholder='Options', index=None,
							help='Text-to-speech voice when supported.' )
					else:
						st.selectbox( label='Language', options=language_options,
							key='audio_language', placeholder='Options', index=None,
							help='Language hint or translation target when supported.' )
				
				# ---------- Format ------------
				with aud_c4:
					st.selectbox( label='Format', options=format_options,
						key='audio_response_format', placeholder='Options', index=None,
						help='Audio output or text response format.' )
					st.session_state[ 'audio_format' ] = st.session_state.get(
						'audio_response_format', '' )
				
				# ---------- Speed ------------
				with aud_c5:
					st.slider( label='Speed', min_value=0.25, max_value=4.00, step=0.25,
						key='audio_speed',
						help='Playback/synthesis speed when supported by the provider.' )
				
				sr_c1, sr_c2 = st.columns( [ 0.50, 0.50 ], border=True, gap='xxsmall' )
				
				# ---------- Sample Rate ------------
				with sr_c1:
					st.selectbox( label='Sample Rate', options=sample_rate_options,
						key='audio_sample_rate', index=None, placeholder='Options',
						help='Optional TTS sample rate. Zero/blank means provider default.' )
				
				# ---------- Bit Rate ------------
				with sr_c2:
					st.selectbox( label='Bit Rate', options=bit_rate_options, key='audio_bit_rate',
						index=None, placeholder='Options',
						help='Optional TTS MP3 bit rate. Zero/blank means provider default.' )
				
				# ----- Reset Button -----
				st.button( label='Reset', key='audio_task_reset', width='stretch',
					on_click=reset_audio_task_controls, icon='🔄' )
			
			# ------------------------------------------------------------------
			# Expander - Inference Settings (Audio)
			# ------------------------------------------------------------------
			with st.expander( label='Inference Settings', icon='🎚️', expanded=False,
					width='stretch' ):
				inf_c1, inf_c2, inf_c3, inf_c4, inf_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], gap='xxsmall', border=True )
				
				# ---------- Top-P ------------
				with inf_c1:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0, step=0.01,
						key='audio_top_percent', help=cfg.TOP_P )
				
				# ---------- Temperature ------------
				with inf_c2:
					st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.01,
						key='audio_temperature', help=cfg.TEMPERATURE )
				
				# ---------- Frequency Penalty ------------
				with inf_c3:
					st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='audio_frequency_penalty', help=cfg.FREQUENCY_PENALTY )
				
				# ---------- Presence Penalty ------------
				with inf_c4:
					st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='audio_presence_penalty', help=cfg.PRESENCE_PENALTY )
					st.session_state[ 'audio_presense_penalty' ] = st.session_state.get(
						'audio_presence_penalty', 0.0 )
				
				# ---------- Max Tokens ------------
				with inf_c5:
					st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
						key='audio_max_tokens', help=cfg.MAX_OUTPUT_TOKENS )
				
				ctl_c1, ctl_c2, ctl_c3, ctl_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					gap='xxsmall', border=True )
				
				# ---------- Include ------------
				with ctl_c1:
					st.multiselect( label='Include', options=include_options, key='audio_include',
						placeholder='Options', help=cfg.INCLUDE )
				
				# ---------- Store ------------
				with ctl_c2:
					st.toggle( label='Store', key='audio_store', help=cfg.STORE )
				
				# ---------- Stream ------------
				with ctl_c3:
					st.toggle( label='Stream', key='audio_stream', help=cfg.STREAM )
				
				# ---------- Background ------------
				with ctl_c4:
					st.toggle( label='Background', key='audio_background',
						help=cfg.BACKGROUND_MODE )
				
				# ----- Reset Button -----
				st.button( label='Reset', key='audio_inference_reset', width='stretch',
					on_click=reset_audio_inference_controls, icon='🔄' )
			
			# ------------------------------------------------------------------
			# Expander - Playback
			# ------------------------------------------------------------------
			with st.expander( label='Playback Settings', icon='🔊', expanded=False,
					width='stretch' ):
				play_c1, play_c2, play_c3, play_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					gap='xxsmall', border=True )
				
				# ---------- Start Time ------------
				with play_c1:
					st.number_input( label='Start Time', min_value=0.0, step=1.0,
						key='audio_start_time', help='Optional segment start time.' )
				
				# ---------- End Time ------------
				with play_c2:
					st.number_input( label='End Time', min_value=0.0, step=1.0,
						key='audio_end_time', help='Optional segment end time.' )
				
				# ---------- Loop ------------
				with play_c3:
					st.toggle( label='Loop', key='audio_loop',
						help='Loop playback when Streamlit supports it.' )
				
				# ---------- Autoplay ------------
				with play_c4:
					st.toggle( label='Autoplay', key='audio_autoplay',
						help='Autoplay playback when Streamlit supports it.' )
				
				# ----- Reset Button ------
				st.button( label='Reset', key='audio_playback_reset', width='stretch',
					on_click=reset_audio_playback_controls, icon='🔄' )
		
		# ------------------------------------------------------------------
		# Expander — Audio System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			# ----- Prompt Categories ------
			audio_prompt_categories = fetch_prompt_categories( 'Audio' )
			current_audio_category = st.session_state.get( 'audio_prompt_category' )
			if current_audio_category not in audio_prompt_categories:
				st.session_state[ 'audio_prompt_category' ] = None
			
			selected_audio_category = st.session_state.get( 'audio_prompt_category' )
			audio_prompt_options = fetch_prompt_options(
				selected_audio_category ) if selected_audio_category else [ ]
			
			audio_prompt_ids = [ int( option[ 'ID' ] ) for option in audio_prompt_options ]
			if st.session_state.get( 'audio_prompt_id' ) not in audio_prompt_ids:
				st.session_state[ 'audio_prompt_id' ] = None
			
			# ----- Instruction Text -----
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					key='audio_system_instructions', help=cfg.SYSTEM_INSTRUCTIONS )
			
			# ------ Template Selection ------
			with in_right:
				st.selectbox( label='Category', options=audio_prompt_categories, index=None,
					key='audio_prompt_category', placeholder='Select Category',
					help='Limits prompt templates to categories associated with audio workflows.',
					on_change=reset_prompt_template_selection, args=('audio_prompt_id',), )
				
				st.selectbox( label='Use Template', options=audio_prompt_ids, index=None,
					key='audio_prompt_id', placeholder='Select Template',
					disabled=not audio_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						audio_prompt_options, ),
					help='Loads the selected prompt into the Audio system-instruction field.',
					on_change=load_audio_instruction_template, )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ----- Clear Button -----
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_audio_instructions, icon='🧹' )
			
			# ----- Convert Button -----
			with btn_c2:
				st.button( label='XML ↔️ Markdown', width='stretch',
					on_click=convert_audio_system_instructions, )
		
		# ------------------------------------------------------------------
		# Audio Workflows
		# ------------------------------------------------------------------
		workflows = [ 'Transcribe / Translate', 'Text-to-Speech', 'Playback' ]
		tab_process, tab_tts, tab_playback = st.tabs( workflows )
		
		# ----- Audio Process -----
		with tab_process:
			render_audio_messages( )
			audio_input_c1, audio_input_c2 = st.columns( [ 0.50, 0.50 ], gap='small' )
			
			# ---------- Upload ------------
			with audio_input_c1:
				uploaded_audio = st.file_uploader( label='Upload Audio',
					type=[ 'wav', 'mp3', 'mpeg', 'mp4', 'm4a', 'webm', 'ogg', 'flac', ],
					accept_multiple_files=False, key='audio_uploaded_file' )
			
			# ---------- Record ------------
			with audio_input_c2:
				recorded_audio = None
				if hasattr( st, 'audio_input' ):
					recorded_audio = st.audio_input( label='Record Audio',
						key='audio_recorded_file' )
			
			audio_prompt_c1, audio_prompt_c2 = st.columns( [ 0.50, 0.50 ], gap='small' )
			
			# ---------- Transcription ------------
			with audio_prompt_c1:
				transcription_prompt = st.text_area( label='Transcription Prompt',
					key='transcription_prompt', height=80, width='stretch',
					placeholder=('Optional transcription prompt or vocabulary/context hints.') )
			
			# ---------- Translation ------------
			with audio_prompt_c2:
				translation_prompt = st.text_area( label='Translation Prompt',
					key='translation_prompt', height=80, width='stretch',
					placeholder='Optional translation prompt or instructions.' )
			
			# ------------------------------------------------------------------
			# Audio Source Resolution
			# ------------------------------------------------------------------
			audio_path = None
			if uploaded_audio is not None:
				audio_path = save_audio_upload( uploaded_audio )
				st.session_state[ 'audio_upload_path' ] = audio_path or ''
				
				try:
					st.audio( uploaded_audio )
				except Exception:
					pass
			
			elif recorded_audio is not None:
				audio_path = save_audio_upload( recorded_audio )
				st.session_state[ 'audio_recorded_path' ] = audio_path or ''
				
				try:
					st.audio( recorded_audio )
				except Exception:
					pass
			
			process_c1, process_c2 = st.columns( [ 0.50, 0.50 ] )
			
			# ----- Process Button -----
			with process_c1:
				if st.button( 'Process Audio', key='process_audio', width='stretch', icon='⚡' ):
					with st.spinner( 'Processing audio…' ):
						try:
							selected_task = st.session_state.get( 'audio_task' )
							if selected_task not in [ 'Transcribe', 'Translate' ]:
								st.warning(
									'Select Transcribe or Translate before processing audio.' )
							
							elif not audio_path:
								st.warning( 'Upload or record audio before processing.' )
							
							elif not st.session_state.get( 'audio_model' ):
								st.warning( 'Select a model before processing audio.' )
							
							elif selected_task == 'Transcribe':
								result_text = run_audio_transcription( audio_path,
									transcription_prompt )
								if result_text:
									append_audio_message( 'user',
										'Transcribe uploaded/recorded audio.' )
									append_audio_message( 'assistant', result_text )
									st.text_area( 'Transcript', value=result_text, height=300 )
								else:
									st.warning( 'No transcript was returned.' )
							
							elif selected_task == 'Translate':
								result_text = run_audio_translation( audio_path,
									translation_prompt )
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
			
			# ----- Clear Button -----
			with process_c2:
				if st.button( 'Clear Messages', key='audio_clear_process_messages',
						width='stretch',
						on_click=clear_audio_messages, icon='🧹' ):
					st.rerun( )
			
			if st.session_state.get( 'audio_output' ):
				st.download_button( label='Download Text Output',
					data=st.session_state.get( 'audio_output', '' ), file_name='audio_output.txt',
					mime='text/plain', width='stretch' )
		
		# ----- Text To Speech -----
		with tab_tts:
			render_audio_messages( )
			tts_input = st.text_area( label='Enter Text to Synthesize', key='audio_tts_input',
				height=160, width='stretch', placeholder='Enter text for speech synthesis.' )
			
			tts_c1, tts_c2 = st.columns( [ 0.50, 0.50 ] )
			
			# ----- Generate Button -----
			with tts_c1:
				if st.button( 'Generate Audio', key='generate_tts_audio', width='stretch',
						icon='🗣️️' ):
					with st.spinner( 'Synthesizing speech…' ):
						try:
							if st.session_state.get( 'audio_task' ) != 'Text-to-Speech':
								st.warning( 'Select Text-to-Speech as the Audio mode first.' )
							elif not isinstance( tts_input, str ) or not tts_input.strip( ):
								st.warning( 'Enter text before generating speech.' )
							elif not st.session_state.get( 'audio_model' ):
								st.warning( 'Select a model before generating speech.' )
							else:
								audio_bytes = run_audio_tts( tts_input.strip( ) )
								
								if audio_bytes:
									append_audio_message( 'user', tts_input.strip( ) )
									append_audio_message( 'assistant',
										'Text-to-speech audio generated successfully.' )
									st.audio( audio_bytes, format=get_audio_mime_type(
										st.session_state.get( 'audio_response_format', 'mp3' ) ) )
								else:
									st.warning( 'No audio bytes were returned.' )
						
						except Exception as exc:
							err = Error( exc )
							st.error( f'Text-to-speech failed: {err.info}' )
			
			# ----- Clear Button -----
			with tts_c2:
				if st.button( 'Clear Messages', key='audio_clear_tts_messages', width='stretch',
						on_click=clear_audio_messages, icon='🧹' ):
					st.rerun( )
			
			if st.session_state.get( 'audio_output_bytes' ):
				audio_format = st.session_state.get( 'audio_response_format', 'mp3' ) or 'mp3'
				audio_extension = get_audio_file_extension( audio_format )
				audio_mime_type = get_audio_mime_type( audio_format )
				st.download_button( label='Download Audio',
					data=st.session_state.get( 'audio_output_bytes' ),
					file_name=f'tts_output.{audio_extension}', mime=audio_mime_type,
					width='stretch' )
		
		# ----- Playback ------
		with tab_playback:
			st.caption( 'Playback generated output, uploaded/recorded audio, or a local file.' )
			if st.session_state.get( 'audio_output_bytes' ):
				st.audio( st.session_state.get( 'audio_output_bytes' ), format=get_audio_mime_type(
					st.session_state.get( 'audio_response_format', 'mp3' ) ) )
			
			playback_path = (st.session_state.get( 'audio_upload_path' ) or st.session_state.get(
				'audio_recorded_path' ) or st.session_state.get( 'audio_output_path' ) or '')
			
			if playback_path:
				try:
					st.audio( playback_path,
						start_time=float( st.session_state.get( 'audio_start_time', 0.0 ) or 0.0 ),
						end_time=float(
							st.session_state.get( 'audio_end_time', 0.0 ) or 0.0 ) if float(
							st.session_state.get( 'audio_end_time', 0.0 ) or 0.0 ) > 0 else None,
						loop=bool( st.session_state.get( 'audio_loop', False ) ),
						autoplay=bool( st.session_state.get( 'audio_autoplay', False ) ) )
				except TypeError:
					st.audio( playback_path )
				except Exception as exc:
					st.warning( f'Could not play audio file: {exc}' )
			
			local_audio = getattr( cfg, 'AUDIO_TEST_FILE', None )
			if local_audio:
				try:
					st.audio( local_audio,
						start_time=float( st.session_state.get( 'audio_start_time', 0.0 ) or 0.0 ),
						end_time=float(
							st.session_state.get( 'audio_end_time', 0.0 ) or 0.0 ) if float(
							st.session_state.get( 'audio_end_time', 0.0 ) or 0.0 ) > 0 else None,
						loop=bool( st.session_state.get( 'audio_loop', False ) ),
						autoplay=bool( st.session_state.get( 'audio_autoplay', False ) ) )
				except TypeError:
					st.audio( local_audio )
				except Exception as exc:
					st.warning( f'Could not play local audio file: {exc}' )
			else:
				st.info( 'No local audio test file is configured.' )
		
		# ------------------------------------------------------------------
		# Expander - Metadata
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
	
	# ------------------------------------------------------------------
	# Document Q&A State Safety
	# ------------------------------------------------------------------
	for key, value in { 'docqna_number': 0, 'docqna_max_calls': 0, 'docqna_max_searches': 0,
		'docqna_max_tokens': 0, 'docqna_top_percent': 0.0, 'docqna_top_k': 0,
		'docqna_frequency_penalty': 0.0, 'docqna_presense_penalty': 0.0,
		'docqna_presence_penalty': 0.0, 'docqna_temperature': 0.0, 'docqna_stream': False,
		'docqna_parallel_tools': False, 'docqna_store': False, 'docqna_background': False,
		'docqna_reasoning': '', 'docqna_media_resolution': '', 'docqna_response_format': '',
		'docqna_tool_choice': '', 'docqna_tools': [ ], 'docqna_modalities': [ ],
		'docqna_include': [ ], 'docqna_messages': [ ], }.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = value
	
	def parse_docqna_list( value: Any ) -> List[ str ]:
		"""Parse Document Q&A list.
		
		Purpose:
		    Converts a comma-delimited string or existing sequence into a normalized list of
		    non-empty values for provider arguments.
		
		Args:
		    value (Any): String or sequence containing provider option values.
		
		Returns:
		    List[str]: Normalized provider argument values.
		"""
		if isinstance( value, str ):
			return [ item.strip( ) for item in value.split( ',' ) if item.strip( ) ]
		
		if isinstance( value, (list, tuple, set) ):
			return [ str( item ).strip( ) for item in value if str( item ).strip( ) ]
		
		return [ ]
	
	def run_document_query( prompt: str ) -> str:
		"""Run Document Q&A query.
		
		Purpose:
		    Builds document-grounded input and forwards every active Document Q&A UI setting to
		    the selected provider text-generation wrapper.
		
		Args:
		    prompt (str): User question submitted for the active document.
		
		Returns:
		    str: Text returned by the selected provider.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			throw_if( 'prompt', prompt )
			chat = get_chat_module( provider_name )
			model = st.session_state.get( 'docqna_model', '' )
			if not model:
				model_options = list( getattr( chat, 'model_options', [ ] ) or [ ] )
				model = model_options[ 0 ] if model_options else None
			
			if not model:
				raise ValueError(
					f'No Document Q&A model is configured for provider "{provider_name}".' )
			
			top_k = int( st.session_state.get( 'docqna_top_k', 0 ) )
			system_instructions = st.session_state.get( 'system_instructions', '' )
			st.session_state[ 'system_instructions' ] = ''
			try:
				user_input = build_document_user_input( prompt, k=top_k or 6 )
			finally:
				st.session_state[ 'system_instructions' ] = system_instructions
			
			if not user_input:
				user_input = prompt.strip( )
			
			domains_value = st.session_state.get( 'docqna_domains_input',
				st.session_state.get( 'docqna_domains', [ ] ) )
			stops_value = st.session_state.get( 'docqna_stops_input',
				st.session_state.get( 'docqna_stops', [ ] ) )
			presence = st.session_state.get( 'docqna_presence_penalty',
				st.session_state.get( 'docqna_presense_penalty', 0.0 ) )
			
			kwargs: Dict[ str, Any ] = { 'model': model, 'prompt': user_input,
				'number': int( st.session_state.get( 'docqna_number', 0 ) ),
				'temperature': float( st.session_state.get( 'docqna_temperature', 0.0 ) ),
				'top_p': float( st.session_state.get( 'docqna_top_percent', 0.0 ) ), 'top_k':
					top_k,
				'frequency': float( st.session_state.get( 'docqna_frequency_penalty', 0.0 ) ),
				'presence': float( presence ),
				'max_tokens': int( st.session_state.get( 'docqna_max_tokens', 0 ) ),
				'store': bool( st.session_state.get( 'docqna_store', False ) ),
				'stream': bool( st.session_state.get( 'docqna_stream', False ) ),
				'background': bool( st.session_state.get( 'docqna_background', False ) ),
				'instruct': st.session_state.get( 'docqna_system_instructions', '' ),
				'tools': st.session_state.get( 'docqna_tools', [ ] ),
				'include': st.session_state.get( 'docqna_include', [ ] ),
				'tool_choice': st.session_state.get( 'docqna_tool_choice' ) or None,
				'reasoning': st.session_state.get( 'docqna_reasoning' ) or None,
				'response_format': st.session_state.get( 'docqna_response_format' ) or None,
				'modalities': st.session_state.get( 'docqna_modalities', [ ] ),
				'media_resolution': st.session_state.get( 'docqna_media_resolution' ) or None,
				'allowed_domains': parse_docqna_list( domains_value ),
				'stops': parse_docqna_list( stops_value ),
				'is_parallel': bool( st.session_state.get( 'docqna_parallel_tools', False ) ),
				'max_tools': int( st.session_state.get( 'docqna_max_calls', 0 ) ),
				'max_searches': int( st.session_state.get( 'docqna_max_searches', 0 ) ),
				'context': st.session_state.get( 'docqna_context', [ ] ),
				'content': st.session_state.get( 'docqna_content', '' ) or None, }
			
			answer = chat.generate_text( **kwargs )
			if isinstance( answer, str ):
				return answer
			
			output_text = getattr( chat, 'output_text', None )
			if isinstance( output_text, str ) and output_text.strip( ):
				return output_text.strip( )
			
			output_text = getattr( answer, 'output_text', None )
			if isinstance( output_text, str ) and output_text.strip( ):
				return output_text.strip( )
			
			return str( answer or '' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Document Q&A'
			ex.method = 'run_document_query( prompt: str ) -> str'
			Logger( ).write( ex )
			raise ex
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'docqna_system_instructions' ] = ''
		st.session_state[ 'clear_docqa_instructions' ] = False
		st.session_state[ 'clear_instructions' ] = False
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '📚 Document Q & A', help=cfg.DOCUMENT_Q_AND_A )
		st.divider( )
		# ------------------------------------------------------------------
		# Expander — DocQ&A LLM Configuration (Grok)
		# ------------------------------------------------------------------
		if provider_name == 'Grok':
			with st.expander( label='Mind Controls', icon='🧠', expanded=False,
					width='stretch' ):
				# ------------------------------------------------------------------
				# Expander - DocQ&A Model (Grok)
				# ------------------------------------------------------------------
				with st.expander( label='LLM Settings', expanded=False, width='stretch' ):
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
						set_docqna_include = st.multiselect( label='Include:',
							options=include_options, key='docqna_include', help=cfg.INCLUDE,
							placeholder='Options' )
						
						docqna_include = [ d.strip( ) for d in set_docqna_include if d.strip( ) ]
						docqna_include = st.session_state[ 'docqna_include' ]
					
					# ------------- Reasoning Options ----------
					with llm_c3:
						docqna = get_chat_module( provider_name )
						
						def get_docqna_options( instance: Any, attr_name: str,
							fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
							"""Get docqna options.
							
							Purpose:
							    Returns normalized information for the application component. The
							    method provides a stable view
							    of provider capabilities, stored state, or response metadata so UI
							    controls and downstream logic
							    can consume it consistently.
							
							Args:
							    instance (Any): Instance value used by the operation.
							    attr_name (str): Attr name value used by the operation.
							    fallback (Optional[List[Any]]): Fallback value used by the
									operation.
							
							Returns:
							    List[Any]: Return value produced by the operation."""
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
						
						model_options = get_docqna_options( docqna, 'model_options', [ ] )
						reasoning_options = get_docqna_options( docqna, 'reasoning_options', [ ] )
						tool_options = get_docqna_options( docqna, 'tool_options', [ ] )
						include_options = get_docqna_options( docqna, 'include_options', [ ] )
						choice_options = get_docqna_options( docqna, 'choice_options',
							[ 'auto', 'required', 'none' ] )
						format_options = get_docqna_options( docqna, 'format_options', [ 'text' ] )
						if not reasoning_options:
							reasoning_options = [ 'none' ]
						
						if not format_options:
							format_options = [ 'text' ]
						if not reasoning_options:
							reasoning_options = [ 'none' ]
						set_docqna_reasoning = st.selectbox( label='Reasoning',
							options=reasoning_options, key='docqna_reasoning', index=0,
							help='Optional reasoning level when supported by the active '
							     'provider.' )
						
						docqna_reasoning = st.session_state[ 'docqna_reasoning' ]
					
					# ------------- Choice Options ----------
					with llm_c4:
						choice_options = list( docqna.choice_options )
						set_docqna_choice = st.multiselect( label='Tool Choice:',
							options=choice_options, key='docqna_tool_choice', help=cfg.CHOICE,
							placeholder='Options' )
						
						docqna_tool_choice = st.session_state[ 'docqna_tool_choice' ]
					
					# ------------- Reset Settings ----------
					if st.button( label='Reset', key='docqna_model_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_model', 'docqna_include', 'docqna_reasoning',
							'docqna_tool_choice' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Inference Settings (Grok)
				# ------------------------------------------------------------------
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
						set_docqna_temperature = st.slider( label='Temperature', min_value=0.0,
							max_value=1.0,
							value=float( st.session_state.get( 'docqna_temperature', 0.0 ) ),
							step=0.01, help=cfg.TEMPERATURE, key='docqna_temperature' )
						
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
						set_docqna_tokens = st.slider( label='Max Tokens', min_value=0,
							max_value=100000, step=500,
							value=int( st.session_state.get( 'docqna_max_tokens', 0 ) ),
							help=cfg.MAX_OUTPUT_TOKENS, key='docqna_max_tokens' )
						
						docqna_tokens = st.session_state[ 'docqna_max_tokens' ]
					
					# ------------- Reset Setting ----------
					if st.button( label='Reset', key='docqna_inference_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_top_percent', 'docqna_max_tokens',
							'docqna_temperature', 'docqna_number', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Tool Settings (Grok)
				# ------------------------------------------------------------------
				with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
					tool_c1, tool_c2, tool_c3, tool_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
					# ------------- Asynchronous  ------------------
					with tool_c1:
						set_docqna_parallel = st.toggle( label='Asynchronous Tool Calls',
							key='docqna_parallel_tools', help=cfg.PARALLEL_TOOL_CALLS )
						
						docqna_parallel_tools = st.session_state[ 'docqna_parallel_tools' ]
					
					# ------------- Max Tool Calls ------------------
					with tool_c2:
						set_docqna_calls = st.slider( label='Max Tool Calls', min_value=0,
							max_value=4, value=int( st.session_state.get( 'docqna_max_calls',
								0 ) ),
							step=1, help=cfg.MAX_TOOL_CALLS, key='docqna_max_calls' )
						
						docqna_max_calls = st.session_state[ 'docqna_max_calls' ]
					
					# -------------  Max Web Searches ------------------
					with tool_c3:
						set_max_results = st.slider( label='Max Websearch Results',
							key='docqna_max_searches',
							value=int( st.session_state.get( 'docqna_max_searches', 0 ) ),
							min_value=0, max_value=30, step=1,
							help='Optional. Upper limit on the number web search results' )
						
						docqna_max_searches = st.session_state[ 'docqna_max_searches' ]
					
					# ------------- Tools ------------------
					with tool_c4:
						tool_options = list( docqna.tool_options )
						set_docqna_tools = st.multiselect( label='Tools:', options=tool_options,
							key='docqna_tools', help=cfg.TOOLS, placeholder='Options' )
						
						docqna_tools = [ d.strip( ) for d in set_docqna_tools if d.strip( ) ]
						docqna_tools = st.session_state[ 'docqna_tools' ]
					
					# ------------- Reset Button -------------
					if st.button( label='Reset', key='docqna_tools_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_parallel_tools', 'docqna_max_searches',
							'docqna_tools',
							'docqna_max_calls' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Response Settings (Grok)
				# ------------------------------------------------------------------
				with st.expander( label='Response Settings', expanded=False, width='stretch' ):
					resp_c1, resp_c2, resp_c3, resp_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
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
						set_docqna_background = st.toggle( label='Background',
							key='docqna_background', help=cfg.BACKGROUND_MODE )
						
						docqna_background = st.session_state[ 'docqna_background' ]
					
					# ------------- Domains  ------------------
					with resp_c4:
						set_docqna_domains = st.text_input( label='Allowed Websites',
							key='docqna_domains_input',
							value=','.join( st.session_state.get( 'docqna_domains', [ ] ) ),
							help=cfg.ALLOWED_DOMAINS, width='stretch',
							placeholder='Enter Web Domains' )
						
						docqna_domains = [ d.strip( ) for d in set_docqna_domains.split( ',' ) if
							d.strip( ) ]
						
						st.session_state[ 'docqna_domains' ] = docqna_domains
					
					# ------------- Reset Settings  ------------------
					if st.button( label='Reset', key='docqna_response_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_stream', 'docqna_store', 'docqna_background',
							'docqna_domains_input', 'docqna_domains', 'docqna_stops_input', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — DocQ&A LLM Configuration (Gemini)
		# ------------------------------------------------------------------
		elif provider_name == 'Gemini':
			with st.expander( label='Mind Controls', icon='🧠', expanded=False,
					width='stretch' ):
				# ------------------------------------------------------------------
				# Expander — Model Settings (Gemini)
				# ------------------------------------------------------------------
				with st.expander( label='LLM Settings', expanded=False, width='stretch' ):
					llm_c1, llm_c2, llm_c3, llm_c4, llm_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Model ------------
					with llm_c1:
						model_options = list( docqna.model_options )
						set_docqna_model = st.selectbox( label='Select Model',
							options=model_options, key='docqna_model', placeholder='Options',
							index=None, help='REQUIRED. Text Generation model used by the AI', )
						
						docqna_model = st.session_state[ 'docqna_model' ]
					
					# ---------- Include ------------
					with llm_c2:
						include_options = list( docqna.include_options )
						set_docqna_include = st.multiselect( label='Include',
							options=include_options, key='docqna_include', help=cfg.INCLUDE,
							placeholder='Options' )
						
						docqna_include = [ d.strip( ) for d in set_docqna_include if d.strip( ) ]
						docqna_include = st.session_state[ 'docqna_include' ]
					
					# ---------- Allowed Domains ------------
					with llm_c3:
						set_docqna_domains = st.text_input( label='Allowed Domains',
							key='docqna_domains_input',
							value=','.join( st.session_state.get( 'docqna_domains', [ ] ) ),
							help=cfg.ALLOWED_DOMAINS, width='stretch', placeholder='Enter '
							                                                       'Domains' )
						
						docqna_domains = [ d.strip( ) for d in set_docqna_domains.split( ',' ) if
							d.strip( ) ]
						
						st.session_state[ 'docqna_domains' ] = docqna_domains
					
					# ---------- Reasoning/Thinking Level ------------
					with llm_c4:
						reasoning_options = list( docqna.reasoning_options )
						set_docqna_reasoning = st.selectbox( label='Thinking Level:',
							options=reasoning_options, key='docqna_reasoning', help=cfg.REASONING,
							index=None, placeholder='Options' )
						
						docqna_reasoning = st.session_state[ 'docqna_reasoning' ]
					
					# ---------- Media Resolution ------------
					with llm_c5:
						media_options = list( docqna.media_options )
						set_media_resolution = st.selectbox( label='Media Resolution',
							options=media_options, key='docqna_media_resolution',
							help=cfg.REASONING, index=None, placeholder='Options' )
						
						media_resolution = st.session_state[ 'docqna_media_resolution' ]
					
					# ---------- Reset Button ------------
					if st.button( label='Reset', key='docqna_model_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_model', 'docqna_include', 'docqna_domains',
							'docqna_reasoning', 'docqna_media_resolution' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Inference Settings (Gemini)
				# ------------------------------------------------------------------
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
						set_docqna_freq = st.slider( label='Frequency Penalty', min_value=-2.0,
							max_value=2.0,
							value=float( st.session_state.get( 'docqna_frequency_penalty', 0.0 ) ),
							step=0.01, help=cfg.FREQUENCY_PENALTY, key='docqna_frequency_penalty' )
						
						docqna_fequency = st.session_state[ 'docqna_frequency_penalty' ]
					
					# ---------- Presense ------------
					with prm_c3:
						set_docqna_presense = st.slider( label='Presense Penalty', min_value=-2.0,
							max_value=2.0,
							value=float( st.session_state.get( 'docqna_presense_penalty', 0.0 ) ),
							step=0.01, help=cfg.PRESENCE_PENALTY, key='docqna_presense_penalty' )
						
						docqna_presense = st.session_state[ 'docqna_presense_penalty' ]
					
					# ---------- Temperature ------------
					with prm_c4:
						set_docqna_temperature = st.slider( label='Temperature', min_value=0.0,
							max_value=1.0,
							value=float( st.session_state.get( 'docqna_temperature', 0.0 ) ),
							step=0.01, help=cfg.TEMPERATURE, key='docqna_temperature' )
						
						docqna_temperature = st.session_state[ 'docqna_temperature' ]
					
					# ---------- Top-K ------------
					with prm_c5:
						set_docqna_topk = st.slider( label='Top K', min_value=0, max_value=20,
							value=int( st.session_state.get( 'docqna_top_k', 0 ) ), step=1,
							help=cfg.TOP_K, key='docqna_top_k' )
						
						docqna_top_k = st.session_state[ 'docqna_top_k' ]
					
					# ---------- Reset Button ------------
					if st.button( label='Reset', key='docqna_inference_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_top_percent', 'docqna_frequency_penalty',
							'docqna_presense_penalty', 'docqna_temperature', 'docqna_top_k', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Tool Settings (Gemini)
				# ------------------------------------------------------------------
				with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
					tool_c1, tool_c2, tool_c3, tool_c4, tool_c5 = st.columns(
						[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					# ---------- Number/Candidates ------------
					with tool_c1:
						set_docqna_number = st.slider( label='Candidates', min_value=0,
							max_value=50, value=int( st.session_state.get( 'docqna_number', 0 ) ),
							step=1,
							help='Optional. Upper limit on the responses returned by the model',
							key='docqna_number' )
						
						docqna_number = st.session_state[ 'docqna_number' ]
					
					# ---------- Max Calls ------------
					with tool_c2:
						set_docqna_calls = st.slider( label='Max Tool Calls', min_value=0,
							max_value=10,
							value=int( st.session_state.get( 'docqna_max_calls', 0 ) ), step=1,
							help=cfg.MAX_TOOL_CALLS, key='docqna_max_calls' )
						
						docqna_max_calls = st.session_state[ 'docqna_max_calls' ]
					
					# ---------- Choice/Calling Mode ------------
					with tool_c3:
						choice_options = list( docqna.choice_options )
						set_docqna_choice = st.selectbox( label='Calling Mode',
							options=choice_options, key='docqna_tool_choice', help=cfg.CHOICE,
							index=None, placeholder='Options' )
						
						docqna_tool_choice = st.session_state[ 'docqna_tool_choice' ]
					
					# ---------- Tools ------------
					with tool_c4:
						tool_options = list( docqna.tool_options )
						set_docqna_tools = st.multiselect( label='Available Tools',
							options=tool_options, key='docqna_tools', help=cfg.TOOLS,
							placeholder='Options' )
						
						docqna_tools = [ d.strip( ) for d in set_docqna_tools if d.strip( ) ]
						docqna_tools = st.session_state[ 'docqna_tools' ]
					
					# ---------- Modalities ------------
					with tool_c5:
						modality_options = list( docqna.modality_options )
						set_docqna_modalities = st.multiselect( label='Response Modalities',
							options=modality_options, key='docqna_modalities',
							help='Optional. Modality of the response', placeholder='Options' )
						
						docqna_modalities = [ d.strip( ) for d in set_docqna_modalities if
							d.strip( ) ]
						
						docqna_modalities = st.session_state[ 'docqna_modalities' ]
					
					# ---------- Reset Button ------------
					if st.button( label='Reset', key='docqna_tools_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_parallel_tools', 'docqna_tool_choice',
							'docqna_number',
							'docqna_tools', 'docqna_max_calls', 'docqna_modalities' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Response Settings (Gemini)
				# ------------------------------------------------------------------
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
						set_docqna_background = st.toggle( label='Background',
							key='docqna_background', help=cfg.BACKGROUND_MODE )
						
						docqna_background = st.session_state[ 'docqna_background' ]
					
					# ---------- Stops ------------
					with resp_c4:
						set_docqna_stops = st.text_input( label='Stop Sequences',
							key='docqna_stops', help=cfg.STOP_SEQUENCE, width='stretch',
							placeholder='Enter Stops' )
						
						docqna_stops = [ d.strip( ) for d in set_docqna_stops.split( ',' ) if
							d.strip( ) ]
					
					# ---------- Max Tokens ------------
					with resp_c5:
						set_docqna_tokens = st.slider( label='Max Tokens', min_value=0,
							max_value=100000,
							value=int( st.session_state.get( 'docqna_max_tokens', 0 ) ), step=500,
							help=cfg.MAX_OUTPUT_TOKENS, key='docqna_max_tokens' )
						
						docqna_tokens = st.session_state[ 'docqna_max_tokens' ]
					
					# ---------- Reset Button ------------
					if st.button( label='Reset', key='docqna_response_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_stream', 'docqna_store', 'docqna_background',
							'docqna_stops', 'docqna_max_tokens' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						# If using separated UI key for stops
						if 'docqna_stops_input' in st.session_state:
							del st.session_state[ 'docqna_stops_input' ]
						
						st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — Mind Controls (GPT)
		# ------------------------------------------------------------------
		elif provider_name == 'GPT':
			with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
				# ------------------------------------------------------------------
				# Expander — LLM Settings (GPT)
				# ------------------------------------------------------------------
				with st.expander( label='LLM Settings', expanded=False, width='stretch' ):
					llm_c1, llm_c2, llm_c3, llm_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
					# ---------- Model ------------
					with llm_c1:
						model_options = list( docqna.model_options )
						set_docqna_model = st.selectbox( label='Select Model',
							options=model_options, key='docqna_model', placeholder='Options',
							index=None, help='REQUIRED. Text Generation model used by the AI', )
						
						docqna_model = st.session_state[ 'docqna_model' ]
					
					# ---------- Include ------------
					with llm_c2:
						include_options = list( docqna.include_options )
						set_docqna_include = st.multiselect( label='Include:',
							options=include_options, key='docqna_include', help=cfg.INCLUDE,
							placeholder='Options' )
						
						docqna_include = [ d.strip( ) for d in set_docqna_include if d.strip( ) ]
						docqna_include = st.session_state[ 'docqna_include' ]
					
					# ---------- Allowed Domains ------------
					with llm_c3:
						set_docqna_domains = st.text_input( label='Allowed Domains',
							key='docqna_domains_input',
							value=','.join( st.session_state.get( 'docqna_domains', [ ] ) ),
							help=cfg.ALLOWED_DOMAINS, width='stretch', placeholder='Enter '
							                                                       'Domains' )
						
						docqna_domains = [ d.strip( ) for d in set_docqna_domains.split( ',' ) if
							d.strip( ) ]
						
						st.session_state[ 'docqna_domains' ] = docqna_domains
					
					# ---------- Reasoning ------------
					with llm_c4:
						reasoning_options = list( docqna.reasoning_options )
						set_docqna_reasoning = st.selectbox( label='Reasoning Effort:',
							options=reasoning_options, key='docqna_reasoning', help=cfg.REASONING,
							index=None, placeholder='Options' )
						
						docqna_reasoning = st.session_state[ 'docqna_reasoning' ]
					
					# ---------- Reset Button------------
					if st.button( label='Reset', key='docqna_model_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_model', 'docqna_include', 'docqna_domains',
							'docqna_reasoning' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Inference Settings (GPT)
				# ------------------------------------------------------------------
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
						set_docqna_freq = st.slider( label='Frequency Penalty', min_value=-2.0,
							max_value=2.0,
							value=float( st.session_state.get( 'docqna_frequency_penalty', 0.0 ) ),
							step=0.01, help=cfg.FREQUENCY_PENALTY, key='docqna_frequency_penalty' )
						
						docqna_fequency = st.session_state[ 'docqna_frequency_penalty' ]
					
					# ---------- Presense ------------
					with prm_c3:
						set_docqna_presense = st.slider( label='Presence Penalty', min_value=-2.0,
							max_value=2.0,
							value=float( st.session_state.get( 'docqna_presense_penalty', 0.0 ) ),
							step=0.01, help=cfg.PRESENCE_PENALTY, key='docqna_presense_penalty' )
						
						docqna_presense = st.session_state[ 'docqna_presense_penalty' ]
					
					# ---------- Temperature ------------
					with prm_c4:
						set_docqna_temperature = st.slider( label='Temperature', min_value=0.0,
							max_value=1.0,
							value=float( st.session_state.get( 'docqna_temperature', 0.0 ) ),
							step=0.01, help=cfg.TEMPERATURE, key='docqna_temperature' )
						
						docqna_temperature = st.session_state[ 'docqna_temperature' ]
					
					# ---------- Number ------------
					with prm_c5:
						set_docqna_number = st.slider( label='Number', min_value=0, max_value=10,
							value=int( st.session_state.get( 'docqna_number', 0 ) ), step=1,
							help='Optional. Upper limit on the responses returned by the model',
							key='docqna_number' )
						
						docqna_number = st.session_state[ 'docqna_number' ]
					
					# ---------- Reset Button ------------
					if st.button( label='Reset', key='docqna_inference_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_top_percent', 'docqna_frequency_penalty',
							'docqna_presense_penalty', 'docqna_temperature', 'docqna_number', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Tool Settings (GPT)
				# ------------------------------------------------------------------
				with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
					tool_c1, tool_c2, tool_c3, tool_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
					# ---------- Allow Parallel ------------
					with tool_c1:
						set_docqna_parallel = st.toggle( label='Asychronous Calls',
							key='docqna_parallel_tools', help=cfg.PARALLEL_TOOL_CALLS )
						
						docqna_parallel_tools = st.session_state[ 'docqna_parallel_tools' ]
					
					# ---------- Max Calls ------------
					with tool_c2:
						set_docqna_calls = st.slider( label='Max Tool Calls', min_value=0,
							max_value=5, value=int( st.session_state.get( 'docqna_max_calls',
								0 ) ),
							step=1, help=cfg.MAX_TOOL_CALLS, key='docqna_max_calls' )
						
						docqna_max_calls = st.session_state[ 'docqna_max_calls' ]
					
					# ---------- Choice ------------
					with tool_c3:
						choice_options = list( docqna.choice_options )
						set_docqna_choice = st.selectbox( label='Tool Choice:',
							options=choice_options, key='docqna_tool_choice', help=cfg.CHOICE,
							index=None, placeholder='Options' )
						
						docqna_tool_choice = st.session_state[ 'docqna_tool_choice' ]
					
					# ---------- Tools ------------
					with tool_c4:
						tool_options = list( docqna.tool_options )
						set_docqna_tools = st.multiselect( label='Available Tools',
							options=tool_options, key='docqna_tools', help=cfg.TOOLS,
							placeholder='Options' )
						
						docqna_tools = [ d.strip( ) for d in set_docqna_tools if d.strip( ) ]
						docqna_tools = st.session_state[ 'docqna_tools' ]
					
					# ---------- Reset Button ------------
					if st.button( label='Reset', key='docqna_tools_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_parallel_tools', 'docqna_tool_choice', 'docqna_tools',
							'docqna_max_calls' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
				
				# ------------------------------------------------------------------
				# Expander — Response Settings (GPT)
				# ------------------------------------------------------------------
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
						set_docqna_background = st.toggle( label='Background',
							key='docqna_background', help=cfg.BACKGROUND_MODE )
						
						docqna_background = st.session_state[ 'docqna_background' ]
					
					# ---------- Stops ------------
					with resp_c4:
						set_docqna_stops = st.text_input( label='Stop Sequences',
							key='docqna_stops', help=cfg.STOP_SEQUENCE, width='stretch',
							placeholder='Enter Stops' )
						
						docqna_stops = [ d.strip( ) for d in set_docqna_stops.split( ',' ) if
							d.strip( ) ]
					
					# ---------- Max Tokens ------------
					with resp_c5:
						set_docqna_tokens = st.slider( label='Max Output Tokens', min_value=0,
							max_value=100000,
							value=int( st.session_state.get( 'docqna_max_tokens', 0 ) ), step=500,
							help=cfg.MAX_OUTPUT_TOKENS, key='docqna_max_tokens' )
						
						docqna_tokens = st.session_state[ 'docqna_max_tokens' ]
					
					# ---------- Reset Button ------------
					if st.button( label='Reset', key='docqna_response_reset', width='stretch',
							icon='🔄' ):
						for key in [ 'docqna_stream', 'docqna_store', 'docqna_background',
							'docqna_stops', 'docqna_max_tokens' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						# If using separated UI key for stops
						if 'docqna_stops_input' in st.session_state:
							del st.session_state[ 'docqna_stops_input' ]
						
						st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander —  System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			# ------ Document Q&A Prompt Categories ------
			docqna_prompt_categories = fetch_prompt_categories( 'Document Q&A' )
			current_docqna_category = st.session_state.get( 'docqna_prompt_category' )
			if current_docqna_category not in docqna_prompt_categories:
				st.session_state[ 'docqna_prompt_category' ] = None
			
			selected_docqna_category = st.session_state.get( 'docqna_prompt_category' )
			docqna_prompt_options = fetch_prompt_options(
				selected_docqna_category ) if selected_docqna_category else [ ]
			
			docqna_prompt_ids = [ int( option[ 'ID' ] ) for option in docqna_prompt_options ]
			if st.session_state.get( 'docqna_prompt_id' ) not in docqna_prompt_ids:
				st.session_state[ 'docqna_prompt_id' ] = None
			
			# ----- Instruction Text ------
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='docqna_system_instructions', )
			
			# ------ Template Selection ------
			with in_right:
				st.selectbox( label='Category', options=docqna_prompt_categories, index=None,
					key='docqna_prompt_category', placeholder='Select Category',
					help=('Limits prompt templates to categories associated with '
					      'Document Q&A workflows.'), on_change=reset_prompt_template_selection,
					args=('docqna_prompt_id',), )
				
				st.selectbox( label='Select Template', options=docqna_prompt_ids, index=None,
					key='docqna_prompt_id', placeholder='Select Template',
					disabled=not docqna_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						docqna_prompt_options, ),
					help=('Loads the selected prompt into the Document Q&A '
					      'system-instruction field.'), on_change=load_prompt_template,
					args=('docqna_prompt_id', 'docqna_system_instructions',), )
			
			def clear_docqna_instructions( ) -> None:
				"""Clear Document Q&A instructions.
				
				Purpose:
				    Clears the active Document Q&A system-instruction text and selected prompt
				    template without changing the selected prompt category or document state.
				
				Returns:
				    None: This function performs its work through side effects and does not return
				    a value.
				"""
				st.session_state[ 'docqna_system_instructions' ] = ''
				st.session_state[ 'docqna_prompt_id' ] = None
			
			# ----- Clear Button -----
			st.button( label='Clear Instructions', width='stretch',
				on_click=clear_docqna_instructions, icon='🧹' )
		
		
		# ------------------------------------------------------------------
		# Expander —  Document Uploader
		# ------------------------------------------------------------------
		with st.expander( label='Documnet Loader', width='stretch', expanded=False, icon='📤' ):
			doc_left, doc_right = st.columns( [ 0.2, 0.8 ], border=True )
			# ----- Document Loader -----
			with doc_left:
				docqna_uploaded = st.file_uploader( 'Upload', type=[ 'pdf', 'txt', 'md', 'docx' ],
					accept_multiple_files=False, label_visibility='visible' )
				
				if docqna_uploaded is not None:
					st.session_state.docqna_active_docs = [ docqna_uploaded.name ]
					st.session_state.doc_bytes = { docqna_uploaded.name: docqna_uploaded.getvalue( ) }
					st.success( f'{docqna_uploaded.name} has been loaded!' )
				else:
					st.info( 'Load a document.' )
				
					
				# ---------- Unload Button ------------
				unload = st.button( label='Unload Document', width='stretch' )
				if unload:
					docqna_uploaded = None
					st.session_state.docqna_active_docs = None
			
			# ----- Document Viewer ------
			with doc_right:
				if st.session_state.get( 'docqna_active_docs' ):
					name = st.session_state.docqna_active_docs[ 0 ]
					file_bytes = st.session_state.doc_bytes.get( name )
					if file_bytes:
						suffix = Path( name ).suffix.lower( )
						if suffix == '.pdf':
							try:
								encoded_pdf = base64.b64encode( file_bytes ).decode( 'utf-8' )
								st.markdown( f"""
									<iframe
										src="data:application/pdf;base64,{encoded_pdf}"
										width="100%"
										height="420"
										type="application/pdf">
									</iframe>
									""", unsafe_allow_html=True )
							except Exception as exc:
								st.warning( f'Could not render PDF preview: {exc}' )
								st.download_button( label='Download Document', data=file_bytes,
									file_name=name, mime='application/pdf', width='stretch' )
						
						elif suffix in [ '.txt', '.md' ]:
							try:
								preview_text = file_bytes.decode( 'utf-8', errors='ignore' )
								st.text_area( label='Document Preview', value=preview_text[ :20000 ],
									height=420, width='stretch', disabled=True )
							except Exception as exc:
								st.warning( f'Could not render text preview: {exc}' )
								st.download_button( label='Download Document', data=file_bytes,
									file_name=name, mime='text/plain', width='stretch' )
						
						else:
							st.info( 'Preview is not available for this document type.' )
							st.download_button( label='Download Document', data=file_bytes,
								file_name=name, mime='application/octet-stream', width='stretch' )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		for msg in st.session_state.docqna_messages:
			with st.chat_message( msg[ 'role' ] ):
				st.markdown( msg[ 'content' ] )
		
		if prompt := st.chat_input( 'Ask a question about the document' ):
			st.session_state.docqna_messages.append( { 'role': 'user', 'content': prompt } )
			response = run_document_query( prompt )
			st.session_state.docqna_messages.append( { 'role': 'assistant', 'content': response } )
			st.rerun( )
			
		# ------------------------------------------------------------------
		# Clear Messages
		# ------------------------------------------------------------------
		if st.button( label='Clear Messages', key='docqna_clear_messages',
				icon='🧹', width='content' ):
			st.session_state[ 'docqna_messages' ] = [ ]
			st.session_state[ 'docqna_history' ] = [ ]
			st.session_state[ 'docqna_answer' ] = ''
			st.session_state[ 'docqna_context' ] = ''
			st.session_state[ 'docqna_sources' ] = [ ]
			st.session_state[ 'last_answer' ] = ''
			st.session_state[ 'last_sources' ] = [ ]
			
			st.rerun( )

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == 'Embeddings':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	
	if not provider_has_class( 'Embeddings', provider_name ):
		st.error( f'{provider_name} does not provide an Embeddings wrapper.' )
		st.stop( )
	
	embedding = get_embeddings_module( provider_name )
	
	# ----- Embeddings Utilities ------
	def get_embedding_help( name: str, fallback: str = '' ) -> str:
		"""Get embedding help.
		
		Purpose:
		    Returns normalized help text for an Embeddings Mode control from the current
		    application configuration.
		
		Args:
		    name (str): Configuration attribute containing the help text.
		    fallback (str): Fallback help text used when the attribute is unavailable.
		
		Returns:
		    str: Configured or fallback help text.
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_embedding_options( instance: Any, attr_name: str,
		fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""Get embedding options.
		
		Purpose:
		    Returns a normalized list of provider options exposed through an Embeddings wrapper
		    property or method.
		
		Args:
		    instance (Any): Provider Embeddings wrapper instance.
		    attr_name (str): Name of the wrapper option property or method.
		    fallback (Optional[List[Any]]): Values used when the wrapper exposes no options.
		
		Returns:
		    List[Any]: Provider-supported option values.
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
		"""Normalize embedding text.
		
		Purpose:
		    Converts source input into normalized text suitable for chunking and provider
		    embedding requests.
		
		Args:
		    value (Any): Source text value.
		
		Returns:
		    str: Normalized source text.
		"""
		if value is None:
			return ''
		
		return str( value ).replace( '\r\n', '\n' ).strip( )
	
	def chunk_embedding_text( text_value: str, chunk_size: int, overlap: int ) -> List[ str ]:
		"""Chunk embedding text.
		
		Purpose:
		    Divides embedding source text into bounded overlapping chunks while preserving the
		    existing application chunking helpers when available.
		
		Args:
		    text_value (str): Source text to divide.
		    chunk_size (int): Maximum words or tokens retained in each chunk.
		    overlap (int): Number of words or tokens shared by adjacent chunks.
		
		Returns:
		    List[str]: Ordered embedding chunks.
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
		"""Normalize embedding vectors.
		
		Purpose:
		    Converts GPT, Gemini, dictionary, response-object, batch, single-vector, and
		    base64-encoded embedding outputs into a consistent collection of floating-point
		    vectors.
		
		Args:
		    vectors (Any): Provider embedding result or response object.
		
		Returns:
		    List[List[float]]: Normalized embedding vectors.
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
		
		if isinstance( vectors, str ):
			try:
				decoded = base64.b64decode( vectors )
				return [ np.frombuffer( decoded, dtype=np.float32, ).astype( float ).tolist( ) ]
			except Exception:
				return [ ]
		
		if isinstance( vectors, list ) and vectors:
			first = vectors[ 0 ]
			
			if isinstance( first, str ):
				rows = [ ]
				for item in vectors:
					rows.extend( normalize_embedding_vectors( item ) )
				
				return rows
			
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
				return [ [ float( value ) for value in getattr( item, 'embedding' ) ] for item in
					vectors if hasattr( item, 'embedding' ) ]
			
			if isinstance( first, list ):
				return [ [ float( value ) for value in row ] for row in vectors if
					isinstance( row, list ) ]
		
		return [ ]
	
	def call_embeddings_create( chunks: List[ str ] ) -> Any:
		"""Call embeddings create.
		
		Purpose:
		    Routes embedding creation to the exact GPT or Gemini wrapper contract using the
		    currently selected provider controls.
		
		Args:
		    chunks (List[str]): Required source-text chunks.
		
		Returns:
		    Any: Provider embedding output.
		
		Raises:
		    Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'chunks', chunks )
			
			input_value = chunks if len( chunks ) != 1 else chunks[ 0 ]
			dimensions = st.session_state.get( 'embedding_dimensions',
				st.session_state.get( 'embeddings_dimensions', 0 ), )
			encoding_format = st.session_state.get( 'embedding_encoding_format',
				st.session_state.get( 'embeddings_encoding_format', '' ), )
			model = st.session_state.get( 'embedding_model' ) or None
			
			throw_if( 'model', model )
			
			if provider_name == 'GPT':
				return embedding.create( text=input_value, model=model,
					format=encoding_format or 'float',
					dimensions=(dimensions if int( dimensions or 0 ) > 0 else None), )
			
			if provider_name == 'Gemini':
				task_type = str( st.session_state.get( 'embedding_task_type', '' ) or '' )
				title = str( st.session_state.get( 'embedding_title', '' ) or '' )
				
				return embedding.create( text=input_value, model=model,
					dimensions=int( dimensions or 0 ), task_type=task_type, title=title, )
			
			raise ValueError( f'{provider_name} does not support embedding creation.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'app'
			exception.cause = 'Embeddings'
			exception.method = ('call_embeddings_create( chunks: List[ str ] ) -> Any')
			Logger( ).write( exception )
			raise exception
	
	def extract_embedding_usage( result: Any ) -> Dict[ str, Any ]:
		"""Extract embedding usage.
		
		Purpose:
		    Extracts provider usage metadata from the active wrapper response or returned
		    embedding result.
		
		Args:
		    result (Any): Provider embedding result.
		
		Returns:
		    Dict[str, Any]: Normalized provider usage metadata.
		"""
		response = getattr( embedding, 'response', None ) or result
		
		if response is None:
			return { }
		
		usage = getattr( response, 'usage', None )
		if isinstance( usage, dict ):
			return usage
		
		if usage is not None:
			if hasattr( usage, 'model_dump' ):
				try:
					dumped_usage = usage.model_dump( )
					if isinstance( dumped_usage, dict ):
						return dumped_usage
				except Exception:
					pass
			
			try:
				return dict( usage )
			except Exception:
				return { 'usage': str( usage ) }
		
		if isinstance( response, dict ):
			response_usage = response.get( 'usage' )
			if isinstance( response_usage, dict ):
				return response_usage
		
		return { }
	
	def build_embedding_metrics( source_text: str, chunks: List[ str ],
		vectors: List[ List[ float ] ], usage: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""Build embedding metrics.
		
		Purpose:
		    Builds source-text, chunk, vector, dimensionality, and usage metrics for the
		    current embedding result.
		
		Args:
		    source_text (str): Complete normalized source text.
		    chunks (List[str]): Source-text chunks submitted to the provider.
		    vectors (List[List[float]]): Normalized embedding vectors.
		    usage (Dict[str, Any]): Provider usage metadata.
		
		Returns:
		    Dict[str, Any]: Embedding metrics.
		"""
		words = source_text.split( )
		total_words = len( words )
		unique_words = len( set( words ) )
		token_total = (count_tokens( source_text ) if 'count_tokens' in globals( ) else
		               total_words)
		dimensions = len( vectors[ 0 ] ) if vectors else 0
		
		return { 'tokens': token_total, 'words': total_words, 'unique_words': unique_words,
			'ttr': (unique_words / total_words if total_words > 0 else 0.0),
			'characters': len( source_text ), 'chunks': len( chunks ), 'vectors': len( vectors ),
			'dimensions': dimensions, 'usage': usage, }
	
	def build_embeddings_dataframe( chunks: List[ str ],
		vectors: List[ List[ float ] ] ) -> pd.DataFrame:
		"""Build embeddings dataframe.
		
		Purpose:
		    Builds a tabular embedding output containing chunk identifiers, source text, and
		    one column for each vector dimension.
		
		Args:
		    chunks (List[str]): Source-text chunks.
		    vectors (List[List[float]]): Normalized embedding vectors.
		
		Returns:
		    pd.DataFrame: Embedding output dataframe.
		"""
		if not vectors:
			return pd.DataFrame( )
		
		df_vectors = pd.DataFrame( vectors,
			columns=[ f'dim_{index}' for index in range( len( vectors[ 0 ] ) ) ], )
		
		df_vectors.insert( 0, 'ChunkIndex', range( 1, len( df_vectors ) + 1 ), )
		
		if chunks:
			df_vectors.insert( 1, 'Text', chunks[ :len( df_vectors ) ], )
		
		return df_vectors
	
	def render_embedding_metrics( metrics: Dict[ str, Any ] ) -> None:
		"""Render embedding metrics.
		
		Purpose:
		    Renders token, chunk, vector, dimensionality, and type-token-ratio metrics for the
		    active embedding result.
		
		Args:
		    metrics (Dict[str, Any]): Embedding metrics.
		
		Returns:
		    None: This function renders Streamlit controls.
		"""
		col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns( 5, border=True, )
		col_m1.metric( 'Tokens', metrics.get( 'tokens', 0 ) )
		col_m2.metric( 'Chunks', metrics.get( 'chunks', 0 ) )
		col_m3.metric( 'Vectors', metrics.get( 'vectors', 0 ) )
		col_m4.metric( 'Dimensions', metrics.get( 'dimensions', 0 ) )
		col_m5.metric( 'TTR', f"{float( metrics.get( 'ttr', 0.0 ) ):.3f}", )
	
	def reset_embeddings_all( ) -> None:
		"""Reset embeddings all.
		
		Purpose:
		    Clears Embeddings Mode source input, provider configuration, generated vectors,
		    metrics, dataframe output, and compatibility aliases.
		
		Returns:
		    None: This function updates Streamlit session state.
		"""
		keys_to_clear = [ 'embedding_input', 'embedding_text', 'embeddings_input_text',
			'embedding_file', 'embedding_file_uploader', 'embedding_model',
			'embedding_encoding_format', 'embedding_encoding_format_display',
			'embeddings_encoding_format', 'embedding_dimensions', 'embeddings_dimensions',
			'embedding_chunk_size', 'embeddings_chunk_size', 'embedding_chunk_overlap',
			'embeddings_overlap_amount', 'embedding_task_type', 'embedding_title',
			'embedding_chunks', 'embeddings_chunks', 'embedding_vectors', 'embeddings',
			'embedding_results', 'embedding_dataframe', 'embeddings_df', 'embedding_metrics',
			'embedding_usage', ]
		
		for key in keys_to_clear:
			if key in st.session_state:
				del st.session_state[ key ]
		
		st.session_state[ 'embedding_input' ] = ''
		st.session_state[ 'embedding_text' ] = ''
		st.session_state[ 'embeddings_input_text' ] = ''
		st.session_state[ 'embedding_file' ] = None
		st.session_state[ 'embedding_model' ] = ''
		st.session_state[ 'embedding_encoding_format' ] = ''
		st.session_state[ 'embeddings_encoding_format' ] = ''
		st.session_state[ 'embedding_dimensions' ] = 0
		st.session_state[ 'embeddings_dimensions' ] = 0
		st.session_state[ 'embedding_chunk_size' ] = 0
		st.session_state[ 'embeddings_chunk_size' ] = 0
		st.session_state[ 'embedding_chunk_overlap' ] = 0
		st.session_state[ 'embeddings_overlap_amount' ] = 0
		st.session_state[ 'embedding_task_type' ] = ''
		st.session_state[ 'embedding_title' ] = ''
		st.session_state[ 'embedding_chunks' ] = [ ]
		st.session_state[ 'embeddings_chunks' ] = [ ]
		st.session_state[ 'embedding_vectors' ] = [ ]
		st.session_state[ 'embeddings' ] = [ ]
		st.session_state[ 'embedding_results' ] = None
		st.session_state[ 'embedding_dataframe' ] = None
		st.session_state[ 'embeddings_df' ] = None
		st.session_state[ 'embedding_metrics' ] = { }
		st.session_state[ 'embedding_usage' ] = { }
	
	def update_embedding_usage( response: Any ) -> None:
		"""Update embedding usage.
		
		Purpose:
		    Updates the shared application token counters when the embedding provider exposes
		    compatible usage metadata.
		
		Args:
		    response (Any): Provider response containing usage metadata.
		
		Returns:
		    None: This function updates Streamlit session state.
		"""
		if response is None:
			return
		
		try:
			update_token_counters( response )
		except Exception:
			pass
	
	if 'embedding_task_type' not in st.session_state:
		st.session_state[ 'embedding_task_type' ] = ''
	
	if 'embedding_title' not in st.session_state:
		st.session_state[ 'embedding_title' ] = ''
	
	# ------------------------------------------------------------------
	# Non-destructive state aliases
	# ------------------------------------------------------------------
	if ('embeddings_input_text' in st.session_state and 'embedding_input' not in st.session_state):
		st.session_state[ 'embedding_input' ] = st.session_state.get( 'embeddings_input_text',
			'', )
	
	if ('embedding_input' in st.session_state and 'embeddings_input_text' not in st.session_state):
		st.session_state[ 'embeddings_input_text' ] = st.session_state.get( 'embedding_input',
			'', )
	
	if (
			'embeddings_dimensions' in st.session_state and 'embedding_dimensions' not in
			st.session_state):
		st.session_state[ 'embedding_dimensions' ] = st.session_state.get( 'embeddings_dimensions',
			0, )
	
	if (
			'embedding_dimensions' in st.session_state and 'embeddings_dimensions' not in
			st.session_state):
		st.session_state[ 'embeddings_dimensions' ] = st.session_state.get( 'embedding_dimensions',
			0, )
	
	if (
			'embeddings_encoding_format' in st.session_state and 'embedding_encoding_format' not
			in st.session_state):
		st.session_state[ 'embedding_encoding_format' ] = st.session_state.get(
			'embeddings_encoding_format', '', )
	
	if (
			'embedding_encoding_format' in st.session_state and 'embeddings_encoding_format' not
			in st.session_state):
		st.session_state[ 'embeddings_encoding_format' ] = st.session_state.get(
			'embedding_encoding_format', '', )
	
	if (
			'embeddings_chunk_size' in st.session_state and 'embedding_chunk_size' not in
			st.session_state):
		st.session_state[ 'embedding_chunk_size' ] = st.session_state.get( 'embeddings_chunk_size',
			0, )
	
	if (
			'embedding_chunk_size' in st.session_state and 'embeddings_chunk_size' not in
			st.session_state):
		st.session_state[ 'embeddings_chunk_size' ] = st.session_state.get( 'embedding_chunk_size',
			0, )
	
	if (
			'embeddings_overlap_amount' in st.session_state and 'embedding_chunk_overlap' not in
			st.session_state):
		st.session_state[ 'embedding_chunk_overlap' ] = st.session_state.get(
			'embeddings_overlap_amount', 0, )
	
	if (
			'embedding_chunk_overlap' in st.session_state and 'embeddings_overlap_amount' not in
			st.session_state):
		st.session_state[ 'embeddings_overlap_amount' ] = st.session_state.get(
			'embedding_chunk_overlap', 0, )
	
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
		st.subheader( '🔢 Embeddings', help=cfg.EMBEDDINGS_API, )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Expander - Embedding Configuration
		# ------------------------------------------------------------------
		with st.expander( label='Configuration', icon='🎚️', expanded=False, width='stretch', ):
			emb_c1, emb_c2, emb_c3, emb_c4, emb_c5 = st.columns( [ 0.20, 0.20, 0.20, 0.20, 0.20 ],
				border=True, gap='xxsmall', )
			
			# --------- Model --------
			with emb_c1:
				model_options = get_embedding_options( embedding, 'model_options', )
				model_options = [ str( item ) for item in model_options if str( item ).strip( ) ]
				
				st.selectbox( label='Embedding Model', options=model_options,
					help=('Required. Embedding model used by the selected '
					      'provider.'), key='embedding_model', index=None, placeholder='Options', )
			
			# --------- Encoding --------
			with emb_c2:
				if provider_name == 'GPT':
					encoding_options = get_embedding_options( embedding, 'encoding_options',
						[ 'float', 'base64' ], )
					encoding_options = [ str( item ) for item in encoding_options if
						str( item ).strip( ) ]
					
					st.selectbox( label='Encoding Format', options=encoding_options,
						key='embedding_encoding_format',
						help=('Optional. Format returned by the embedding '
						      'provider.'), index=None, placeholder='Options', )
				else:
					st.selectbox( label='Encoding Format', options=[ 'Provider Default' ], index=0,
						disabled=True, key='embedding_encoding_format_display',
						help=('Gemini returns floating-point embedding values.'), )
					st.session_state[ 'embedding_encoding_format' ] = ''
				
				st.session_state[ 'embeddings_encoding_format' ] = (
					st.session_state.get( 'embedding_encoding_format', '', ))
			
			# --------- Dimensions --------
			with emb_c3:
				selected_model = str( st.session_state.get( 'embedding_model', '' ) or '' )
				
				if provider_name == 'GPT':
					dimension_support = getattr( embedding, 'model_dimension_support', { }, )
					max_dimensions = getattr( embedding, 'model_max_dimensions', { }, )
					supports_dimensions = bool( dimension_support.get( selected_model, False, ) )
					dimension_max = int( max_dimensions.get( selected_model, 4096, ) or 4096 )
				else:
					supports_dimensions = (bool( selected_model ) and bool(
						embedding.supports_dimensions( selected_model ) ))
					dimension_options = get_embedding_options( embedding, 'dimension_options',
						[ 0 ], )
					dimension_max = max( [ int( item ) for item in dimension_options ] or [ 4096
					] )
				
				current_dimensions = int( st.session_state.get( 'embedding_dimensions', 0, ) or 0 )
				
				if current_dimensions > dimension_max:
					st.session_state[ 'embedding_dimensions' ] = (dimension_max)
				
				st.slider( label='Dimensions', min_value=0, max_value=dimension_max,
					value=int( st.session_state.get( 'embedding_dimensions', 0, ) or 0 ), step=1,
					key='embedding_dimensions', disabled=not supports_dimensions,
					help=('Optional. Embedding output dimensions when '
					      'supported. Zero uses the provider default.'), )
				st.session_state[ 'embeddings_dimensions' ] = (
					st.session_state.get( 'embedding_dimensions', 0, ))
			
			# --------- Chunk Size --------
			with emb_c4:
				st.slider( label='Chunk Size', min_value=0, max_value=8000,
					value=int( st.session_state.get( 'embedding_chunk_size', 0, ) or 0 ), step=50,
					key='embedding_chunk_size',
					help=('Maximum words or tokens per chunk. Zero embeds '
					      'the full input.'), )
				st.session_state[ 'embeddings_chunk_size' ] = (
					st.session_state.get( 'embedding_chunk_size', 0, ))
			
			# --------- Overlap --------
			with emb_c5:
				st.slider( label='Overlap', min_value=0, max_value=2000,
					value=int( st.session_state.get( 'embedding_chunk_overlap', 0, ) or 0 ),
					step=25, key='embedding_chunk_overlap',
					help='Overlap between adjacent chunks.', )
				st.session_state[ 'embeddings_overlap_amount' ] = (
					st.session_state.get( 'embedding_chunk_overlap', 0, ))
			
			# --------- Gemini Task Configuration --------
			if provider_name == 'Gemini':
				task_c1, task_c2 = st.columns( [ 0.50, 0.50 ], border=True, gap='xxsmall', )
				
				with task_c1:
					task_options = get_embedding_options( embedding, 'task_options', [ '' ], )
					st.selectbox( label='Task Type', options=task_options,
						key='embedding_task_type',
						help=('Optional. Gemini embedding task type used '
						      'to optimize the vector.'), )
				
				with task_c2:
					st.text_input( label='Document Title', key='embedding_title', disabled=(
							st.session_state.get( 'embedding_task_type',
								'', ) != 'RETRIEVAL_DOCUMENT'),
						help=('Optional. Used only with the '
						      'RETRIEVAL_DOCUMENT task type.'), )
			
			# ----- Reset Button -----
			if st.button( label='Reset', key='reset_embedding_configuration', icon='🔄',
					width='stretch', ):
				for key in [ 'embedding_model', 'embedding_encoding_format',
					'embeddings_encoding_format', 'embedding_dimensions', 'embeddings_dimensions',
					'embedding_chunk_size', 'embeddings_chunk_size', 'embedding_chunk_overlap',
					'embeddings_overlap_amount', 'embedding_task_type', 'embedding_title',
					'embedding_encoding_format_display', ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander - Embedding Source
		# ------------------------------------------------------------------
		with st.expander( label='Source Text', icon='📝', expanded=True, width='stretch', ):
			source_text = st.text_area( label='Input Text', key='embeddings_input_text',
				height=240,
				width='stretch', placeholder=('Paste text to embed, or upload a text-compatible '
				                              'file below.'), )
			st.session_state[ 'embedding_input' ] = source_text
			st.session_state[ 'embedding_text' ] = source_text
			
			uploaded_embedding_file = st.file_uploader( label='Upload Text File',
				type=[ 'txt', 'md', 'csv', 'json', 'py', 'cs', 'sql', 'xml', 'html', ],
				accept_multiple_files=False, key='embedding_file_uploader', )
			
			if uploaded_embedding_file is not None:
				try:
					file_text = uploaded_embedding_file.getvalue( ).decode( 'utf-8',
						errors='ignore', )
					st.session_state[ 'embeddings_input_text' ] = file_text
					st.session_state[ 'embedding_input' ] = file_text
					st.session_state[ 'embedding_text' ] = file_text
					st.success( f'Loaded {uploaded_embedding_file.name}.' )
				except Exception as exc:
					st.error( f'Could not read uploaded file: {exc}' )
		
		action_c1, action_c2 = st.columns( [ 0.50, 0.50 ] )
		
		# ----- Create Embeddings ------
		with action_c1:
			if st.button( 'Create Embeddings', key='create_embeddings', width='stretch',
					icon='➕', ):
				with st.spinner( 'Creating embeddings…' ):
					try:
						source_text = normalize_embedding_text(
							st.session_state.get( 'embeddings_input_text', '', ) )
						
						if not source_text:
							st.warning( 'Enter text before creating embeddings.' )
						
						elif not st.session_state.get( 'embedding_model' ):
							st.warning( 'Select an embedding model before '
							            'creating embeddings.' )
						
						else:
							chunk_size = int(
								st.session_state.get( 'embedding_chunk_size', 0, ) or 0 )
							overlap = int(
								st.session_state.get( 'embedding_chunk_overlap', 0, ) or 0 )
							
							if overlap >= chunk_size and chunk_size > 0:
								st.warning( 'Overlap must be smaller than the '
								            'chunk size.' )
							else:
								chunks = chunk_embedding_text( source_text, chunk_size, overlap, )
								
								if not chunks:
									st.warning( 'No chunks were created from the '
									            'source text.' )
								else:
									raw_result = call_embeddings_create( chunks )
									vectors = normalize_embedding_vectors( raw_result )
									
									if not vectors:
										raise ValueError( 'The provider returned no '
										                  'embedding vectors.' )
									
									usage = extract_embedding_usage( raw_result )
									df_embeddings = (build_embeddings_dataframe( chunks,
										vectors, ))
									metrics = build_embedding_metrics( source_text, chunks,
										vectors,
										usage, )
									
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
										getattr( embedding, 'response', None, ) or raw_result )
									st.success( 'Embeddings created successfully.' )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Embedding creation failed: {err.info}' )
		
		# ----- Reset Button -----
		with action_c2:
			if st.button( 'Reset All', key='reset_embeddings_all', width='stretch',
					on_click=reset_embeddings_all, icon='🔄', ):
				st.rerun( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
		
		metrics = st.session_state.get( 'embedding_metrics', { }, )
		if isinstance( metrics, dict ) and len( metrics ) > 0:
			render_embedding_metrics( metrics )
		
		df_embeddings = st.session_state.get( 'embedding_dataframe', pd.DataFrame( ), )
		if (isinstance( df_embeddings, pd.DataFrame ) and not df_embeddings.empty):
			st.subheader( 'Embedding Output' )
			st.data_editor( df_embeddings, use_container_width=True, hide_index=True,
				disabled=True,
				key='embedding_dataframe_view', )
		
		chunks = st.session_state.get( 'embedding_chunks', [ ], )
		if isinstance( chunks, list ) and len( chunks ) > 0:
			with st.expander( label='Chunks', icon='🧩', expanded=False, width='stretch', ):
				df_chunks = pd.DataFrame( [ { 'ChunkIndex': index + 1, 'Text': chunk, 'Tokens': (
					count_tokens( chunk ) if 'count_tokens' in globals( ) else len(
						chunk.split( ) )), } for index, chunk in enumerate( chunks ) ] )
				st.data_editor( df_chunks, use_container_width=True, hide_index=True,
					disabled=True,
					key='embedding_chunks_view', )
		
		usage = st.session_state.get( 'embedding_usage', { }, )
		if isinstance( usage, dict ) and len( usage ) > 0:
			with st.expander( label='Embedding Usage', icon='📊', expanded=False,
					width='stretch', ):
				st.json( usage )

# ======================================================================================
# FILES MODE
# ======================================================================================
elif mode == 'Files':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	
	if not provider_has_class( 'Files', provider_name ):
		st.error( f'{provider_name} does not provide a Files wrapper.' )
		st.stop( )
	
	files = get_files_module( provider_name )
	
	# ------------------------------------------------------------------
	# Files Mode State
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
	
	if not isinstance( st.session_state.get( 'files_include' ), list ):
		st.session_state[ 'files_include' ] = [ ]
	
	if not isinstance( st.session_state.get( 'files_tools' ), list ):
		st.session_state[ 'files_tools' ] = [ ]
	
	if 'files_manual_id' not in st.session_state:
		st.session_state[ 'files_manual_id' ] = ''
	
	if 'files_retrieve_id' not in st.session_state:
		st.session_state[ 'files_retrieve_id' ] = ''
	
	if 'files_extract_id' not in st.session_state:
		st.session_state[ 'files_extract_id' ] = ''
	
	if 'files_delete_id' not in st.session_state:
		st.session_state[ 'files_delete_id' ] = ''
	
	if 'files_type' not in st.session_state:
		st.session_state[ 'files_type' ] = ''
	
	if 'files_selected_id' not in st.session_state:
		st.session_state[ 'files_selected_id' ] = ''
	
	if 'files_question' not in st.session_state:
		st.session_state[ 'files_question' ] = ''
	
	if 'files_content' not in st.session_state:
		st.session_state[ 'files_content' ] = None
	
	if 'files_content_text' not in st.session_state:
		st.session_state[ 'files_content_text' ] = ''
	
	if 'files_last_answer' not in st.session_state:
		st.session_state[ 'files_last_answer' ] = ''
	
	if 'files_previous_response_id' not in st.session_state:
		st.session_state[ 'files_previous_response_id' ] = ''
	
	if 'files_conversation_id' not in st.session_state:
		st.session_state[ 'files_conversation_id' ] = ''
	
	if 'files_store' not in st.session_state:
		st.session_state[ 'files_store' ] = False
	
	if 'files_stream' not in st.session_state:
		st.session_state[ 'files_stream' ] = False
	
	if 'files_top_percent' not in st.session_state:
		st.session_state[ 'files_top_percent' ] = 0.0
	
	if 'files_frequency_penalty' not in st.session_state:
		st.session_state[ 'files_frequency_penalty' ] = 0.0
	
	if 'files_presence_penalty' not in st.session_state:
		st.session_state[ 'files_presence_penalty' ] = 0.0
	
	if 'files_download_format' not in st.session_state:
		st.session_state[ 'files_download_format' ] = ''
	
	if 'files_page_number' not in st.session_state:
		st.session_state[ 'files_page_number' ] = 0
	
	if 'files_confirm_delete' not in st.session_state:
		st.session_state[ 'files_confirm_delete' ] = False
	
	if 'files_limit' not in st.session_state:
		st.session_state[ 'files_limit' ] = 100
	
	if 'files_pagination_token' not in st.session_state:
		st.session_state[ 'files_pagination_token' ] = ''
	
	if 'files_expires_after' not in st.session_state:
		st.session_state[ 'files_expires_after' ] = 0
	
	if 'files_max_chars' not in st.session_state:
		st.session_state[ 'files_max_chars' ] = 200000
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'files_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	# ----- Files Mode Utilities -----
	def get_files_help( name: str, fallback: str = '' ) -> str:
		"""Get Files help.
		
		Purpose:
		    Returns configured help text for a Files Mode control.
		
		Args:
		    name (str): Configuration attribute name.
		    fallback (str): Fallback text.
		
		Returns:
		    str: Configured or fallback help text.
		"""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_files_options( instance: Any, attr_name: str,
		fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""Get Files options.
		
		Purpose:
		    Returns provider-supported options exposed by a Files wrapper property or method.
		
		Args:
		    instance (Any): Files wrapper instance.
		    attr_name (str): Option property or method name.
		    fallback (Optional[List[Any]]): Fallback options.
		
		Returns:
		    List[Any]: Provider-supported options.
		"""
		values = getattr( instance, attr_name, None )
		
		if callable( values ):
			try:
				values = values( )
			except Exception:
				values = None
		
		if isinstance( values, tuple ):
			values = list( values )
		
		if isinstance( values, list ):
			return values
		
		return fallback or [ ]
	
	def sanitize_files_selection( key: str, valid_options: List[ Any ], default: Any = '' ) -> \
			None:
		"""Sanitize Files selection.
		
		Purpose:
		    Clears a stored selection when it is not supported by the active provider.
		
		Args:
		    key (str): Session-state key.
		    valid_options (List[Any]): Valid provider options.
		    default (Any): Replacement value.
		
		Returns:
		    None: This function updates session state.
		"""
		current_value = st.session_state.get( key, default )
		
		if current_value in [ None, '' ]:
			return
		
		if current_value not in valid_options:
			st.session_state[ key ] = default
	
	def sanitize_files_multiselect( key: str, valid_options: List[ Any ] ) -> None:
		"""Sanitize Files multiselect.
		
		Purpose:
		    Removes stored multiselect values unsupported by the active provider.
		
		Args:
		    key (str): Session-state key.
		    valid_options (List[Any]): Valid provider options.
		
		Returns:
		    None: This function updates session state.
		"""
		current_values = st.session_state.get( key, [ ] )
		
		if not isinstance( current_values, list ):
			st.session_state[ key ] = [ ]
			return
		
		st.session_state[ key ] = [ value for value in current_values if value in valid_options ]
	
	def normalize_file_id( result: Any ) -> str:
		"""Normalize file identifier.
		
		Purpose:
		    Extracts a stable provider file identifier from a file response.
		
		Args:
		    result (Any): Provider file response.
		
		Returns:
		    str: Provider file identifier or resource name.
		"""
		if result is None:
			return ''
		
		if isinstance( result, dict ):
			return str(
				result.get( 'id' ) or result.get( 'file_id' ) or result.get( 'name' ) or '' )
		
		return str(
			getattr( result, 'id', None ) or getattr( result, 'file_id', None ) or getattr( result,
				'name', None ) or '' )
	
	def normalize_files_list( result: Any ) -> List[ Dict[ str, Any ] ]:
		"""Normalize Files list.
		
		Purpose:
		    Converts provider-specific file collections into rows for the Files Mode table.
		
		Args:
		    result (Any): Provider file-list response.
		
		Returns:
		    List[Dict[str, Any]]: Normalized file records.
		"""
		if result is None:
			return [ ]
		
		items = result
		
		if isinstance( result, dict ):
			items = (result.get( 'data' ) or result.get( 'files' ) or result.get( 'items' ) or [ ])
		
		if hasattr( result, 'data' ):
			items = getattr( result, 'data' )
		
		if hasattr( result, 'files' ):
			items = getattr( result, 'files' )
		
		if not isinstance( items, list ):
			try:
				items = list( items )
			except Exception:
				items = [ items ]
		
		rows: List[ Dict[ str, Any ] ] = [ ]
		
		for item in items:
			if item is None:
				continue
			
			if isinstance( item, dict ):
				file_id = (item.get( 'id' ) or item.get( 'file_id' ) or item.get( 'name' ))
				filename = (
						item.get( 'filename' ) or item.get( 'display_name' ) or item.get( 'name' ))
				purpose = (item.get( 'purpose' ) or item.get( 'mime_type' ) or item.get( 'state' ))
				created = (item.get( 'created_at' ) or item.get( 'create_time' ) or item.get(
					'created' ))
				size = (item.get( 'bytes' ) or item.get( 'size_bytes' ) or item.get( 'size' ))
			else:
				file_id = (
						getattr( item, 'id', None ) or getattr( item, 'file_id', None ) or getattr(
					item, 'name', None ))
				filename = (getattr( item, 'filename', None ) or getattr( item, 'display_name',
					None ) or getattr( item, 'name', None ))
				purpose = (getattr( item, 'purpose', None ) or getattr( item, 'mime_type',
					None ) or getattr( item, 'state', None ))
				created = (getattr( item, 'created_at', None ) or getattr( item, 'create_time',
					None ) or getattr( item, 'created', None ))
				size = (getattr( item, 'bytes', None ) or getattr( item, 'size_bytes',
					None ) or getattr( item, 'size', None ))
			
			rows.append( { 'id': str( file_id or '' ), 'filename': str( filename or '' ),
				'purpose': str( purpose or '' ), 'created': str( created or '' ),
				'size': str( size or '' ), } )
		
		return rows
	
	def normalize_file_metadata( result: Any ) -> Dict[ str, Any ]:
		"""Normalize file metadata.
		
		Purpose:
		    Converts provider file metadata into a dictionary suitable for Streamlit output.
		
		Args:
		    result (Any): Provider file response.
		
		Returns:
		    Dict[str, Any]: Normalized metadata.
		"""
		if result is None:
			return { }
		
		if isinstance( result, dict ):
			return result
		
		if hasattr( result, 'model_dump' ):
			try:
				value = result.model_dump( )
				if isinstance( value, dict ):
					return value
			except Exception:
				pass
		
		if hasattr( files, 'get_file_metadata' ):
			try:
				value = files.get_file_metadata( result )
				if isinstance( value, dict ):
					return value
			except Exception:
				pass
		
		return { 'result': str( result ) }
	
	def save_uploaded_file_for_api( uploaded_file: Any ) -> Optional[ str ]:
		"""Save uploaded file for API.
		
		Purpose:
		    Writes a Streamlit uploaded file to a temporary local path.
		
		Args:
		    uploaded_file (Any): Streamlit uploaded-file object.
		
		Returns:
		    Optional[str]: Temporary file path.
		"""
		if uploaded_file is None:
			return None
		
		suffix = Path( getattr( uploaded_file, 'name', 'upload.bin' ) ).suffix or '.bin'
		
		with tempfile.NamedTemporaryFile( delete=False, suffix=suffix, ) as tmp:
			if hasattr( uploaded_file, 'getbuffer' ):
				tmp.write( uploaded_file.getbuffer( ) )
			elif hasattr( uploaded_file, 'getvalue' ):
				tmp.write( uploaded_file.getvalue( ) )
			elif hasattr( uploaded_file, 'read' ):
				tmp.write( uploaded_file.read( ) )
			else:
				return None
			
			return tmp.name
	
	def normalize_file_content( content: Any ) -> str:
		"""Normalize file content.
		
		Purpose:
		    Converts extracted provider content into displayable text.
		
		Args:
		    content (Any): Provider file content.
		
		Returns:
		    str: Displayable content.
		"""
		if content is None:
			return ''
		
		if isinstance( content, str ):
			return content
		
		if isinstance( content, bytes ):
			try:
				return content.decode( 'utf-8' )
			except UnicodeDecodeError:
				return ''
		
		if isinstance( content, dict ):
			return json.dumps( content, indent=2, default=str )
		
		if hasattr( content, 'text' ):
			text_value = getattr( content, 'text', '' )
			if text_value:
				return str( text_value )
		
		return str( content )
	
	def get_effective_file_id( *keys: str ) -> str:
		"""Get effective file identifier.
		
		Purpose:
		    Returns the first populated provider file identifier from the supplied state keys.
		
		Args:
		    *keys (str): Ordered session-state keys.
		
		Returns:
		    str: Active file identifier.
		"""
		for key in keys:
			value = st.session_state.get( key, '' )
			
			if isinstance( value, str ) and value.strip( ):
				return value.strip( )
		
		return ''
	
	def refresh_files_table( ) -> List[ Dict[ str, Any ] ]:
		"""Refresh Files table.
		
		Purpose:
		    Lists provider files and stores normalized records for display and selection.
		
		Returns:
		    List[Dict[str, Any]]: Normalized provider file records.
		"""
		if provider_name == 'GPT':
			result = files.list( purpose=st.session_state.get( 'files_purpose', '' ), )
		elif provider_name == 'Gemini':
			result = files.list( )
		elif provider_name == 'Grok':
			result = files.list( limit=int( st.session_state.get( 'files_limit', 100 ) or 100 ),
				pagination_token=str(
					st.session_state.get( 'files_pagination_token', '', ) or '' ), )
		else:
			raise ValueError( f'Unsupported Files provider: {provider_name}' )
		
		rows = normalize_files_list( result )
		st.session_state[ 'files_table' ] = rows
		
		if provider_name == 'Grok':
			st.session_state[ 'files_pagination_token' ] = str(
				getattr( files, 'next_token', '' ) or '' )
		
		return rows
	
	def upload_provider_file( uploaded_file: Any, purpose: Optional[ str ] = None ) -> Any:
		"""Upload provider file.
		
		Purpose:
		    Uploads a staged local file through the exact selected-provider Files contract.
		
		Args:
		    uploaded_file (Any): Streamlit uploaded-file object.
		    purpose (Optional[str]): Optional file-purpose value.
		
		Returns:
		    Any: Provider file response.
		"""
		path = save_uploaded_file_for_api( uploaded_file )
		
		if not path:
			raise ValueError( 'Could not create a temporary file for upload.' )
		
		filename = str( getattr( uploaded_file, 'name', '' ) or Path( path ).name )
		mime_type = str( getattr( uploaded_file, 'type', '' ) or '' )
		
		if provider_name == 'GPT':
			return files.upload( path=path, purpose=purpose or 'user_data', )
		
		if provider_name == 'Gemini':
			return files.upload( path=path, display_name=filename, mime_type=mime_type, )
		
		if provider_name == 'Grok':
			return files.upload( file_path=path, file_name=filename,
				purpose=purpose or 'assistants',
				expires_after=int( st.session_state.get( 'files_expires_after', 0, ) or 0 ), )
		
		raise ValueError( f'Unsupported Files provider: {provider_name}' )
	
	def retrieve_provider_file( file_id: str ) -> Any:
		"""Retrieve provider file.
		
		Purpose:
		    Retrieves file metadata through the exact selected-provider Files contract.
		
		Args:
		    file_id (str): Provider file identifier or resource name.
		
		Returns:
		    Any: Provider file metadata.
		"""
		throw_if( 'file_id', file_id )
		
		if provider_name == 'GPT':
			return files.retrieve( id=file_id )
		
		return files.retrieve( file_id=file_id )
	
	def extract_provider_file( file_id: str ) -> Any:
		"""Extract provider file.
		
		Purpose:
		    Retrieves file content through the exact selected-provider Files contract.
		
		Args:
		    file_id (str): Provider file identifier or resource name.
		
		Returns:
		    Any: Extracted content or downloaded bytes.
		"""
		throw_if( 'file_id', file_id )
		
		if provider_name == 'GPT':
			return files.extract( id=file_id )
		
		return files.extract( file_id=file_id )
	
	def delete_provider_file( file_id: str ) -> Any:
		"""Delete provider file.
		
		Purpose:
		    Deletes a file through the exact selected-provider Files contract.
		
		Args:
		    file_id (str): Provider file identifier or resource name.
		
		Returns:
		    Any: Provider deletion response.
		"""
		throw_if( 'file_id', file_id )
		
		if provider_name == 'GPT':
			return files.delete( id=file_id )
		
		return files.delete( file_id=file_id )
	
	def ask_provider_file( file_id: str, prompt: str ) -> str:
		"""Ask provider file.
		
		Purpose:
		    Executes a file-aware question through the exact search contract implemented by the
		    selected provider wrapper.
		
		Args:
		    file_id (str): Provider file identifier or resource name.
		    prompt (str): Question asked about the file.
		
		Returns:
		    str: Provider-generated answer.
		"""
		throw_if( 'file_id', file_id )
		throw_if( 'prompt', prompt )
		
		model = str( st.session_state.get( 'files_model', '' ) or '' )
		throw_if( 'model', model )
		
		if provider_name == 'GPT':
			result = files.search( id=file_id, query=prompt, model=model,
				max_chars=int( st.session_state.get( 'files_max_chars', 200000, ) or 200000 ), )
		elif provider_name == 'Gemini':
			result = files.search( prompt=prompt, file_id=file_id, model=model,
				temperature=float( st.session_state.get( 'files_temperature', 0.0, ) or 0.0 ),
				top_p=float( st.session_state.get( 'files_top_percent', 0.0, ) or 0.0 ), top_k=0,
				frequency=float( st.session_state.get( 'files_frequency_penalty', 0.0, ) or 0.0 ),
				presence=float( st.session_state.get( 'files_presence_penalty', 0.0, ) or 0.0 ),
				max_tokens=int( st.session_state.get( 'files_max_tokens', 0, ) or 0 ),
				instruct=str( st.session_state.get( 'files_system_instructions', '', ) or '' ),
				response_format=str( st.session_state.get( 'files_response_format', '', ) or ''
				), )
		elif provider_name == 'Grok':
			result = files.search( file_id=file_id, query=prompt, model=model,
				instruct=str( st.session_state.get( 'files_system_instructions', '', ) or '' ),
				temperature=float( st.session_state.get( 'files_temperature', 0.0, ) or 0.0 ),
				top_p=float( st.session_state.get( 'files_top_percent', 0.0, ) or 0.0 ),
				frequency=float( st.session_state.get( 'files_frequency_penalty', 0.0, ) or 0.0 ),
				presence=float( st.session_state.get( 'files_presence_penalty', 0.0, ) or 0.0 ),
				max_tokens=int( st.session_state.get( 'files_max_tokens', 0, ) or 0 ),
				store=bool( st.session_state.get( 'files_store', False, ) ),
				stream=bool( st.session_state.get( 'files_stream', False, ) ),
				include=list( st.session_state.get( 'files_include', [ ], ) or [ ] ),
				previous_id=str(
					st.session_state.get( 'files_previous_response_id', '', ) or '' ), )
		else:
			raise ValueError( f'Unsupported Files provider: {provider_name}' )
		
		if isinstance( result, str ):
			return result
		
		output_text = getattr( files, 'output_text', '' )
		
		if isinstance( output_text, str ) and output_text.strip( ):
			return output_text.strip( )
		
		return str( result or '' )
	
	def clear_files_outputs( ) -> None:
		"""Clear Files outputs.
		
		Purpose:
		    Clears loaded file records, metadata, extracted content, and operation results.
		
		Returns:
		    None: This function updates session state.
		"""
		st.session_state[ 'files_table' ] = [ ]
		st.session_state[ 'files_metadata' ] = { }
		st.session_state[ 'files_delete_result' ] = { }
		st.session_state[ 'files_results' ] = None
		st.session_state[ 'files_content' ] = None
		st.session_state[ 'files_content_text' ] = ''
	
	def clear_files_messages( ) -> None:
		"""Clear Files messages.
		
		Purpose:
		    Clears Files Mode messages and the latest generated answer.
		
		Returns:
		    None: This function updates session state.
		"""
		st.session_state[ 'files_messages' ] = [ ]
		st.session_state[ 'files_last_answer' ] = ''
	
	def append_files_message( role: str, content: str ) -> None:
		"""Append Files message.
		
		Purpose:
		    Adds a user or assistant message to Files Mode history.
		
		Args:
		    role (str): Message role.
		    content (str): Message content.
		
		Returns:
		    None: This function updates session state.
		"""
		st.session_state[ 'files_messages' ].append( { 'role': role, 'content': content, } )
	
	def render_files_messages( ) -> None:
		"""Render Files messages.
		
		Purpose:
		    Renders Files Mode conversation history.
		
		Returns:
		    None: This function renders Streamlit output.
		"""
		for message in st.session_state.get( 'files_messages', [ ], ):
			if not isinstance( message, dict ):
				continue
			
			with st.chat_message( message.get( 'role', 'assistant' ), ):
				st.markdown( message.get( 'content', '' ) )
	
	def clear_files_instructions( ) -> None:
		"""Clear Files instructions.
		
		Purpose:
		    Clears Files Mode system instructions and its selected prompt template.
		
		Returns:
		    None: This function updates session state.
		"""
		st.session_state[ 'files_system_instructions' ] = ''
		st.session_state[ 'files_prompt_id' ] = None
	
	def convert_files_system_instructions( ) -> None:
		"""Convert Files system instructions.
		
		Purpose:
		    Converts Files Mode instructions between XML blocks and Markdown headings.
		
		Returns:
		    None: This function updates session state.
		"""
		text_value = st.session_state.get( 'files_system_instructions', '', )
		
		if not isinstance( text_value, str ):
			return
		
		if not text_value.strip( ):
			return
		
		source = text_value.strip( )
		
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'files_system_instructions' ] = converted
	
	def load_files_instruction_template( ) -> None:
		"""Load Files instruction template.
		
		Purpose:
		    Loads the selected Files Mode prompt template into the system-instruction field.
		
		Returns:
		    None: This function updates session state.
		"""
		load_prompt_template( prompt_id_key='files_prompt_id',
			instructions_key='files_system_instructions', )
	
	# ------------------------------------------------------------------
	# Provider Capabilities
	# ------------------------------------------------------------------
	extract_supported = callable( getattr( files, 'extract', None ) )
	ask_supported = callable( getattr( files, 'search', None ) )
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	
	with center:
		st.subheader( '📁 Files API', help=get_files_help( 'FILES_API' ), )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Expander - Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch', ):
			# ------------------------------------------------------------------
			# Expander - File Management
			# ------------------------------------------------------------------
			with st.expander( label='File Management', icon='📂', expanded=False,
					width='stretch', ):
				mgmt_c1, mgmt_c2, mgmt_c3, mgmt_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='xxsmall', )
				
				# ----- Purpose -----
				with mgmt_c1:
					purpose_options = get_files_options( files, 'purpose_options',
						[ 'assistants', 'batch', 'fine-tune', 'user_data' ], )
					purpose_options = [ str( item ) for item in purpose_options if
						str( item ).strip( ) ]
					sanitize_files_selection( 'files_purpose', purpose_options, )
					st.selectbox( label='Purpose', options=purpose_options, key='files_purpose',
						index=None, placeholder='Options', help='Optional provider file '
						                                        'purpose.', )
				
				# ----- Type ------
				with mgmt_c2:
					st.selectbox( label='File Type',
						options=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json', 'csv',
							'xlsx', 'xls', ], key='files_type', index=None, placeholder='Options',
						help='Optional local filter for uploaded file types.', )
				
				# ----- ID ------
				with mgmt_c3:
					st.text_input( label='Manual File ID', key='files_manual_id',
						help=('Optional. Paste a provider file ID/name for '
						      'retrieve, extract, ask, or delete.'), width='stretch', )
				
				# ----- Selected File -----
				with mgmt_c4:
					table_rows = st.session_state.get( 'files_table', [ ], )
					file_options = [ row.get( 'id', '' ) for row in table_rows if
						isinstance( row, dict ) and row.get( 'id', '' ) ]
					sanitize_files_selection( 'files_selected_id', file_options, )
					st.selectbox( label='Selected File', options=file_options,
						key='files_selected_id', index=None, placeholder='Options',
						help='File selected from the latest provider list.', )
			
			# ------------------------------------------------------------------
			# Expander - Request Settings
			# ------------------------------------------------------------------
			with st.expander( label='Request Settings', icon='⚙️', expanded=False,
					width='stretch', ):
				req_c1, req_c2, req_c3, req_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='xxsmall', )
				
				# ----- Model -----
				with req_c1:
					model_options = get_files_options( files, 'model_options', [ ], )
					model_options = [ str( item ) for item in model_options if
						str( item ).strip( ) ]
					sanitize_files_selection( 'files_model', model_options, )
					st.selectbox( label='Model', options=model_options, key='files_model',
						index=None, placeholder='Options',
						help='Provider model for file-aware operations.', )
				
				# ----- Max Tokens -----
				with req_c2:
					st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
						key='files_max_tokens', help=cfg.MAX_OUTPUT_TOKENS, )
				
				# ----- Temperature -----
				with req_c3:
					st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.01,
						key='files_temperature', help=cfg.TEMPERATURE,
						disabled=provider_name == 'GPT', )
				
				# ----- Format -----
				with req_c4:
					format_options = get_files_options( files, 'format_options', [ ], )
					format_options = [ str( item ) for item in format_options if
						str( item ).strip( ) ]
					sanitize_files_selection( 'files_response_format', format_options, )
					st.selectbox( label='Response Format', options=format_options,
						key='files_response_format', index=None, placeholder='Options',
						help='Optional response format.', disabled=provider_name == 'GPT', )
				
				req2_c1, req2_c2, req2_c3, req2_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='xxsmall', )
				
				# ----- Top-P -----
				with req2_c1:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0, step=0.01,
						key='files_top_percent', help=cfg.TOP_P, disabled=provider_name == 'GPT', )
				
				# ----- Frequency -----
				with req2_c2:
					st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='files_frequency_penalty', help=cfg.FREQUENCY_PENALTY,
						disabled=provider_name == 'GPT', )
				
				# ----- Presence -----
				with req2_c3:
					st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='files_presence_penalty', help=cfg.PRESENCE_PENALTY,
						disabled=provider_name == 'GPT', )
				
				# ----- Choice -----
				with req2_c4:
					choice_options = get_files_options( files, 'choice_options', [ ], )
					choice_options = [ str( item ) for item in choice_options if
						str( item ).strip( ) ]
					sanitize_files_selection( 'files_tool_choice', choice_options, )
					st.selectbox( label='Tool Choice', options=choice_options,
						key='files_tool_choice', index=None, placeholder='Options',
						help=cfg.CHOICE,
						disabled=True, )
				
				req3_c1, req3_c2, req3_c3, req3_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='xxsmall', )
				
				# ----- Tools -----
				with req3_c1:
					tool_options = get_files_options( files, 'tool_options', [ ], )
					tool_options = [ str( item ) for item in tool_options if str( item ).strip( ) ]
					sanitize_files_multiselect( 'files_tools', tool_options, )
					st.multiselect( label='Tools', options=tool_options, key='files_tools',
						placeholder='Options', help=cfg.TOOLS, disabled=True, )
				
				# ----- Include -----
				with req3_c2:
					include_options = get_files_options( files, 'include_options', [ ], )
					include_options = [ str( item ) for item in include_options if
						str( item ).strip( ) ]
					sanitize_files_multiselect( 'files_include', include_options, )
					st.multiselect( label='Include', options=include_options, key='files_include',
						placeholder='Options', help=cfg.INCLUDE, disabled=provider_name !=
						                                                  'Grok', )
				
				# ----- Store -----
				with req3_c3:
					st.toggle( label='Store', key='files_store', help=cfg.STORE,
						disabled=provider_name != 'Grok', )
				
				# ----- Stream -----
				with req3_c4:
					st.toggle( label='Stream', key='files_stream', help=cfg.STREAM,
						disabled=provider_name != 'Grok', )
				
				# ----- Reset Button -----
				if st.button( label='Reset', key='files_request_settings_reset', width='stretch',
						icon='🔄', ):
					for key in [ 'files_model', 'files_max_tokens', 'files_temperature',
						'files_response_format', 'files_top_percent', 'files_frequency_penalty',
						'files_presence_penalty', 'files_tool_choice', 'files_tools',
						'files_include', 'files_store', 'files_stream', ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — Files System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch', ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			# ----- Prompt Categories -----
			files_prompt_categories = fetch_prompt_categories( 'Files' )
			current_files_category = st.session_state.get( 'files_prompt_category' )
			
			if current_files_category not in files_prompt_categories:
				st.session_state[ 'files_prompt_category' ] = None
			
			selected_files_category = st.session_state.get( 'files_prompt_category' )
			files_prompt_options = (
				fetch_prompt_options( selected_files_category ) if selected_files_category else
				[ ])
			files_prompt_ids = [ int( option[ 'ID' ] ) for option in files_prompt_options ]
			
			if st.session_state.get( 'files_prompt_id' ) not in files_prompt_ids:
				st.session_state[ 'files_prompt_id' ] = None
			
			# ----- Instruction Text ------
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					key='files_system_instructions', help=cfg.SYSTEM_INSTRUCTIONS, )
			
			# ----- Template Selection ------
			with in_right:
				st.selectbox( label='Category', options=files_prompt_categories, index=None,
					key='files_prompt_category', placeholder='Select Category',
					help=('Limits prompt templates to categories associated '
					      'with file-processing workflows.'),
					on_change=reset_prompt_template_selection, args=('files_prompt_id',), )
				
				st.selectbox( label='Use Template', options=files_prompt_ids, index=None,
					key='files_prompt_id', placeholder='Select Template',
					disabled=not files_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						files_prompt_options, ), help=('Loads the selected prompt into the Files '
					                                   'system-instruction field.'),
					on_change=load_files_instruction_template, )
			
			# ----- Instruction Actions -----
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ----- Clear Button -----
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_files_instructions, icon='🧹', )
			
			# ----- Convert Button ----
			with btn_c2:
				st.button( label='XML ↔️ Markdown', width='stretch',
					on_click=convert_files_system_instructions, )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
		
		upload_tab, list_tab, retrieve_tab, extract_tab, ask_tab, delete_tab = st.tabs(
			[ 'Upload', 'List', 'Retrieve', 'Extract', 'Ask', 'Delete' ] )
		
		# ----- Upload -----
		with upload_tab:
			allowed_types = [ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json', 'csv',
				'xlsx', 'xls', ]
			selected_file_type = st.session_state.get( 'files_type' )
			upload_types = ([ selected_file_type ] if selected_file_type else allowed_types)
			uploaded_file = st.file_uploader( label='Upload File', type=upload_types,
				accept_multiple_files=False, key='files_uploader', )
			
			if uploaded_file is not None:
				st.caption( f'Selected: {uploaded_file.name}' )
			
			# ----- Upload Button -----
			if st.button( 'Upload File', key='files_upload_button', width='stretch', icon='📤', ):
				with st.spinner( 'Uploading file…' ):
					try:
						if uploaded_file is None:
							st.warning( 'Select a file before uploading.' )
						else:
							result = upload_provider_file( uploaded_file=uploaded_file,
								purpose=st.session_state.get( 'files_purpose' ) or None, )
							file_id = normalize_file_id( result )
							
							st.session_state[ 'files_results' ] = result
							st.session_state[ 'files_selected_id' ] = file_id
							st.session_state[ 'files_uploaded' ].append(
								{ 'id': file_id, 'filename': uploaded_file.name,
									'provider': provider_name, } )
							st.success( f'Uploaded file: {file_id}' )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Upload failed: {err.info}' )
			
			if st.session_state.get( 'files_results' ) is not None:
				with st.expander( label='Upload Result', icon='📄', expanded=False,
						width='stretch', ):
					st.write( st.session_state.get( 'files_results' ) )
		
		# ----- List -----
		with list_tab:
			list_c1, list_c2 = st.columns( [ 0.50, 0.50 ] )
			
			# ----- List Button -----
			with list_c1:
				if st.button( 'List Files', key='files_list_button', width='stretch', icon='🔠', ):
					with st.spinner( 'Listing files…' ):
						try:
							rows = refresh_files_table( )
							st.success( f'Loaded {len( rows )} file record(s).' )
						except Exception as exc:
							st.session_state[ 'files_table' ] = [ ]
							err = Error( exc )
							st.error( f'List files failed: {err.info}' )
			
			# ----- Clear Button ----
			with list_c2:
				if st.button( 'Clear Outputs', key='files_clear_outputs', width='stretch',
						on_click=clear_files_outputs, icon='🧹', ):
					st.rerun( )
			
			df_files = pd.DataFrame( st.session_state.get( 'files_table', [ ] ) )
			
			if not df_files.empty:
				st.data_editor( df_files, use_container_width=True, hide_index=True, disabled=True,
					key='files_table_view', )
			else:
				st.info( 'No file records loaded yet.' )
		
		# ----- Retrieve -----
		with retrieve_tab:
			if not st.session_state.get( 'files_retrieve_id' ):
				st.session_state[ 'files_retrieve_id' ] = (
					get_effective_file_id( 'files_selected_id', 'files_manual_id', ))
			
			st.text_input( label='Retrieve File ID', key='files_retrieve_id',
				help='Provider file ID/name to retrieve.', width='stretch', )
			
			if st.button( 'Retrieve File', key='files_retrieve_button', width='stretch',
					icon='🐕', ):
				with st.spinner( 'Retrieving file metadata…' ):
					try:
						file_id = st.session_state.get( 'files_retrieve_id', '', ).strip( )
						
						if not file_id:
							st.warning( 'Select or enter a file ID before retrieving.' )
						else:
							result = retrieve_provider_file( file_id )
							st.session_state[ 'files_metadata' ] = (
								normalize_file_metadata( result ))
							st.session_state[ 'files_results' ] = result
							st.success( 'File metadata retrieved.' )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Retrieve failed: {err.info}' )
			
			if st.session_state.get( 'files_metadata' ):
				st.json( st.session_state.get( 'files_metadata' ) )
		
		# ----- Extract -----
		with extract_tab:
			if not extract_supported:
				st.info( f'{provider_name} Files wrapper does not expose '
				         f'an extract method.' )
			
			if not st.session_state.get( 'files_extract_id' ):
				st.session_state[ 'files_extract_id' ] = (
					get_effective_file_id( 'files_selected_id', 'files_manual_id', ))
			
			ext_c1, ext_c2 = st.columns( [ 0.50, 0.50 ], border=True, gap='xxsmall', )
			
			# ----- Extract ------
			with ext_c1:
				st.text_input( label='Extract File ID', key='files_extract_id',
					help='Provider file ID/name to download or extract.', width='stretch', )
			
			with ext_c2:
				st.selectbox( label='Download Format',
					options=[ '', 'DOWNLOAD_FORMAT_TEXT', 'DOWNLOAD_FORMAT_BYTES', ],
					key='files_download_format', index=None, placeholder='Options',
					help=('Retained for compatible provider download '
					      'workflows. The current extract wrappers choose '
					      'the provider-native content representation.'), disabled=True, )
			
			st.number_input( label='Page Number', min_value=0, step=1, key='files_page_number',
				help=('Retained for provider compatibility. Current Files '
				      'wrappers extract complete content.'), disabled=True, )
			
			if st.button( 'Extract File Content', key='files_extract_button', width='stretch',
					disabled=not extract_supported, icon='🦷', ):
				with st.spinner( 'Extracting file content…' ):
					try:
						file_id = st.session_state.get( 'files_extract_id', '', ).strip( )
						
						if not file_id:
							st.warning( 'Select or enter a file ID before '
							            'extracting content.' )
						else:
							content = extract_provider_file( file_id )
							content_text = normalize_file_content( content )
							st.session_state[ 'files_content' ] = content
							st.session_state[ 'files_content_text' ] = content_text
							st.session_state[ 'files_results' ] = content
							st.success( 'File content extracted.' )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Extract failed: {err.info}' )
			
			if st.session_state.get( 'files_content_text' ):
				st.text_area( label='Extracted Content',
					value=st.session_state.get( 'files_content_text', '', ), height=300,
					width='stretch', disabled=True, )
				
				st.download_button( label='Download Extracted Text',
					data=st.session_state.get( 'files_content_text', '', ),
					file_name='file_content.txt', mime='text/plain', width='stretch', )
			elif isinstance( st.session_state.get( 'files_content' ), bytes, ):
				st.download_button( label='Download File Content',
					data=st.session_state.get( 'files_content' ), file_name='file_content.bin',
					mime='application/octet-stream', width='stretch', )
		
		# ----- Ask ------
		with ask_tab:
			if not ask_supported:
				st.info( f'{provider_name} Files wrapper does not expose a '
				         f'compatible file-aware search method.' )
			
			render_files_messages( )
			file_id = get_effective_file_id( 'files_selected_id', 'files_manual_id',
				'files_retrieve_id', 'files_extract_id', )
			
			if file_id:
				st.caption( f'Active File ID: {file_id}' )
			else:
				st.info( 'Select or enter a file ID before asking a '
				         'file-aware question.' )
			
			st.text_area( label='Question', key='files_question', height=120, width='stretch',
				placeholder='Ask a question about the selected file.', )
			
			ask_c1, ask_c2 = st.columns( [ 0.50, 0.50 ] )
			
			# ----- Ask Button -----
			with ask_c1:
				if st.button( 'Ask File', key='files_ask_button', width='stretch',
						disabled=not ask_supported, icon='❓', ):
					with st.spinner( 'Asking file-aware question…' ):
						try:
							active_file_id = get_effective_file_id( 'files_selected_id',
								'files_manual_id', 'files_retrieve_id', 'files_extract_id', )
							question = st.session_state.get( 'files_question', '', ).strip( )
							
							if not active_file_id:
								st.warning( 'Select or enter a file ID before '
								            'asking a question.' )
							elif not question:
								st.warning( 'Enter a question before asking the file.' )
							elif not st.session_state.get( 'files_model' ):
								st.warning( 'Select a model before asking a '
								            'file-aware question.' )
							else:
								append_files_message( 'user', question, )
								answer = ask_provider_file( active_file_id, question, )
								st.session_state[ 'files_last_answer' ] = answer
								
								previous_id = (
										getattr( files, 'previous_id', None, ) or getattr( files,
									'previous_response_id', None, ) or st.session_state.get(
									'files_previous_response_id', '', ) or '')
								st.session_state[ 'files_previous_response_id' ] = previous_id
								
								append_files_message( 'assistant', answer, )
								st.markdown( answer )
						except Exception as exc:
							err = Error( exc )
							st.error( f'File question failed: {err.info}' )
			
			# ----- Clear Button -----
			with ask_c2:
				if st.button( 'Clear Messages', key='files_clear_messages_button', width='stretch',
						on_click=clear_files_messages, icon='🧹', ):
					st.rerun( )
			
			if st.session_state.get( 'files_last_answer' ):
				st.download_button( label='Download Answer',
					data=st.session_state.get( 'files_last_answer', '', ),
					file_name='file_answer.txt', mime='text/plain', width='stretch', )
		
		# ----- Delete -----
		with delete_tab:
			if not st.session_state.get( 'files_delete_id' ):
				st.session_state[ 'files_delete_id' ] = (
					get_effective_file_id( 'files_selected_id', 'files_manual_id', ))
			
			st.text_input( label='Delete File ID', key='files_delete_id',
				help='Provider file ID/name to delete.', width='stretch', )
			
			confirm_delete = st.checkbox( 'Confirm Delete', key='files_confirm_delete', )
			
			# ----- Delete Button -----
			if st.button( 'Delete File', key='files_delete_button', width='stretch',
					disabled=not confirm_delete, icon='❌', ):
				with st.spinner( 'Deleting file…' ):
					try:
						file_id = st.session_state.get( 'files_delete_id', '', ).strip( )
						
						if not file_id:
							st.warning( 'Select or enter a file ID before deleting.' )
						else:
							result = delete_provider_file( file_id )
							st.session_state[ 'files_delete_result' ] = normalize_file_metadata(
								result )
							st.session_state[ 'files_results' ] = result
							st.session_state[ 'files_table' ] = [ row for row in
								st.session_state.get( 'files_table', [ ], ) if
								isinstance( row, dict ) and row.get( 'id' ) != file_id ]
							
							if st.session_state.get( 'files_selected_id' ) == file_id:
								st.session_state[ 'files_selected_id' ] = ''
							
							st.success( 'File deleted.' )
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
	
	# ------ Vector Stores Utilities -----
	def get_storage_help( name: str, fallback: str = '' ) -> str:
		"""Get storage help.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and downstream logic
		    can consume it consistently.
		
		Args:
		    name (str): Name value used by the operation.
		    fallback (str): Fallback value used by the operation.
		
		Returns:
		    str: Return value produced by the operation."""
		return str( getattr( cfg, name, fallback ) or fallback )
	
	def get_storage_options( instance: Any, attr_name: str,
		fallback: Optional[ List[ Any ] ] = None ) -> List[ Any ]:
		"""Get storage options.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view  of provider capabilities, stored state, or response metadata so UI
		    controls
		    and downstream logic can consume it consistently.
		
		Args:
		    instance (Any): Instance value used by the operation.
		    attr_name (str): Attr name value used by the operation.
		    fallback (Optional[List[Any]]): Fallback value used by the operation.
		
		Returns:
		    List[Any]: Return value produced by the operation."""
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
		"""Parse storage json.
		
		Purpose:
		    Performs the parse_storage_json workflow using the inputs supplied by the caller and
		    the current
		    runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    value (Any): Value value used by the operation.
		    label (str): Label value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation."""
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
		"""Parse storage ids.
		
		Purpose:
		    Performs the parse_storage_ids workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so
		    related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    value (Any): Value value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		raw = str( value or '' )
		return [ item.strip( ) for item in raw.split( ',' ) if item.strip( ) ]
	
	def call_storage_method( instance: Any, method_names: List[ str ],
		kwargs: Optional[ Dict[ str, Any ] ] = None ) -> Any:
		"""Call storage method.
		
		Purpose:
		    Performs the call_storage_method workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so
		    related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    instance (Any): Instance value used by the operation.
		    method_names (List[str]): Method names value used by the operation.
		    kwargs (Optional[Dict[str, Any]]): Kwargs value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation."""
		kwargs = kwargs or { }
		
		for method_name in method_names:
			method = getattr( instance, method_name, None )
			if not callable( method ):
				continue
			
			try:
				import inspect
				
				signature = inspect.signature( method )
				parameters = signature.parameters
				has_kwargs = any( parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in
					parameters.values( ) )
				
				if has_kwargs:
					call_kwargs = { key: value for key, value in kwargs.items( ) if
						value is not None and value != '' and value != [ ] }
				else:
					call_kwargs = { key: value for key, value in kwargs.items( ) if
						key in parameters and value is not None and value != '' and value != [ ] }
				
				return method( **call_kwargs )
			except TypeError:
				clean_kwargs = { key: value for key, value in kwargs.items( ) if
					value is not None and value != '' and value != [ ] }
				
				if len( clean_kwargs ) == 1:
					return method( list( clean_kwargs.values( ) )[ 0 ] )
				
				raise
		
		raise AttributeError(
			f'Provider "{provider_name}" does not expose a compatible method from: '
			f'{", ".join( method_names )}.' )
	
	def normalize_storage_object( value: Any ) -> Dict[ str, Any ]:
		"""Normalize storage object.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application
		    processing. The
		    function reduces provider, user-input, or serialization differences before values are
		    stored or displayed.
		
		Args:
		    value (Any): Value value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation."""
		if value is None:
			return { }
		
		if isinstance( value, dict ):
			result = dict( value )
		elif hasattr( value, 'model_dump' ):
			try:
				dumped = value.model_dump( )
				result = dumped if isinstance( dumped, dict ) else { 'result': dumped }
			except Exception:
				result = { 'result': str( value ) }
		elif hasattr( value, 'dict' ):
			try:
				dumped = value.dict( )
				result = dumped if isinstance( dumped, dict ) else { 'result': dumped }
			except Exception:
				result = { 'result': str( value ) }
		else:
			result = { }
			for attr_name in [ 'id', 'name', 'display_name', 'description', 'status', 'state',
				'file_counts', 'usage_bytes', 'created_at', 'expires_at', 'metadata', 'deleted',
				'collection_id', 'collection_name', 'collection_description', 'documents_count',
				'document_count', 'size_bytes', ]:
				if hasattr( value, attr_name ):
					result[ attr_name ] = getattr( value, attr_name )
			
			if not result:
				result = { 'result': str( value ) }
		
		collection_id = result.get( 'collection_id' ) or result.get( 'id' ) or ''
		collection_name = result.get( 'collection_name' ) or result.get( 'display_name' )
		collection_name = collection_name or result.get( 'name' ) or collection_id or ''
		description = result.get( 'collection_description' ) or result.get( 'description' ) or ''
		status = result.get( 'status' ) or result.get( 'state' ) or ''
		file_counts = result.get( 'file_counts' )
		file_counts = file_counts if file_counts is not None else result.get( 'documents_count' )
		file_counts = file_counts if file_counts is not None else result.get( 'document_count' )
		usage_bytes = result.get( 'usage_bytes' )
		usage_bytes = usage_bytes if usage_bytes is not None else result.get( 'size_bytes' )
		
		result[ 'id' ] = str( result.get( 'id' ) or collection_id or '' )
		result[ 'name' ] = str( result.get( 'name' ) or collection_name or '' )
		result[ 'display_name' ] = str( result.get( 'display_name' ) or collection_name or '' )
		result[ 'description' ] = str( result.get( 'description' ) or description or '' )
		result[ 'status' ] = str( status or '' )
		result[ 'file_counts' ] = file_counts if file_counts is not None else ''
		result[ 'usage_bytes' ] = usage_bytes if usage_bytes is not None else ''
		
		if collection_id:
			result[ 'collection_id' ] = str( collection_id )
		
		if collection_name:
			result[ 'collection_name' ] = str( collection_name )
		
		if description:
			result[ 'collection_description' ] = str( description )
		
		return result
	
	def normalize_storage_rows( result: Any ) -> List[ Dict[ str, Any ] ]:
		"""Normalize storage rows.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application
		    processing. The
		    function reduces provider, user-input, or serialization differences before values are
		    stored or
		    displayed.
		
		Args:
		    result (Any): Result value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation."""
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
		elif isinstance( result, dict ) and isinstance( result.get( 'collections' ), list ):
			items = result.get( 'collections' )
		elif isinstance( result, list ):
			items = result
		else:
			items = [ result ]
		
		rows = [ ]
		for item in items:
			obj = normalize_storage_object( item )
			if not obj:
				continue
			
			store_id = obj.get( 'id' ) or obj.get( 'collection_id' ) or ''
			store_name = obj.get( 'name' ) or obj.get( 'display_name' )
			store_name = store_name or obj.get( 'collection_name' ) or store_id or ''
			rows.append( { 'id': str( store_id or '' ), 'name': str( store_name or '' ),
				'status': str( obj.get( 'status', '' ) or '' ),
				'file_counts': str( obj.get( 'file_counts', '' ) or '' ),
				'usage_bytes': str( obj.get( 'usage_bytes', '' ) or '' ), } )
		
		return rows
	
	def normalize_search_results( result: Any ) -> List[ Dict[ str, Any ] ]:
		"""Normalize search results.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application
			processing. The function reduces provider, user-input, or serialization differences
			before values are stored or displayed.
		
		Args:
		    result (Any): Result value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation."""
		if result is None:
			return [ ]
		
		if isinstance( result, str ):
			return [ { 'text': result } ]
		
		if hasattr( result, 'data' ):
			items = getattr( result, 'data' )
		elif isinstance( result, dict ) and isinstance( result.get( 'data' ), list ):
			items = result.get( 'data' )
		elif isinstance( result, dict ) and isinstance( result.get( 'results' ), list ):
			items = result.get( 'results' )
		elif isinstance( result, dict ) and isinstance( result.get( 'matches' ), list ):
			items = result.get( 'matches' )
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
		"""Save uploaded storage file.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application
		    workflows. The function standardizes file handling and returns a stable reference for
		    downstream
		    processing.
		
		Args:
		    uploaded_file (Any): Uploaded file value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation."""
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
		"""Get selected store id.
		
		Purpose:
		    Returns normalized information for the application component. The method provides a
		    stable view of provider capabilities, stored state, or response metadata so UI
		    controls and
		    downstream logic can consume it consistently.
		
		Args:
		    table_key (str): Table key value used by the operation.
		    manual_key (str): Manual key value used by the operation.
		    selected_key (str): Selected key value used by the operation.
		
		Returns:
		    str: Return value produced by the operation."""
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
		"""Render storage table.
		
		Purpose:
		    Renders the requested user interface element or result block in Streamlit using
		    normalized
		    inputs. The function keeps presentation logic isolated from provider calls and
		    data-processing steps so the screen output remains predictable.
		
		Args:
		    rows (List[Dict[str, Any]]): Rows value used by the operation.
		    key (str): Key value used by the operation.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		df_rows = pd.DataFrame( rows or [ ] )
		if df_rows.empty:
			st.info( 'No storage records loaded yet.' )
			return
		
		st.data_editor( df_rows, use_container_width=True, hide_index=True, key=key )
	
	def render_storage_metadata( metadata: Dict[ str, Any ] ) -> None:
		"""Render storage metadata.
		
		Purpose:
		    Renders the requested user interface element or result block in Streamlit using
		    normalized inputs. The function keeps presentation logic isolated from provider calls
		    and
		    data-processing steps so the screen output remains predictable.
		
		Args:
		    metadata (Dict[str, Any]): Metadata value used by the operation.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		if not isinstance( metadata, dict ) or len( metadata ) == 0:
			st.info( 'No metadata loaded yet.' )
			return
		
		st.json( metadata )
	
	def render_storage_search_results( rows: List[ Dict[ str, Any ] ] ) -> None:
		"""Render storage search results.
		
		Purpose:
		    Renders the requested user interface element or result block in Streamlit using
		    normalized inputs. The function keeps presentation logic isolated from provider calls
		    and data-processing
		    steps so the screen output remains predictable.
		
		Args:
		    rows (List[Dict[str, Any]]): Rows value used by the operation.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		if not isinstance( rows, list ) or len( rows ) == 0:
			st.info( 'No search results loaded yet.' )
			return
		
		df_results = pd.DataFrame( rows )
		st.data_editor( df_results, use_container_width=True, hide_index=True,
			key='stores_search_results_view' )
	
	def clear_vector_store_outputs( ) -> None:
		"""Clear vector store outputs.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled
		    manner. The function keeps cleanup behavior centralized so callers do not duplicate
		    lifecycle
		    logic.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		st.session_state[ 'stores_store_metadata' ] = { }
		st.session_state[ 'stores_search_results' ] = [ ]
		st.session_state[ 'stores_files_table' ] = [ ]
		st.session_state[ 'stores_batch_result' ] = { }
	
	def clear_stores_instructions( ) -> None:
		"""Clear Vector Stores instructions.
		
		Purpose:
		    Clears the Vector Stores system-instruction text and selected prompt template without
		    changing the selected prompt category, vector-store configuration, or provider results.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		st.session_state[ 'stores_system_instructions' ] = ''
		st.session_state[ 'stores_prompt_id' ] = None
	
	def convert_stores_system_instructions( ) -> None:
		"""Convert Vector Stores system instructions.
		
		Purpose:
		    Converts the active Vector Stores system instructions between supported XML-style
		    instruction blocks and Markdown headings.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		text_value = st.session_state.get( 'stores_system_instructions', '' )
		
		if not isinstance( text_value, str ) or not text_value.strip( ):
			return
		
		source = text_value.strip( )
		
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'stores_system_instructions' ] = converted
	
	def load_stores_instruction_template( ) -> None:
		"""Load Vector Stores instruction template.
		
		Purpose:
		    Loads the selected Vector Stores prompt template into the Vector Stores
		    system-instruction field using its stable prompt identifier.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			load_prompt_template( prompt_id_key='stores_prompt_id',
				instructions_key='stores_system_instructions', )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Vector Stores Mode'
			ex.method = 'load_stores_instruction_template( ) -> None'
			Logger( ).write( ex )
			raise ex
	
	if provider_name == 'Gemini':
		st.warning( 'Gemini storage available via File Search Stores and/or Google Cloud Buckets.' )
		st.stop( )
	
	vector = get_vectorstores_module( provider_name )
	for key, default_value in { 'stores_table': [ ], 'stores_files_table': [ ],
		'stores_store_metadata': { }, 'stores_batch_result': { }, 'stores_search_results': [ ],
		'stores_messages': [ ], }.items( ):
		if key not in st.session_state or not isinstance( st.session_state.get( key ),
				type( default_value ) ):
			st.session_state[ key ] = default_value
	
	for key, default_value in { 'stores_name': '', 'stores_id': '', 'stores_manual_id': '',
		'stores_description': '', 'stores_metadata': '', 'stores_query': '', 'stores_file_id': '',
		'stores_file_ids_text': '', 'stores_selected_id': '', }.items( ):
		if key not in st.session_state:
			st.session_state[ key ] = default_value
	
	if 'stores_max_tokens' not in st.session_state:
		st.session_state[ 'stores_max_tokens' ] = 0
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( '🧊 Vector Stores', help=cfg.VECTORSTORES_API )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Expander - Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
				border=True, gap='xxsmall' )
			
			# ----- Store Name -----
			with ctrl_c1:
				st.text_input( label='Store Name', key='stores_name',
					help='Name used when creating a vector store.', width='stretch',
					placeholder='Enter store name' )
			
			# ----- Store ID -----
			with ctrl_c2:
				st.text_input( label='Manual Store ID', key='stores_manual_id',
					help='Optional. Paste a vector store or collection ID.', width='stretch' )
			
			# ----- Model -----
			with ctrl_c3:
				st.selectbox( label='Answer Model',
					options=get_storage_options( vector, 'model_options', [ ] ),
					key='stores_model',
					index=None, placeholder='Options',
					help='Optional. Model used for store-backed answers when supported.' )
			
			# ----- Max Tokens ----
			with ctrl_c4:
				st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
					key='stores_max_tokens', help=cfg.MAX_OUTPUT_TOKENS )
			
			desc_c1, desc_c2 = st.columns( [ 0.50, 0.50 ], border=True, gap='xxsmall' )
			
			# ----- Description -----
			with desc_c1:
				st.text_area( label='Description', key='stores_description', height=80,
					width='stretch', help='Optional. Vector store description when supported.' )
			
			# ----- Metadata -----
			with desc_c2:
				st.text_area( label='Metadata JSON', key='stores_metadata', height=80,
					width='stretch',
					help='Optional. JSON object metadata for create/update calls.' )
			
			# ----- Reset Button -----
			if st.button( label='Reset', key='reset_store_config', icon='🔄', width='stretch' ):
				for key in [ 'stores_name', 'stores_manual_id', 'stores_model',
					'stores_max_tokens', 'stores_description', 'stores_metadata', ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — File Uploader
		# ------------------------------------------------------------------
		with st.expander( label='File Uploading', expanded=False, width='stretch', icon='📤' ):
			batch_c1, batch_c2, batch_c3 = st.columns( [ 0.25, 0.35, 0.40 ], border=True,
				gap='xxsmall' )
			
			# ----- File ID -----
			with batch_c1:
				st.text_input( label='File ID', key='stores_file_id',
					help='Existing provider file ID used by Attach File and Delete File.',
					width='stretch', placeholder='file_abc123' )
			
			# ----- File IDs -----
			with batch_c2:
				st.text_area( label='File IDs', key='stores_file_ids_text', height=80,
					width='stretch', help='Existing provider file IDs used to create a file '
					                      'batch.',
					placeholder='file_abc,file_def,file_xyz' )
			
			# ----- File Upload -----
			with batch_c3:
				uploaded_store_file = st.file_uploader( label='Upload File to Store',
					type=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json', 'csv' ],
					key='stores_file_upload',
					help='Uploads a local file and attaches it to the selected store.' )
			
			fileop_c1, fileop_c2, fileop_c3, fileop_c4, fileop_c5 = st.columns(
				[ 0.2, 0.2, 0.2, 0.2, 0.2 ] )
			
			rows = st.session_state.get( 'stores_table', [ ] )
			store_ids = [ row.get( 'id', '' ) for row in rows if
				isinstance( row, dict ) and row.get( 'id', '' ) ]
			
			selected_store_id = get_selected_store_id( table_key='stores_table',
				manual_key='stores_manual_id', selected_key='stores_selected_id' )
			
			# ----- Attach Button -----
			with fileop_c1:
				if st.button( 'Attach File', key='attach_vector_store_file', width='stretch',
						icon='📎' ):
					with st.spinner( 'Attaching file…' ):
						try:
							if provider_name == 'Grok':
								st.warning( 'Grok collection file attachment requires '
								            'collection-management capability.' )
							elif not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							elif not st.session_state.get( 'stores_file_id', '' ).strip( ):
								st.warning( 'Enter a file ID first.' )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'create_file', 'attach_file', 'add_file' ],
									kwargs={ 'store_id': selected_store_id, 'id':
										selected_store_id,
										'file_id': st.session_state.get( 'stores_file_id',
											'' ).strip( ), } )
								st.session_state[
									'stores_batch_result' ] = normalize_storage_object( result )
								st.success( 'File attach request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Attach file failed: {err.info}' )
			
			# ----- List Button -----
			with fileop_c2:
				if st.button( 'List Files', key='list_vector_store_files', width='stretch',
						icon='🔠' ):
					with st.spinner( 'Listing store files…' ):
						try:
							if provider_name == 'Grok':
								st.warning(
									'Grok list-files requires collection-management capability.' )
							elif not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'list_files', 'files', 'list_store_files' ],
									kwargs={ 'store_id': selected_store_id, 'id':
										selected_store_id,
										'limit': 100, 'order': 'desc', } )
								rows = normalize_storage_rows( result )
								st.session_state[ 'stores_files_table' ] = rows
								st.success( f'Loaded {len( rows )} store file record(s).' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'List store files failed: {err.info}' )
			
			# ----- Delete Button -----
			with fileop_c3:
				if st.button( 'Delete File', key='delete_vector_store_file', width='stretch',
						icon='❌' ):
					with st.spinner( 'Deleting store file…' ):
						try:
							if provider_name == 'Grok':
								st.warning(
									'Grok delete-file requires collection-management capability.' )
							elif not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							elif not st.session_state.get( 'stores_file_id', '' ).strip( ):
								st.warning( 'Enter a file ID first.' )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'delete_file', 'remove_file' ],
									kwargs={ 'store_id': selected_store_id, 'id':
										selected_store_id,
										'file_id': st.session_state.get( 'stores_file_id',
											'' ).strip( ), } )
								st.session_state[
									'stores_batch_result' ] = normalize_storage_object( result )
								st.success( 'Store file delete request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Delete store file failed: {err.info}' )
			
			# ----- Create Button -----
			with fileop_c4:
				if st.button( 'Create Batch', key='create_vector_store_batch', width='stretch',
						icon='➕' ):
					with st.spinner( 'Creating file batch…' ):
						try:
							if provider_name == 'Grok':
								st.warning( 'Grok batch attachment requires collection-management '
								            'capability.' )
							elif not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							else:
								file_ids = parse_storage_ids(
									st.session_state.get( 'stores_file_ids_text', '' ) )
								
								if not file_ids:
									st.warning( 'Enter one or more file IDs first.' )
								else:
									result = call_storage_method( instance=vector,
										method_names=[ 'create_file_batch', 'create_batch',
											'batch' ], kwargs={ 'store_id': selected_store_id,
											'id': selected_store_id, 'file_ids': file_ids, } )
									st.session_state[
										'stores_batch_result' ] = normalize_storage_object(
										result )
									st.success( 'Batch request submitted.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Batch request failed: {err.info}' )
			
			# ----- Upload Button -----
			with fileop_c5:
				if st.button( 'Upload + Attach', key='upload_attach_vector_store_file',
						width='stretch', icon='📤' ):
					with st.spinner( 'Uploading and attaching file…' ):
						try:
							if provider_name == 'Grok':
								st.warning(
									'Grok upload-to-collection requires collection-management '
									'capability.' )
							elif not selected_store_id:
								st.warning( 'Select or enter a store ID first.' )
							elif uploaded_store_file is None:
								st.warning( 'Select a file first.' )
							else:
								path = save_uploaded_storage_file( uploaded_store_file )
								result = call_storage_method( instance=vector,
									method_names=[ 'upload_file', 'upload', 'files_upload' ],
									kwargs={ 'store_id': selected_store_id, 'id':
										selected_store_id,
										'path': path, 'file_path': path, } )
								st.session_state[
									'stores_batch_result' ] = normalize_storage_object( result )
								st.success( 'Upload request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Upload attach failed: {err.info}' )
			
			# ----- Reset Button -----
			if st.button( label='Reset', key='reset_batches', icon='🔄', width='stretch' ):
				for key in [ 'stores_file_id', 'stores_file_ids_text', 'stores_file_upload', ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — Store Life-cycle
		# ------------------------------------------------------------------
		with st.expander( label='Store Management', expanded=False, width='stretch', icon='🏛️' ):
			create_c1, create_c2, create_c3 = st.columns( [ 0.33, 0.33, 0.33 ] )
			
			# ----- Create Button ------
			with create_c1:
				if st.button( 'Create Store', key='create_vector_store', width='stretch',
						icon='➕' ):
					with st.spinner( 'Creating vector store…' ):
						try:
							name = st.session_state.get( 'stores_name', '' ).strip( )
							
							if not name:
								st.warning( 'Enter a vector store name before creating.' )
							elif provider_name == 'Grok':
								st.warning(
									'Grok collection creation requires collection-management '
									'capability. Use configured collections for search.' )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'create', 'create_store', 'create_collection' ],
									kwargs={ 'name': name,
										'description': st.session_state.get( 'stores_description',
											'' ) or None, 'metadata': parse_storage_json(
											st.session_state.get( 'stores_metadata', '' ),
											'Vector store metadata' ),
										'file_ids': parse_storage_ids(
											st.session_state.get( 'stores_file_ids_text',
												'' ) ), } )
								metadata = normalize_storage_object( result )
								st.session_state[ 'stores_store_metadata' ] = metadata
								st.session_state[ 'stores_id' ] = metadata.get( 'id', name )
								st.success( f'Created store: {st.session_state[ "stores_id" ]}' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Create vector store failed: {err.info}' )
			
			# ----- List Button -----
			with create_c2:
				if st.button( 'List Stores', key='list_vector_stores', width='stretch', icon='🔠' ):
					with st.spinner( 'Listing vector stores…' ):
						try:
							if provider_name == 'Grok':
								result = call_storage_method( instance=vector,
									method_names=[ 'list' ], kwargs={ } )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'list_stores', 'list', 'list_collections' ],
									kwargs={ 'limit': 100, 'order': 'desc' } )
							
							rows = normalize_storage_rows( result )
							st.session_state[ 'stores_table' ] = rows
							st.success( f'Loaded {len( rows )} store record(s).' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'List vector stores failed: {err.info}' )
			
			rows = st.session_state.get( 'stores_table', [ ] )
			store_ids = [ row.get( 'id', '' ) for row in rows if
				isinstance( row, dict ) and row.get( 'id', '' ) ]
			
			# ----- ID Selection -----
			with create_c3:
				st.selectbox( label='Selected Store', options=store_ids, key='stores_selected_id',
					index=None, placeholder='Options', help='Store selected from latest list.' )
				
				selected_store_id = get_selected_store_id( table_key='stores_table',
					manual_key='stores_manual_id', selected_key='stores_selected_id' )
					
			retrieve_c1, retrieve_c2, retrieve_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
			
			# ----- Retrieve Button -----
			with retrieve_c1:
				if st.button( 'Retrieve Store', key='retrieve_vector_store', width='stretch', icon='🐕' ):
					with st.spinner( 'Retrieving vector store…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID before retrieving.' )
							elif provider_name == 'Grok':
								result = call_storage_method( instance=vector,
									method_names=[ 'retrieve' ],
									kwargs={ 'store_id': selected_store_id } )
								metadata = normalize_storage_object( result )
								st.session_state[ 'stores_store_metadata' ] = metadata
								st.session_state[ 'stores_id' ] = selected_store_id
								st.success( 'Store metadata retrieved.' )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'retrieve', 'retrieve_store',
										'get_collection' ],
									kwargs={ 'store_id': selected_store_id,
										'id': selected_store_id } )
								metadata = normalize_storage_object( result )
								st.session_state[ 'stores_store_metadata' ] = metadata
								st.session_state[ 'stores_id' ] = selected_store_id
								st.success( 'Store metadata retrieved.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Retrieve vector store failed: {err.info}' )
			
			# ----- Update Button -----
			with retrieve_c2:
				if st.button( 'Update Store', key='update_vector_store', width='stretch', icon='🔼' ):
					with st.spinner( 'Updating vector store…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID before updating.' )
							elif provider_name == 'Grok':
								st.warning( 'Use configured collections for search.' )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'update', 'update_store', 'update_collection' ],
									kwargs={ 'store_id': selected_store_id, 'id':
										selected_store_id,
										'name': st.session_state.get( 'stores_name', '' ) or None,
										'description': st.session_state.get( 'stores_description',
											'' ) or None, 'metadata': parse_storage_json(
											st.session_state.get( 'stores_metadata', '' ),
											'Vector store metadata' ), } )
								st.session_state[ 'stores_store_metadata' ] = normalize_storage_object( result )
								st.success( 'Store update submitted.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Update vector store failed: {err.info}' )
			
			# ----- Delete Button -----
			with retrieve_c3:
				if st.button( 'Delete Store', key='delete_vector_store', width='stretch', icon='❌' ):
					with st.spinner( 'Deleting vector store…' ):
						try:
							if not selected_store_id:
								st.warning( 'Select or enter a store ID before deleting.' )
							elif provider_name == 'Grok':
								st.warning( 'Use configured collections for search.' )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'delete', 'delete_store', 'delete_collection' ],
									kwargs={ 'store_id': selected_store_id,
										'id': selected_store_id } )
								st.session_state[ 'stores_store_metadata' ] = \
									normalize_storage_object( result )
								st.success( 'Delete request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Delete vector store failed: {err.info}' )
			
			render_storage_table( st.session_state.get( 'stores_table', [ ] ),
				'vector_stores_table_view' )
			
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			# ------ Vector Stores Prompt Categories -----
			stores_prompt_categories = fetch_prompt_categories( 'Vector Stores' )
			current_stores_category = st.session_state.get( 'stores_prompt_category' )
			if current_stores_category not in stores_prompt_categories:
				st.session_state[ 'stores_prompt_category' ] = None
			
			selected_stores_category = st.session_state.get( 'stores_prompt_category' )
			stores_prompt_options = fetch_prompt_options(
				selected_stores_category ) if selected_stores_category else [ ]
			
			stores_prompt_ids = [ int( option[ 'ID' ] ) for option in stores_prompt_options ]
			if st.session_state.get( 'stores_prompt_id' ) not in stores_prompt_ids:
				st.session_state[ 'stores_prompt_id' ] = None
			
			# ----- Instruction Text -------
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					key='stores_system_instructions',
					help=get_storage_help( 'SYSTEM_INSTRUCTIONS' ), )
			
			# ----- Template Selection -----
			with in_right:
				st.selectbox( label='Category', options=stores_prompt_categories, index=None,
					key='stores_prompt_category', placeholder='Select Category',
					help=('Limits prompt templates to categories associated with '
					      'vector-store and retrieval workflows.'),
					on_change=reset_prompt_template_selection, args=('stores_prompt_id',), )
				
				st.selectbox( label='Use Template', options=stores_prompt_ids, index=None,
					key='stores_prompt_id', placeholder='Select Template',
					disabled=not stores_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						stores_prompt_options, ),
					help=('Loads the selected prompt into the Vector Stores '
					      'system-instruction field.'),
					on_change=load_stores_instruction_template, )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ----- Clear Button -----
			with btn_c1:
				st.button( label='Clear Instructions', key='clear_stores_instructions',
					width='stretch', on_click=clear_stores_instructions, icon='🧹' )
			
			# ----- Convert Button ------
			with btn_c2:
				st.button( label='XML ↔️ Markdown', key='convert_stores_instructions',
					width='stretch', on_click=convert_stores_system_instructions, )
				
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		st.text_area( label='Search Query', key='stores_query', height=90, width='stretch',
			placeholder='Search this vector store or collection.' )
		
		search_c1, search_c2 = st.columns( [ 0.50, 0.50 ] )
		
		# ----- Search Button ----
		with search_c1:
			if st.button( 'Search Store', key='search_vector_store', width='stretch', icon='🔍' ):
				with st.spinner( 'Searching store…' ):
					try:
						if not selected_store_id:
							st.warning( 'Select or enter a store ID before searching.' )
						elif not st.session_state.get( 'stores_query', '' ).strip( ):
							st.warning( 'Enter a search query first.' )
						else:
							query_text = st.session_state.get( 'stores_query', '' ).strip( )
							
							if provider_name == 'Grok':
								result = call_storage_method( instance=vector,
									method_names=[ 'search' ],
									kwargs={ 'store_id': selected_store_id,
										'prompt': query_text, 'model': st.session_state.get(
											'stores_model' ) or 'grok-4-fast', } )
							else:
								result = call_storage_method( instance=vector,
									method_names=[ 'search', 'search_store', 'query',
										'query_collection' ],
									kwargs={ 'store_id': selected_store_id,
										'id': selected_store_id, 'query': query_text, } )
							
							rows = normalize_search_results( result )
							st.session_state[ 'stores_search_results' ] = rows
							st.success( f'Returned {len( rows )} result(s).' )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Store search failed: {err.info}' )
		
		# ----- Clear Button -----
		with search_c2:
			st.button( label='Clear Outputs', key='clear_vector_store_outputs',
				width='stretch',
				on_click=clear_vector_store_outputs, icon='🧹' )
		
		render_storage_search_results( st.session_state.get( 'stores_search_results', [ ] ) )

# ======================================================================================
# FILE SEARCH STORES MODE
# ======================================================================================
elif mode == 'File Search Stores':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	
	if provider_name != 'Gemini':
		st.warning( 'File Search Stores are available for Gemini only.' )
		st.stop( )
	
	searcher = get_file_search_module( provider_name )
	
	# ----- File Search Store Utilities
	def call_file_search_method( method_names: List[ str ],
		kwargs: Optional[ Dict[ str, Any ] ] = None ) -> Any:
		"""Call file search method.
		
		Purpose:
		    Calls the first compatible File Search Store wrapper method using the supplied
		    provider arguments. The function resolves supported argument aliases before invoking
		    the provider method.
		
		Args:
		    method_names (List[str]): Ordered provider method names to evaluate.
		    kwargs (Optional[Dict[str, Any]]): Candidate arguments for the provider method.
		
		Returns:
		    Any: Result returned by the compatible provider method.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			import inspect
			
			throw_if( 'method_names', method_names )
			candidate_kwargs = kwargs or { }
			
			for method_name in method_names:
				method = getattr( searcher, method_name, None )
				
				if not callable( method ):
					continue
				
				signature = inspect.signature( method )
				parameters = signature.parameters
				accepted_names = set( parameters.keys( ) )
				accepts_kwargs = any(
					parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in
					parameters.values( ) )
				
				clean_kwargs = { key: value for key, value in candidate_kwargs.items( ) if
					value is not None and value != '' and value != [ ] }
				
				if 'store_id' in accepted_names and not clean_kwargs.get( 'store_id' ):
					if clean_kwargs.get( 'id' ):
						clean_kwargs[ 'store_id' ] = clean_kwargs[ 'id' ]
					elif clean_kwargs.get( 'name' ) and method_name not in [ 'create',
						'create_store', ]:
						clean_kwargs[ 'store_id' ] = clean_kwargs[ 'name' ]
				
				if 'path' in accepted_names and not clean_kwargs.get( 'path' ):
					if clean_kwargs.get( 'file_path' ):
						clean_kwargs[ 'path' ] = clean_kwargs[ 'file_path' ]
				
				if 'name' in accepted_names and not clean_kwargs.get( 'name' ):
					if clean_kwargs.get( 'display_name' ):
						clean_kwargs[ 'name' ] = clean_kwargs[ 'display_name' ]
					elif clean_kwargs.get( 'store_id' ) and method_name in [ 'create',
						'create_store', ]:
						clean_kwargs[ 'name' ] = clean_kwargs[ 'store_id' ]
				
				if accepts_kwargs:
					return method( **clean_kwargs )
				
				method_kwargs = { key: value for key, value in clean_kwargs.items( ) if
					key in accepted_names }
				
				required_names = [ name for name, parameter in parameters.items( ) if
					parameter.default == inspect.Parameter.empty and parameter.kind in [
						inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
					] ]
				
				missing_names = [ name for name in required_names if name not in method_kwargs ]
				
				if missing_names:
					continue
				
				return method( **method_kwargs )
			
			raise AttributeError( f'Gemini FileSearch does not expose any compatible method from: '
			                      f'{", ".join( method_names )}.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'app'
			exception.cause = 'File Search Stores'
			exception.method = ('call_file_search_method( method_names: List[ str ], '
			                    'kwargs: Optional[ Dict[ str, Any ] ] = None ) -> Any')
			Logger( ).write( exception )
			raise exception
	
	def clear_filestore_outputs( ) -> None:
		"""Clear filestore outputs.
		
		Purpose:
		    Removes File Search Store operation results without changing store selections,
		    request settings, uploaded-file controls, queries, or system instructions.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		st.session_state[ 'filestore_results' ] = None
		st.session_state[ 'filestore_metadata' ] = { }
		st.session_state[ 'filestore_upload_result' ] = { }
	
	def clear_filestore_instructions( ) -> None:
		"""Clear File Search Stores instructions.
		
		Purpose:
		    Clears the File Search Stores system-instruction text and selected prompt template
		    without changing the selected prompt category, store selection, provider results,
		    or uploaded files.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		st.session_state[ 'filestore_system_instructions' ] = ''
		st.session_state[ 'filestore_prompt_id' ] = None
	
	def convert_filestore_system_instructions( ) -> None:
		"""Convert File Search Stores system instructions.
		
		Purpose:
		    Converts the active File Search Stores system instructions between supported XML-style
		    instruction blocks and Markdown headings.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		text_value = st.session_state.get( 'filestore_system_instructions', '' )
		
		if not isinstance( text_value, str ) or not text_value.strip( ):
			return
		
		source = text_value.strip( )
		
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'filestore_system_instructions' ] = converted
	
	def load_filestore_instruction_template( ) -> None:
		"""Load File Search Stores instruction template.
		
		Purpose:
		    Loads the selected File Search Stores prompt template into the mode-specific
		    system-instruction field using its stable prompt identifier.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			load_prompt_template( prompt_id_key='filestore_prompt_id',
				instructions_key='filestore_system_instructions', )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'File Search Stores Mode'
			ex.method = 'load_filestore_instruction_template( ) -> None'
			Logger( ).write( ex )
			raise ex
	
	def reset_filestore_management_settings( ) -> None:
		"""Reset File Search Store management settings.
		
		Purpose:
		    Returns the File Search Store management controls to their initial values without
		    clearing provider resources, operation results, request settings, or instructions.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		for key in [ 'filestore_name', 'filestore_selected_id', 'filestore_selected_label',
			'filestore_manual_id', ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_filestore_request_settings( ) -> None:
		"""Reset File Search Store request settings.
		
		Purpose:
		    Returns the File Search Store model, inference, and response controls to their initial
		    values without clearing store selections, uploads, queries, results, or instructions.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		    value."""
		for key in [ 'filestore_model', 'filestore_temperature', 'filestore_top_percent',
			'filestore_max_tokens', 'filestore_frequency_penalty', 'filestore_presence_penalty',
			'filestore_response_format', 'filestore_tool_choice', 'filestore_reasoning',
			'filestore_store', 'filestore_stream', 'filestore_background', ]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	# ------------------------------------------------------------------
	# Mode State
	# ------------------------------------------------------------------
	for key, default_value in { 'filestore_table': [ ], 'filestore_metadata': { },
		'filestore_upload_result': { }, }.items( ):
		if key not in st.session_state or not isinstance( st.session_state.get( key ),
				type( default_value ) ):
			st.session_state[ key ] = default_value
	
	if 'filestore_name' not in st.session_state:
		st.session_state[ 'filestore_name' ] = ''
	
	if 'filestore_manual_id' not in st.session_state:
		st.session_state[ 'filestore_manual_id' ] = ''
	
	# ------------------------------------------------------------------
	# Provider Options
	# ------------------------------------------------------------------
	model_options = getattr( searcher, 'model_options', [ ] ) or [ ]
	if callable( model_options ):
		model_options = model_options( )
	
	model_options = [ str( option ) for option in model_options if str( option ).strip( ) ]
	format_options = getattr( searcher, 'format_options', [ ] ) or [ ]
	if callable( format_options ):
		format_options = format_options( )
	
	format_options = [ str( option ) for option in format_options if str( option ).strip( ) ]
	choice_options = getattr( searcher, 'choice_options', [ ] ) or [ ]
	if callable( choice_options ):
		choice_options = choice_options( )
	
	choice_options = [ str( option ) for option in choice_options if str( option ).strip( ) ]
	reasoning_options = getattr( searcher, 'reasoning_options', [ ] ) or [ ]
	if callable( reasoning_options ):
		reasoning_options = reasoning_options( )
	
	reasoning_options = [ str( option ) for option in reasoning_options if str( option ).strip( ) ]
	
	# ------------------------------------------------------------------
	# Store Selection
	# ------------------------------------------------------------------
	store_options = [ str( row.get( 'id', '' ) ) for row in
		st.session_state.get( 'filestore_table', [ ] ) if
		isinstance( row, dict ) and row.get( 'id' ) ]
	
	if st.session_state.get( 'filestore_selected_id' ) not in store_options:
		st.session_state[ 'filestore_selected_id' ] = None
	
	st.session_state[ 'filestore_selected_label' ] = (
			st.session_state.get( 'filestore_selected_id' ) or '')
	
	selected_store_id = (st.session_state.get( 'filestore_manual_id', '' ) or st.session_state.get(
		'filestore_selected_id', '' ))
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	
	with (center):
		st.subheader( '📦 File Search Stores', help=cfg.VECTORSTORES_API )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Expander — Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Expander — Store Management
			# ------------------------------------------------------------------
			with st.expander( label='Store Management', icon='📦', expanded=False,
					width='stretch' ):
				management_c1, management_c2, management_c3 = st.columns( [ 0.34, 0.33, 0.33 ],
					border=True, gap='xxsmall', )
				
				# ----- New Store Name -----
				with management_c1:
					st.text_input( label='New Store Name', key='filestore_name', width='stretch',
						help='Name used when creating a Gemini File Search Store.',
						placeholder='Enter store name', )
				
				# ----- Selected Store -----
				with management_c2:
					st.selectbox( label='Selected Store', options=store_options,
						key='filestore_selected_id', index=None,
						placeholder='Options' if store_options else 'No stores loaded',
						help='Select a File Search Store returned by List Stores.',
						disabled=not bool( store_options ), )
				
				# ----- Manual Store ID -----
				with management_c3:
					st.text_input( label='Manual Store ID', key='filestore_manual_id',
						width='stretch',
						help='Optional. Enter a File Search Store identifier directly.',
						placeholder='Enter store ID', )
				
				lifecycle_c1, lifecycle_c2, lifecycle_c3, lifecycle_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ] )
				
				# ----- Create Store -----
				with lifecycle_c1:
					if st.button( label='Create Store', key='filestore_create', icon='➕',
							width='stretch' ):
						try:
							name = st.session_state.get( 'filestore_name', '', ).strip( )
							
							if not name:
								st.warning( 'Enter a File Search Store name.' )
							else:
								result = call_file_search_method( [ 'create_store', 'create' ],
									{ 'name': name, 'display_name': name,
										'instruct': st.session_state.get(
											'filestore_system_instructions', '', ), }, )
								
								st.session_state[ 'filestore_metadata' ] = \
									normalize_storage_object( result )
								
								st.success( 'File Search Store created.' )
						except Exception as exc:
							st.error( f'Create failed: {exc}' )
				
				# ----- List Stores -----
				with lifecycle_c2:
					if st.button( label='List Stores', key='filestore_list', icon='📋',
							width='stretch' ):
						try:
							result = call_file_search_method( [ 'list_stores', 'list' ] )
							
							rows = [ normalize_storage_object( item ) for item in
								(getattr( result, 'data', result ) or [ ]) ]
							
							st.session_state[ 'filestore_table' ] = rows
						except Exception as exc:
							st.error( f'List failed: {exc}' )
				
				# ----- Retrieve Store -----
				with lifecycle_c3:
					if st.button( label='Retrieve Store', key='filestore_retrieve', icon='🔎',
							width='stretch', disabled=not bool( selected_store_id ) ):
						try:
							result = call_file_search_method(
								[ 'retrieve_store', 'retrieve', 'get' ],
								{ 'store_id': selected_store_id, 'id': selected_store_id,
									'name': selected_store_id, }, )
							
							st.session_state[ 'filestore_metadata' ] = normalize_storage_object(
								result )
						except Exception as exc:
							st.error( f'Retrieve failed: {exc}' )
				
				# ----- Delete Store -----
				with lifecycle_c4:
					if st.button( label='Delete Store', key='filestore_delete', icon='🗑️',
							width='stretch', disabled=not bool( selected_store_id ) ):
						try:
							call_file_search_method( [ 'delete_store', 'delete' ],
								{ 'store_id': selected_store_id, 'id': selected_store_id,
									'name': selected_store_id, }, )
							
							clear_filestore_outputs( )
							st.success( 'File Search Store deleted.' )
						except Exception as exc:
							st.error( f'Delete failed: {exc}' )
				
				# ----- Store Table -----
				if st.session_state.get( 'filestore_table' ):
					df_filestores = pd.DataFrame( st.session_state[ 'filestore_table' ] )
					
					st.dataframe( df_filestores, width='stretch', hide_index=True, )
				
				# ----- Store Metadata -----
				if st.session_state.get( 'filestore_metadata' ):
					with st.expander( label='Store Metadata', icon='📄', expanded=False,
							width='stretch' ):
						render_storage_metadata( st.session_state[ 'filestore_metadata' ] )
				
				# ----- Reset Button -----
				st.button( label='Reset', key='filestore_management_reset', width='stretch',
					icon='🔄', on_click=reset_filestore_management_settings, )
			
			# ------------------------------------------------------------------
			# Expander — Request Settings
			# ------------------------------------------------------------------
			with st.expander( label='Request Settings', icon='⚙️', expanded=False, width='stretch' ):
				request_c1, request_c2, request_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True,
					gap='xxsmall', )
				
				# ----- Model -----
				with request_c1:
					st.selectbox( label='Model', options=model_options, index=None,
						key='filestore_model', placeholder='Options',
						help='Optional model used for File Search Store queries.', )
				
				# ----- Max Tokens -----
				with request_c2:
					st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
						key='filestore_max_tokens', help=cfg.MAX_OUTPUT_TOKENS, )
				
				# ----- Response Format -----
				with request_c3:
					st.selectbox( label='Response Format', options=format_options, index=None,
						key='filestore_response_format', placeholder='Options',
						help='Optional response format when supported.', )
				
				request_c4, request_c5, request_c6, request_c7 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall', )
				
				# ----- Temperature -----
				with request_c4:
					st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.01,
						key='filestore_temperature', help=cfg.TEMPERATURE, )
				
				# ----- Top-P -----
				with request_c5:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0, step=0.01,
						key='filestore_top_percent', help=cfg.TOP_P, )
				
				# ----- Frequency Penalty -----
				with request_c6:
					st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='filestore_frequency_penalty', help=cfg.FREQUENCY_PENALTY, )
				
				# ----- Presence Penalty -----
				with request_c7:
					st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='filestore_presence_penalty', help=cfg.PRESENCE_PENALTY, )
				
				request_c8, request_c9, request_c10 = st.columns( [ 0.34, 0.33, 0.33 ],
					border=True, gap='xxsmall', )
				
				# ----- Tool Choice -----
				with request_c8:
					st.selectbox( label='Tool Choice', options=choice_options, index=None,
						key='filestore_tool_choice', placeholder='Options', help=cfg.CHOICE, )
				
				# ----- Reasoning -----
				with request_c9:
					st.selectbox( label='Reasoning', options=reasoning_options, index=None,
						key='filestore_reasoning', placeholder='Options', help=cfg.REASONING, )
				
				# ----- Store -----
				with request_c10:
					st.toggle( label='Store', key='filestore_store', help=cfg.STORE, )
				
				request_c11, request_c12 = st.columns( [ 0.50, 0.50 ], border=True,
					gap='xxsmall', )
				
				# ----- Stream -----
				with request_c11:
					st.toggle( label='Stream', key='filestore_stream', help=cfg.STREAM, )
				
				# ----- Background -----
				with request_c12:
					st.toggle( label='Background', key='filestore_background',
						help=cfg.BACKGROUND_MODE, )
				
				# ----- Reset Button -----
				st.button( label='Reset', key='filestore_request_settings_reset', width='stretch',
					icon='🔄', on_click=reset_filestore_request_settings, )
	
		# ------------------------------------------------------------------
		# Expander — File Uploader
		# ------------------------------------------------------------------
		with st.expander( label='File Uploader', icon='📤', expanded=False, width='stretch' ):
			uploaded_file = st.file_uploader( label='Upload File to File Search Store',
				key='filestore_input',
				help='Upload a file to the selected Gemini File Search Store.', )
			
			if st.button( label='Upload to Store', key='filestore_upload', icon='⬆️',
					width='stretch',
					disabled=uploaded_file is None or not bool( selected_store_id ) ):
				try:
					file_path = save_uploaded_storage_file( uploaded_file )
					
					result = call_file_search_method( [ 'upload_file', 'upload', 'add_file' ],
						{ 'store_id': selected_store_id, 'id': selected_store_id,
							'file_path': file_path, 'path': file_path,
							'instruct': st.session_state.get( 'filestore_system_instructions',
								'', ), }, )
					
					st.session_state[ 'filestore_upload_result' ] = normalize_storage_object(
						result )
					
					st.success( 'File uploaded to the File Search Store.' )
				except Exception as exc:
					st.error( f'Upload failed: {exc}' )
			
			if st.session_state.get( 'filestore_upload_result' ):
				with st.expander( label='Upload Result', icon='📄', expanded=False,
						width='stretch' ):
					render_storage_metadata( st.session_state[ 'filestore_upload_result' ] )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			
			filestore_prompt_categories = fetch_prompt_categories( 'File Search Stores' )
			current_filestore_category = st.session_state.get( 'filestore_prompt_category' )
			if current_filestore_category not in filestore_prompt_categories:
				st.session_state[ 'filestore_prompt_category' ] = None
			
			selected_filestore_category = st.session_state.get( 'filestore_prompt_category' )
			filestore_prompt_options = fetch_prompt_options(
				selected_filestore_category ) if selected_filestore_category else [ ]
			
			filestore_prompt_ids = [ int( option[ 'ID' ] ) for option in filestore_prompt_options ]
			if st.session_state.get( 'filestore_prompt_id' ) not in filestore_prompt_ids:
				st.session_state[ 'filestore_prompt_id' ] = None
			
			# ----- Instruction Text -----
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					key='filestore_system_instructions',
					help=getattr( cfg, 'SYSTEM_INSTRUCTIONS', '' ), )
			
			# ----- Template Selection -----
			with in_right:
				st.selectbox( label='Category', options=filestore_prompt_categories, index=None,
					key='filestore_prompt_category', placeholder='Select Category',
					help=('Limits prompt templates to categories associated with '
					      'file-search, retrieval, and corpus-management workflows.'),
					on_change=reset_prompt_template_selection, args=('filestore_prompt_id',), )
				
				st.selectbox( label='Use Template', options=filestore_prompt_ids, index=None,
					key='filestore_prompt_id', placeholder='Select Template',
					disabled=not filestore_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						filestore_prompt_options, ),
					help=('Loads the selected prompt into the File Search Stores '
					      'system-instruction field.'),
					on_change=load_filestore_instruction_template, )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ----- Clear Instructions -----
			with btn_c1:
				st.button( label='Clear Instructions', key='clear_filestore_instructions',
					width='stretch', on_click=clear_filestore_instructions, icon='🧹', )
			
			# ----- Convert Instructions -----
			with btn_c2:
				st.button( label='XML ↔️ Markdown', key='convert_filestore_instructions',
					width='stretch', on_click=convert_filestore_system_instructions, )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Expander — Vector Store Query
		# ------------------------------------------------------------------
		st.text_area( label='Query', key='filestore_query', height=120,
			help='Search the selected Gemini File Search Store.', )
		
		query_c1, query_c2 = st.columns( 2 )
		
		# ------ Search -----
		with query_c1:
			if st.button( label='Search Store', key='filestore_search', icon='🔍', width='stretch',
					disabled=not bool( selected_store_id ) or not bool(
						st.session_state.get( 'filestore_query', '', ).strip( ) ) ):
				try:
					query = st.session_state.get( 'filestore_query', '', ).strip( )
					
					result = call_file_search_method( [ 'search', 'query', 'generate_text' ],
						{ 'store_id': selected_store_id, 'id': selected_store_id, 'query': query,
							'prompt': query,
							'model': st.session_state.get( 'filestore_model', '', ),
							'temperature': st.session_state.get( 'filestore_temperature', 0.0, ),
							'top_p': st.session_state.get( 'filestore_top_percent', 0.0, ),
							'max_tokens': st.session_state.get( 'filestore_max_tokens', 0, ),
							'frequency': st.session_state.get( 'filestore_frequency_penalty',
								0.0, ),
							'presence': st.session_state.get( 'filestore_presence_penalty', 0.0, ),
							'response_format': st.session_state.get( 'filestore_response_format',
								'', ),
							'tool_choice': st.session_state.get( 'filestore_tool_choice', '', ),
							'reasoning': st.session_state.get( 'filestore_reasoning', '', ),
							'background': st.session_state.get( 'filestore_background', False, ),
							'store': st.session_state.get( 'filestore_store', False, ),
							'stream': st.session_state.get( 'filestore_stream', False, ),
							'instruct': st.session_state.get( 'filestore_system_instructions',
								'', ), }, )
					
					st.session_state[ 'filestore_results' ] = result
				except Exception as exc:
					st.error( f'Search failed: {exc}' )
		
		# ----- Clear Outputs -----
		with query_c2:
			st.button( label='Clear Outputs', key='filestore_clear_outputs', icon='🧹',
				width='stretch', on_click=clear_filestore_outputs, )
		
		# ------------------------------------------------------------------
		# Expander — Search Results
		# ------------------------------------------------------------------
		if st.session_state.get( 'filestore_results' ) is not None:
			with st.expander( label='Search Results', icon='🔎', expanded=True, width='stretch' ):
				result = st.session_state[ 'filestore_results' ]
				
				if isinstance( result, str ):
					st.markdown( result )
				else:
					st.json( normalize( result ) )

# ======================================================================================
# GOOGLE CLOUD BUCKETS MODE
# ======================================================================================
elif mode == 'Google Cloud Buckets':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	if provider_name != 'Gemini':
		st.warning( 'Google Cloud Buckets are available for Gemini / Google Cloud only.' )
		st.stop( )
	
	buckets = get_cloud_buckets_module( provider_name )
	
	# ----- Google Cloud Bucket Utilities -----
	
	def call_bucket_method( method_names: List[ str ],
		kwargs: Optional[ Dict[ str, Any ] ] = None ) -> Any:
		"""Call bucket method.
		
		Purpose:
		    Calls the first compatible Google Cloud Bucket wrapper method using the supplied
		    provider arguments. The function resolves supported argument aliases before invoking
		    the provider method.
		
		Args:
		    method_names (List[str]): Ordered provider method names to evaluate.
		    kwargs (Optional[Dict[str, Any]]): Candidate arguments for the provider method.
		
		Returns:
		    Any: Result returned by the compatible provider method.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			import inspect
			
			throw_if( 'method_names', method_names )
			candidate_kwargs = kwargs or { }
			
			for method_name in method_names:
				method = getattr( buckets, method_name, None )
				
				if not callable( method ):
					continue
				
				signature = inspect.signature( method )
				parameters = signature.parameters
				accepted_names = set( parameters.keys( ) )
				accepts_kwargs = any(
					parameter.kind == inspect.Parameter.VAR_KEYWORD
					for parameter in parameters.values( )
				)
				
				clean_kwargs = {
					key: value
					for key, value in candidate_kwargs.items( )
					if value is not None and value != '' and value != [ ]
				}
				
				if 'bucket' in accepted_names and not clean_kwargs.get( 'bucket' ):
					if clean_kwargs.get( 'bucket_name' ):
						clean_kwargs[ 'bucket' ] = clean_kwargs[ 'bucket_name' ]
					elif clean_kwargs.get( 'store_id' ):
						clean_kwargs[ 'bucket' ] = clean_kwargs[ 'store_id' ]
					elif clean_kwargs.get( 'id' ):
						clean_kwargs[ 'bucket' ] = clean_kwargs[ 'id' ]
				
				if 'name' in accepted_names and not clean_kwargs.get( 'name' ):
					if method_name in [ 'create', 'create_bucket' ] and clean_kwargs.get(
							'bucket' ):
						clean_kwargs[ 'name' ] = clean_kwargs[ 'bucket' ]
					elif clean_kwargs.get( 'object_name' ):
						clean_kwargs[ 'name' ] = clean_kwargs[ 'object_name' ]
					elif clean_kwargs.get( 'file_name' ):
						clean_kwargs[ 'name' ] = clean_kwargs[ 'file_name' ]
					elif clean_kwargs.get( 'display_name' ):
						clean_kwargs[ 'name' ] = clean_kwargs[ 'display_name' ]
				
				if 'path' in accepted_names and not clean_kwargs.get( 'path' ):
					if clean_kwargs.get( 'file_path' ):
						clean_kwargs[ 'path' ] = clean_kwargs[ 'file_path' ]
				
				if accepts_kwargs:
					return method( **clean_kwargs )
				
				method_kwargs = {
					key: value
					for key, value in clean_kwargs.items( )
					if key in accepted_names
				}
				
				required_names = [
					name
					for name, parameter in parameters.items( )
					if parameter.default == inspect.Parameter.empty
					and parameter.kind in [
						inspect.Parameter.POSITIONAL_OR_KEYWORD,
						inspect.Parameter.KEYWORD_ONLY,
					]
				]
				
				missing_names = [
					name
					for name in required_names
					if name not in method_kwargs
				]
				
				if missing_names:
					continue
				
				return method( **method_kwargs )
			
			raise AttributeError(
				f'Gemini CloudBuckets does not expose any compatible method from: '
				f'{", ".join( method_names )}.'
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'app'
			exception.cause = 'Google Cloud Buckets'
			exception.method = (
				'call_bucket_method( method_names: List[ str ], '
				'kwargs: Optional[ Dict[ str, Any ] ] = None ) -> Any'
			)
			Logger( ).write( exception )
			raise exception
	
	def clear_bucket_outputs( ) -> None:
		"""Clear bucket outputs.
		
		Purpose:
		    Clears Google Cloud Bucket operation results without changing bucket selections,
		    request settings, uploaded files, queries, or system instructions.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		st.session_state[ 'bucket_results' ] = None
		st.session_state[ 'bucket_metadata' ] = { }
		st.session_state[ 'bucket_upload_result' ] = { }
	
	def clear_bucket_instructions( ) -> None:
		"""Clear Google Cloud Buckets instructions.
		
		Purpose:
		    Clears the Google Cloud Buckets system-instruction text and selected prompt template
		    without changing the selected category, bucket configuration, results, or uploaded
		    files.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		st.session_state[ 'bucket_system_instructions' ] = ''
		st.session_state[ 'bucket_prompt_id' ] = None
	
	def convert_bucket_system_instructions( ) -> None:
		"""Convert Google Cloud Buckets system instructions.
		
		Purpose:
		    Converts the active Google Cloud Buckets system instructions between supported
		    XML-style instruction blocks and Markdown headings.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		text_value = st.session_state.get( 'bucket_system_instructions', '' )
		
		if not isinstance( text_value, str ) or not text_value.strip( ):
			return
		
		source = text_value.strip( )
		
		if cfg.XML_BLOCK_PATTERN.search( source ):
			converted = convert_xml( source )
		else:
			converted = convert_markdown( source )
		
		st.session_state[ 'bucket_system_instructions' ] = converted
	
	def load_bucket_instruction_template( ) -> None:
		"""Load Google Cloud Buckets instruction template.
		
		Purpose:
		    Loads the selected Google Cloud Buckets prompt template into the mode-specific
		    system-instruction field using its stable prompt identifier.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			load_prompt_template(
				prompt_id_key='bucket_prompt_id',
				instructions_key='bucket_system_instructions',
			)
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Google Cloud Buckets Mode'
			ex.method = 'load_bucket_instruction_template( ) -> None'
			Logger( ).write( ex )
			raise ex
	
	def reset_bucket_management_settings( ) -> None:
		"""Reset bucket management settings.
		
		Purpose:
		    Returns the Google Cloud Bucket management controls to their initial values without
		    clearing provider resources, operation results, request settings, or instructions.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		for key in [
			'bucket_name',
			'bucket_select',
			'bucket_selected_id',
			'bucket_selected_label',
			'bucket_manual_id',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	def reset_bucket_request_settings( ) -> None:
		"""Reset bucket request settings.
		
		Purpose:
		    Returns the Google Cloud Bucket model, inference, and response controls to their
		    initial values without clearing bucket selections, queries, results, or instructions.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		for key in [
			'bucket_model',
			'bucket_number',
			'bucket_temperature',
			'bucket_top_percent',
			'bucket_max_tokens',
			'bucket_frequency_penalty',
			'bucket_presence_penalty',
			'bucket_response_format',
			'bucket_tool_choice',
			'bucket_reasoning',
			'bucket_store',
			'bucket_stream',
			'bucket_background',
		]:
			if key in st.session_state:
				del st.session_state[ key ]
	
	# ------------------------------------------------------------------
	# Mode State
	# ------------------------------------------------------------------
	for key, default_value in {
		'bucket_table': [ ],
		'bucket_metadata': { },
		'bucket_upload_result': { }, }.items( ):
		if key not in st.session_state or not isinstance(
				st.session_state.get( key ),
				type( default_value ) ):
			st.session_state[ key ] = default_value
	
	if 'bucket_name' not in st.session_state:
		st.session_state[ 'bucket_name' ] = ''
	
	if 'bucket_manual_id' not in st.session_state:
		st.session_state[ 'bucket_manual_id' ] = ''
	
	# ------------------------------------------------------------------
	# Provider Options
	# ------------------------------------------------------------------
	model_options = getattr( buckets, 'model_options', [ ] ) or [ ]
	if callable( model_options ):
		model_options = model_options( )
	
	model_options = [ str( option )
		for option in model_options
		if str( option ).strip( ) ]
	
	format_options = getattr( buckets, 'format_options', [ ] ) or [ ]
	
	if callable( format_options ):
		format_options = format_options( )
	
	format_options = [
		str( option )
		for option in format_options
		if str( option ).strip( ) ]
	
	choice_options = getattr( buckets, 'choice_options', [ ] ) or [ ]
	if callable( choice_options ):
		choice_options = choice_options( )
	
	choice_options = [
		str( option )
		for option in choice_options
		if str( option ).strip( ) ]
	reasoning_options = getattr( buckets, 'reasoning_options', [ ] ) or [ ]
	if callable( reasoning_options ):
		reasoning_options = reasoning_options( )
	
	reasoning_options = [
		str( option )
		for option in reasoning_options
		if str( option ).strip( )
	]
	
	collections = getattr( buckets, 'collections', None )
	bucket_options = list( collections.items( ) ) if isinstance( collections, dict ) else [ ]
	bucket_option_labels = [
		f'{name} — {bucket_id}'
		for name, bucket_id in bucket_options
	]
	
	if st.session_state.get( 'bucket_select' ) not in bucket_option_labels:
		st.session_state[ 'bucket_select' ] = None
	
	selected_bucket_label = st.session_state.get( 'bucket_select' )
	selected_bucket_id = ''
	
	for bucket_name, bucket_id in bucket_options:
		if f'{bucket_name} — {bucket_id}' == selected_bucket_label:
			selected_bucket_id = str( bucket_id )
			break
	
	st.session_state[ 'bucket_selected_id' ] = selected_bucket_id
	st.session_state[ 'bucket_selected_label' ] = selected_bucket_label or ''
	target_bucket = (st.session_state.get( 'bucket_manual_id', '' ) or st.session_state.get(
		'bucket_selected_id', '' ))
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	
	with center:
		st.subheader( '🧊 Google Cloud Buckets', help=getattr( cfg, 'VECTORSTORES_API', '' ), )
		st.divider( )
		project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		location = cfg.GOOGLE_CLOUD_LOCATION
		st.caption( f'Project: {project_id or "Not configured"} | '
		            f'Location: {location or "Not configured"}' )
		
		# ------------------------------------------------------------------
		# Expander — Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Expander — Bucket Management
			# ------------------------------------------------------------------
			with st.expander( label='Bucket Management', icon='🪣', expanded=False, width='stretch' ):
				management_c1, management_c2, management_c3 = st.columns( [ 0.34, 0.33, 0.33 ],
					border=True, gap='xxsmall', )
				
				# ----- New Bucket Name -----
				with management_c1:
					st.text_input( label='New Bucket Name', key='bucket_name', width='stretch',
						help='Name used when creating a new Google Cloud Bucket.',
						placeholder='Enter bucket name', )
				
				# ----- Configured Bucket -----
				with management_c2:
					if bucket_option_labels:
						st.selectbox( label='Configured Bucket', options=bucket_option_labels,
							key='bucket_select', index=None, placeholder='Options',
							help='Select a Google Cloud Bucket exposed by the provider wrapper.', )
					else:
						st.selectbox( label='Configured Bucket', options=[ ], key='bucket_select',
							index=None, placeholder='No configured buckets',
							help='No Google Cloud Bucket collections are exposed by the wrapper.',
							disabled=True, )
				
				# ----- Manual Bucket ID -----
				with management_c3:
					st.text_input( label='Manual Bucket ID / Name', key='bucket_manual_id',
						width='stretch',
						help='Optional. Enter a bucket identifier or name directly.',
						placeholder='Enter bucket ID or name', )
				
				lifecycle_c1, lifecycle_c2, lifecycle_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
				
				# ----- Create Bucket -----
				with lifecycle_c1:
					if st.button( label='Create Bucket', key='create_bucket', width='stretch',
							icon='➕' ):
						with st.spinner( 'Creating cloud bucket…' ):
							try:
								name = st.session_state.get( 'bucket_name', '' ).strip( )
								
								if not name:
									st.warning( 'Enter a Cloud Bucket name.' )
								else:
									result = call_bucket_method( [ 'create', 'create_bucket' ],
										{ 'name': name, 'bucket_name': name, 'project_id': project_id,
											'location': location, }, )
									
									st.session_state[ 'bucket_metadata' ] = normalize_storage_object(
										result )
									
									st.success( f'Created Cloud Bucket: {name}' )
							except Exception as exc:
								err = Error( exc )
								st.error( f'Create bucket failed: {err.info}' )
				
				# ----- Retrieve Bucket -----
				with lifecycle_c2:
					if st.button( label='Retrieve Bucket', key='retrieve_bucket', width='stretch',
							icon='🔎' ):
						with st.spinner( 'Retrieving cloud bucket…' ):
							try:
								if not target_bucket:
									st.warning( 'Select or enter a Cloud Bucket ID.' )
								else:
									result = call_bucket_method(
										[ 'retrieve', 'retrieve_bucket', 'get' ],
										{ 'store_id': target_bucket, 'id': target_bucket,
											'name': target_bucket, 'bucket_name': target_bucket,
											'project_id': project_id, 'location': location, }, )
									
									st.session_state[ 'bucket_metadata' ] = normalize_storage_object(
										result )
									
									st.success( 'Cloud Bucket metadata retrieved.' )
							except Exception as exc:
								err = Error( exc )
								st.error( f'Retrieve bucket failed: {err.info}' )
				
				# ----- Delete Bucket -----
				with lifecycle_c3:
					if st.button( label='Delete Bucket', key='delete_bucket', width='stretch',
							icon='❌' ):
						with st.spinner( 'Deleting cloud bucket…' ):
							try:
								if not target_bucket:
									st.warning( 'Select or enter a Cloud Bucket ID.' )
								else:
									result = call_bucket_method(
										[ 'delete', 'delete_bucket', 'remove' ],
										{ 'store_id': target_bucket, 'id': target_bucket,
											'name': target_bucket, 'bucket_name': target_bucket,
											'project_id': project_id, 'location': location, }, )
									
									st.session_state[ 'bucket_metadata' ] = normalize_storage_object(
										result )
									
									st.success( 'Delete request completed.' )
							except Exception as exc:
								err = Error( exc )
								st.error( f'Delete bucket failed: {err.info}' )
				
				# ----- Reset Button -----
				st.button( label='Reset', key='bucket_management_reset', width='stretch',
					icon='🔄', on_click=reset_bucket_management_settings, )
			
			# ------------------------------------------------------------------
			# File Upload
			# ------------------------------------------------------------------
			with st.expander( label='File Uploader', expanded=False, icon='📤' ):
				# ----- File Uploader -----
				uploaded_file = st.file_uploader( label='Upload File to Cloud Bucket',
					type=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', 'json', 'csv', ],
					key='bucket_uploader',
					help='Select a local file to upload to the active Google Cloud Bucket.', )
			
				# ----- Upload Button -----
				if st.button( label='Upload File', key='upload_bucket_file', width='stretch',
						icon='⬆️' ):
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
									{ 'path': path, 'file_path': path, 'bucket_name': target_bucket,
										'store_id': target_bucket, 'id': target_bucket,
										'project_id': project_id, 'location': location, }, )
								
								st.session_state[ 'bucket_upload_result' ] = normalize_storage_object(
									result )
								
								st.success( 'Upload request completed.' )
						except Exception as exc:
							err = Error( exc )
							st.error( f'Bucket upload failed: {err.info}' )
		
			# ------------------------------------------------------------------
			# Expander — Request Settings
			# ------------------------------------------------------------------
			with st.expander( label='Request Settings', icon='⚙️', expanded=False,
					width='stretch' ):
				request_c1, request_c2, request_c3, request_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall', )
				
				# ----- Model -----
				with request_c1:
					st.selectbox( label='Model', options=model_options, index=None,
						key='bucket_model', placeholder='Options',
						help='Optional model used for bucket-backed queries.', )
				
				# ----- Responses -----
				with request_c2:
					st.slider( label='Responses', min_value=0, max_value=8, step=1,
						key='bucket_number',
						help='Optional number of candidate responses when supported.', )
				
				# ----- Max Tokens -----
				with request_c3:
					st.slider( label='Max Tokens', min_value=0, max_value=100000, step=500,
						key='bucket_max_tokens', help=cfg.MAX_OUTPUT_TOKENS, )
				
				# ----- Response Format -----
				with request_c4:
					st.selectbox( label='Response Format', options=format_options, index=None,
						key='bucket_response_format', placeholder='Options',
						help='Optional response format when supported.', )
				
				request_c5, request_c6, request_c7, request_c8 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall', )
				
				# ----- Temperature -----
				with request_c5:
					st.slider( label='Temperature', min_value=0.0, max_value=2.0, step=0.01,
						key='bucket_temperature', help=cfg.TEMPERATURE, )
				
				# ----- Top-P -----
				with request_c6:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0, step=0.01,
						key='bucket_top_percent', help=cfg.TOP_P, )
				
				# ----- Frequency Penalty -----
				with request_c7:
					st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='bucket_frequency_penalty', help=cfg.FREQUENCY_PENALTY, )
				
				# ----- Presence Penalty -----
				with request_c8:
					st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0, step=0.01,
						key='bucket_presence_penalty', help=cfg.PRESENCE_PENALTY, )
				
				request_c9, request_c10, request_c11, request_c12 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall', )
				
				# ----- Tool Choice -----
				with request_c9:
					st.selectbox( label='Tool Choice', options=choice_options, index=None,
						key='bucket_tool_choice', placeholder='Options', help=cfg.CHOICE, )
				
				# ----- Reasoning -----
				with request_c10:
					st.selectbox( label='Reasoning', options=reasoning_options, index=None,
						key='bucket_reasoning', placeholder='Options', help=cfg.REASONING, )
				
				# ----- Store -----
				with request_c11:
					st.toggle( label='Store', key='bucket_store', help=cfg.STORE, )
				
				# ----- Stream -----
				with request_c12:
					st.toggle( label='Stream', key='bucket_stream', help=cfg.STREAM, )
				
				# ----- Background -----
				st.toggle( label='Background', key='bucket_background', help=cfg.BACKGROUND_MODE, )
				
				st.button( label='Reset', key='bucket_request_settings_reset', width='stretch',
					icon='🔄', on_click=reset_bucket_request_settings, )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			bucket_prompt_categories = fetch_prompt_categories( 'Cloud Buckets' )
			current_bucket_category = st.session_state.get( 'bucket_prompt_category' )
			if current_bucket_category not in bucket_prompt_categories:
				st.session_state[ 'bucket_prompt_category' ] = None
			
			selected_bucket_category = st.session_state.get( 'bucket_prompt_category' )
			bucket_prompt_options = fetch_prompt_options(
				selected_bucket_category ) if selected_bucket_category else [ ]
			
			bucket_prompt_ids = [ int( option[ 'ID' ] ) for option in bucket_prompt_options ]
			if st.session_state.get( 'bucket_prompt_id' ) not in bucket_prompt_ids:
				st.session_state[ 'bucket_prompt_id' ] = None
			
			# ----- Instruction Text -----
			with in_left:
				st.text_area( label='Enter Text', height=140, width='stretch',
					key='bucket_system_instructions', help=cfg.SYSTEM_INSTRUCTIONS, )
			
			# ----- Template Selection -----
			with in_right:
				st.selectbox( label='Category', options=bucket_prompt_categories, index=None,
					key='bucket_prompt_category', placeholder='Select Category',
					help=('Limits prompt templates to categories associated with '
					      'cloud-storage, governance, and data-management workflows.'),
					on_change=reset_prompt_template_selection, args=('bucket_prompt_id',), )
				
				st.selectbox( label='Use Template', options=bucket_prompt_ids, index=None,
					key='bucket_prompt_id', placeholder='Select Template',
					disabled=not bucket_prompt_ids,
					format_func=lambda prompt_id: format_prompt_option( prompt_id,
						bucket_prompt_options, ),
					help=('Loads the selected prompt into the Google Cloud Buckets '
					      'system-instruction field.'),
					on_change=load_bucket_instruction_template, )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			# ----- Clear Button -----
			with btn_c1:
				st.button( label='Clear Instructions', key='clear_bucket_instructions',
					width='stretch', on_click=clear_bucket_instructions, icon='🧹', )
			
			# ----- Convert Button -----
			with btn_c2:
				st.button( label='XML ↔️ Markdown', key='convert_bucket_instructions',
					width='stretch', on_click=convert_bucket_system_instructions, )
	
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Bucket Query
		# ------------------------------------------------------------------
		st.text_area( label='Query', key='bucket_query', height=120,
			help=('Ask a question using content stored in the selected '
			      'Google Cloud Bucket.'), )
		
		qry_c1, qry_c2 = st.columns( 2 )
		
		with qry_c1:
			# ----- Query Button -----
			if st.button( label='Query Cloud Bucket', key='query_bucket', icon='🔍', width='stretch',
					disabled=not bool( target_bucket ) or not bool(
						st.session_state.get( 'bucket_query', '' ).strip( ) ) ):
				
				with st.spinner( 'Querying cloud bucket…' ):
					try:
						query = st.session_state.get( 'bucket_query', '', ).strip( )
						
						result = call_bucket_method( [ 'search', 'query', 'ask', 'generate_text' ],
							{ 'bucket': target_bucket, 'bucket_name': target_bucket,
								'store_id': target_bucket, 'id': target_bucket, 'query': query,
								'prompt': query,
								'model': st.session_state.get( 'bucket_model', '', ),
								'number': st.session_state.get( 'bucket_number', 0, ),
								'temperature': st.session_state.get( 'bucket_temperature', 0.0, ),
								'top_p': st.session_state.get( 'bucket_top_percent', 0.0, ),
								'max_tokens': st.session_state.get( 'bucket_max_tokens', 0, ),
								'frequency': st.session_state.get( 'bucket_frequency_penalty',
									0.0, ),
								'presence': st.session_state.get( 'bucket_presence_penalty',
									0.0, ),
								'response_format': st.session_state.get( 'bucket_response_format',
									'', ),
								'tool_choice': st.session_state.get( 'bucket_tool_choice', '', ),
								'reasoning': st.session_state.get( 'bucket_reasoning', '', ),
								'background': st.session_state.get( 'bucket_background', False, ),
								'store': st.session_state.get( 'bucket_store', False, ),
								'stream': st.session_state.get( 'bucket_stream', False, ),
								'instruct': st.session_state.get( 'bucket_system_instructions',
									'', ), 'project_id': project_id, 'location': location, }, )
						
						st.session_state[ 'bucket_results' ] = result
					except Exception as exc:
						err = Error( exc )
						st.error( f'Bucket query failed: {err.info}' )
			
		if st.session_state.get( 'bucket_results' ) is not None:
			with st.expander( label='Query Results', icon='📄', expanded=True, width='stretch' ):
				result = st.session_state[ 'bucket_results' ]
				
				if isinstance( result, str ):
					st.markdown( result )
				else:
					st.json( normalize( result ) )
	
		with qry_c2:
			# ----- Clear Outputs -----
			if st.button( label='Clear Outputs', key='clear_bucket_outputs', width='stretch',
					on_click=clear_bucket_outputs, icon='🧹' ):
				st.rerun( )

# ======================================================================================
# PROMPT ENGINEERING MODE
# ======================================================================================
elif mode == 'Prompt Engineering':
	import math
	
	TABLE = 'Prompts'
	PAGE_SIZE = 10
	
	# ------------------------------------------------------------------
	# Prompt Engineering State
	# ------------------------------------------------------------------
	st.session_state.setdefault( 'pe_page', 1 )
	st.session_state.setdefault( 'pe_search', '' )
	st.session_state.setdefault( 'pe_sort_col', 'ID' )
	st.session_state.setdefault( 'pe_sort_dir', 'ASC' )
	st.session_state.setdefault( 'pe_selected_id', None )
	st.session_state.setdefault( 'pe_caption', '' )
	st.session_state.setdefault( 'pe_name', '' )
	st.session_state.setdefault( 'pe_category', None )
	st.session_state.setdefault( 'pe_prompt', '' )
	
	# ------------------------------------------------------------------
	# Prompt Engineering Helpers
	# ------------------------------------------------------------------
	def get_prompt_connection( ) -> sqlite3.Connection:
		"""Get prompt connection.
		
		Purpose:
		    Creates a SQLite connection to the configured application database for Prompt
		    Engineering read and write operations.
		
		Returns:
		    sqlite3.Connection: Open SQLite connection to the application database.
		"""
		return sqlite3.connect( cfg.DB_PATH )
	
	def reset_prompt_page( ) -> None:
		"""Reset prompt page.
		
		Purpose:
		    Returns the Prompt Engineering result grid to its first page when a search or sort
		    control changes.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		st.session_state[ 'pe_page' ] = 1
	
	def reset_prompt_selection( ) -> None:
		"""Reset prompt selection.
		
		Purpose:
		    Clears the selected Prompt Engineering record and resets the authoritative editor
		    fields without changing search, sorting, or paging controls.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		st.session_state[ 'pe_selected_id' ] = None
		st.session_state[ 'pe_caption' ] = ''
		st.session_state[ 'pe_name' ] = ''
		st.session_state[ 'pe_category' ] = None
		st.session_state[ 'pe_prompt' ] = ''
	
	def load_prompt_record( prompt_id: int ) -> None:
		"""Load prompt record.
		
		Purpose:
		    Loads the selected category-aware prompt record into the authoritative Prompt
		    Engineering editor fields.
		
		Args:
		    prompt_id (int): Numeric primary key of the prompt record to load.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			record = fetch_prompt_by_id( int( prompt_id ) )
			
			if record is None:
				reset_prompt_selection( )
				st.warning( f'Prompt {prompt_id} was not found.' )
				return
			
			st.session_state[ 'pe_selected_id' ] = int( record[ 'ID' ] )
			st.session_state[ 'pe_caption' ] = str( record[ 'Caption' ] )
			st.session_state[ 'pe_name' ] = str( record[ 'Name' ] )
			st.session_state[ 'pe_category' ] = str( record[ 'Category' ] )
			st.session_state[ 'pe_prompt' ] = str( record[ 'Prompt' ] )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Prompt Engineering'
			ex.method = 'load_prompt_record( prompt_id: int ) -> None'
			Logger( ).write( ex )
			raise ex
	
	def fetch_prompt_editor_categories( ) -> List[ str ]:
		"""Fetch prompt editor categories.
		
		Purpose:
		    Returns the combined set of configured and persisted prompt categories available to
		    the Prompt Engineering editor.
		
		Returns:
		    List[str]: Sorted prompt categories available for record creation and editing.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			configured_categories = { category for categories in PROMPT_CATEGORY_MODE_MAP.values( )
				for category in categories if isinstance( category, str ) and category.strip( ) }
			
			with get_prompt_connection( ) as conn:
				rows = conn.execute( f"""
					SELECT DISTINCT Category
					FROM {TABLE}
					WHERE Category IS NOT NULL
						AND TRIM(Category) <> '';
					""" ).fetchall( )
			
			persisted_categories = { str( row[ 0 ] ).strip( ) for row in rows if
				row and row[ 0 ] is not None and str( row[ 0 ] ).strip( ) }
			
			return sorted( configured_categories | persisted_categories )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Prompt Engineering'
			ex.method = 'fetch_prompt_editor_categories( ) -> List[ str ]'
			Logger( ).write( ex )
			raise ex
	
	def validate_prompt_editor( ) -> Dict[ str, str ]:
		"""Validate prompt editor.
		
		Purpose:
		    Validates and normalizes the authoritative Prompt Engineering editor values before a
		    prompt record is inserted or updated.
		
		Returns:
		    Dict[str, str]: Normalized Caption, Name, Category, and Prompt values.
		
		Raises:
		    ValueError: Raised when a required prompt field is empty.
		"""
		data = { 'Caption': str( st.session_state.get( 'pe_caption', '' ) or '' ).strip( ),
			'Name': str( st.session_state.get( 'pe_name', '' ) or '' ).strip( ),
			'Category': str( st.session_state.get( 'pe_category', '' ) or '' ).strip( ),
			'Prompt': str( st.session_state.get( 'pe_prompt', '' ) or '' ).strip( ), }
		
		for field_name, field_value in data.items( ):
			if not field_value:
				raise ValueError( f'{field_name} is required.' )
		
		return data
	
	def save_prompt_record( ) -> None:
		"""Save prompt record.
		
		Purpose:
		    Creates or updates the authoritative Prompt Engineering record using the canonical
		    category-aware prompt schema.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			data = validate_prompt_editor( )
			selected_id = st.session_state.get( 'pe_selected_id' )
			
			if selected_id is None:
				insert_prompt( data )
				message = 'Prompt created.'
			else:
				update_prompt( int( selected_id ), data )
				message = 'Prompt updated.'
			
			reset_prompt_selection( )
			st.success( message )
			st.rerun( )
		except ValueError as e:
			st.warning( str( e ) )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Prompt Engineering'
			ex.method = 'save_prompt_record( ) -> None'
			Logger( ).write( ex )
			raise ex
	
	def delete_prompt_record( ) -> None:
		"""Delete prompt record.
		
		Purpose:
		    Deletes the selected Prompt Engineering record and resets the authoritative editor
		    state.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger.
		"""
		try:
			selected_id = st.session_state.get( 'pe_selected_id' )
			
			if selected_id is None:
				st.warning( 'Select a prompt before deleting.' )
				return
			
			delete_prompt( int( selected_id ) )
			reset_prompt_selection( )
			st.success( 'Prompt deleted.' )
			st.rerun( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'app'
			ex.cause = 'Prompt Engineering'
			ex.method = 'delete_prompt_record( ) -> None'
			Logger( ).write( ex )
			raise ex
	
	def convert_prompt_xml_to_markdown( ) -> None:
		"""Convert prompt XML to Markdown.
		
		Purpose:
		    Converts XML-style instruction blocks in the authoritative prompt editor to Markdown.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		prompt_value = st.session_state.get( 'pe_prompt', '' )
		
		if isinstance( prompt_value, str ) and prompt_value.strip( ):
			st.session_state[ 'pe_prompt' ] = convert_xml( prompt_value )
	
	def convert_prompt_markdown_to_xml( ) -> None:
		"""Convert prompt Markdown to XML.
		
		Purpose:
		    Converts Markdown headings in the authoritative prompt editor to XML-style instruction
		    blocks.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		"""
		prompt_value = st.session_state.get( 'pe_prompt', '' )
		
		if isinstance( prompt_value, str ) and prompt_value.strip( ):
			st.session_state[ 'pe_prompt' ] = convert_markdown( prompt_value )
	
	# ------------------------------------------------------------------
	# Sanitize Prompt Engineering State
	# ------------------------------------------------------------------
	valid_sort_columns = [ 'ID', 'Caption', 'Name', 'Category', ]
	
	if st.session_state.get( 'pe_sort_col' ) not in valid_sort_columns:
		st.session_state[ 'pe_sort_col' ] = 'ID'
	
	if st.session_state.get( 'pe_sort_dir' ) not in [ 'ASC', 'DESC' ]:
		st.session_state[ 'pe_sort_dir' ] = 'ASC'
	
	editor_categories = fetch_prompt_editor_categories( )
	current_editor_category = st.session_state.get( 'pe_category' )
	
	if current_editor_category and current_editor_category not in editor_categories:
		editor_categories.append( str( current_editor_category ) )
		editor_categories.sort( )
	
	if current_editor_category == '':
		st.session_state[ 'pe_category' ] = None
	
	# ------------------------------------------------------------------
	# Controls
	# ------------------------------------------------------------------
	c1, c2, c3, c4 = st.columns( [ 4, 2, 2, 3 ] )
	
	with c1:
		st.text_input( label='Search', key='pe_search',
			placeholder='Caption, name, category, or prompt text', on_change=reset_prompt_page, )
	
	with c2:
		st.selectbox( label='Sort by', options=valid_sort_columns, key='pe_sort_col',
			on_change=reset_prompt_page, )
	
	with c3:
		st.selectbox( label='Direction', options=[ 'ASC', 'DESC' ], key='pe_sort_dir',
			on_change=reset_prompt_page, )
	
	with c4:
		st.markdown( "<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.25rem;'>"
		             "Go to ID</div>", unsafe_allow_html=True, )
		
		a1, a2, a3 = st.columns( [ 2, 1, 1 ] )
		
		with a1:
			jump_id = st.number_input( label='Go to ID', min_value=0, step=1,
				label_visibility='collapsed', key='pe_jump_id', )
		
		with a2:
			if st.button( label='Go', key='pe_go_to_id', width='stretch' ):
				load_prompt_record( int( jump_id ) )
		
		with a3:
			if st.button( label='Undo', key='pe_undo_selection', width='stretch' ):
				reset_prompt_selection( )
				st.rerun( )
	
	# ------------------------------------------------------------------
	# Query Prompt Records
	# ------------------------------------------------------------------
	where_clause = ''
	query_params: List[ Any ] = [ ]
	search_value = str( st.session_state.get( 'pe_search', '' ) or '' ).strip( )
	
	if search_value:
		where_clause = """
			WHERE Caption LIKE ?
				OR Name LIKE ?
				OR Category LIKE ?
				OR Prompt LIKE ?
		"""
		
		search_pattern = f'%{search_value}%'
		query_params.extend( [ search_pattern, search_pattern, search_pattern, search_pattern, ] )
	
	count_query = f"""
		SELECT COUNT(*)
		FROM {TABLE}
		{where_clause};
	"""
	
	with get_prompt_connection( ) as conn:
		total_rows = int( conn.execute( count_query, tuple( query_params ) ).fetchone( )[ 0 ] )
	
	total_pages = max( 1, math.ceil( total_rows / PAGE_SIZE ) )
	
	if st.session_state[ 'pe_page' ] > total_pages:
		st.session_state[ 'pe_page' ] = total_pages
	
	if st.session_state[ 'pe_page' ] < 1:
		st.session_state[ 'pe_page' ] = 1
	
	offset = (int( st.session_state[ 'pe_page' ] ) - 1) * PAGE_SIZE
	
	data_query = f"""
		SELECT
			ID,
			Caption,
			Name,
			Category
		FROM {TABLE}
		{where_clause}
		ORDER BY {st.session_state[ 'pe_sort_col' ]}
			{st.session_state[ 'pe_sort_dir' ]}
		LIMIT ?
		OFFSET ?;
	"""
	
	data_params = query_params + [ PAGE_SIZE, offset ]
	
	with get_prompt_connection( ) as conn:
		rows = conn.execute( data_query, tuple( data_params ) ).fetchall( )
	
	# ------------------------------------------------------------------
	# Prompt Table
	# ------------------------------------------------------------------
	df_prompt_rows = pd.DataFrame( [
		{ 'Selected': int( row[ 0 ] ) == st.session_state.get( 'pe_selected_id' ),
			'ID': int( row[ 0 ] ), 'Caption': str( row[ 1 ] or '' ), 'Name': str( row[ 2 ] or '' ),
			'Category': str( row[ 3 ] or '' ), } for row in rows ],
		columns=[ 'Selected', 'ID', 'Caption', 'Name', 'Category', ], )
	
	df_edited_prompts = st.data_editor( df_prompt_rows, hide_index=True, width='stretch',
		disabled=[ 'ID', 'Caption', 'Name', 'Category', ], column_config={
			'Selected': st.column_config.CheckboxColumn( label='Selected', width='small', ),
			'ID': st.column_config.NumberColumn( label='ID', format='%d', width='small', ),
			'Caption': st.column_config.TextColumn( label='Caption', width='medium', ),
			'Name': st.column_config.TextColumn( label='Name', width='medium', ),
			'Category': st.column_config.TextColumn( label='Category', width='medium', ), },
		key='pe_prompt_table', )
	
	if isinstance( df_edited_prompts, pd.DataFrame ) and not df_edited_prompts.empty:
		df_selected_prompts = df_edited_prompts.loc[ df_edited_prompts[ 'Selected' ] == True ]
		
		if len( df_selected_prompts.index ) == 1:
			selected_id = int( df_selected_prompts.iloc[ 0 ][ 'ID' ] )
			
			if selected_id != st.session_state.get( 'pe_selected_id' ):
				load_prompt_record( selected_id )
				st.rerun( )
		
		elif len( df_selected_prompts.index ) > 1:
			st.warning( 'Select only one prompt record at a time.' )
	
	# ------------------------------------------------------------------
	# Paging
	# ------------------------------------------------------------------
	p1, p2, p3 = st.columns( [ 1, 2, 1 ] )
	
	with p1:
		if st.button( label='◀ Prev', key='pe_previous_page', width='stretch',
				disabled=st.session_state[ 'pe_page' ] <= 1, ):
			st.session_state[ 'pe_page' ] -= 1
			st.rerun( )
	
	with p2:
		st.markdown( f"Page **{st.session_state[ 'pe_page' ]}** of **{total_pages}** "
		             f"— **{total_rows:,} prompts**" )
	
	with p3:
		if st.button( label='Next ▶', key='pe_next_page', width='stretch',
				disabled=st.session_state[ 'pe_page' ] >= total_pages, ):
			st.session_state[ 'pe_page' ] += 1
			st.rerun( )
	
	st.divider( )
	
	# ------------------------------------------------------------------
	# XML / Markdown Converter
	# ------------------------------------------------------------------
	with st.expander( label='XML ↔ Markdown Converter', expanded=False ):
		b1, b2 = st.columns( 2 )
		
		with b1:
			st.button( label='Convert XML → Markdown', key='pe_convert_xml_to_markdown',
				width='stretch', on_click=convert_prompt_xml_to_markdown, )
		
		with b2:
			st.button( label='Convert Markdown → XML', key='pe_convert_markdown_to_xml',
				width='stretch', on_click=convert_prompt_markdown_to_xml, )
	
	# ------------------------------------------------------------------
	# Create / Edit Prompt
	# ------------------------------------------------------------------
	with st.expander( label='Create / Edit Prompt', expanded=True ):
		st.text_input( label='ID',
			value=st.session_state.get( 'pe_selected_id' ) if st.session_state.get(
				'pe_selected_id' ) is not None else '', disabled=True, key='pe_display_id', )
		
		editor_c1, editor_c2, editor_c3 = st.columns( [ 0.34, 0.33, 0.33 ] )
		
		with editor_c1:
			st.text_input( label='Caption', key='pe_caption',
				placeholder='Human-readable template caption', )
		
		with editor_c2:
			st.text_input( label='Name', key='pe_name', placeholder='Programmatic prompt name', )
		
		with editor_c3:
			st.selectbox( label='Category', options=editor_categories, index=None,
				key='pe_category', placeholder='Select Category', )
		
		st.text_area( label='Prompt', key='pe_prompt', height=260, width='stretch', )
		
		c1, c2, c3 = st.columns( 3 )
		
		with c1:
			st.button( label='Save Changes' if st.session_state.get(
				'pe_selected_id' ) is not None else 'Create Prompt', key='pe_save_prompt',
				width='stretch', on_click=save_prompt_record, )
		
		with c2:
			st.button( label='Delete', key='pe_delete_prompt', width='stretch',
				disabled=st.session_state.get( 'pe_selected_id' ) is None,
				on_click=delete_prompt_record, )
		
		with c3:
			st.button( label='Clear Selection', key='pe_clear_selection', width='stretch',
				on_click=reset_prompt_selection, )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '🏛️ Data Management', help=cfg.DATA_MANAGEMENT )
		tabs = st.tabs(
			[ 'Import', 'Browse', 'CRUD', 'Explore', 'Filter', 'Aggregate', 'Visualize', 'Admin',
				'SQL' ] )
		
		tables = list_tables( )
		if not tables:
			st.info( "No tables available." )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			overwrite = st.checkbox( 'Overwrite existing tables', value=True )
			if uploaded_file:
				try:
					sheets = pd.read_excel( uploaded_file, sheet_name=None )
					with create_connection( ) as conn:
						conn.execute( 'BEGIN' )
						for sheet_name, df in sheets.items( ):
							table_name = create_identifier( sheet_name )
							if overwrite:
								conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
							
							# --- Create Table ---
							columns = [ ]
							df.columns = [ create_identifier( c ) for c in df.columns ]
							for col in df.columns:
								sql_type = get_sqlite_type( df[ col ].dtype )
								columns.append( f'"{col}" {sql_type}' )
							
							create_stmt = (f'CREATE TABLE "{table_name}" '
							               f'({", ".join( columns )});')
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (f'INSERT INTO "{table_name}" '
							               f'VALUES ({placeholders});')
							
							conn.executemany( insert_stmt,
								df.where( pd.notnull( df ), None ).values.tolist( ) )
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				render_table( df )
			else:
				st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				crud_header_c1, crud_header_c2, crud_header_c3 = st.columns( [ 0.45, 0.25, 0.30 ],
					border=True )
				
				with crud_header_c1:
					table = st.selectbox( 'Select Table', tables, key='crud_table' )
				
				df = read_table( table )
				schema = create_schema( table )
				
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				with crud_header_c2:
					st.metric( 'Rows', len( df.index ) )
				
				with crud_header_c3:
					st.metric( 'Columns', len( type_map ) )
				
				st.divider( )
				
				insert_col, update_col = st.columns( [ 0.50, 0.50 ], border=True )
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				with insert_col:
					st.markdown( '#### Insert Row' )
					insert_data = { }
					
					for column, col_type in type_map.items( ):
						if 'INT' in col_type:
							insert_data[ column ] = st.number_input( column, step=1,
								key=f'ins_{table}_{column}' )
						
						elif 'REAL' in col_type:
							insert_data[ column ] = st.number_input( column, format='%.6f',
								key=f'ins_{table}_{column}' )
						
						elif 'BOOL' in col_type:
							insert_data[ column ] = 1 if st.checkbox( column,
								key=f'ins_{table}_{column}' ) else 0
						
						else:
							insert_data[ column ] = st.text_input( column,
								key=f'ins_{table}_{column}' )
					
					if st.button( 'Insert Row', key=f'insert_row_{table}',
							use_container_width=True ):
						cols = list( insert_data.keys( ) )
						quoted_cols = [ f'"{c}"' for c in cols ]
						placeholders = ', '.join( [ '?' ] * len( cols ) )
						stmt = (f'INSERT INTO "{table}" ({", ".join( quoted_cols )}) '
						        f'VALUES ({placeholders});')
						
						with create_connection( ) as conn:
							conn.execute( stmt, list( insert_data.values( ) ) )
							conn.commit( )
						
						st.success( 'Row inserted.' )
						st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				with update_col:
					st.markdown( '#### Update Row' )
					rowid = st.number_input( 'Row ID', min_value=1, step=1,
						key=f'crud_update_rowid_{table}' )
					
					update_data = { }
					
					for column, col_type in type_map.items( ):
						if 'INT' in col_type:
							val = st.number_input( column, step=1, key=f'upd_{table}_{column}' )
							update_data[ column ] = val
						
						elif 'REAL' in col_type:
							val = st.number_input( column, format='%.6f',
								key=f'upd_{table}_{column}' )
							update_data[ column ] = val
						
						elif 'BOOL' in col_type:
							val = 1 if st.checkbox( column, key=f'upd_{table}_{column}' ) else 0
							update_data[ column ] = val
						
						else:
							val = st.text_input( column, key=f'upd_{table}_{column}' )
							update_data[ column ] = val
					
					if st.button( 'Update Row', key=f'update_row_{table}',
							use_container_width=True ):
						set_clause = ', '.join( [ f'"{c}"=?' for c in update_data ] )
						stmt = f'UPDATE "{table}" SET {set_clause} WHERE rowid=?;'
						
						with create_connection( ) as conn:
							conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
							conn.commit( )
						
						st.success( 'Row updated.' )
						st.rerun( )
				
				st.divider( )
				
				delete_col, preview_col = st.columns( [ 0.35, 0.65 ], border=True )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				with delete_col:
					st.markdown( '#### Delete Row' )
					delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1,
						key=f'crud_delete_rowid_{table}' )
					
					if st.button( 'Delete Row', key=f'delete_row_{table}',
							use_container_width=True ):
						with create_connection( ) as conn:
							conn.execute( f'DELETE FROM "{table}" WHERE rowid=?;', (delete_id,) )
							conn.commit( )
						
						st.success( 'Row deleted.' )
						st.rerun( )
				
				# ------------------------------------------------------------------
				# PREVIEW
				# ------------------------------------------------------------------
				with preview_col:
					st.markdown( '#### Current Data Preview' )
					st.data_editor( df.head( 25 ), key=f'dm_crud_preview_{table}',
						use_container_width=True, disabled=True )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				exp_c1, exp_c2, exp_c3 = st.columns( [ 0.4, 0.4, 0.2 ], border=True )
				with exp_c1:
					table = st.selectbox( 'Table', tables, key='explore_table' )
				with exp_c2:
					page_size = st.slider( 'Rows per page', 10, 500, 50 )
				with exp_c3:
					page = st.number_input( 'Page', min_value=1, step=1 )
					offset = (page - 1) * page_size
					df_page = read_table( table, page_size, offset )
				
				st.data_editor( df_page )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				tbl_c1, tbl_c2, tbl_c3 = st.columns( [ 0.25, 0.25, 0.5 ], border=True )
				with tbl_c1:
					table = st.selectbox( 'Select Table', tables, key='filter_table' )
					df = read_table( table )
				with tbl_c2:
					column = st.selectbox( 'Select Field', df.columns )
				with tbl_c3:
					value = st.text_input( 'Contains', placeholder='Enter Text for Lookup' )
					if value:
						df = df[ df[ column ].astype( str ).str.contains( value ) ]
				
				st.data_editor( df )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				agg_c1, agg_c2, agg_c3, agg_c4 = st.columns( [ 0.2, 0.2, 0.2, 0.4 ], border=True )
				with agg_c1:
					table = st.selectbox( 'Table', tables, key='agg_table' )
					df = read_table( table )
					numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
					with agg_c2:
						if numeric_cols:
							col = st.selectbox( 'Column', numeric_cols )
					with agg_c3:
						agg = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ] )
					with agg_c4:
						if agg == 'SUM':
							st.metric( 'Result', df[ col ].sum( ), width='stretch',
								format='accounting' )
						
						elif agg == 'AVG':
							st.metric( 'Result', df[ col ].mean( ), width='stretch',
								format='accounting' )
						
						elif agg == 'COUNT':
							st.metric( 'Result', df[ col ].count( ), width='stretch',
								format='accounting' )
		
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				create_visualization( df )
		
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			
			st.subheader( 'Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					render_table( profile_df )
			
			st.subheader( 'Drop Table' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
					            'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			st.divider( )
			st.subheader( 'Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20,
				value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( { 'name': col_name, 'type': col_type, 'not_null': not_null,
					'primary_key': primary_key, 'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.subheader( 'Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				
				# Column schema
				schema = create_schema( table )
				schema_df = pd.DataFrame( schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "### Columns" )
				st.data_editor( make_display_safe( schema_df ), hide_index=True,
					use_container_width=True, disabled=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute( f'SELECT COUNT(*) FROM "{table}"' ).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame( indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ] )
					st.markdown( "### Indexes" )
					st.data_editor( make_display_safe( idx_df ), hide_index=True,
						use_container_width=True, disabled=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.subheader( "ALTER TABLE Operations" )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ] )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ] )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
		
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.subheader( 'SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv, 'query_results.csv',
								'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ======================================================================================
# FOOTER — SECTION
# ======================================================================================
st.markdown( """
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""", unsafe_allow_html=True, )

# ---- Fixed Container
st.markdown( """
	<style>
	.boo-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(27, 27, 27, 0.95);
		border-top: 1px solid #4d4d4d;
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #4aa2f7;
		z-index: 1000;
	}
	.boo-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""", unsafe_allow_html=True, )

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================
_mode_to_model_key = { 'Text': 'text_model', 'Images': 'image_model', 'Audio': 'audio_model',
	'Embeddings': 'embed_model', 'Document Q&A': 'doc_model', 'Files': 'files_model',
	'Vector Stores': 'stores_model', 'File Search Stores': 'filestore_model',
	'Google Cloud Buckets': 'bucket_model', 'Prompt Engineering': 'prompt_model',
	'Data Management': 'data_model', }

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
	audio_input = st.session_state.get( 'audio_input' )
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
st.markdown( f"""
    <div class="boo-status-bar">
        <div class="boo-status-inner">
            <span>{provider_val} — {mode_val}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True, )
