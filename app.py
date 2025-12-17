# ******************************************************************************************
# Assembly:                Boo
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Created:                 12-16-2025
# ******************************************************************************************

from __future__ import annotations

import streamlit as st
import tempfile
from typing import List, Dict, Any
import fitz

from boo import (
	Chat,
	Image,
	Embedding,
	Transcription,
	Translation,
)

# =========================================================================================
# Streamlit Configuration
# =========================================================================================

st.set_page_config( page_title='Boo • Multimodal AI Assistant', page_icon='👻', layout='wide' )

# =========================================================================================
# Instantiate Boo Components (Read-Only Introspection)
# =========================================================================================

chat = Chat( )
image = Image( )
embedding = Embedding( )
transcriber = Transcription( )
translator = Translation( )

# =========================================================================================
# Session State
# =========================================================================================

if 'messages' not in st.session_state:
	st.session_state.messages: List[ Dict[ str, Any ] ] = [ ]

if 'files' not in st.session_state:
	st.session_state.files: List[ str ] = [ ]

# =========================================================================================
# Utilities
# =========================================================================================

def save_temp( upload ) -> str:
	with tempfile.NamedTemporaryFile( delete=False ) as tmp:
		tmp.write( upload.read( ) )
		return tmp.name

# =========================================================================================
# Sidebar — Controls
# =========================================================================================

with st.sidebar:
	st.markdown( '## 👻 Boo' )
	st.caption( 'Multimodal AI Framework' )
	
	st.markdown( '---' )
	st.markdown( '### 🧠 Chat Model' )
	
	model = st.selectbox(
		'Model',
		chat.model_options,
		index=chat.model_options.index( 'gpt-4o-mini' )
		if 'gpt-4o-mini' in chat.model_options else 0,
	)
	
	st.markdown( '---' )
	st.markdown( '### 🧩 Response Includes' )
	
	include = st.multiselect(
		'Include',
		chat.include_options,
	)
	
	st.markdown( '---' )
	st.markdown( '### 🧭 Mode' )
	
	mode = st.radio(
		'Mode',
		[
				'Chat',
				'Images',
				'Audio',
				'Embeddings',
		],
		label_visibility='collapsed',
	)
	
	st.markdown( '---' )
	st.markdown( '### 📄 Documents' )
	
	uploads = st.file_uploader( 'Upload files', type=[ 'pdf', 'txt','md', 'docx' ], accept_multiple_files=True, )
	
	if uploads:
		st.session_state.files.clear( )
		for f in uploads:
			st.session_state.files.append( save_temp( f ) )
	
	st.markdown( '---' )
	
	if st.button( 'Clear Conversation' ):
		st.session_state.messages.clear( )

# =========================================================================================
# Header
# =========================================================================================

st.markdown(
	"""
	<h1 style='margin-bottom:0.25rem;'>👻 Boo</h1>
	<p style='color:#9aa0a6;'>
		Introspection-driven, multimodal AI assistant
	</p>
	""",
	unsafe_allow_html=True,
)

st.divider( )

# =========================================================================================
# CHAT MODE
# =========================================================================================

if mode == 'Chat':
	store_name = st.selectbox(
		'Vector Store',
		list( chat.vector_stores.keys( ) ),
	)
	
	chat.vector_store_ids = [ chat.vector_stores[ store_name ] ]
	chat.include = include
	
	for msg in st.session_state.messages:
		with st.chat_message( msg[ 'role' ] ):
			st.markdown( msg[ 'content' ] )
	
	prompt = st.chat_input( 'Ask Boo a question…' )
	
	if prompt:
		st.session_state.messages.append(
			{
					'role': 'user',
					'content': prompt }
		)
		
		with st.chat_message( 'assistant' ):
			with st.spinner( 'Thinking…' ):
				response = chat.generate_text(
					prompt=prompt,
					model=model,
				)
				
				st.markdown( response or "" )
				st.session_state.messages.append(
					{
							'role': 'assistant',
							'content': response or "" }
				)

# =========================================================================================
# IMAGE MODE
# =========================================================================================
elif mode == 'Images':
	tab_gen, tab_analyze = st.tabs( [ '🖼️ Generate',  '🔍 Analyze' ] )
	
	with tab_gen:
		prompt = st.text_area( 'Prompt', height=120 )
		
		col1, col2 = st.columns( 2 )
		
		with col1:
			size = st.selectbox( 'Size', image.size_options )
			quality = st.selectbox( 'Quality', [ 'standard',  'hd' ] )
		
		with col2:
			fmt = st.selectbox( 'Format', image.format_options )
			detail = st.selectbox( 'Detail', image.detail_options )
		
		if st.button( 'Generate Image' ):
			with st.spinner( 'Generating…' ):
				url = image.generate(
					prompt=prompt,
					model='dall-e-3',
					size=size,
					quality=quality,
				)
				st.image( url )
	
	with tab_analyze:
		img = st.file_uploader( 'Upload image', type=[ 'png', 'jpg',  'jpeg' ] )
		prompt = st.text_area( 'Analysis prompt', value='Describe this image in detail.', )
		
		if img and st.button( 'Analyze Image' ):
			path = save_temp( img )
			with st.spinner( 'Analyzing…' ):
				result = image.analyze( text=prompt, path=path,)
				st.markdown( result )

# =========================================================================================
# AUDIO MODE
# =========================================================================================
elif mode == 'Audio':
	audio = st.audio_input( 'Record or upload audio' )
	task = st.radio( 'Task', [ 'Transcription', 'Translation' ], horizontal=True )
	
	if audio and st.button( 'Process' ):
		path = save_temp( audio )
		
		with st.spinner( 'Processing…' ):
			if task == 'Transcription':
				text = transcriber.transcribe( path )
				st.markdown( '### 📝 Transcription' )
				st.markdown( text )
			
			else:
				lang = st.text_input( 'Target language', value='en' )
				text = translator.translate( path, lang )
				st.markdown( '### 🌍 Translation' )
				st.markdown( text )

# =========================================================================================
# EMBEDDINGS MODE
# =========================================================================================
elif mode == 'Embeddings':
	text = st.text_area( 'Text to embed', height=150 )
	
	col1, col2 = st.columns( 2 )
	
	with col1:
		model = st.selectbox( 'Model', embedding.model_options )
	
	with col2:
		encoding = st.selectbox( 'Encoding', embedding.encoding_options )
	
	if st.button( 'Create Embedding' ):
		with st.spinner( 'Embedding…' ):
			vector = embedding.create(
				text=text,
				model=model,
				format=encoding,
			)
			st.success( f'Vector length: {len( vector )}' )
			st.json( vector[ :10 ] )

# =========================================================================================
# Footer
# =========================================================================================

st.markdown(
	"""
	<hr/>
	<div style='display:flex; justify-content:space-between; color:#9aa0a6; font-size:0.85rem;'>
		<span>Boo Framework</span>
		<span>Single-page • Introspection-driven</span>
	</div>
	""", unsafe_allow_html=True, )
