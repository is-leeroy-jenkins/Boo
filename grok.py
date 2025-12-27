'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                grok.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="grok.py" company="Terry D. Eppler">

	     grok.py
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
    grok.py
  </summary>
  ******************************************************************************************
'''

import os
import base64
import requests
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
import groq
from groq import Groq
import config as cfg
from boogr import ErrorDialog, Error

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encodes a local image to a base64 string for vision API requests."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class Endpoints:
	'''

	    Purpose:
	    ---------
	    The class containing endpoints for the Groq API and Hybrid services.

	    Attributes:
	    -----------
	    base_url           : str - The base URL for Groq API
	    chat_completions   : str - Endpoint for chat completion requests
	    speech_generations : str - Endpoint for TTS generation
	    translations       : str - Endpoint for audio translation
	    transcriptions     : str - Endpoint for audio transcription
	    image_generations  : str - Endpoint for image generation
	    image_edits        : str - Endpoint for image editing
	    embeddings         : str - Endpoint for text embeddings
	    wolfram            : str - Endpoint for Wolfram Alpha API
	    google_search      : str - Endpoint for Google/Tavily search API

    '''
	base_url: Optional[ str ]
	chat_completions: Optional[ str ]
	speech_generations: Optional[ str ]
	translations: Optional[ str ]
	transcriptions: Optional[ str ]
	image_generations: Optional[ str ]
	image_edits: Optional[ str ]
	embeddings: Optional[ str ]
	wolfram: Optional[ str ]
	google_search: Optional[ str ]
	
	def __init__( self ):
		self.base_url = f'https://api.groq.com/'
		self.chat_completions = f'https://api.groq.com/openai/v1/chat/completions'
		self.speech_generations = f'https://api.openai.com/v1/audio/speech'
		self.translations = f'https://api.groq.com/openai/v1/audio/translations'
		self.transcriptions = f'https://api.groq.com/openai/v1/audio/transcriptions'
		self.image_generations = f'https://api.openai.com/v1/images/generations'
		self.image_edits = f'https://api.openai.com/v1/images/edits'
		self.embeddings = f'https://api.openai.com/v1/embeddings'
		self.wolfram = f'https://api.wolframalpha.com/v1/result'
		self.google_search = f'https://api.tavily.com/search'

class Header:
	'''

	    Purpose:
	    --------
	    Encapsulates HTTP header configurations for API requests.

	    Attributes:
	    -----------
	    content_type  : str - The MIME type for request bodies
	    api_key       : str - The API key used for authentication
	    authorization : str - The Bearer token string

	    Methods:
	    --------
	    get_header( ) : Returns the header dictionary for requests

    '''
	content_type: Optional[ str ]
	api_key: Optional[ str ]
	authorization: Optional[ str ]
	
	def __init__( self, key: str = cfg.GROQ_API_KEY ):
		self.content_type = 'application/json'
		self.api_key = key
		self.authorization = f'Bearer {key}'
	
	def get_header( self ) -> Dict[ str, str ]:
		"""
		Purpose: Returns the standard HTTP header dictionary.
		Returns: Dict[ str, str ] - Header dictionary.
		"""
		return {
				'Content-Type': self.content_type,
				'Authorization': self.authorization }

class Grok:
	'''

		Purpose:
		-------
		Base class for Groq AI functionality and configuration.

		Attributes:
		-----------
		api_key           : str - Groq API Key
		instructions      : str - System instructions for the model
		prompt            : str - User input prompt
		model             : str - Model identifier
		max_tokens        : int - Maximum completion tokens
		temperature       : float - Sampling temperature
		top_p             : float - Nucleus sampling parameter
		top_k             : int - Top-k sampling parameter
		modalities        : list - List of input/output modalities
		frequency_penalty : float - Frequency penalty coefficient
		presence_penalty  : float - Presence penalty coefficient
		response_format   : dict - Formatting configuration for response
		candidate_count   : int - Number of outputs per request

	'''
	api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	modalities: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ Union[ str, Dict[ str, str ] ] ]
	candidate_count: Optional[ int ]
	
	def __init__( self ):
		self.api_key = cfg.GROQ_API_KEY
		self.model = None
		self.temperature = 0.7
		self.top_p = 0.9
		self.top_k = 40
		self.candidate_count = 1
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 4096
		self.instructions = None
		self.prompt = None
		self.modalities = None
		self.response_format = None

class Chat( Grok ):
	'''

	    Purpose:
	    _______
	    Class handling text completions, vision, and tool integration via Groq.

	    Attributes:
	    -----------
	    client           : Groq - The Groq API client
	    contents         : list - Collection of message parts
	    response         : any - The raw response object from Groq
	    image_url        : str - URL of the image being processed
	    file_path        : str - Local path of the file being processed
	    url              : str - Target URL for web tools

	    Methods:
	    --------
	    generate_text( prompt, model )      : Generates text response
	    generate_image( prompt, model )     : Generates image via hybrid API
	    analyze_image( prompt, filepath )   : Analyzes images with vision models
	    summarize_document( prompt, path )  : Summarizes document contents
	    search_file( prompt, filepath )     : Searches local file for information
	    web_search( prompt, model )         : Performs a broad web search
	    search_website( prompt, url, mod )  : Extracts content from a URL
	    wolfram_alpha( prompt, model )      : Computational query via Wolfram

    '''
	client: Optional[ Groq ]
	contents: Optional[ Union[ str, List[ Dict[ str, Any ] ] ] ]
	response: Optional[ Any ]
	image_url: Optional[ str ]
	file_path: Optional[ str ]
	url: Optional[ str ]
	
	def __init__( self, model: str = 'llama-3.3-70b-versatile', temperature: float = 0.8, top_p: float = 0.9,
			frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 4096,
			instruct: str = None ):
		super( ).__init__( )
		self.model = model;
		self.top_p = top_p;
		self.temperature = temperature
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.instructions = instruct
		self.client = Groq( api_key=self.api_key );
		self.contents = None
		self.response = None;
		self.image_url = None;
		self.file_path = None;
		self.url = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Returns the currently available high-performance LLMs on Groq."""
		return [ 'llama-3.3-70b-versatile',
		         'llama-3.3-70b-specdec',
		         'llama-3.1-70b-versatile',
		         'llama-3.1-8b-instant',
		         'mixtral-8x7b-32768',
		         'gemma2-9b-it' ]
	
	def generate_text( self, prompt: str, model: str = 'llama-3.3-70b-versatile' ) -> Optional[
		str ]:
		"""
		Purpose: Generates a text completion based on prompt and configuration.
		Parameters:
		-----------
		prompt: str - The user input string.
		model: str - The model identifier to use.
		Returns:
		--------
		Optional[ str ] - Generated text or None.
		"""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			messages = [ ]
			if self.instructions:
				messages.append( {
						"role": "system",
						"content": self.instructions } )
			messages.append( {
					"role": "user",
					"content": self.prompt } )
			self.response = self.client.chat.completions.create( model=self.model, messages=messages,
				temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p,
				frequency_penalty=self.frequency_penalty, presence_penalty=self.presence_penalty )
			return self.response.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def generate_image( self, prompt: str, model: str = 'dall-e-3' ) -> Optional[ str ]:
		"""
		Purpose: Routes an image generation request to the hybrid service.
		Parameters:
		-----------
		prompt: str - Description of the image.
		model: str - Image generation model ID.
		Returns:
		--------
		Optional[ str ] - URL of the generated image.
		"""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			image_gen = Image( temperature=self.temperature, top_p=self.top_p )
			return image_gen.generate( prompt=self.prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'generate_image( self, prompt: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def analyze_image( self, prompt: str, filepath: str, model: str = 'llama-3.2-11b-vision-preview' ) -> \
	Optional[ str ]:
		"""
		Purpose: Analyzes a local image file using a vision-capable model.
		Parameters:
		-----------
		prompt: str - Analysis instructions or questions.
		filepath: str - Path to local image file.
		model: str - Vision model ID.
		Returns:
		--------
		Optional[ str ] - Model analysis text.
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'filepath', filepath )
			self.prompt = prompt;
			self.file_path = filepath;
			self.model = model
			base64_image = encode_image( self.file_path )
			messages = [ {
					             "role": "user",
					             "content": [ {
							                          "type": "text",
							                          "text": self.prompt },
					                          {
							                          "type": "image_url",
							                          "image_url": {
									                          "url": f"data:image/jpeg;base64,{base64_image}" } } ] } ]
			self.response = self.client.chat.completions.create( model=self.model, messages=messages,
				temperature=self.temperature, max_tokens=self.max_tokens )
			return self.response.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, filepath: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def web_search( self, prompt: str, model: str = 'llama-3.3-70b-versatile' ) -> Optional[ str ]:
		"""
		Purpose: Performs a web search and synthesizes a response via LLM.
		Parameters:
		-----------
		prompt: str - Search query.
		model: str - Synthesis model ID.
		Returns:
		--------
		Optional[ str ] - Research summary.
		"""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			endpoint = Endpoints( ).google_search
			payload = {
					"api_key": cfg.TAVILY_API_KEY,
					"query": self.prompt,
					"search_depth": "advanced" }
			search_resp = requests.post( endpoint, json=payload )
			search_context = search_resp.json( ).get( 'results', [ ] )
			synth_prompt = f"Summarize these search results for query '{self.prompt}': {search_context}"
			return self.generate_text( prompt=synth_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def search_website( self, prompt: str, url: str, model: str = 'llama-3.3-70b-versatile' ) -> \
	Optional[ str ]:
		"""
		Purpose: Extracts content from a specific URL to answer a question.
		Parameters:
		-----------
		prompt: str - Specific question.
		url: str - Target website URL.
		model: str - Processing model ID.
		Returns:
		--------
		Optional[ str ] - Extracted answer.
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'url', url )
			self.prompt = prompt;
			self.url = url;
			self.model = model
			web_resp = requests.get( self.url, timeout=10 )
			web_text = web_resp.text[ :15000 ]
			site_prompt = f"Answer query '{self.prompt}' using this URL content: {web_text}"
			return self.generate_text( prompt=site_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'search_website( self, prompt: str, url: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def wolfram_alpha( self, prompt: str, model: str = 'llama-3.3-70b-versatile' ) -> Optional[
		str ]:
		"""
		Purpose: Retrieves computational data from Wolfram Alpha.
		Parameters:
		-----------
		prompt: str - Mathematical or factual query.
		model: str - Explanatory model ID.
		Returns:
		--------
		Optional[ str ] - Calculated result.
		"""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			endpoint = Endpoints( ).wolfram
			params = {
					"appid": cfg.WOLFRAM_APP_ID,
					"i": self.prompt }
			wolf_resp = requests.get( endpoint, params=params )
			math_prompt = f"Explain this result for '{self.prompt}': {wolf_resp.text}"
			return self.generate_text( prompt=math_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'wolfram_alpha( self, prompt: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def summarize_document( self, prompt: str, filepath: str, model: str = 'llama-3.3-70b-versatile' ) -> \
	Optional[ str ]:
		"""
		Purpose: Summarizes document text provided via local path.
		Parameters:
		-----------
		prompt: str - Formatting instructions.
		filepath: str - Path to text document.
		model: str - LLM identifier.
		Returns:
		--------
		Optional[ str ] - Summary text.
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'filepath', filepath )
			self.prompt = prompt;
			self.file_path = filepath;
			self.model = model
			with open( self.file_path, 'r', encoding='utf-8' ) as f:
				doc_text = f.read( )
			full_prompt = f"{self.prompt}\n\nDocument Contents:\n{doc_text}"
			return self.generate_text( full_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, filepath: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def search_file( self, prompt: str, filepath: str, model: str = 'llama-3.3-70b-versatile' ) -> \
	Optional[ str ]:
		"""
		Purpose: Searches local file contents to answer specific queries.
		Parameters:
		-----------
		prompt: str - Search query.
		filepath: str - Target file path.
		model: str - Search model identifier.
		Returns:
		--------
		Optional[ str ] - Found information.
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'filepath', filepath )
			self.prompt = prompt;
			self.file_path = filepath;
			self.model = model
			with open( self.file_path, 'r', encoding='utf-8' ) as f:
				doc_text = f.read( )
			search_prompt = f"Using this content, answer '{self.prompt}': {doc_text}"
			return self.generate_text( search_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'search_file( self, prompt: str, filepath: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )

class Embedding( Grok ):
	'''

		Purpose:
		--------
		Class providing text embedding generation via Hybrid services.

		Attributes:
		-----------
		client                : Groq - The API client instance
		response              : any - The API response object
		embedding             : list - The resulting float vector
		encoding_format       : str - Data format (float/base64)
		dimensions            : int - Vector size
		input_text            : str - Input text string
		top_percent           : float - Top-p probability
		max_completion_tokens : int - Token limit
		contents              : list - Input collection
		http_options          : dict - Network options

		Methods:
		--------
		create( text, model, format ) : Generates embedding vector
		count_tokens( text, coding )   : Estimates token count for input text

	'''
	client: Optional[ Groq ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	input_text: Optional[ str ]
	top_percent: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	contents: Optional[ List[ str ] ]
	http_options: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'text-embedding-3-small', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 10000 ):
		super( ).__init__( )
		self.api_key = cfg.GROQ_API_KEY
		self.client = Groq( api_key=self.api_key )
		self.model = model;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens;
		self.contents = [ ]
		self.http_options = { };
		self.encoding_format = 'float'
		self.input_text = None;
		self.embedding = None;
		self.response = None
		self.dimensions = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Returns list of standard hybrid embedding models."""
		return [ 'text-embedding-3-small',
		         'text-embedding-3-large',
		         'text-embedding-ada-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Returns list of available format options."""
		return [ 'float',
		         'base64' ]
	
	def create( self, text: str, model: str = 'text-embedding-3-small',
			format: str = 'float' ) -> Optional[ List[ float ] ]:
		"""
		Purpose: Generates an embedding vector for the provided text using Hybrid logic.
		Parameters:
		-----------
		text: str - Input text string.
		model: str - Embedding model identifier.
		format: str - Response data format.
		Returns:
		--------
		Optional[ List[ float ] ] - Float vector.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text;
			self.model = model;
			self.encoding_format = format
			endpoint = Endpoints( ).embeddings
			headers = Header( key=cfg.OPENAI_API_KEY ).get_header( )
			payload = {
					"input": self.input_text,
					"model": self.model,
					"encoding_format": self.encoding_format }
			resp = requests.post( endpoint, headers=headers, json=payload )
			if resp.status_code == 200:
				self.response = resp.json( )
				self.embedding = self.response[ 'data' ][ 0 ][ 'embedding' ]
				return self.embedding
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'groq';
			exception.cause = 'Embedding'
			exception.method = 'create( self, text: str, model: str, format: str ) -> Optional[ List[ float ] ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def count_tokens( self, text: str, coding: str = 'cl100k_base' ) -> Optional[ int ]:
		"""
		Purpose: Returns the number of tokens in the input string.
		Parameters:
		-----------
		text: str - Input string.
		coding: str - Encoding identifier.
		Returns:
		--------
		Optional[ int ] - Word count estimate.
		"""
		try:
			throw_if( 'text', text )
			return len( text.split( ) )
		except Exception as e:
			exception = Error( e );
			exception.module = 'groq';
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> Optional[ int ]';
			error = ErrorDialog( exception );
			error.show( )

class TTS( Grok ):
	"""

	    Purpose
	    ___________
	    Class for converting text into spoken audio using Hybrid providers.

	    Attributes:
	    -----------
	    speed                 : float - Playback speed rate
	    voice                 : str - Selected voice persona
	    response              : any - API response object
	    client                : Groq - API client instance
	    audio_path            : str - Output path for audio file
	    response_format       : str - Audio file format (wav, mp3, etc)
	    input_text            : str - Original text string to convert
	    store                 : bool - Flag to store request on server
	    stream                : bool - Flag to stream response

	    Methods:
	    --------
	    create_audio( text, path, format, speed, model ) : Generates audio file via POST request

    """
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ Any ]
	client: Optional[ Groq ]
	audio_path: Optional[ str ]
	response_format: Optional[ str ]
	input_text: Optional[ str ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	number: Optional[ int ]
	top_percent: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	stops: Optional[ List[ str ] ]
	messages: Optional[ List[ Dict[ str, str ] ] ]
	tools: Optional[ List[ Any ] ]
	vector_store_ids: Optional[ List[ str ] ]
	descriptions: Optional[ List[ str ] ]
	assistants: Optional[ List[ Any ] ]
	
	def __init__( self, number: int = 1, temperature: float = 0.8, top_p: float = 0.9, frequency: float = 0.0,
			presence: float = 0.0, max_tokens: int = 4096, store: bool = True, stream: bool = True,
			instruct: str = None ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key );
		self.model = 'tts-1'
		self.number = number;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.store = store;
		self.stream = stream
		self.instructions = instruct;
		self.audio_path = None;
		self.response = None
		self.response_format = 'mp3';
		self.speed = 1.0;
		self.voice = 'onyx'
		self.input_text = None;
		self.stops = None;
		self.messages = None;
		self.tools = None
		self.vector_store_ids = None;
		self.descriptions = None;
		self.assistants = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Returns list of available Hybrid TTS models."""
		return [ 'tts-1',
		         'tts-1-hd' ]
	
	@property
	def voice_options( self ) -> List[ str ]:
		"""Returns list of standard voices for hybrid TTS providers."""
		return [ 'alloy',
		         'echo',
		         'fable',
		         'onyx',
		         'nova',
		         'shimmer' ]
	
	@property
	def output_options( self ) -> List[ str ]:
		"""Returns list of audio file formats."""
		return [ 'mp3',
		         'opus',
		         'aac',
		         'flac',
		         'wav' ]
	
	@property
	def sample_options( self ) -> List[ int ]:
		"""Returns list of available sample rates (Hz) for UI selection."""
		return [ 8000,
		         16000,
		         22050,
		         24000,
		         32000,
		         44100,
		         48000 ]
	
	def create_audio( self, text: str, filepath: str, format: str = 'mp3',
			speed: float = 1.0, model: str = 'tts-1' ) -> Optional[ str ]:
		"""
		Purpose: Generates a spoken audio file from the provided text via POST.
		Parameters:
		-----------
		text: str - Input text string.
		filepath: str - Target path for audio file.
		format: str - MP3, WAV, etc.
		speed: float - Playback speed multiplier.
		model: str - TTS model ID.
		Returns:
		--------
		Optional[ str ] - Path to created file.
		"""
		try:
			throw_if( 'text', text );
			throw_if( 'filepath', filepath )
			self.input_text = text;
			self.audio_path = filepath
			self.response_format = format;
			self.speed = speed;
			self.model = model
			endpoint = Endpoints( ).speech_generations
			headers = Header( key=cfg.OPENAI_API_KEY ).get_header( )
			payload = {
					"model": self.model,
					"input": self.input_text,
					"voice": self.voice,
					"response_format": self.response_format,
					"speed": self.speed }
			resp = requests.post( endpoint, headers=headers, json=payload )
			if resp.status_code == 200:
				with open( self.audio_path, 'wb' ) as f:
					f.write( resp.content )
				return self.audio_path
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'TTS'
			exception.method = 'create_audio( self, text: str, filepath: str, format: str, speed: float ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )

class Transcription( Grok ):
	"""

	    Purpose
	    ___________
	    Class for transcribing audio recordings into text using Whisper on Groq LPU.

	    Attributes:
	    -----------
	    transcript            : str - Resulting text transcription
	    response              : any - RAW API response object

	    Methods:
	    --------
	    transcribe( path, model ) : Processes audio and returns text

    """
	client: Optional[ Groq ]
	audio_file: Optional[ Any ]
	transcript: Optional[ str ]
	response: Optional[ Any ]
	input_text: Optional[ str ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	number: Optional[ int ]
	top_percent: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	messages: Optional[ List[ Dict[ str, str ] ] ]
	stops: Optional[ List[ str ] ]
	
	def __init__( self, number: int = 1, temperature: float = 0.8, top_p: float = 0.9, frequency: float = 0.0,
			presence: float = 0.0, max_tokens: int = 4096, store: bool = True, stream: bool = True,
			instruct: str = None ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key )
		self.number = number;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.store = store;
		self.stream = stream
		self.instructions = instruct;
		self.input_text = None;
		self.audio_file = None
		self.transcript = None;
		self.response = None;
		self.messages = None;
		self.stops = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Returns list of available transcription models on Groq."""
		return [ 'whisper-large-v3-turbo',
		         'whisper-large-v3',
		         'distil-whisper-large-v3-en' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Returns list of available output response formats."""
		return [ 'text',
		         'json',
		         'verbose_json',
		         'vtt',
		         'srt' ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Returns common ISO language codes supported by Whisper."""
		return [ 'en',
		         'es',
		         'fr',
		         'de',
		         'it',
		         'pt',
		         'hi',
		         'ja',
		         'zh',
		         'ru' ]
	
	def transcribe( self, path: str, model: str = 'whisper-large-v3-turbo' ) -> Optional[ str ]:
		"""
		Purpose: Transcribes an audio file into text using Whisper LPU inference.
		Parameters:
		-----------
		path: str - Path to local audio file.
		model: str - Whisper model identifier.
		Returns:
		--------
		Optional[ str ] - Transcribed text.
		"""
		try:
			throw_if( 'path', path );
			self.audio_file = path;
			self.model = model
			with open( self.audio_file, 'rb' ) as audio:
				self.response = self.client.audio.transcriptions.create( file=(self.audio_file,
				                                                               audio.read( )), model=self.model, response_format="text" )
			self.transcript = str( self.response );
			return self.transcript
		except Exception as e:
			ex = Error( e );
			ex.module = 'grok';
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( ex );
			error.show( )

class Translation( Grok ):
	"""

	    Purpose
	    ___________
	    Class for translating foreign audio speech into English text via Groq Whisper.

	    Attributes:
	    -----------
	    target_language       : str - English (Default)
	    client                : Groq - API Client

	    Methods:
	    --------
	    translate( path, model ) : Translates audio to English text

    """
	target_language: Optional[ str ]
	client: Optional[ Groq ]
	audio_file: Optional[ Any ]
	response: Optional[ Any ]
	voice: Optional[ str ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	number: Optional[ int ]
	top_percent: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	audio_path: Optional[ str ]
	messages: Optional[ List[ Dict[ str, str ] ] ]
	stops: Optional[ List[ str ] ]
	completion: Optional[ str ]
	
	def __init__( self, number: int = 1, temperature: float = 0.8, top_p: float = 0.9, frequency: float = 0.0,
			presence: float = 0.0, max_tokens: int = 4096, store: bool = True, stream: bool = True,
			instruct: str = None ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key )
		self.model = 'whisper-large-v3';
		self.number = number;
		self.temperature = temperature
		self.top_percent = top_p;
		self.frequency_penalty = frequency
		self.presence_penalty = presence;
		self.max_tokens = max_tokens
		self.store = store;
		self.stream = stream;
		self.instructions = instruct
		self.audio_file = None;
		self.response = None;
		self.voice = None
		self.audio_path = None;
		self.messages = None;
		self.stops = None
		self.completion = None;
		self.target_language = 'English'
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Returns list of translation-capable models on Groq."""
		return [ 'whisper-large-v3',
		         'whisper-large-v3-turbo' ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Returns the primary target language for Groq translation."""
		return [ 'English' ]
	
	def translate( self, path: str, model: str = 'whisper-large-v3' ) -> Optional[ str ]:
		"""
		Purpose: Translates speech from a foreign audio file into English text.
		Parameters:
		-----------
		path: str - Path to local audio.
		model: str - Whisper model ID.
		Returns:
		--------
		Optional[ str ] - Translated text.
		"""
		try:
			throw_if( 'path', path );
			self.audio_path = path;
			self.model = model
			with open( self.audio_path, 'rb' ) as audio:
				self.response = self.client.audio.translations.create( file=(self.audio_path,
				                                                             audio.read( )), model=self.model )
			return self.response.text
		except Exception as e:
			ex = Error( e );
			ex.module = 'grok';
			ex.cause = 'Translation'
			ex.method = 'translate( self, path: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( ex );
			error.show( )

class Image( Grok ):
	'''

	    Purpose
	    ___________
	    Class handling vision analysis (Groq) and hybrid image generation/editing.

	    Attributes:
	    -----------
	    image_url             : str - URL of the generated or analyzed image
	    size                  : str - Dimensions of the image (e.g., 1024x1024)

	    Methods:
	    --------
	    generate( prompt, model, quality, size ) : Generates images via Hybrid POST
	    analyze( text, path, model )             : Analyzes image content via Groq Vision LPU
	    edit( prompt, path, model )              : Edits local image via Hybrid POST

    '''
	image_url: Optional[ str ]
	quality: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
	tool_choice: Optional[ str ]
	style: Optional[ str ]
	response_format: Optional[ str ]
	client: Optional[ Groq ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	number: Optional[ int ]
	top_percent: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	input: Optional[ List[ Any ] ]
	input_text: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ Any ]
	stops: Optional[ List[ str ] ]
	messages: Optional[ List[ Dict[ str, str ] ] ]
	completion: Optional[ str ]
	
	def __init__( self, n: int = 1, temperature: float = 0.8, top_p: float = 0.9,
			frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 4096,
			store: bool = False, stream: bool = False ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key )
		self.number = n;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.store = store;
		self.stream = stream
		self.tool_choice = 'auto';
		self.input_text = None;
		self.file_path = None
		self.image_url = None;
		self.quality = 'standard';
		self.size = '1024x1024'
		self.style = 'natural';
		self.response_format = 'url';
		self.input = None
		self.detail = None;
		self.response = None;
		self.stops = None;
		self.messages = None
		self.completion = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Returns the Llama 3.2 Vision models currently available on Groq."""
		return [ 'llama-3.2-90b-vision-preview',
		         'llama-3.2-11b-vision-preview' ]
	
	@property
	def gen_model_options( self ) -> List[ str ]:
		"""Returns list of image generation models (Hybrid/OpenAI)."""
		return [ 'dall-e-3',
		         'dall-e-2' ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Returns list of supported image sizes."""
		return [ '1024x1024',
		         '1024x1792',
		         '1792x1024',
		         '512x512',
		         '256x256' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Returns list of available image response formats."""
		return [ 'url',
		         'b64_json' ]
	
	def generate( self, prompt: str, model: str = 'dall-e-3', quality: str = 'standard',
			size: str = '1024x1024' ) -> Optional[ str ]:
		"""
		Purpose: Generates a new image based on prompt via Hybrid POST.
		Parameters:
		-----------
		prompt: str - Image description.
		model: str - Hybrid model ID.
		quality: str - Quality setting.
		size: str - Dimensions.
		Returns:
		--------
		Optional[ str ] - URL.
		"""
		try:
			throw_if( 'text', prompt );
			self.input_text = prompt;
			self.model = model
			self.quality = quality;
			self.size = size
			endpoint = Endpoints( ).image_generations
			headers = Header( key=cfg.OPENAI_API_KEY ).get_header( )
			payload = {
					"model": self.model,
					"prompt": self.input_text,
					"size": self.size,
					"quality": self.quality,
					"n": self.number }
			resp = requests.post( endpoint, headers=headers, json=payload )
			if resp.status_code == 200:
				self.response = resp.json( );
				return self.response[ 'data' ][ 0 ][ 'url' ]
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Image'
			exception.method = 'generate( self, prompt: str, model: str, quality: str, size: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def analyze( self, text: str, path: str, model: str = 'llama-3.2-90b-vision-preview' ) -> \
	Optional[ str ]:
		"""
		Purpose: Analyzes image contents using Groq Vision.
		Parameters:
		-----------
		text: str - Query about the image.
		path: str - Local image path.
		model: str - Vision model ID.
		Returns:
		--------
		Optional[ str ] - Analysis text.
		"""
		try:
			throw_if( 'text', text );
			throw_if( 'path', path )
			self.input_text = text;
			self.file_path = path;
			self.model = model
			base64_image = encode_image( self.file_path )
			messages = [ {
					             "role": "user",
					             "content": [ {
							                          "type": "text",
							                          "text": self.input_text },
					                          {
							                          "type": "image_url",
							                          "image_url": {
									                          "url": f"data:image/jpeg;base64,{base64_image}" } } ] } ]
			self.response = self.client.chat.completions.create( model=self.model, messages=messages )
			self.completion = self.response.choices[ 0 ].message.content
			return self.completion
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Image'
			exception.method = 'analyze( self, text: str, path: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )
	
	def edit( self, prompt: str, path: str, model: str = 'dall-e-2' ) -> Optional[ str ]:
		"""
		Purpose: Edits a local image using a hybrid image service.
		Parameters:
		-----------
		prompt: str - Edit instructions.
		path: str - Local source image.
		model: str - Editing model ID.
		Returns:
		--------
		Optional[ str ] - Edited image URL.
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'path', path )
			self.input_text = prompt;
			self.file_path = path;
			self.model = model
			endpoint = Endpoints( ).image_edits
			headers = {
					"Authorization": f"Bearer {cfg.OPENAI_API_KEY}" }
			files = {
					"image": open( self.file_path, "rb" ) }
			data = {
					"prompt": self.input_text,
					"n": self.number,
					"size": self.size,
					"model": self.model }
			resp = requests.post( endpoint, headers=headers, files=files, data=data )
			if resp.status_code == 200:
				return resp.json( )[ 'data' ][ 0 ][ 'url' ]
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Image'
			exception.method = 'edit( self, prompt: str, path: str, model: str ) -> Optional[ str ]';
			error = ErrorDialog( exception );
			error.show( )