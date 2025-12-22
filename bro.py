'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                bro.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="bro.py" company="Terry D. Eppler">

	     bro.py
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
    bro.py
  </summary>
  ******************************************************************************************
'''
import os

import requests

from app import temperature
from boogr import ErrorDialog, Error
import config as cfg
import google
from google import genai
from google.genai import types
from pathlib import Path
from PIL import Image
from requests import Response
from typing import Any, List, Optional, Dict

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Gemini( ):
	'''
	
		Purpose:
		-------
		Base class for Gemma AI Functionality
		
	'''
	project_id: Optional[ str ]
	api_key: Optional[ str ]
	cloud_location: Optional[ str ]
	instructions: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	content_config: Optional[ types.GenerateContentConfig ]
	image_config: Optional[ types.GenerateImagesConfig ]
	function_config: Optional[ types.FunctionCallingConfig ]
	candidate_count: Optional[ int ]
	modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	
	def __init__( self ):
		self.api_key = cfg.GOOGLE_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT
		self.cloud_location = cfg.GOOGLE_CLOUD_LOCATION
		self.model = None
		self.content_config = None
		self.image_config = None
		self.api_version = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.candidate_count = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None

class Chat( Gemini ):
	'''

	    Purpose:
	    _______
	    Class containing lists of OpenAI models by generation

    '''
	use_vertex: Optional[ bool ]
	http_options: Optional[ types.HttpOptions ]
	client: Optional[ genai.Client ]
	contents: Optional[ List[ str ] ]
	response: Optional[ Response ]
	image_uri: Optional[ str ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, model: str='gemini-2.5-flash', version: str='v1alpha',
			use_ai: bool=True, temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			candidates: int=1, instruct: str=None, contents: List[ str ]=None ):
		super( ).__init__( )
		self.model = model
		self.api_version = version
		self.top_p = top_p
		self.temperature = temperature
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.candidate_count = candidates
		self.max_tokens = max_tokens
		self.use_vertex = use_ai
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = genai.Client( vertexai=self.use_ai, api_key=self.api_key,
			project=self.project_id, location=self.cloud_location, http_options=self.http_options )
		self.contents = contents
		self.instructions = instruct
		self.response_modalities = [ 'TEXT', 'IMAGE' ]
		self.content_config = None
		self.image_config = None
		self.function_config = None
		self.response = None
		self.image_uri = None
		self.file_path = None
		
	@property
	def model_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			_______
			List[ str ] - list of available models

		'''
		return [ 'gemini-3-flash-preview',
		         'gemini-2.5-flash',
		         'gemini-2.5-flash-lite',
		         'gemini-2.5-flash-image',
		         'gemini-2.5-flash-native-audio-preview-12-2025',
		         'gemini-2.5-flash-tts',
		         'gemini-2.5-flash-lite-preview-tts',
		         'gemini-2.0-flash-001',
		         'gemini-2.0-flash-lite',
		         'gemini-2.5-computer-use-preview-10-2025',
		         'translate-llm',
		         'imagen-3.0-capability-002',
		         'imagen-4.0-ultra-generate-preview-06-06',
		         'imagen-4.0-generate-001',
		         'imagen-4.0-ultra-generate-001',
		         'imagen-4.0-fast-generate-001', ]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ '1:1',
		         '2:3',
		         '3:2',
		         '3:4',
		         '4:3',
		         '9:16',
		         '16:9',
		         '21:9' ]
	
	@property
	def size_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ '1K',
		         '2K',
		         '4K' ]
	
	@property
	def version_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			--------
			List[ str ] - list of available api versions
			
		'''
		return [ 'v1', 'v1alpha', 'v1beta1' ]
		
	def generate_text( self, prompt: str, model: str='gemini-2.5-flash' ) -> str | None:
		try:
			throw_if( 'propmpt', prompt )
			self.contents = prompt
			self.model = model
			self.content_config = types.GenerateContentConfig( temperature=self.temperature,
				top_p=self.top_p, max_output_tokens=self.max_tokens,
				candidate_count=self.candidate_count, frequency_penalty=self.frequency_penalty,
				presence_penalty=self.presence_penalty, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str, model: str ) -> str:'
			error = ErrorDialog( exception )
			error.show( )

	def generate_image( self, prompt: str, model: str='gemini-2.5-flash-image' ) -> str | None:
		try:
			throw_if( 'propmpt', prompt )
			self.contents = prompt
			self.model = model
			self.image_config = types.GenerateImagesConfig( http_options=self.http_options,)
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = 'generate_image( self, prompt: str, model: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )

	def analyze_image( self, prompt: str, filepath: str, model: str='gemini-2.5-flash-image' ) -> str | None:
		try:
			throw_if( 'propmpt', prompt )
			throw_if( 'filepath', filepath )
			self.contents = prompt
			self.filepath = filepath
			self.model = model
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def summarize_document( self, prompt: str, filepath: str, model: str='gemini-2.5-flash' ) -> str | None:
		try:
			throw_if( 'propmpt', prompt )
			throw_if( 'filepath', filepath )
			self.contents = prompt
			self.filepath = filepath
			self.model = model
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def search_file( self, prompt: str, file_id:str ) -> str | None:
		try:
			throw_if( 'propmpt', prompt )
			throw_if( 'file_id', file_id )
			self.contents = prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
		
	def upload_file( self, prompt: str, file_id: str ) -> str | None:
		try:
			throw_if( 'propmpt', prompt )
			throw_if( 'file_id', file_id )
			self.contents = prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

	def retreive_file( self, prompt: str, file_id: str ) -> str | None:
		try:
			throw_if( 'propmpt', prompt )
			throw_if( 'file_id', file_id )
			self.contents = prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

	def list_files( self, purpose: str ) -> str | None:
		try:
			throw_if( 'purpose', purpose )
			self.contents = purpose
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

	def delete_file( self, file_id: str ) -> str | None:
		try:
			throw_if( 'file_id', file_id )
			self.contents = file_id
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )

class Embedding( Gemini ):
	'''
		
		Purpose:
		--------
		Class providing embedding functionality
		
		
	'''
	client: Optional[ genai.Client ]
	response: Optional[ Response ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	use_vertex: Optional[ bool ]
	task_type: Optional[ str ]
	http_options: Optional[ Dict[ str, Any ] ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	content_config: Optional[ types.GenerateContentConfig ]
	client: Optional[ genai.Client ]
	contents: Optional[ List[ str ] ]
	input_text: Optional[ str ]
	
	def __init__( self, model: str='gemini-embedding-001', version: str='v1alpha',
			use_ai: bool=True, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000 ):
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.model = model
		self.version = version
		self.use_ai = use_ai
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = genai.Client( vertexai=self.use_ai, api_key=self.api_key,
			project=self.project_id, location=self.cloud_location, http_options=self.http_options )
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.contents = [ ]
		self.encoding_format = None
		self.input_text = None
		self.content_config = None
		self.model = None
		self.embedding = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		'''
			
			Returns:
			--------
			List[ str ] of embedding models

		'''
		return [ 'gemini-embedding-001',
		         'text-embedding-005',
		         'text-multilingual-embedding-002',
		         'multilingual-e5-small',
		         'multilingual-e5-large', ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		'''
			
			Returns:
			--------
			List[ str ] of available format options

		'''
		return [ 'float',
		         'base64' ]
	
	def embed( self, text: str, model: str='gemini-embedding-001', format: str='float' ) -> List[ float ] | None:
		"""
	
	        Purpose
	        _______
	        Creates an embedding ginve a text
	
	
	        Parameters
	        ----------
	        text: str
	
	
	        Returns
	        -------
	        get_list[ float

        """
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.model = model
			self.encoding_format = format
			self.response = self.client.embeddings.create( input=self.input, model=self.model,
				encoding_format=self.encoding_format )
			self.embedding = self.response.data[ 0 ].embedding
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'Embedding'
			exception.method = 'create( self, text: str, model: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def count_tokens( self, text: str, coding: str ) -> int:
		'''

	        Purpose:
	        -------
	        Returns the num of words in a documents path.
	
	        Parameters:
	        -----------
	        text: str - The string that is tokenized
	        coding: str - The encoding to use for tokenizing
	
	        Returns:
	        --------
	        int - The number of words

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'coding', coding )
			return 0
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> int'
			error = ErrorDialog( exception )
			error.show( )

class TTS( Gemini ):
	"""

	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (TTS)


	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True

	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model,
	    self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.descriptions, self.assistants

	    Methods
	    ------------
	    get_model_options( self ) -> str
	    create_small_embedding( self, prompt: str, path: str )

    """
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ requests.Response ]
	client: Optional[ genai.Client ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True, instruct: str=None ):
		'''

	        Purpose:
	        --------
	        Constructor to  create_small_embedding TTS objects

        '''
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.client = genai.Client( api_key=cfg.GOOGLE_API_KEY )
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.model = None
		self.audio_path = None
		self.response = None
		self.response_format = None
		self.speed = None
		self.voice = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of tts model names

        '''
		return [ 'gemini-2.5-flash-tts',
		         'gemini-2.5-flash-lite-preview-tts', ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ 'en-US',
		         'es-MX',
		         'fr-FR',
		         'ceb-PH',
		         'ja-JP',
		         'pt-PT',
		         'la-VA',
		         'he-IL',
		         'el-GR',
		         'fil-PH',
		         'ru-RU',
		         'ar-001',
		         'cmn-CN' ]

	@property
	def voice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available voices for TTS

		'''
		return [ 'Achernar',
		         'Achird',
		         'Aoede',
		         'Enceladus',
		         'Erinome',
		         'Iapetus',
		         'Kore',
		         'Orus',
		         'Pulcherrima',
		         'Puck',
		         'Zephyr' ]
	
	@property
	def output_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ 'ALAW',
		         'MULAW',
		         'MP3',
		         'OGG_OPUS',
		         'PCM', ]
	
	def create_audio( self, text: str, filepath: str, format: str='MP3',
			speed: float=1.0, voice: str='Kore' ) -> str:
		"""

	        Purpose
	        _______
	        Generates audio given a text prompt less than
	        4096 characters and a path to audio file


	        Parameters
	        ----------
	        prompt: str
	        path: str


	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'text', text )
			throw_if( 'filepath', filepath )
			self.input_text = text
			self.speed = speed
			self.response_format = format
			self.voice = voice
			out_path = Path( filepath )
			if not out_path.parent.exists( ):
				out_path.parent.mkdir( parents=True, exist_ok=True )
			with self.client.audio.speech.with_streaming_response.create( model=self.model, speed=self.speed,
					voice=self.voice, response_format=self.response_format, input=self.input_text ) as resp:
				resp.stream_to_file( str( out_path ) )
			return str( out_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'TTS'
			exception.method = 'create_audio( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method returns a list of strings representing members

	        Parameters:
	        ----------
	        self

	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'system_instructions',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'content',
		         'prompt',
		         'response',
		         'completion',
		         'file',
		         'path',
		         'messages',
		         'image_url',
		         'response_format',
		         'tools',
		         'vector_store_ids',
		         'name',
		         'id',
		         'description',
		         'generate_text',
		         'get_format_options',
		         'get_model_options',
		         'reasoning_effort',
		         'get_effort_options',
		         'get_speed_options',
		         'input_text',
		         'metadata',
		         'get_files',
		         'get_data', ]

class Transcription( Gemini ):
	"""

	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (whisper-1)


	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True

	    Attributes
	    -----------
	    api_key, 
	    system_instructions, 
	    client, 
	    small_model, 
	    reasoning_effort,
	    response, 
	    num, 
	    temperature, 
	    top_percent,
	    frequency_penalty, 
	    presence_penalty, s
	    elf.max_completion_tokens,
	    store, 
	    stream, 
	    modalities, 
	    stops, 
	    content,
	    input_text, 
	    response, 
	    completion, 
	    audio_file, 
	    transcript


	    Methods
	    ------------
	    get_model_options( self ) -> str
	    create_small_embedding( self, path: str  ) -> str


    """
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True, instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.client = genai.Client( api_key=cfg.GOOGLE_API_KEY )
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.model = None
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
	
	@property
	def model_options( self ) -> str:
		'''


	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'whisper-1',
		         'gpt-4o-mini-transcribe',
		         'gpt-4o-transcribe',
		         'gpt-4o-transcribe-diarize' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ 'en-US',
		         'es-MX',
		         'fr-FR',
		         'ceb-PH',
		         'ja-JP',
		         'pt-PT',
		         'la-VA',
		         'he-IL',
		         'el-GR',
		         'fil-PH',
		         'ru-RU',
		         'ar-001',
		         'cmn-CN' ]
	
	@property
	def output_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ 'ALAW',
		         'MULAW',
		         'MP3',
		         'OGG_OPUS',
		         'PCM', ]
	
	def transcribe( self, path: str, model: str='whisper-1' ) -> str:
		"""

            Transcribe audio with Whisper.

        """
		try:
			throw_if( 'path', path )
			self.model = model
			with open( path, 'rb' ) as self.audio_file:
				resp = self.client.audio.transcriptions.create( model=self.model,
					file=self.audio_file )
			return resp.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'boo'
			ex.cause = 'Transcription'
			ex.method = 'transcribe(self, path)'
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method returns a list of strings representing members

	        Parameters:
	        ----------
	        self

	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'audio_file',
		         'messages',
		         'response_format',
		         'api_key',
		         'client',
		         'input_text',
		         'transcript', ]

class Translation( Gemini ):
	"""

	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (whisper-1)


	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True

	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model,  self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.descriptions, self.assistants

	    Methods
	    ------------
	    create_small_embedding( self, prompt: str, path: str )

    """
	target_language: Optional[ str ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True, instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = genai.Client( api_key=self.api_key )
		self.model = 'whisper-1'
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.audio_file = None
		self.response = None
		self.voice = None
	
	@property
	def model_options( self ) -> str:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'whisper-1',
		         'text-davinci-003',
		         'gpt-4-0613',
		         'gpt-4-0314',
		         'gpt-4-turbo-2024-04-09', ]
	
	@property
	def language_options( self ):
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'English',
		         'Spanish',
		         'Tagalog',
		         'French',
		         'Japanese',
		         'German',
		         'Italian',
		         'Chinese' ]
	
	@property
	def voice_options( self ):
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'alloy',
		         'ash',
		         'ballad',
		         'coral',
		         'echo',
		         'fable',
		         'onyx',
		         'nova',
		         'sage',
		         'shiver', ]
	
	def create( self, text: str, path: str ) -> str | None:
		"""

	        Purpose
	        _______
	        Generates a translation given a string to an audio file


	        Parameters
	        ----------
	        text: str
	        path: str


	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			with open( path, 'rb' ) as audio_file:
				resp = self.client.audio.translations.create( model='whisper-1',
					file=audio_file, prompt=text )
			return resp.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'Translation'
			exception.method = 'create( self, text: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def translate( self, path: str ) -> str | None:
		"""

            Translate non-English speech to English with Whisper.

        """
		try:
			throw_if( 'path', path )
			with open( path, 'rb' ) as audio_file:
				resp = self.client.audio.translations.create( model=self.model, file=audio_file )
			return resp.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'boo'
			ex.cause = 'Translation'
			ex.method = 'translate(self, path)'
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method returns a list of strings representing members

	        Parameters:
	        ----------
	        self

	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'completion',
		         'audio_path',
		         'path',
		         'messages',
		         'response_format',
		         'tools',
		         'api_key',
		         'client',
		         'model',
		         'translate',
		         'model_options', ]

class Image( Gemini ):
	"""

	    Purpose
	    ___________
	    Class used for generating images Google's Imagen3/4 API


	    Parameters
	    ------------
	    n: int=1
	    temperature: float=0.8
	    top_p: float=0.9
	    frequency: float=0.0
	    presence: float=0.0
	    max_tokens: int=10000
	    store: bool=True
	    stream: bool=True

	    Attributes
	    -----------
	    self.api_key, self.client, self.small_model,  self.embedding,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.prompt, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.input_text, self.image_url

		Properties:
		----------
	    detail_options( self ) -> list[ str ]
	    format_options( self ) -> list[ str ]
	    size_options( self ) -> list[ str ]
	    model_options( self ) -> str

	    Methods
	    ------------
	    generate( self, path: str ) -> str
	    analyze( self, path: str, text: str ) -> str

    """
	image_url: Optional[ str ]
	quality: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
	tool_choice: Optional[ str ]
	style: Optional[ str ]
	response_format: Optional[ str ]
	
	def __init__( self, n: int = 1, temperture: float = 0.8, top_p: float = 0.9, frequency: float = 0.0,
			presence: float = 0.0, max_tokens: int = 10000, store: bool = False, stream: bool = False, ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = genai.Client( api_key=self.api_key )
		self.number = n
		self.temperature = temperture
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.tool_choice = 'auto'
		self.input = [ ]
		self.input_text = None
		self.file_path = None
		self.image_url = None
		self.quality = None
		self.detail = None
		self.model = None
		self.size = None
		self.style = None
		self.response_format = None
	
	@property
	def style_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Methods that returns a list of style options for dall-e-3

        '''
		return [ 'vivid',
		         'natural', ]
	
	@property
	def model_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Methods that returns a list of small_model names

        '''
		return [ 'dall-e-2',
		         'dall-e-3',
		         'gpt-4o-mini',
		         'gpt-4o',
		         'gpt-image-1',
		         'gpt-image-1-mini',
		         'gpt-image-1.5', ]
	
	@property
	def size_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Method that returns a  list of sizes

	        - For gpt-image-1, the size must be one of '1024x1024', '1536x1024' (landscape),
	        '1024x1536' (portrait), or 'auto' (default value).

	        - For dall-e-2, the size must be one of '256x256', '512x512', or '1024x1024'

	        - For dall-e-3, the sie must be one of '1024x1024', '1792x1024', or '1024x1792'

        '''
		return [ 'auto',
		         '256x256',
		         '512x512',
		         '1024x1024',
		         '1536x1024',
		         '1024x1792',
		         '1792x1024' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ '.png',
		         '.jpeg',
		         '.webp',
		         '.gif' ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of quality options

        '''
		return [ 'low',
		         'medium',
		         'hi', ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of detail options

        '''
		return [ 'auto',
		         'low',
		         'high' ]
	
	def generate( self, prompt: str, model: str, quality: str, size: str,
			style: str = 'natural', format: str = 'url' ) -> str:
		"""

                Purpose
                _______
                Generate an image from a text prompt.


                Parameters
                ----------
                prompt: str


                Returns
                -------
                Image object

        """
		try:
			throw_if( 'text', prompt )
			throw_if( 'model', model )
			throw_if( 'quality', quality )
			throw_if( 'size', size )
			self.input_text = prompt
			self.model = model
			self.quality = quality
			self.size = size
			self.style = style
			self.response_format = format
			self.response = self.client.images.generate( model=self.model, prompt=self.input_text,
				size=self.size, style=self.style, response_format=self.response_format,
				quality=self.quality, n=self.number )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'Image'
			exception.method = 'generate( self, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze( self, text: str, path: str, model: str = 'gpt-4o-mini', ) -> str:
		'''

	        Purpose:
	        ________

	        Method providing image analysis functionality given a prompt and path

	        Parameters:
	        ----------
	        input: str
	        path: str

	        Returns:
	        --------
	        str | None

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			self.input_text = text
			self.model = model
			self.file_path = path
			self.input = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'input_text',
											'text': self.input_text },
									{
											'type': 'input_image',
											'image_url': self.file_path },
							],
					} ]
			
			self.response = self.client.responses.create( model=self.model, input=self.input,
				max_output_tokens=self.max_completion_tokens, temperature=self.temperature,
				tool_choice=self.tool_choice, stream=self.stream, store=self.store )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'Image'
			exception.method = 'analyze( self, path: str, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def edit( self, prompt: str, path: str, size: str = '1024x1024' ) -> str:
		"""

	        Purpose
	        _______
	        Method that analyzeses an image given a path prompt,

	        Parameters
	        ----------
	        prompt: str
	        url: str

	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'input', prompt )
			throw_if( 'path', path )
			self.input_text = prompt
			self.file_path = path
			self.response = self.client.images.edit( model=self.model,
				image=open( self.file_path, 'rb' ), prompt=self.input_text, n=self.number,
				size=self.size, )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'bro'
			exception.cause = 'Image'
			exception.method = 'edit( self, text: str, path: str, size: str=1024x1024 ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method returns a list of strings representing members

	        Parameters:
	        ----------
	        self

	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [  # Attributes
				'number',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_completion_tokens',
				'store',
				'stream',
				'modalities',
				'stops',
				'api_key',
				'client',
				'path',
				'input_text',
				'image_url',
				'size',
				'quality',
				'detail',
				'model',
				# Properties
				'style_options',
				'model_options',
				'detail_options',
				'format_options',
				'size_options',
				# Methods
				'generate',
				'analyze',
				'edit', ]
