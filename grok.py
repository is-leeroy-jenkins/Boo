'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                grok.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
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
from pathlib import Path
import groq
from app import temperature
from boogr import ErrorDialog, Error
import config as cfg
from groq import Groq
from requests import Response
from typing import Any, List, Optional, Dict

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class GroqEndpoints:
	'''

	    Purpose:
	    ---------
	    The class containing endpoints for the Groq API

    '''
	base_url: Optional[ str ]
	chat_completions: Optional[ str ]
	responses: Optional[ str ]
	speech_generations: Optional[ str ]
	translations: Optional[ str ]
	transcriptions: Optional[ str ]
	finetunings: Optional[ str ]
	files: Optional[ str ]
	
	def __init__( self ):
		self.base_url = f'https://api.groq.com/'
		self.chat_completions = f'https://api.groq.com/openai/v1/chat/completions'
		self.responses = f'https://api.groq.com/openai/v1/responses'
		self.speech_generation = f'https://api.groq.com/openai/v1/audio/speech'
		self.translations = f'https://api.groq.com/openai/v1/audio/translations'
		self.transcriptions = f'https://api.groq.com/openai/v1/audio/transcriptions'
		self.finetuning = f'https://api.groq.com/v1/fine_tunings'
		self.files = f'https://api.groq.com/openai/v1/files'

class GroqHeader:
	'''

	    Purpose:
	    --------
	    Encapsulates HTTP header stores for Groq API requests.

	    Attributes:
	    -----------
	    content_type : str
	    api_key      : str | None
	    authorization: str
	    stores         : dict[str, str]

    '''
	content_type: Optional[ str ]
	api_key: Optional[ str ]
	authorization: Optional[ str ]
	
	def __init__( self ):
		self.content_type = 'application/json'
		self.api_key = cfg.GROQ_API_KEY
		self.authorization = f'Bearer {cfg.GROQ_API_KEY}'
	
	def __dir__( self ) -> list[ str ] | None:
		return [ 'content_type',
		         'api_key',
		         'authorization',
		         'get_data' ]

class Grok( ):
	'''

		Purpose:
		-------
		Base class for Gemma AI Functionality

	'''
	api_key: Optional[ str ]
	instructions: Optional[ str ]
	model: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	modalities: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ List[ str ] ]
	
	def __init__( self ):
		self.api_key = cfg.GROQ_API_KEY
		self.model = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.candidates = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None

class Chat( Grok ):
	'''

	    Purpose:
	    _______
	    Class containing lists of OpenAI models by generation

    '''
	client: Optional[ Groq ]
	contents: Optional[ List[ str ] ]
	response: Optional[ Response ]
	image_url: Optional[ str ]
	
	def __init__( self, model: str='llama-3.1-8b-instant', temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			instruct: str=None ):
		super( ).__init__( )
		self.model = model
		self.top_p = top_p
		self.temperature = temperature
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = Groq( api_key=self.api_key  )
		self.client = None
		self.response = None
		self.image_url = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

			Returns:
			_______
			List[ str ] - list of available models

		'''
		return [ 'llama-3.1-8b-instant',
		         'llama-3.3-70b-versatile',
		         'meta-llama/llama-guard-4-12b',
		         'meta-llama/llama-4-scout-17b-16e-instruct',
		         'meta-llama/llama-4-maverick-17b-128e-instruct',
		         'openai/gpt-oss-120b',
		         'openai/gpt-oss-20b',
		         'whisper-large-v3',
		         'whisper-large-v3-turbo',
		         'groq/compound',
		         'groq/compound-mini', ]

	def generate_text( self, prompt: str, model: str='llama-3.1-8b-instant' ) -> str | None:
		pass
	
	def generate_image( self, prompt: str, model: str='llama-3.1-8b-instant' ) -> str | None:
		pass
	
	def analyze_image( self, prompt: str, filepath: str, model: str='llama-3.1-8b-instant' ) -> str | None:
		pass
	
	def summarize_document( self, prompt: str, filepath: str, model: str='llama-3.1-8b-instant' ) -> str | None:
		pass
	
	def search_file( self, prompt: str, filepath: str, model: str='llama-3.1-8b-instant' ) -> str | None:
		pass

class Embedding( Grok ):
	'''

		Purpose:
		--------
		Class providing embedding functionality


	'''
	client: Optional[ Groq ]
	response: Optional[ Response ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	input_text: Optional[ str ]
	
	def __init__( self, model: str='llama-3.1-8b-instant', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0, max_tokens: int=10000 ):
		super( ).__init__( )
		self.api_key = cfg.GROQ_API_KEY
		self.model = model
		self.client = Groq( api_key=self.api_key  )
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.contents = [ ]
		self.http_options = { }
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
		return [ '', ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		'''

			Returns:
			--------
			List[ str ] of available format options

		'''
		return [ 'float',
		         'base64' ]
	
	def create( self, text: str, model: str='gemini-embedding-001', format: str='float' ) -> List[ float ] | None:
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
			exception.module = 'groq'
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
			exception.module = 'groq'
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> int'
			error = ErrorDialog( exception )
			error.show( )

class TTS( Grok ):
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
	    self.api_key, self.system_instructions, self.client, self.small_model, self.reasoning_effort,
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
	response: Optional[ Response ]
	client: Optional[ groq.Groq ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True, instruct: str=None ):
		'''

	        Purpose:
	        --------
	        Constructor to  create_small_embedding TTS objects

        '''
		super( ).__init__( )
		self.api_key = cfg.GROQ_API_KEY
		self.client = Groq( api_key=self.api_key  )
		self.model = 'gpt-4o-mini-tts'
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
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
		return [ 'canopylabs/orpheus-v1-english',
		         'canopylabs/orpheus-arabic-saudi',]
	
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
		return [ 'flac', 'mp3', 'mulaw', 'ogg', 'wav' ]

	@property
	def sample_options( self ) -> List[ int ] | None:
		'''
			
			Returns:
			--------
			List[ int ] - speed rate options
			
		'''
		return [ 8000, 16000, 22050, 24000, 32000, 44100, 48000 ]
	
	def create_audio( self, text: str, filepath: str, format: str='wav',
			speed: float=1.0, model: str='canopylabs/orpheus-v1-english' ) -> str:
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
			self.model = model
			out_path = Path( filepath )
			if not out_path.parent.exists( ):
				out_path.parent.mkdir( parents=True, exist_ok=True )
			with self.client.audio.speech.with_streaming_response.create( model=self.model, speed=self.speed,
					voice=self.voice, response_format=self.response_format, input=self.input_text ) as resp:
				resp.stream_to_file( str( out_path ) )
			return str( out_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
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
		         'get_data',
		         'dump', ]

class Transcription( Grok ):
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
	    self.api_key, self.system_instructions, self.client, self.small_model, self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.audio_file, self.transcript


	    Methods
	    ------------
	    get_model_options( self ) -> str
	    create_small_embedding( self, path: str  ) -> str


    """
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True, instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.GROQ_API_KEY
		self.client = Groq( api_key=self.api_key  )
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
		return [ 'whisper-large-v3-turbo',
		         'whisper-large-v3',]

	@property
	def output_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ 'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm' ]
	
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
	def format_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ 'json', 'verbose_json', 'text' ]
	
	def transcribe( self, path: str, model: str='whisper-large-v3-turbo' ) -> str | None:
		"""

            Transcribe audio with Whisper.

        """
		try:
			throw_if( 'path', path )
			self.model = model
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

class Translation( Grok ):
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
		self.instructions = instruct
		self.api_key = cfg.GROQ_API_KEY
		self.client = Groq( api_key=self.api_key  )
		self.model = 'whisper-1'
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.audio_file = None
		self.response = None
		self.voice = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'whisper-large-v3',
		         'whisper-large-v3-turbo', ]
	
	@property
	def response_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			List[ str ] - response format options
			
		'''
		return [ 'json', 'verbose_json', 'text' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
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
	def output_options( self ) -> List[ str ] | None:
		'''
		
		Returns:
		--------
		List[ str ] - List of file types used by the Translation API
		
		'''
		return [ 'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm' ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
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
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
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

class Image( Grok ):
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
	
	def __init__( self, n: int=1, temperture: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=False, stream: bool=False, ):
		super( ).__init__( )
		self.api_key = cfg.GROQ_API_KEY
		self.client = Groq( api_key=self.api_key  )
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
		return [ 'meta-llama/llama-4-scout-17b-16e-instruct',
		         'meta-llama/llama-4-maverick-17b-128e-instruct',]
	
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
			exception.module = 'grok'
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
			exception.module = 'grok'
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
			exception.module = 'grok'
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
