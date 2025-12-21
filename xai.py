'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                xai.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="xai.py" company="Terry D. Eppler">

	     xai.py
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
    xai.py
  </summary>
  ******************************************************************************************
'''

import os

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

class XAI( ):
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

class Chat( XAI ):
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

class Embedding( XAI ):
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
		self.api_key = cfg.GOOGLE_API_KEY
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
	
	def create( self, text: str, model: str = 'gemini-embedding-001', format: str = 'float' ) -> \
	List[ float ] | None:
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