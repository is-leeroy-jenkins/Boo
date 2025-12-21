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
	image: Optional[ Image ]
	
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
		self.contents = contents
		self.instructions = instruct
		self.client = None
		self.content_config = None
		self.image_config = None
		self.function_config = None
		self.response = None
		self.image = None
		
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
		         'gemini-2.5-flash-preview-tts',
		         'gemini-2.0-flash-001',
		         'gemini-2.0-flash-lite',
		         'gemini-2.5-computer-use-preview-10-2025',
		         'translate-llm ',
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
		         '9:16',
		         '16:9',
		         '3:4',
		         '4:3', ]
	
	@property
	def size_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			List[ str ] - list of available aspect ratios for Imagen 4

		'''
		return [ '1K',
		         '2K', ]
	
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
		pass

	def analyze_image( self, prompt: str, filepath: str, model: str='gemini-2.5-flash-image' ) -> str | None:
		pass
	
	def summarize_document( self, prompt: str, filepath: str, model: str='gemini-2.5-flash' ) -> str | None:
		pass
	
	def search_file( self, prompt: str, file_id:str ) -> str | None:
		pass
		
	def upload_file( self, prompt: str, file_id: str ) -> str | None:
		pass

	def retreive_file( self, prompt: str, file_id: str ) -> str | None:
		pass

	def list_files( self ) -> str | None:
		pass

	def delete_files( self ) -> str | None:
		pass

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
		self.client = genai.Client( vertexai=self.use_ai, api_key=self.api_key,
			project=self.project_id, location=self.cloud_location )
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
	