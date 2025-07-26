'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                Boo.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="boo.py" company="Terry D. Eppler">

	     Boo is a df analysis tool integrating various Generative AI, Text-Processing, and
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
    Boo.py
  </summary>
  ******************************************************************************************
  '''
import asyncio
from boogr import ErrorDialog, Error, ChatBot, ChatWindow
import datetime as dt
import numpy as np
import os
from openai import OpenAI, AssistantEventHandler
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import pandas as pd
from pydantic import BaseModel, Field, validator
from pygments.lexers.csound import newline
import requests
from static import Requests, Roles, Languages
from typing_extensions import override
import tiktoken
from typing import Any, List, Tuple, Optional, Dict
from guro import Prompt


class Prompt( BaseModel ):
	'''

		Purpose:
		--------
		Class for the user's location

	'''
	instruction: Optional[ str ]
	context: Optional[ str ]
	output_indicator: Optional[ str ]
	input_data: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class UserLocation( BaseModel ):
	'''

		Purpose:
		--------
		Class for the user's location
		
	'''
	type: Optional[ str ]
	city: Optional[ str ]
	country: Optional[ str ]
	region: Optional[ str ]
	timezone: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class Text( BaseModel ):
	'''

		Purpose:
		--------
		A class used to generate text responses.
		 
	'''
	type: str='text'

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class File( BaseModel ):
	'''

		Purpose:
		--------
		A class used to represent File objects.
		 
	'''
	filename: Optional[ str ]
	bytes: Optional[ int ]
	created_at: Optional[ int ]
	expires_at: Optional[ int ]
	id: Optional[ str ]
	object: Optional[ str ]
	purpose: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class Error( BaseModel ):
	'''

		Purpose:
		--------
		A class for exceptions
	
	'''
	code: Optional[ str ]
	message: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class JsonSchema( BaseModel ):
	'''

		Purpose:
		--------
		A class used to generate json schema responses.
		
	'''
	type: str
	schema: str
	name: str
	description: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class JsonObject( BaseModel ):
	'''

		Purpose:
		--------
		A class used to generate json schema responses.
		
	'''
	type: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class Format( BaseModel ):
	'''

		Purpose:
		--------
		A class for objects specifying the format that the model must output.
		 
	'''
	text: Optional[ Text ]
	json_schema: Optional[ JsonSchema ]
	json_object: Optional[ JsonObject ]

	class Config:
		arbitrary_types_allowed = True

	
class Reasoning( BaseModel ):
	'''

		Purpose:
		--------
		Class providing reasoning functionality.

	'''
	effort: Optional[ str ]
	summary: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class Response( BaseModel ):
	'''

		Purpose:
		--------
		Base class for GPT responses.

	'''
	id: Optional[ str ]
	object: Optional[ object ]
	input: Optional[ str ]
	model: Optional[ str ]
	include: Optional[ List[ str ] ]
	instructions: Optional[ str ]
	max_output_tokens: Optional[ int ]
	previous_response_id: Optional[ int ]
	reasoning: Optional[ object ]
	role: Optional[ str ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	parallel_tool_calls: Optional[ bool ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	top_p: Optional[ int ]
	truncation: Optional[ str ]
	text: Optional[ object ]
	status: Optional[ str ]
	created: Optional[ str ]
	data: Optional[ Dict ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class FileSearch( BaseModel ):
	'''

		Purpose:
		--------
		A tool that searches for relevant content from uploaded file
	
	'''
	type: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	max_num_results: Optional[ int ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class WebSearch( BaseModel ):
	'''

		Purpose:
		--------
		Class for a tool that searches the web
		for relevant results to use in a response.

	'''
	type: Optional[ str ]
	search_context_size: Optional[ str ]
	user_location: Optional[ object ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class ComputerUse( BaseModel ):
	'''

		Purpose:
		--------
		A class for a tool that controls a virtual computer
		
	'''
	type: Optional[ str ]
	display_height: Optional[ int ]
	display_width: Optional[ int ]
	environment: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class Function( BaseModel ):
	'''

		Class for a function the model can choose to call

	'''
	name: Optional[ str ]
	type: Optional[ str ]
	description: Optional[ str ]
	parameters: Optional[ List[ str ] ]
	strict: Optional[ bool ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class Message( BaseModel ):
	'''

		Purpose:
		--------
		Class representing the system message

	'''
	content: str
	role: str
	type: Optional[ str ]
	instructions: Optional[ str ]
	data: Optional[ Dict ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )


class EndPoint( ):
	'''

		Purpose:
		---------
		The class containing endpoints for OpenAI
		
	'''
	
	
	def __init__( self ):
		self.base_url = f'https://api.openai.com/'
		self.text_generation = f'https://api.openai.com/v1/chat/completions'
		self.image_generation = f'https://api.openai.com/v1/images/generations'
		self.chat_completion = f'https://api.openai.com/v1/chat/completions'
		self.responses = f'https://api.openai.com/v1/responses'
		self.speech_generation = f'https://api.openai.com/v1/audio/speech'
		self.translations = f'https://api.openai.com/v1/audio/translations'
		self.assistants = f'https://api.openai.com/v1/assistants'
		self.transcriptions = f'https://api.openai.com/v1/audio/transcriptions'
		self.finetuning = f'https://api.openai.com/v1/fineTuning/jobs'
		self.embeddings = f'https://api.openai.com/v1/embeddings'
		self.uploads = f'https://api.openai.com/v1/uploads'
		self.files = f'https://api.openai.com/v1/files'
		self.vector_stores = f'https://api.openai.com/v1/vector_stores'
		
		
	def get_data( self ) -> Dict[ str, str ] | None:
		'''

			Purpose:
			--------
			Returns: dict[ str ] of members

		'''
		return { 'base_url': self.base_url,
		         'text_generation': self.text_generation,
		         'image_generation': self.image_generation,
		         'chat_completion': self.chat_completion,
		         'responses': self.responses,
		         'speech_generation': self.speech_generation,
		         'translations': self.translations,
		         'assistants': self.assistants,
		         'transcriptions': self.transcriptions,
		         'finetuning': self.finetuning,
		         'vectors': self.embeddings,
		         'uploads': self.uploads,
		         'files': self.files,
		         'vector_stores': self.vector_stores }
		
		
	def dump( self ) -> str:
		'''

			Purpose:
			--------
			Returns: path of "member = value", pairs

		'''
		new = r'\r\n'
		return 'base_url' + f' = {self.base_url}' + new + \
			'text_generation' + f' = {self.text_generation}' + new + \
			'image_generation' + f' = {self.image_generation}' + new + \
			'chat_completion' + f' = {self.chat_completion}' + new + \
			'speech_generation' + f' = {self.speech_generation}' + new + \
			'translations' + f' = {self.translations}' + new + \
			'assistants' + f' = {self.assistants}' + new + \
			'transcriptions' + f' = {self.transcriptions}' + new + \
			'finetuning' + f' = {self.finetuning}' + new + \
			'vectors' + f' = {self.files}' + new + \
			'uploads' + f' = {self.uploads}' + new + \
			'files' + f' = {self.files}' + new + \
			'vector_stores' + f' = {self.vector_stores}' + new


class Header( ):
	'''

		Purpose:
		-------
		Class used to encapsulate GPT headers
		
	'''
	
	
	def __init__( self ):
		self.content_type = 'application/json'
		self.api_key = os.environ.get( 'OPENAI_API_KEY' )
		self.authoriztion = 'Bearer ' + os.environ.get( 'OPENAI_API_KEY' )
		self.data = { 'content-scaler': self.content_type,
		              'Authorization': self.authoriztion }
	
	
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
		return [ 'content_type', 'api_key', 'authorization' ]



class Models( ):
	'''
	
		Purpose:
		_______
		Class containing lists of OpenAI models by generation
		
	'''
	
	def __init__( self ):
		self.text_generation = [ 'text-davinci-003', 'text-curie-001',
		                         'gpt-4-0613', 'gpt-4-0314',
		                         'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
		                         'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
		                         'gpt-4o-mini-2024-07-18', 'gpt-4.1-mini-2025-04-14',
		                         'gpt-4.1-nano-2025-04-14',
		                         'o1-pro-2025-03-19', 'o1-2024-12-17',
		                         'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.image_generation = [ 'dall-e-2', 'dall-e-3',
		                          'gpt-4-0613', 'gpt-4-0314',
		                          'gpt-4o-mini-2024-07-18' ]
		self.chat_completion = [ 'gpt-4-0613', 'gpt-4-0314',
		                         'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
		                         'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
		                         'gpt-4o-mini-2024-07-18', 'gpt-4.1-mini-2025-04-14',
		                         'gpt-4.1-nano-2025-04-14', 'o1-2024-12-17',
		                         'o1-mini-2024-09-12', 'o3-mini-2025-01-31',
		                         'gpt-4o-search-preview-2025-03-11',
		                         'gpt-4o-mini-search-preview-2025-03-11' ]
		self.speech_generation = [ 'tts-1', 'tts-1-hd', 'gpt-4o-mini-tts',
		                           'gpt-4o-audio-preview-2024-12-17',
		                           'gpt-4o-audio-preview-2024-10-01',
		                           'gpt-4o-mini-audio-preview-2024-12-17' ]
		self.transcription = [ 'whisper-1', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe' ]
		self.translation = [ 'whisper-1', 'text-davinci-003',
		                     'gpt-4-0613', 'gpt-4-0314',
		                     'gpt-4-turbo-2024-04-09' ]
		self.responses = [ 'gpt-4o-mini-search-preview-2025-03-11',
		                   'gpt-4o-search-preview-2025-03-11',
		                   'computer-use-preview-2025-03-11' ]
		self.reasoning  = [ 'o1-2024-12-17', 'o1-mini-2024-09-12',
		                    'o3-mini-2025-01-31', 'o1-pro-2025-03-19' ]
		self.finetuning = [ 'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18',
		                    'gpt-4-0613', 'gpt-3.5-turbo-0125',
		                    'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613' ]
		self.embeddings = [ 'text-embedding-3-small', 'text-embedding-3-large',
		                    'text-embedding-ada-002' ]
		self.uploads = [ 'gpt-4-0613', 'gpt-4-0314', 'gpt-4-turbo-2024-04-09',
		                 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20',
		                 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18',
		                 'o1-2024-12-17', 'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.files = [ 'gpt-4-0613', 'gpt-4-0314', 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20',
		               'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18',
		               'o1-2024-12-17', 'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.vectorstores = [ 'gpt-4-0613', 'gpt-4-0314', 'gpt-4-turbo-2024-04-09',
		                       'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
		                       'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
		                       'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.bubba = \
		[
			'ft:gpt-4.1-2025-04-14:leeroy-jenkins:budget-execution-gpt-4-1-2025-20-05:BZO7tKJy',
			'ft:gpt-4.1-nano-2025-04-14:leeroy-jenkins:bubba-gpt-4-1-nano-2025-29-05:BcfJJTtv',
			'ft:gpt-4.1-mini-2025-04-14:leeroy-jenkins:bubba-gpt-4-1-mini-2025-05-05:BcekjucJ',
			'ft:gpt-4o-mini-2024-07-18:leeroy-jenkins:bubba-gpt-4o-mini-2025-30-05:BcrX4S0l',
	        'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-base-training:BGVAJg57' ]

		self.bro = \
		[
			'ft:gpt-4.1-2025-04-14:leeroy-jenkins:bro-gpt-4-1-df-analysis-2025-21-05:BZetxEQa',
			'ft:gpt-4.1-nano-2025-04-14:leeroy-jenkins:bro-gpt-4-1-nano-2025-29-05:BchzJVjL',
			'ft:gpt-4.1-mini-2025-04-14:leeroy-jenkins:bro-gpt-4-1-mini-2025-29-05:BcgMfu1w',
			'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-fine-tuned:BTc3PMb5',
			'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-analytics:BTX4TYqY' ]
	
	
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
		return [ 'text_generation', 'image_generation', 'chat_completion',
		         'speech_generation', 'responses', 'reasoning', 'translations', 'assistants',
		         'transcriptions', 'finetuning', 'vectors', 'uploads', 'files', 'vectorstores',
		         'bubba_instructions', 'bro_instructions', 'get_data', ]
	
	
	def get_data( self ) -> Dict[ str, List[ str ] ] | None:
		'''

			Purpose:
			--------
			Method that returns a dictionary of a list of strings.
		
		'''
		_data = { 'text_generation': self.text_generation,
		          'image_generation': self.image_generation,
		          'chat_completion': self.chat_completion,
		          'speech_generation': self.speech_generation,
		          'translations': self.translation,
		          'finetuning': self.finetuning,
		          'vectors': self.embeddings,
		          'uploads': self.uploads,
		          'files': self.files,
		          'vectorstores': self.vectorstores }
		return _data


class AI( ):
	'''

		Purpose:
		--------
		AI is the base class for all OpenAI functionalityl
	
	'''
	
	
	def __init__( self ):
		self.header = Header( )
		self.endpoint = EndPoint( )
		self.prompt = Prompt( )
		self.api_key = self.header.api_key
		self.client = OpenAI( api_key=self.api_key )
		self.bro_instructions = prompt.data_bro
		self.bubba_instructions = prompt.budget_analyst


class Chat( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with OpenAI's
		Chat Completions API
		
		
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
		
		self.self.num, self.self.temperature, self.self.top_percent,
		self.frequency_penalty, self.presence_penalty,
		self.store, self.stream, self.maximum_completion_tokens,
		self.api_key, self.client, self.small_model,  self.embedding,
		self.response, self.num, self.temperature, self.top_percent,
		self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
		self.store, self.stream, self.modalities, self.stops, self.content,
		self.prompt, self.response, self.completion, self.file, self.path,
		self.path, self.messages, self.image_url, self.response_format,
		self.tools, self.vector_store_ids
		
		Methods
		------------
		get_model_options( self ) -> list[ str ]
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize_document( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		dump( self ) -> str
		get_data( self ) -> { }
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'gpt-4o-mini'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.tool_choice = None
		self.content = None
		self.prompt = None
		self.response = None
		self.completion = None
		self.file = None
		self.file_path = None
		self.input = [ ]
		self.messages = [ ]
		self.image_url = None
		self.response_format = 'auto'
		self.tools = [ ]
		self.vector_stores = \
		{
			'Apporopriations': 'vs_8fEoYp1zVvk5D8atfWLbEupN',
			'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
			'Code': 'vs_67e83bdf8abc81918bda0d6b39a19372',
			'Hawaii': 'vs_67a777291d548191b9fa42956a7f6cb9'
		}
	
	
	def get_model_options( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Methods that returns a list of small_model names
		
		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
	
	
	def generate_text( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Generates a chat completion given a prompt
			
			
			Parameters
			----------
			prompt: str
			
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.prompt = prompt
				self.response = self.client.responses.create( model=self.model, input=self.prompt )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def generate_image( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Generates an image given a prompt
			
			
			Parameters
			----------
			prompt: str
			
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.prompt = prompt
				self.response = self.client.images.generate( model='dall-e-3',
					prompt=self.prompt, size='1024x1024', quality='standard', n=1 )

			return self.response.data[0].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_image( self, prompt: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )
		
		
	def analyze_image( self, prompt: str, url: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a  prompt.

			Parameters
			----------
			prompt: str
			url: str
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif url is None:
				raise Exception( 'Argument "url" cannot be None' )
			else:
				self.prompt = prompt
				self.image_url = url
				self.input = [
				{
					'role': 'user',
					'content':
						[
							{
									'scaler': 'input_text',
							        'text': self.prompt
							},
							{
									'scaler': 'input_image',
									'image_url': self.image_url
							}
						]
				} ]
				
				self.response = self.client.responses.create( model=self.model, input=self.input )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def summarize_document( self, prompt: str, path: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that summarizes a document given a
			path prompt, and a path

			Parameters
			----------
			prompt: str
			path: str
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif path is None:
				raise Exception( 'Argument "path" cannot be None' )
			else:
				self.file_path = path
				self.file = self.client.files.create( file=open( path, 'rb' ), purpose='user_data' )
				self.messages = [
				{
					'role': 'user',
					'content': [
					{
						'scaler': 'file',
						'file':
						{
							'file_id': file.id,
						}
					},
					{
						'scaler': 'text',
						'text': 'What is the first dragon in the book?',
					}, ]
				} ]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a prompt,

			Parameters
			----------
			prompt: str
			url: str
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.messages = [
		        {
		            'role': 'user',
		            'content': prompt,
		        } ]
				
				self.response = self.client.chat.completions.create( model=self.model,
					web_search_options={ }, messages=self.messages )
				
				return  self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


	def search_file( self, prompt: str ) -> str | None:
		"""
		
			Purpose:
			_______
			Method that analyzeses an image given a prompt,

			Parameters:
			----------
			prompt: str
			url: str
			
			Returns:
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.tools = [
				{
					'scaler': 'file_search',
					'vector_store_ids': self.vector_store_ids,
					'max_num_results': 20
				} ]
				
				self.response = self.client.responses.create( model=self.model,
					tools=self.tools, input=prompt )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def translate( self, text: str ) -> str | None:
		pass
	
	
	def transcribe( self, text: str ) -> str | None:
		pass
	
	
	def get_data( self ) -> Dict[ str, float ] | None:
		'''
			Purpose:
			--------
			Returns: dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str | None:
		'''

			Purpose:
			--------
			Returns: dict of members

		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'messages', 'image_url', 'respose_format', 'tools',
		         'vector_store_ids', 'size', 'api_key', 'client', 'small_model',
		         'generate_text', 'analyze_image', 'summarize_document', 'generate_image',
		         'translate', 'transcribe',
		         'search_web', 'search_files', 'get_data', 'dump' ]
	
	
class Assistant( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with OpenAI's Assistants API

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
		
		Methods
		------------
		get_model_options( self ) -> str
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize_document( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.system_instructions = AI( ).bubba_instructions
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'gpt-4o-mini'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio', 'auto' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'auto'
		self.reasoning_effort = 'auto'
		self.input_text = None
		self.name = 'Boo'
		self.description = 'Generic Assistant'
		self.id = None
		self.metadata = { }
		self.tools = [ ]
		self.vector_stores = \
		{
			'Apporopriations': 'vs_8fEoYp1zVvk5D8atfWLbEupN',
			'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
			'Code': 'vs_67e83bdf8abc81918bda0d6b39a19372',
			'Hawaii': 'vs_67a777291d548191b9fa42956a7f6cb9'
		}
		self.vector_store_ids = [ 'vs_67a777291d548191b9fa42956a7f6cb9' ]
		self.assistants = \
		{
			'Bro': 'asst_2Yu2yfINGD5en4e0aUXAKxyu',
            'Bubba': 'asst_2IpP4nE85lXLKbY6Zewwqtqe',
			'FNG': 'asst_FQXRnDVgvnBxslZQit8hIbXY'
        }
	
	
	def generate_text( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Generates a chat completion given a prompt
			
			
			Parameters
			----------
			prompt: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.input_text = prompt
				self.response = self.client.responses.create( model=self.model,
					input=self.input_text )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
			
			
	def generate_image( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Generates a chat completion given a prompt
			
			
			Parameters
			----------
			prompt: str
			
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.prompt = prompt
				self.response = self.client.images.generate( model='dall-e-3',
					prompt=self.prompt, size='1024x1024', quality='standard', n=1 )

			return self.response.data[0].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
		
		
	def analyze_image( self, prompt: str, url: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a prompt,

			Parameters
			----------
			prompt: str
			url: str
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif url is None:
				raise Exception( 'Argument "url" cannot be None' )
			else:
				self.prompt = prompt
				self.image_url = url
				self.input = [
				{
					'role': 'user',
					'content': [
					{ 'scaler': 'input_text',
					  'text': self.prompt
					},
					{
						'scaler': 'input_image',
						'image_url': self.image_url
					}]
				}]
				
				self.response = self.client.responses.create( model=self.model, input=self.input )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def summarize_document( self, prompt: str, path: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that summarizes a document given a path prompt, and a path
			
			Parameters
			----------
			prompt: str
			path: str
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif path is None:
				raise Exception( 'Argument "path" cannot be None' )
			else:
				self.file_path = path
				self.input_text = prompt
				self.file = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose='user_data' )
				
				self.messages = [
				{
					'role': 'user',
					'content': [
					{
						'scaler': 'file',
						'file':
						{
							'file_id': self.file.id,
						}
					},
					{
						'scaler': 'text',
						'text': self.input_text,
					},]
				}]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a prompt,

			Parameters
			----------
			prompt: str
			url: str
			
			Returns
			-------
			str | None
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.input_text = prompt
				self.messages = [
		        {
		            'role': 'user',
		            'content': self.input_text,
		        } ]
				
				self.response = self.client.chat.completions.create( model=self.model,
					web_search_options={ }, messages=self.messages )
				return  self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'search_web( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


	def search_files( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a prompt,
			
			Parameters
			----------
			prompt: str
			url: str
			
			Returns
			-------
			str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.tools = [
				{
					'scaler': 'file_search',
					'vector_store_ids': self.vector_store_ids,
					'max_num_results': 20
				}, ]
				
				self.response = self.client.responses.create( model=self.model,
					tools=self.tools, input=prompt )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'search_files( self, prompt: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def translate( self, text: str ) -> str | None:
		pass
	
	
	def transcribe( self, text: str ) -> str | None:
		pass
	

	def get_list( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Method that returns a list of available assistants
			
		'''
		try:
			self.assistants = self.client.beta.assistants.list( order='desc', limit='20' )
			return self.assistants.data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'get_list( ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )


	def get_format_options( ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Method that returns a list of formatting options
		
		'''
		return [ 'auto', 'text', 'json' ]
	
	
	def get_model_options( ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Method that returns a list of available models

		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31'  ]
	
	
	def get_effort_options( ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Method that returns a list of available models

		'''
		return [ 'auto', 'low', 'high' ]
	
	
	def get_data( self ) -> Dict[ str, float ] | None:
		'''

			Returns: dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str | None:
		'''

			Returns:
			_______
			Dict of strings representing members

		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens', 'system_instructions',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'messages', 'image_url', 'respose_format', 'tools',
		         'vector_store_ids', 'name', 'id', 'description', 'generate_text',
		         'get_format_options', 'get_model_options', 'reasoning_effort'
		         'get_effort_options', 'input_text', 'metadata',
		         'get_files', 'get_data',
		         'dump', 'translate', 'transcribe' ]

	
class Bubba( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with a Budget Execution Assistant
		
		
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
		
		
		Methods
		------------
		get_model_options( self ) -> str
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize_document( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		dump( self ) -> str
		get_data( self ) -> { }

	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.system_instructions = AI( ).bubba_instructions
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'ft:gpt-4.1-mini-2025-04-14:leeroy-jenkins:bubba-gpt-4-1-mini-2025-05-05:BcekjucJ'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'auto'
		self.reasoning_effort = 'auto'
		self.input_text = None
		self.name = 'Bubba'
		self.description = 'A Budget Execution Assistant'
		self.id = 'asst_J6SAABzDixkTYi2k39OGgjPv'
		self.metadata = { }
		self.tools = [ ]
		self.vector_stores = \
		{
			'Apporopriations': 'vs_8fEoYp1zVvk5D8atfWLbEupN',
			'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK'
		}


	def generate_text( self, prompt: str ) -> str | None:
		"""
		
			Purpose:
			_______
			Generates a chat completion given a prompt
			
			
			Parameters:
			----------
				prompt: str
			
			
			Returns:
			-------
			str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.input_text = prompt
				self.response = self.client.responses.create( model=self.model,
					input=self.input_text )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Bubba'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
			
			
	def generate_image( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Generates a chat completion given a prompt
			
			
			Parameters
			----------
			prompt: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.prompt = prompt
				self.response = self.client.images.generate( model='dall-e-3',
					prompt=self.prompt, size='1024x1024', quality='standard', n=1 )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
		
		
	def analyze_image( self, prompt: str, url: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a prompt,
			
			
			
			Parameters
			----------
			prompt: str
			url: str
			
			Returns
			-------
			str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif url is None:
				raise Exception( 'Argument "url" cannot be None' )
			else:
				self.prompt = prompt
				self.image_url = url
				self.input = [
				{
					'role': 'user',
					'content': [
					{
							'scaler': 'input_text',
							'text': self.prompt
					},
					{
						'scaler': 'input_image',
						'image_url': self.image_url
					} ]
				} ]
				
				self.response = self.client.responses.create( model=self.model, input=self.input )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def summarize_document( self, prompt: str, path: str ) -> str | None:
		"""
		
			Purpose:
			_______
				Method that summarizes a document given a
				path prompt, and a path

			
			Parameters:
			----------
				prompt: str
				path: str
			
			Returns:
			-------
				str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif path is None:
				raise Exception( 'Argument "path" cannot be None' )
			else:
				self.prompt = prompt
				self.file_path = path
				self.file = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose='user_data' )
				
				self.messages = [
				{
					'role': 'user',
					'content': [
					{
						'scaler': 'file',
						'file':
						{
							'file_id': self.file.id,
						}
					},
					{
						'scaler': 'text',
						'text': self.prompt,
					}, ]
				} ]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str | None:
		"""
		
			Purpose:
			_______
				Method that analyzeses an image given a prompt,
			
			
			
			Parameters:
			----------
				prompt: str
				url: str
			
			Returns:
			-------
				str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.prompt = prompt
				self.messages = [
		        {
		            'role': 'user',
		            'content': self.prompt,
		        } ]
				
				self.response = self.client.chat.completions.create( model=self.model,
					web_search_options={ }, messages=self.messages )
				return  self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


	def search_files( self, prompt: str ) -> str | None:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a prompt,
			
			
			
			Parameters
			----------
			prompt: str
			url: str
			
			Returns
			-------
			str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.prompt = prompt
				self.tools = [
				{
					'scaler': 'file_search',
					'vector_store_ids': self.vector_store_ids,
					'max_num_results': 20
				} ]
				
				self.response = self.client.responses.create( model=self.model,
					tools=self.tools, input=self.prompt )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def translate( self, text: str ) -> str | None:
		pass
	
	
	def transcribe( self, text: str ) -> str | None:
		pass
	
	
	def get_files( self ) -> List[ str ] | None:
		'''

			Method that returns a list of files in a vector stores

		'''
		try:
			_aid = self.vector_stores[ 'Appropriations' ]
			_files = self.client.vector_stores.files.list( vector_store_id=_aid )
			_list = [ f for f in _files  ]
			_doc_id = self.vector_stores[ 'Guidance' ]
			_doc_files = self.client.vector_stores.files.list( vector_store_id=_doc_id )
			_doc_list = [ d for d in _doc_files  ]
			return  _list.extend( _doc_list )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Bubba'
			exception.method = 'get_files'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def get_format_options( ) -> List[ str ] | None:
		'''
		
			Method that returns a list of formatting options
		
		'''
		return [ 'auto', 'text', 'json' ]
	
	
	def get_model_options( ) -> List[ str ] | None:
		'''

			Method that returns a list of available models

		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31',
		         'ft:gpt-4.1-mini-2025-04-14:leeroy-jenkins:bubba-gpt-4-1-mini-2025-05-05:BcekjucJ',
		         'ft:gpt-4.1-nano-2025-04-14:leeroy-jenkins:bubba-gpt-4-1-nano-2025-29-05:BcfJJTtv',
		         'ft:gpt-4.1-2025-04-14:leeroy-jenkins:budget-execution-gpt-4-1-2025-20-05:BZO7tKJy',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-fine-tuned-2025-05-06:BUF6o5Xa',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-fine-tuned-2025-05-05:BU7RK1Dq',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-budget-training:BGVjoSXv',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:budget-base-training:BGVk5Ii1',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-base-training:BGVAJg57' ]
	
	
	def get_effort_options( ) -> List[ str ] | None:
		'''

			Method that returns a list of available models

		'''
		return [ 'auto', 'low', 'high' ]
	
	
	def get_data( self ) -> Dict[ str, float ] | None:
		'''

			Returns: dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str | None:
		'''
			Returns: dict of members
		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
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
		return  [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens', 'system_instructions',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'messages', 'image_url', 'respose_format', 'tools',
		         'vector_store_ids', 'name', 'id', 'description', 'generate_text',
		         'get_format_options', 'get_model_options', 'reasoning_effort'
		         'get_effort_options', 'input_text', 'metadata', 'get_data', 'dump',
		          'translate', 'transcribe' ]


class Bro( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with a Data Science & Programming assistant
		
		
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
		
		
		Methods
		------------
		get_model_options( self ) -> str
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize_document( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		dump( self ) -> str
		get_data( self ) -> { }
		
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.system_instructions = AI( ).bro_instructions
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'ft:gpt-4.1-2025-04-14:leeroy-jenkins:bro-gpt-4-1-df-analysis-2025-21-05:BZetxEQa'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'auto'
		self.reasoning_effort = None
		self.input_text = None
		self.name = 'Bro'
		self.description = 'A Computer Programming and Data Science Assistant'
		self.id = 'asst_2Yu2yfINGD5en4e0aUXAKxyu'
		self.vector_store_ids = [ 'vs_67e83bdf8abc81918bda0d6b39a19372', ]
		self.metadata = { }
		self.tools = [ ]
		self.vector_stores = \
		{
			'Code': 'vs_67e83bdf8abc81918bda0d6b39a19372',
		}
	
	
	def generate_text( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Generates a chat completion given a prompt
			
			
			Parameters
			----------
			prompt: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.input_text = prompt
				self.response = self.client.responses.create( model=self.model,
					input=self.input_text )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def generate_image( self, prompt: str ) -> str:
		"""

			Purpose
			_______
			Generates a chat completion given a prompt


			Parameters
			----------
			prompt: str


			Returns
			-------
			str

		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.prompt = prompt
				self.response = self.client.images.generate( model='dall-e-3',
					prompt=self.prompt, size='1024x1024', quality='standard', n=1 )
			
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze_image( self, prompt: str, url: str ) -> str:
		"""

			Purpose
			_______
			Method that analyzeses an image given a prompt,



			Parameters
			----------
			prompt: str
			url: str

			Returns
			-------
			str

		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif url is None:
				raise Exception( 'Argument "url" cannot be None' )
			else:
				self.prompt = prompt
				self.image_url = url
				self.input = [
				{
					'role': 'user',
					'content':[
					{
						'scaler': 'input_text',
					    'text': self.prompt
					},
					{
						'scaler': 'input_image',
						'image_url': self.image_url
					}]
				}]
				
				self.response = self.client.responses.create( model=self.model, input=self.input )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def summmarize_document( self, prompt: str, path: str ) -> str:
		"""

			Purpose
			_______
			Method that summarizes a document given a
			path prompt, and a path




			Parameters
			----------
			prompt: str
			path: str

			Returns
			-------
			str

		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif path is None:
				raise Exception( 'Argument "path" cannot be None' )
			else:
				self.input_text = prompt
				self.file_path = path
				self.file = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose='user_data' )
				
				self.messages = [
				{
					'role': 'user',
					'content': [
					{
						'scaler': 'file',
						'file':
							{
								'file_id': self.file.id,
							}
					},
					{
						'scaler': 'text',
						'text': self.input_text,
					}, ]
				} ]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str:
		"""

			Purpose
			_______
			Method that analyzeses an image given a prompt,



			Parameters
			----------
			prompt: str
			url: str

			Returns
			-------
			str

		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.messages = [
					{
						'role': 'user',
						'content': prompt,
					} ]
				
				self.response = self.client.chat.completions.create( model=self.model,
					web_search_options={ }, messages=self.messages )
				
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_files( self, prompt: str ) -> str:
		"""

			Purpose
			_______
			Method that analyzeses an image given a prompt,



			Parameters
			----------
			prompt: str
			url: str

			Returns
			-------
			str

		"""
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.tools = [
				{
					'scaler': 'file_search',
					'vector_store_ids': self.vector_store_ids,
					'max_num_results': 20
				} ]
				
				self.response = self.client.responses.create( model=self.model,
					tools=self.tools, input=prompt )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def translate( self, text: str ) -> str:
		pass
	
	
	def transcribe( self, text: str ) -> str:
		pass
	
	
	def get_format_options( ) -> List[ str ]:
		'''
		
			Method that returns a list of formatting options
		
		'''
		return [ 'auto', 'text', 'json' ]
	
	
	def get_model_options( ) -> List[ str ]:
		'''

			Method that returns a list of available models

		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31',
		         'ft:gpt-4.1-2025-04-14:leeroy-jenkins:bro-gpt-4-1-df-analysis-2025-21-05:BZetxEQa',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-fine-tuned-05052025:BTryvkMx',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-analytics:BTX4TYqY',
		         'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-fine-tuned-05052025:BTryvkMx' ]
	
	
	def get_effort_options( ) -> List[ str ]:
		'''

			Method that returns a list of available models

		'''
		return [ 'auto', 'low', 'high' ]
	
	
	def get_data( self ) -> dict:
		'''

			Returns: dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str:
		'''
			Returns: dict of members
		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	

	def __dir__(self) -> List[ str ]:
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
		return  [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens', 'system_instructions',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'messages', 'image_url', 'respose_format', 'tools',
		         'vector_store_ids', 'name', 'id', 'description', 'generate_text',
		         'get_format_options', 'get_model_options', 'reasoning_effort'
		         'get_effort_options', 'input_text', 'metadata',
		          'get_data', 'dump' ]


class Embedding( AI ):
	"""

		Purpose
		___________
		Class used for creating vectors using
		OpenAI' embedding-3-small embedding small_model

		Parameters
		------------
		None

		Attributes
		-----------
		self.api_key
		self.client
		self.small_model
		self.embedding
		self.response

		Methods
		------------
		create_small_embedding( self, text: str ) -> get_list[ float ]


	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.small_model = 'text-embedding-3-small'
		self.large_model = 'text-embedding-3-large'
		self.ada_model = 'text-embedding-ada-002'
		self.encoding_format = 'float'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.embedding = None
		self.response = None
	
	
	def create_small_embedding( self, text: str ) -> List[ float ]:
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
			if text is None:
				raise Exception( 'Argument "text" is required.' )
			else:
				self.input = text
				self.response = self.client.embeddings.create( self.input, self.small_model )
				self.embedding = self.response.data[ 0 ].embedding
				return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Embedding'
			exception.method = 'create_small_embedding( self, text: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_large_embedding( self, text: str ) -> List[ float ] | None:
		"""

			Purpose:
			_______
			Creates an Large embedding given a text

			Parameters:
			----------
			text: str

			Returns:
			-------
			List[ float ] | None

		"""
		try:
			if text is None:
				raise Exception( 'Argument "text" is required.' )
			else:
				self.input = text
				self.response = self.client.embeddings.create( self.input, self.large_model )
				self.embedding = self.response.data[ 0 ].embedding
				return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Embedding'
			exception.method = 'create_large_embedding( self, text: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_ada_embedding( self, text: str ) -> List[ float ]:
		"""

			Purpose
			_______
			Creates an ADA embedding given a text

			Parameters
			----------
			text: str

			Returns
			-------
			List[ float ]

		"""
		try:
			if text is None:
				raise Exception( 'Argument "text" is required.' )
			else:
				self.input = text
				self.response = self.client.embeddings.create( self.input, self.ada_model )
				self.embedding = self.response.data[ 0 ].embedding
				return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Embedding'
			exception.method = 'create_ada_embedding( self, text: str ) -> List[ float ]'
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
			if text is None:
				raise Exception( 'Argument "text" is required.' )
			elif coding is None:
				raise Exception( 'Argument "coding" is required.' )
			else:
				_encoding = tiktoken.get_encoding( coding )
				_tokens = len( _encoding.encode( text ) )
				return _tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> int'
			error = ErrorDialog( exception )
			error.show( )


	def get_data( self ) -> Dict:
		'''

			Purpose:
			-------

			Returns:
			--------
			dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str | None:
		'''

			Purpose:
			--------
			Returns: dict of members

		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops', 'create_ada_embedding',
		         'api_key', 'client', 'small_model', 'count_tokens', 'create_large_embedding',
		         'path', 'create_small_embedding', 'get_model_options' ]


class TTS( AI ):
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
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		'''

			Purpose:
			--------
			Constructor to  create_small_embedding TTS objects

		'''
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'gpt-4o-mini-tts'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.audio_path = None
		self.response = None
		self.prompt = None
		self.voice = 'alloy'
	
	
	def get_model_options( self ) -> str:
		'''

			Purpose:
			--------
			Methods that returns a list of small_model names
		
		'''
		return [ 'tts-1', 'tts-1-hd',
		         'gpt-4o-mini-tts',
                 'gpt-4o-audio-preview-2024-12-17',
                 'gpt-4o-audio-preview-2024-10-01',
                 'gpt-4o-mini-audio-preview-2024-12-17' ]
	
	
	def get_voice_options( self ):
		'''

			Purpose:
			--------
			Method that returns a list of voice names
		
		'''
		return [ 'alloy', 'ash', 'ballad', 'coral',
		         'echo', 'fable', 'onyx', 'nova',
		         'sage', 'shiver' ]
	
	
	def get_format_options( self ):
		'''

			Purpose:
			--------
			Method that returns a list of image formats
		
		'''
		return [ 'mp3', 'wav', 'aac', 'flac', 'opus', 'pcm']
	
	
	def save_audio( self, prompt: str, filepath: str ) -> str:
		"""
		
			Purpose
			_______
			Generates audio given a text prompt and path to audio file
			
			
			Parameters
			----------
			prompt: str
			path: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if filepath is None:
				raise Exception( 'Argument "path" is required.' )
			elif prompt is None:
				raise Exception( 'Argument "prompt" is required.' )
			else:
				self.audio_path = Path( filepath ).parent
				self.prompt = prompt
				self.response = self.client.audio.speech.with_streaming_response( model=self.model,
					voice=self.voice, input=self.prompt )
				_retval = self.response.stream_to_file( self.audio_path )
				return _retval
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'TTS'
			exception.method = 'save_audio( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def get_data( self ) -> Dict[ str, str ]:
		'''

			Purpose:
			--------
			Returns: dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str:
		'''

			Purpose:
			--------
			Returns: dict of members

		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'prompt', 'response', 'completion', 'audio_path',
		         'path', 'messages', 'respose_format', 'tools',
		         'size', 'api_key', 'client', 'small_model', 'voice',
		         'generate_text', 'get_model_options' ]


class Transcription( AI ):
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
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'whisper-1'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
	
	
	def get_model_options( self ) -> str:
		'''

			Purpose:
			--------
			Methods that returns a list of small_model names

		'''
		return [ 'whisper-1',
		         'gpt-4o-mini-transcribe',
		         'gpt-4o-transcribe' ]
	
	
	def create( self, text: str, path: str ) -> str:
		"""
		
			Purpose
			_______
			Generates a transcription given a text text to an audio file
			
			
			Parameters
			----------
			text: str
			path: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if text is None:
				raise Exception( 'Argument "text" is required.' )
			elif path is None:
				raise Exception( 'Argument "path" is required.' )
			else:
				self.audio_file = open( filepath=path, encoding='utf-8', errors='ignore' )
				self.input_text = text
				self.response = self.client.audio.speech.create( model=self.model,
					voice='alloy', input=self.input_text )
				self.response.stream_to_file( self.audio_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Transcription'
			exception.method = 'create( self, text: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def get_data( self ) -> dict:
		'''

			Purpose:
			--------
			Returns: dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str:
		'''
			Returns: dict of members
		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'prompt', 'response', 'audio_file',
		         'messages', 'respose_format',
		         'api_key', 'client',
		         'input_text', 'transcript' ]


class Translation( AI ):
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
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'whisper-1'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.audio_file = None
		self.response = None
		self.voice = None
	
	
	def get_model_options( self ) -> str:
		'''

			Purpose:
			--------
			Methods that returns a list of small_model names

		'''
		return [ 'whisper-1', 'text-davinci-003',
		         'gpt-4-0613', 'gpt-4-0314',
		         'gpt-4-turbo-2024-04-09' ]
	
	
	def get_voice_options( self ):
		'''

			Purpose:
			--------
			Method that returns a list of voice names
		
		'''
		return [ 'alloy', 'ash', 'ballad', 'coral',
		         'echo', 'fable', 'onyx', 'nova',
		         'sage', 'shiver' ]
	
	
	def create( self, text: str, path: str ) -> str:
		"""

			Purpose
			_______
			Generates a transcription given a text text to an audio file


			Parameters
			----------
			text: str
			path: str


			Returns
			-------
			str

		"""
		try:
			if text is None:
				raise Exception( 'Argument "text" is required.' )
			elif path is None:
				raise Exception( 'Argument "path" is required.' )
			else:
				self.audio_file = open( filepath=path, mode='rb', encoding='utf-8', errors='ignore' )
				self.response = self.client.audio.translations.create( model='whisper-1',
					file=self.audio_file )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Translation'
			exception.method = 'create_small_embedding( self, text: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def get_data( self ) -> dict:
		'''

			Purpose:
			--------
			Returns: dict[ str ] of members

		'''
		return { 'num': self.number,
		         'temperature': self.temperature,
		         'top_percent': self.top_percent,
		         'frequency_penalty': self.frequency_penalty,
		         'presence_penalty': self.presence_penalty,
		         'store': self.store,
		         'stream': self.stream,
		         'size': self.size }
	
	
	def dump( self ) -> str:
		'''
		
			Returns: dict of members
			
		'''
		new = '\r\n'
		return 'num' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'prompt', 'response', 'completion', 'audio_path',
		         'path', 'messages', 'respose_format', 'tools',
		         'size', 'api_key', 'client',
		         'small_model', 'create_small_embedding', 'get_model_options' ]


class LargeImage( AI ):
	"""

		Purpose
		___________
		Class used for generating images OpenAI's Images API and dall-e-3


		Parameters
		------------
		num: int
		temperature: float
		top_percent: float
		frequency_penalty: float
		presence_penalty: float
		maximum_completion_tokens: int
		store: bool
		stream: bool

		Methods
		------------
		generate( self, path: str ) -> str:
		analyze( self, path: str, text: str ) -> str
		get_detail_options( self ) -> list[ str ]
		get_format_options( self ) -> list[ str ]:
		get_size_options( self ) -> list[ str ]

	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=False, stream: bool=False ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.quality = 'hd'
		self.model = 'dall-e-3'
		self.size = '1024X1024'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.input_text = None
		self.file_path = None
		self.image_url = None
	
	
	def generate( self, input: str ) -> str:
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
			if input is None:
				raise Exception( 'The "text" argument is required.' )
			else:
				self.input_text = input
				self.response = self.client.images.generate(
					model=self.model,
					prompt=self.input_text,
					size=self.size,
					quality=self.quality,
					n=self.number
				)
				
				return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Image'
			exception.method = 'generate( self, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, input: str, path: str ) -> str:
		'''

			Purpose:
			--------
			Method providing image analysis functionality given a prompt and path

		'''
		try:
			if input is None:
				raise Exception( 'The argument "path" cannot be None' )
			elif path is None:
				raise Exception( 'The argument "path" cannot be None' )
			else:
				self.input_text = input
				self.file_path = path
				self.input = \
					[ {
						'role': 'user',
						'content':
							[
								{ 'scaler': 'input_text',
								  'text': self.input_text
								  },
								{
									'scaler': 'input_image',
									'image_url': self.file_path
								},
							],
					}
					]
				
				self.response = self.client.responses.create(
					model='gpt-4o-mini',
					input=self.input,
				)
				
				return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Image'
			exception.method = 'analyze( self, text: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def get_model_options( self ) -> List[ str ]:
		'''

			Purpose:
			--------
			Methods that returns a list of small_model names

		'''
		return [ 'dall-e-3', 'gpt-4-0613',
		         'gpt-4-0314', 'gpt-4o-mini',
		         'gpt-4o-mini-2024-07-18' ]
	
	
	def get_format_options( self ) -> list[ str ]:
		'''

			Purpose:
			--------
			Method that returns a  list of format options

		'''
		return [ '.png', '.mpeg', '.jpeg',
		         '.webp', '.gif' ]
	
	
	def get_detail_options( self ) -> list[ str ]:
		'''

			Purpose:
			--------
			Method that returns a  list of reasoning effort options

		'''
		return [ 'auto', 'low', 'high' ]
	
	
	def get_size_options( self ) -> list[ str ]:
		'''

			Purpose:
			--------
			Method that returns a  list of sizes

		'''
		return [ '1024x1024',
		         '1024x1792',
		         '1792x1024' ]
	
	
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'input_text', 'image_url', 'path',
		         'api_key', 'client', 'small_model', 'generate',
		         'get_detail_options', 'get_format_options', 'get_size_options' ]


class Image( AI ):
	"""

		Purpose
		___________
		Class used for generating images OpenAI's Images API and dall-e-2


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
		self.api_key, self.client, self.small_model,  self.embedding,
		self.response, self.num, self.temperature, self.top_percent,
		self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
		self.store, self.stream, self.modalities, self.stops, self.content,
		self.prompt, self.response, self.completion, self.file, self.path,
		self.messages, self.image_url, self.response_format,
		self.tools, self.vector_store_ids, self.input_text, self.image_url

		Methods
		------------
		get_model_options( self ) -> str
		generate( self, path: str ) -> str
		analyze( self, path: str, text: str ) -> str
		get_detail_options( self ) -> list[ str ]
		get_format_options( self ) -> list[ str ]
		get_size_options( self ) -> list[ str ]

	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=10000, store: bool=False, stream: bool=False ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.quality = 'standard'
		self.detail = 'auto'
		self.model = 'dall-e-2'
		self.large_model = 'dall-e-3'
		self.small_model = 'dall-e-2'
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.input = [ ]
		self.input_text = None
		self.file_path = None
		self.image_url = None
	
	
	def get_model_options( self ) -> List[ str ]:
		'''
		
			Purpose:
			________
			Methods that returns a list of small_model names

		'''
		return [ 'dall-e-2', 'gpt-4-0613',
		         'gpt-4-0314', 'gpt-4o-mini',
		         'gpt-4o-mini-2024-07-18' ]
	
	
	def get_size_options( self ) -> list[ str ]:
		'''
		
			Purpose:
			________
			Method that returns a  list of small_model options

		'''
		return [ '256x256', '512x512', '1024x1024' ]
	
	
	def get_format_options( self ) -> list[ str ]:
		'''
		
			Purpose:
			________
			Method that returns a  list of format options

		'''
		return [ '.png', '.mpeg', '.jpeg', '.webp', '.gif' ]
	
	
	def get_detail_options( self ) -> list[ str ]:
		'''
		
			Purpose:
			________
			Method that returns a  list of reasoning effort options

		'''
		return [ 'auto', 'low', 'high' ]
	
	
	def generate( self, text: str ) -> str:
		"""

			Purpose
			_______
			Generates an image given a text path


			Parameters
			----------
			text: str


			Returns
			-------
			Image object

		"""
		try:
			if text is None:
				raise Exception( 'The "text" argument is required.' )
			else:
				self.input_text = text
				self.response = self.client.images.generate(
					model=self.model,
					prompt=self.input_text,
					size=self.size,
					quality=self.quality,
					n=self.number )
				
				return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Image'
			exception.method = 'generate( self, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, input: str, path: str ) -> str:
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
			if input is None:
				raise Exception( 'The argument "text" cannot be None' )
			elif path is None:
				raise Exception( 'The argument "path" cannot be None' )
			else:
				self.input_text = input
				self.file_path = path
				self.input = [
				{
						'role': 'user',
						'content':
						[
							{
								'scaler': 'input_text',
								'text': self.input_text
							},
							{
								'scaler': 'input_image',
								'image_url': self.file_path
							},
						],
				}]
				
				self.response = self.client.responses.create( model='gpt-4o-mini',
					input=self.input )
				
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Image'
			exception.method = 'analyze( self, path: str, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def edit( self, input: str, path: str, size: str='1024X1024' ) -> str:
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
			if input is None:
				raise Exception( 'The argument "text" cannot be None' )
			elif path is None:
				raise Exception( 'The argument "path" cannot be None' )
			else:
				self.input_text = input
				self.file_path = path
				self.response = self.client.images.edit( model=self.model,
					image=open( self.file_path, 'rb' ), prompt=self.input_text, n=self.number,
					size=self.size )
				
				return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Image'
			exception.method = 'edit( self, text: str, path: str, size: str=1024X1024 ) -> str'
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
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'api_key', 'client', 'small_model', 'path', 'analyze',
		         'input_text', 'image_url', 'edit',
		         'generate', 'quality', 'detail', 'small_model', 'get_model_options',
		         'get_detail_options', 'get_format_options', 'get_size_options' ]
	
	

