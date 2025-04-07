'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                boo.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="boo.py" company="Terry D. Eppler">

     Bobo is a values analysis tool for EPA Analysts.
     Copyright ©  2024  Terry Eppler

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
    boo.py
  </summary>
  ******************************************************************************************
  '''
import os
import datetime as dt
from openai import OpenAI
import requests
from pygments.lexers.csound import newline
from static import GptRequests, GptRoles, GptLanguages
from booger import ErrorDialog, Error


class Header( ):
	'''
		Class used to encapsulate GPT headers
	'''
	def __init__( self ):
		self.content_type = 'application/json'
		self.api_key = os.environ.get( 'OPENAI_API_KEY' )
		self.authoriztion = 'Bearer ' + os.environ.get( 'OPENAI_API_KEY' )
		self.data = { 'content-type': self.content_type,
		              'Authorization': self.authoriztion }
	
	
	def __dir__( self ):
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'content_type', 'api_key', 'authorization', 'values' ]


class EndPoint( ):
	'''
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
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'base_url', 'text_generation', 'image_generation', 'chat_completions',
		         'speech_generation', 'translations', 'assistants', 'transcriptions',
		         'finetuning', 'embeddings', 'uploads', 'files', 'vector_stores',
		         'responses', 'get_data', 'dump' ]
	
	
	def get_data( self ) -> dict:
		'''

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
		         'embeddings': self.embeddings,
		         'uploads': self.uploads,
		         'files': self.files,
		         'vector_stores': self.vector_stores }
	
	
	def dump( self ) -> str:
		'''
			Returns: string of "member = value", pairs
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
			'embeddings' + f' = {self.files}' + new + \
			'uploads' + f' = {self.uploads}' + new + \
			'files' + f' = {self.files}' + new + \
			'vector_stores' + f' = {self.vector_stores}' + new


class Models( ):
	'''
		Class containing lists of OpenAI models by generation
	'''
	def __init__( self ):
		self.text_generation = [ 'documents-davinci-003', 'documents-curie-001',
		                         'gpt-4-0613', 'gpt-4-0314',
		                         'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
		                         'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
		                         'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
		                         'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.image_generation = [ 'dall-e-2', 'dall-e-3',
		                          'gpt-4-0613', 'gpt-4-0314',
		                          'gpt-4o-mini-2024-07-18' ]
		self.chat_completion = [ 'gpt-4-0613', 'gpt-4-0314',
		                         'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
		                         'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
		                         'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
		                         'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.speech_generation = [ 'tts-1', 'tts-1-hd',
		                           'gpt-4o-audio-preview-2024-12-17',
		                           'gpt-4o-audio-preview-2024-10-01',
		                           'gpt-4o-mini-audio-preview-2024-12-17' ]
		self.transcription = [ 'whisper-1', 'gpt-4o-mini-transcribe', ' gpt-4o-transcribe' ]
		self.translation = [ 'whisper-1', 'documents-davinci-003',
		                     'gpt-4-0613', 'gpt-4-0314',
		                     'gpt-4-turbo-2024-04-09' ]
		self.finetuning = [ 'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18',
		                    'gpt-4-0613', 'gpt-3.5-turbo-0125',
		                    'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613' ]
		self.embeddings = [ 'embedding-3-small', 'embedding-3-large',
		                    'embedding-ada-002' ]
		self.uploads = [ 'gpt-4-0613', 'gpt-4-0314', 'gpt-4-turbo-2024-04-09',
		                 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20',
		                 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18',
		                 'o1-2024-12-17', 'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.files = [ 'gpt-4-0613', 'gpt-4-0314', 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20',
		               'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18',
		               'o1-2024-12-17', 'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
		self.vector_stores = [ 'gpt-4-0613', 'gpt-4-0314', 'gpt-4-turbo-2024-04-09',
		                       'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
		                       'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
		                       'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'base_url', 'text_generation', 'image_generation', 'chat_completion',
		         'speech_generation', 'translations', 'assistants', 'transcriptions',
		         'finetuning', 'embeddings', 'uploads', 'files', 'vector_stores' ]
	
	
	def get_data( self ) -> dict:
		'''
			Method that returns a list of dictionaries
		'''
		_data = { 'text_generation': self.text_generation,
		          'image_generation': self.image_generation,
		          'chat_completion': self.chat_completion,
		          'speech_generation': self.speech_generation,
		          'translations': self.translation,
		          'finetuning': self.finetuning,
		          'embeddings': self.embeddings,
		          'uploads': self.uploads,
		          'files': self.files,
		          'vector_stores': self.vector_stores }


class AI( ):
	'''
	AI is the base class for all OpenAI functionalityl
	'''
	def __init__( self ):
		self.header = Header( )
		self.endpoint = EndPoint( )
		self.api_key = self.header.api_key
		self.client = OpenAI( api_key=self.api_key )
		self.system_instructions = '''You are the most knowledgeable Budget Analyst in the
		federal government who provides detailed responses based on your vast knowledge
		of budget legislation, and federal appropriations.  Your responses to questions
		about federal finance are complete, transparent, and very detailed using
		an academic format.   Your vast knowledge of and experience in Data Science makes you the
		best Data Analyst in the world. You are also an expert programmer who is
		proficient in C#, Python, SQL, C++, JavaScript, and VBA.
		You are famous for the accuracy of your responses so you verify all your answers.
		This makes the quality of your code very high and it always works.
		Your responses are always accurate and complete!  Your name is Bubba.
        '''


class GptOptions( ):
	'''

		The base class used by all parameter classes.

	'''
	def __init__( self, num: int=1, temp: float=0.80, top: float=0.90,
	              freq: float=0.00, max: int=2048, pres: float=0.00,
	              store: bool=False, stream: bool=True, size: str='1024X1024' ):
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.max_completion_tokens = max
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		self.size = size
		self.modalities = [ 'text', 'audio' ]
		self.response_format = 'text'
	
	
	def __dir__( self ) -> list[ str ]:
		'''

			Methods that returns a list of member names
			Returns: list[ str ]

		'''
		return [ 'number', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens', 'store', 'stream', 'size',
		         'get_voices', 'get_sizes', 'response_format',
		         'get_file_formats', 'get_response_formats',
		         'get_output_formats', 'get_input_formats',
		         'get_data' ]
	
	
	def get_voices( self ) -> list[ str ]:
		'''

			Returns: list[ str ] of voices used by the audio api

		'''
		return [ 'alloy', 'cash', 'coral', 'echo',
		         'onyx', 'fable', 'nova', 'sage' ]
	
	
	def get_sizes( self ) -> list[ str ]:
		'''

			Returns: list[ str ] of size used by the audio api

		'''
		return [ '256X256', '512X512', '1024X1024',
		         '1024x1792', '1792x1024' ]
	
	
	def get_response_formats( self ) -> list[ str ]:
		'''

			Returns: list[ str ] of response formats used by the GPT

		'''
		return [ 'documents', 'audio', 'url' ]
	
	
	def get_output_formats( self ) -> list[ str ]:
		'''

			Returns: list[ str ] of audio formats output by the audio api

		'''
		return [ 'mp3', 'opus', 'aac', 'flac', 'pcm' ]
	
	
	def get_input_formats( self ) -> list[ str ]:
		'''

			Returns: list[ str ] of audio formats uploaded into the audio api

		'''
		return [ 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm' ]
	
	
	def get_data( self ) -> dict:
		'''

			Returns: dict[ str ] of members

		'''
		return { 'number': self.number,
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
		return 'number' + f' = {self.number}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_percent' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new


class Payload( ):
	'''

		Class used to capture request parameters.

	'''
	def __init__( self, model: str='gpt-4o', number: int = 1, temp: float=0.80,
	              top_p: float=0.90, freq: float=0.00, max: int=2048,
	              presence: float=0.00, store: bool=False,
	              stream: bool=True, size: str='1024X1024' ):
		self.model = model
		self.number = number
		self.temperature = temp
		self.top_percent = top_p
		self.frequency_penalty = freq
		self.max_completion_tokens = max
		self.presence_penalty = presence
		self.store = store
		self.stream = stream
		self.size = size
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'max_completion_tokens': f'{self.max_completion_tokens}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'size': f'{self.size}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		Methods that returns a list of member names
		Returns: list[ str ]
		'''
		return [ 'number', 'model', 'temperature',
		         'top_percent', 'frequency_penalty',
		         'max_completion_tokens', 'presence_penalty',
		         'store', 'stream', 'endpoint',
		         'size', 'values', 'dump', 'parse' ]
	
	
	def dump( self ) -> str:
		'''

			Returns: a string of key value pairs

		'''
		new = '\r\n'
		return 'n' + f' = {self.number}' + new + \
			'model' + f' = {self.model}' + new + \
			'endpoint' + f' = {self.endpoint}' + new + \
			'temperature' + f' = {self.temperature}' + new + \
			'top_p' + f' = {self.top_percent}' + new + \
			'frequency_penalty' + f' = {self.frequency_penalty}' + new + \
			'presence_penalty' + f' = {self.presence_penalty}' + new + \
			'max_completion_tokens' + f' = {self.max_completion_tokens}' + new + \
			'store' + f' = {self.store}' + new + \
			'stream' + f' = {self.stream}' + new + \
			'size' + f' = {self.size}' + new
	
	
	def parse( self ) -> dict:
		pass


class GptRequest( AI ):
	'''
		Base class for GPT requests.
	'''
	def __init__( self, num: int = 1, temp: float = 0.8, top: float = 0.9, freq: float = 0.0,
	              pres: float = 0.0, max: int = 2048, store: bool = False, stream: bool = True ):
		super( ).__init__( )
		self.header = super( ).header
		self.api_key = super( ).api_key
		self.instructions = super( ).system_instructions
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
		self.response_format = 'text'
		self.content = None
		self.response = None
		self.prompt = None
		self.messages = [ ]
		self.data = { 'number': f'{self.number}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }


class TextRequest( GptRequest ):
	'''

	Class provides the functionality fo the Text Generation API

	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
		self.endpoint = EndPoint( ).text_generation
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
		self.response_format = 'text'
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}'}
	
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'content', 'response', 'prompt', 'size', 'create', 'messages', 'values' ]
	
	
	def create( self, prompt: str ) -> str:
		'''

			Given an input prompt 'prompt', function generates a documents generation
			request from the openai api.

		Args:
			prompt: query provided by the user to the GPT application

		Returns:

		'''
		try:
			if prompt is None:
				alert = f'The prompt argument is not available'
				raise Exception( alert )
			else:
				self.prompt = prompt
			
			self.client.api_key = self.header.api_key
			_system_prompt = 'You are a helpful assistant and Budget Analyst'
			_system = GptMessage( prompt=_system_prompt, role='system', type='documents' )
			_user = GptMessage( prompt=self.prompt, role='user', type='documents' )
			self.messages.append( _system )
			self.messages.append( _user )
			
			completion = self.client.chat.completions.create(
				model=self.model,
				messages=self.messages,
				temperature=self.temperature,
				max_completion_tokens=self.max_completion_tokens,
				top_p=self.top_percent,
				n=self.number,
				frequency_penalty=self.frequency_penalty,
				presence_penalty=self.presence_penalty,
			)
			
			self.content = completion[ 'choices' ][ 0 ][ 'message' ][ 'content' ]
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'TextRequest'
			exception.method = 'create( prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


class CompletionRequest( GptRequest ):
	'''

		Class provides the functionality fo the Completions API

	'''
	def __init__( self, num: int = 1, temp: float = 0.8, top: float = 0.9, freq: float = 0.0,
	              pres: float = 0.0, max: int = 2048, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
		self.endpoint = EndPoint( ).chat_completion
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
		self.response_format = 'text'
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint',
		         'model', 'number', 'messages',
		         'content', 'response', 'instructions', 'prompt',
		         'size', 'create', 'messages', 'values' ]
	
	
	def create( self, prompt: str ) -> str:
		'''
				Function that generates chat completions given a prompt
			Args:
				prompt:

			Returns:

		'''
		
		try:
			if prompt is None:
				alert = 'The prompt argument is not available'
				raise Exception( alert )
			else:
				self.prompt = prompt
			self.client.api_key = self.header.api_key
			_sys = 'You are a helpful assistant and Budget Analyst'
			_system = GptMessage( prompt=_sys, role='system', type='documents' )
			_user = GptMessage( prompt=self.prompt, role='user', type='documents' )
			self.messages.append( _system )
			self.messages.append( _user )
			self.response = self.client.chat.completions.create(
				model=self.model,
				messages=self.messages,
				temperature=self.temperature,
				max_completion_tokens=self.max_completion_tokens,
				top_p=self.top_percent,
				n=self.number,
				frequency_penalty=self.frequency_penalty,
				presence_penalty=self.presence_penalty,
			)
			
			self.content = self.response[ 'choices' ][ 0 ][ 'message' ][ 'content' ]
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'CompletionRequest'
			exception.method = 'create( prompt: str )'
			error = ErrorDialog( exception )
			error.show( )


class ImageRequest( GptRequest ):
	'''
		Class provides the functionality fo the Image Generation API
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'dall-e-3'
		self.endpoint = EndPoint( ).image_generation
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.size = '1024X1024'
		self.detail = 'standard'
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'text'
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'size': f'{self.size}',
		              'detail': f'{self.detail}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'content', 'response', 'prompt', 'size', 'create', 'messages', 'values',
		         'instructions' ]
	
	
	def create( self, prompt: str, num: int = 1,
	            quality='standard', size: str = '1024X1024' ) -> str:
		'''
			Function geerates chat completq

			Args:
				prompt: str, num: int, size: str


			Returns:

		'''
		try:
			if prompt is None:
				alert = 'The prompt argument is not available'
				raise Exception( alert )
			else:
				self.prompt = prompt
				self.number = num
				self.size = size
				openai.api_key = self.header.api_key
				_sys = 'You are a helpful assistant and Budget Analyst'
				_system = GptMessage( prompt=_sys, role='system', type='documents' )
				_user = GptMessage( prompt=self.prompt, role='user', type='documents' )
				self.messages.append( _system )
				self.messages.append( _user )
				self.response = self.client.chat.completions.create(
					model=self.model,
					messages=self.messages,
					temperature=self.temperature,
					max_completion_tokens=self.max_completion_tokens,
					top_p=self.top_percent,
					n=self.number,
					frequency_penalty=self.frequency_penalty,
					presence_penalty=self.presence_penalty,
				)
				
				self.url = self.response[ 'values' ][ 0 ][ 'url' ]
				self.content = requests.get( url ).content
				with open( 'image_name.png', 'wb' ) as file:
					file.write( self.content )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'ImageRequest'
			exception.method = 'create( prompt: str )'
			error = ErrorDialog( exception )
			error.show( )


class SpeechRequest( GptRequest ):
	'''
		Class encapsulating requests for speech generations.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'tts-1-hd'
		self.endpoint = EndPoint( ).speech_generation
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.size = None
		self.detail = None
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'mp3'
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'size': f'{self.size}',
		              'detail': f'{self.detail}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }



class TranslationRequest( GptRequest ):
	'''
		Class encapsulating requests for translation.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'whisper-1'
		self.endpoint = EndPoint( ).translations
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = max
		self.store = store
		self.stream = stream
		self.size = None
		self.detail = None
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'mp3'
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'size': f'{self.size}',
		              'detail': f'{self.detail}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'content', 'response', 'prompt', 'size', 'create', 'messages', 'values' ]
	
	
class TranscriptionRequest( GptRequest ):
	'''
		Class encapsulating requests for transcriptions.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'whisper-1'
		self.endpoint = EndPoint( ).transcriptions
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
		self.response_format = 'text'
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'endpoint', 'model', 'number', 'messages',
		         'content', 'response', 'prompt', 'size', 'create', 'messages', 'values' ]



class EmbeddingRequest( GptRequest ):
	'''
		Class encapsulating requests for embedding.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'text-embedding-3-large'
		self.endpoint = EndPoint( ).embeddings
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
		self.response_format = 'text'
		self.messages = [ GptMessage ]
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'response_format', 'modalities', 'max_completion_tokens', 'frequency_penalty',
		         'presence_penalty', 'temperature', 'top_percent', 'store', 'stream',
		         'stops', 'content', 'response', 'prompt', 'create', 'messages', 'values',
		         'instructions' ]



class VectorStoreRequest( GptRequest ):
	'''
		Class encapsulating requests for vectors.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'gt-4o-mini'
		self.endpoint = EndPoint( ).vector_stores
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
		self.response_format = 'text'
		self.messages = [ GptMessage ]
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'response_format', 'modalities', 'max_completion_tokens', 'frequency_penalty',
		         'presence_penalty', 'temperature', 'top_percent', 'store', 'stream',
		         'stops', 'content', 'response', 'prompt', 'create', 'messages', 'values',
		         'instructions' ]



class GptFileRequest( GptRequest ):
	'''
		Class encapsulating requests for GPT files.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'gt-4o-mini'
		self.endpoint = EndPoint( ).files
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
		self.response_format = 'text'
		self.messages = [ GptMessage ]
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'response_format', 'modalities', 'max_completion_tokens', 'frequency_penalty',
		         'presence_penalty', 'temperature', 'top_percent', 'store', 'stream',
		         'stops', 'content', 'response', 'prompt', 'create', 'messages', 'values' ]



class GptUploadRequest( GptRequest ):
	'''
		Class encapsulating requests for GPT uploads.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'gt-4o-mini'
		self.endpoint = EndPoint( ).uploads
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
		self.response_format = 'text'
		self.messages = [ GptMessage ]
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'response_format', 'modalities', 'max_completion_tokens', 'frequency_penalty',
		         'presence_penalty', 'temperature', 'top_percent', 'store', 'stream',
		         'stops', 'content', 'response', 'prompt', 'create', 'messages', 'values',
		         'instructions' ]



class FineTuningRequest( GptRequest ):
	'''
		Class encapsulating requests for fine-tuning.
	'''
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=True  ):
		super( ).__init__( num, temp, top, freq, pres, max, store, stream )
		self.api_key = super( ).api_key
		self.header = super( ).header
		self.instructions = super( ).system_instructions
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
		self.endpoint = EndPoint( ).finetuning
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
		self.response_format = 'text'
		self.messages = [ GptMessage ]
		self.content = None
		self.response = None
		self.prompt = None
		self.data = { 'number': f'{self.number}',
		              'model': f'{self.model}',
		              'endpoint': f'{self.endpoint}',
		              'temperature': f'{self.temperature}',
		              'top_percent': f'{self.top_percent}',
		              'frequency_penalty': f'{self.frequency_penalty}',
		              'presence_penalty': f'{self.presence_penalty}',
		              'store': f'{self.store}',
		              'stream': f'{self.stream}',
		              'response_format': f'{self.response_format}',
		              'modalities': f'{self.modalities}',
		              'stops': f'{self.stops}',
		              'messages': f'{self.messages}',
		              'authorization': f'{self.header.authoriztion}',
		              'content-type': f'{self.header.content_type}',
		              'instructions': f'{self.instructions}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
				Methods that returns a list of member names
				Returns: list[ str ]
		'''
		return [ 'header', 'client', 'request_type', 'endpoint', 'model', 'number', 'messages',
		         'response_format', 'modalities', 'max_completion_tokens', 'frequency_penalty',
		         'presence_penalty', 'temperature', 'top_percent', 'store', 'stream',
		         'stops', 'content', 'response', 'prompt', 'create', 'messages', 'values',
		         'instructions' ]


class GptResponse( ):
	'''
		Base class for GPT responses.
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class CompletionResponse( GptResponse ):
	'''
		Class containing the GPT response for the chat completion request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class TextResponse( GptResponse ):
	'''
		Class containing the GPT response for the documents generation
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class EmbeddingResponse( GptResponse ):
	'''
		Class containing the GPT response for the embedding request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class FineTuningResponse( GptResponse ):
	'''
		Class containing the GPT response for the fine tuning request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class VectorResponse( GptResponse ):
	'''
		Class containing the GPT response for the vector request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class FileResponse( GptResponse ):
	'''
		Class containing the GPT response for the file request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class UploadResponse( GptResponse ):
	'''
		References
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class ImageResponse( GptResponse ):
	'''
		Class containing the GPT response for the image request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		self.data = { 'id': f'{self.id}',
		              'object': f'{self.object}',
		              'model': f'{self.model}',
		              'created': f'{self.created}' }
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'id', 'object', 'model', 'created' ]


class GptMessage( ):
	'''

		Base class for all messages used in the GPT application

	'''
	def __init__( self, prompt: str, role: str, type: str ):
		self.content = prompt
		self.role = role
		self.type = type
		self.data = { 'role': f'{self.role}',
		              'type': f'{self.type}',
		              'content': f'{self.content}' }
	
	
	def __str__( self ) -> str:
		'''

			Returns: the json string representation of the message.

		'''
		new = '\r\n'
		if not self.content is None:
			_pair = f'''
            'role': '{self.role}', \r\n
            'type': '{self.type}', \r\n
            'content': '{self.content}'
            '''
			_retval = '{ ' + _pair + ' }'
			return _retval
	
	
	def dump( self ) -> str:
		'''

			Returns: key value pairs in a string

		'''
		new = '\r\n'
		return 'role' + f' = {self.role}' + new + \
			'type' + f' = {self.type}' + new + \
			'content' + f' = {self.content}'
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'role', 'content', 'type' ]
	
	
class SystemMessage( GptMessage ):
	'''
		
		Class representing the system message
		
	'''
	def __init__( self, prompt: str, role: str='system', type: str='documents' ) -> None:
		super( ).__init__( prompt, role, type )
		self.content = prompt
		self.role = role
		self.type = type
		self.data = { 'role': f'{self.role}',
		              'type': f'{self.type}',
		              'content': f'{self.content}' }
	
	
	def __str__( self ) -> str:
		'''

			Returns: the json string representation of the message.

		'''
		new = '\r\n'
		if not self.content is None:
			_pair = f'''
            'role': '{self.role}', \r\n
            'type': '{self.type}', \r\n
            'content': '{self.content}'
            '''
			_retval = '{ ' + _pair + ' }'
			return _retval
	
	
	def dump( self ) -> str:
		'''

			Returns: key value pairs in a string

		'''
		new = '\r\n'
		return 'role' + f' = {self.role}' + new + \
			'type' + f' = {self.type}' + new + \
			'content' + f' = {self.content}'
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'role', 'content', 'type' ]
		
		
class UserMessage( GptMessage ):
	'''
		
		Class representing the system message
		
	'''
	def __init__( self, prompt: str, role: str='user', type: str='text' ) -> None:
		super( ).__init__( prompt, role, type )
		self.content = prompt
		self.role = role
		self.type = type
		self.data = { 'role': f'{self.role}',
		              'type': f'{self.type}',
		              'content': f'{self.content}' }
	
	
	def __str__( self ) -> str:
		'''

			Returns: the json string representation of the message.

		'''
		new = '\r\n'
		if not self.content is None:
			_pair = f'''
            'role': '{self.role}', \r\n
            'type': '{self.type}', \r\n
            'content': '{self.content}'
            '''
			_retval = '{ ' + _pair + ' }'
			return _retval
	
	
	def dump( self ) -> str:
		'''

			Returns: key value pairs in a string

		'''
		new = '\r\n'
		return 'role' + f' = {self.role}' + new + \
			'type' + f' = {self.type}' + new + \
			'content' + f' = {self.content}'
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'role', 'content', 'type' ]
		
		
class DeveloperMessage( GptMessage ):
	'''
		
		Class representing the system message
		
	'''
	def __init__( self, prompt: str, role: str='developer', type: str='text' ) -> None:
		super( ).__init__( prompt, role, type )
		self.content = prompt
		self.role = role
		self.type = type
		self.data = { 'role': f'{self.role}',
		              'type': f'{self.type}',
		              'content': f'{self.content}' }
	
	
	def __str__( self ) -> str:
		'''

			Returns: the json string representation of the message.

		'''
		new = '\r\n'
		if not self.content is None:
			_pair = f'''
            'role': '{self.role}', \r\n
            'type': '{self.type}', \r\n
            'content': '{self.content}'
            '''
			_retval = '{ ' + _pair + ' }'
			return _retval
	
	
	def dump( self ) -> str:
		'''

			Returns: key value pairs in a string

		'''
		new = '\r\n'
		return 'role' + f' = {self.role}' + new + \
			'type' + f' = {self.type}' + new + \
			'content' + f' = {self.content}'
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'role', 'content', 'type' ]
		
		
class AssistantMessage( GptMessage ):
	'''
		
		Class representing the system message
		
	'''
	def __init__( self, prompt: str, role: str='assistant', type: str='text' ) -> None:
		super( ).__init__( prompt, role, type )
		self.content = prompt
		self.role = role
		self.type = type
		self.data = { 'role': f'{self.role}',
		              'type': f'{self.type}',
		              'content': f'{self.content}' }
	
	
	def __str__( self ) -> str:
		'''

			Returns: the json string representation of the message.

		'''
		new = '\r\n'
		if not self.content is None:
			_pair = f'''
            'role': '{self.role}', \r\n
            'type': '{self.type}', \r\n
            'content': '{self.content}'
            '''
			_retval = '{ ' + _pair + ' }'
			return _retval
	
	
	def dump( self ) -> str:
		'''

			Returns: key value pairs in a string

		'''
		new = '\r\n'
		return 'role' + f' = {self.role}' + new + \
			'type' + f' = {self.type}' + new + \
			'content' + f' = {self.content}'
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'role', 'content', 'type' ]


class ChatLog(  ):
	'''
	
		Class used to encapsulate a collection of chat messages.
	
	'''
	def __init__( self ):
		self.messages = [ GptMessage ]
		
		
class Perceptron:
	'''
		Purpose
		________
		
		Class to train models via fit function
		
		
		Parameters
		------------
		eta : float
		Learning rate (between 0.0 and 1.0)
		n_iter : int
		Passes over the training dataset.
		random_state : int
		Random number generator seed for random weight
		initialization.
		
		
		Attributes
		-----------
		w_ : 1d-array
		Weights after fitting.
		b_ : Scalar
		Bias unit after fitting.
		errors_ : list
		Number of misclassifications (updates) in each epoch.
	
	
	'''
	def __init__( self, eta=0.01, n_iter=50, random_state=1 ):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state
	
	
	def fit( self, X, y ):
		"""
		
			Purpose
			_______
			Fit training values.
			
			
			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vectors, where n_examples is the number of
			examples and n_features is the number of features.
			
			y : array-like, shape = [n_examples]
			Target values.
			
			Returns
			-------
			self : object
		
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			elif y is None:
				raise Exception( 'y is not provided.' )
			else:
				rgen = np.random.RandomState( self.random_state )
				self.w_ = rgen.normal( loc=0.0, scale=0.01, size=X.shape[ 1 ] )
				self.b_ = np.float_( 0. )
				self.errors_ = [ ]
				
				for _ in range( self.n_iter ):
					errors = 0
					
				for xi, target in zip( X, y ):
					update = self.eta * ( target - self.predict( xi ) )
					
				self.w_ += update * xi
				self.b_ += update
				errors += int( update != 0.0 )
				self.errors_.append( errors )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Perceptron'
			exception.method = 'fit( self, values, y )'
			error = ErrorDialog( exception )
			error.show( )
		
		
	def net_input( self, X ):
		"""
			Calculate net input
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			else:
				return np.dot( X, self.w_ ) + self.b_
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Perceptron'
			exception.method = 'net_input( self, values ):'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X ):
		"""
		Return class label after unit step
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			else:
				return np.where( self.net_input( X ) >= 0.0, 1, 0 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Perceptron'
			exception.method = 'predict( self, values )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
			Methods that returns a list of member names
			Returns: list[ str ]
		'''
		return [ 'fit', 'net_input', 'predict' ]


class AdaptiveLinearNeuron( ):
	"""
		
		Purpose
		___________
		Adaptive Linear Neuron classifier.
		
		Parameters
		------------
		eta : float
		Learning rate (between 0.0 and 1.0)
		n_iter : int
		Passes over the training dataset.
		random_state : int
		Random number generator seed for random weight initialization.
		
		Attributes
		-----------
		w_ : 1d-array
		Weights after fitting.
		b_ : Scalar
		Bias unit after fitting.
		losses_ : list
		Mean squared error loss function values in each epoch.
	
	"""
	def __init__( self, eta=0.01, n_iter=50, random_state=1 ):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state
		
		
	def fit( self, X, y ):
		""" Fit training values.

			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vectors, where n_examples
			is the number of examples and
			n_features is the number of features.
			y : array-like, shape = [n_examples] Target values.
	
			Returns
			-------
			self : object
		
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			elif y is None:
				raise Exception( 'y is not provided.' )
			else:
				rgen = np.random.RandomState(self.random_state)
				self.w_ = rgen.normal(loc=0.0, scale=0.01,
				size=X.shape[1])
				self.b_ = np.float_(0.)
				self.losses_ = []
				for i in range(self.n_iter):
					net_input = self.net_input(X)
					
				output = self.activation(net_input)
				errors = (y - output)
				self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
				self.b_ += self.eta * 2.0 * errors.mean()
				loss = ( errors**2 ).mean( )
				self.losses_.append( loss )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'AdaptiveLinearNeuron'
			exception.method = 'fit( self, values, y )'
			error = ErrorDialog( exception )
			error.show( )


	def net_input( self, X ):
		"""
			
			Calculate net
			
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			else:
				return np.dot( X, self.w_ ) + self.b_
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'AdaptiveLinearNeuron'
			exception.method = 'net_input( self, values )'
			error = ErrorDialog( exception )
			error.show( )


	def activation( self, X):
		"""
		
			Compute linear activation
		
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			else:
				return X
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'AdaptiveLinearNeuron'
			exception.method = 'activation( self, values)'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X ):
		"""
			
			Return class label after unit step
		
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			else:
				return np.where( self.activation( self.net_input( X ) ) >= 0.5, 1, 0 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'AdaptiveLinearNeuron'
			exception.method = 'predict( self, values )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'fit', 'net_input', 'activation', 'predict' ]


class Embedding( AI ):
	'''
	
		Class proiding embedding objects
		
	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super().api_key
		self.client = OpenAI( self.api_key )
		self.model = 'text-embedding-3-large'
		
	
	def create( self, input: str ) -> list[ float ]:
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.input = input
				_request = self.client.embeddings.create( input, self.model )
				_emedding = _request.data[ 0 ].embedding
				return _emedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Embedding'
			exception.method = 'create( self, input: str ) -> list[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'create' ]


class Response( AI ):
	'''

		Class proiding Response objects

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
	
	
	def create( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'create' ]



class Chat( AI ):
	'''

		Class proiding Response objects

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
	
	
	def create( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'create' ]



class Image( AI ):
	'''

		Class proiding Response objects

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'dall-e-3'
	
	
	def create( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'create' ]


class TTS( AI ):
	'''

		Class proiding Response objects

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'tts-1-hd'
	
	
	def create( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'create' ]


class Transcription( AI ):
	'''

		Class proiding Transciprtion objects

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'whisper-1'
	
	
	def create( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'create' ]


class Translation( AI ):
	'''

		Class proiding Response objects

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'whisper-1'
	
	
	def create( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'create' ]
	
	
class Bro( AI ):
	'''
	
		Class providing open ai functionality
		
	'''
	def __init__( self ):
		pass