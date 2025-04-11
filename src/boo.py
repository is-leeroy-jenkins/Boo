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
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import requests
import tiktoken
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
		self.system_instructions = '''You are the most knowledgeable Budget Analyst in the federal
		government who provides detailed responses based on your vast knowledge of budget
		legislation, and federal appropriations. Your responses to questions about federal finance
		are complete, transparent, and very detailed using an academic format. Your vast knowledge
		of and experience in Data Science makes you the best Data Analyst in the world. You are
		also an expert programmer who is proficient in C#, Python, SQL, C++, JavaScript, and VBA.
		You are famous for the accuracy of your responses so you verify all your answers. This
		makes the quality of your code very high because it always works. Your responses are
		always accurate and complete!  Your name is Bubba.'''


class Perceptron( ):
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
					update = self.eta * (target - self.predict( xi ))
				
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
		return [ 'fit', 'net_input', 'predict',
		         'w_', 'b_', 'errors_',
		         'n_iter', 'random_state', 'eta' ]


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
		"""
		
			Fit training values.

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
				rgen = np.random.RandomState( self.random_state )
				self.w_ = rgen.normal( loc=0.0, scale=0.01,
					size=X.shape[ 1 ] )
				self.b_ = np.float_( 0. )
				self.losses_ = [ ]
				for i in range( self.n_iter ):
					net_input = self.net_input( X )
				
				output = self.activation( net_input )
				errors = (y - output)
				self.w_ += self.eta * 2.0 * X.T.dot( errors ) / X.shape[ 0 ]
				self.b_ += self.eta * 2.0 * errors.mean( )
				loss = (errors ** 2).mean( )
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
	
	
	def activation( self, X ):
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
		return [ 'fit', 'net_input', 'activation',
		         'predict', 'losses_', 'b_', 'w_',
		         'n_iter', 'eta', 'random_state', 'embedding-3-small' ]


class SmallEmbedding( AI ):
	'''

		Class proiding embedding objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'embedding-3-small'
		self.embedding = None
		self.response = None
	
	
	def create( self, input: str ) -> list[ float ]:
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.input = input
				self.response = self.client.embeddings.create( input, self.model )
				self.embedding = self.response.data[ 0 ].embedding
				return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'SmallEmbedding'
			exception.method = 'create( self, input: str ) -> list[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''

			Methods that returns a list of member names
			Returns: list[ str ]

		'''
		return [ 'api_key', 'client', 'model', 'input', 'generate_text' ]


class AdaEmbedding( AI ):
	'''

		Class proiding embedding objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'embedding-ada-002'
		self.embedding = None
		self.response = None
	
	
	def create( self, input: str ) -> list[ float ]:
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.input = input
				self.response = self.client.embeddings.create( input, self.model )
				self.embedding = self.response.data[ 0 ].embedding
				return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'AdaEmbedding'
			exception.method = 'create( self, input: str ) -> list[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''

			Methods that returns a list of member names
			Returns: list[ str ]

		'''
		return [ 'api_key', 'client', 'model', 'input', 'generate_text' ]


class LargeEmbedding( AI ):
	'''
	
		Class proiding embedding objects
		
	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'text-embedding-3-large'
		self.embedding = None
		self.response = None
	
	
	def create( self, input: str ) -> list[ float ]:
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.input = input
				self.response = self.client.embeddings.create( input, self.model )
				self.embedding = self.response.data[ 0 ].embedding
				return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'LargeEmbedding'
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

		Class proiding Chat object functionality

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
		self.response = None
		self.input = [ ]
		self.image_url = None
		self.response_format = 'auto'
		self.tools = [ ]
		self.top_p = 0.9
		self.temperature = 0.8
	
	
	def generate_text( self, prompt: str ) -> str:
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			else:
				self.response = self.client.responses.create( model=self.model, input=prompt )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze_image( self, prompt: str, url: str ) -> str:
		try:
			if prompt is None:
				raise Exception( 'Argument "prompt" cannot be None' )
			elif url is None:
				raise Exception( 'Argument "url" cannot be None' )
			else:
				self.input =\
				[
					{
						'role': 'user',
						'content':
						[
							{ 'type': 'input_text',
							  'text': prompt
							},
							{
								'type': 'input_image',
								'image_url': url
							}
						]
					}
				]
				
				self.response = self.client.responses.create( model=self.model, input=self.input )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input',
		         'generate_text', 'analyze_image' ]


class LargeImage( AI ):
	'''

		Class proiding Response objects

	'''
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.quality = 'hd'
		self.model = 'dall-e-3'
	
	
	def generate( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'generate' ]


class Image( AI ):
	'''

		Class proiding Response objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.quality = 'standard'
		self.model = 'dall-e-2'
	
	
	def generate( self, input: str ) -> str:
		pass
	
	
	def __dir__( self ) -> list[ str ]:
		'''

			Methods that returns a list of member names
			Returns: list[ str ]

		'''
		return [ 'api_key', 'client', 'model', 'input', 'generate' ]


class Assistant( AI ):
	'''

		Class proiding Transciprtion objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
		self.response_format = 'auto'
		self.input_text = None
		self.name = None
		self.description = None
		self.id = None
		self.metadata = { }
		self.tools = [ ]
		self.top_p = 0.9
		self.temperature = 0.8
	
	
	def generate_text( self, input: str ):
		'''
			method creating transciption object given an input: str
		'''
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.input_text = input
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'create( self, input: str )'
			error = ErrorDialog( exception )
			error.show( )


class Bubba( AI ):
	'''

		Class proiding Transciprtion objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
		self.response_format = 'auto'
		self.input_text = None
		self.name = 'Bubba'
		self.description = None
		self.id = 'asst_J6SAABzDixkTYi2k39OGgjPv'
		self.metadata = { }
		self.tools = [ ]
		self.top_p = 0.9
		self.temperature = 0.8
	
	
	def generate_text( self, input: str ):
		'''
			method creating transciption object given an input: str
		'''
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.input_text = input
				self.client.beta.threads.create( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'generate_text( self, input: str )'
			error = ErrorDialog( exception )
			error.show( )


class Bro( AI ):
	'''

		Class proiding Transciprtion objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'gpt-4o-mini'
		self.response_format = 'auto'
		self.input_text = None
		self.name = 'Bro'
		self.description = None
		self.id = 'asst_2Yu2yfINGD5en4e0aUXAKxyu'
		self.metadata = { }
		self.tools = [ ]
		self.top_p = 0.9
		self.temperature = 0.8
	
	
	def generate_text( self, input: str ):
		'''
			method creating transciption object given an input: str
		'''
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.input_text = input
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Assistant'
			exception.method = 'generate_text( self, input: str )'
			error = ErrorDialog( exception )
			error.show( )


class TextToSpeech( AI ):
	'''

		Class proiding Response objects

	'''
	
	
	def __init__( self ):
		'''
			Constructor to generate_text TextToSpeech objects
		'''
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'tts-1-hd'
		self.audio_path = None
		self.response = None
		self.prompt = None
	
	
	def create( self, prompt: str, path: str ):
		'''
			method providing TextToSpeech functionality
		'''
		try:
			if path is None:
				raise Exception( 'Argument "url" is required.' )
			elif prompt is None:
				raise Exception( 'Argument "prompt" is required.' )
			else:
				self.audio_path = Path( path ).parent  # 'speech.mp3'
				self.prompt = prompt
				self.response = self.client.audio.speech.create( model=self.model, voice='alloy',
					input=prompt )
				self.response.stream_to_file( self.audio_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'TextToSpeech'
			exception.method = 'create( self, prompt: str, input: str )]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'generate_text' ]


class Transcription( AI ):
	'''

		Class proiding Transciprtion objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'whisper-1'
		self.input_text = None
		self.audio_file = None
		self.transcript = None
	
	
	def create( self, input: str ):
		'''
			method creating transciption object given an input: str
		'''
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.audio_file = open( 'boo.mp3', 'rb' )
				self.input_text = input
				self.response = self.client.audio.speech.create( model=self.model,
					voice='alloy', input=self.input_text )
				self.response.stream_to_file( self.audio_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Transcription'
			exception.method = 'creat( self, input: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'generate_text' ]


class Translation( AI ):
	'''

		Class proiding Response objects

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = super( ).api_key
		self.client = OpenAI( self.api_key )
		self.model = 'whisper-1'
		self.audio_file = None
		self.response = None
	
	
	def create( self, input: str ):
		'''
			Method creating a translation from one language to another
		'''
		try:
			if input is None:
				raise Exception( 'Argument "input" is required.' )
			else:
				self.audio_file = open( 'boo.mp3', 'rb' )
				self.response = self.client.audio.translations.create( model='whisper-1',
					file=self.audio_file )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Translation'
			exception.method = 'create( self, input: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a list of member names
			Returns: list[ str ]
			
		'''
		return [ 'api_key', 'client', 'model', 'input', 'generate_text' ]
