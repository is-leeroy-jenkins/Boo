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
import numpy as np
import pandas as pd
import datetime as dt
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import requests
import tiktoken
from pygments.lexers.csound import newline
from static import GptRequests, GptRoles, GptLanguages
from booger import ErrorDialog, Error
from typing import Any, List, Tuple, Optional, Dict


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
		
			Methods that returns a get_list of member names
			Returns: list[ str ]
			
		'''
		return [ 'content_type', 'api_key', 'authorization', 'values' ]

	def __dir__( self ) -> List[ str ]:
		'''
		
			Methods that returns a get_list of member names
			Returns: list[ str ]
			
		'''
		return [ 'base_url', 'text_generation', 'image_generation', 'chat_completions',
		         'speech_generation', 'translations', 'assistants', 'transcriptions',
		         'finetuning', 'vectors', 'uploads', 'files', 'vector_stores',
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
		         'vectors': self.embeddings,
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
			'vectors' + f' = {self.files}' + new + \
			'uploads' + f' = {self.uploads}' + new + \
			'files' + f' = {self.files}' + new + \
			'vector_stores' + f' = {self.vector_stores}' + new


class Models( ):
	'''
	
		Purpose
		_______
		
		Class containing lists of OpenAI models by generation
		
	'''
	
	def __init__( self ):
		self.text_generation = [ 'pages-davinci-003', 'pages-curie-001',
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
		self.transcription = [ 'whisper-1', 'gpt-4o-mini-transcribe', ' openai-4o-transcribe' ]
		self.translation = [ 'whisper-1', 'pages-davinci-003',
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
		self.embeddings = [ 'pages-embedding-3-small', 'pages-embedding-3-large',
		                    'pages-embedding-ada-002' ]
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
		self.bubba = [ 'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-budget-training:BGVjoSXv',
		                'ft:gpt-4o-2024-08-06:leeroy-jenkins:budget-base-training:BGVk5Ii1',
		                'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-base-training:BGVAJg57' ]

		self.bro = [ 'ft:gpt-4o-2024-08-06:leeroy-jenkins:bubba-budget-training:BGVjoSXv',
		                    'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-fine-tuned:BTc3PMb5',
		                    'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-analytics:BTX4TYqY' ]
	
	
	def __dir__( self ) -> List[ str ]:
		'''
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
		'''
		return [ 'base_url', 'text_generation', 'image_generation', 'chat_completion',
		         'speech_generation', 'responses', 'reasoning',
		         'translations', 'assistants', 'transcriptions',
		         'finetuning', 'vectors', 'uploads', 'files', 'vector_stores',
		         'bubba', 'bro' ]
	
	
	def get_data( self ) -> dict:
		'''
			
			Method that returns a get_list of dictionaries
		
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
		self.bro_instructions = '''
		You are an assistant who is the most knowledgeable Data Scientist in the world.   You
		provide detailed responses based on your vast knowledge of federal appropriations and your
		knowledge of computer programming.  Your responses to questions are always complete and
		detailed using an academic format.  Your vast knowledge of and experience in Data Science
		makes you the best Analyst in the world. You are an expert programmer proficient in C#,
		Python, SQL, C++, JavaScript, and VBA.  Your name is Bro because your code just works!
		'''
		self.bubba_instructions = '''You are the most knowledgeable Budget Analyst in the federal
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
		Random num generator seed for random weight
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
		"""
		
			Purpose
			_______
			Initializes Perceptron opbjects
			
			
			Parameters
			----------
			eta: flaot.
			The learning rate (between 0.0 and 1.0)
			
			n_iter: int
			Target values.
			
			random_state: int
			Epochs.
			
			Returns
			-------
			self : object
		
		"""
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
			Training vec, where n_examples is the num of
			examples and n_features is the num of features.
			
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
				self.b_ = np.float64( 0. )
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
			exception.method = 'fit( self, X y )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def net_input( self, X ):
		"""
		
			Purpose
			_______
			Calculates net text
			
			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vec, where n_examples is the num of
			examples and n_features is the num of features.
			

			Returns
			-------
			np.array
		
		"""
		try:
			if X is None:
				raise Exception( 'Aurguent "X" is not provided.' )
			else:
				return np.dot( X, self.w_ ) + self.b_
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Perceptron'
			exception.method = 'net_input( self, X ):'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X ):
		"""
		
			Purpose
			_______
			Calculates prediction
			
			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vec, where n_examples is the num of
			examples and n_features is the num of features.
			

			Returns
			-------
			np.array
		
		"""
		try:
			if X is None:
				raise Exception( 'Aurguent "X" is not provided.' )
			else:
				return np.where( self.net_input( X ) >= 0.0, 1, 0 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Perceptron'
			exception.method = 'predict( self, X )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> List[ str ]:
		'''
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
		'''
		return [ 'fit', 'net_input', 'predict',
		         'w_', 'b_', 'errors_',
		         'n_iter', 'random_state', 'eta' ]


class LinearGradientDescent( ):
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
		Random num generator seed for random weight initialization.
		
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
		"""
		
			Purpose
			_______
			Initializes LinearGradientDescent opbjects
			
			
			Parameters
			----------
			eta: flaot=0.01
			The learning rate (between 0.0 and 1.0)
			
			n_iter: int: 50
			Target values.
			
			random_state: int:1
			Epochs.
			
			Returns
			-------
			self : object
		
		"""
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state
	
	
	def fit( self, X, y ):
		"""
		
			Fit training values.

			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vec, where n_examples
			is the num of examples and
			n_features is the num of features.
			
			y : array-like, shape = [n_examples]
			Target values.
	
			Returns
			-------
			self : object
		
		"""
		try:
			if X is None:
				raise Exception( 'Aurguent "X" is not provided.' )
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
			exception.cause = 'LinearGradientDescent'
			exception.method = 'fit( self, X, y )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def net_input( self, X ):
		"""
		
			Purpose
			_______
			Calculates net text
			
			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vec, where n_examples is the num of
			examples and n_features is the num of features.
			

			Returns
			-------
			np.array
		
		"""
		try:
			if X is None:
				raise Exception( 'values is not provided.' )
			else:
				return np.dot( X, self.w_ ) + self.b_
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'LinearGradientDescent'
			exception.method = 'net_input( self, X )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def activation( self, X ):
		"""
		
			Purpose
			_______
			Computes linear activation
			
			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vec, where n_examples is the num of
			examples and n_features is the num of features.
			

			Returns
			-------
			X : {array-like}, shape = [n_examples, n_features]
			Training vec, where n_examples is the num of
			examples and n_features is the num of features.
		
		"""
		try:
			if X is None:
				raise Exception( 'Aurguent "X" is not provided.' )
			else:
				return X
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'LinearGradientDescent'
			exception.method = 'activation( self, X )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X ):
		"""
		
			Purpose
			_______
			Computes linear activation
			
			Parameters
			----------
			X : {array-like}, shape = [n_examples, n_features]
			Training vec, where n_examples is the num of
			examples and n_features is the num of features.
			

			Returns
			-------
			np.array
		
		"""
		try:
			if X is None:
				raise Exception( 'Aurguent "X" is not provided.' )
			else:
				return np.where( self.activation( self.net_input( X ) ) >= 0.5, 1, 0 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'LinearGradientDescent'
			exception.method = 'predict( self, values )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
			
		'''
		return [ 'fit', 'net_input', 'activation',
		         'predict', 'losses_', 'b_', 'w_',
		         'n_iter', 'eta', 'random_state'  ]

class SystemMessage(  ):
	'''

		Class representing the system message

	'''
	def __init__( self, prompt: str, type: str='text' ) -> None:
		self.content = prompt
		self.role = 'system'
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
		
			Methods that returns a get_list of member names
			Returns: List[ str ]
			
		'''
		return [ 'role', 'content', 'type' ]


class UserMessage( ):
	'''

		Class representing the system message

	'''
	def __init__( self, prompt: str, type: str='text' ) -> None:
		self.content = prompt
		self.role = 'user'
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
	
	
	def __dir__( self ) -> List[ str ]:
		'''

			Methods that returns a get_list of member names
			Returns: get_list[ str ]

		'''
		return [ 'role', 'content', 'type' ]


class DeveloperMessage( ):
	'''

		Class representing the system message

	'''
	def __init__( self, prompt: str, type: str='text' ) -> None:
		self.content = prompt
		self.role = 'developer'
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
	
	
	def __dir__( self ) -> List[ str ]:
		'''
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
		'''
		return [ 'role', 'content', 'type' ]


class AssistantMessage( ):
	'''

		Class representing the system message

	'''
	
	
	def __init__( self, prompt: str, type: str='text' ) -> None:
		self.content = prompt
		self.role = 'assistant'
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
	
	
	def __dir__( self ) -> List[ str ]:
		'''
		
			Methods that returns a get_list of member names
			Returns: List[ str ]
			
		'''
		return [ 'role', 'content', 'type' ]


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
		max: int=2048
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
		self.text, self.messages, self.image_url, self.response_format,
		self.tools, self.vector_store_ids
		
		Methods
		------------
		get_model_options( self ) -> list[ str ]
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		dump( self ) -> str
		get_data( self ) -> { }
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
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
		self.vector_store_ids = [ 'vs_712r5W5833G6aLxIYIbuvVcK', 'vs_8fEoYp1zVvk5D8atfWLbEupN' ]
	
	
	def get_model_options( self ) -> str:
		'''
		
			Methods that returns a list of small_model names
		
		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31' ]
	
	
	def generate_text( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Generates a chat completion given a string prompt
			
			
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
				self.response = self.client.responses.create( model=self.model, input=self.prompt )
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
			Generates a chat completion given a string prompt
			
			
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

			return self.response.data[0].url
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
			Method that analyzeses an image given a string prompt,
			
			
			
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
						'content':
							[
								{ 'type': 'input_text',
								  'pages': self.prompt
								  },
								{
									'type': 'input_image',
									'image_url': self.image_url
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
	
	
	def summarize( self, prompt: str, path: str ) -> str:
		"""
		
			Purpose
			_______
			Method that summarizes a document given a
			string prompt, and a path
	
			
			
			
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
				self.file_path = path
				self.file = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose='user_data' )
				
				self.messages = [
					{
						'role': 'user',
						'content': [
							{
								'type': 'file',
								'file':
									{
										'file_id': file.id,
									}
							},
							{
								'type': 'text',
								'pages': 'What is the first dragon in the book?',
							},
						]
					}
				]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'summarize( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a string prompt,
			
			
			
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
				
				return  self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


	def search_file( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a string prompt,
			
			
			
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
					'type': 'file_search',
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
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
			
		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'text', 'messages', 'image_url', 'respose_format', 'tools',
		         'vector_store_ids', 'size', 'api_key', 'client', 'small_model',
		         'generate_text', 'analyze_image', 'summarize', 'generate_image',
		         'translate', 'transcribe',
		         'search_web', 'search_files', 'get_data', 'dump' ]
	
	
class Assistant( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with OpenAI's
		Assistants API
		
		
		Parameters
		------------
		num: int=1
		temp: float=0.8
		top: float=0.9
		freq: float=0.0
		pres: float=0.0
		max: int=2048
		store: bool=True
		stream: bool=True
		
		Methods
		------------
		get_model_options( self ) -> str
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
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
		self.modalities = [ 'pages', 'audio', 'auto' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'auto'
		self.reasoning_effort = 'auto'
		self.input_text = None
		self.name = 'Boo'
		self.description = 'Generic Assistant'
		self.id = None
		self.metadata = { }
		self.tools = [ ]
		self.assistants = [ ]
	
	
	def generate_text( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Generates a chat completion given a string prompt
			
			
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
			Generates a chat completion given a string prompt
			
			
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

			return self.response.data[0].url
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
			Method that analyzeses an image given a string prompt,
			
			
			
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
						'content':
							[
								{ 'type': 'input_text',
								  'pages': self.prompt
								  },
								{
									'type': 'input_image',
									'image_url': self.image_url
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
	
	
	def summarize( self, prompt: str, path: str ) -> str:
		"""
		
			Purpose
			_______
			Method that summarizes a document given a
			string prompt, and a path
	
			
			
			
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
				self.file_path = path
				self.file = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose='user_data' )
				
				self.messages = [
					{
						'role': 'user',
						'content': [
							{
								'type': 'file',
								'file':
									{
										'file_id': file.id,
									}
							},
							{
								'type': 'text',
								'pages': 'What is the first dragon in the book?',
							},
						]
					}
				]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'summarize( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a string prompt,
			
			
			
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
				
				return  self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


	def search_file( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a string prompt,
			
			
			
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
					'type': 'file_search',
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
	

	def get_list( self ) -> List[ str ]:
		'''
		
			Method that returns a list of available assistants
			
		'''
		try:
			self.assistants = self.client.beta.assistants.list( order="desc", limit="20" )
			return self.assistants.data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'get_list( ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )


	def get_format_options( ):
		'''
		
			Method that returns a list of formatting options
		
		'''
		return [ 'auto', 'text', 'json' ]
	
	
	def get_model_options( ):
		'''

			Method that returns a list of available models

		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31'  ]
	
	
	def get_effort_options( ):
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
	
	
	def __dir__( self ):
		'''
		
			Method that returns a list of members
		
		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens', 'system_instructions',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'text', 'messages', 'image_url', 'respose_format', 'tools',
		         'vector_store_ids', 'name', 'id', 'description', 'generate_text',
		         'get_format_options', 'get_model_options', 'reasoning_effort'
		         'get_effort_options', 'input_text', 'metadata',
		         'get_list', 'get_data',
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
		max: int=2048
		store: bool=True
		stream: bool=True
		
		
		Methods
		------------
		get_model_options( self ) -> str
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		dump( self ) -> str
		get_data( self ) -> { }
		
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
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
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'auto'
		self.reasoning_effort = 'auto'
		self.input_text = None
		self.name = 'Bubba'
		self.description = 'A Budget Execution & Data Analysis Assistant'
		self.id = 'asst_J6SAABzDixkTYi2k39OGgjPv'
		self.vector_store_ids = [ 'vs_8fEoYp1zVvk5D8atfWLbEupN', 'vs_712r5W5833G6aLxIYIbuvVcK' ]
		self.metadata = { }
		self.tools = [ ]
	
	
	def generate_text( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Generates a chat completion given a string prompt
			
			
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
			Generates a chat completion given a string prompt
			
			
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

			return self.response.data[0].url
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
			Method that analyzeses an image given a string prompt,
			
			
			
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
					{ 'type': 'input_text',
					  'pages': self.prompt
					},
					{
						'type': 'input_image',
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
	
	
	def summarize( self, prompt: str, path: str ) -> str:
		"""
		
			Purpose
			_______
			Method that summarizes a document given a
			string prompt, and a path
	
			
			
			
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
				self.file_path = path
				self.file = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose='user_data' )
				
				self.messages = [
					{
						'role': 'user',
						'content': [
							{
								'type': 'file',
								'file':
									{
										'file_id': file.id,
									}
							},
							{
								'type': 'text',
								'pages': 'What is the first dragon in the book?',
							},
						]
					}
				]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'summarize( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Method that analyzeses an image given a string prompt,
			
			
			
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
				
				return  self.response.output_text
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
			Method that analyzeses an image given a string prompt,
			
			
			
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
					'type': 'file_search',
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
	
	
	def get_format_options( ):
		'''
		
			Method that returns a list of formatting options
		
		'''
		return [ 'auto', 'text', 'json' ]
	
	
	def get_model_options( ):
		'''

			Method that returns a list of available models

		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31'  ]
	
	
	def get_effort_options( ):
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
	
	
	def __dir__( self ):
		'''
		
			Method that returns a list of members
		
		'''
		return  [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens', 'system_instructions',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'text', 'messages', 'image_url', 'respose_format', 'tools',
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
		max: int=2048
		store: bool=True
		stream: bool=True
		
		
		Methods
		------------
		get_model_options( self ) -> str
		generate_text( self, prompt: str ) -> str:
		analyze_image( self, prompt: str, url: str ) -> str:
		summarize( self, prompt: str, path: str ) -> str
		search_web( self, prompt: str ) -> str
		search_files( self, prompt: str ) -> str
		dump( self ) -> str
		get_data( self ) -> { }
		
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.system_instructions = AI( ).bro_instructions
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
		self.response_format = 'auto'
		self.reasoning_effort = None
		self.input_text = None
		self.name = 'Bro'
		self.description = 'A Computer Programming, Data Science and Analysis Assistant'
		self.id = 'asst_2Yu2yfINGD5en4e0aUXAKxyu'
		self.vector_store_ids = [ 'vs_67e83bdf8abc81918bda0d6b39a19372', ]
		self.metadata = { }
		self.tools = [ ]
	
	
	def generate_text( self, prompt: str ) -> str:
		"""
		
			Purpose
			_______
			Generates a chat completion given a string prompt
			
			
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
			Generates a chat completion given a string prompt


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
			Method that analyzeses an image given a string prompt,



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
						'content':
							[
								{ 'type': 'input_text',
								  'pages': self.prompt
								  },
								{
									'type': 'input_image',
									'image_url': self.image_url
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
	
	
	def summarize( self, prompt: str, path: str ) -> str:
		"""

			Purpose
			_______
			Method that summarizes a document given a
			string prompt, and a path




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
				self.file_path = path
				self.file = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose='user_data' )
				
				self.messages = [
				{
					'role': 'user',
					'content': [
						{
							'type': 'file',
							'file':
								{
									'file_id': file.id,
								}
						},
						{
							'type': 'text',
							'pages': 'What is the first dragon in the book?',
						},
					]
				} ]
				
				self.completion = self.client.chat.completions.create( model=self.model,
					messages=self.messages )
				return self.completion.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Chat'
			exception.method = 'summarize( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def search_web( self, prompt: str ) -> str:
		"""

			Purpose
			_______
			Method that analyzeses an image given a string prompt,



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
	
	
	def search_file( self, prompt: str ) -> str:
		"""

			Purpose
			_______
			Method that analyzeses an image given a string prompt,



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
					'type': 'file_search',
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
	
	
	def get_format_options( ):
		'''
		
			Method that returns a list of formatting options
		
		'''
		return [ 'auto', 'text', 'json' ]
	
	
	def get_model_options( ):
		'''

			Method that returns a list of available models

		'''
		return [ 'gpt-4-0613', 'gpt-4-0314',
                 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06',
                 'gpt-4o-2024-11-20', 'gpt-4o-2024-05-13',
                 'gpt-4o-mini-2024-07-18', 'o1-2024-12-17',
                 'o1-mini-2024-09-12', 'o3-mini-2025-01-31'  ]
	
	
	def get_effort_options( ):
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
	
	

	def __dir__(self):
		'''
		
			Method that returns a list of members
		
		'''
		return  [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens', 'system_instructions',
		         'store', 'stream', 'modalities', 'stops', 'content',
		         'prompt', 'response', 'completion', 'file', 'path',
		         'text', 'messages', 'image_url', 'respose_format', 'tools',
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
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
		super( ).__init__( )
		self.client = OpenAI( self.api_key )
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
	
	
	def create_small( self, text: str ) -> List[ float ]:
		"""

			Purpose
			_______
			Creates an embedding ginve a string text


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
			exception.method = 'create_small( self, text: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_large( self, text: str ) -> List[ float ]:
		"""

			Purpose
			_______
			Creates an Large embedding given a string text


			Parameters
			----------
			text: str


			Returns
			-------
			list[ float ]

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
			exception.method = 'create_large( self, text: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_ada( self, text: str ) -> List[ float ]:
		"""

			Purpose
			_______
			Creates an ADA embedding given a string text


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
				self.response = self.client.embeddings.create( self.input, self.ada_model )
				self.embedding = self.response.data[ 0 ].embedding
				return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Embedding'
			exception.method = 'create_ada( self, text: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def get_data( self ) -> Dict:
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
	
	
	def __dir__( self ) -> List[ str ]:
		'''

			Methods that returns a get_list of member names
			Returns: get_list[ str ]

		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'api_key', 'client', 'small_model',
		         'text', 'create_small_embedding', 'get_model_options' ]


class TTS( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with OpenAI's Audio API (TTS)
		
		
		Parameters
		------------
		num: int=1
		temp: float=0.8
		top: float=0.9
		freq: float=0.0
		pres: float=0.0
		max: int=2048
		store: bool=True
		stream: bool=True
		
		Attributes
		-----------
		self.api_key, self.system_instructions, self.client, self.small_model,  self.reasoning_effort,
		self.response, self.num, self.temperature, self.top_percent,
		self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
		self.store, self.stream, self.modalities, self.stops, self.content,
		self.input_text, self.response, self.completion, self.file, self.path,
		self.text, self.messages, self.image_url, self.response_format,
		self.tools, self.vector_store_ids, self.descriptions, self.assistants
		
		Methods
		------------
		get_model_options( self ) -> str
		create_small_embedding( self, prompt: str, path: str )
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
		'''
			Constructor to  create_small_embedding TTS objects
		'''
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.model = 'tts-1-hd'
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
		
			Methods that returns a list of small_model names
		
		'''
		return [ 'tts-1', 'tts-1-hd',
                 'gpt-4o-audio-preview-2024-12-17',
                 'gpt-4o-audio-preview-2024-10-01',
                 'gpt-4o-mini-audio-preview-2024-12-17' ]
	
	
	def get_voice_options( self ):
		'''
		
			Method that returns a list of voice names
		
		'''
		return [ 'alloy', 'ash', 'ballad', 'coral',
		         'echo', 'fable', 'onyx', 'nova',
		         'sage', 'shiver' ]
	
	
	def get_format_options( self ):
		'''
		
			Method that returns a list of image formats
		
		'''
		return [ 'mp3', 'wav', 'aac', 'flac', 'opus', 'pcm']
	
	
	def create( self, prompt: str, path: str ):
		"""
		
			Purpose
			_______
			Generates audio given a string prompt and path to audio file
			
			
			Parameters
			----------
			prompt: str
			path: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if path is None:
				raise Exception( 'Argument "url" is required.' )
			elif prompt is None:
				raise Exception( 'Argument "prompt" is required.' )
			else:
				self.audio_path = Path( path ).parent
				self.prompt = prompt
				self.response = self.client.audio.speech.with_streaming_response( model=self.model,
					voice=self.voice, input=self.prompt )
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'TTS'
			exception.method = 'create_small_embedding( self, prompt: str, text: str )]'
			error = ErrorDialog( exception )
			error.show( )
	
	
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
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
			
		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'prompt', 'response', 'completion', 'audio_path',
		         'text', 'messages', 'respose_format', 'tools',
		         'size', 'api_key', 'client', 'small_model', 'voice',
		         'generate_text', 'get_model_options' ]


class Transcription( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with OpenAI's Audio API (whisper-1)
		
		
		Parameters
		------------
		num: int=1
		temp: float=0.8
		top: float=0.9
		freq: float=0.0
		pres: float=0.0
		max: int=2048
		store: bool=True
		stream: bool=True
		
		Attributes
		-----------
		self.api_key, self.system_instructions, self.client, self.small_model,  self.reasoning_effort,
		self.response, self.num, self.temperature, self.top_percent,
		self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
		self.store, self.stream, self.modalities, self.stops, self.content,
		self.input_text, self.response, self.completion, self.audio_file, self.transcript
		
		
		Methods
		------------
		get_model_options( self ) -> str
		create_small_embedding( self, text: str  ) -> str
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
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

			Methods that returns a list of small_model names

		'''
		return [ 'whisper-1', 'gpt-4o-mini-transcribe', ' openai-4o-transcribe' ]
	
	
	def create( self, input: str ) -> str:
		"""
		
			Purpose
			_______
			Generates a transcription given a string path to an audio file
			
			
			Parameters
			----------
			input: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if input is None:
				raise Exception( 'Argument "text" is required.' )
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
			exception.method = 'create_small_embedding( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
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
	
	
	def __dir__( self ) -> list[ str ]:
		'''
		
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
			
		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'prompt', 'response', 'audio_file',
		         'text', 'messages', 'respose_format',
		         'api_key', 'client', 'small_model', 'create_small_embedding',
		         'input_text', 'transcript' ]


class Translation( AI ):
	"""
		
		Purpose
		___________
		Class used for interacting with OpenAI's Audio API (whisper-1)
		
		
		Parameters
		------------
		num: int=1
		temp: float=0.8
		top: float=0.9
		freq: float=0.0
		pres: float=0.0
		max: int=2048
		store: bool=True
		stream: bool=True
		
		Attributes
		-----------
		self.api_key, self.system_instructions, self.client, self.small_model,  self.reasoning_effort,
		self.response, self.num, self.temperature, self.top_percent,
		self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
		self.store, self.stream, self.modalities, self.stops, self.content,
		self.input_text, self.response, self.completion, self.file, self.path,
		self.text, self.messages, self.image_url, self.response_format,
		self.tools, self.vector_store_ids, self.descriptions, self.assistants
		
		Methods
		------------
		create_small_embedding( self, prompt: str, path: str )
		
	
	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=True, stream: bool=True ):
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

			Methods that returns a list of small_model names

		'''
		return [ 'whisper-1', 'text-davinci-003',
		         'gpt-4-0613', 'gpt-4-0314',
		         'gpt-4-turbo-2024-04-09' ]
	
	
	def get_voice_options( self ):
		'''
		
			Method that returns a list of voice names
		
		'''
		return [ 'alloy', 'ash', 'ballad', 'coral',
		         'echo', 'fable', 'onyx', 'nova',
		         'sage', 'shiver' ]
	
	
	def create( self, input: str ):
		"""
		
			Purpose
			_______
			Generates a transcription given a string path to an audio file
			
			
			Parameters
			----------
			input: str
			
			
			Returns
			-------
			str
		
		"""
		try:
			if input is None:
				raise Exception( 'Argument "text" is required.' )
			else:
				self.audio_file = open( 'boo.mp3', 'rb' )
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
	
	
	def __dir__( self ) -> List[ str ]:
		'''
		
			Methods that returns a get_list of member names
			Returns: get_list[ str ]
			
		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'prompt', 'response', 'completion', 'audio_path',
		         'text', 'messages', 'respose_format', 'tools',
		         'size', 'api_key', 'client',
		         'small_model', 'create_small_embedding', 'get_model_options' ]


class LargeImage( AI ):
	"""

		Purpose
		___________
		Class used for generating images OpenAI's
		Images API and dall-e-3


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
		generate( self, text: str ) -> str:
		analyze( self, text: str, path: str ) -> str
		get_detail_options( self ) -> list[ str ]
		get_format_options( self ) -> list[ str ]:
		get_size_options( self ) -> list[ str ]

	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=False ):
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
			Method that analyzeses an image given a string prompt,



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
				raise Exception( 'The "input" argument is required.' )
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
			exception.method = 'generate( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, input: str, path: str ) -> str:
		'''

			Method providing image analysis functionality given a prompt and filepath

		'''
		try:
			if input is None:
				raise Exception( 'The argument "text" cannot be None' )
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
								{ 'type': 'input_text',
								  'text': self.input_text
								  },
								{
									'type': 'input_image',
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

			Methods that returns a list of small_model names

		'''
		return [ 'dall-e-3', 'gpt-4-0613',
		         'gpt-4-0314', 'gpt-4o-mini',
		         'gpt-4o-mini-2024-07-18' ]
	
	
	def get_format_options( self ) -> list[ str ]:
		'''

			Method that returns a  list of format options

		'''
		return [ '.png', '.mpeg', '.jpeg',
		         '.webp', '.gif' ]
	
	
	def get_detail_options( self ) -> list[ str ]:
		'''

			Method that returns a  list of reasoning effort options

		'''
		return [ 'auto', 'low', 'high' ]
	
	
	def get_size_options( self ) -> list[ str ]:
		'''

			Method that returns a  list of sizes

		'''
		return [ '1024x1024', '1024x1792',
		         '1792x1024' ]
	
	
	def __dir__( self ) -> list[ str ]:
		'''

			Methods that returns a get_list of member names
			Returns: get_list[ str ]

		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'input_text', 'image_url', 'path',
		         'api_key', 'client', 'small_model', 'text', 'generate',
		         'get_detail_options', 'get_format_options', 'get_size_options' ]


class Image( AI ):
	"""

		Purpose
		___________
		Class used for generating images OpenAI's
		Images API and dall-e-2


		Parameters
		------------
		num: int=1
		temp: float=0.8
		top: float=0.9
		freq: float=0.0
		pres: float=0.0
		max: int=2048
		store: bool=True
		stream: bool=True

		Attributes
		-----------
		self.api_key, self.client, self.small_model,  self.embedding,
		self.response, self.num, self.temperature, self.top_percent,
		self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
		self.store, self.stream, self.modalities, self.stops, self.content,
		self.prompt, self.response, self.completion, self.file, self.path,
		self.text, self.messages, self.image_url, self.response_format,
		self.tools, self.vector_store_ids, self.input_text, self.path, self.image_url

		Methods
		------------
		get_model_options( self ) -> str
		generate( self, text: str ) -> str
		analyze( self, text: str, path: str ) -> str
		get_detail_options( self ) -> list[ str ]
		get_format_options( self ) -> list[ str ]
		get_size_options( self ) -> list[ str ]

	"""
	
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9, freq: float=0.0,
	              pres: float=0.0, max: int=2048, store: bool=False, stream: bool=False ):
		super( ).__init__( )
		self.api_key = Header( ).api_key
		self.client = OpenAI( )
		self.client.api_key = Header( ).api_key
		self.quality = 'standard'
		self.detail = 'auto'
		self.model = 'dall-e-2'
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
	
	
	def get_model_options( self ) -> list[ str ]:
		'''

			Methods that returns a list of small_model names

		'''
		return [ 'dall-e-2', 'gpt-4-0613',
		         'gpt-4-0314', 'gpt-4o-mini',
		         'gpt-4o-mini-2024-07-18' ]
	
	
	def generate( self, input: str ) -> str:
		"""

			Purpose
			_______
			Generates an image given a string text


			Parameters
			----------
			input: str


			Returns
			-------
			Image object

		"""
		try:
			if input is None:
				raise Exception( 'The "input" argument is required.' )
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
			exception.method = 'generate( self, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, input: str, path: str ) -> str:
		'''

			Method providing image analysis functionality given a prompt and filepath

		'''
		try:
			if input is None:
				raise Exception( 'The argument "input" cannot be None' )
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
								{ 'type': 'input_text',
								  'text': self.input_text
								  },
								{
									'type': 'input_image',
									'image_url': self.file_path
								},
							],
					}
					]
				
				self.response = self.client.responses.create( model='gpt-4o-mini',
					input=self.input )
				
				return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Image'
			exception.method = 'analyze( self, text: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def edit( self, input: str, path: str, size: str = '1024X1024' ) -> str:
		"""

			Purpose
			_______
			Method that analyzeses an image given a string prompt,



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
				raise Exception( 'The argument "input" cannot be None' )
			elif path is None:
				raise Exception( 'The argument "path" cannot be None' )
			else:
				self.input_text = input
				self.file_path = path
				self.response = self.client.images.edit( model=self.model,
					image=open( self.file_path, "rb" ), prompt=self.input_text, n=self.number,
					size=self.size )
				
				return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'Boo'
			exception.cause = 'Image'
			exception.method = 'analyze( self, text: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ) -> list[ str ]:
		'''

			Methods that returns a get_list of member names
			Returns: get_list[ str ]

		'''
		return [ 'num', 'temperature', 'top_percent', 'frequency_penalty',
		         'presence_penalty', 'max_completion_tokens',
		         'store', 'stream', 'modalities', 'stops',
		         'api_key', 'client', 'small_model', 'text', 'analyze',
		         'input_text', 'image_url', 'path', 'edit',
		         'generate', 'quality', 'detail', 'small_model', 'get_model_options',
		         'get_detail_options', 'get_format_options', 'get_size_options' ]
	
	
	def get_size_options( self ) -> list[ str ]:
		'''

			Method that returns a  list of small_model options

		'''
		return [ '256x256', '512x512', '1024x1024' ]
	
	
	def get_format_options( self ) -> list[ str ]:
		'''

			Method that returns a  list of format options

		'''
		return [ '.png', '.mpeg', '.jpeg',
		         '.webp', '.gif' ]
	
	
	def get_detail_options( self ) -> list[ str ]:
		'''

			Method that returns a  list of reasoning effort options

		'''
		return [ 'auto', 'low', 'high' ]

