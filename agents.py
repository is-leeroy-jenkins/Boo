'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
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
    name.py
  </summary>
  ******************************************************************************************
'''
import os
from pathlib import Path
from typing import Any, List, Optional, Dict
import tiktoken
from openai import OpenAI
from models import Prompt, Reasoning, Text, Format
from .boogr import ErrorDialog, Error


def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Agent(  ):
	'''
	
		Purpose:
		--------
		Base class for all agent prompts/requests/responses.
	
	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Reasoning ]
	text: Optional[ str ]
	format: Optional[ Format ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	store: Optional[ bool ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	version: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	
	def __init__( self ):
		''''
		
			Purpose:
			--------
			Constructor
		
		'''
		self.client = None
		self.client.api_key = None
		self.question = None
		self.max_output_tokens = 10000
		self.store = True
		self.temperature = 0.8
		self.top_p = 0.9

class ApportionmentAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68a34b1eb99481969acf77a71b51ff25018476307b10d0b5'
		self.version = '15'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.vector_store_ids = [ 'vs_68a34aaff93481918c3b3fef8c4e8fea' ]
		self.file_ids = [ 'file-XfTDeZNv7M1toGMsZcnP24',
		                  'file-8wQZAAZpdHAjVrUdE45TiL',
		                  'file-N5QJtZHnU6vFdHSszwvAZn',
		                  'file-AukoekscMxBsxfgyoXLb5z',
		                  'file-7oRCvxc3W4VNaXhTQpsNFq',
		                  'file-BKUENFQD67naMN3kx6PrHe' ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			container = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': container }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ApportionmentAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DataAnalyst( Agent ):
	'''
	
		
		Purpose:
		--------
		
	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	include: Optional[ List ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68913db1bddc8194931a6c743d6fe2cd03a4dc1797022fcc'
		self.version = '8'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',
		                 'web_search_call.action.sources' ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str  ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			self.client = OpenAI( )
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			_prompt = { 'id': self.id, 'version': self.version,
			            'variables': { 'question': self.question } }
			_response = self.client.responses.create( model=self.model, prompt=_prompt,
				temperature=self.temperature, store=self.store, tool_choice=self.tool_choice, include=self.include )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DataAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PythonAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68a0fb2b65408194a68164a99b0e104a06fddb113af66a94'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ 'vs_6900bd53b400819182cca77ee4fbc143' ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			self.tools.append( search_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning, tool_choice=self.tool_choice )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PythonAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class AppropriationsAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''
		
			Purpose:
			-------
			Contructor for class objects
		
		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68c5b8dd376c8190a2090cb28cefa2b000113be4688382f5'
		self.version = 5
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' },
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources', 'reasoning.encrypted_content' ]
		self.input = [ ]
		self.vector_store_ids = ['vs_712r5W5833G6aLxIYIbuvVcK' ]
		self.file_ids = [ 'file-B4bKRt3Sfg1opRcNL1DRdk',
          'file-21MLeKkao1x3J4u19sYofq',
          'file-SEPUd6zDZ9Kku19pFdguxR',
          'file-Dmd8C3aFALXK7zgify3YKm',
          'file-RvPTUjEyXfN77c9qbh5TBg' ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			container = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': container }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning  )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'AppropriationsAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ScheduleXAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68c58f4e6c0c8190907ebd7e5dd85fd8028ee0257b6020e0'
		self.version = 3
		self.format = 'text'
		self.reasoning = { }
		self.include =[ 'code_interpreter_call.outputs',
		                'reasoning.encrypted_content',
		                'web_search_call.action.sources' ]

		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include  )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ScheduleXAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class BudgetGandolf( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68bac6f657f08194b230e580a82e15e50006cdfe61dc331d'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include =[ 'code_interpreter_call.outputs',
		                'reasoning.encrypted_content',
		                'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ 'file-XfTDeZNv7M1toGMsZcnP24',
          'file-8wQZAAZpdHAjVrUdE45TiL',
          'file-N5QJtZHnU6vFdHSszwvAZn',
          'file-AukoekscMxBsxfgyoXLb5z',
          'file-7oRCvxc3W4VNaXhTQpsNFq',
          'file-BKUENFQD67naMN3kx6PrHe', ]
		self.vector_store_ids = [ 'vs_68a34aaff93481918c3b3fef8c4e8fea', ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			meta = { 'id': self.id, 'version': self.version, 'variables': { 'question': self.question } }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				temperature=self.temperature, top_p=self.top_p, tool_choice=self.tool_choice,
				reasoning=self.reasoning  )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'BudgetGandolf'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class OutlookAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ Dict[ str, Any ] ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6894fe07f204819685a6e340004618840f802573eeac1f4a'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			meta = { 'id': self.id, 'version': self.version, 'variables': { 'question': self.question } }
			_response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				temperature=self.temperature, tools=self.tools, )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'OutlookAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ProcurementAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6894de0a7c6c8196a67581f1a40e83ed031e560f0d172c13'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'web_search_call.action.sources' ]
		self.input = [ ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ 'vs_712r5W5833G6aLxIYIbuvVcK' ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			self.tools.append( search_tool )
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning, tool_choice=self.tool_choice )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ProcurementAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class WhatIfAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6894ddcdff6c819088d5e1cbc8f612c30a8ec3da3496500d'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			meta = { 'id': self.id, 'version': self.version, 'variables': { 'question': self.question } }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning  )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'WhatIfAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class InnovationAnalyst( Agent ):
	'''


		Purpose:
		--------
		
		
		Attributes:
		__________
		
		
		
		Methods:
		----------
		

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6894dd3e952c8194a667670a5c6af01901c8a63112266fb1'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			meta = { 'id': self.id, 'version': self.version, 'variables': { 'question': self.question } }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include  )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'InnovationAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class StatisticsAnalyst( Agent ):
	'''


		Purpose:
		--------
		
		
		Attributes:
		__________
		
		
		
		Methods:
		----------
		

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6894dc961ce881958a585b1d883e60c90133afd64b4ec8a0'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { }
		self.include =[ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include, )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'StatisticsAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PbiExpert( Agent ):
	'''


		Purpose:
		--------
		
		
		Attributes:
		__________
		
		
		
		Methods:
		----------
		

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6894dc2073ec8197b2821fdec0cec32909b600c3c67452d6'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning)
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PbiExpert'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ExcelNinja( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	example: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6894dbd8e46081958a101d6829a2290f0456a555875b6de3'
		self.version = 4
		self.format = 'text'
		self.reasoning = {  }
		self.include =[ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, example: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'example', example )
			self.question = question
			self.example = example
			variable = { 'question': self.question, 'xlsx': self.example }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include  )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ExcelNinja'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ResearchAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'o4-mini-2025-04-16'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68935e63580c8193af06187bae8f9ede01e5f4fd3773b2a6'
		self.version = 4
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ResearchAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class BrainStormer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68933cb36fa0819693121e3b029cf41302980715c4c8625a'
		self.version = 3
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			response = self.client.responses.create( model=self.model, reasoning=self.reasoning,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'BrainStormer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PbiAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689265a62c08819481fda29f423e6625020dd21903e967e0'
		self.version = '5'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str, document: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'document', document )
			self.question = question
			self.document = document
			variable = { 'question': self.question, 'document': self.document }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PbiAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class AutomationAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689339ed50d881939f5fcc265cda026d0b4df3e15cc51bc1'
		self.version = '4'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store,
				include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'AutomationAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class AgendaMaker( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	person: Optional[ str ]
	duration: Optional[ str ]
	date: Optional[ str ]
	attendees: Optional[ List[ str ] ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689336cd3ae88197810ce513dd1e12b70a89ec7bba3af876'
		self.version = '3'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str, duration: str, date: str, attendees: List[ str ] ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			self.duration = duration
			self.date = date
			self.attendees = attendees
			variables = { 'question': self.question,
			             'duration': self.duration,
			             'date': self.date,
			             'attendees': self.attendees }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'AgendaMaker'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ExcelAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	document: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68926c31f2a88190b92866147ef190880abbd30cc10783c4'
		self.version = '5'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str, document: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'document', document )
			self.question = question
			self.document = document
			variable = { 'question': self.question, 'document': self.document }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ExcelAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class FinancialAdvisor( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6892586bcc8c8194bedae3a4b31c0e81058a8b8f3319ffec'
		self.version = '9'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'FinancialAdvisor'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class SpeechWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689254423e908193888fb93a093c71d3053ccef2d2a59be2'
		self.version = '4'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = {'question': self.question }
			meta = {'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'SpeechWriter'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DashboardAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689265a62c08819481fda29f423e6625020dd21903e967e0'
		self.version = '4'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store,
				include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DashboardAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class WealthAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6892535fa7588197a56a6bca44b9b8970fcf2aa7f5f18b30'
		self.version = '3'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'WealthAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class RandomWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6892535fa7588197a56a6bca44b9b8970fcf2aa7f5f18b30'
		self.version = '4'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'RandomWriter'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ExploratoryDataAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	variables: Optional[ List[ str ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	document: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6892054450ac81938b386357144a590305d63be465dc6622'
		self.version = '7'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str, document: str ) -> str | None:
		'''

			Purpose:
			-------
			Method for sending a request to an agent.

			Parameters:
			-----------
			question: str
			A string containing the user message for the request payload.

			Returns:
			---------
			A string containing the response output content

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			self.document = document
			variable = { 'question': self.question, 'document': self.document }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ExploratoryDataAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ComplexProblemAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68914d78489c8190a8721685937b2a530604c9bb3d2ea367'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ComplexProblemAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class EmailAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	document: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689139adf5448190b8307b55ad0384cb01beed075060eede'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, document: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'documnet', document )
			self.question = question
			self.document = document
			variable = { 'question': self.question, 'document': self.document }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ResearchEvaluator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	summary: Optional[ str ]
	article: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689139230710819095711ad1b3f59e9301017a586373075f'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, summary: str, article: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'summary', summary )
			throw_if( 'article', article)
			self.question = question
			self.summary = summary
			self.article = article
			variable = { 'question': self.question, 'summary': self.summary, 'article': self.article }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'EssayWriter'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ExecutiveAssistant( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6891385c2a3c8195babb7ab819fd0dbb0b89cf339e6c6291'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning  )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ExecutiveAssistant'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class EvaluationExpert( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	article: Optional[ str ]
	summary: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = '3'
		self.version = 'pmpt_6891385c2a3c8195babb7ab819fd0dbb0b89cf339e6c6291'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, article: str, summary: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'article', article )
			throw_if( 'summary', summary )
			self.question = question
			self.article = article
			self.summary = summary
			variable = { 'question': self.question, 'article': self.article, 'summary': self.summary }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning  )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'EvaluationExpert'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ExpertProgrammer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68913810fb3081909e90afd11b7d54ba01c2eeac10a06125'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning  )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ExpertProgrammer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class FeatureExtractor( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6891373a1f6c81908484fb1d75ccf61c0648e00599529f7f'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'FeatureExtractor'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class FinancialAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6891369156fc81958b24c0ce84c7deda01be94bfa9bf7a2e'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'FinancialAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class FinancialPlanner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6891361e4418819483480f77083823d108cc20456900f165'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'FinancialPlanner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class FormBuilder( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6891359401008195956bf1855321e27508eea3cf6957065f'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'FormBuilder'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class LegalAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	parties: Optional[ str ]
	purpose: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6891335308b081909903f694ab6fc7fd04de43be735450f4'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, parties: str, purpose: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'parties', parties )
			throw_if( 'purpose', purpose )
			self.question = question
			self.purpose = purpose
			self.parties = parties
			variable = { 'question': self.question, 'parties': self.parties, 'purpose': self.purpose }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'LegalAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PromptEngineer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	number: Optional[ str ]
	name: Optional[ str ]
	answer: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689126c5c5b081908ad6ee27b78377d400fae2713e5ad3d1'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, number:str, name: str, answer: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			self.number = number
			self.name = name
			self.answer = answer
			variables = { 'question': self.question, 'number': self.number,
			             'name': self.name, 'answer': self.answer }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PromptEngineer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ProjectArchitect( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68912680a44881949684ff8775796bc209168e3555d8cd38'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ProjectArchitect'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ProjectPlanner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_689125dadf1081979650dd0c4b2ee1b801700c40557fa1b2'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ProjectPlanner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TrainingWheels( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	articles: Optional[ str ]
	transcript: Optional[ str ]
	message: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68911b7dee3881908524ee0bae8564e30974790f13ab110f'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, articles: str, transcript: str, message: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'articles', articles )
			throw_if( 'transcript', trancsript)
			throw_if( 'message', message )
			self.articles = articles
			self.transcript = transcript
			self.message = message
			variables = { 'message': self.message, 'articles': self.articles, 'transcript': self.transcript }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'TrainingWheels'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class RedTeamAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6891239b51188195ad7555c872b88359016fcad277ff9cb8'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'RedTeamAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class SentimentAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	number: Optional[ str ]
	sources: Optional[ str ]
	purpose: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68911ff2807081909da7edb2505324d10e8cd40e99552e6e'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, purpose: str, sources: str, number: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'sources', sources )
			throw_if( 'number', number )
			self.question = question
			variable = { 'question': self.question, 'sources': self.sources, 'number': self.number }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'SentimentAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TrainingPlanner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68911be7a5948196b42bbc5a33bcd2c8061c4a703bbd4aaa'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, role: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'TrainingPlanner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class WebSearchOptimizer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	topic: Optional[ str ]
	keyword: Optional[ str ]
	wordcount: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68911aecfae48194aa3e9f9f09b51a1105c57f5c67dc6eb9'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, topic: str, keyword: str, wordcount: int ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'topic', topic )
			throw_if( 'keyword', keyword )
			throw_if( 'wordcount', wordcount )
			self.question = question
			self.topic = topic
			self.keyword = keyword
			self.wordcount = wordcount
			variables = { 'question': self.question, 'topic': self.topic,
			             'keyword': self.keyword, 'wordcount': self.wordcount }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'WebSearchOptimizer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class BudgetAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68668be09f2c8193b0c16b0d3a0e6a560c08f132c9c0f5e7'
		self.version = '20'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs',
		                 'reasoning.encrypted_content',
		                 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ 'file-EQSpQdb6a4tr2koPSLY9tX',
          'file-R514TVxo99RDas5yaf5KYa',
          'file-Fqd25bRD5EczveCvomeLqi',
          'file-SPq96KKrk7aX1E2igGNGMm' ]
		self.vector_store_ids = [ 'vs_712r5W5833G6aLxIYIbuvVcK' ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			container = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': container }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class YoutubeSummarizer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	transcript: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68668411a80c81948e1eb2a36e1028f208d2942c73667285'
		self.version = '9'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, transcript: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'transcript', transcript )
			self.question = question
			self.transcript = transcript
			variable = { 'question': self.question, 'transcripts': self.transcript }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'YoutubeSummarizer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class YoutubeScribe( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	transcript: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686683daaa5481938be88f4238564e03043a3c94c3739613'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, transcript: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'transcript', transcript )
			self.question = question
			self.transcript = transcript
			variables = { 'question': self.question, 'transcript': self.transcript }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'YoutubeScribe'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class WritingEditor( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6866839610448196b582a0361d5d0df30659d47624fc5b3d'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			container = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': container }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'WritingEditor'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class WebDesigner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686682f561d88190bf88ded7ccf34f4a0d3a290a3fd122c9'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'WebDesigner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class Guardrails( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	articles: Optional[ str ]
	transcript: Optional[ str ]
	version: Optional[ str ]
	
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686682b75214819694c85baf4f397a8a09cbbe9c7769a5e6'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: st, articles: str, transcript: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'articles', articles )
			throw_if( 'transcript', transcript )
			self.question = question
			self.articles = articles
			self.transcript = transcript
			variable = { 'question': self.question, 'articles': self.articles, 'transcript': self.transcript }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'Guardrails'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TrainingProgramDesigner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6866820482688195a57fcc2328d93d2f063953206e5f1928'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TrainingContentDesigner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6866820482688195a57fcc2328d93d2f063953206e5f1928'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'TrainingContentDesigner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TopicResearcher( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'o4-mini-2025-04-16'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68668176d5188194aefe11dfd4583b9b02b4196e31725700'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'TopicResearcher'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TechSupportAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6866812c56b48197876d9f4e4613e1d4051c557c6cb9b2d2'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'TechSupportAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TaskPlanner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68668095c8008193942013ac51274c3b01cce05b3c75d7a2'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			container = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': container }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				tool_choice=self.tool_choice, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'TaskPlanner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class TeachingAssistant( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686680e5e1ac8197aaf54867b7033b200bcf5dc3663ac8fb'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'TeachingAssistant'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class SustainabilityPlanner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68668061eb2081978c9cb0ffcfdcdb340993936662ab0928'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'code_interpreter_call.outputs',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			files = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': files }
			self.tools.append( search_tool )
			self.tools.append( code_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			container = { 'type': 'auto', 'file_ids': self.file_ids }
			code_tool = { 'type': 'code_interpreter', 'container': container }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'SustainabilityPlanner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class StructuredProblemSolver( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667e1aace08190996d29d77860db1f058ab8059c4e500b'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'StructuredProblemSolver'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class StrategicThinker( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	context: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667dc870288197b541efafb8555eb8016ceae87d60aa2c'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, context: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'context', context )
			self.question = question
			self.context = context
			variable = { 'question': self.question, 'context': self.context }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta,store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'StrategicThinker'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class SqlAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667d7172a08197ba0534dbc33f043a07cf7cbaa4a40d82'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'SqlAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class SearchOptimizer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	keyword: Optional[ str ]
	wordcount: Optional[ int ]
	audience: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667d1f893c81978a8bdbf1c74d3693052dac9e9971b065'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, keyword: str, audience: str, wordcount: int ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'keyword', keyword )
			throw_if( 'audience', audience )
			throw_if( 'wordcount', wordcount )
			self.question = question
			self.keyword = keyword
			self.audience = audience
			self.wordcount = wordcount
			variables = { 'question': self.question, 'keyword': self.keyword,
			             'audience': self.audience, 'wordcount': self.wordcount }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'SearchOptimizer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class SearchOptimizedWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667bafc350819794290cccac7b68900b62cd807a9f94a0'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',
		                 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = {
					'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'SearchOptimizedWriter'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class RootCauseAnalyzer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667b61e3988197ae445df0887e73840f0980a3cdd03e58'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'RootCauseAnalyzer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class RevenueProjector( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667a943d848194ae90520e54e5d56d075ade6f0cdda41d'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'RevenueProjector'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ResumeWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667a4978b481949947dd25f72af4ee03da24bb3fa42cfe'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ResumeWriter'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ResumeBuilder( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	industry: Optional[ str ]
	experience: Optional[ int ]
	title: Optional[ str ]
	resume: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686679fa18a08194a967e16b7dab69d003562ba364f35707'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, resume: str, title: str, industry:str, expeience: int ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'resume', resume )
			throw_if( 'title', title )
			throw_if( 'industry', industry )
			throw_if( 'experience', expeience )
			self.question = question
			self.resume = resume
			self.title = title
			self.industry = industry
			self.experience = experience
			variables = { 'question': self.question,
			             'resume': self.resume,
			             'title': self.title,
			             'industry': self.industry,
			             'experience': self.experience }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ResumeBuilder'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ResultsCreator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6866790f642881949762b6f280426a500e02b90b2f12b679'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ResultsCreator'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class RequirementsGenerator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	product: Optional[ str ]
	stage: Optional[ str ]
	team: Optional[ str ]
	challenges: Optional[ str ]
	timeline: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686677c322b081949427d4700b4f624101da857c0c08b6e8'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, product: str, team: str, challenges: str, timeline: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'product', product )
			throw_if( 'team', team )
			throw_if( 'challenges', challenges )
			throw_if( 'timeline', timeline )
			self.question = question
			self.prompt = product
			self.team = team
			self.challenges = challenges
			self.timeline = timeline
			variables = { 'question': self.question,
			             'product': self.product,
			             'team': self.team,
			             'challenges': self.challenges,
			             'timeline': self.timeline }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'RequirementsGenerator'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ResearchExpert( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686678c70e64819085ff39d93154d389011cd5cddb7d5ee8'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = {'question': self.question }
			meta = {'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ResearchExpert'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ReasoningAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6866775201508197ae4f82159524c2f80008112f498e8a6d'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ReasoningAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )


class ProofReader( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68667707cc5c819386bd8fc446cd3b5201b4de6b06fefbb0'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ProofReader'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class QuickProblemSolver( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865c956a5308194974efb2e16195eb00dd298e96e32be36'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'QuickProblemSolver'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PowerPointAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865c90d03b081949a0457d8f3901ac109daff424fc5dd7c'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = {'question': self.question }
			meta = {'id': self.id,'version': self.version,'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PowerPointAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PortraitGenerator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865c88a40b88196ad8b14799875d6460fd474cbe64a347c'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PortraitGenerator'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PersonalAssistant( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865c8343cdc81948de056ff1b6c35c00dc55e6d399d18b6'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = {'question': self.question }
			meta = {'id': self.id,'version': self.version,'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PersonalAssistant'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PdfParser( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	title: Optional[ str ]
	document: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865c715485c8194a17a9fb6fb3060a6080334be02a74646'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, title: str, document: str  ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'title', title )
			throw_if( 'document', document )
			self.question = question
			self.title = title
			self.document = document
			variables = { 'question': self.question, 'title': self.title, 'document': self.document }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PdfParser'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class NicheResearcher( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	topic: Optional[ str ]
	audience: Optional[ str ]
	frequency: Optional[ str ]
	version: Optional[ str ]
	
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865c5fe64108194b740e05393fecfcc05f4fc2eca4618d4'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, topic: str, audience: str, frequency: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'audience', audience )
			throw_if( 'topic', topic )
			throw_if( 'frequency', freuency )
			self.question = question
			self.audience = audience
			self.topic = topic
			self.frequency = frequency
			variables = { 'question': self.question,
			             'audience': self.audience,
			             'topic': self.topic,
			             'frequency': self.frequency }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'NicheResearcher'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class NewsLetterWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	topic: Optional[ str ]
	audience: Optional[ str ]
	frequency: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865be7996808195b57be1a53f895af50b9013256a736343'
		self.version = '10'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, topic: str, audience: str, frequency: str  ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'topic', topic )
			throw_if( 'audience', audience )
			throw_if( 'frequency', frequency)
			self.question = question
			self.topic = topic
			self.audience = audience
			self.frequency = frequency
			variables = { 'question': self.question,
			             'topic': self.topic,
			             'audience': self.audience,
			             'frequency': self.frequency }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class MultiProfessor( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865be0bc6548194893cacab4f0b495607f7bb2c4087a50a'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'MultiProfessor'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class MeetingSummarizer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865bd9745c481909aca8e4caa45bc0e03e2fbe2dbd48450'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'MeetingSummarizer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class MeetingOptimizer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	meeting: Optional[ str ]
	participants: Optional[ str ]
	goals: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865bd5677308194ad9ac837991d7a150a514e33f1a21e05'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, meeting: str, participants: str, goals: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'meeting', meeting )
			throw_if( 'participants', participants )
			throw_if( 'goals', goals )
			self.question = question
			self.meeting = meeting
			self.participants = participants
			self.goals = goals
			variables = { 'question': self.question,
			             'meeting': self.meeting,
			             'participants': self.participants,
			             'goals': self.goals }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'MeetingOptimizer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class MathyMagician( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865bc0ac92881959abe4b990a5b588a07d9b74212eacd6c'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'MathyMagician'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class MarketResearcher( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	industry: Optional[ str ]
	company: Optional[ str ]
	depth: Optional[ str ]
	region: Optional[ str ]
	timeframe: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865bb7c469881949a32ebdea794bcf70b884fff46bae3d2'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, industry: str, company: str,
			depth: str, region: str, timeframe: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'industry', industry )
			throw_if( 'company', company )
			throw_if( 'depth', depth )
			throw_if( 'region', region )
			throw_if( 'timeframe', timeframe )
			self.question = question
			self.industry = industry
			self.company = company
			self.depth = depth
			self.region = region
			self.timeframe = timeframe
			variables = { 'question': self.question,
			             'industry': self.industry,
			             'company': self.company,
			             'depth': self.depth,
			             'region': self.region,
			             'timeframe': self.timeframe }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'MarketResearcher'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class MarketPlanner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	product: Optional[ str ]
	version: Optional[ str ]
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865ba3d30d48193886c3d7400a7bce60a55bd77959a79f2'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, product: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'product', product )
			self.question = question
			self.product = product
			variables = { 'question': self.question, 'product': self.product }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'MarketingPlanner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class MarketForecaster( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	industry: Optional[ str ]
	problem: Optional[ str ]
	trend: Optional[ str ]
	region: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865b9ae72048194ba4bd37883c4ee5a00e3d0b846f4d15d'
		self.version = '5'
		self.format = 'text'
		self.reasoning = {'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content','web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, industry: str, trend: str,
			region: str, problem: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'industry', industry )
			throw_if( 'trend', trend )
			throw_if( 'problem', problem )
			self.question = question
			self.industry = industry
			self.trend = trend
			self.problem = problem
			variables = {'question': self.question,
			            'industry': self.industry,
			            'trend': self.trend,
			            'problem': self.problem }
			meta = {'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'MarketForecaster'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ManagementConsultant( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865b8929fd88196b1bd772e7037aef206a2512c498edd67'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ManagementConsultant'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class KeywordGenerator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865b7671178819691e5ee1b092723ff05d3ac222b9985f7'
		self.version = '9'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class JackOfAllTrades( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865ae9765388190b42964801eb3e1500f42db71260dedcd'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'JackOfAllTrades'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class InterviewCoach( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	role: Optional[ str ]
	company: Optional[ str ]
	skills: Optional[ str ]
	experience: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865ae25524c81968e171bd843b891bc0246e299bc057886'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, role: str, company: str, skills: str, experience: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'role', role )
			throw_if( 'company', company )
			throw_if( 'skills', skills )
			throw_if( 'experience', experience )
			self.question = question
			variables = { 'question': self.question,
			             'role': self.role,
			             'company': self.company,
			             'skills': self.skills,
			             'experience': self.experience }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class InvestmentAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865ad450b00819592c7783a8b8dd50604d79a3339872985'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'InvestmentAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class EducationalWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a905e68081949da4f6e0abd7a43008822934901b9761'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class HowToBuilder( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	topic: Optional[ str ]
	skill: Optional[ str ]
	format: Optional[ str ]
	version: Optional[ str ]
	
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865ac6e61c48195ad8a82d66c6bd95a0aac37f459d9a377'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, topic: str, skill: str, format: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'topic', topic )
			throw_if( 'skill', skill )
			throw_if( 'format', format )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'HowToBuilder'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class EssayWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	topic: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865aaa288f481908547597ea36c21cf0b3e7db8b571e3d7'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, topic: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'topic', topic )
			self.question = question
			variables = { 'question': self.question, 'topic': self.topic }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'EssayWriter'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class EmailAssistant( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865aa043f788197a6900a111c1d87750d51002ac8927974'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variables = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DocumentSummarizer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a858f9508195a936864371bc52740c999fbbc53593e1'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DocumentSummarizer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DocumentInterrogator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	document: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a7d7da708196b3f1643fe34af06e088c920d959706a7'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, document: str  ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'qdocument', document )
			self.question = question
			self.document = document
			variable = { 'question': self.question, 'document': self.document }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DocumentInterrogator'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DependencyIndentifier( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a7662f6481909f7856938c7e93b0017264a41177c6aa'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DependencyIdentifier'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DecisionMaker( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a70af9208197ae3c71f8b67b6e3d0485d8b7da9ba122'
		self.version = 4
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DecisionMaker'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DecisionMaker( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	data: Optional[ str ]
	version: Optional[ str ]
	
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a683d9248190ab81fcc2323d1b270f09afb1ae2c0f08'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, data: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'stores', data )
			self.question = question
			self.data = data
			variables = { 'question': self.question, 'stores': self.data }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DecisionMaker'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DataScientist( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a62d44c48193858d1b35ddb577720c820bbf2e79ced7'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DataScientist'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DatasetAnalyzer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a5dea2688197ba54961d40cf1b9a00895973bc015ddb'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DatasetAnalyzer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DataPlumber( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a4bcafb08197b54e6f160c0f7e98066f753cc10b0128'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DataPlumber'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DataFarmer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a4bcafb08197b54e6f160c0f7e98066f753cc10b0128'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DataCleaner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a3d7067c8193af52faec4761470101e2fa0480275266'
		self.version = ''
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DataCleaner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DataCleaner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a3d7067c8193af52faec4761470101e2fa0480275266'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DataCleaner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DataBro( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a382ab3881968f43106958e2460005c97cb2abbabbc7'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DataBro'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class DatabaseSpecialist( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a327e2588194a524170d0a198b4100a43cda801d84e2'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'DatabaseSpecialist'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class CriticalThinker( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865a076c4f081978ea9992c408f39f60ecd0aee4bf0e7fa'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'CriticalThinker'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class CriticalReasoningAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68659e443ae8819095857201a4f035210a9a9128f0605de1'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'CriticalReasoningAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class CourseCreator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	subject: Optional[ str ]
	audience: Optional[ str ]
	duration: Optional[ str ]
	frequency: Optional[ str ]
	time: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68659d7a84588193a8d901eae0b4ad250a771174e3b18ccc'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, subject: str, audience: str,
			duration: str, frequency: str, time: str  ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'duration', duration )
			throw_if( 'audience', audience)
			throw_if( 'duration', duration )
			throw_if( 'frequency', frequency )
			throw_if( 'time', time )
			self.question = question
			self.duration = duration
			self.audience = audience
			self.frequency = frequency
			self.time = time
			variables = { 'question': self.question,
			             'duration': self.duration,
			             'audience': self.audience,
			             'frequency': self.frequency,
			             'time': self.time }
			meta = { 'id': self.id, 'version': self.version, 'variables': variables }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'CourseCreator'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class CompanyResearcher( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655ca202688190a0a3a4ae57771574039e126fd5c37ecc'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'CompanyResearcher'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class CognitiveProfiler( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655c5795408195988f70d11ba0e155020c59ca546a6755'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'CognitiveProfiler'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class CodeReviewer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655c0318fc8196adc0d8718775c2e40f4819ec31b29e70'
		self.version = '7'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'CodeReviewer'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ChecklistCreator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655ba61afc81949ab8bac0ec6615320614ec9128dec201'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ChecklistCreator'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ChainOfDensity( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	document: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655b5048888190916aae3e0401b86609d234efe0126fa8'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, document: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'document', document )
			self.question = question
			self.document = documnet
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ChainOfDensity'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class BusinessResearcher( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	buisness: Optional[ str ]
	industry: Optional[ str ]
	product: Optional[ str ]
	timeframe: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655ad6ea6881909b7d348866af27910beeb7966664cf8f'
		self.version = '8'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, business: str, industry: str, product: str, timeframe: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'business', business )
			throw_if( 'industry', industry )
			throw_if( 'product', product )
			throw_if( 'timeframe', timeframe )
			self.question = question
			self.buisness = business
			self.industry = industry
			self.product = product
			self.timeframe = timeframe
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'BusinessResearcher'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class BusinessPlanner( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655a8e8e9881908f858f54c361bb760e2c93d271d3125a'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'BusinessPlanner'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class BusinessAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	ticker: Optional[ str ]
	company: Optional[ str ]
	sector: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655a41a1748196a7c864aaee0af331041858ced4469344'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str, ticker: str, company: str, sector: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			throw_if( 'ticker', ticker )
			throw_if( 'company', company )
			throw_if( 'sector', sector )
			self.question = question
			self.ticker = ticker
			self.company = company
			self.sector = sector
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'BusinessAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class PowerQueryAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_69038d1954a08190bc876d003b771556002a558c5cc0e5ca'
		self.version = '2'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content',  'web_search_call.action.sources' ]
		self.input = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning )
			output = _response.output_text
			return output
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'PowerQueryAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class BookSummarizer( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686559d29d2c81969454aaa5bf1518820c54175869340641'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs',
		                 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning,
			tool_choice = self.tool_choice  )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class AuthorEmulator( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865598842288195b61fd663cfbcf0930025515c2331ed97'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning,
			tool_choice=self.tool_choice )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'AuthorEmulator'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class AsciiArtist( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865590d73d48194acd1f75d7c8961ce0fed37fa3ea81306'
		self.version = '4'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning,
				tool_choice=self.tool_choice )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'AsciiArtist'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class ArtsyFartsy( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_686558c7dda08194a684d49d057a62ce0157d5ff5bfda345'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning,
				tool_choice=self.tool_choice )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'ArtsyFartsy'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class AdaptiveAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_6865586fc73c8190aebddd1d4f7b57680ba7c4db40cd45c8'
		self.version = '6'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning,
				tool_choice=self.tool_choice )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'AdaptiveAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class AcademicWriter( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ str ]
	format: Optional[ str ]
	max_output_tokens: Optional[ int ]
	input: Optional[ List ]
	store: Optional[ bool ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	include: Optional[ List ]
	question: Optional[ str ]
	variables: Optional[ List[ str ] ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self ):
		'''

			Purpose:
			-------
			Contructor for class objects

		'''
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-5-nano-2025-08-07'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68655623b2e0819099bc136d3c8fbf5b04420f5632d48e2d'
		self.version = '8'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.input = [ ]
		self.vector_store_ids = [ ]
		self.file_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			-------

			Parameters:
			-----------

			Returns:
			---------

		'''
		try:
			throw_if( 'question', question )
			self.question = question
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			_response = self.client.responses.create( model=self.model, prompt=meta, store=self.store,
				max_output_tokens=self.max_output_tokens, include=self.include, reasoning=self.reasoning,
				tool_choice=self.tool_choice )
			return _response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'AcademicWriter'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )
