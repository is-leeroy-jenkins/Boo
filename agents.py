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
from boogr import ErrorDialog, Error


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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Reasoning ]
	text: Optional[ Text ]
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
	prompt: Optional[ Prompt ]
	text: Optional[ Text ]
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
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-4o-mini'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68a34b1eb99481969acf77a71b51ff25018476307b10d0b5'
		self.version = '12'
		self.format = 'text'
		self.tools = [ ]
		self.reasoning = {}
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.vector_store_ids = [ 'vs_68a34aaff93481918c3b3fef8c4e8fea' ]
		self.file_ids =  [
          'file-XfTDeZNv7M1toGMsZcnP24',
          'file-8wQZAAZpdHAjVrUdE45TiL',
          'file-N5QJtZHnU6vFdHSszwvAZn',
          'file-AukoekscMxBsxfgyoXLb5z',
          'file-7oRCvxc3W4VNaXhTQpsNFq',
          'file-BKUENFQD67naMN3kx6PrHe'
        ]
	
	def ask( self, question: str ):
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
	
	def __init__( self ):
		super( ).__init__( )
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-4o-mini'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68913db1bddc8194931a6c743d6fe2cd03a4dc1797022fcc'
		self.version = '5'
		self.format = 'text'
		self.reasoning = { }
		self.include = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str  ):
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.model = 'gpt-4o-mini'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68a0fb2b65408194a68164a99b0e104a06fddb113af66a94'
		self.version = '2'
		self.format = 'text'
		self.reasoning = { }
		self.include = [ 'web_search_call.action.sources' ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ 'vs_6900bd53b400819182cca77ee4fbc143' ]
	
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
			self.tools.append( search_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include )
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.model = 'gpt-4o-mini'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68c5b8dd376c8190a2090cb28cefa2b000113be4688382f5'
		self.version = '2'
		self.format = 'text'
		self.reasoning = { }
		self.tools = [ ]
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
		self.vector_store_ids = [ 'vs_712r5W5833G6aLxIYIbuvVcK' ]
		self.file_ids = [ 'file-B4bKRt3Sfg1opRcNL1DRdk',
          'file-21MLeKkao1x3J4u19sYofq',
          'file-SEPUd6zDZ9Kku19pFdguxR',
          'file-Dmd8C3aFALXK7zgify3YKm',
          'file-RvPTUjEyXfN77c9qbh5TBg' ]
	
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.model = 'gpt-4.1-mini-2025-04-14'
		self.tool_choice = 'auto'
		self.id = 'pmpt_68c58f4e6c0c8190907ebd7e5dd85fd8028ee0257b6020e0'
		self.version = '2'
		self.format = 'text'
		self.reasoning = { }
		self.include = [ 'code_interpreter_call.outputs', 'web_search_call.action.sources' ]
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
			self.tools.append( { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids } )
			self.tools.append( { 'type': 'code_interpreter',
			                     'container': { 'type': 'auto','file_ids': self.file_ids }  } )
			_prompt = { 'id': self.id, 'version': self.version,
			            'variables': { 'question': self.question } }
			_response = self.client.responses.create( prompt=_prompt, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				temperature=self.temperature, top_p=self.top_p, tool_choice=self.tool_choice,
				model=self.model )
			return _response.output_text
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.id = 'pmpt_68bac6f657f08194b230e580a82e15e50006cdfe61dc331d'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include =[ 'code_interpreter_call.outputs',
		                'reasoning.encrypted_content',
		                'web_search_call.action.sources' ]
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.version = '4'
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
			meta = { 'id': self.id, 'version': self.version, 'variables': { 'question': self.question } }
			_response = self.client.responses.create( model=self.model, prompt=meta,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.id = 'pmpt_6894de0a7c6c8196a67581f1a40e83ed031e560f0d172c13'
		self.version = '3'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'web_search_call.action.sources' ]
		self.tools = [ ]
		self.file_ids = [ ]
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
			self.tools.append( search_tool )
			variable = { 'question': self.question }
			meta = { 'id': self.id, 'version': self.version, 'variables': variable }
			search_tool = { 'type': 'file_search', 'vector_store_ids': self.vector_store_ids }
			response = self.client.responses.create( model=self.model, prompt=meta, tools=self.tools,
				max_output_tokens=self.max_output_tokens, store=self.store, include=self.include,
				reasoning=self.reasoning )
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
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.id = 'pmpt_6894ddcdff6c819088d5e1cbc8f612c30a8ec3da3496500d'
		self.version = '2'
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
			exception.cause = 'WhatIfAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class InnovationAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.id = 'pmpt_6894dd3e952c8194a667670a5c6af01901c8a63112266fb1'
		self.version = '2'
		self.format = 'text'
		self.reasoning = { 'effort': 'medium' }
		self.include = [ 'reasoning.encrypted_content', 'web_search_call.action.sources' ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			--------

			Parameters:
			----------

			Returns:
			--------

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
				reasoning=self.reasoning )
			return response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'agents'
			exception.cause = 'InnovationAnalyst'
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class NewAnalyst( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.model = ''
		self.tool_choice = 'auto'
		self.id = ''
		self.version = ''
		self.format = 'text'
		self.reasoning = { }
		self.include = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			--------

			Parameters:
			----------

			Returns:
			--------

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
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class NewAnalyst2( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.model = ''
		self.tool_choice = 'auto'
		self.id = ''
		self.version = ''
		self.format = 'text'
		self.reasoning = { }
		self.include = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			--------

			Parameters:
			----------

			Returns:
			--------

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
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )

class NewAnalyst3( Agent ):
	'''


		Purpose:
		--------

	'''
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	prompt: Optional[ Prompt ]
	reasoning: Optional[ Dict[ str, str ] ]
	text: Optional[ Text ]
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
		self.model = ''
		self.tool_choice = 'auto'
		self.id = ''
		self.version = ''
		self.format = 'text'
		self.reasoning = { }
		self.include = [ ]
		self.tools = [ ]
		self.file_ids = [ ]
		self.vector_store_ids = [ ]
	
	def ask( self, question: str ) -> str | None:
		'''

			Purpose:
			--------

			Parameters:
			----------

			Returns:
			--------

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
			exception.cause = ''
			exception.method = 'ask( self, question: str ) -> str | None'
			error = ErrorDialog( exception )
			error.show( )
