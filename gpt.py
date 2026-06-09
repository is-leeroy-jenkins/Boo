'''
	******************************************************************************************
	    Assembly:                Boo
	    Filename:                Boo.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2022
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="gpt.py" company="Terry D. Eppler">
	
	           Boo is a df analysis tool integrating various Generative GPT, GptText-Processing, and
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
from __future__ import annotations
import json
import os
from pathlib import Path
import tiktoken
from openai import OpenAI
from typing import Optional, List, Dict, Any
from openai.types.responses import Response
import base64
from openai.types import CreateEmbeddingResponse, VectorStore, FileObject
from boogr import Error, Logger
import config as cfg
import tempfile

def throw_if( name: str, value: object ) -> None:
	"""Throw if.
	
	Purpose:
	    Validates that a required argument contains a usable value before the surrounding workflow
	    continues. This guard centralizes early validation so provider wrappers and UI routines fail
	    with consistent, readable error messages.
	
	Args:
	    name (str): Name value used by the operation.
	    value (object): Value value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode image.
	
	Purpose:
	    Performs the encode_image workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
	    data-processing paths can call it consistently.
	
	Args:
	    image_path (str): Image path value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class GPT:
	"""GPT class.
	
	Purpose:
	    Defines the GPT component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    client (Optional[OpenAI]): Stores client for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    temperature (Optional[float]): Stores temperature for the component runtime state.
	    top_percent (Optional[float]): Stores top percent for the component runtime state.
	    frequency_penalty (Optional[float]): Stores frequency penalty for the component runtime state.
	    presence_penalty (Optional[float]): Stores presence penalty for the component runtime state.
	    max_tokens (Optional[int]): Stores max tokens for the component runtime state.
	    stops (Optional[List[str]]): Stores stops for the component runtime state.
	    store (Optional[bool]): Stores store for the component runtime state.
	    stream (Optional[bool]): Stores stream for the component runtime state.
	    background (Optional[bool]): Stores background for the component runtime state.
	    number (Optional[int]): Stores number for the component runtime state.
	    response_format (Optional[Dict[str, str]]): Stores response format for the component runtime state.
	    context (Optional[List[Dict[str, str]]]): Stores context for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state."""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_tokens: Optional[ int ]
	stops: Optional[ List[ str ] ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	background: Optional[ bool ]
	number: Optional[ int ]
	response_format: Optional[ Dict[ str, str ] ]
	context: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	
	def __init__( self ):
		self.api_key = cfg.OPENAI_API_KEY
		self.model = None
		self.client = None
		self.number = None
		self.stops = [ ]
		self.response_format = { }
		self.number = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.prompt = None
		self.store = None
		self.stream = None
		self.background = None
		self.instructions = None
		self.context = [ ]

class Chat( GPT ):
	"""Chat class.
	
	Purpose:
	    Defines the Chat component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    include (Optional[List[str]]): Stores include for the component runtime state.
	    tool_choice (Optional[str]): Stores tool choice for the component runtime state.
	    previous_id (Optional[str]): Stores previous id for the component runtime state.
	    conversation_id (Optional[str]): Stores conversation id for the component runtime state.
	    parallel_tools (Optional[bool]): Stores parallel tools for the component runtime state.
	    max_tools (Optional[int]): Stores max tools for the component runtime state.
	    input (Optional[List[Dict[str, Any]] | str]): Stores input for the component runtime state.
	    tools (Optional[List[Dict[str, Any]]]): Stores tools for the component runtime state.
	    reasoning (Optional[Dict[str, str]]): Stores reasoning for the component runtime state.
	    image_url (Optional[str]): Stores image url for the component runtime state.
	    image_path (Optional[str]): Stores image path for the component runtime state.
	    file_url (Optional[str]): Stores file url for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    allowed_domains (Optional[List[str]]): Stores allowed domains for the component runtime state.
	    max_search_results (Optional[int]): Stores max search results for the component runtime state.
	    output_text (Optional[str]): Stores output text for the component runtime state.
	    vector_stores (Optional[Dict[str, str]]): Stores vector stores for the component runtime state.
	    files (Optional[Dict[str, str]]): Stores files for the component runtime state.
	    content (Optional[str]): Stores content for the component runtime state.
	    vector_store_ids (Optional[List[str]]): Stores vector store ids for the component runtime state.
	    file_ids (Optional[List[str]]): Stores file ids for the component runtime state.
	    response (Optional[Response]): Stores response for the component runtime state.
	    file (Optional[FileObject]): Stores file for the component runtime state.
	    purpose (Optional[str]): Stores purpose for the component runtime state."""
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	previous_id: Optional[ str ]
	conversation_id: Optional[ str ]
	parallel_tools: Optional[ bool ]
	max_tools: Optional[ int ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	file_url: Optional[ str ]
	file_path: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	max_search_results: Optional[ int ]
	output_text: Optional[ str ]
	vector_stores: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	content: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	response: Optional[ Response ]
	file: Optional[ FileObject ]
	purpose: Optional[ str ]
	
	def __init__( self, model: str = 'gpt-5-nano', prompt: str = None, temperature: float = None,
			top_p: float = None, presense: float = None, presence: float = None, store: bool = None,
			stream: bool = None, stops: List[ str ] = None,
			response_format: Dict[ str, Any ] = None,
			number: int = None, instruct: str = None, context: List[ Dict[ str, str ] ] = None,
			allowed_domains: List[ str ] = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, max_tools: int = None,
			tool_choice: str = None, file_path: str = None, background: bool = None,
			is_parallel: bool = None, max_tokens: int = None, frequency: float = None,
			input: List[ Dict[ str, Any ] ] = None, file_ids: List[ str ] = None,
			previous_id: str = None, conversation_id: str = None,
			reasoning: Dict[ str, str ] | str = None, output_text: str = None,
			max_search_results: int = None, content: str = None,
			vector_store_ids: List[ str ] = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Chat object with its default configuration, runtime state, provider settings,
		    and compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment.
		
		Args:
		    model (str): Model value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    presense (float): Presense value used by the operation.
		    presence (float): Presence value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    response_format (Dict[str, Any]): Response format value used by the operation.
		    number (int): Number value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    context (List[Dict[str, str]]): Context value used by the operation.
		    allowed_domains (List[str]): Allowed domains value used by the operation.
		    include (List[str]): Include value used by the operation.
		    tools (List[Dict[str, Any]]): Tools value used by the operation.
		    max_tools (int): Max tools value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    file_path (str): File path value used by the operation.
		    background (bool): Background value used by the operation.
		    is_parallel (bool): Is parallel value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    input (List[Dict[str, Any]]): Input value used by the operation.
		    file_ids (List[str]): File ids value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		    reasoning (Dict[str, str] | str): Reasoning value used by the operation.
		    output_text (str): Output text value used by the operation.
		    max_search_results (int): Max search results value used by the operation.
		    content (str): Content value used by the operation.
		    vector_store_ids (List[str]): Vector store ids value used by the operation."""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = prompt
		self.number = number
		self.response_format = response_format if response_format is not None else { }
		self.temperature = temperature
		self.top_percent = top_p
		self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
		self.frequency_penalty = frequency
		self.presence_penalty = presence if presence is not None else presense
		self.max_tokens = max_tokens
		self.context = context if context is not None else [ ]
		self.stream = stream
		self.store = store
		self.instructions = instruct
		self.stops = stops if stops is not None else [ ]
		self.background = background
		self.input = input if input is not None else [ ]
		self.include = include if include is not None else [ ]
		self.output_text = output_text
		self.max_tools = max_tools
		self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
		self.file_ids = file_ids if file_ids is not None else [ ]
		self.tools = tools if tools is not None else [ ]
		self.previous_id = previous_id
		self.conversation_id = conversation_id
		self.reasoning = reasoning
		self.parallel_tools = is_parallel
		self.tool_choice = tool_choice
		self.response = None
		self.file = None
		self.file_url = file_path
		self.file_path = file_path
		self.image_url = None
		self.content = content
		self.max_search_results = max_search_results
		self.purpose = None
		self.request = { }
		self.messages = [ ]
		self.vector_stores = cfg.GPT_VECTOR_STORES
		self.files = {
				'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
				'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
				'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
				'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn',
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
		         'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'file_search_call.results',
				'web_search_call.results',
				'web_search_call.action.sources',
				'code_interpreter_call.outputs',
				'reasoning.encrypted_content',
				'message.output_text.logprobs',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'web_search',
				'file_search',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		"""Purpose options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'assistants',
				'batch',
				'fine-tune',
				'vision',
				'user_data',
				'evals',
		]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'text',
				'json_object',
				'json_schema',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'none',
				'minimal',
				'low',
				'medium',
				'high',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'text',
		]
	
	def build_reasoning( self, reasoning: str | Dict[ str, str ] = None ) -> Dict[
		                                                                         str, str ] | None:
		"""Build reasoning.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    reasoning (str | Dict[str, str]): Reasoning value used by the operation.
		
		Returns:
		    Dict[str, str] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if reasoning is None:
				return None
			
			if isinstance( reasoning, dict ):
				value = reasoning.get( 'effort' )
				if isinstance( value, str ) and value.strip( ) in self.reasoning_options:
					if value.strip( ) == 'none':
						return None
					
					return { 'effort': value.strip( ) }
				
				return None
			
			if isinstance( reasoning, str ) and reasoning.strip( ):
				value = reasoning.strip( )
				if value == 'none':
					return None
				
				if value in self.reasoning_options:
					return { 'effort': value }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_reasoning( self, reasoning )'
			Logger( ).write( exception )
			raise exception
	
	def build_input( self, prompt: str, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build input.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    context (List[Dict[str, str]]): Context value used by the operation.
		    input_data (List[Dict[str, Any]]): Input data value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.messages = [ ]
			
			if input_data is not None and len( input_data ) > 0:
				self.messages.extend( input_data )
			elif context is not None and len( context ) > 0:
				for item in context:
					if not isinstance( item, dict ):
						continue
					
					role = str( item.get( 'role', '' ) or '' ).strip( )
					content = item.get( 'content', '' )
					
					if role not in [ 'user', 'assistant', 'system', 'developer' ]:
						continue
					
					if not isinstance( content, str ) or not content.strip( ):
						continue
					
					self.messages.append(
						{
								'role': role,
								'content': [
										{
												'type': 'input_text',
												'text': content.strip( ),
										}, ],
						} )
			
			self.messages.append(
				{
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt,
								}, ],
				} )
			
			return self.messages
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_input( self, prompt, context, input_data )'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: List[ Any ] = None, allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""Build tools.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    tools (List[Any]): Tools value used by the operation.
		    allowed_domains (List[str]): Allowed domains value used by the operation.
		    vector_store_ids (List[str]): Vector store ids value used by the operation.
		
		Returns:
		    List[Dict[str, Any]] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			
			if tools is None or len( tools ) == 0:
				return None
			
			self.built_tools = [ ]
			for tool in tools:
				if isinstance( tool, dict ):
					tool_type = str( tool.get( 'type', '' ) or '' ).strip( )
				else:
					tool_type = str( tool or '' ).strip( )
				
				if not tool_type:
					continue
				
				if tool_type in [ 'web_search', 'web_search_preview', 'web_search_2025_08_26' ]:
					built_tool = { 'type': 'web_search' }
					if len( self.allowed_domains ) > 0:
						built_tool[ 'filters' ] = { 'allowed_domains': self.allowed_domains }
					
					self.built_tools.append( built_tool )
					continue
				
				if tool_type == 'file_search':
					resolved_store_ids = self.vector_store_ids
					if isinstance( tool, dict ) and len( resolved_store_ids ) == 0:
						tool_store_ids = tool.get( 'vector_store_ids', [ ] )
						if isinstance( tool_store_ids, list ):
							resolved_store_ids = tool_store_ids
					
					if len( resolved_store_ids ) == 0:
						continue
					
					self.built_tools.append(
						{
								'type': 'file_search',
								'vector_store_ids': resolved_store_ids,
						} )
					continue
				
				if tool_type == 'code_interpreter':
					self.built_tools.append( { 'type': 'code_interpreter' } )
					continue
			
			return self.built_tools if len( self.built_tools ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, tools, allowed_domains, vector_store_ids )'
			Logger( ).write( exception )
			raise exception
	
	def build_text_format( self, format: Dict[ str, Any ] | str = None ) -> Dict[ str, Any ] | None:
		"""Build text format.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    format (Dict[str, Any] | str): Format value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if format is None:
				return None
			
			if isinstance( format, str ) and format.strip( ):
				value = format.strip( )
				if value == 'text':
					return { 'format': { 'type': 'text' } }
				
				if value == 'json_object':
					return { 'format': { 'type': 'json_object' } }
				
				return None
			
			if not isinstance( format, dict ) or len( format ) == 0:
				return None
			
			if 'format' in format and isinstance( format.get( 'format' ), dict ):
				text_format = dict( format.get( 'format' ) )
			elif 'type' in format:
				text_format = dict( format )
			else:
				return None
			
			format_type = str( text_format.get( 'type', '' ) or '' ).strip( )
			if not format_type:
				return None
			
			if format_type == 'text':
				return { 'format': { 'type': 'text' } }
			
			if format_type == 'json_object':
				return { 'format': { 'type': 'json_object' } }
			
			if format_type == 'json_schema':
				json_schema = text_format.get( 'json_schema' )
				if isinstance( json_schema, dict ):
					schema_name = str( json_schema.get( 'name', '' ) or '' ).strip( )
					schema = json_schema.get( 'schema' )
					description = json_schema.get( 'description' )
					strict = json_schema.get( 'strict' )
				else:
					schema_name = str( text_format.get( 'name', '' ) or '' ).strip( )
					schema = text_format.get( 'schema' )
					description = text_format.get( 'description' )
					strict = text_format.get( 'strict' )
				
				if not schema_name:
					schema_name = 'response_schema'
				
				if not isinstance( schema, dict ) or len( schema ) == 0:
					return None
				
				normalized = {
						'type': 'json_schema',
						'name': schema_name,
						'schema': schema,
				}
				
				if isinstance( description, str ) and description.strip( ):
					normalized[ 'description' ] = description.strip( )
				
				if isinstance( strict, bool ):
					normalized[ 'strict' ] = strict
				
				return { 'format': normalized }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_text_format( self, format )'
			Logger( ).write( exception )
			raise exception
	
	def build_tool_choice( self, tool_choice: str = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> str | None:
		"""Build tool choice.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    tool_choice (str): Tool choice value used by the operation.
		    tools (List[Dict[str, Any]]): Tools value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if not isinstance( tool_choice, str ) or not tool_choice.strip( ):
				return None
			
			choice = tool_choice.strip( )
			if choice not in self.choice_options:
				return None
			
			if choice == 'none':
				return 'none'
			
			if tools is None or len( tools ) == 0:
				return None
			
			return choice
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_tool_choice( self, tool_choice, tools )'
			Logger( ).write( exception )
			raise exception
	
	def build_include( self, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> List[ str ] | None:
		"""Build include.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    include (List[str]): Include value used by the operation.
		    tools (List[Dict[str, Any]]): Tools value used by the operation.
		
		Returns:
		    List[str] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if include is None or len( include ) == 0:
				return None
			
			tool_types = [ ]
			if isinstance( tools, list ):
				for tool in tools:
					if isinstance( tool, dict ) and tool.get( 'type' ):
						tool_types.append( str( tool.get( 'type' ) ) )
			
			allowed = [ ]
			for value in include:
				if not isinstance( value, str ) or not value.strip( ):
					continue
				
				name = value.strip( )
				if name == 'reasoning.encrypted_content':
					allowed.append( name )
					continue
				
				if name == 'message.output_text.logprobs':
					allowed.append( name )
					continue
				
				if name.startswith( 'web_search_call.' ) and 'web_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'file_search_call.results' and 'file_search' in tool_types:
					allowed.append( name )
					continue
			
			return allowed if len( allowed ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_include( self, include, tools )'
			Logger( ).write( exception )
			raise exception
	
	def build_request( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None,
			background: bool = False, reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, allowed_domains: List[ str ] = None,
			previous_id: str = None, tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None, input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    format (Dict[str, Any]): Format value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    max_tools (int): Max tools value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    background (bool): Background value used by the operation.
		    reasoning (str): Reasoning value used by the operation.
		    include (List[str]): Include value used by the operation.
		    tools (List[Dict[str, Any]]): Tools value used by the operation.
		    allowed_domains (List[str]): Allowed domains value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    is_parallel (bool): Is parallel value used by the operation.
		    context (List[Dict[str, str]]): Context value used by the operation.
		    input_data (List[Dict[str, Any]]): Input data value used by the operation.
		    vector_store_ids (List[str]): Vector store ids value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.model = model
			self.prompt = prompt
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.background = background
			self.instructions = instruct
			self.response_format = self.build_text_format( format )
			self.max_tools = max_tools
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			self.previous_id = previous_id if isinstance( previous_id, str ) else None
			self.conversation_id = conversation_id if isinstance( conversation_id, str ) else None
			self.parallel_tools = is_parallel
			self.reasoning = self.build_reasoning( reasoning )
			self.tools = self.build_tools( tools=tools, allowed_domains=allowed_domains,
				vector_store_ids=self.vector_store_ids )
			self.tool_choice = self.build_tool_choice( tool_choice=tool_choice, tools=self.tools )
			self.include = self.build_include( include=include, tools=self.tools )
			self.input = self.build_input( prompt=prompt, context=context, input_data=input_data )
			self.request = {
					'model': self.model,
					'input': self.input,
			}
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if self.reasoning is not None:
				self.request[ 'reasoning' ] = self.reasoning
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'top_p' ] = self.top_percent
			
			if self.frequency_penalty is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'presence_penalty' ] = self.presence_penalty
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.include is not None and len( self.include ) > 0:
				self.request[ 'include' ] = self.include
			
			if self.tools is not None and len( self.tools ) > 0:
				self.request[ 'tools' ] = self.tools
			
			if self.tool_choice:
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.parallel_tools is not None and self.tools is not None:
				self.request[ 'parallel_tool_calls' ] = self.parallel_tools
			
			if self.previous_id and self.previous_id.strip( ):
				self.request[ 'previous_response_id' ] = self.previous_id.strip( )
			
			if self.conversation_id and self.conversation_id.strip( ):
				self.request[ 'conversation' ] = self.conversation_id.strip( )
			
			if isinstance( self.max_tools, int ) and self.max_tools > 0 and self.tools is not None:
				self.request[ 'max_tool_calls' ] = self.max_tools
			
			if self.response_format is not None and len( self.response_format ) > 0:
				self.request[ 'text' ] = self.response_format
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_request( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""Get output text.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if self.response is None:
				return None
			
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			if hasattr( self.response, 'output' ) and self.response.output:
				text_parts = [ ]
				for item in self.response.output:
					if getattr( item, 'type', None ) != 'message':
						continue
					
					if not hasattr( item, 'content' ) or item.content is None:
						continue
					
					for block in item.content:
						if getattr( block, 'type', None ) == 'output_text':
							text = getattr( block, 'text', None )
							if text:
								text_parts.append( text )
				
				if len( text_parts ) > 0:
					self.output_text = ''.join( text_parts ).strip( )
					return self.output_text
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def get_usage( self ) -> Any:
		"""Get usage.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if self.response is None:
				return None
			
			return getattr( self.response, 'usage', None )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'get_usage( self ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None, background: bool = False,
			reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None, tool_choice: str = None,
			is_parallel: bool = None, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None, vector_store_ids: List[ str ] = None,
			conversation_id: str = None ) -> str | None:
		"""Generate text.
		
		Purpose:
		    Generates provider output for the Chat workflow using validated model settings and request
		    inputs. The method coordinates request construction, provider execution, response capture, and
		    logged exception handling.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    format (Dict[str, Any]): Format value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    max_tools (int): Max tools value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    background (bool): Background value used by the operation.
		    reasoning (str): Reasoning value used by the operation.
		    include (List[str]): Include value used by the operation.
		    tools (List[Dict[str, Any]]): Tools value used by the operation.
		    allowed_domains (List[str]): Allowed domains value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    is_parallel (bool): Is parallel value used by the operation.
		    context (List[Dict[str, str]]): Context value used by the operation.
		    input_data (List[Dict[str, Any]]): Input data value used by the operation.
		    vector_store_ids (List[str]): Vector store ids value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			
			self.stream_requested = bool( stream )
			self.background_requested = bool( background )
			
			self.request = self.build_request( prompt=prompt, model=model,
				temperature=temperature, format=format, top_p=top_p, frequency=frequency,
				max_tools=max_tools, presence=presence, max_tokens=max_tokens, store=store,
				stream=False, instruct=instruct, background=False, reasoning=reasoning,
				include=include, tools=tools, allowed_domains=allowed_domains,
				previous_id=previous_id, tool_choice=tool_choice, is_parallel=is_parallel,
				context=context, input_data=input_data, vector_store_ids=vector_store_ids,
				conversation_id=conversation_id )
			
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', None )
			self.output_text = self.get_output_text( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Chat.__dir__ workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'client',
				'model',
				'prompt',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'stops',
				'store',
				'stream',
				'background',
				'number',
				'response_format',
				'context',
				'instructions',
				'include',
				'tool_choice',
				'previous_id',
				'conversation_id',
				'parallel_tools',
				'max_tools',
				'input',
				'tools',
				'reasoning',
				'allowed_domains',
				'max_search_results',
				'output_text',
				'vector_store_ids',
				'file_ids',
				'response',
				'file',
				'purpose',
				'model_options',
				'include_options',
				'tool_options',
				'choice_options',
				'purpose_options',
				'format_options',
				'reasoning_options',
				'modality_options',
				'build_reasoning',
				'build_input',
				'build_tools',
				'build_tool_choice',
				'build_include',
				'build_text_format',
				'build_request',
				'get_output_text',
				'get_usage',
				'generate_text',
		]

class Images( GPT ):
	"""Images class.
	
	Purpose:
	    Defines the Images component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    quality (Optional[str]): Stores quality for the component runtime state.
	    detail (Optional[str]): Stores detail for the component runtime state.
	    size (Optional[str]): Stores size for the component runtime state.
	    previous_id (Optional[str]): Stores previous id for the component runtime state.
	    include (Optional[List[str]]): Stores include for the component runtime state.
	    tool_choice (Optional[str]): Stores tool choice for the component runtime state.
	    parallel_tools (Optional[bool]): Stores parallel tools for the component runtime state.
	    input (Optional[List[Dict[str, Any]] | str]): Stores input for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    max_tools (Optional[int]): Stores max tools for the component runtime state.
	    tools (Optional[List[Dict[str, Any]]]): Stores tools for the component runtime state.
	    messages (Optional[List[Dict[str, Any]]]): Stores messages for the component runtime state.
	    reasoning (Optional[Dict[str, Any]]): Stores reasoning for the component runtime state.
	    allowed_domains (Optional[List[str]]): Stores allowed domains for the component runtime state.
	    max_tokens (Optional[int]): Stores max tokens for the component runtime state.
	    temperature (Optional[float]): Stores temperature for the component runtime state.
	    store (Optional[bool]): Stores store for the component runtime state.
	    stream (Optional[bool]): Stores stream for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    request (Optional[Dict[str, Any]]): Stores request for the component runtime state.
	    output_text (Optional[str]): Stores output text for the component runtime state.
	    output (Optional[Any]): Stores output for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    image_path (Optional[str]): Stores image path for the component runtime state.
	    image_url (Optional[str]): Stores image url for the component runtime state.
	    mask_path (Optional[str]): Stores mask path for the component runtime state."""
	quality: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
	previous_id: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	parallel_tools: Optional[ bool ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	instructions: Optional[ str ]
	max_tools: Optional[ int ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning: Optional[ Dict[ str, Any ] ]
	allowed_domains: Optional[ List[ str ] ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	response: Optional[ Any ]
	request: Optional[ Dict[ str, Any ] ]
	output_text: Optional[ str ]
	output: Optional[ Any ]
	file_path: Optional[ str ]
	image_path: Optional[ str ]
	image_url: Optional[ str ]
	mask_path: Optional[ str ]
	
	def __init__( self, prompt: str = None, model: str = 'gpt-image-1', temperature: float = None,
			top_p: float = None, presence: float = None, frequency: float = None,
			max_tokens: int = None,
			store: bool = None, stream: bool = False, backcolor: str = None, instruct: str = None,
			background: bool = None, number: int = None, image_format: str = None,
			include: List[ Dict[ str, str ] ] = None, tools: List[ Dict[ str, str ] ] = None,
			max_tools: int = None, respose_format: Dict[ str, str ] = None,
			response_format: Dict[ str, str ] = None, tool_choice: str = None,
			image_path: str = None,
			is_parallel: bool = None, input: List[ Dict[ str, str ] ] = None,
			previous_id: str = None,
			reasoning: Dict[ str, str ] = None, input_text: str = None, image_url: str = None,
			content: List[ Dict[ str, str ] ] = None, quality: str = None, size: str = None,
			detail: str = None, style: str = None, compression: float = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Images object with its default configuration, runtime state, provider settings,
		    and compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    presence (float): Presence value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    backcolor (str): Backcolor value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    background (bool): Background value used by the operation.
		    number (int): Number value used by the operation.
		    image_format (str): Image format value used by the operation.
		    include (List[Dict[str, str]]): Include value used by the operation.
		    tools (List[Dict[str, str]]): Tools value used by the operation.
		    max_tools (int): Max tools value used by the operation.
		    respose_format (Dict[str, str]): Respose format value used by the operation.
		    response_format (Dict[str, str]): Response format value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    image_path (str): Image path value used by the operation.
		    is_parallel (bool): Is parallel value used by the operation.
		    input (List[Dict[str, str]]): Input value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    reasoning (Dict[str, str]): Reasoning value used by the operation.
		    input_text (str): Input text value used by the operation.
		    image_url (str): Image url value used by the operation.
		    content (List[Dict[str, str]]): Content value used by the operation.
		    quality (str): Quality value used by the operation.
		    size (str): Size value used by the operation.
		    detail (str): Detail value used by the operation.
		    style (str): Style value used by the operation.
		    compression (float): Compression value used by the operation."""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = prompt
		self.input_text = input_text
		self.temperature = temperature
		self.top_percent = top_p
		self.presence_penalty = presence
		self.frequency_penalty = frequency
		self.max_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.background = backcolor if backcolor is not None else background
		self.backcolor = backcolor
		self.instructions = instruct
		self.number = number
		self.mime_format = image_format
		self.output_format = image_format
		self.include = include if include is not None else [ ]
		self.tools = tools if tools is not None else [ ]
		self.max_tools = max_tools
		self.response_format = response_format if response_format is not None else respose_format
		self.tool_choice = tool_choice
		self.image_path = image_path
		self.file_path = image_path
		self.parallel_tools = is_parallel
		self.input = input
		self.previous_id = previous_id
		self.reasoning = reasoning
		self.image_url = image_url
		self.content = content
		self.quality = quality
		self.size = size
		self.detail = detail
		self.style = style
		self.compression = compression
		self.messages = [ ]
		self.allowed_domains = [ ]
		self.request = None
		self.response = None
		self.output = None
		self.output_text = None
		self.mask_path = None
	
	@property
	def style_options( self ) -> List[ str ]:
		"""Style options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'vivid',
				'natural',
		]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
		]
	
	@property
	def generation_model_options( self ) -> List[ str ] | None:
		"""Generation model options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
		]
	
	@property
	def edit_model_options( self ) -> List[ str ] | None:
		"""Edit model options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
		]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Size options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'auto',
				'1024x1024',
				'1024x1536',
				'1536x1024',
		]
	
	@property
	def analysis_model_options( self ) -> List[ str ] | None:
		"""Analysis model options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-5.4',
				'gpt-5.4-mini',
				'gpt-5',
				'gpt-5-mini',
				'gpt-4.1',
				'gpt-4.1-mini',
				'gpt-4o',
				'gpt-4o-mini',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'url',
				'b64_json',
		]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'png',
				'jpeg',
				'webp',
		]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'file_search_call.results',
				'web_search_call.results',
				'web_search_call.action.sources',
				'message.input_image.image_url',
				'computer_call_output.output.image_url',
				'code_interpreter_call.outputs',
				'reasoning.encrypted_content',
				'message.output_text.logprobs',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'web_search',
				'image_generation',
				'file_search',
				'code_interpreter',
				'computer_use_preview',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
				'required',
				'none',
		]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		"""Backcolor options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'auto',
				'transparent',
				'opaque',
		]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""Quality options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'auto',
				'low',
				'medium',
				'high',
		]
	
	@property
	def detail_options( self ) -> List[ str ]:
		"""Detail options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'auto',
				'low',
				'high',
				'original',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'low',
				'medium',
				'high',
				'none',
				'minimal',
				'xhigh',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'text',
				'auto',
				'image',
				'audio',
		]
	
	def generate( self, prompt: str = None, number: int = 1, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None, style: str = None,
			**kwargs: Any ) -> str | bytes | list[ str | bytes ] | None:
		"""Generate.
		
		Purpose:
		    Generates provider output for the Images workflow using validated model settings and request
		    inputs. The method coordinates request construction, provider execution, response capture, and
		    logged exception handling.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    number (int): Number value used by the operation.
		    model (str): Model value used by the operation.
		    size (str): Size value used by the operation.
		    quality (str): Quality value used by the operation.
		    fmt (str): Fmt value used by the operation.
		    compression (float): Compression value used by the operation.
		    background (str): Background value used by the operation.
		    style (str): Style value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    str | bytes | List[str | bytes] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			input_text = (prompt or kwargs.get( 'text' ) or kwargs.get( 'input_text' )
			              or kwargs.get( 'content' ))
			throw_if( 'input_text', input_text )
			throw_if( 'model', model )
			if cfg.OPENAI_API_KEY is None or not str( cfg.OPENAI_API_KEY ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			self.api_key = cfg.OPENAI_API_KEY
			self.prompt = str( input_text ).strip( )
			self.model = str( model ).strip( )
			self.number = number if isinstance( number, int ) and number > 0 else 1
			self.size = str( size or '1024x1024' ).strip( )
			self.quality = str( quality or 'auto' ).strip( )
			self.output_format = str( fmt or 'jpeg' ).strip( ).lower( ).replace( '.', '' )
			self.compression = compression
			self.background = background
			self.style = style
			
			if self.output_format.startswith( 'image/' ):
				self.output_format = self.output_format.replace( 'image/', '' )
			
			self.client = OpenAI( api_key=self.api_key )
			self.request = {
					'model': self.model,
					'prompt': self.prompt,
			}
			
			if self.model == 'dall-e-2':
				if self.size not in [ '256x256', '512x512', '1024x1024' ]:
					self.size = '1024x1024'
				
				if self.output_format not in self.format_options:
					self.output_format = 'url'
				
				self.request[ 'size' ] = self.size
				self.request[ 'n' ] = self.number
				self.request[ 'response_format' ] = self.output_format
			
			elif self.model == 'dall-e-3':
				if self.size not in [ '1024x1024', '1792x1024', '1024x1792' ]:
					self.size = '1024x1024'
				
				if self.quality not in [ 'standard', 'hd' ]:
					self.quality = 'standard'
				
				self.request[ 'size' ] = self.size
				self.request[ 'quality' ] = self.quality
				self.request[ 'n' ] = 1
				
				if self.style in self.style_options:
					self.request[ 'style' ] = self.style
				
				if self.output_format in self.format_options:
					self.request[ 'response_format' ] = self.output_format
			
			else:
				if self.size not in self.size_options:
					self.size = '1024x1024'
				
				if self.quality not in self.quality_options:
					self.quality = 'auto'
				
				if self.output_format not in self.mime_options:
					self.output_format = 'jpeg'
				
				if self.background not in self.backcolor_options:
					self.background = None
				
				if self.model.startswith( 'gpt-image-2' ) and self.background == 'transparent':
					self.background = 'auto'
				
				self.request[ 'n' ] = self.number
				self.request[ 'size' ] = self.size
				self.request[ 'quality' ] = self.quality
				self.request[ 'output_format' ] = self.output_format
				
				if self.background:
					self.request[ 'background' ] = self.background
				
				if self.compression is not None and self.output_format in [ 'jpeg', 'webp' ]:
					if isinstance( self.compression, float ) and self.compression <= 1.0:
						self.compression = int( round( self.compression * 100 ) )
					else:
						self.compression = int( round( self.compression ) )
					
					self.compression = max( 0, min( 100, int( self.compression ) ) )
					self.request[ 'output_compression' ] = self.compression
			
			self.response = self.client.images.generate( **self.request )
			
			if getattr( self.response, 'data', None ):
				self.output = [ ]
				for item in self.response.data:
					if getattr( item, 'b64_json', None ):
						self.output.append( base64.b64decode( item.b64_json ) )
					elif getattr( item, 'url', None ):
						self.output.append( item.url )
				
				if len( self.output ) == 1:
					return self.output[ 0 ]
				
				if len( self.output ) > 1:
					return self.output
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt: str=None, model: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, text: str = None, path: str = None, model: str = 'gpt-4o-mini',
			instruct: str = None, max_tokens: int = None, temperature: float = None,
			include: List[ str ] = None, store: bool = None, stream: bool = None,
			detail: str = 'auto', **kwargs: Any ) -> str | None:
		"""Analyze.
		
		Purpose:
		    Performs the Images.analyze workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    text (str): Text value used by the operation.
		    path (str): Path value used by the operation.
		    model (str): Model value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    include (List[str]): Include value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    detail (str): Detail value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			input_text = (text or kwargs.get( 'prompt' ) or kwargs.get( 'input_text' )
			              or kwargs.get( 'content' ))
			file_path = path or kwargs.get( 'image_path' ) or kwargs.get( 'file_path' )
			throw_if( 'input_text', input_text )
			throw_if( 'file_path', file_path )
			throw_if( 'model', model )
			
			if cfg.OPENAI_API_KEY is None or not str( cfg.OPENAI_API_KEY ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			self.api_key = cfg.OPENAI_API_KEY
			self.input_text = str( input_text ).strip( )
			self.file_path = str( file_path ).strip( )
			self.image_path = self.file_path
			self.model = str( model ).strip( )
			self.instructions = instruct
			self.max_tokens = max_tokens
			self.temperature = temperature
			self.include = include if include is not None else [ ]
			self.store = store
			self.stream = stream
			self.detail = str( detail or 'auto' ).strip( )
			
			if self.model.startswith( 'gpt-image' ):
				self.model = 'gpt-4o-mini'
			
			if self.detail == 'original':
				self.detail = 'auto'
			
			if self.detail not in [ 'auto', 'low', 'high' ]:
				self.detail = 'auto'
			
			self.file_suffix = Path( self.file_path ).suffix.lower( ).replace( '.', '' )
			if self.file_suffix == 'jpg':
				self.file_suffix = 'jpeg'
			
			if self.file_suffix not in [ 'jpeg', 'png', 'webp', 'gif' ]:
				self.file_suffix = 'png'
			
			with open( self.file_path, 'rb' ) as image_file:
				self.encoded_image = base64.b64encode( image_file.read( ) ).decode( 'utf-8' )
			
			self.input = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'input_text',
											'text': self.input_text,
									},
									{
											'type': 'input_image',
											'image_url': f'data:image/{self.file_suffix};base64,{self.encoded_image}',
											'detail': self.detail,
									},
							],
					}
			]
			
			self.request = {
					'model': self.model,
					'input': self.input,
			}
			
			if self.instructions and str( self.instructions ).strip( ):
				self.request[ 'instructions' ] = str( self.instructions ).strip( )
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if isinstance( self.include, list ) and len( self.include ) > 0:
				self.request[ 'include' ] = self.include
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.stream is not None:
				self.request[ 'stream' ] = self.stream
			
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.responses.create( **self.request )
			
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			try:
				for item in self.response.output:
					if getattr( item, 'type', None ) != 'message':
						continue
					
					for block in item.content:
						if getattr( block, 'type', None ) == 'output_text':
							self.output_text = getattr( block, 'text', None )
							if self.output_text:
								return self.output_text
			except Exception:
				pass
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'analyze( self, text: str=None, path: str=None, model: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str = None, path: str = None, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None, number: int = 1,
			mask_path: str = None, mask: str = None,
			**kwargs: Any ) -> str | bytes | list[ str | bytes ] | None:
		"""Edit.
		
		Purpose:
		    Performs the Images.edit workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    path (str): Path value used by the operation.
		    model (str): Model value used by the operation.
		    size (str): Size value used by the operation.
		    quality (str): Quality value used by the operation.
		    fmt (str): Fmt value used by the operation.
		    compression (float): Compression value used by the operation.
		    background (str): Background value used by the operation.
		    number (int): Number value used by the operation.
		    mask_path (str): Mask path value used by the operation.
		    mask (str): Mask value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    str | bytes | List[str | bytes] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		source = None
		mask_source = None
		
		try:
			input_text = (prompt or kwargs.get( 'text' )
			              or kwargs.get( 'input_text' ) or kwargs.get( 'content' ))
			file_path = path or kwargs.get( 'image_path' ) or kwargs.get( 'file_path' )
			throw_if( 'input_text', input_text )
			throw_if( 'file_path', file_path )
			throw_if( 'model', model )
			
			if cfg.OPENAI_API_KEY is None or not str( cfg.OPENAI_API_KEY ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			self.api_key = cfg.OPENAI_API_KEY
			self.input_text = str( input_text ).strip( )
			self.prompt = self.input_text
			self.file_path = str( file_path ).strip( )
			self.image_path = self.file_path
			self.mask_path = mask_path or mask or kwargs.get( 'mask_path' ) or kwargs.get( 'mask' )
			self.model = str( model ).strip( )
			self.number = number if isinstance( number, int ) and number > 0 else 1
			self.size = str( size or '1024x1024' ).strip( )
			self.quality = str( quality or 'auto' ).strip( )
			self.output_format = str( fmt or 'jpeg' ).strip( ).lower( ).replace( '.', '' )
			self.compression = compression
			self.background = background
			if self.output_format.startswith( 'image/' ):
				self.output_format = self.output_format.replace( 'image/', '' )
			
			self.request = {
					'model': self.model,
					'prompt': self.input_text,
					'n': self.number,
					'size': self.size,
					'quality': self.quality,
			}
			
			if self.output_format in self.mime_options:
				self.request[ 'output_format' ] = self.output_format
			elif self.output_format in self.format_options:
				self.request[ 'response_format' ] = self.output_format
			
			if self.background in self.backcolor_options:
				self.request[ 'background' ] = self.background
			
			if self.compression is not None and self.output_format in [ 'jpeg', 'webp' ]:
				if isinstance( self.compression, float ) and self.compression <= 1.0:
					self.compression = int( round( self.compression * 100 ) )
				else:
					self.compression = int( round( self.compression ) )
				
				self.compression = max( 0, min( 100, int( self.compression ) ) )
				self.request[ 'output_compression' ] = self.compression
			
			self.client = OpenAI( api_key=self.api_key )
			source = open( self.file_path, 'rb' )
			
			if self.mask_path:
				mask_source = open( self.mask_path, 'rb' )
				self.response = self.client.images.edit( image=source, mask=mask_source,
					**self.request )
			else:
				self.response = self.client.images.edit( image=source, **self.request )
			
			if getattr( self.response, 'data', None ):
				self.output = [ ]
				for item in self.response.data:
					if getattr( item, 'b64_json', None ):
						self.output.append( base64.b64decode( item.b64_json ) )
					elif getattr( item, 'url', None ):
						self.output.append( item.url )
				
				if len( self.output ) == 1:
					return self.output[ 0 ]
				
				if len( self.output ) > 1:
					return self.output
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'edit( self, prompt: str=None, path: str=None, model: str=None )'
			Logger( ).write( exception )
			raise exception
		finally:
			if source is not None:
				source.close( )
			
			if mask_source is not None:
				mask_source.close( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Images.__dir__ workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'client',
				'model',
				'prompt',
				'input_text',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'store',
				'stream',
				'background',
				'number',
				'response_format',
				'instructions',
				'include',
				'tool_choice',
				'previous_id',
				'parallel_tools',
				'max_tools',
				'input',
				'tools',
				'reasoning',
				'allowed_domains',
				'output_text',
				'response',
				'request',
				'output',
				'image_path',
				'image_url',
				'file_path',
				'mask_path',
				'size',
				'quality',
				'detail',
				'style',
				'compression',
				'style_options',
				'model_options',
				'generation_model_options',
				'edit_model_options',
				'size_options',
				'analysis_model_options',
				'format_options',
				'mime_options',
				'include_options',
				'tool_options',
				'choice_options',
				'backcolor_options',
				'quality_options',
				'detail_options',
				'reasoning_options',
				'modality_options',
				'generate',
				'analyze',
				'edit',
		]

class TTS( ):
	"""TTS class.
	
	Purpose:
	    Defines the TTS component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    client (Optional[OpenAI]): Stores client for the component runtime state.
	    speed (Optional[float]): Stores speed for the component runtime state.
	    voice (Optional[str]): Stores voice for the component runtime state.
	    input (Optional[str]): Stores input for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    audio_bytes (Optional[bytes]): Stores audio bytes for the component runtime state.
	    request (Optional[Dict[str, Any]]): Stores request for the component runtime state."""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	input: Optional[ str ]
	instructions: Optional[ str ]
	response: Optional[ Any ]
	response_format: Optional[ str ]
	file_path: Optional[ str ]
	model: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, input: str = None, model: str = 'gpt-4o-mini-tts', format: str = None,
			instruct: str = None, voice: str = None, speed: float = None, file_path: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the TTS object with its default configuration, runtime state, provider settings, and
		    compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment.
		
		Args:
		    input (str): Input value used by the operation.
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    voice (str): Voice value used by the operation.
		    speed (float): Speed value used by the operation.
		    file_path (str): File path value used by the operation."""
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.input = input
		self.model = model
		self.instructions = instruct
		self.response_format = format
		self.voice = voice
		self.file_path = file_path
		self.speed = speed
		self.response = None
		self.audio_bytes = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-4o-mini-tts',
				'gpt-4o-mini-tts-2025-12-15',
				'tts-1',
				'tts-1-hd',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'mp3',
				'opus',
				'aac',
				'flac',
				'wav',
				'pcm',
		]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		"""Voice options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'alloy',
				'ash',
				'ballad',
				'coral',
				'echo',
				'fable',
				'nova',
				'onyx',
				'sage',
				'shimmer',
				'verse',
				'marin',
				'cedar',
		]
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		"""Speed options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[float] | None: Return value produced by the operation."""
		return [
				0.25,
				0.50,
				0.75,
				1.0,
				1.25,
				1.50,
				2.0,
				3.0,
				4.0,
		]
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		Purpose:
		    Performs the TTS.validate_model workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini-tts'
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported TTS model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_format( self, format: str = None ) -> str:
		"""Validate format.
		
		Purpose:
		    Performs the TTS.validate_format workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    format (str): Format value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = format if isinstance( format, str ) and format.strip( ) else 'mp3'
			value = value.strip( ).lower( )
			if value not in self.mime_options:
				raise ValueError( f'Unsupported TTS output format: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_format( self, format: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_voice( self, voice: str = None ) -> str:
		"""Validate voice.
		
		Purpose:
		    Performs the TTS.validate_voice workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    voice (str): Voice value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = voice if isinstance( voice, str ) and voice.strip( ) else 'alloy'
			value = value.strip( )
			if value not in self.voice_options:
				raise ValueError( f'Unsupported TTS voice: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_voice( self, voice: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_speed( self, speed: float = None ) -> float:
		"""Validate speed.
		
		Purpose:
		    Performs the TTS.validate_speed workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    speed (float): Speed value used by the operation.
		
		Returns:
		    float: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = 1.0 if speed is None else float( speed )
			if value < 0.25:
				return 0.25
			
			if value > 4.0:
				return 4.0
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_speed( self, speed: float=None ) -> float'
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str = None, model: str = 'gpt-4o-mini-tts',
			format: str = 'mp3', speed: float = 1.0, voice: str = 'alloy',
			instruct: str = None, file_path: str = None, **kwargs: Any ) -> bytes | None:
		"""Create speech.
		
		Purpose:
		    Creates the requested resource, connection, schema object, or user interface artifact using
		    validated inputs. The function encapsulates setup details so callers can rely on a consistent
		    resource lifecycle.
		
		Args:
		    text (str): Text value used by the operation.
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		    speed (float): Speed value used by the operation.
		    voice (str): Voice value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    file_path (str): File path value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    bytes | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			input_text = (text or kwargs.get( 'prompt' ) or kwargs.get( 'input' )
			              or kwargs.get( 'content' ))
			throw_if( 'input_text', input_text )
			model = str( model or 'gpt-4o-mini-tts' ).strip( )
			throw_if( 'model', model )
			api_key = cfg.OPENAI_API_KEY
			if api_key is None or not str( api_key ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			response_format = format or kwargs.get( 'response_format' ) or kwargs.get( 'mime_type' )
			response_format = self.validate_format( response_format )
			voice = self.validate_voice( voice or kwargs.get( 'voice' ) )
			speed = self.validate_speed( speed if speed is not None else kwargs.get( 'speed' ) )
			file_path = file_path or kwargs.get( 'filepath' ) or kwargs.get( 'audio_path' )
			instructions = instruct if instruct is not None else kwargs.get( 'instructions' )
			
			self.input = input_text
			self.model = self.validate_model( model )
			self.response_format = response_format
			self.voice = voice
			self.speed = speed
			self.instructions = instructions
			self.file_path = file_path
			self.client = OpenAI( api_key=api_key )
			self.response = None
			self.audio_bytes = None
			
			with tempfile.NamedTemporaryFile(
					suffix=f'.{self.response_format}', delete=False ) as tmp:
				temp_path = tmp.name
			
			try:
				request = {
						'model': self.model,
						'voice': self.voice,
						'input': self.input,
						'response_format': self.response_format,
						'speed': self.speed,
				}
				
				if self.instructions and self.model not in ('tts-1', 'tts-1-hd'):
					request[ 'instructions' ] = self.instructions
				
				self.request = request
				
				with self.client.audio.speech.with_streaming_response.create(
						**self.request ) as response:
					self.response = response
					response.stream_to_file( temp_path )
				
				with open( temp_path, 'rb' ) as source:
					audio_bytes = source.read( )
				
				if audio_bytes is None or len( audio_bytes ) == 0:
					raise ValueError( 'No audio bytes were returned by OpenAI Speech.' )
				
				self.audio_bytes = audio_bytes
				
				if self.file_path:
					with open( self.file_path, 'wb' ) as target:
						target.write( self.audio_bytes )
				
				return self.audio_bytes
			finally:
				try:
					if os.path.exists( temp_path ):
						os.remove( temp_path )
				except Exception:
					pass
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, text: str=None, **kwargs ) -> bytes | None'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the TTS.__dir__ workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'input',
				'file_path',
				'voice',
				'client',
				'response_format',
				'speed',
				'model',
				'instructions',
				'response',
				'audio_bytes',
				'request',
				'model_options',
				'mime_options',
				'voice_options',
				'speed_options',
				'validate_model',
				'validate_format',
				'validate_voice',
				'validate_speed',
				'create_speech',
		]

class Transcription( GPT ):
	"""Transcription class.
	
	Purpose:
	    Defines the Transcription component used by the Boo application. The class groups related
	    provider configuration, runtime state, helper methods, and API-facing behavior so Streamlit
	    workflows can call a consistent interface.
	
	Attributes:
	    client (Optional[OpenAI]): Stores client for the component runtime state.
	    language (Optional[str]): Stores language for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    include (Optional[List[str]]): Stores include for the component runtime state.
	    normalized_result (Optional[Dict[str, Any]]): Stores normalized result for the component runtime state."""
	client: Optional[ OpenAI ]
	language: Optional[ str ]
	instructions: Optional[ str ]
	include: Optional[ List[ str ] ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'gpt-4o-transcribe', temperature: float = None,
			prompt: str = None, number: int = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stream: bool = None, store: bool = None,
			language: str = None, instruct: str = None, format: str = None, background: bool = None,
			messages: List[ Dict[ str, str ] ] = None, stops: List[ str ] = None,
			include: List[ str ] = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Transcription object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment.
		
		Args:
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    number (int): Number value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stream (bool): Stream value used by the operation.
		    store (bool): Store value used by the operation.
		    language (str): Language value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    format (str): Format value used by the operation.
		    background (bool): Background value used by the operation.
		    messages (List[Dict[str, str]]): Messages value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    include (List[str]): Include value used by the operation."""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.prompt = prompt
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.stream = stream
		self.response_format = format
		self.background = background
		self.message = messages
		self.stops = stops
		self.store = store
		self.language = language
		self.instructions = instruct
		self.model = model
		self.number = number
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
		self.include = include if include is not None else [ ]
		self.normalized_result = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-4o-transcribe',
				'gpt-4o-mini-transcribe',
				'gpt-4o-mini-transcribe-2025-12-15',
				'whisper-1',
				'gpt-4o-transcribe-diarize',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'flac',
				'mp3',
				'mp4',
				'mpeg',
				'mpga',
				'm4a',
				'ogg',
				'wav',
				'webm',
		]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ru',
				'uk',
				'el',
				'he',
				'ar',
				'hi',
				'zh',
				'ja',
				'ko',
				'vi',
				'th',
		]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		"""Language labels.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    Dict[str, str] | None: Return value produced by the operation."""
		return {
				'en': 'English',
				'es': 'Spanish',
				'fr': 'French',
				'de': 'German',
				'it': 'Italian',
				'pt': 'Portuguese',
				'ru': 'Russian',
				'uk': 'Ukrainian',
				'el': 'Greek',
				'he': 'Hebrew',
				'ar': 'Arabic',
				'hi': 'Hindi',
				'zh': 'Chinese',
				'ja': 'Japanese',
				'ko': 'Korean',
				'vi': 'Vietnamese',
				'th': 'Thai',
		}
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'logprobs',
		]
	
	@property
	def response_format_options( self ) -> Dict[ str, List[ str ] ]:
		"""Response format options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    Dict[str, List[str]]: Return value produced by the operation."""
		return {
				'whisper-1': [
						'json',
						'text',
						'srt',
						'verbose_json',
						'vtt',
				],
				'gpt-4o-transcribe': [
						'json',
				],
				'gpt-4o-mini-transcribe': [
						'json',
				],
				'gpt-4o-mini-transcribe-2025-12-15': [
						'json',
				],
				'gpt-4o-transcribe-diarize': [
						'json',
						'text',
						'diarized_json',
				],
		}
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		Purpose:
		    Performs the Transcription.validate_model workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-transcribe'
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported transcription model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_format( self, model: str, format: str = None ) -> str | None:
		"""Validate format.
		
		Purpose:
		    Performs the Transcription.validate_format workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			options = self.response_format_options.get( model, [ 'json' ] )
			if not isinstance( format, str ) or not format.strip( ):
				return options[ 0 ] if len( options ) > 0 else None
			
			value = format.strip( )
			if value not in options:
				return options[ 0 ] if len( options ) > 0 else None
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'validate_format( self, model: str, format: str=None ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def validate_include( self, model: str, include: List[ str ] = None ) -> List[ str ]:
		"""Validate include.
		
		Purpose:
		    Performs the Transcription.validate_include workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		    include (List[str]): Include value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if include is None or len( include ) == 0:
				return [ ]
			
			if model not in [ 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe',
			                  'gpt-4o-mini-transcribe-2025-12-15' ]:
				return [ ]
			
			values = [ ]
			for item in include:
				if isinstance( item, str ) and item.strip( ) in self.include_options:
					values.append( item.strip( ) )
			
			return values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'validate_include( self, model: str, include: List[ str ]=None )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""Normalize response.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    response (Any): Response value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			result: Dict[ str, Any ] = {
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ] = response
				result[ 'raw' ] = response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ] = response.model_dump( )
				except Exception:
					result[ 'raw' ] = str( response )
			else:
				result[ 'raw' ] = str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ] = text
			
			segments = getattr( response, 'segments', None )
			if isinstance( segments, list ):
				normalized_segments = [ ]
				for segment in segments:
					if hasattr( segment, 'model_dump' ):
						normalized_segments.append( segment.model_dump( ) )
					elif isinstance( segment, dict ):
						normalized_segments.append( segment )
					else:
						normalized_segments.append( { 'text': str( segment ) } )
				
				result[ 'segments' ] = normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ] = language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ] = duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ] = '\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ] = str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str = None, model: str = 'gpt-4o-transcribe',
			language: str = None, prompt: str = None, format: str = None,
			temperature: float = None, include: List[ str ] = None,
			**kwargs: Any ) -> str | None:
		"""Transcribe.
		
		Purpose:
		    Performs the Transcription.transcribe workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		    model (str): Model value used by the operation.
		    language (str): Language value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    format (str): Format value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    include (List[str]): Include value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			file_path = (path or kwargs.get( 'filepath' ) or kwargs.get( 'file_path' )
			             or kwargs.get( 'audio_file' ))
			throw_if( 'file_path', file_path )
			
			model = str( model or 'gpt-4o-transcribe' ).strip( )
			throw_if( 'model', model )
			
			api_key = cfg.OPENAI_API_KEY
			if api_key is None or not str( api_key ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			language = language or kwargs.get( 'language' ) or kwargs.get( 'source_language' )
			if isinstance( language, str ) and language.lower( ) == 'auto':
				language = None
			
			prompt = prompt or kwargs.get( 'instructions' ) or kwargs.get( 'instruct' )
			if isinstance( prompt, str ) and not prompt.strip( ):
				prompt = None
			
			response_format = format or kwargs.get( 'response_format' )
			temperature = temperature if temperature is not None else kwargs.get( 'temperature' )
			include = include if include is not None else kwargs.get( 'include' )
			
			self.file_path = file_path
			self.model = self.validate_model( model )
			self.language = language if isinstance( language, str ) and language.strip( ) else None
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else None
			self.response_format = self.validate_format( self.model, response_format )
			self.temperature = temperature
			self.include = self.validate_include( self.model, include )
			self.client = OpenAI( api_key=api_key )
			
			request = {
					'model': self.model,
			}
			
			if self.language:
				request[ 'language' ] = self.language
			
			if self.prompt:
				request[ 'prompt' ] = self.prompt
			
			if self.response_format:
				request[ 'response_format' ] = self.response_format
			
			if self.include:
				request[ 'include' ] = self.include
			
			if self.temperature is not None and self.model == 'whisper-1':
				request[ 'temperature' ] = self.temperature
			
			self.request = request
			
			with open( self.file_path, 'rb' ) as audio_file:
				self.response = self.client.audio.transcriptions.create(
					file=audio_file,
					**self.request )
			
			self.normalized_result = self.normalize_response( self.response )
			self.transcript = self.normalized_result.get( 'text' )
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path: str=None, **kwargs ) -> str | None'
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Transcription.__dir__ workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'number',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'store',
				'stream',
				'stops',
				'prompt',
				'response',
				'audio_file',
				'messages',
				'response_format',
				'api_key',
				'client',
				'input_text',
				'transcript',
				'language',
				'model',
				'include',
				'normalized_result',
				'model_options',
				'mime_options',
				'language_options',
				'language_labels',
				'include_options',
				'response_format_options',
				'validate_model',
				'validate_format',
				'validate_include',
				'normalize_response',
				'transcribe',
		]

class Translation( GPT ):
	"""Translation class.
	
	Purpose:
	    Defines the Translation component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    client (Optional[OpenAI]): Stores client for the component runtime state.
	    target_language (Optional[str]): Stores target language for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    normalized_result (Optional[Dict[str, Any]]): Stores normalized result for the component runtime state."""
	client: Optional[ OpenAI ]
	target_language: Optional[ str ]
	response_format: Optional[ str ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'whisper-1', temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None,
			stream: bool = None, instruct: str = None, audio_file: str = None, format: str = None,
			language: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Translation object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment.
		
		Args:
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    audio_file (str): Audio file value used by the operation.
		    format (str): Format value used by the operation.
		    language (str): Language value used by the operation."""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.audio_file = audio_file
		self.response = None
		self.response_format = format
		self.target_language = language
		self.normalized_result = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'whisper-1',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'flac',
				'mp3',
				'mp4',
				'mpeg',
				'mpga',
				'm4a',
				'ogg',
				'wav',
				'webm',
		]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ru',
				'uk',
				'el',
				'he',
				'ar',
				'hi',
				'zh',
				'ja',
				'ko',
				'vi',
				'th',
		]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		"""Language labels.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Dict[str, str] | None: Return value produced by the operation."""
		return {
				'en': 'English',
				'es': 'Spanish',
				'fr': 'French',
				'de': 'German',
				'it': 'Italian',
				'pt': 'Portuguese',
				'ru': 'Russian',
				'uk': 'Ukrainian',
				'el': 'Greek',
				'he': 'Hebrew',
				'ar': 'Arabic',
				'hi': 'Hindi',
				'zh': 'Chinese',
				'ja': 'Japanese',
				'ko': 'Korean',
				'vi': 'Vietnamese',
				'th': 'Thai',
		}
	
	@property
	def response_format_options( self ) -> List[ str ] | None:
		"""Response format options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'json',
				'text',
				'srt',
				'verbose_json',
				'vtt',
		]
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		Purpose:
		    Performs the Translation.validate_model workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = model if isinstance( model, str ) and model.strip( ) else 'whisper-1'
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported translation model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_format( self, format: str = None ) -> str | None:
		"""Validate format.
		
		Purpose:
		    Performs the Translation.validate_format workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    format (str): Format value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if not isinstance( format, str ) or not format.strip( ):
				return 'json'
			
			value = format.strip( )
			if value not in self.response_format_options:
				return 'json'
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'validate_format( self, format: str=None ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""Normalize response.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    response (Any): Response value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			result: Dict[ str, Any ] = {
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ] = response
				result[ 'raw' ] = response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ] = response.model_dump( )
				except Exception:
					result[ 'raw' ] = str( response )
			else:
				result[ 'raw' ] = str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ] = text
			
			segments = getattr( response, 'segments', None )
			if isinstance( segments, list ):
				normalized_segments = [ ]
				for segment in segments:
					if hasattr( segment, 'model_dump' ):
						normalized_segments.append( segment.model_dump( ) )
					elif isinstance( segment, dict ):
						normalized_segments.append( segment )
					else:
						normalized_segments.append( { 'text': str( segment ) } )
				
				result[ 'segments' ] = normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ] = language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ] = duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ] = '\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ] = str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def translate( self, filepath: str = None, model: str = 'whisper-1', prompt: str = None,
			format: str = None, temperature: float = None, language: str = None,
			**kwargs: Any ) -> str | None:
		"""Translate.
		
		Purpose:
		    Performs the Translation.translate workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    filepath (str): Filepath value used by the operation.
		    model (str): Model value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    format (str): Format value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    language (str): Language value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			file_path = (filepath or kwargs.get( 'path' ) or kwargs.get( 'file_path' )
			             or kwargs.get( 'audio_file' ))
			throw_if( 'file_path', file_path )
			model = str( model or 'whisper-1' ).strip( )
			throw_if( 'model', model )
			api_key = cfg.OPENAI_API_KEY
			if api_key is None or not str( api_key ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			prompt = prompt or kwargs.get( 'instructions' ) or kwargs.get( 'instruct' )
			if isinstance( prompt, str ) and not prompt.strip( ):
				prompt = None
			
			response_format = format or kwargs.get( 'response_format' )
			temperature = temperature if temperature is not None else kwargs.get( 'temperature' )
			self.file_path = file_path
			self.model = self.validate_model( model )
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else None
			self.response_format = self.validate_format( response_format )
			self.temperature = temperature
			self.target_language = language or kwargs.get( 'language' )
			self.client = OpenAI( api_key=api_key )
			request = { 'model': self.model, }
			if self.prompt:
				request[ 'prompt' ] = self.prompt
			
			if self.response_format:
				request[ 'response_format' ] = self.response_format
			
			if self.temperature is not None:
				request[ 'temperature' ] = self.temperature
			
			self.request = request
			
			with open( self.file_path, 'rb' ) as audio_file:
				self.response = self.client.audio.translations.create(
					file=audio_file,
					**self.request )
			
			self.normalized_result = self.normalize_response( self.response )
			return self.normalized_result.get( 'text' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Translation'
			ex.method = 'translate( self, filepath: str=None, **kwargs ) -> str | None'
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Translation.__dir__ workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'store',
				'stream',
				'prompt',
				'response',
				'audio_file',
				'response_format',
				'api_key',
				'client',
				'model',
				'target_language',
				'normalized_result',
				'model_options',
				'mime_options',
				'language_options',
				'language_labels',
				'response_format_options',
				'validate_model',
				'validate_format',
				'normalize_response',
				'translate',
		]

class Embeddings( GPT ):
	"""Embeddings class.
	
	Purpose:
	    Defines the Embeddings component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    client (Optional[OpenAI]): Stores client for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    input (Optional[str | List[str]]): Stores input for the component runtime state.
	    encoding_format (Optional[str]): Stores encoding format for the component runtime state.
	    dimensions (Optional[int]): Stores dimensions for the component runtime state.
	    user (Optional[str]): Stores user for the component runtime state.
	    response (Optional[CreateEmbeddingResponse]): Stores response for the component runtime state.
	    embedding (Optional[List[float] | str]): Stores embedding for the component runtime state.
	    embeddings (Optional[List[List[float]] | List[str]]): Stores embeddings for the component runtime state.
	    usage (Optional[Any]): Stores usage for the component runtime state.
	    request (Optional[Dict[str, Any]]): Stores request for the component runtime state."""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	input: Optional[ str | List[ str ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	user: Optional[ str ]
	response: Optional[ CreateEmbeddingResponse ]
	embedding: Optional[ List[ float ] | str ]
	embeddings: Optional[ List[ List[ float ] ] | List[ str ] ]
	usage: Optional[ Any ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, text: str | List[ str ] = None, model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None, user: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Embeddings object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		    dimensions (int): Dimensions value used by the operation.
		    user (str): User value used by the operation."""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.input = text
		self.encoding_format = format
		self.dimensions = dimensions
		self.user = user
		self.response = None
		self.embedding = None
		self.embeddings = None
		self.usage = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'text-embedding-3-small',
				'text-embedding-3-large',
				'text-embedding-ada-002',
		]
	
	@property
	def encoding_options( self ) -> List[ str ] | None:
		"""Encoding options.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'float',
				'base64',
		]
	
	@property
	def model_default_dimensions( self ) -> Dict[ str, int ]:
		"""Model default dimensions.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Dict[str, int]: Return value produced by the operation."""
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_max_dimensions( self ) -> Dict[ str, int ]:
		"""Model max dimensions.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Dict[str, int]: Return value produced by the operation."""
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_dimension_support( self ) -> Dict[ str, bool ]:
		"""Model dimension support.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Dict[str, bool]: Return value produced by the operation."""
		return {
				'text-embedding-3-small': True,
				'text-embedding-3-large': True,
				'text-embedding-ada-002': False,
		}
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		Purpose:
		    Performs the Embeddings.validate_model workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = model if isinstance( model, str ) and model.strip( ) else \
				'text-embedding-3-small'
			
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported embedding model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_encoding_format( self, format: str = None ) -> str:
		"""Validate encoding format.
		
		Purpose:
		    Performs the Embeddings.validate_encoding_format workflow using the inputs supplied by the
		    caller and the current runtime configuration. The function keeps this behavior isolated so
		    related UI, provider, and data-processing paths can call it consistently.
		
		Args:
		    format (str): Format value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = format if isinstance( format, str ) and format.strip( ) else 'float'
			value = value.strip( ).lower( )
			if value not in self.encoding_options:
				raise ValueError( f'Unsupported embedding encoding format: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_encoding_format( self, format: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_dimensions( self, model: str, dimensions: int = None ) -> int | None:
		"""Validate dimensions.
		
		Purpose:
		    Performs the Embeddings.validate_dimensions workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		    dimensions (int): Dimensions value used by the operation.
		
		Returns:
		    Optional[int]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if dimensions is None:
				return None
			
			try:
				value = int( dimensions )
			except Exception:
				return None
			
			if value <= 0:
				return None
			
			supports_dimensions = self.model_dimension_support.get( model, False )
			if not supports_dimensions:
				return None
			
			max_dimensions = self.get_max_dimensions( model )
			if value > max_dimensions:
				return max_dimensions
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_dimensions( self, model: str, dimensions: int=None )'
			Logger( ).write( exception )
			raise exception
	
	def validate_input( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Validate input.
		
		Purpose:
		    Performs the Embeddings.validate_input workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		
		Returns:
		    str | List[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'text', text )
			
			if isinstance( text, str ):
				value = text.strip( )
				throw_if( 'text', value )
				return value
			
			if isinstance( text, list ):
				values = [ ]
				for item in text:
					if not isinstance( item, str ):
						continue
					
					clean = item.strip( )
					if clean:
						values.append( clean )
				
				throw_if( 'text', values )
				return values
			
			raise ValueError( 'Embedding input must be a string or list of strings.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_input( self, text: str | List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def get_default_dimensions( self, model: str ) -> int:
		"""Get default dimensions.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    int: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return int( self.model_default_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_default_dimensions( self, model: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def get_max_dimensions( self, model: str ) -> int:
		"""Get max dimensions.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    int: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return int( self.model_max_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_max_dimensions( self, model: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def count_tokens( self, text: str, encoding_name: str = 'cl100k_base' ) -> int:
		"""Count tokens.
		
		Purpose:
		    Performs the Embeddings.count_tokens workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    text (str): Text value used by the operation.
		    encoding_name (str): Encoding name value used by the operation.
		
		Returns:
		    int: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if not isinstance( text, str ) or not text:
				return 0
			
			encoding = tiktoken.get_encoding( encoding_name )
			return len( encoding.encode( text ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'count_tokens( self, text: str, encoding_name: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def count_total_tokens( self, text: str | List[ str ],
			encoding_name: str = 'cl100k_base' ) -> int:
		"""Count total tokens.
		
		Purpose:
		    Performs the Embeddings.count_total_tokens workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		    encoding_name (str): Encoding name value used by the operation.
		
		Returns:
		    int: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if isinstance( text, str ):
				return self.count_tokens( text, encoding_name=encoding_name )
			
			if isinstance( text, list ):
				return sum( self.count_tokens( item, encoding_name=encoding_name )
				            for item in text if isinstance( item, str ) )
			
			return 0
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'count_total_tokens( self, text: str | List[ str ] ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def validate_token_limits( self, text: str | List[ str ],
			max_input_tokens: int = 8192, max_total_tokens: int = 300000 ) -> None:
		"""Validate token limits.
		
		Purpose:
		    Performs the Embeddings.validate_token_limits workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		    max_input_tokens (int): Max input tokens value used by the operation.
		    max_total_tokens (int): Max total tokens value used by the operation.
		    
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			values = text if isinstance( text, list ) else [ text ]
			for index, item in enumerate( values ):
				token_count = self.count_tokens( item )
				if token_count > max_input_tokens:
					raise ValueError(
						f'Embedding input item {index + 1} has {token_count} tokens, '
						f'which exceeds the {max_input_tokens} token per-input limit.' )
			
			total_tokens = self.count_total_tokens( text )
			if total_tokens > max_total_tokens:
				raise ValueError(
					f'Embedding request has {total_tokens} total tokens, which exceeds '
					f'the {max_total_tokens} token request limit.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_token_limits( self, text: str | List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def build_request( self, text: str | List[ str ], model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None, user: str = None ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
		    Builds the normalized data structure required by the Embeddings workflow. The function converts
		    caller input, session state, or provider-specific options into a stable shape that downstream
		    API calls and rendering code can consume safely.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		    dimensions (int): Dimensions value used by the operation.
		    user (str): User value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		
		try:
			throw_if( 'text', text )
			self.input_text = text
			
			throw_if( 'model', model )
			self.model = model
			
			self.format = format
			self.dimensions = dimensions
			self.user = user
			
			self.input = self.validate_input( self.input_text )
			self.model = self.validate_model( self.model )
			self.encoding_format = self.validate_encoding_format( self.format )
			self.dimensions = self.validate_dimensions( self.model, self.dimensions )
			self.user = self.user if isinstance( self.user, str ) and self.user.strip( ) else None
			
			self.validate_token_limits( self.input )
			
			self.request = {
					'model': self.model,
					'input': self.input,
					'encoding_format': self.encoding_format,
			}
			
			if self.dimensions is not None:
				self.request[ 'dimensions' ] = self.dimensions
			
			if self.user:
				self.request[ 'user' ] = self.user.strip( )
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'build_request( self, text, model, format, dimensions, user )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None,
			user: str = None ) -> List[ float ] | List[ List[ float ] ] | str | List[ str ] | None:
		"""Create.
		
		Purpose:
		    Performs the Embeddings.create workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		    dimensions (int): Dimensions value used by the operation.
		    user (str): User value used by the operation.
		
		Returns:
		    List[float] | List[List[float]] | str | List[str] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'text', text )
			self.input_text = text
			
			throw_if( 'model', model )
			self.model = model
			
			self.format = format
			self.dimensions = dimensions
			self.user = user
			self.api_key = cfg.OPENAI_API_KEY
			
			if self.api_key is None or not str( self.api_key ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			self.client = OpenAI( api_key=self.api_key )
			self.request = self.build_request(
				text=self.input_text,
				model=self.model,
				format=self.format,
				dimensions=self.dimensions,
				user=self.user
			)
			
			self.response = self.client.embeddings.create( **self.request )
			self.usage = getattr( self.response, 'usage', None )
			self.data = getattr( self.response, 'data', None )
			self.embeddings = [ ]
			
			if self.data is None or len( self.data ) == 0:
				self.embedding = None
				return None
			
			for item in self.data:
				embedding = getattr( item, 'embedding', None )
				if embedding is not None:
					self.embeddings.append( embedding )
			
			if len( self.embeddings ) == 0:
				self.embedding = None
				return None
			
			self.embedding = self.embeddings[ 0 ]
			
			if isinstance( self.input, str ):
				return self.embedding
			
			return self.embeddings
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'create( self, text, model, format, dimensions, user )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Embeddings.__dir__ workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'client',
				'model',
				'input',
				'encoding_format',
				'dimensions',
				'user',
				'response',
				'embedding',
				'embeddings',
				'usage',
				'request',
				'model_options',
				'encoding_options',
				'model_default_dimensions',
				'model_max_dimensions',
				'model_dimension_support',
				'validate_model',
				'validate_encoding_format',
				'validate_dimensions',
				'validate_input',
				'get_default_dimensions',
				'get_max_dimensions',
				'count_tokens',
				'count_total_tokens',
				'validate_token_limits',
				'build_request',
				'create',
		]

class Files( GPT ):
	"""Files class.
	
	Purpose:
	    Defines the Files component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    client (Optional[OpenAI]): Stores client for the component runtime state.
	    file (Optional[Any]): Stores file for the component runtime state.
	    file_id (Optional[str]): Stores file id for the component runtime state.
	    filepath (Optional[str]): Stores filepath for the component runtime state.
	    filename (Optional[str]): Stores filename for the component runtime state.
	    purpose (Optional[str]): Stores purpose for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    content (Optional[str | bytes | Dict[str, Any]]): Stores content for the component runtime state.
	    files (Optional[List[Dict[str, Any]]]): Stores files for the component runtime state.
	    request (Optional[Dict[str, Any]]): Stores request for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    output_text (Optional[str]): Stores output text for the component runtime state."""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	file: Optional[ Any ]
	file_id: Optional[ str ]
	filepath: Optional[ str ]
	filename: Optional[ str ]
	purpose: Optional[ str ]
	response: Optional[ Any ]
	content: Optional[ str | bytes | Dict[ str, Any ] ]
	files: Optional[ List[ Dict[ str, Any ] ] ]
	request: Optional[ Dict[ str, Any ] ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	output_text: Optional[ str ]
	
	def __init__( self, id: str = None, filepath: str = None, purpose: str = 'user_data',
			model: str = 'gpt-4o-mini', prompt: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Files object with its default configuration, runtime state, provider settings,
		    and compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment.
		
		Args:
		    id (str): Id value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    purpose (str): Purpose value used by the operation.
		    model (str): Model value used by the operation.
		    prompt (str): Prompt value used by the operation."""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.file = None
		self.file_id = id
		self.filepath = filepath
		self.filename = None
		self.purpose = purpose
		self.response = None
		self.content = None
		self.files = [ ]
		self.request = None
		self.model = model
		self.prompt = prompt
		self.output_text = None
	
	@property
	def upload_purpose_options( self ) -> List[ str ] | None:
		"""Upload purpose options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'assistants',
				'batch',
				'fine-tune',
				'vision',
				'user_data',
				'evals',
		]
	
	@property
	def file_purpose_options( self ) -> List[ str ] | None:
		"""File purpose options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'assistants',
				'assistants_output',
				'batch',
				'batch_output',
				'fine-tune',
				'fine-tune-results',
				'vision',
				'user_data',
				'evals',
		]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		"""Purpose options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return self.upload_purpose_options
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'none',
				'minimal',
				'low',
				'medium',
				'high',
		]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'file_search_call.results',
				'web_search_call.results',
				'web_search_call.action.sources',
				'code_interpreter_call.outputs',
				'reasoning.encrypted_content',
				'message.output_text.logprobs',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'web_search',
				'file_search',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'text',
		]
	
	def validate_upload_purpose( self, purpose: str = None ) -> str:
		"""Validate upload purpose.
		
		Purpose:
		    Performs the Files.validate_upload_purpose workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    purpose (str): Purpose value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = purpose if isinstance( purpose, str ) and purpose.strip( ) else 'user_data'
			value = value.strip( )
			
			if value not in self.upload_purpose_options:
				raise ValueError( f'Unsupported upload purpose: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'validate_upload_purpose( self, purpose: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_file_id( self, id: str = None ) -> str:
		"""Validate file id.
		
		Purpose:
		    Performs the Files.validate_file_id workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = id if isinstance( id, str ) and id.strip( ) else self.file_id
			throw_if( 'id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'validate_file_id( self, id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_object( self, file: Any ) -> Dict[ str, Any ]:
		"""Normalize file object.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    file (Any): File value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if file is None:
				return { }
			
			if isinstance( file, dict ):
				source = file
			elif hasattr( file, 'model_dump' ):
				source = file.model_dump( )
			else:
				source = {
						'id': getattr( file, 'id', None ),
						'bytes': getattr( file, 'bytes', None ),
						'created_at': getattr( file, 'created_at', None ),
						'expires_at': getattr( file, 'expires_at', None ),
						'filename': getattr( file, 'filename', None ),
						'object': getattr( file, 'object', None ),
						'purpose': getattr( file, 'purpose', None ),
						'status': getattr( file, 'status', None ),
						'status_details': getattr( file, 'status_details', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'filename': source.get( 'filename' ),
					'purpose': source.get( 'purpose' ),
					'bytes': source.get( 'bytes' ),
					'created_at': source.get( 'created_at' ),
					'expires_at': source.get( 'expires_at' ),
					'object': source.get( 'object' ),
					'status': source.get( 'status' ),
					'status_details': source.get( 'status_details' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_object( self, file: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_list( self, response: Any, purpose: str = None ) -> List[ Dict[ str, Any ] ]:
		"""Normalize file list.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    response (Any): Response value used by the operation.
		    purpose (str): Purpose value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if response is None:
				return [ ]
			
			if isinstance( response, list ):
				items = response
			elif isinstance( response, dict ):
				items = response.get( 'data', [ ] )
			else:
				items = getattr( response, 'data', [ ] )
			
			rows: List[ Dict[ str, Any ] ] = [ ]
			for item in items:
				row = self.normalize_file_object( item )
				
				if not row.get( 'id' ):
					continue
				
				if isinstance( purpose, str ) and purpose.strip( ):
					if row.get( 'purpose' ) != purpose.strip( ):
						continue
				
				rows.append( row )
			
			return rows
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_list( self, response: Any, purpose: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_content( self, content: Any ) -> str | bytes | Dict[ str, Any ] | None:
		"""Normalize file content.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    content (Any): Content value used by the operation.
		
		Returns:
		    str | bytes | Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if content is None:
				return None
			
			if isinstance( content, (str, bytes) ):
				return content
			
			if hasattr( content, 'read' ):
				value = content.read( )
				if isinstance( value, bytes ):
					try:
						return value.decode( 'utf-8' )
					except Exception:
						return value
				
				return value
			
			if hasattr( content, 'text' ):
				value = getattr( content, 'text' )
				if isinstance( value, str ):
					return value
			
			if hasattr( content, 'content' ):
				value = getattr( content, 'content' )
				if isinstance( value, bytes ):
					try:
						return value.decode( 'utf-8' )
					except Exception:
						return value
				
				return value
			
			if hasattr( content, 'model_dump' ):
				return content.model_dump( )
			
			return str( content )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_content( self, content: Any )'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, filepath: str, purpose: str = 'user_data' ) -> Dict[ str, Any ] | None:
		"""Upload.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    filepath (str): Filepath value used by the operation.
		    purpose (str): Purpose value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'filepath', filepath )
			
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.filepath = filepath
			self.purpose = self.validate_upload_purpose( purpose )
			self.request = {
					'file': filepath,
					'purpose': self.purpose,
			}
			
			with open( filepath, 'rb' ) as source:
				self.response = self.client.files.create(
					file=source,
					purpose=self.purpose )
			
			self.file = self.response
			metadata = self.normalize_file_object( self.response )
			self.file_id = metadata.get( 'id' )
			self.filename = metadata.get( 'filename' )
			return metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'upload( self, filepath: str, purpose: str )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, purpose: str = None ) -> List[ Dict[ str, Any ] ]:
		"""List.
		
		Purpose:
		    Performs the Files.list workflow using the inputs supplied by the caller and the current runtime
		    configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    purpose (str): Purpose value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.purpose = purpose if isinstance( purpose, str ) and purpose.strip( ) else None
			self.request = { }
			
			if self.purpose:
				self.request[ 'purpose_filter' ] = self.purpose
			
			self.response = self.client.files.list( )
			self.files = self.normalize_file_list( self.response, purpose=self.purpose )
			return self.files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'list( self, purpose: str=None ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve.
		
		Purpose:
		    Performs the Files.retrieve workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.file_id = self.validate_file_id( id )
			self.request = {
					'file_id': self.file_id,
			}
			
			self.response = self.client.files.retrieve( file_id=self.file_id )
			self.file = self.response
			metadata = self.normalize_file_object( self.response )
			self.filename = metadata.get( 'filename' )
			return metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'retrieve( self, id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def extract( self, id: str ) -> str | bytes | Dict[ str, Any ] | None:
		"""Extract.
		
		Purpose:
		    Performs the Files.extract workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		
		Returns:
		    str | bytes | Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.file_id = self.validate_file_id( id )
			self.request = {
					'file_id': self.file_id,
			}
			
			self.response = self.client.files.content( file_id=self.file_id )
			self.content = self.normalize_file_content( self.response )
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'extract( self, id: str )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, id: str ) -> Dict[ str, Any ] | None:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    id (str): Id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.file_id = self.validate_file_id( id )
			self.request = {
					'file_id': self.file_id,
			}
			
			self.response = self.client.files.delete( file_id=self.file_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.file_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'delete( self, id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def summarize( self, id: str, prompt: str = None, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""Summarize.
		
		Purpose:
		    Performs the Files.summarize workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    max_chars (int): Max chars value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_id = self.validate_file_id( id )
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else \
				'Summarize the selected file content.'
			self.model = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini'
			
			content = self.extract( self.file_id )
			if isinstance( content, bytes ):
				try:
					content_text = content.decode( 'utf-8' )
				except Exception:
					content_text = str( content )
			elif isinstance( content, dict ):
				content_text = str( content )
			else:
				content_text = content if isinstance( content, str ) else ''
			
			throw_if( 'content_text', content_text )
			content_text = content_text[ :max_chars ] if isinstance( max_chars,
				int ) else content_text
			
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = {
					'model': self.model,
					'input': [
							{
									'role': 'user',
									'content': [
											{
													'type': 'input_text',
													'text': f'{self.prompt}\n\nFile ID: {self.file_id}\n\n{content_text}',
											}, ],
							}, ],
			}
			
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', None )
			
			if self.output_text:
				return self.output_text
			
			return str( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'summarize( self, id: str, prompt: str=None ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def search( self, id: str, query: str, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""Search.
		
		Purpose:
		    Performs the Files.search workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		    query (str): Query value used by the operation.
		    model (str): Model value used by the operation.
		    max_chars (int): Max chars value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'query', query )
			prompt = (
					'Answer the user question using only the selected file content when possible. '
					f'Question: {query}'
			)
			
			return self.summarize( id=id, prompt=prompt, model=model, max_chars=max_chars )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'search( self, id: str, query: str ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, id: str, max_chars: int = 4000 ) -> Dict[ str, Any ]:
		"""Survey.
		
		Purpose:
		    Performs the Files.survey workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		    max_chars (int): Max chars value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_id = self.validate_file_id( id )
			metadata = self.retrieve( self.file_id )
			content = self.extract( self.file_id )
			
			if isinstance( content, bytes ):
				try:
					content_text = content.decode( 'utf-8' )
				except Exception:
					content_text = str( content )
			elif isinstance( content, dict ):
				content_text = str( content )
			else:
				content_text = content if isinstance( content, str ) else ''
			
			preview = content_text[ :max_chars ] if isinstance( max_chars, int ) else content_text
			
			return {
					'metadata': metadata,
					'preview': preview,
					'file_id': self.file_id,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'survey( self, id: str ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Files.__dir__ workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'client',
				'file',
				'file_id',
				'filepath',
				'filename',
				'purpose',
				'response',
				'content',
				'files',
				'request',
				'model',
				'prompt',
				'output_text',
				'upload_purpose_options',
				'file_purpose_options',
				'purpose_options',
				'model_options',
				'validate_upload_purpose',
				'validate_file_id',
				'normalize_file_object',
				'normalize_file_list',
				'normalize_file_content',
				'upload',
				'list',
				'retrieve',
				'extract',
				'delete',
				'summarize',
				'search',
				'survey',
		]

class VectorStores( GPT ):
	"""VectorStores class.
	
	Purpose:
	    Defines the VectorStores component used by the Boo application. The class groups related
	    provider configuration, runtime state, helper methods, and API-facing behavior so Streamlit
	    workflows can call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    client (Optional[OpenAI]): Stores client for the component runtime state.
	    name (Optional[str]): Stores name for the component runtime state.
	    description (Optional[str]): Stores description for the component runtime state.
	    store_id (Optional[str]): Stores store id for the component runtime state.
	    file_id (Optional[str]): Stores file id for the component runtime state.
	    batch_id (Optional[str]): Stores batch id for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    vector_store (Optional[Dict[str, Any]]): Stores vector store for the component runtime state.
	    vector_stores (Optional[List[Dict[str, Any]]]): Stores vector stores for the component runtime state.
	    vector_file (Optional[Dict[str, Any]]): Stores vector file for the component runtime state.
	    vector_files (Optional[List[Dict[str, Any]]]): Stores vector files for the component runtime state.
	    file_batch (Optional[Dict[str, Any]]): Stores file batch for the component runtime state.
	    search_results (Optional[List[Dict[str, Any]]]): Stores search results for the component runtime state.
	    output_text (Optional[str]): Stores output text for the component runtime state.
	    request (Optional[Dict[str, Any]]): Stores request for the component runtime state.
	    collections (Optional[Dict[str, str]]): Stores collections for the component runtime state.
	    max_search_results (Optional[int]): Stores max search results for the component runtime state."""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	name: Optional[ str ]
	description: Optional[ str ]
	store_id: Optional[ str ]
	file_id: Optional[ str ]
	batch_id: Optional[ str ]
	response: Optional[ Any ]
	vector_store: Optional[ Dict[ str, Any ] ]
	vector_stores: Optional[ List[ Dict[ str, Any ] ] ]
	vector_file: Optional[ Dict[ str, Any ] ]
	vector_files: Optional[ List[ Dict[ str, Any ] ] ]
	file_batch: Optional[ Dict[ str, Any ] ]
	search_results: Optional[ List[ Dict[ str, Any ] ] ]
	output_text: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	collections: Optional[ Dict[ str, str ] ]
	max_search_results: Optional[ int ]
	
	def __init__( self, name: str = None, store_id: str = None, file_id: str = None,
			model: str = 'gpt-4o-mini', max_search_results: int = 10 ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the VectorStores object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment.
		
		Args:
		    name (str): Name value used by the operation.
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		    model (str): Model value used by the operation.
		    max_search_results (int): Max search results value used by the operation."""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.name = name
		self.description = None
		self.store_id = store_id
		self.file_id = file_id
		self.batch_id = None
		self.model = model
		self.response = None
		self.vector_store = None
		self.vector_stores = [ ]
		self.vector_file = None
		self.vector_files = [ ]
		self.file_batch = None
		self.search_results = [ ]
		self.output_text = None
		self.request = None
		self.max_search_results = max_search_results
		self.collections = {
				'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	@property
	def ranker_options( self ) -> List[ str ] | None:
		"""Ranker options.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
				'default-2024-11-15',
		]
	
	@property
	def chunking_strategy_options( self ) -> List[ str ] | None:
		"""Chunking strategy options.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
				'static',
		]
	
	def validate_store_name( self, name: str = None ) -> str:
		"""Validate store name.
		
		Purpose:
		    Performs the VectorStores.validate_store_name workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    name (str): Name value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = name if isinstance( name, str ) and name.strip( ) else self.name
			throw_if( 'name', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_store_name( self, name: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_store_id( self, store_id: str = None ) -> str:
		"""Validate store id.
		
		Purpose:
		    Performs the VectorStores.validate_store_id workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = store_id if isinstance( store_id, str ) and store_id.strip( ) else self.store_id
			throw_if( 'store_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_store_id( self, store_id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_file_id( self, file_id: str = None ) -> str:
		"""Validate file id.
		
		Purpose:
		    Performs the VectorStores.validate_file_id workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = file_id if isinstance( file_id, str ) and file_id.strip( ) else self.file_id
			throw_if( 'file_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_file_id( self, file_id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_batch_id( self, batch_id: str = None ) -> str:
		"""Validate batch id.
		
		Purpose:
		    Performs the VectorStores.validate_batch_id workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    batch_id (str): Batch id value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = batch_id if isinstance( batch_id, str ) and batch_id.strip( ) else self.batch_id
			throw_if( 'batch_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_batch_id( self, batch_id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_file_ids( self, file_ids: List[ str ] = None ) -> List[ str ]:
		"""Validate file ids.
		
		Purpose:
		    Performs the VectorStores.validate_file_ids workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    file_ids (List[str]): File ids value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if file_ids is None:
				return [ ]
			
			values = [ ]
			for item in file_ids:
				if isinstance( item, str ) and item.strip( ):
					values.append( item.strip( ) )
			
			return values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_file_ids( self, file_ids: List[ str ]=None )'
			Logger( ).write( exception )
			raise exception
	
	def validate_max_num_results( self, max_num_results: int = None ) -> int:
		"""Validate max num results.
		
		Purpose:
		    Performs the VectorStores.validate_max_num_results workflow using the inputs supplied by the
		    caller and the current runtime configuration. The function keeps this behavior isolated so
		    related UI, provider, and data-processing paths can call it consistently.
		
		Args:
		    max_num_results (int): Max num results value used by the operation.
		
		Returns:
		    int: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = self.max_search_results if max_num_results is None else int( max_num_results )
			
			if value < 1:
				return 1
			
			if value > 50:
				return 50
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_max_num_results( self, max_num_results: int=None )'
			Logger( ).write( exception )
			raise exception
	
	def build_expires_after( self, anchor: str = None, days: int = None ) -> Dict[
		                                                                         str, Any ] | None:
		"""Build expires after.
		
		Purpose:
		    Builds the normalized data structure required by the VectorStores workflow. The function
		    converts caller input, session state, or provider-specific options into a stable shape that
		    downstream API calls and rendering code can consume safely.
		
		Args:
		    anchor (str): Anchor value used by the operation.
		    days (int): Days value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if days is None:
				return None
			
			value = int( days )
			if value <= 0:
				return None
			
			anchor_value = anchor if isinstance( anchor,
				str ) and anchor.strip( ) else 'last_active_at'
			
			return {
					'anchor': anchor_value.strip( ),
					'days': value,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'build_expires_after( self, anchor: str=None, days: int=None )'
			Logger( ).write( exception )
			raise exception
	
	def build_chunking_strategy( self, strategy: str = 'auto', max_chunk_size_tokens: int = None,
			chunk_overlap_tokens: int = None ) -> Dict[ str, Any ] | None:
		"""Build chunking strategy.
		
		Purpose:
		    Builds the normalized data structure required by the VectorStores workflow. The function
		    converts caller input, session state, or provider-specific options into a stable shape that
		    downstream API calls and rendering code can consume safely.
		
		Args:
		    strategy (str): Strategy value used by the operation.
		    max_chunk_size_tokens (int): Max chunk size tokens value used by the operation.
		    chunk_overlap_tokens (int): Chunk overlap tokens value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			strategy_value = strategy if isinstance( strategy,
				str ) and strategy.strip( ) else 'auto'
			strategy_value = strategy_value.strip( )
			
			if strategy_value == 'auto':
				return { 'type': 'auto', }
			
			if strategy_value != 'static':
				return None
			
			max_value = 800 if max_chunk_size_tokens is None else int( max_chunk_size_tokens )
			overlap_value = 400 if chunk_overlap_tokens is None else int( chunk_overlap_tokens )
			
			if max_value < 100:
				max_value = 100
			
			if max_value > 4096:
				max_value = 4096
			
			if overlap_value < 0:
				overlap_value = 0
			
			if overlap_value > max_value // 2:
				overlap_value = max_value // 2
			
			return {
					'type': 'static',
					'static': {
							'max_chunk_size_tokens': max_value,
							'chunk_overlap_tokens': overlap_value,
					},
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'build_chunking_strategy( self, strategy: str, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_vector_store( self, store: Any ) -> Dict[ str, Any ]:
		"""Normalize vector store.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    store (Any): Store value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if store is None:
				return { }
			
			if isinstance( store, dict ):
				source = store
			elif hasattr( store, 'model_dump' ):
				source = store.model_dump( )
			else:
				source = {
						'id': getattr( store, 'id', None ),
						'name': getattr( store, 'name', None ),
						'description': getattr( store, 'description', None ),
						'created_at': getattr( store, 'created_at', None ),
						'object': getattr( store, 'object', None ),
						'usage_bytes': getattr( store, 'usage_bytes', None ),
						'file_counts': getattr( store, 'file_counts', None ),
						'status': getattr( store, 'status', None ),
						'expires_after': getattr( store, 'expires_after', None ),
						'expires_at': getattr( store, 'expires_at', None ),
						'last_active_at': getattr( store, 'last_active_at', None ),
						'metadata': getattr( store, 'metadata', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'name': source.get( 'name' ),
					'description': source.get( 'description' ),
					'created_at': source.get( 'created_at' ),
					'object': source.get( 'object' ),
					'usage_bytes': source.get( 'usage_bytes' ),
					'file_counts': source.get( 'file_counts' ),
					'status': source.get( 'status' ),
					'expires_after': source.get( 'expires_after' ),
					'expires_at': source.get( 'expires_at' ),
					'last_active_at': source.get( 'last_active_at' ),
					'metadata': source.get( 'metadata' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_vector_store( self, store: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_vector_store_file( self, file: Any ) -> Dict[ str, Any ]:
		"""Normalize vector store file.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    file (Any): File value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if file is None:
				return { }
			
			if isinstance( file, dict ):
				source = file
			elif hasattr( file, 'model_dump' ):
				source = file.model_dump( )
			else:
				source = {
						'id': getattr( file, 'id', None ),
						'object': getattr( file, 'object', None ),
						'created_at': getattr( file, 'created_at', None ),
						'vector_store_id': getattr( file, 'vector_store_id', None ),
						'status': getattr( file, 'status', None ),
						'last_error': getattr( file, 'last_error', None ),
						'chunking_strategy': getattr( file, 'chunking_strategy', None ),
						'attributes': getattr( file, 'attributes', None ),
						'usage_bytes': getattr( file, 'usage_bytes', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'object': source.get( 'object' ),
					'created_at': source.get( 'created_at' ),
					'vector_store_id': source.get( 'vector_store_id' ),
					'status': source.get( 'status' ),
					'last_error': source.get( 'last_error' ),
					'chunking_strategy': source.get( 'chunking_strategy' ),
					'attributes': source.get( 'attributes' ),
					'usage_bytes': source.get( 'usage_bytes' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_vector_store_file( self, file: Any )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_batch( self, batch: Any ) -> Dict[ str, Any ]:
		"""Normalize file batch.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    batch (Any): Batch value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if batch is None:
				return { }
			
			if isinstance( batch, dict ):
				source = batch
			elif hasattr( batch, 'model_dump' ):
				source = batch.model_dump( )
			else:
				source = {
						'id': getattr( batch, 'id', None ),
						'object': getattr( batch, 'object', None ),
						'created_at': getattr( batch, 'created_at', None ),
						'vector_store_id': getattr( batch, 'vector_store_id', None ),
						'status': getattr( batch, 'status', None ),
						'file_counts': getattr( batch, 'file_counts', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'object': source.get( 'object' ),
					'created_at': source.get( 'created_at' ),
					'vector_store_id': source.get( 'vector_store_id' ),
					'status': source.get( 'status' ),
					'file_counts': source.get( 'file_counts' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_file_batch( self, batch: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_search_results( self, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""Normalize search results.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    response (Any): Response value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if response is None:
				return [ ]
			
			if isinstance( response, dict ):
				items = response.get( 'data', [ ] )
			elif isinstance( response, list ):
				items = response
			else:
				items = getattr( response, 'data', [ ] )
			
			rows: List[ Dict[ str, Any ] ] = [ ]
			for item in items:
				if isinstance( item, dict ):
					source = item
				elif hasattr( item, 'model_dump' ):
					source = item.model_dump( )
				else:
					source = {
							'file_id': getattr( item, 'file_id', None ),
							'filename': getattr( item, 'filename', None ),
							'score': getattr( item, 'score', None ),
							'attributes': getattr( item, 'attributes', None ),
							'content': getattr( item, 'content', None ),
					}
				
				rows.append( {
						'file_id': source.get( 'file_id' ),
						'filename': source.get( 'filename' ),
						'score': source.get( 'score' ),
						'attributes': source.get( 'attributes' ),
						'content': source.get( 'content' ),
				} )
			
			return rows
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_search_results( self, response: Any )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str, description: str = None, metadata: Dict[ str, Any ] = None,
			expires_after: Dict[ str, Any ] = None, file_ids: List[ str ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Create.
		
		Purpose:
		    Performs the VectorStores.create workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    name (str): Name value used by the operation.
		    description (str): Description value used by the operation.
		    metadata (Dict[str, Any]): Metadata value used by the operation.
		    expires_after (Dict[str, Any]): Expires after value used by the operation.
		    file_ids (List[str]): File ids value used by the operation.
		    chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.name = self.validate_store_name( name )
			self.description = description if isinstance( description,
				str ) and description.strip( ) else None
			
			self.request = { 'name': self.name, }
			
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				self.request[ 'metadata' ] = dict( metadata )
			
			if self.description:
				if 'metadata' not in self.request:
					self.request[ 'metadata' ] = { }
				
				self.request[ 'metadata' ][ 'description' ] = self.description
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ] = expires_after
			
			clean_file_ids = self.validate_file_ids( file_ids )
			if len( clean_file_ids ) > 0:
				self.request[ 'file_ids' ] = clean_file_ids
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.create( **self.request )
			self.vector_store = self.normalize_vector_store( self.response )
			
			if self.description:
				self.vector_store[ 'description' ] = self.description
			
			self.store_id = self.vector_store.get( 'id' )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create( self, name: str, **kwargs ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def update( self, store_id: str, name: str = None, description: str = None,
			metadata: Dict[ str, Any ] = None,
			expires_after: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Update.
		
		Purpose:
		    Performs the VectorStores.update workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    name (str): Name value used by the operation.
		    description (str): Description value used by the operation.
		    metadata (Dict[str, Any]): Metadata value used by the operation.
		    expires_after (Dict[str, Any]): Expires after value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.description = description if isinstance( description,
				str ) and description.strip( ) else None
			self.request = { }
			
			if isinstance( name, str ) and name.strip( ):
				self.request[ 'name' ] = name.strip( )
			
			if isinstance( metadata, dict ):
				self.request[ 'metadata' ] = dict( metadata )
			
			if self.description:
				if 'metadata' not in self.request:
					self.request[ 'metadata' ] = { }
				
				self.request[ 'metadata' ][ 'description' ] = self.description
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ] = expires_after
			
			if len( self.request ) == 0:
				return self.retrieve( self.store_id )
			
			self.response = self.client.vector_stores.update(
				vector_store_id=self.store_id,
				**self.request )
			
			self.vector_store = self.normalize_vector_store( self.response )
			
			if self.description:
				self.vector_store[ 'description' ] = self.description
			
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update( self, store_id: str, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list_stores( self, limit: int = 100, order: str = 'desc',
			after: str = None, before: str = None ) -> List[ Dict[ str, Any ] ]:
		"""List stores.
		
		Purpose:
		    Performs the VectorStores.list_stores workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    limit (int): Limit value used by the operation.
		    order (str): Order value used by the operation.
		    after (str): After value used by the operation.
		    before (str): Before value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = {
					'limit': limit,
					'order': order,
			}
			
			if isinstance( after, str ) and after.strip( ):
				self.request[ 'after' ] = after.strip( )
			
			if isinstance( before, str ) and before.strip( ):
				self.request[ 'before' ] = before.strip( )
			
			self.response = self.client.vector_stores.list( **self.request )
			items = getattr( self.response, 'data', [ ] )
			self.vector_stores = [ self.normalize_vector_store( item ) for item in items ]
			return self.vector_stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_stores( self, limit: int=100 )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve.
		
		Purpose:
		    Performs the VectorStores.retrieve workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'vector_store_id': self.store_id,
			}
			
			self.response = self.client.vector_stores.retrieve(
				vector_store_id=self.store_id )
			self.vector_store = self.normalize_vector_store( self.response )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve( self, store_id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'vector_store_id': self.store_id,
			}
			
			self.response = self.client.vector_stores.delete(
				vector_store_id=self.store_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.store_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'delete( self, store_id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def attach_file( self, store_id: str, file_id: str, attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Attach file.
		
		Purpose:
		    Performs the VectorStores.attach_file workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		    attributes (Dict[str, Any]): Attributes value used by the operation.
		    chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			self.request = {
					'file_id': self.file_id,
			}
			
			if isinstance( attributes, dict ) and len( attributes ) > 0:
				self.request[ 'attributes' ] = attributes
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.files.create(
				vector_store_id=self.store_id,
				**self.request )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'attach_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def upload_file( self, store_id: str, path: str, file_path: str = None,
			purpose: str = 'assistants', attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Upload file.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    path (str): Path value used by the operation.
		    file_path (str): File path value used by the operation.
		    purpose (str): Purpose value used by the operation.
		    attributes (Dict[str, Any]): Attributes value used by the operation.
		    chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		
		source = None
		try:
			throw_if( 'store_id', store_id )
			self.store_id = self.validate_store_id( store_id )
			
			throw_if( 'path', path )
			self.file_path = path
			
			throw_if( 'purpose', purpose )
			self.purpose = purpose
			
			self.api_key = cfg.OPENAI_API_KEY
			if self.api_key is None or not str( self.api_key ).strip( ):
				raise ValueError( 'OPENAI_API_KEY is required.' )
			
			self.client = OpenAI( api_key=self.api_key )
			source = open( self.file_path, 'rb' )
			self.file = self.client.files.create(
				file=source,
				purpose=self.purpose
			)
			
			self.uploaded_file = (
					self.file.model_dump( )
					if hasattr( self.file, 'model_dump' )
					else dict( self.file ) if isinstance( self.file, dict ) else {
							'id': getattr( self.file, 'id', None ),
							'object': getattr( self.file, 'object', None ),
							'bytes': getattr( self.file, 'bytes', None ),
							'created_at': getattr( self.file, 'created_at', None ),
							'filename': getattr( self.file, 'filename',
								Path( self.file_path ).name ),
							'purpose': getattr( self.file, 'purpose', self.purpose ),
					}
			)
			
			self.file_id = self.uploaded_file.get( 'id' )
			if self.file_id is None or not str( self.file_id ).strip( ):
				raise ValueError( 'OpenAI did not return a file ID for the uploaded file.' )
			
			self.vector_file = self.attach_file(
				store_id=self.store_id,
				file_id=self.file_id,
				attributes=attributes,
				chunking_strategy=chunking_strategy
			)
			
			self.result = dict( self.vector_file or { } )
			self.result[ 'uploaded_file_id' ] = self.file_id
			self.result[ 'uploaded_filename' ] = self.uploaded_file.get(
				'filename', Path( self.file_path ).name )
			self.result[ 'uploaded_file' ] = self.uploaded_file
			self.result[ 'vector_store_id' ] = self.store_id
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'upload_file( self, store_id, path, file_path, purpose, attributes, chunking_strategy )'
			Logger( ).write( exception )
			raise exception
		finally:
			if source is not None:
				source.close( )
	
	def upload( self, store_id: str, path: str, file_path: str = None,
			purpose: str = 'assistants', attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Upload.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    path (str): Path value used by the operation.
		    file_path (str): File path value used by the operation.
		    purpose (str): Purpose value used by the operation.
		    attributes (Dict[str, Any]): Attributes value used by the operation.
		    chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			
			throw_if( 'path', path )
			self.file_path = path
			
			return self.upload_file(
				store_id=self.store_id,
				path=self.file_path,
				file_path=file_path,
				purpose=purpose,
				attributes=attributes,
				chunking_strategy=chunking_strategy
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'upload( self, store_id, path, file_path, purpose, attributes, chunking_strategy )'
			Logger( ).write( exception )
			raise exception
	
	def files_upload( self, store_id: str, path: str, file_path: str = None,
			purpose: str = 'assistants', attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Files upload.
		
		Purpose:
		    Performs the VectorStores.files_upload workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    path (str): Path value used by the operation.
		    file_path (str): File path value used by the operation.
		    purpose (str): Purpose value used by the operation.
		    attributes (Dict[str, Any]): Attributes value used by the operation.
		    chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			
			throw_if( 'path', path )
			self.file_path = path
			
			return self.upload_file(
				store_id=self.store_id,
				path=self.file_path,
				file_path=file_path,
				purpose=purpose,
				attributes=attributes,
				chunking_strategy=chunking_strategy
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'files_upload( self, store_id, path, file_path, purpose, attributes, chunking_strategy )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, store_id: str, limit: int = 100, order: str = 'desc' ) -> List[
		Dict[ str, Any ] ]:
		"""List.
		
		Purpose:
		    Performs the VectorStores.list workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    limit (int): Limit value used by the operation.
		    order (str): Order value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.list_files( store_id=store_id, limit=limit, order=order )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list( self, store_id: str ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def list_files( self, store_id: str, limit: int = 100, order: str = 'desc' ) -> List[
		Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
		    Performs the VectorStores.list_files workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    limit (int): Limit value used by the operation.
		    order (str): Order value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'limit': limit,
					'order': order,
			}
			
			self.response = self.client.vector_stores.files.list(
				vector_store_id=self.store_id,
				**self.request )
			
			items = getattr( self.response, 'data', [ ] )
			self.vector_files = [ self.normalize_vector_store_file( item ) for item in items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_files( self, store_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve file.
		
		Purpose:
		    Performs the VectorStores.retrieve_file workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.retrieve(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def update_file( self, store_id: str, file_id: str,
			attributes: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Update file.
		
		Purpose:
		    Performs the VectorStores.update_file workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		    attributes (Dict[str, Any]): Attributes value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			self.request = { }
			
			if isinstance( attributes, dict ):
				self.request[ 'attributes' ] = attributes
			
			self.response = self.client.vector_stores.files.update(
				vector_store_id=self.store_id,
				file_id=self.file_id,
				**self.request )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def delete_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""Delete file.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.delete(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.file_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'delete_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_content( self, store_id: str, file_id: str ) -> Any:
		"""Retrieve file content.
		
		Purpose:
		    Performs the VectorStores.retrieve_file_content workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.content(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_content( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def create_file_batch( self, store_id: str, file_ids: List[ str ],
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Create file batch.
		
		Purpose:
		    Creates the requested resource, connection, schema object, or user interface artifact using
		    validated inputs. The function encapsulates setup details so callers can rely on a consistent
		    resource lifecycle.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_ids (List[str]): File ids value used by the operation.
		    attributes (Dict[str, Any]): Attributes value used by the operation.
		    chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			clean_file_ids = self.validate_file_ids( file_ids )
			throw_if( 'file_ids', clean_file_ids )
			
			if len( clean_file_ids ) > 2000:
				raise ValueError( 'Vector store file batches cannot exceed 2000 files.' )
			
			self.request = {
					'file_ids': clean_file_ids,
			}
			
			if isinstance( attributes, dict ) and len( attributes ) > 0:
				self.request[ 'attributes' ] = attributes
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.file_batches.create(
				vector_store_id=self.store_id,
				**self.request )
			
			self.file_batch = self.normalize_file_batch( self.response )
			self.batch_id = self.file_batch.get( 'id' )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create_file_batch( self, store_id: str, file_ids: List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve file batch.
		
		Purpose:
		    Performs the VectorStores.retrieve_file_batch workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    batch_id (str): Batch id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.retrieve(
				vector_store_id=self.store_id,
				batch_id=self.batch_id )
			
			self.file_batch = self.normalize_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_batch( self, store_id: str, batch_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def list_file_batch_files( self, store_id: str, batch_id: str, limit: int = 100 ) -> List[
		Dict[ str, Any ] ]:
		"""List file batch files.
		
		Purpose:
		    Performs the VectorStores.list_file_batch_files workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    batch_id (str): Batch id value used by the operation.
		    limit (int): Limit value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.files.list(
				vector_store_id=self.store_id,
				batch_id=self.batch_id,
				limit=limit )
			
			items = getattr( self.response, 'data', [ ] )
			self.vector_files = [ self.normalize_vector_store_file( item ) for item in items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_file_batch_files( self, store_id: str, batch_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def cancel_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""Cancel file batch.
		
		Purpose:
		    Performs the VectorStores.cancel_file_batch workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    batch_id (str): Batch id value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.cancel(
				vector_store_id=self.store_id,
				batch_id=self.batch_id )
			
			self.file_batch = self.normalize_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'cancel_file_batch( self, store_id: str, batch_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def search( self, store_id: str, query: str, max_num_results: int = 10,
			filters: Dict[ str, Any ] = None, ranking_options: Dict[ str, Any ] = None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""Search.
		
		Purpose:
		    Performs the VectorStores.search workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    query (str): Query value used by the operation.
		    max_num_results (int): Max num results value used by the operation.
		    filters (Dict[str, Any]): Filters value used by the operation.
		    ranking_options (Dict[str, Any]): Ranking options value used by the operation.
		    rewrite_query (bool): Rewrite query value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.search_store( store_id=store_id, query=query,
				max_num_results=max_num_results,
				filters=filters, ranking_options=ranking_options, rewrite_query=rewrite_query )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, store_id: str, query: str )'
			Logger( ).write( exception )
			raise exception
	
	def search_store( self, store_id: str, query: str, max_num_results: int = 10,
			filters: Dict[ str, Any ] = None, ranking_options: Dict[ str, Any ] = None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""Search store.
		
		Purpose:
		    Performs the VectorStores.search_store workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    query (str): Query value used by the operation.
		    max_num_results (int): Max num results value used by the operation.
		    filters (Dict[str, Any]): Filters value used by the operation.
		    ranking_options (Dict[str, Any]): Ranking options value used by the operation.
		    rewrite_query (bool): Rewrite query value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			throw_if( 'query', query )
			
			self.request = {
					'query': query.strip( ),
					'max_num_results': self.validate_max_num_results( max_num_results ),
			}
			
			if isinstance( filters, dict ) and len( filters ) > 0:
				self.request[ 'filters' ] = filters
			
			if isinstance( ranking_options, dict ) and len( ranking_options ) > 0:
				self.request[ 'ranking_options' ] = ranking_options
			
			if isinstance( rewrite_query, bool ):
				self.request[ 'rewrite_query' ] = rewrite_query
			
			self.response = self.client.vector_stores.search(
				vector_store_id=self.store_id,
				**self.request )
			
			self.search_results = self.normalize_search_results( self.response )
			return self.search_results
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search_store( self, store_id: str, query: str )'
			Logger( ).write( exception )
			raise exception
	
	def answer_with_file_search( self, store_ids: List[ str ], prompt: str,
			model: str = 'gpt-4o-mini', max_num_results: int = 10,
			instructions: str = None ) -> str | None:
		"""Answer with file search.
		
		Purpose:
		    Performs the VectorStores.answer_with_file_search workflow using the inputs supplied by the
		    caller and the current runtime configuration. The function keeps this behavior isolated so
		    related UI, provider, and data-processing paths can call it consistently.
		
		Args:
		    store_ids (List[str]): Store ids value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    max_num_results (int): Max num results value used by the operation.
		    instructions (str): Instructions value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			clean_store_ids = [
					item.strip( ) for item in store_ids
					if isinstance( item, str ) and item.strip( )
			]
			
			throw_if( 'store_ids', clean_store_ids )
			throw_if( 'prompt', prompt )
			model_value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini'
			input_items: List[ Dict[ str, Any ] ] = [ ]
			if isinstance( instructions, str ) and instructions.strip( ):
				input_items.append(
					{
							'role': 'developer',
							'content': [
									{
											'type': 'input_text',
											'text': instructions.strip( ),
									}, ],
					} )
			
			input_items.append(
				{
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt.strip( ),
								}, ],
				} )
			
			self.request = {
					'model': model_value,
					'input': input_items,
					'tools': [
							{
									'type': 'file_search',
									'vector_store_ids': clean_store_ids,
									'max_num_results': self.validate_max_num_results(
										max_num_results ),
							}, ],
			}
			
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			return str( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'answer_with_file_search( self, store_ids: List[ str ], prompt: str )'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, store_ids: List[ str ], prompt: str = None, model: str = 'gpt-4o-mini',
			max_num_results: int = 10, instructions: str = None ) -> str | None:
		"""Survey.
		
		Purpose:
		    Performs the VectorStores.survey workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_ids (List[str]): Store ids value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    max_num_results (int): Max num results value used by the operation.
		    instructions (str): Instructions value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			query = prompt if isinstance( prompt, str ) and prompt.strip( ) else \
				'Summarize the most relevant information available in the selected vector stores.'
			
			return self.answer_with_file_search(
				store_ids=store_ids,
				prompt=query,
				model=model,
				max_num_results=max_num_results,
				instructions=instructions )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'survey( self, store_ids: List[ str ], prompt: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the VectorStores.__dir__ workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'client',
				'name',
				'description',
				'store_id',
				'file_id',
				'batch_id',
				'model',
				'response',
				'vector_store',
				'vector_stores',
				'vector_file',
				'vector_files',
				'file_batch',
				'search_results',
				'output_text',
				'request',
				'collections',
				'max_search_results',
				'model_options',
				'ranker_options',
				'chunking_strategy_options',
				'validate_store_name',
				'validate_store_id',
				'validate_file_id',
				'validate_batch_id',
				'validate_file_ids',
				'validate_max_num_results',
				'build_expires_after',
				'build_chunking_strategy',
				'normalize_vector_store',
				'normalize_vector_store_file',
				'normalize_file_batch',
				'normalize_search_results',
				'create',
				'list_stores',
				'retrieve',
				'update',
				'delete',
				'attach_file',
				'list',
				'list_files',
				'retrieve_file',
				'update_file',
				'delete_file',
				'retrieve_file_content',
				'create_file_batch',
				'retrieve_file_batch',
				'list_file_batch_files',
				'cancel_file_batch',
				'search',
				'search_store',
				'answer_with_file_search',
				'survey',
		]