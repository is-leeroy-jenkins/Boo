'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                grok.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="grok.py" company="Terry D. Eppler">

	     grok.py
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
    Groq Cloud API wrapper for Streamlit with Hybrid Tool support.
  </summary>
  ******************************************************************************************
'''
import os
import base64
import requests
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from google.genai.types import ListFilesResponse
import config as cfg
from boogr import Error, Logger
import config as cfg
from openai import OpenAI
from xai_sdk.aio.image import ImageResponse
from xai_sdk import Client
from xai_sdk.tools import web_search, x_search, collections_search, code_execution
from xai_sdk.chat import user, system, image, file

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
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Grok( ):
	"""Grok class.
	
	Purpose:
	    Defines the Grok component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    timeout (Optional[float]): Stores timeout for the component runtime state.
	    base_url (Optional[str]): Stores base url for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    store_messages (Optional[bool]): Stores store messages for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    temperature (Optional[float]): Stores temperature for the component runtime state.
	    top_percent (Optional[float]): Stores top percent for the component runtime state.
	    frequency_penalty (Optional[float]): Stores frequency penalty for the component runtime state.
	    presence_penalty (Optional[float]): Stores presence penalty for the component runtime state.
	    max_output_tokens (Optional[int]): Stores max output tokens for the component runtime state.
	    tool_choice (Optional[str]): Stores tool choice for the component runtime state.
	    tools (Optional[List[str]]): Stores tools for the component runtime state.
	    stops (Optional[List[str]]): Stores stops for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    content (Optional[str]): Stores content for the component runtime state.
	    messages (Optional[List[Dict[str, Any]]]): Stores messages for the component runtime state.
	    stores (Optional[Dict[str, str]]): Stores stores for the component runtime state.
	    files (Optional[Dict[str, str]]): Stores files for the component runtime state."""
	api_key: Optional[ str ]
	timeout: Optional[ float ]
	base_url: Optional[ str ]
	model: Optional[ str ]
	store_messages: Optional[ bool ]
	response_format: Optional[ str ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_output_tokens: Optional[ int ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	instructions: Optional[ str ]
	content: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	stores: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Grok object with its default configuration, runtime state, provider settings,
		    and compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment."""
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.timeout = None
		self.instructions = None
		self.content = None
		self.store_messages = None
		self.model = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
		self.tool_choice = None
		self.tools = [ ]
		self.frequency_penalty = None
		self.presence_penalty = None
		self.response_format = None
		self.messages = [ ]
		self.stops = [ ]
		self.collections = None
		self.files = None

class Chat( Grok ):
	"""Chat class.
	
	Purpose:
	    Defines the Chat component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    include (Optional[List[str]]): Stores include for the component runtime state.
	    tool_choice (Optional[str]): Stores tool choice for the component runtime state.
	    previous_id (Optional[str]): Stores previous id for the component runtime state.
	    previous_response_id (Optional[str]): Stores previous response id for the component runtime state.
	    conversation_id (Optional[str]): Stores conversation id for the component runtime state.
	    parallel_tools (Optional[bool]): Stores parallel tools for the component runtime state.
	    max_tools (Optional[int]): Stores max tools for the component runtime state.
	    input (Optional[List[Dict[str, Any]] | str]): Stores input for the component runtime state.
	    tools (Optional[List[Any]]): Stores tools for the component runtime state.
	    reasoning (Optional[Dict[str, str] | str]): Stores reasoning for the component runtime state.
	    allowed_domains (Optional[List[str]]): Stores allowed domains for the component runtime state.
	    max_search_results (Optional[int]): Stores max search results for the component runtime state.
	    output_text (Optional[str]): Stores output text for the component runtime state.
	    collections (Optional[Dict[str, str]]): Stores collections for the component runtime state.
	    files (Optional[Dict[str, str]]): Stores files for the component runtime state.
	    content (Optional[str]): Stores content for the component runtime state.
	    vector_store_ids (Optional[List[str]]): Stores vector store ids for the component runtime state.
	    file_ids (Optional[List[str]]): Stores file ids for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    chat (Optional[Any]): Stores chat for the component runtime state."""
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	previous_id: Optional[ str ]
	previous_response_id: Optional[ str ]
	conversation_id: Optional[ str ]
	parallel_tools: Optional[ bool ]
	max_tools: Optional[ int ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	tools: Optional[ List[ Any ] ]
	reasoning: Optional[ Dict[ str, str ] | str ]
	allowed_domains: Optional[ List[ str ] ]
	max_search_results: Optional[ int ]
	output_text: Optional[ str ]
	collections: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	content: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	response: Optional[ Any ]
	file_path: Optional[ str ]
	chat: Optional[ Any ]
	
	def __init__( self, model: str = 'grok-4.20', prompt: str = None,
			temperature: float = None, top_p: float = None, presense: float = None,
			presence: float = None, store: bool = None, stream: bool = None,
			stops: List[ str ] = None, response_format: Dict[ str, Any ] = None,
			number: int = None, instruct: str = None,
			context: List[ Dict[ str, str ] ] = None,
			allowed_domains: List[ str ] = None, include: List[ str ] = None,
			tools: List[ Any ] = None, max_tools: int = None, tool_choice: str = None,
			file_path: str = None, background: bool = None, is_parallel: bool = None,
			max_tokens: int = None, frequency: float = None,
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
		    tools (List[Any]): Tools value used by the operation.
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
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.client = None
		self.chat = None
		self.model = model
		self.prompt = prompt
		self.number = number
		self.response_format = response_format if response_format is not None else { }
		self.temperature = temperature
		self.top_percent = top_p
		self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
		self.frequency_penalty = frequency
		self.presence_penalty = presence if presence is not None else presense
		self.max_output_tokens = max_tokens
		self.context = context if context is not None else [ ]
		self.stream = stream
		self.store_messages = store
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
		self.previous_response_id = previous_id
		self.conversation_id = conversation_id
		self.reasoning = reasoning
		self.parallel_tools = is_parallel
		self.tool_choice = tool_choice
		self.response = None
		self.file_path = file_path
		self.content = content
		self.max_search_results = max_search_results
		self.extra_kwargs = { }
		self.collections = cfg.GROK_COLLECTIONS
		self.files = getattr( cfg, 'GROK_DOCUMENTS', { } )
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'grok-4.20',
				'grok-4.20-reasoning',
				'grok-4.20-multi-agent',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-code-fast-1',
				'grok-3',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
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
				'verbose_streaming',
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
				'x_search',
				'collections_search',
				'code_execution',
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
		return [
				'auto',
				'required',
				'none',
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
				'low',
				'medium',
				'high',
				'xhigh',
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
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
		]
	
	def generate_text( self, prompt: str, model: str = 'grok-4.20',
			temperature: float = None, format: Dict[ str, Any ] | str = None,
			top_p: float = None, frequency: float = None, max_tools: int = None,
			presence: float = None, max_tokens: int = None, store: bool = None,
			stream: bool = False, instruct: str = None, background: bool = False,
			reasoning: str | Dict[ str, str ] = None, include: List[ str ] = None,
			tools: List[ Any ] = None, allowed_domains: List[ str ] = None,
			previous_id: str = None, tool_choice: str = None,
			is_parallel: bool = None, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None,
			response_schema: Any = None, **kwargs: Any ) -> str | None:
		"""Generate text.
		
		Purpose:
		    Generates provider output for the Chat workflow using validated model settings and request
		    inputs. The method coordinates request construction, provider execution, response capture, and
		    logged exception handling.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    format (Dict[str, Any] | str): Format value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    max_tools (int): Max tools value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    background (bool): Background value used by the operation.
		    reasoning (str | Dict[str, str]): Reasoning value used by the operation.
		    include (List[str]): Include value used by the operation.
		    tools (List[Any]): Tools value used by the operation.
		    allowed_domains (List[str]): Allowed domains value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    is_parallel (bool): Is parallel value used by the operation.
		    context (List[Dict[str, str]]): Context value used by the operation.
		    input_data (List[Dict[str, Any]]): Input data value used by the operation.
		    vector_store_ids (List[str]): Vector store ids value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		    response_schema (Any): Response schema value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			
			self.prompt = str( prompt ).strip( )
			self.model = str( model ).strip( )
			self.temperature = temperature
			self.response_format = format
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tools = max_tools
			self.max_output_tokens = max_tokens
			self.store_messages = store
			self.stream = stream
			self.background = background
			self.instructions = instruct
			self.reasoning = reasoning
			self.include = include if include is not None else [ ]
			self.tools = tools if tools is not None else [ ]
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.previous_id = previous_id
			self.previous_response_id = previous_id
			self.tool_choice = tool_choice
			self.parallel_tools = is_parallel
			self.context = context if context is not None else [ ]
			self.input = input_data if input_data is not None else [ ]
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			self.conversation_id = conversation_id
			self.response_schema = response_schema
			self.extra_kwargs = kwargs or { }
			self.tool_objects = [ ]
			
			for selected_tool in self.tools:
				if isinstance( selected_tool, dict ):
					self.tool_name = str( selected_tool.get( 'type', '' ) or '' ).strip( )
				else:
					self.tool_name = str( selected_tool or '' ).strip( )
				
				if not self.tool_name:
					continue
				
				if self.tool_name == 'web_search':
					if len( self.allowed_domains ) > 0:
						self.tool_objects.append(
							web_search( allowed_domains=self.allowed_domains )
						)
					else:
						self.tool_objects.append( web_search( ) )
					
					continue
				
				if self.tool_name == 'x_search':
					self.tool_objects.append( x_search( ) )
					continue
				
				if self.tool_name == 'collections_search':
					if len( self.vector_store_ids ) == 0:
						continue
					
					self.tool_objects.append(
						collections_search( collection_ids=self.vector_store_ids )
					)
					continue
				
				if self.tool_name == 'code_execution':
					self.tool_objects.append( code_execution( ) )
					continue
			
			self.client = Client( api_key=self.api_key, timeout=3600 )
			self.chat_kwargs = {
					'model': self.model,
			}
			
			if len( self.tool_objects ) > 0:
				self.chat_kwargs[ 'tools' ] = self.tool_objects
			
			if self.store_messages is not None:
				self.chat_kwargs[ 'store_messages' ] = self.store_messages
			
			if isinstance( self.include, list ) and 'verbose_streaming' in self.include:
				self.chat_kwargs[ 'include' ] = [ 'verbose_streaming' ]
			
			if isinstance( self.previous_id, str ) and self.previous_id.strip( ):
				self.chat_kwargs[ 'previous_response_id' ] = self.previous_id.strip( )
			
			self.chat = self.client.chat.create( **self.chat_kwargs )
			
			if self.instructions and str( self.instructions ).strip( ):
				self.chat.append( system( str( self.instructions ).strip( ) ) )
			
			for item in self.context:
				if not isinstance( item, dict ):
					continue
				
				self.role = str( item.get( 'role', '' ) or '' ).strip( )
				self.message_content = str( item.get( 'content', '' ) or '' ).strip( )
				
				if not self.message_content:
					continue
				
				if self.role == 'system':
					self.chat.append( system( self.message_content ) )
					continue
				
				if self.role == 'user':
					self.chat.append( user( self.message_content ) )
					continue
				
				if self.role == 'assistant':
					self.chat.append(
						user( f'Previous assistant response: {self.message_content}' ) )
					continue
			
			self.chat.append( user( self.prompt ) )
			
			if self.stream:
				self.parts = [ ]
				self.response = None
				
				for response, chunk in self.chat.stream( ):
					self.response = response
					if getattr( chunk, 'content', None ):
						self.parts.append( chunk.content )
				
				self.output_text = ''.join( self.parts ).strip( )
				if self.output_text:
					return self.output_text
				
				if self.response is not None and getattr( self.response, 'content', None ):
					self.output_text = self.response.content
					return self.output_text
				
				return None
			
			self.response = self.chat.sample( )
			self.previous_id = getattr( self.response, 'id', None )
			self.previous_response_id = self.previous_id
			self.output_text = getattr( self.response, 'content', None )
			
			if self.output_text:
				return self.output_text
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Chat'
			ex.method = 'generate_text( self, prompt: str, model: str ) -> str | None'
			Logger( ).write( ex )
			raise ex
	
	def get_usage( self ) -> Dict[ str, Any ] | None:
		"""Get usage.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if self.response is None:
				return None
			
			self.usage = getattr( self.response, 'usage', None )
			if self.usage is None:
				return None
			
			try:
				return dict( self.usage )
			except Exception:
				return self.usage
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Chat'
			ex.method = 'get_usage( self ) -> Dict[ str, Any ] | None'
			Logger( ).write( ex )
			raise ex
	
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
				'base_url',
				'client',
				'chat',
				'model',
				'prompt',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_output_tokens',
				'stops',
				'store_messages',
				'stream',
				'background',
				'number',
				'response_format',
				'context',
				'instructions',
				'include',
				'tool_choice',
				'previous_id',
				'previous_response_id',
				'conversation_id',
				'parallel_tools',
				'max_tools',
				'input',
				'tools',
				'tool_objects',
				'reasoning',
				'allowed_domains',
				'max_search_results',
				'output_text',
				'vector_store_ids',
				'file_ids',
				'response',
				'file_path',
				'collections',
				'files',
				'model_options',
				'include_options',
				'tool_options',
				'choice_options',
				'format_options',
				'reasoning_options',
				'modality_options',
				'media_options',
				'generate_text',
				'get_usage',
		]

class Images( Grok ):
	"""Images class.
	
	Purpose:
	    Defines the Images component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    model (Optional[str]): Stores model for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    aspect_ratio (Optional[str]): Stores aspect ratio for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    client (Optional[Client]): Stores client for the component runtime state.
	    image_path (Optional[str]): Stores image path for the component runtime state.
	    image_url (Optional[str]): Stores image url for the component runtime state.
	    detail (Optional[str]): Stores detail for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    output (Optional[Any]): Stores output for the component runtime state."""
	model: Optional[ str ]
	prompt: Optional[ str ]
	aspect_ratio: Optional[ str ]
	response_format: Optional[ str ]
	client: Optional[ Client ]
	image_path: Optional[ str ]
	image_url: Optional[ str ]
	detail: Optional[ str ]
	response: Optional[ Any ]
	output: Optional[ Any ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Images object with its default configuration, runtime state, provider settings,
		    and compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment."""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.client = None
		self.model = None
		self.prompt = None
		self.number = None
		self.aspect_ratio = None
		self.size = None
		self.quality = None
		self.style = None
		self.detail = None
		self.response_format = None
		self.mime_type = None
		self.compression = None
		self.background = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
		self.store = None
		self.stream = None
		self.instructions = None
		self.image_path = None
		self.image_url = None
		self.file_path = None
		self.mask_path = None
		self.response = None
		self.output = None
		self.output_text = None
		self.extra_kwargs = { }
	
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
				'grok-imagine-image-quality',
				'grok-imagine-image',
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
				'grok-4.20-reasoning',
				'grok-4.20',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-3',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		"""Aspect options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
				'1:1',
				'16:9',
				'9:16',
				'4:3',
				'3:4',
				'3:2',
				'2:3',
				'2:1',
				'1:2',
				'19.5:9',
				'9:19.5',
				'20:9',
				'9:20',
		]
	
	@property
	def size_options( self ) -> List[ str ] | None:
		"""Size options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
		]
	
	@property
	def quality_options( self ) -> List[ str ] | None:
		"""Quality options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
				'quality',
		]
	
	@property
	def style_options( self ) -> List[ str ] | None:
		"""Style options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'natural',
		]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'url',
				'b64_json',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'jpeg',
				'png',
		]
	
	@property
	def detail_options( self ) -> List[ str ] | None:
		"""Detail options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'auto',
				'low',
				'high',
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
				'image',
		]
	
	def generate( self, prompt: str, model: str = 'grok-imagine-image-quality',
			number: int = 1, size: str = None, quality: str = None, style: str = None,
			fmt: str = None, mime_type: str = None, compression: float = None,
			background: str = None, aspect_ratio: str = None,
			response_modalities: str = None, **kwargs: Any ) -> Any:
		"""Generate.
		
		Purpose:
		    Generates provider output for the Images workflow using validated model settings and request
		    inputs. The method coordinates request construction, provider execution, response capture, and
		    logged exception handling.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    number (int): Number value used by the operation.
		    size (str): Size value used by the operation.
		    quality (str): Quality value used by the operation.
		    style (str): Style value used by the operation.
		    fmt (str): Fmt value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    compression (float): Compression value used by the operation.
		    background (str): Background value used by the operation.
		    aspect_ratio (str): Aspect ratio value used by the operation.
		    response_modalities (str): Response modalities value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			self.api_key = cfg.XAI_API_KEY
			self.prompt = str( prompt ).strip( )
			self.model = str( model ).strip( )
			self.number = number if isinstance( number, int ) and number > 0 else 1
			self.size = size
			self.quality = quality
			self.style = style
			self.response_format = fmt or kwargs.get( 'response_format' )
			self.mime_type = mime_type
			self.compression = compression
			self.background = background
			self.aspect_ratio = aspect_ratio
			self.response_modalities = response_modalities
			self.extra_kwargs = kwargs or { }
			self.client = Client( api_key=self.api_key, timeout=3600 )
			
			if self.number > 1:
				if self.aspect_ratio in self.aspect_options and self.aspect_ratio != 'auto':
					self.response = self.client.image.sample_batch(
						prompt=self.prompt,
						model=self.model,
						n=self.number,
						aspect_ratio=self.aspect_ratio
					)
				else:
					self.response = self.client.image.sample_batch(
						prompt=self.prompt,
						model=self.model,
						n=self.number
					)
				
				self.output = [ ]
				for item in self.response:
					if getattr( item, 'url', None ):
						self.output.append( item.url )
					elif getattr( item, 'b64_json', None ):
						self.output.append( base64.b64decode( item.b64_json ) )
					else:
						self.output.append( item )
				
				return self.output
			
			if self.aspect_ratio in self.aspect_options and self.aspect_ratio != 'auto':
				self.response = self.client.image.sample(
					prompt=self.prompt,
					model=self.model,
					aspect_ratio=self.aspect_ratio
				)
			else:
				self.response = self.client.image.sample(
					prompt=self.prompt,
					model=self.model
				)
			
			if getattr( self.response, 'url', None ):
				return self.response.url
			
			if getattr( self.response, 'b64_json', None ):
				return base64.b64decode( self.response.b64_json )
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'generate( self, prompt: str, model: str )'
			Logger( ).write( ex )
			raise ex
	
	def edit( self, image_path: str = None, prompt: str = None,
			model: str = 'grok-imagine-image-quality', image_url: str = None,
			path: str = None, number: int = 1, size: str = None,
			aspect_ratio: str = None, fmt: str = None, mask_path: str = None,
			mask: str = None, **kwargs: Any ) -> Any:
		"""Edit.
		
		Purpose:
		    Performs the Images.edit workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    image_path (str): Image path value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    image_url (str): Image url value used by the operation.
		    path (str): Path value used by the operation.
		    number (int): Number value used by the operation.
		    size (str): Size value used by the operation.
		    aspect_ratio (str): Aspect ratio value used by the operation.
		    fmt (str): Fmt value used by the operation.
		    mask_path (str): Mask path value used by the operation.
		    mask (str): Mask value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.image_path = image_path or path
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			if not image_url:
				throw_if( 'image_path', self.image_path )
			
			self.api_key = cfg.XAI_API_KEY
			self.prompt = str( prompt ).strip( )
			self.model = str( model ).strip( )
			self.image_url = image_url
			self.image_path = str( self.image_path ).strip( ) if self.image_path else None
			self.file_path = self.image_path
			self.number = number if isinstance( number, int ) and number > 0 else 1
			self.size = size
			self.aspect_ratio = aspect_ratio
			self.response_format = fmt or kwargs.get( 'response_format' )
			self.mask_path = mask_path or mask
			self.extra_kwargs = kwargs or { }
			
			if self.mask_path:
				raise ValueError( 'xAI image editing does not support mask-based edits.' )
			
			if not self.image_url:
				if not os.path.exists( self.image_path ):
					raise FileNotFoundError( f'Image file not found: {self.image_path}' )
				
				self.suffix = Path( self.image_path ).suffix.lower( ).replace( '.', '' )
				if self.suffix == 'jpg':
					self.suffix = 'jpeg'
				
				if self.suffix not in [ 'jpeg', 'png' ]:
					raise ValueError( 'xAI image editing supports JPEG and PNG source images.' )
				
				with open( self.image_path, 'rb' ) as image_file:
					self.encoded_image = base64.b64encode(
						image_file.read( ) ).decode( 'utf-8' )
				
				self.image_url = f'data:image/{self.suffix};base64,{self.encoded_image}'
			
			self.client = Client( api_key=self.api_key, timeout=3600 )
			
			if self.number > 1:
				if self.aspect_ratio in self.aspect_options and self.aspect_ratio != 'auto':
					self.response = self.client.image.sample_batch(
						prompt=self.prompt,
						model=self.model,
						n=self.number,
						image_url=self.image_url,
						aspect_ratio=self.aspect_ratio
					)
				else:
					self.response = self.client.image.sample_batch(
						prompt=self.prompt,
						model=self.model,
						n=self.number,
						image_url=self.image_url
					)
				
				self.output = [ ]
				for item in self.response:
					if getattr( item, 'url', None ):
						self.output.append( item.url )
					elif getattr( item, 'b64_json', None ):
						self.output.append( base64.b64decode( item.b64_json ) )
					else:
						self.output.append( item )
				
				return self.output
			
			if self.aspect_ratio in self.aspect_options and self.aspect_ratio != 'auto':
				self.response = self.client.image.sample(
					prompt=self.prompt,
					model=self.model,
					image_url=self.image_url,
					aspect_ratio=self.aspect_ratio
				)
			else:
				self.response = self.client.image.sample(
					prompt=self.prompt,
					model=self.model,
					image_url=self.image_url
				)
			
			if getattr( self.response, 'url', None ):
				return self.response.url
			
			if getattr( self.response, 'b64_json', None ):
				return base64.b64decode( self.response.b64_json )
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'edit( self, image_path: str=None, prompt: str=None, model: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def analyze( self, prompt: str, image_url: str = None,
			model: str = 'grok-4.20-reasoning', max_output_tokens: int = 10000,
			temperature: float = None, top_p: float = None, detail: str = 'high',
			image_path: str = None, path: str = None, store: bool = False,
			**kwargs: Any ) -> str | None:
		"""Analyze.
		
		Purpose:
		    Performs the Images.analyze workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    image_url (str): Image url value used by the operation.
		    model (str): Model value used by the operation.
		    max_output_tokens (int): Max output tokens value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    detail (str): Detail value used by the operation.
		    image_path (str): Image path value used by the operation.
		    path (str): Path value used by the operation.
		    store (bool): Store value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.image_path = image_path or path
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			if not image_url:
				throw_if( 'image_path', self.image_path )
			
			self.api_key = cfg.XAI_API_KEY
			self.prompt = str( prompt ).strip( )
			self.model = str( model ).strip( )
			self.detail = str( detail or 'high' ).strip( )
			self.max_output_tokens = max_output_tokens
			self.temperature = temperature
			self.top_percent = top_p
			self.store = store
			self.extra_kwargs = kwargs or { }
			
			if image_url:
				self.image_url = str( image_url ).strip( )
			else:
				self.image_path = str( self.image_path ).strip( )
				if not os.path.exists( self.image_path ):
					raise FileNotFoundError( f'Image file not found: {self.image_path}' )
				
				self.suffix = Path( self.image_path ).suffix.lower( ).replace( '.', '' )
				if self.suffix == 'jpg':
					self.suffix = 'jpeg'
				
				if self.suffix not in [ 'jpeg', 'png' ]:
					raise ValueError( 'xAI image understanding supports JPEG and PNG inputs.' )
				
				with open( self.image_path, 'rb' ) as image_file:
					self.encoded_image = base64.b64encode(
						image_file.read( ) ).decode( 'utf-8' )
				
				self.image_url = f'data:image/{self.suffix};base64,{self.encoded_image}'
			
			self.client = Client( api_key=self.api_key, timeout=3600 )
			self.chat = self.client.chat.create( model=self.model )
			
			if self.instructions and str( self.instructions ).strip( ):
				self.chat.append( system( str( self.instructions ).strip( ) ) )
			
			self.chat.append( user( self.prompt, image( self.image_url ) ) )
			self.response = self.chat.sample( )
			self.output_text = getattr( self.response, 'content', None )
			
			if self.output_text:
				return self.output_text
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'analyze( self, prompt: str, image_url: str=None, model: str=None )'
			Logger( ).write( ex )
			raise ex
	
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
				'base_url',
				'client',
				'model',
				'prompt',
				'number',
				'aspect_ratio',
				'size',
				'quality',
				'style',
				'detail',
				'response_format',
				'mime_type',
				'compression',
				'background',
				'max_output_tokens',
				'temperature',
				'top_percent',
				'store',
				'stream',
				'instructions',
				'image_path',
				'image_url',
				'file_path',
				'mask_path',
				'response',
				'output',
				'output_text',
				'extra_kwargs',
				'model_options',
				'analysis_model_options',
				'aspect_options',
				'size_options',
				'quality_options',
				'style_options',
				'format_options',
				'mime_options',
				'detail_options',
				'modality_options',
				'generate',
				'edit',
				'analyze',
		]

class Files( Grok ):
	"""Files class.
	
	Purpose:
	    Defines the Files component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    client (Optional[Client]): Stores client for the component runtime state.
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    base_url (Optional[str]): Stores base url for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    file_name (Optional[str]): Stores file name for the component runtime state.
	    file_id (Optional[str]): Stores file id for the component runtime state.
	    file_ids (Optional[List[str]]): Stores file ids for the component runtime state.
	    file_paths (Optional[List[str]]): Stores file paths for the component runtime state.
	    file_names (Optional[List[str]]): Stores file names for the component runtime state.
	    purpose (Optional[str]): Stores purpose for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    file_content (Optional[Any]): Stores file content for the component runtime state.
	    output_text (Optional[str]): Stores output text for the component runtime state.
	    documents (Optional[Dict[str, str]]): Stores documents for the component runtime state.
	    temperature (Optional[float]): Stores temperature for the component runtime state.
	    top_percent (Optional[float]): Stores top percent for the component runtime state.
	    frequency_penalty (Optional[float]): Stores frequency penalty for the component runtime state.
	    presence_penalty (Optional[float]): Stores presence penalty for the component runtime state.
	    max_output_tokens (Optional[int]): Stores max output tokens for the component runtime state.
	    store (Optional[bool]): Stores store for the component runtime state.
	    stream (Optional[bool]): Stores stream for the component runtime state.
	    tools (Optional[List[Any]]): Stores tools for the component runtime state.
	    previous_id (Optional[str]): Stores previous id for the component runtime state.
	    previous_response_id (Optional[str]): Stores previous response id for the component runtime state.
	    limit (Optional[int]): Stores limit for the component runtime state.
	    pagination_token (Optional[str]): Stores pagination token for the component runtime state.
	    next_token (Optional[str]): Stores next token for the component runtime state.
	    download_format (Optional[str]): Stores download format for the component runtime state.
	    params (Optional[Dict[str, Any]]): Stores params for the component runtime state."""
	client: Optional[ Client ]
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	file_path: Optional[ str ]
	file_name: Optional[ str ]
	file_id: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	file_paths: Optional[ List[ str ] ]
	file_names: Optional[ List[ str ] ]
	purpose: Optional[ str ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	instructions: Optional[ str ]
	response: Optional[ Any ]
	file_content: Optional[ Any ]
	output_text: Optional[ str ]
	documents: Optional[ Dict[ str, str ] ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_output_tokens: Optional[ int ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	tools: Optional[ List[ Any ] ]
	previous_id: Optional[ str ]
	previous_response_id: Optional[ str ]
	limit: Optional[ int ]
	pagination_token: Optional[ str ]
	next_token: Optional[ str ]
	download_format: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Files object with its default configuration, runtime state, provider settings,
		    and compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment."""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1' )
		self.client = None
		self.model = None
		self.instructions = None
		self.prompt = None
		self.response = None
		self.file_content = None
		self.output_text = None
		self.file_id = None
		self.file_ids = [ ]
		self.file_path = None
		self.file_name = None
		self.file_paths = [ ]
		self.file_names = [ ]
		self.purpose = 'assistants'
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_output_tokens = None
		self.store = None
		self.stream = None
		self.tools = [ ]
		self.previous_id = None
		self.previous_response_id = None
		self.limit = None
		self.pagination_token = None
		self.next_token = None
		self.download_format = None
		self.params = { }
		self.extra_kwargs = { }
		self.documents = getattr( cfg, 'GROK_DOCUMENTS', { } )
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'grok-4.20-reasoning',
				'grok-4.20',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-code-fast-1',
				'grok-3',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""Purpose options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'assistants',
				'batch',
				'fine-tune',
				'user_data',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'text',
		]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Tool options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'code_execution',
		]
	
	def upload( self, filepath: str = None, filename: str = None, purpose: str = 'assistants',
			file_path: str = None, name: str = None, expires_after: int = None,
			**kwargs: Any ) -> Dict[ str, Any ]:
		"""Upload.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    filepath (str): Filepath value used by the operation.
		    filename (str): Filename value used by the operation.
		    purpose (str): Purpose value used by the operation.
		    file_path (str): File path value used by the operation.
		    name (str): Name value used by the operation.
		    expires_after (int): Expires after value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		source = None
		
		try:
			self.file_path = filepath or file_path
			throw_if( 'file_path', self.file_path )
			
			self.api_key = cfg.XAI_API_KEY
			self.file_path = str( self.file_path ).strip( )
			if not os.path.exists( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			
			self.file_name = str( filename or name or Path( self.file_path ).name ).strip( )
			self.purpose = str( purpose or 'assistants' ).strip( )
			self.expires_after = expires_after
			self.extra_kwargs = kwargs or { }
			self.client = Client( api_key=self.api_key, timeout=3600 )
			source = open( self.file_path, 'rb' )
			
			if self.expires_after is not None:
				self.response = self.client.files.upload(
					source,
					filename=self.file_name,
					expires_after=self.expires_after
				)
			else:
				self.response = self.client.files.upload(
					source,
					filename=self.file_name
				)
			
			self.file_id = getattr( self.response, 'id', None )
			self.file_name = getattr( self.response, 'filename', self.file_name )
			self.file = {
					'id': self.file_id,
					'name': self.file_name,
					'filename': self.file_name,
					'bytes': getattr( self.response, 'bytes', None ),
					'created_at': getattr( self.response, 'created_at', None ),
					'expires_at': getattr( self.response, 'expires_at', None ),
					'purpose': getattr( self.response, 'purpose', self.purpose ),
					'object': getattr( self.response, 'object', 'file' ),
					'metadata': self.response,
			}
			return self.file
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'upload( self, filepath: str=None, filename: str=None ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
		finally:
			if source is not None:
				source.close( )
	
	def list( self, limit: int = None, pagination_token: str = None,
			next_token: str = None, order: str = None, sort_by: str = None,
			filter: str = None, team_id: str = None,
			**kwargs: Any ) -> List[ Dict[ str, Any ] ]:
		"""List.
		
		Purpose:
		    Performs the Files.list workflow using the inputs supplied by the caller and the current runtime
		    configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    limit (int): Limit value used by the operation.
		    pagination_token (str): Pagination token value used by the operation.
		    next_token (str): Next token value used by the operation.
		    order (str): Order value used by the operation.
		    sort_by (str): Sort by value used by the operation.
		    filter (str): Filter value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.api_key = cfg.XAI_API_KEY
			self.limit = limit
			self.pagination_token = pagination_token or next_token
			self.next_token = self.pagination_token
			self.order = order
			self.sort_by = sort_by
			self.filter = filter
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.params = {
					'limit': self.limit,
					'pagination_token': self.pagination_token,
					'order': self.order,
					'sort_by': self.sort_by,
					'filter': self.filter,
			}
			self.response = requests.get(
				url=f'{self.base_url.rstrip( "/" )}/files',
				headers={ 'Authorization': f'Bearer {self.api_key}' },
				params={
						key: value
						for key, value in self.params.items( )
						if value is not None and value != ''
				},
				timeout=3600
			)
			self.response.raise_for_status( )
			self.result = self.response.json( )
			self.items = self.result.get( 'data', [ ] ) if isinstance( self.result, dict ) else [ ]
			self.pagination_token = self.result.get( 'pagination_token' ) if isinstance(
				self.result, dict ) else None
			self.files = [ ]
			
			for item in self.items:
				self.files.append( {
						'id': item.get( 'id' ),
						'name': item.get( 'filename' ) or item.get( 'name' ),
						'filename': item.get( 'filename' ) or item.get( 'name' ),
						'bytes': item.get( 'bytes' ) or item.get( 'size_bytes' ),
						'created_at': item.get( 'created_at' ),
						'expires_at': item.get( 'expires_at' ),
						'purpose': item.get( 'purpose' ),
						'object': item.get( 'object', 'file' ),
						'metadata': item,
				} )
			
			return self.files
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'list( self ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( ex )
			raise ex
	
	def list_files( self, limit: int = None, pagination_token: str = None,
			next_token: str = None, order: str = None, sort_by: str = None,
			filter: str = None, team_id: str = None,
			**kwargs: Any ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
		    Performs the Files.list_files workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    limit (int): Limit value used by the operation.
		    pagination_token (str): Pagination token value used by the operation.
		    next_token (str): Next token value used by the operation.
		    order (str): Order value used by the operation.
		    sort_by (str): Sort by value used by the operation.
		    filter (str): Filter value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.list( limit=limit, pagination_token=pagination_token,
				next_token=next_token, order=order, sort_by=sort_by, filter=filter,
				team_id=team_id, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'list_files( self ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, file_id: str = None, id: str = None, name: str = None,
			team_id: str = None, **kwargs: Any ) -> Dict[ str, Any ]:
		"""Retrieve.
		
		Purpose:
		    Performs the Files.retrieve workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    name (str): Name value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_id = file_id or id or name
			throw_if( 'file_id', self.file_id )
			
			self.api_key = cfg.XAI_API_KEY
			self.file_id = str( self.file_id ).strip( )
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.response = requests.get(
				url=f'{self.base_url.rstrip( "/" )}/files/{self.file_id}',
				headers={ 'Authorization': f'Bearer {self.api_key}' },
				timeout=3600
			)
			self.response.raise_for_status( )
			self.result = self.response.json( )
			return {
					'id': self.result.get( 'id', self.file_id ),
					'name': self.result.get( 'filename' ) or self.result.get( 'name' ),
					'filename': self.result.get( 'filename' ) or self.result.get( 'name' ),
					'bytes': self.result.get( 'bytes' ) or self.result.get( 'size_bytes' ),
					'created_at': self.result.get( 'created_at' ),
					'expires_at': self.result.get( 'expires_at' ),
					'purpose': self.result.get( 'purpose' ),
					'object': self.result.get( 'object', 'file' ),
					'metadata': self.result,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id: str=None, id: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def extract( self, file_id: str = None, id: str = None, name: str = None,
			format: str = None, page_number: int = None, team_id: str = None,
			**kwargs: Any ) -> bytes | str | None:
		"""Extract.
		
		Purpose:
		    Performs the Files.extract workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    name (str): Name value used by the operation.
		    format (str): Format value used by the operation.
		    page_number (int): Page number value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    bytes | Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_id = file_id or id or name
			throw_if( 'file_id', self.file_id )
			
			self.api_key = cfg.XAI_API_KEY
			self.file_id = str( self.file_id ).strip( )
			self.download_format = format
			self.page_number = page_number
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.response = requests.get(
				url=f'{self.base_url.rstrip( "/" )}/files/{self.file_id}/content',
				headers={ 'Authorization': f'Bearer {self.api_key}' },
				timeout=3600
			)
			self.response.raise_for_status( )
			self.file_content = self.response.content
			
			content_type = self.response.headers.get( 'Content-Type', '' )
			if content_type.startswith( 'text/' ) or self.download_format in [ 'text', 'txt',
			                                                                   'md' ]:
				try:
					self.file_content = self.response.content.decode( 'utf-8' )
				except Exception:
					self.file_content = self.response.text
			
			return self.file_content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'extract( self, file_id: str=None, id: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def download( self, file_id: str = None, id: str = None, name: str = None,
			format: str = None, page_number: int = None, team_id: str = None,
			**kwargs: Any ) -> bytes | str | None:
		"""Download.
		
		Purpose:
		    Performs the Files.download workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    name (str): Name value used by the operation.
		    format (str): Format value used by the operation.
		    page_number (int): Page number value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    bytes | Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.extract( file_id=file_id, id=id, name=name, format=format,
				page_number=page_number, team_id=team_id, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'download( self, file_id: str=None, id: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def content( self, file_id: str = None, id: str = None, name: str = None,
			format: str = None, page_number: int = None, team_id: str = None,
			**kwargs: Any ) -> bytes | str | None:
		"""Content.
		
		Purpose:
		    Performs the Files.content workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    name (str): Name value used by the operation.
		    format (str): Format value used by the operation.
		    page_number (int): Page number value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    bytes | Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.extract( file_id=file_id, id=id, name=name, format=format,
				page_number=page_number, team_id=team_id, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'content( self, file_id: str=None, id: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def delete( self, file_id: str = None, id: str = None, name: str = None,
			team_id: str = None, **kwargs: Any ) -> Dict[ str, Any ]:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    name (str): Name value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_id = file_id or id or name
			throw_if( 'file_id', self.file_id )
			
			self.api_key = cfg.XAI_API_KEY
			self.file_id = str( self.file_id ).strip( )
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.response = requests.delete(
				url=f'{self.base_url.rstrip( "/" )}/files/{self.file_id}',
				headers={ 'Authorization': f'Bearer {self.api_key}' },
				timeout=3600
			)
			self.response.raise_for_status( )
			
			if self.response.content:
				self.result = self.response.json( )
			else:
				self.result = { 'id': self.file_id, 'deleted': True }
			
			return {
					'id': self.result.get( 'id', self.file_id ),
					'deleted': self.result.get( 'deleted', True ),
					'object': self.result.get( 'object', 'file.deleted' ),
					'metadata': self.result,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'delete( self, file_id: str=None, id: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def summarize( self, file_id: str = None, id: str = None, name: str = None,
			file_path: str = None, filepath: str = None, filename: str = None,
			prompt: str = 'Summarize the attached file.', model: str = 'grok-4.20',
			instruct: str = None, temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, include: List[ str ] = None,
			tools: List[ Any ] = None, tool_choice: str = None,
			previous_id: str = None, conversation_id: str = None,
			response_format: Any = None, purpose: str = 'assistants',
			**kwargs: Any ) -> str | None:
		"""Summarize.
		
		Purpose:
		    Performs the Files.summarize workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    name (str): Name value used by the operation.
		    file_path (str): File path value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    filename (str): Filename value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    include (List[str]): Include value used by the operation.
		    tools (List[Any]): Tools value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		    response_format (Any): Response format value used by the operation.
		    purpose (str): Purpose value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			self.api_key = cfg.XAI_API_KEY
			self.file_path = file_path or filepath
			self.file_id = file_id or id or name
			
			if not self.file_id and self.file_path:
				self.uploaded_file = self.upload( filepath=self.file_path, filename=filename,
					purpose=purpose )
				self.file_id = self.uploaded_file.get( 'id' )
			
			throw_if( 'file_id', self.file_id )
			
			self.file_id = str( self.file_id ).strip( )
			self.prompt = str( prompt ).strip( )
			self.model = str( model ).strip( )
			self.instructions = instruct
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.tools = tools if tools is not None else [ ]
			self.previous_id = previous_id
			self.previous_response_id = previous_id
			self.conversation_id = conversation_id
			self.extra_kwargs = kwargs or { }
			self.client = Client( api_key=self.api_key, timeout=3600 )
			self.chat = self.client.chat.create( model=self.model )
			
			if self.instructions and str( self.instructions ).strip( ):
				self.chat.append( system( str( self.instructions ).strip( ) ) )
			
			self.chat.append( user( self.prompt, file( file_id=self.file_id ) ) )
			
			if self.stream:
				self.parts = [ ]
				self.response = None
				
				for response, chunk in self.chat.stream( ):
					self.response = response
					if getattr( chunk, 'content', None ):
						self.parts.append( chunk.content )
				
				self.output_text = ''.join( self.parts ).strip( )
				if self.output_text:
					return self.output_text
				
				if self.response is not None and getattr( self.response, 'content', None ):
					self.output_text = self.response.content
					return self.output_text
				
				return None
			
			self.response = self.chat.sample( )
			self.output_text = getattr( self.response, 'content', None )
			
			if self.output_text:
				return self.output_text
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'summarize( self, file_id: str=None, prompt: str=None, model: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def search( self, id: str = None, file_id: str = None, query: str = None,
			model: str = 'grok-4.20', **kwargs: Any ) -> str | None:
		"""Search.
		
		Purpose:
		    Performs the Files.search workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		    file_id (str): File id value used by the operation.
		    query (str): Query value used by the operation.
		    model (str): Model value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'query', query )
			
			self.prompt = (
					'Answer the user question using only the selected file content when possible. '
					f'Question: {query}'
			)
			self.file_id = file_id or id
			return self.summarize( file_id=self.file_id, prompt=self.prompt, model=model,
				**kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'search( self, id: str=None, file_id: str=None, query: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def survey( self, id: str = None, file_id: str = None, name: str = None,
			max_chars: int = 4000, **kwargs: Any ) -> Dict[ str, Any ]:
		"""Survey.
		
		Purpose:
		    Performs the Files.survey workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    id (str): Id value used by the operation.
		    file_id (str): File id value used by the operation.
		    name (str): Name value used by the operation.
		    max_chars (int): Max chars value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_id = file_id or id or name
			throw_if( 'file_id', self.file_id )
			
			self.file_id = str( self.file_id ).strip( )
			self.metadata = self.retrieve( file_id=self.file_id )
			self.file_content = self.extract( file_id=self.file_id )
			
			if isinstance( self.file_content, bytes ):
				try:
					self.content_text = self.file_content.decode( 'utf-8' )
				except Exception:
					self.content_text = str( self.file_content )
			elif isinstance( self.file_content, str ):
				self.content_text = self.file_content
			else:
				self.content_text = str( self.file_content )
			
			self.preview = self.content_text[ :max_chars ] if isinstance(
				max_chars, int ) else self.content_text
			return {
					'metadata': self.metadata,
					'preview': self.preview,
					'file_id': self.file_id,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'survey( self, id: str=None, file_id: str=None ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
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
				'base_url',
				'client',
				'file_path',
				'file_name',
				'file_id',
				'file_ids',
				'file_paths',
				'file_names',
				'purpose',
				'model',
				'prompt',
				'instructions',
				'response',
				'file_content',
				'output_text',
				'documents',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_output_tokens',
				'store',
				'stream',
				'tools',
				'previous_id',
				'previous_response_id',
				'limit',
				'pagination_token',
				'next_token',
				'download_format',
				'params',
				'model_options',
				'purpose_options',
				'format_options',
				'tool_options',
				'upload',
				'list',
				'list_files',
				'retrieve',
				'extract',
				'download',
				'content',
				'delete',
				'summarize',
				'search',
				'survey',
		]

class TTS( Grok ):
	"""TTS class.
	
	Purpose:
	    Defines the TTS component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    base_url (Optional[str]): Stores base url for the component runtime state.
	    text (Optional[str]): Stores text for the component runtime state.
	    language (Optional[str]): Stores language for the component runtime state.
	    voice_id (Optional[str]): Stores voice id for the component runtime state.
	    output_format (Optional[str | Dict[str, Any]]): Stores output format for the component runtime state.
	    speed (Optional[float]): Stores speed for the component runtime state.
	    optimize_streaming_latency (Optional[int]): Stores optimize streaming latency for the component runtime state.
	    text_normalization (Optional[bool]): Stores text normalization for the component runtime state.
	    sample_rate (Optional[int]): Stores sample rate for the component runtime state.
	    bit_rate (Optional[int]): Stores bit rate for the component runtime state.
	    audio_path (Optional[str]): Stores audio path for the component runtime state.
	    filepath (Optional[str]): Stores filepath for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    audio (Optional[bytes]): Stores audio for the component runtime state.
	    params (Optional[Dict[str, Any]]): Stores params for the component runtime state."""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	text: Optional[ str ]
	language: Optional[ str ]
	voice_id: Optional[ str ]
	output_format: Optional[ str | Dict[ str, Any ] ]
	speed: Optional[ float ]
	optimize_streaming_latency: Optional[ int ]
	text_normalization: Optional[ bool ]
	sample_rate: Optional[ int ]
	bit_rate: Optional[ int ]
	audio_path: Optional[ str ]
	filepath: Optional[ str ]
	response: Optional[ Any ]
	audio: Optional[ bytes ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the TTS object with its default configuration, runtime state, provider settings, and
		    compatibility fields. This constructor prepares the instance for later method calls without
		    performing external work beyond local attribute assignment."""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1' )
		self.text = None
		self.language = None
		self.voice_id = None
		self.output_format = None
		self.speed = None
		self.optimize_streaming_latency = None
		self.text_normalization = None
		self.sample_rate = None
		self.bit_rate = None
		self.audio_path = None
		self.filepath = None
		self.response = None
		self.audio = None
		self.params = { }
	
	@property
	def voice_options( self ) -> List[ str ]:
		"""Voice options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'eve',
				'ara',
				'rex',
				'sal',
				'leo',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'mp3',
				'wav',
				'pcm',
				'mulaw',
				'alaw',
		]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Language options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'auto',
				'en',
				'ar-EG',
				'ar-SA',
				'ar-AE',
				'bn',
				'zh',
				'fr',
				'de',
				'hi',
				'id',
				'it',
				'ja',
				'ko',
				'pt-BR',
				'pt-PT',
				'ru',
				'es-MX',
				'es-ES',
				'tr',
				'vi',
		]
	
	def create_speech( self, text: str, language: str = 'en', voice_id: str = 'eve',
			output_format: str | Dict[ str, Any ] = 'mp3', speed: float = None,
			optimize_streaming_latency: int = None, text_normalization: bool = None,
			sample_rate: int = None, bit_rate: int = None, filepath: str = None,
			audio_path: str = None ) -> bytes:
		"""Create speech.
		
		Purpose:
		    Creates the requested resource, connection, schema object, or user interface artifact using
		    validated inputs. The function encapsulates setup details so callers can rely on a consistent
		    resource lifecycle.
		
		Args:
		    text (str): Text value used by the operation.
		    language (str): Language value used by the operation.
		    voice_id (str): Voice id value used by the operation.
		    output_format (str | Dict[str, Any]): Output format value used by the operation.
		    speed (float): Speed value used by the operation.
		    optimize_streaming_latency (int): Optimize streaming latency value used by the operation.
		    text_normalization (bool): Text normalization value used by the operation.
		    sample_rate (int): Sample rate value used by the operation.
		    bit_rate (int): Bit rate value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    audio_path (str): Audio path value used by the operation.
		
		Returns:
		    bytes: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'text', text )
			throw_if( 'language', language )
			
			self.api_key = cfg.XAI_API_KEY
			self.text = str( text ).strip( )
			self.language = str( language ).strip( )
			self.voice_id = str( voice_id or 'eve' ).strip( )
			self.output_format = output_format
			self.speed = speed
			self.optimize_streaming_latency = optimize_streaming_latency
			self.text_normalization = text_normalization
			self.sample_rate = sample_rate
			self.bit_rate = bit_rate
			self.filepath = filepath or audio_path
			self.audio_path = self.filepath
			
			if isinstance( self.output_format, dict ):
				self.output_format_payload = {
						key: value
						for key, value in self.output_format.items( )
						if value is not None and value != ''
				}
			else:
				self.codec = str( self.output_format or 'mp3' ).strip( ).lower( )
				self.output_format_payload = { 'codec': self.codec }
			
			if self.sample_rate is not None:
				self.output_format_payload[ 'sample_rate' ] = self.sample_rate
			
			if self.bit_rate is not None:
				self.output_format_payload[ 'bit_rate' ] = self.bit_rate
			
			self.params = {
					'text': self.text,
					'language': self.language,
					'voice_id': self.voice_id,
					'output_format': self.output_format_payload,
			}
			
			if self.speed is not None:
				self.params[ 'speed' ] = self.speed
			
			if self.optimize_streaming_latency is not None:
				self.params[ 'optimize_streaming_latency' ] = self.optimize_streaming_latency
			
			if self.text_normalization is not None:
				self.params[ 'text_normalization' ] = self.text_normalization
			
			self.response = requests.post(
				url=f'{self.base_url.rstrip( "/" )}/tts',
				headers={
						'Authorization': f'Bearer {self.api_key}',
						'Content-Type': 'application/json',
				},
				json=self.params,
				timeout=3600
			)
			self.response.raise_for_status( )
			self.audio = self.response.content
			
			if self.filepath:
				self.filepath = str( self.filepath ).strip( )
				with open( self.filepath, 'wb' ) as target:
					target.write( self.audio )
			
			return self.audio
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'create_speech( self, text: str, language: str ) -> bytes'
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the TTS.__dir__ workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'base_url',
				'text',
				'language',
				'voice_id',
				'output_format',
				'speed',
				'optimize_streaming_latency',
				'text_normalization',
				'sample_rate',
				'bit_rate',
				'audio_path',
				'filepath',
				'response',
				'audio',
				'params',
				'voice_options',
				'format_options',
				'language_options',
				'create_speech',
		]

class Translation( Grok ):
	"""Translation class.
	
	Purpose:
	    Defines the Translation component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    base_url (Optional[str]): Stores base url for the component runtime state.
	    audio_path (Optional[str]): Stores audio path for the component runtime state.
	    filepath (Optional[str]): Stores filepath for the component runtime state.
	    file_name (Optional[str]): Stores file name for the component runtime state.
	    mime_type (Optional[str]): Stores mime type for the component runtime state.
	    source_language (Optional[str]): Stores source language for the component runtime state.
	    target_language (Optional[str]): Stores target language for the component runtime state.
	    output_format (Optional[str]): Stores output format for the component runtime state.
	    keyterm (Optional[str]): Stores keyterm for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    transcript (Optional[str]): Stores transcript for the component runtime state.
	    translation (Optional[str]): Stores translation for the component runtime state.
	    result (Optional[Dict[str, Any]]): Stores result for the component runtime state.
	    params (Optional[Dict[str, Any]]): Stores params for the component runtime state.
	    chat (Optional[Any]): Stores chat for the component runtime state.
	    client (Optional[Client]): Stores client for the component runtime state."""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	audio_path: Optional[ str ]
	filepath: Optional[ str ]
	file_name: Optional[ str ]
	mime_type: Optional[ str ]
	source_language: Optional[ str ]
	target_language: Optional[ str ]
	output_format: Optional[ str ]
	keyterm: Optional[ str ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	response: Optional[ Any ]
	transcript: Optional[ str ]
	translation: Optional[ str ]
	result: Optional[ Dict[ str, Any ] ]
	params: Optional[ Dict[ str, Any ] ]
	chat: Optional[ Any ]
	client: Optional[ Client ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Translation object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment."""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1' )
		self.audio_path = None
		self.filepath = None
		self.file_name = None
		self.mime_type = None
		self.source_language = None
		self.target_language = None
		self.output_format = None
		self.keyterm = None
		self.model = None
		self.prompt = None
		self.response = None
		self.transcript = None
		self.translation = None
		self.result = None
		self.params = { }
		self.chat = None
		self.client = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'grok-4.20',
				'grok-4.20-reasoning',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-3',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'json',
				'verbose_json',
				'text',
		]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Language options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'auto',
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ja',
				'ko',
				'zh',
				'ar',
				'hi',
				'id',
				'ru',
				'tr',
				'vi',
		]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'audio/mpeg',
				'audio/mp3',
				'audio/wav',
				'audio/x-wav',
				'audio/flac',
				'audio/ogg',
				'audio/webm',
				'audio/mp4',
				'audio/aac',
				'audio/m4a',
		]
	
	def translate( self, path: str = None, audio_path: str = None, filepath: str = None,
			target_language: str = 'en', source_language: str = None,
			model: str = 'grok-4.20', format: str = 'json', mime_type: str = None,
			keyterm: str = None ) -> str | None:
		"""Translate.
		
		Purpose:
		    Performs the Translation.translate workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		    audio_path (str): Audio path value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    target_language (str): Target language value used by the operation.
		    source_language (str): Source language value used by the operation.
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    keyterm (str): Keyterm value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		source = None
		
		try:
			self.audio_path = path or audio_path or filepath
			throw_if( 'audio_path', self.audio_path )
			throw_if( 'target_language', target_language )
			throw_if( 'model', model )
			
			self.api_key = cfg.XAI_API_KEY
			self.audio_path = str( self.audio_path ).strip( )
			self.target_language = str( target_language ).strip( )
			self.source_language = source_language
			self.model = str( model ).strip( )
			self.output_format = str( format or 'json' ).strip( )
			self.mime_type = mime_type
			self.keyterm = keyterm
			
			if not os.path.exists( self.audio_path ):
				raise FileNotFoundError( f'Audio file not found: {self.audio_path}' )
			
			self.file_name = Path( self.audio_path ).name
			
			if not self.mime_type:
				self.suffix = Path( self.audio_path ).suffix.lower( )
				if self.suffix == '.mp3':
					self.mime_type = 'audio/mpeg'
				elif self.suffix == '.wav':
					self.mime_type = 'audio/wav'
				elif self.suffix == '.flac':
					self.mime_type = 'audio/flac'
				elif self.suffix == '.ogg':
					self.mime_type = 'audio/ogg'
				elif self.suffix == '.webm':
					self.mime_type = 'audio/webm'
				elif self.suffix in [ '.m4a', '.mp4' ]:
					self.mime_type = 'audio/mp4'
				elif self.suffix == '.aac':
					self.mime_type = 'audio/aac'
				else:
					self.mime_type = 'application/octet-stream'
			
			self.params = {
					'format': self.output_format,
					'language': self.source_language,
					'keyterm': self.keyterm,
			}
			
			source = open( self.audio_path, 'rb' )
			self.response = requests.post(
				url=f'{self.base_url.rstrip( "/" )}/stt',
				headers={ 'Authorization': f'Bearer {self.api_key}' },
				data={
						key: value
						for key, value in self.params.items( )
						if value is not None and value != '' and value != [ ]
				},
				files={
						'file': (self.file_name, source, self.mime_type),
				},
				timeout=3600
			)
			self.response.raise_for_status( )
			
			content_type = self.response.headers.get( 'Content-Type', '' )
			if 'application/json' in content_type:
				self.result = self.response.json( )
				self.transcript = self.result.get( 'text' )
			else:
				self.transcript = self.response.text
			
			throw_if( 'transcript', self.transcript )
			
			self.prompt = (
					f'Translate the following transcript into {self.target_language}. '
					'Return only the translated text without commentary.\n\n'
					f'{self.transcript}'
			)
			self.client = Client( api_key=self.api_key, timeout=3600 )
			self.chat = self.client.chat.create( model=self.model )
			self.chat.append( user( self.prompt ) )
			self.response = self.chat.sample( )
			self.translation = getattr( self.response, 'content', None )
			
			if self.translation:
				return self.translation
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'translate( self, path: str=None, target_language: str=None )'
			Logger( ).write( ex )
			raise ex
		finally:
			if source is not None:
				source.close( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Translation.__dir__ workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'base_url',
				'audio_path',
				'filepath',
				'file_name',
				'mime_type',
				'source_language',
				'target_language',
				'output_format',
				'keyterm',
				'model',
				'prompt',
				'response',
				'transcript',
				'translation',
				'result',
				'params',
				'chat',
				'client',
				'model_options',
				'format_options',
				'language_options',
				'mime_options',
				'translate',
		]

class Transcription( Grok ):
	"""Transcription class.
	
	Purpose:
	    Defines the Transcription component used by the Boo application. The class groups related
	    provider configuration, runtime state, helper methods, and API-facing behavior so Streamlit
	    workflows can call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    base_url (Optional[str]): Stores base url for the component runtime state.
	    audio_path (Optional[str]): Stores audio path for the component runtime state.
	    filepath (Optional[str]): Stores filepath for the component runtime state.
	    file_name (Optional[str]): Stores file name for the component runtime state.
	    mime_type (Optional[str]): Stores mime type for the component runtime state.
	    language (Optional[str]): Stores language for the component runtime state.
	    output_format (Optional[str]): Stores output format for the component runtime state.
	    keyterm (Optional[str]): Stores keyterm for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    transcript (Optional[str]): Stores transcript for the component runtime state.
	    result (Optional[Dict[str, Any]]): Stores result for the component runtime state.
	    words (Optional[List[Dict[str, Any]]]): Stores words for the component runtime state.
	    channels (Optional[List[Dict[str, Any]]]): Stores channels for the component runtime state.
	    duration (Optional[float]): Stores duration for the component runtime state.
	    params (Optional[Dict[str, Any]]): Stores params for the component runtime state."""
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	audio_path: Optional[ str ]
	filepath: Optional[ str ]
	file_name: Optional[ str ]
	mime_type: Optional[ str ]
	language: Optional[ str ]
	output_format: Optional[ str ]
	keyterm: Optional[ str ]
	response: Optional[ Any ]
	transcript: Optional[ str ]
	result: Optional[ Dict[ str, Any ] ]
	words: Optional[ List[ Dict[ str, Any ] ] ]
	channels: Optional[ List[ Dict[ str, Any ] ] ]
	duration: Optional[ float ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Transcription object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment."""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1' )
		self.audio_path = None
		self.filepath = None
		self.file_name = None
		self.mime_type = None
		self.language = None
		self.output_format = None
		self.keyterm = None
		self.response = None
		self.transcript = None
		self.result = None
		self.words = [ ]
		self.channels = [ ]
		self.duration = None
		self.params = { }
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'text',
				'json',
				'verbose_json',
				'vtt',
				'srt',
		]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Language options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'',
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ja',
				'ko',
				'zh',
		]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'audio/mpeg',
				'audio/mp3',
				'audio/wav',
				'audio/x-wav',
				'audio/flac',
				'audio/ogg',
				'audio/webm',
				'audio/mp4',
				'audio/aac',
				'audio/m4a',
		]
	
	def transcribe( self, path: str = None, audio_path: str = None,
			filepath: str = None, language: str = None, format: str = 'json',
			mime_type: str = None, keyterm: str = None ) -> str | Dict[ str, Any ]:
		"""Transcribe.
		
		Purpose:
		    Performs the Transcription.transcribe workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		    audio_path (str): Audio path value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    language (str): Language value used by the operation.
		    format (str): Format value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    keyterm (str): Keyterm value used by the operation.
		
		Returns:
		    str | Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		source = None
		
		try:
			audio_path_value = path or audio_path or filepath
			audio_path_value = str(
				audio_path_value ).strip( ) if audio_path_value is not None else None
			throw_if( 'audio_path', audio_path_value )
			
			format_value = str( format or 'json' ).strip( )
			throw_if( 'format', format_value )
			
			self.api_key = cfg.XAI_API_KEY
			self.audio_path = audio_path_value
			self.filepath = audio_path_value
			self.output_format = format_value
			self.language = str( language ).strip( ) if language is not None else None
			self.mime_type = str( mime_type ).strip( ) if mime_type is not None else None
			self.keyterm = str( keyterm ).strip( ) if keyterm is not None else None
			
			if not os.path.exists( self.audio_path ):
				raise FileNotFoundError( f'Audio file not found: {self.audio_path}' )
			
			self.file_name = Path( self.audio_path ).name
			
			if not self.mime_type:
				self.suffix = Path( self.audio_path ).suffix.lower( )
				if self.suffix == '.mp3':
					self.mime_type = 'audio/mpeg'
				elif self.suffix == '.wav':
					self.mime_type = 'audio/wav'
				elif self.suffix == '.flac':
					self.mime_type = 'audio/flac'
				elif self.suffix == '.ogg':
					self.mime_type = 'audio/ogg'
				elif self.suffix == '.webm':
					self.mime_type = 'audio/webm'
				elif self.suffix in [ '.m4a', '.mp4' ]:
					self.mime_type = 'audio/mp4'
				elif self.suffix == '.aac':
					self.mime_type = 'audio/aac'
				else:
					self.mime_type = 'application/octet-stream'
			
			self.params = {
					'format': self.output_format,
			}
			
			if self.language:
				self.params[ 'language' ] = self.language
			
			if self.keyterm:
				self.params[ 'keyterm' ] = self.keyterm
			
			source = open( self.audio_path, 'rb' )
			self.response = requests.post(
				url=f'{self.base_url.rstrip( "/" )}/stt',
				headers={ 'Authorization': f'Bearer {self.api_key}' },
				data=self.params,
				files={
						'file': (self.file_name, source, self.mime_type),
				},
				timeout=3600
			)
			self.response.raise_for_status( )
			
			content_type = self.response.headers.get( 'Content-Type', '' )
			if 'application/json' in content_type:
				self.result = self.response.json( )
				self.transcript = self.result.get( 'text' )
				self.language = self.result.get( 'language', self.language )
				self.duration = self.result.get( 'duration' )
				self.words = self.result.get( 'words', [ ] )
				self.channels = self.result.get( 'channels', [ ] )
				
				if self.output_format in [ 'text', 'txt' ] and self.transcript:
					return self.transcript
				
				return self.result
			
			self.transcript = self.response.text
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path: str=None, language: str=None )'
			Logger( ).write( ex )
			raise ex
		finally:
			if source is not None:
				source.close( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Performs the Transcription.__dir__ workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'api_key',
				'base_url',
				'audio_path',
				'filepath',
				'file_name',
				'mime_type',
				'language',
				'output_format',
				'keyterm',
				'response',
				'transcript',
				'result',
				'words',
				'channels',
				'duration',
				'params',
				'format_options',
				'language_options',
				'mime_options',
				'transcribe',
		]

class VectorStores( Grok ):
	"""VectorStores class.
	
	Purpose:
	    Defines the VectorStores component used by the Boo application. The class groups related
	    provider configuration, runtime state, helper methods, and API-facing behavior so Streamlit
	    workflows can call a consistent interface.
	
	Attributes:
	    client (Optional[Client]): Stores client for the component runtime state.
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    management_key (Optional[str]): Stores management key for the component runtime state.
	    base_url (Optional[str]): Stores base url for the component runtime state.
	    management_base_url (Optional[str]): Stores management base url for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    number (Optional[int]): Stores number for the component runtime state.
	    content (Optional[str]): Stores content for the component runtime state.
	    name (Optional[str]): Stores name for the component runtime state.
	    description (Optional[str]): Stores description for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    file_name (Optional[str]): Stores file name for the component runtime state.
	    file_id (Optional[str]): Stores file id for the component runtime state.
	    file_ids (Optional[List[str]]): Stores file ids for the component runtime state.
	    store_ids (Optional[List[str]]): Stores store ids for the component runtime state.
	    store_id (Optional[str]): Stores store id for the component runtime state.
	    collection_ids (Optional[List[str]]): Stores collection ids for the component runtime state.
	    collection_id (Optional[str]): Stores collection id for the component runtime state.
	    documents (Optional[Dict[str, str]]): Stores documents for the component runtime state.
	    collections (Optional[Dict[str, str]]): Stores collections for the component runtime state.
	    request (Optional[Dict[str, Any]]): Stores request for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    result (Optional[Any]): Stores result for the component runtime state.
	    params (Optional[Dict[str, Any]]): Stores params for the component runtime state.
	    payload (Optional[Dict[str, Any]]): Stores payload for the component runtime state.
	    headers (Optional[Dict[str, str]]): Stores headers for the component runtime state.
	    team_id (Optional[str]): Stores team id for the component runtime state.
	    limit (Optional[int]): Stores limit for the component runtime state.
	    order (Optional[str]): Stores order for the component runtime state.
	    sort_by (Optional[str]): Stores sort by for the component runtime state.
	    pagination_token (Optional[str]): Stores pagination token for the component runtime state.
	    filter (Optional[str]): Stores filter for the component runtime state."""
	client: Optional[ Client ]
	api_key: Optional[ str ]
	management_key: Optional[ str ]
	base_url: Optional[ str ]
	management_base_url: Optional[ str ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	number: Optional[ int ]
	content: Optional[ str ]
	name: Optional[ str ]
	description: Optional[ str ]
	file_path: Optional[ str ]
	file_name: Optional[ str ]
	file_id: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	store_ids: Optional[ List[ str ] ]
	store_id: Optional[ str ]
	collection_ids: Optional[ List[ str ] ]
	collection_id: Optional[ str ]
	documents: Optional[ Dict[ str, str ] ]
	collections: Optional[ Dict[ str, str ] ]
	request: Optional[ Dict[ str, Any ] ]
	response: Optional[ Any ]
	result: Optional[ Any ]
	params: Optional[ Dict[ str, Any ] ]
	payload: Optional[ Dict[ str, Any ] ]
	headers: Optional[ Dict[ str, str ] ]
	team_id: Optional[ str ]
	limit: Optional[ int ]
	order: Optional[ str ]
	sort_by: Optional[ str ]
	pagination_token: Optional[ str ]
	filter: Optional[ str ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the VectorStores object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment."""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.management_key = cfg.XAI_MANAGEMENT_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.management_base_url = getattr( cfg, 'XAI_MANAGEMENT_BASE_URL',
			'https://management-api.x.ai/v1' )
		self.client = None
		self.model = None
		self.prompt = None
		self.response_format = None
		self.number = None
		self.content = None
		self.name = None
		self.description = None
		self.response = None
		self.result = None
		self.request = { }
		self.params = { }
		self.payload = { }
		self.headers = { }
		self.file_id = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.collection_ids = [ ]
		self.file_path = None
		self.file_name = None
		self.store_id = None
		self.collection_id = None
		self.team_id = None
		self.limit = None
		self.order = None
		self.sort_by = None
		self.pagination_token = None
		self.filter = None
		self.collections = cfg.GROK_COLLECTIONS
		self.documents = getattr( cfg, 'GROK_DOCUMENTS', { } )
	
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
				'grok-4.20',
				'grok-4.20-reasoning',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-3',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def order_options( self ) -> List[ str ] | None:
		"""Order options.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'asc',
				'desc',
		]
	
	@property
	def collection_sort_options( self ) -> List[ str ] | None:
		"""Collection sort options.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'collection_name',
				'created_at',
				'documents_count',
		]
	
	@property
	def document_sort_options( self ) -> List[ str ] | None:
		"""Document sort options.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'name',
				'created_at',
				'size_bytes',
				'status',
		]
	
	def get_collection_id( self, store_id: str ) -> str:
		"""Get collection id.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			
			if self.store_id in self.collections:
				self.collection_id = self.collections[ self.store_id ]
			else:
				self.collection_id = self.store_id
			
			return self.collection_id
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'get_collection_id( self, store_id: str ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def get_collection_name( self, collection_id: str ) -> str:
		"""Get collection name.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Args:
		    collection_id (str): Collection id value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'collection_id', collection_id )
			self.collection_id = str( collection_id ).strip( )
			
			for name, configured_id in self.collections.items( ):
				if configured_id == self.collection_id:
					self.name = name
					return self.name
			
			self.name = self.collection_id
			return self.name
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'get_collection_name( self, collection_id: str ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def build_management_headers( self ) -> Dict[ str, str ]:
		"""Build management headers.
		
		Purpose:
		    Builds the normalized data structure required by the VectorStores workflow. The function
		    converts caller input, session state, or provider-specific options into a stable shape that
		    downstream API calls and rendering code can consume safely.
		
		Returns:
		    Dict[str, str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'XAI_MANAGEMENT_KEY', self.management_key )
			self.headers = {
					'Authorization': f'Bearer {self.management_key}',
					'Content-Type': 'application/json',
			}
			return self.headers
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'build_management_headers( self ) -> Dict[ str, str ]'
			Logger( ).write( ex )
			raise ex
	
	def execute_management_request( self, method: str, endpoint: str,
			params: Dict[ str, Any ] = None, payload: Dict[ str, Any ] = None ) -> Any:
		"""Execute management request.
		
		Purpose:
		    Performs the VectorStores.execute_management_request workflow using the inputs supplied by the
		    caller and the current runtime configuration. The function keeps this behavior isolated so
		    related UI, provider, and data-processing paths can call it consistently.
		
		Args:
		    method (str): Method value used by the operation.
		    endpoint (str): Endpoint value used by the operation.
		    params (Dict[str, Any]): Params value used by the operation.
		    payload (Dict[str, Any]): Payload value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'method', method )
			throw_if( 'endpoint', endpoint )
			self.method = str( method ).strip( ).upper( )
			self.endpoint = str( endpoint ).strip( )
			self.params = {
					key: value
					for key, value in (params or { }).items( )
					if value is not None and value != '' and value != [ ]
			}
			self.payload = {
					key: value
					for key, value in (payload or { }).items( )
					if value is not None and value != '' and value != [ ]
			}
			self.headers = self.build_management_headers( )
			self.url = f'{self.management_base_url.rstrip( "/" )}/{self.endpoint.lstrip( "/" )}'
			
			self.response = requests.request( method=self.method, url=self.url,
				headers=self.headers, params=self.params if self.params else None,
				json=self.payload if self.payload else None, timeout=3600 )
			self.response.raise_for_status( )
			
			if not self.response.content:
				self.result = { }
				return self.result
			
			try:
				self.result = self.response.json( )
				return self.result
			except Exception:
				self.result = self.response.text
				return self.result
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'execute_management_request( self, method: str, endpoint: str ) -> Any'
			Logger( ).write( ex )
			raise ex
	
	def normalize_collection( self, item: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""Normalize collection.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    item (Dict[str, Any]): Item value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'item', item )
			self.collection = item
			self.collection_id = (self.collection.get( 'collection_id' )
			                      or self.collection.get( 'id' ) or '')
			self.name = (self.collection.get( 'collection_name' ) or self.collection.get( 'name' )
			             or self.get_collection_name( self.collection_id ))
			
			return {
					'id': self.collection_id,
					'name': self.name,
					'display_name': self.name,
					'description': self.collection.get( 'collection_description', '' ),
					'status': self.collection.get( 'status', '' ),
					'file_counts': self.collection.get( 'documents_count', '' ),
					'usage_bytes': '',
					'collection_id': self.collection_id,
					'collection_name': self.name,
					'collection_description': self.collection.get( 'collection_description', '' ),
					'created_at': self.collection.get( 'created_at', '' ),
					'documents_count': self.collection.get( 'documents_count', '' ),
					'collection_type': self.collection.get( 'collection_type', '' ),
					'metadata': self.collection,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'normalize_collection( self, item: Dict[ str, Any ] ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def normalize_collection_list( self, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""Normalize collection list.
		
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
			self.response = response
			
			if isinstance( self.response, dict ):
				self.items = self.response.get( 'collections', [ ] )
			elif isinstance( self.response, list ):
				self.items = self.response
			else:
				self.items = [ ]
			
			self.collection_rows = [
					self.normalize_collection( item )
					for item in self.items
					if isinstance( item, dict )
			]
			return self.collection_rows
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'normalize_collection_list( self, response: Any ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( ex )
			raise ex
	
	def get_text_output( self, response: Any ) -> Any:
		"""Get text output.
		
		Purpose:
		    Returns normalized information for the VectorStores component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Args:
		    response (Any): Response value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if response is None:
				return None
			
			self.output_text = getattr( response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			self.output_text = getattr( response, 'text', None )
			if self.output_text:
				return self.output_text
			
			if isinstance( response, dict ):
				self.output_text = response.get( 'output_text' ) or response.get( 'text' )
				if self.output_text:
					return self.output_text
			
			return response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'get_text_output( self, response: Any ) -> Any'
			Logger( ).write( ex )
			raise ex
	
	def create( self, name: str, model: str = None, description: str = None,
			index_configuration: Dict[ str, Any ] = None,
			chunk_configuration: Dict[ str, Any ] = None,
			field_definitions: List[ Dict[ str, Any ] ] = None,
			**kwargs: Any ) -> Dict[ str, Any ]:
		"""Create.
		
		Purpose:
		    Performs the VectorStores.create workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    name (str): Name value used by the operation.
		    model (str): Model value used by the operation.
		    description (str): Description value used by the operation.
		    index_configuration (Dict[str, Any]): Index configuration value used by the operation.
		    chunk_configuration (Dict[str, Any]): Chunk configuration value used by the operation.
		    field_definitions (List[Dict[str, Any]]): Field definitions value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'name', name )
			self.name = str( name ).strip( )
			self.description = description
			self.model = model
			self.index_configuration = index_configuration
			self.chunk_configuration = chunk_configuration
			self.field_definitions = field_definitions
			self.extra_kwargs = kwargs or { }
			self.payload = {
					'collection_name': self.name,
					'collection_description': self.description,
					'index_configuration': self.index_configuration,
					'chunk_configuration': self.chunk_configuration,
					'field_definitions': self.field_definitions,
			}
			
			if self.model and self.index_configuration is None:
				self.payload[ 'index_configuration' ] = { 'model_name': self.model }
			
			for key, value in self.extra_kwargs.items( ):
				if value is not None:
					self.payload[ key ] = value
			
			self.result = self.execute_management_request( method='POST', endpoint='/collections',
				payload=self.payload )
			return self.normalize_collection( self.result )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'create( self, name: str, model: str=None ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def list( self, team_id: str = None, limit: int = None, order: str = None,
			sort_by: str = None, pagination_token: str = None,
			filter: str = None ) -> List[ Dict[ str, Any ] ]:
		"""List.
		
		Purpose:
		    Performs the VectorStores.list workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    team_id (str): Team id value used by the operation.
		    limit (int): Limit value used by the operation.
		    order (str): Order value used by the operation.
		    sort_by (str): Sort by value used by the operation.
		    pagination_token (str): Pagination token value used by the operation.
		    filter (str): Filter value used by the operation.
		
		Returns:
		    List[Dict[str, Any]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.team_id = team_id
			self.limit = limit
			self.order = order
			self.sort_by = sort_by
			self.pagination_token = pagination_token
			self.filter = filter
			self.params = {
					'team_id': self.team_id,
					'limit': self.limit,
					'order': self.order,
					'sort_by': self.sort_by,
					'pagination_token': self.pagination_token,
					'filter': self.filter,
			}
			self.result = self.execute_management_request( method='GET', endpoint='/collections',
				params=self.params )
			return self.normalize_collection_list( self.result )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'list( self ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, store_id: str, team_id: str = None ) -> Dict[ str, Any ]:
		"""Retrieve.
		
		Purpose:
		    Performs the VectorStores.retrieve workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    team_id (str): Team id value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.team_id = team_id
			self.params = { 'team_id': self.team_id }
			self.result = self.execute_management_request( method='GET',
				endpoint=f'/collections/{self.collection_id}', params=self.params )
			return self.normalize_collection( self.result )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'retrieve( self, store_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def update( self, store_id: str, name: str = None, description: str = None,
			chunk_configuration: Dict[ str, Any ] = None,
			index_configuration: Dict[ str, Any ] = None,
			field_definitions: List[ Dict[ str, Any ] ] = None,
			team_id: str = None, **kwargs: Any ) -> Dict[ str, Any ]:
		"""Update.
		
		Purpose:
		    Performs the VectorStores.update workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    name (str): Name value used by the operation.
		    description (str): Description value used by the operation.
		    chunk_configuration (Dict[str, Any]): Chunk configuration value used by the operation.
		    index_configuration (Dict[str, Any]): Index configuration value used by the operation.
		    field_definitions (List[Dict[str, Any]]): Field definitions value used by the operation.
		    team_id (str): Team id value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.name = name
			self.description = description
			self.chunk_configuration = chunk_configuration
			self.index_configuration = index_configuration
			self.field_definitions = field_definitions
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.params = { 'team_id': self.team_id }
			self.payload = {
					'collectionName': self.name,
					'collectionDescription': self.description,
					'chunkConfiguration': self.chunk_configuration,
					'indexConfiguration': self.index_configuration,
					'fieldDefinitions': self.field_definitions,
			}
			
			for key, value in self.extra_kwargs.items( ):
				if value is not None:
					self.payload[ key ] = value
			
			self.result = self.execute_management_request( method='PUT',
				endpoint=f'/collections/{self.collection_id}', params=self.params,
				payload=self.payload )
			return self.normalize_collection( self.result )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'update( self, store_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def delete( self, store_id: str, team_id: str = None ) -> Dict[ str, Any ]:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    team_id (str): Team id value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.team_id = team_id
			self.params = { 'team_id': self.team_id }
			self.result = self.execute_management_request( method='DELETE',
				endpoint=f'/collections/{self.collection_id}', params=self.params )
			return {
					'id': self.collection_id,
					'collection_id': self.collection_id,
					'deleted': True,
					'object': 'collection.deleted',
					'metadata': self.result,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, store_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def add_document( self, store_id: str, file_id: str,
			fields: Dict[ str, Any ] = None, team_id: str = None ) -> Dict[ str, Any ]:
		"""Add document.
		
		Purpose:
		    Performs the VectorStores.add_document workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		    fields (Dict[str, Any]): Fields value used by the operation.
		    team_id (str): Team id value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.file_id = str( file_id ).strip( )
			self.fields = fields if fields is not None else { }
			self.team_id = team_id
			self.params = { 'team_id': self.team_id }
			self.payload = { 'fields': self.fields }
			self.result = self.execute_management_request( method='POST',
				endpoint=f'/collections/{self.collection_id}/documents/{self.file_id}',
				params=self.params, payload=self.payload )
			return {
					'collection_id': self.collection_id,
					'file_id': self.file_id,
					'added': True,
					'metadata': self.result,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'add_document( self, store_id: str, file_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def list_documents( self, store_id: str, team_id: str = None, limit: int = None,
			order: str = None, sort_by: str = None, pagination_token: str = None,
			name: str = None, filter: str = None ) -> Any:
		"""List documents.
		
		Purpose:
		    Performs the VectorStores.list_documents workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    team_id (str): Team id value used by the operation.
		    limit (int): Limit value used by the operation.
		    order (str): Order value used by the operation.
		    sort_by (str): Sort by value used by the operation.
		    pagination_token (str): Pagination token value used by the operation.
		    name (str): Name value used by the operation.
		    filter (str): Filter value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.team_id = team_id
			self.limit = limit
			self.order = order
			self.sort_by = sort_by
			self.pagination_token = pagination_token
			self.name = name
			self.filter = filter
			self.params = {
					'team_id': self.team_id,
					'limit': self.limit,
					'order': self.order,
					'sort_by': self.sort_by,
					'pagination_token': self.pagination_token,
					'name': self.name,
					'filter': self.filter,
			}
			self.result = self.execute_management_request( method='GET',
				endpoint=f'/collections/{self.collection_id}/documents', params=self.params )
			return self.result
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'list_documents( self, store_id: str ) -> Any'
			Logger( ).write( ex )
			raise ex
	
	def retrieve_document( self, store_id: str, file_id: str,
			team_id: str = None ) -> Any:
		"""Retrieve document.
		
		Purpose:
		    Performs the VectorStores.retrieve_document workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		    team_id (str): Team id value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.file_id = str( file_id ).strip( )
			self.team_id = team_id
			self.params = { 'team_id': self.team_id }
			self.result = self.execute_management_request(
				method='GET',
				endpoint=f'/collections/{self.collection_id}/documents/{self.file_id}',
				params=self.params
			)
			return self.result
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'retrieve_document( self, store_id: str, file_id: str ) -> Any'
			Logger( ).write( ex )
			raise ex
	
	def regenerate_document( self, store_id: str, file_id: str,
			team_id: str = None ) -> Dict[ str, Any ]:
		"""Regenerate document.
		
		Purpose:
		    Performs the VectorStores.regenerate_document workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		    team_id (str): Team id value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.file_id = str( file_id ).strip( )
			self.team_id = team_id
			self.params = { 'team_id': self.team_id }
			self.result = self.execute_management_request(
				method='PATCH',
				endpoint=f'/collections/{self.collection_id}/documents/{self.file_id}',
				params=self.params
			)
			return {
					'collection_id': self.collection_id,
					'file_id': self.file_id,
					'regenerated': True,
					'metadata': self.result,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'regenerate_document( self, store_id: str, file_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def remove_document( self, store_id: str, file_id: str,
			team_id: str = None ) -> Dict[ str, Any ]:
		"""Remove document.
		
		Purpose:
		    Performs the VectorStores.remove_document workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_id (str): File id value used by the operation.
		    team_id (str): Team id value used by the operation.
		
		Returns:
		    Dict[str, Any]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.file_id = str( file_id ).strip( )
			self.team_id = team_id
			self.params = { 'team_id': self.team_id }
			self.result = self.execute_management_request(
				method='DELETE',
				endpoint=f'/collections/{self.collection_id}/documents/{self.file_id}',
				params=self.params
			)
			return {
					'collection_id': self.collection_id,
					'file_id': self.file_id,
					'removed': True,
					'metadata': self.result,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'remove_document( self, store_id: str, file_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def batch_get_documents( self, store_id: str, file_ids: List[ str ],
			team_id: str = None ) -> Any:
		"""Batch get documents.
		
		Purpose:
		    Performs the VectorStores.batch_get_documents workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    file_ids (List[str]): File ids value used by the operation.
		    team_id (str): Team id value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_ids', file_ids )
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.file_ids = file_ids
			self.team_id = team_id
			self.params = {
					'team_id': self.team_id,
					'file_ids': self.file_ids,
			}
			self.result = self.execute_management_request(
				method='GET',
				endpoint=f'/collections/{self.collection_id}/documents:batchGet',
				params=self.params
			)
			return self.result
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'batch_get_documents( self, store_id: str, file_ids: List[ str ] ) -> Any'
			Logger( ).write( ex )
			raise ex
	
	def search( self, prompt: str, store_id: str, model: str = 'grok-4-fast' ) -> Any:
		"""Search.
		
		Purpose:
		    Performs the VectorStores.search workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    store_id (str): Store id value used by the operation.
		    model (str): Model value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_id )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = str( prompt ).strip( )
			self.model = model
			self.store_id = str( store_id ).strip( )
			self.collection_id = self.get_collection_id( self.store_id )
			self.store_ids = [ self.collection_id ]
			self.collection_ids = [ self.collection_id ]
			self.client = Client( api_key=self.api_key )
			self.response = self.client.collections.search(
				query=self.prompt,
				collection_ids=self.collection_ids
			)
			return self.get_text_output( self.response )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'search( self, prompt: str, store_id: str, model: str ) -> Any'
			Logger( ).write( ex )
			raise ex
	
	def survey( self, prompt: str, store_ids: List[ str ],
			model: str = 'grok-4-fast' ) -> Any:
		"""Survey.
		
		Purpose:
		    Performs the VectorStores.survey workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    store_ids (List[str]): Store ids value used by the operation.
		    model (str): Model value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_ids', store_ids )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = str( prompt ).strip( )
			self.model = model
			self.store_ids = store_ids
			self.collection_ids = [ self.get_collection_id( store_id )
			                        for store_id in self.store_ids ]
			self.client = Client( api_key=self.api_key )
			self.response = self.client.collections.search( query=self.prompt,
				collection_ids=self.collection_ids )
			return self.get_text_output( self.response )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'survey( self, prompt: str, store_ids: List[ str ], model: str ) -> Any'
			Logger( ).write( ex )
			raise ex
	
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
				'management_key',
				'base_url',
				'management_base_url',
				'client',
				'model',
				'prompt',
				'response_format',
				'number',
				'content',
				'name',
				'description',
				'file_path',
				'file_name',
				'file_id',
				'file_ids',
				'store_ids',
				'store_id',
				'collection_ids',
				'collection_id',
				'documents',
				'collections',
				'request',
				'response',
				'result',
				'params',
				'payload',
				'headers',
				'team_id',
				'limit',
				'order',
				'sort_by',
				'pagination_token',
				'filter',
				'model_options',
				'order_options',
				'collection_sort_options',
				'document_sort_options',
				'get_collection_id',
				'get_collection_name',
				'build_management_headers',
				'execute_management_request',
				'normalize_collection',
				'normalize_collection_list',
				'get_text_output',
				'create',
				'list',
				'retrieve',
				'update',
				'delete',
				'add_document',
				'list_documents',
				'retrieve_document',
				'regenerate_document',
				'remove_document',
				'batch_get_documents',
				'search',
				'survey',
		]