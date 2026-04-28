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
from boogr import ErrorDialog, Error
import config as cfg
from openai import OpenAI
from xai_sdk.aio.image import ImageResponse
from xai_sdk import Client
from xai_sdk.chat import user, system, image, file

def encode_image( image_path: str ) -> str:
	"""Encodes a local image to a base64 string for vision API requests."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

def throw_if( name: str, value: object ) -> None:
	"""
	
		Purpose:
		--------
		Validate that a required value is not empty.
		
		Parameters:
		-----------
		name (str): Name of the argument being validated.
		value (object): Value to validate.
		
		Returns:
		--------
		None
		
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )
	
class Grok( ):
	"""
	
		Purpose:
		--------
		Base class for xAI (Grok) REST API functionality.
	
		This class provides:
			- API key and base URL management
			- Common request headers
			- Shared HTTP helpers for JSON and streaming requests
	
		Notes:
		------
		xAI exposes an OpenAI-compatible REST surface at:
			https://api.x.ai/v1
	
		All child capability classes (Chat, Images, Embeddings, Files, etc.)
		are expected to route through the helpers defined here.
	
	"""
	api_key: Optional[ str ]
	timeout: Optional[ float ]
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
		"""
			
			Purpose:
			--------
			Initialize the Grok (xAI) API client.
	
			Parameters:
			-----------
			cfg : object
				Configuration object providing API credentials and options.
			
		"""
		self.api_key = cfg.XAI_API_KEY
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
	"""
	
	    Purpose:
	    --------
	    Provides a wrapper around the xAI Responses API for text-generation,
	    retrieval-augmented, and tool-enabled Grok chat workflows.

	    Attributes:
	    -----------
	    include:
	        Optional Responses API include fields.

	    tool_choice:
	        Optional Responses API tool-choice policy.

	    previous_id:
	        Optional previous response identifier used for stateful Responses API calls.

	    conversation_id:
	        Optional Responses API conversation identifier.

	    parallel_tools:
	        Optional flag allowing parallel tool calls.

	    max_tools:
	        Optional maximum number of tool calls.

	    input:
	        Responses API input payload.

	    tools:
	        Normalized xAI Responses API tool definitions.

	    reasoning:
	        Optional xAI Responses API reasoning configuration.

	    allowed_domains:
	        Optional list of web-search allowed domains.

	    output_text:
	        Text output from the most recent response.

	    vector_store_ids:
	        xAI Collection identifiers used for collections/file search.

	    file_ids:
	        File identifiers retained for compatibility.

	    response:
	        Last Responses API response object.

	    Methods:
	    --------
	    generate_text:
	        Generates a text response through the xAI Responses API.

	    build_reasoning:
	        Builds a valid xAI Responses API reasoning object.

	    build_input:
	        Builds the Responses API input payload.

	    build_tools:
	        Builds valid built-in xAI Responses API tool objects.

	    build_tool_choice:
	        Builds a safe tool-choice value based on the final tool list.

	    build_include:
	        Filters include values to a conservative supported subset.

	    build_text_format:
	        Builds the Responses API text-format object.

	    build_request:
	        Builds the full Responses API request dictionary.

	    get_output_text:
	        Extracts output text from a completed response.

	    get_usage:
	        Returns usage metadata from the last response.

    """
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	previous_id: Optional[ str ]
	previous_response_id: Optional[ str ]
	conversation_id: Optional[ str ]
	parallel_tools: Optional[ bool ]
	max_tools: Optional[ int ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
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
	
	def __init__( self, model: str = 'grok-4.20', prompt: str = None, temperature: float = None,
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
		"""
		
			Purpose:
			--------
			Initialize a Grok Chat wrapper instance with optional Responses API defaults.

			Parameters:
			-----------
			model: str
				Default xAI model name.

			prompt: str
				Optional default user prompt.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			presense: float
				Backward-compatible misspelled presence penalty argument.

			presence: float
				Optional presence penalty value.

			store: bool
				Optional Responses API store flag.

			stream: bool
				Optional stream flag retained for compatibility.

			stops: List[ str ]
				Optional stop sequences retained for compatibility.

			response_format: Dict[ str, Any ]
				Optional Responses API text formatting object.

			number: int
				Optional number retained for compatibility.

			instruct: str
				Optional system/developer instructions.

			context: List[ Dict[ str, str ] ]
				Optional prior message context.

			allowed_domains: List[ str ]
				Optional web-search allowed-domain list.

			include: List[ str ]
				Optional include fields.

			tools: List[ Dict[ str, Any ] ]
				Optional tool definitions or selected tool-name dictionaries.

			max_tools: int
				Optional maximum tool-call count.

			tool_choice: str
				Optional tool-choice policy.

			file_path: str
				Optional file path retained for compatibility.

			background: bool
				Optional background flag retained for compatibility.

			is_parallel: bool
				Optional parallel tool-call flag.

			max_tokens: int
				Optional maximum output token count.

			frequency: float
				Optional frequency penalty value.

			input: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input payload.

			file_ids: List[ str ]
				Optional file identifiers retained for compatibility.

			previous_id: str
				Optional previous response identifier.

			conversation_id: str
				Optional Responses API conversation identifier.

			reasoning: Dict[ str, str ] | str
				Optional reasoning configuration.

			output_text: str
				Optional output text retained for compatibility.

			max_search_results: int
				Optional maximum search-result count retained for compatibility.

			content: str
				Optional content retained for compatibility.

			vector_store_ids: List[ str ]
				Optional xAI Collection identifiers used by collections/file search.

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
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
		self.request = { }
		self.messages = [ ]
		self.stream_requested = False
		self.background_requested = False
		self.collections = {
				'Federal Financial Regulations': 'collection_9195d847-03a1-443c-9240-294c64dd01e2',
				'Federal Financial Data': 'collection_e28cdcc2-a9e5-430a-bdf5-94fbaf44b6a4',
				'Explanatory Statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93',
				'Public Laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee',
		}
		self.files = {
				'Outlays.csv': 'file_b0a448b3-904a-40c7-bae1-64df657fde1c',
				'Authority.csv': 'file_c6ad236f-0c52-45f4-8883-d3be032d07c2',
				'Balances.csv': 'file_0f63d120-406f-49e6-97e5-7855f2cb26b5',
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return xAI text-capable model names used by the Text mode selector.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Model option names.

		'''
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
		'''
		
			Purpose:
			--------
			Return conservative xAI Responses API include options supported by Text mode.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Include option names.

		'''
		return [
				'web_search_call_output',
				'x_search_call_output',
				'code_execution_call_output',
				'collections_search_call_output',
				'attachment_search_call_output',
				'mcp_call_output',
				'inline_citations',
				'verbose_streaming',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return built-in xAI tool options that Text mode can configure.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Tool option names.

		'''
		return [
				'web_search',
				'x_search',
				'collections_search',
				'code_execution',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return supported tool-choice policies.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Tool-choice option names.

		'''
		return [ 'auto', 'required', 'none', ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return Text mode response-format options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Response-format names.

		'''
		return [
				'text',
				'json_object',
				'json_schema',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return xAI reasoning effort options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Reasoning effort names.

		'''
		return [
				'none',
				'low',
				'medium',
				'high',
				'xhigh',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return modality options retained for Text UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Modality names.

		'''
		return [ 'text', ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return media-resolution options retained for Text UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Media-resolution option names.

		'''
		return [ 'auto', ]
	
	def build_reasoning( self, reasoning: str | Dict[ str, str ] = None ) -> Dict[ str, str ] | None:
		"""
		
			Purpose:
			--------
			Create a valid xAI Responses API reasoning object from a string or dictionary.

			Parameters:
			-----------
			reasoning: str | Dict[ str, str ]
				Reasoning effort string or prebuilt reasoning dictionary.

			Returns:
			--------
			Dict[ str, str ] | None:
				Reasoning object or None.

		"""
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
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_reasoning( self, reasoning )'
			raise exception
	
	def build_input( self, prompt: str, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""
		
			Purpose:
			--------
			Create the Responses API input payload for text-generation requests.

			Parameters:
			-----------
			prompt: str
				User prompt submitted to the Responses API.

			context: List[ Dict[ str, str ] ]
				Prior user/assistant/developer/system messages.

			input_data: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input objects.

			Returns:
			--------
			List[ Dict[ str, Any ] ]:
				Responses API input payload.

		"""
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
										},
								],
						} )
			
			self.messages.append(
				{
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt,
								},
						],
				} )
			
			return self.messages
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_input( self, prompt, context, input_data )'
			raise exception
	
	def build_tools( self, tools: List[ Any ] = None, allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""
		
			Purpose:
			--------
			Normalize supported built-in xAI Responses API tool objects for Text mode.

			Parameters:
			-----------
			tools: List[ Any ]
				Tool strings or dictionaries selected by the application UI.

			allowed_domains: List[ str ]
				Optional list of allowed domains for web_search.

			vector_store_ids: List[ str ]
				Optional xAI Collection IDs used by collections_search.

			Returns:
			--------
			List[ Dict[ str, Any ] ] | None:
				Normalized tool dictionaries or None.

		"""
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
				
				if tool_type == 'web_search':
					built_tool = { 'type': 'web_search' }
					if len( self.allowed_domains ) > 0:
						built_tool[ 'allowed_domains' ] = self.allowed_domains
					
					self.built_tools.append( built_tool )
					continue
				
				if tool_type == 'x_search':
					self.built_tools.append( { 'type': 'x_search' } )
					continue
				
				if tool_type in [ 'collections_search', 'file_search' ]:
					if len( self.vector_store_ids ) == 0:
						continue
					
					self.built_tools.append(
						{
								'type': 'collections_search',
								'collection_ids': self.vector_store_ids,
						} )
					continue
				
				if tool_type in [ 'code_execution', 'code_interpreter' ]:
					self.built_tools.append( { 'type': 'code_execution' } )
					continue
			
			return self.built_tools if len( self.built_tools ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, tools, allowed_domains, vector_store_ids )'
			raise exception
	
	def build_tool_choice( self, tool_choice: str = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Build a safe tool-choice value based on the final normalized tool list.

			Parameters:
			-----------
			tool_choice: str
				Requested tool-choice policy.

			tools: List[ Dict[ str, Any ] ]
				Final normalized tool list.

			Returns:
			--------
			str | None:
				Tool-choice policy or None.

		"""
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
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_tool_choice( self, tool_choice, tools )'
			raise exception
	
	def build_include( self, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Filter include values to a conservative subset supported by selected tools.

			Parameters:
			-----------
			include: List[ str ]
				Requested include values.

			tools: List[ Dict[ str, Any ] ]
				Final normalized tool list.

			Returns:
			--------
			List[ str ] | None:
				Filtered include values or None.

		"""
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
				if name in [ 'inline_citations', 'verbose_streaming', 'mcp_call_output' ]:
					allowed.append( name )
					continue
				
				if name == 'web_search_call_output' and 'web_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'x_search_call_output' and 'x_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'code_execution_call_output' and 'code_execution' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'collections_search_call_output' and 'collections_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'attachment_search_call_output' and 'collections_search' in tool_types:
					allowed.append( name )
					continue
			
			return allowed if len( allowed ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_include( self, include, tools )'
			raise exception
	
	def build_text_format( self, format: Dict[ str, Any ] | str = None,
			response_schema: Any = None ) -> Dict[ str, Any ] | None:
		"""
		
			Purpose:
			--------
			Build or validate a Responses API text-format object.

			Parameters:
			-----------
			format: Dict[ str, Any ] | str
				Response format dictionary or response format name.

			response_schema: Any
				Optional JSON schema for json_schema output.

			Returns:
			--------
			Dict[ str, Any ] | None:
				Responses API text-format object or None.

		"""
		try:
			if format is None:
				return None
			
			if isinstance( format, dict ) and len( format ) > 0:
				if 'format' in format and isinstance( format.get( 'format' ), dict ):
					return format
				
				if 'type' in format:
					return { 'format': format }
				
				return None
			
			if isinstance( format, str ) and format.strip( ):
				value = format.strip( )
				if value == 'text':
					return { 'format': { 'type': 'text' } }
				
				if value == 'json_object':
					return { 'format': { 'type': 'json_object' } }
				
				if value == 'json_schema' and isinstance( response_schema, dict ):
					return { 'format': { 'type': 'json_schema', 'json_schema': response_schema } }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_text_format( self, format, response_schema )'
			raise exception
	
	def build_request( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None,
			background: bool = False, reasoning: str = None, include: List[ str ] = None,
			tools: List[ Any ] = None, allowed_domains: List[ str ] = None,
			previous_id: str = None, tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None, input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None,
			response_schema: Any = None ) -> Dict[ str, Any ]:
		"""
		
			Purpose:
			--------
			Create a normalized xAI Responses API request payload for text generation.
	
			Parameters:
			-----------
			prompt: str
				User prompt submitted to the model.
	
			model: str
				xAI model identifier.
	
			temperature: float
				Optional sampling temperature.
	
			format: Dict[ str, Any ]
				Optional Responses API text formatting object.
	
			top_p: float
				Optional nucleus sampling value.
	
			frequency: float
				Optional frequency penalty.
	
			max_tools: int
				Optional maximum number of tool calls.
	
			presence: float
				Optional presence penalty.
	
			max_tokens: int
				Optional maximum output token count.
	
			store: bool
				Optional flag controlling whether xAI stores the response.
	
			stream: bool
				Optional stream flag retained for compatibility. This non-streaming wrapper path
				does not send stream=True.
	
			instruct: str
				Optional system or developer instructions.
	
			background: bool
				Optional background flag retained for compatibility. This immediate wrapper path
				does not send background=True.
	
			reasoning: str
				Optional reasoning effort value.
	
			include: List[ str ]
				Optional Responses API include fields.
	
			tools: List[ Any ]
				Optional tool dictionaries or tool names.
	
			allowed_domains: List[ str ]
				Optional web_search allowed-domain filters.
	
			previous_id: str
				Optional previous response ID.
	
			tool_choice: str
				Optional tool-choice policy.
	
			is_parallel: bool
				Optional flag allowing parallel tool calls.
	
			context: List[ Dict[ str, str ] ]
				Optional conversation context.
	
			input_data: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input items.
	
			vector_store_ids: List[ str ]
				Optional xAI Collection IDs for collections_search.
	
			conversation_id: str
				Optional Responses API conversation identifier.
	
			response_schema: Any
				Optional JSON schema for structured output.
	
			Returns:
			--------
			Dict[ str, Any ]:
				Responses API request dictionary.
	
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.model = model
			self.prompt = prompt
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store_messages = store
			self.stream = stream
			self.background = background
			self.instructions = instruct
			self.response_format = self.build_text_format( format, response_schema=response_schema )
			self.max_tools = max_tools
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			self.previous_id = previous_id if isinstance( previous_id, str ) else None
			self.previous_response_id = self.previous_id
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
			
			if self.reasoning is not None and self.model == 'grok-4.20-multi-agent':
				self.request[ 'reasoning' ] = self.reasoning
			
			if isinstance( self.max_output_tokens, int ) and self.max_output_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_output_tokens
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None:
				self.request[ 'top_p' ] = self.top_percent
			
			if self.frequency_penalty is not None:
				self.request[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None:
				self.request[ 'presence_penalty' ] = self.presence_penalty
			
			if self.store_messages is not None:
				self.request[ 'store' ] = self.store_messages
			
			# Stream and background are retained on self for layout/UI parity. This path returns final text.
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
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_request( self, **kwargs )'
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""
		
			Purpose:
			--------
			Return text output from the last completed Responses API call.

			Parameters:
			-----------
			None

			Returns:
			--------
			str | None:
				Output text when available.

		"""
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
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> str | None'
			raise exception
	
	def get_usage( self ) -> Any:
		"""
		
			Purpose:
			--------
			Return usage metadata from the last Responses API call.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				Usage metadata when available.

		"""
		try:
			if self.response is None:
				return None
			
			return getattr( self.response, 'usage', None )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_usage( self ) -> Any'
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, top_k: int = None,
			frequency: float = None, max_tools: int = None, presence: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = None,
			instruct: str = None, background: bool = False, reasoning: str = None,
			include: List[ str ] = None, tools: List[ Any ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None,
			tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None, input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None,
			response_format: Dict[ str, Any ] | str = None, response_schema: Any = None,
			number: int = None, modalities: List[ str ] = None, media_resolution: str = None,
			content: str = None, urls: List[ str ] = None, max_urls: int = None,
			safety_profile: str = None, **kwargs: Any ) -> str | None:
		"""
		
			Purpose:
			--------
			Generate a text response through the xAI Responses API.

			Parameters:
			-----------
			prompt: str
				User prompt submitted to the Responses API.

			model: str
				xAI model name.

			temperature: float
				Optional sampling temperature.

			format: Dict[ str, Any ]
				Optional Responses API text formatting object.

			top_p: float
				Optional nucleus sampling value.

			top_k: int
				Optional top-k value retained for compatibility.

			frequency: float
				Optional frequency penalty value.

			max_tools: int
				Optional maximum number of tool calls.

			presence: float
				Optional presence penalty value.

			max_tokens: int
				Optional maximum output token value.

			store: bool
				Optional Responses API store flag.

			stream: bool
				Optional Responses API stream flag. This non-streaming wrapper path does
				not send stream=True.

			instruct: str
				Optional system or developer instructions.

			background: bool
				Optional background execution flag. This immediate wrapper path does not
				send background=True.

			reasoning: str
				Optional reasoning effort value.

			include: List[ str ]
				Optional include fields returned by the Responses API.

			tools: List[ Any ]
				Optional built-in tool names or definitions.

			allowed_domains: List[ str ]
				Optional web-search domain allowlist.

			previous_id: str
				Optional previous response identifier.

			tool_choice: str
				Optional tool-choice mode.

			is_parallel: bool
				Optional parallel tool-call flag.

			context: List[ Dict[ str, str ] ]
				Optional prior conversation context.

			input_data: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input payload.

			vector_store_ids: List[ str ]
				Optional xAI Collection identifiers used by collections_search.

			conversation_id: str
				Optional Responses API conversation identifier.

			response_format: Dict[ str, Any ] | str
				Optional Boo UI response-format value.

			response_schema: Any
				Optional structured-output JSON schema.

			number: int
				Optional number retained for UI compatibility.

			modalities: List[ str ]
				Optional modalities retained for UI compatibility.

			media_resolution: str
				Optional media resolution retained for UI compatibility.

			content: str
				Optional content retained for UI compatibility.

			urls: List[ str ]
				Optional URLs retained for UI compatibility.

			max_urls: int
				Optional maximum URL count retained for UI compatibility.

			safety_profile: str
				Optional safety profile retained for UI compatibility.

			**kwargs: Any
				Additional provider-neutral UI arguments.

			Returns:
			--------
			str | None
				Assistant output text when available.

		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.client = OpenAI( api_key=self.api_key, base_url=self.base_url )
			self.number = number
			self.top_k = top_k
			self.modalities = modalities if modalities is not None else [ ]
			self.media_resolution = media_resolution
			self.content = content
			self.urls = urls if urls is not None else [ ]
			self.max_urls = max_urls
			self.safety_profile = safety_profile
			self.extra_kwargs = kwargs or { }
			self.stream_requested = bool( stream )
			self.background_requested = bool( background )
			self.request = self.build_request( prompt=prompt, model=model,
				temperature=temperature, format=response_format or format, top_p=top_p,
				frequency=frequency, max_tools=max_tools, presence=presence,
				max_tokens=max_tokens, store=store, stream=False, instruct=instruct,
				background=False, reasoning=reasoning, include=include, tools=tools,
				allowed_domains=allowed_domains, previous_id=previous_id,
				tool_choice=tool_choice, is_parallel=is_parallel, context=context,
				input_data=input_data, vector_store_ids=vector_store_ids,
				conversation_id=conversation_id, response_schema=response_schema )
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', None )
			self.previous_response_id = self.previous_id
			self.output_text = self.get_output_text( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str ) -> str | None'
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]:
		"""
		
			Purpose:
			--------
			Return source/citation records from the current response when available.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ Dict[ str, Any ] ]:
				Source dictionaries.

		"""
		try:
			if self.response is None:
				return [ ]
			
			self.sources = [ ]
			output = getattr( self.response, 'output', None )
			if isinstance( output, list ):
				for item in output:
					item_type = getattr( item, 'type', None )
					
					if item_type in [ 'web_search_call', 'x_search_call' ]:
						action = getattr( item, 'action', None )
						raw_sources = getattr( action, 'sources', None ) if action else None
						if isinstance( raw_sources, list ):
							for source in raw_sources:
								self.sources.append(
									{
											'title': getattr( source, 'title', None ),
											'url': getattr( source, 'url', None ),
											'snippet': getattr( source, 'snippet', None ),
											'file_id': None,
									} )
					
					if item_type in [ 'collections_search_call', 'file_search_call' ]:
						results = getattr( item, 'results', None )
						if isinstance( results, list ):
							for result in results:
								self.sources.append(
									{
											'title': getattr( result, 'file_name',
												None ) or getattr(
												result, 'title', None ),
											'url': None,
											'snippet': getattr( result, 'text', None ),
											'file_id': getattr( result, 'file_id', None ),
									} )
			
			return self.sources
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]'
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Member names.

		'''
		return [
				'api_key',
				'base_url',
				'client',
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
				'reasoning',
				'allowed_domains',
				'max_search_results',
				'output_text',
				'vector_store_ids',
				'file_ids',
				'response',
				'file_path',
				'model_options',
				'include_options',
				'tool_options',
				'choice_options',
				'format_options',
				'reasoning_options',
				'modality_options',
				'media_options',
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
				'get_grounding_sources',
		]
	
class TTS( Grok ):
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
	client: Optional[ Client ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	language: Optional[ str ]
	prompt: Optional[ str ]
	
	def __init__( self, model: str='grok-3-mini-fast' ):
		'''

	        Purpose:
	        --------
	        Constructor to  create_small_embedding TTS objects

        '''
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.client = None
		self.model = model
		self.number = None
		self.prompt = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_completion_tokens = None
		self.store = None
		self.stream = None
		self.instructions = None
		self.messages = []
		self.audio_path = None
		self.response = None
		self.response_format = None
		self.speed = None
		self.voice = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI text-capable models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-4',
		         'grok-4-0709',
		         'grok-4-latest',
		         'grok-4-1-fast',
		         'grok-4-1-fast-reasoning',
		         'grok-4-1-fast-reasoning-latest',
		         'grok-4-1-fast-non-reasoning',
		         'grok-4-1-fast-non-reasoning-latest',
		         'grok-4-fast',
		         'grok-4-fast-reasoning',
		         'grok-4-fast-reasoning-latest',
		         'grok-4-fast-non-reasoning',
		         'grok-4-fast-non-reasoning-latest',
		         'grok-code-fast-1',
		         'grok-3',
		         'grok-3-latest',
		         'grok-3-mini',
		         'grok-3-fast',
		         'grok-3-fast-latest',
		         'grok-3-mini-fast',
		         'grok-3-mini-fast-latest' ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'alloy',
		         'ash',
		         'ballad',
		         'coral',
		         'echo',
		         'fable',
		         'onyx',
		         'nova',
		         'sage',
		         'shiver', ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of image formats

        '''
		return [ 'mp3',
		         'wav',
		         'aac',
		         'flac',
		         'opus',
		         'pcm' ]
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of floats
	        representing different audio speeds

        '''
		return [ 0.25,
		         1.0,
		         4.0 ]
	
	def generate( self, prompt: str, model: str='grok-3-mini', max_tokens: int=None,
			temperature: float=None, top_p: float=None, effort: str=None, format: str=None,
			store: bool=None, include: List[ str ]=None, instruct: str=None ):
		"""
		
			Purpose:
			--------
			Generate text using the xAI Responses API.

			If previous_response_id is set, the conversation will be
			continued server-side.

			Parameters:
			-----------
			prompt : str
				User input prompt.
			model : str | None
				Model identifier.
			max_output_tokens : int | None
				Maximum number of tokens in the response.
			temperature : float | None
			top_p : float | None
			include_reasoning : bool | None
				Whether to include encrypted reasoning content.
			reasoning_effort : str | None
				Reasoning effort level (grok-3-mini only).

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.max_output_tokens = max_tokens
			self.temperature = temperature
			self.top_percent = top_p
			self.instructions = instruct
			self.reasoning_effort = effort
			self.store = store
			self.response_format = format
			self.include = include
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {self.api_key}',
					'Content-Type': 'application/json', } )
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.user ) )
			self.chat = self.client.chat.create( model=self.model, messages=self.messages,
				store_messages=self.store, temperature=self.temperature, top_p=self.top_p,
				reasoning_effort=self.reasoning_effort, max_tokens=self.max_output_tokens,
				response_format=self.response_format )
			return self.chat
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'TTS'
			exception.method = 'generate( self, prompt: str, path: str ) -> str'
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
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'system_instructions',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'content',
		         'prompt',
		         'response',
		         'completion',
		         'file',
		         'path',
		         'messages',
		         'image_url',
		         'response_format',
		         'tools',
		         'name',
		         'id',
		         'description',
		         'generate_text',
		         'format_options',
		         'model_options',
		         'reasoning_effort',
		         'effort_options',
		         'speed_options',
		         'input_text', ]

class Transcription( Grok ):
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
	client: Optional[ Client ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	language: Optional[ str ]
	prompt: Optional[ str ]
	chat: Optional[ Any ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int =10000, store: bool=True,
			stream: bool=True, language: str='en', instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.client = None
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.language = language
		self.instructions = instruct
		self.prompt = None
		self.messages = [ ]
		self.model = None
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
		self.chat = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI text-capable models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-4',
		         'grok-4-0709',
		         'grok-4-latest',
		         'grok-4-1-fast',
		         'grok-4-1-fast-reasoning',
		         'grok-4-1-fast-reasoning-latest',
		         'grok-4-1-fast-non-reasoning',
		         'grok-4-1-fast-non-reasoning-latest',
		         'grok-4-fast',
		         'grok-4-fast-reasoning',
		         'grok-4-fast-reasoning-latest',
		         'grok-4-fast-non-reasoning',
		         'grok-4-fast-non-reasoning-latest',
		         'grok-code-fast-1',
		         'grok-3',
		         'grok-3-latest',
		         'grok-3-mini',
		         'grok-3-fast',
		         'grok-3-fast-latest',
		         'grok-3-mini-fast',
		         'grok-3-mini-fast-latest' ]
	
	@property
	def file_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of image formats

        '''
		return [ 'mp3',
		         'wav',
		         'aac',
		         'flac',
		         'opus',
		         'pcm' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			-------
			List[ str ] output  format options
			
		'''
		return [ 'json',
		         'text',
		         'srt',
		         'verbose_json',
		         'vtt',
		         'diarized_json' ]
	
	@property
	def language_options( self ):
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'English',
		         'Spanish',
		         'Tagalog',
		         'French',
		         'Japanese',
		         'German',
		         'Italian',
		         'Chinese' ]
	
	def transcribe( self, prompt: str, path: str, model: str='grok-3-mini-fast', language: str='en',
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, store: bool=None, stream: bool=None,
			instruct: str=None ) -> str:
		"""
		
			Purpose:
			----------
            Transcribe audio with Grok.
        
        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.model = model
			self.prompt = prompt
			self.language = language
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.prompt ) )
			self.client = Client( api_key=cfg.XAI_API_KEY )
			with open( path, 'rb' ) as self.audio_file:
				self.chat = self.client.chat.create( model=self.model,
					file=self.audio_file, messages=self.messages )
				self.response = self.chat.sample( )
			return self.response.output_text
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'transcribe(self, path)'
			raise ex
	
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
		return [ 'number',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'audio_file',
		         'messages',
		         'response_format',
		         'api_key',
		         'client',
		         'input_text',
		         'transcript', ]

class Translation( Grok ):
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
	client: Optional[ Client ]
	target_language: Optional[ str ]
	prompt: Optional[ str ]
	chat: Optional[ Any ]
	messages = Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self, model: str='grok-3-fast' ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.number = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_completion_tokens = None
		self.store = None
		self.stream = None
		self.instructions = None
		self.prompt = None
		self.audio_file = None
		self.response = None
		self.voice = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI text-capable models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-4',
		         'grok-4-0709',
		         'grok-4-latest',
		         'grok-4-1-fast',
		         'grok-4-1-fast-reasoning',
		         'grok-4-1-fast-reasoning-latest',
		         'grok-4-1-fast-non-reasoning',
		         'grok-4-1-fast-non-reasoning-latest',
		         'grok-4-fast',
		         'grok-4-fast-reasoning',
		         'grok-4-fast-reasoning-latest',
		         'grok-4-fast-non-reasoning',
		         'grok-4-fast-non-reasoning-latest',
		         'grok-code-fast-1',
		         'grok-3',
		         'grok-3-latest',
		         'grok-3-mini',
		         'grok-3-fast',
		         'grok-3-fast-latest',
		         'grok-3-mini-fast',
		         'grok-3-mini-fast-latest' ]
	
	@property
	def language_options( self ):
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'English',
		         'Spanish',
		         'Tagalog',
		         'French',
		         'Japanese',
		         'German',
		         'Italian',
		         'Chinese' ]
	
	@property
	def voice_options( self ):
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'alloy',
		         'ash',
		         'ballad',
		         'coral',
		         'echo',
		         'fable',
		         'onyx',
		         'nova',
		         'sage',
		         'shiver', ]
	
	def translate( self, text: str, path: str, number: int=None, temperature: float=None,
			top_p: float=None, frequency: float=None, presence: float=None, max_tokens: int=None,
			store: bool=None, stream: bool=None, instruct: str=None ) -> str | None:
		"""

	        Purpose
	        _______
	        Generates a translation given a string to an audio file
	
	
	        Parameters
	        ----------
	        text: str
	        path: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			self.number = number
			self.prompt = text
			self.audio_file = path
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_completion_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.instructions = instruct
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.prompt ) )
			self.client = Client( api_key=cfg.XAI_API_KEY )
			with open( self.audio_file, 'rb' ) as self.audio_file:
				self.chat = self.client.chat.create( model=self.model,
					file=self.audio_file, messages=self.messages )
			return self.chat
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Translation'
			exception.method = 'translate( self, text: str )'
			raise exception
	
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
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'audio_path',
		         'path',
		         'messages',
		         'response_format',
		         'tools',
		         'api_key',
		         'client',
		         'model',
		         'translate',
		         'model_options', ]

class Images( Grok ):
	"""
	
		Purpose:
		--------
		Provide image generation and image editing functionality using
		the xAI Images REST API.

		This class models the /images/generations and /images/edits
		endpoints exactly as exposed by xAI.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	
	"""
	model: Optional[ str ]
	aspect_ratio: Optional[ str ]
	resolution: Optional[ str ]
	response_format: Optional[ str ]
	client: Optional[ Client ]
	image: Optional[ image ]
	image_path: Optional[ str ]
	detail: Optional[ str ]
	response_format: Optional[ str ]
	response: Optional[ ImageResponse ]
	
	def __init__( self ):
		"""
		
			Purpose:
			--------
			Initialize the Images API client.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.client = None
		self.model = None
		self.aspect_ratio = None
		self.resolution = None
		self.quality = None
		self.detail = None
		self.response_format = None
		self.client = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI image generation models.

			Returns:
			--------
			List[str]
		
		"""
		return [ "grok-2-image-1212", 'grok-imagine-image' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'web_search',
		         'x_search',
		         'collections_search',
		         'code_interpreter' ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		return [ '1:1',
		         '3:4',
		         '4:3',
		         '9:16',
		         '16:9',
		         '2:3',
		         '3:2',
		         '9:19.5',
		         '19.5:9',
		         '9:20',
		         '20:9',
		         '1:2',
		         '2:1']
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported reasoning effort levels.

			Notes:
			------
			Only valid for model = 'grok-3-mini'.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'low', 'high' ]
	
	@property
	def size_options( self ) -> List[ str ]:
		return [ '1K',  '2K' ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		return [ 'low', 'medium', 'high' ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		return [ 'auto',
		         'low',
		         'high' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		return [ 'base64', 'url' ]
	
	@property
	def include_options( self ) -> List[ str ]:
		return [ 'web_search_call_output',
		         'x_search_call_output',
		         'code_execution_call_output',
		         'collections_search_call_output',
		         'attachment_search_call_output',
		         'mcp_call_output',
		         'inline_citations',
		         'verbose_streaming' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'auto', 'required', 'none' ]
	
	def create( self, prompt: str, model: str='grok-imagine-image', resolution: str=None,
			aspect_ratio: str=None,  format: str=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Generate one or more images from a text prompt.

			Parameters:
			-----------
			prompt : str
			model : str | None
			n : int | None
			aspect_ratio : str | None
			resolution : str | None
			quality : str | None
			style : str | None
			response_format : str | None

			Returns:
			--------
			List[dict]
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.model = model
			self.resolution = resolution
			self.aspect_ratio = aspect_ratio
			self.response_format = format
			self.client = Client( api_key=self.api_key )
			self.client.headers.update({ 'Authorization': f'Bearer {self.api_key}',
					'Content-Type': 'application/json', } )
			self.response = self.client.image.sample( prompt=self.prompt, resolution=self.resolution,
				model="grok-imagine-image", aspect_ratio=self.aspect_ratio,
				image_format=self.response_format )
			return self.response.base64
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'create( prompt: str, model: str )'
			raise ex
	
	def edit( self, image_path: str, prompt: str, model: str='grok-imagine-image',
			aspect_ratio: str=None, resolution: str=None, quality: str=None,
			response_format: str=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Edit an existing image using a text prompt and optional mask.

			Parameters:
			-----------
			image_path : str
			prompt : str
			mask_path : str | None
			model : str | None
			n : int | None
			aspect_ratio : str | None
			resolution : str | None
			quality : str | None
			style : str | None
			response_format : str | None

			Returns:
			--------
			List[dict]
		
		"""
		try:
			throw_if( 'image_path', image_path )
			throw_if( 'prompt', prompt )
			self.model = model
			self.image_path = image_path
			self.aspect_ratio = aspect_ratio
			self.resolution = resolution
			self.quality = quality
			self.response_format = response_format
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			with open( self.image_path, "rb" ) as f:
				image_data = base64.b64encode( f.read( ) ).decode( "utf-8" )
				self.response = self.client.image.sample( prompt=self.prompt, model=self.model,
					aspect_ratio=self.aspect_ratio, image_format=self.response_format,
					image_url=f"data:image/jpeg;base64,{image_data}", )
				return self.response.base64
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = 'edit( self, **kwarge ) -> str'
			raise ex
	
	def analyze( self, prompt: str, image_url: str, model: str='grok-4-1-fast-reasoning',
			max_output_tokens: int=10000, temperature: float=0.9, top_p: float=0.8,
			reasoning_effort: str='medium', detail: str='medium'  ):
		"""
		
			Purpose:
			--------
			Analyze an image (image understanding) using a text prompt and an image URL.

			This method uses xAI's multimodal input format via the Responses API and
			returns a text response describing or reasoning about the image.

			Parameters:
			-----------
			prompt : str
			image_url : str
			model : str | None
			max_output_tokens : int | None
			temperature : float | None
			top_p : float | None
			include_reasoning : bool | None
			reasoning_effort : str | None
			store : bool
			previous_response_id : str | None

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( "prompt", prompt )
			throw_if( "image_url", image_url )
			self.model = model
			self.prompt = prompt
			self.image_url = image_url
			self.max_output_tokens = max_output_tokens
			self.temperature = temperature
			self.top_percent = top_p
			self.detail = detail
			self.reasoning_effort = reasoning_effort
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {self.api_key}',
					'Content-Type': 'application/json', } )
			chat_response = self.client.chat.create( model=self.model )
			chat_response.append( user( self.prompt,
				image( image_url=self.image_url, detail=self.detail ) ) )
			image_respose = chat_response.sample()
			return image_respose.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'analyze( prompt: str, image_url: str  )'
			raise ex

class Files( Grok ):
	"""
	
		Purpose:
		--------
		Provide file upload, retrieval, listing, deletion, and
		file-based querying functionality using the xAI (Grok) REST API.

		This class manages file storage and enables file-based chat
		via the Responses API.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	
	"""
	client: Optional[ Client ]
	prompt: Optional[ str ]
	file_name: Optional[ str ]
	response_format: Optional[ str ]
	instructions: Optional[ str ]
	file_path: Optional[ str ]
	file_paths: Optional[ List[ str ] ]
	file_names: Optional[ List[ str ] ]
	file_id: Optional[ str ]
	purpose: Optional[ str ]
	content: Optional[ List[ Dict[ str, Any ] ] ]
	file_ids: Optional[ List[ str ] ]
	documents: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		"""
		
			Purpose:
			--------
			Initialize the Files capability.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.client = None
		self.model = None
		self.instructions = None
		self.content = None
		self.prompt = None
		self.response = None
		self.file_id = None
		self.file_path = None
		self.file_Name = None
		self.input = None
		self.purpose = None
		self.documents = \
		{
			'AccountBalances.csv': 'file_4731bb8c-d8ff-48c0-9dae-3092fbcab214',
			'SF133.csv': 'file_41037cc2-e1f4-4cce-b25a-5c1d1f0172b2',
			'Authority.csv': 'file_cbde06d5-988b-483f-880c-441613bfe54f',
			'Outlays.csv': 'file_78479189-7d47-4edb-9abc-2931172430e9'
		}

	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return list of efficient file interaction models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-4', 'grok-4-0709', 'grok-4-latest', 'grok-4-1-fast',
		         'grok-4-1-fast-reasoning', 'grok-4-1-fast-reasoning-latest',
		         'grok-4-1-fast-non-reasoning', 'grok-4-1-fast-non-reasoning-latest', 'grok-4-fast',
		         'grok-4-fast-reasoning', 'grok-4-fast-reasoning-latest',
		         'grok-4-fast-non-reasoning', 'grok-4-fast-non-reasoning-latest',
		         'grok-code-fast-1', 'grok-3', 'grok-3-latest', 'grok-3-mini', 'grok-3-fast',
		         'grok-3-fast-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest' ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return list of efficient file interaction models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'code_execution()' ]
	
	@property
	def include_options( self ) -> List[ str ]:
		return [ 'web_search_call_output',
		         'x_search_call_output',
		         'code_execution_call_output',
		         'collections_search_call_output',
		         'attachment_search_call_output',
		         'mcp_call_output',
		         'inline_citations',
		         'verbose_streaming' ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported reasoning effort levels.

			Notes:
			------
			Only valid for model = 'grok-3-mini'.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'low', 'high' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'auto', 'required', 'none' ]
	
	def upload( self, filepath: str, filename: str ) -> None:
		"""
		
			Purpose:
			--------
			Upload a local file to xAI file storage.

			Parameters:
			-----------
			filepath : str
			filename : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'filename', filename )
			self.file_path = filepath
			self.file_name = filename
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
				'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.client.files.upload( file=open( self.file_path, mode='rb' ),
				filename=self.file_name )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'upload( self, filepath: str, filename: str ) -> None'
			raise ex
	
	def list( self ) -> ListFilesResponse | None:
		"""
		
			Purpose:
			--------
			List all stored files.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[dict]
		
		"""
		try:
			self.client = Client( api_key=self.api_key )
			files_response = self.client.files.list( )
			return files_response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'list( self ) -> List[ Any ]'
			raise ex
	
	def retrieve( self, file_id: str ) -> Any | None:
		"""
		
			Purpose:
			--------
			Retrieve metadata for a specific file.

			Parameters:
			-----------
			file_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			metadata = self.client.files.get( file_id=self.file_id )
			return metadata
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id: str ) -> Any | None'
			raise ex
	
	def summarize( self, filepath: str, filename: str, prompt: str, model: str='grok-4-fast',
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, store: bool=None,
			stream: bool=None, instruct: str=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Chat with an uploaded file by attaching it to a Responses API
			request and asking a question about its contents.

			Parameters:
			-----------
			file_id : str
			prompt : str
			model : str | None
			max_output_tokens : int | None
			temperature : float | None
			top_p : float | None
			store : bool
			previous_response_id : str | None

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'filename', filename )
			throw_if( 'prompt', prompt )
			self.model = model
			self.prompt = prompt
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.user ) )
			self.file_path = filepath
			self.filename = filename
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.file = self.client.files.upload( open( self.file_path, 'rb' ),
				filename=self.file_name )
			self.chat = self.client.chat.create( model=self.model )
			self.chat.append( user( self.prompt, file( self.file.id ) ) )
			_response = self.chat.sample( )
			return _response.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'search( self, filepath: str, filename: str, prompt: str, model: str ) -> str'
			raise ex
	
	def search( self, filepath: str, filename: str, prompt: str, model: str='grok-4-fast',
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, store: bool=None,
			stream: bool=None, instruct: str=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Chat with an uploaded file by attaching it to a Responses API
			request and asking a question about its contents.

			Parameters:
			-----------
			file_id : str
			prompt : str
			model : str | None
			max_output_tokens : int | None
			temperature : float | None
			top_p : float | None
			store : bool
			previous_response_id : str | None

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'filename', filename )
			throw_if( 'prompt', prompt )
			self.model = model
			self.prompt = prompt
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.user ) )
			self.file_path = filepath
			self.filename = filename
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.file = self.client.files.upload( open( self.file_path, 'rb' ),
				filename=self.file_name )
			self.chat = self.client.chat.create( model=self.model )
			self.chat.append( user( self.prompt, file( self.file.id ) ) )
			_response = self.chat.sample( )
			return _response.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'search( self, filepath: str, filename: str, prompt: str, model: str ) -> str'
			raise ex
	
	def survey( self, filepaths: List[ str ], filenames: List[ str ], prompt: str,
			model: str='grok-4-fast', temperature: float=None, top_p: float=None,
			frequency: float=None, presence: float=None, max_tokens: int=None, store: bool=None,
			stream: bool=None, instruct: str=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Chat with an uploaded file by attaching it to a Responses API
			request and asking a question about its contents.

			Parameters:
			-----------
			file_id : str
			prompt : str
			model : str | None
			max_output_tokens : int | None
			temperature : float | None
			top_p : float | None
			store : bool
			previous_response_id : str | None

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( 'filepath', filepaths )
			throw_if( 'filename', filenames )
			throw_if( 'prompt', prompt )
			self.model = model
			self.prompt = prompt
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.user ) )
			self.file_paths = filepaths
			self.filenames = filenames
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.file = self.client.files.upload( open( self.file_path, 'rb' ),
				filename=self.file_name )
			self.chat = self.client.chat.create( model=self.model )
			self.chat.append( user( self.prompt, file( self.file.id ) ) )
			_response = self.chat.sample( )
			return _response.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = ('survey( self, filepaths: List[ str ], filenames: List[ str ], '
			             'prompt: str, model: str ) -> str')
			raise ex
	
	def extract( self, file_id: str ) -> bytes | None:
		"""
		
			Purpose:
			--------
			Retrieve raw content of a stored file.

			Parameters:
			-----------
			file_id : str

			Returns:
			--------
			bytes
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			_content = self.client.files.content( file_id=self.file_id )
			return _content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'extract( self, file_id: str ) -> bytes | None'
			raise ex
	
	def delete( self, file_id: str ) -> None:
		"""
		
			Purpose:
			--------
			Delete a file from storage.

			Parameters:
			-----------
			file_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.client.files.delete( file_id=self.file_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'delete( self, file_id: str ) -> None'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'documents',
		         'response',
		         'name',
		         'model',
		         'file_id',
		         'list',
		         'retrieve',
		         'search',
		         'delete',
		         'upload', ]

class VectorStores( Grok ):
	"""
	
		Purpose:
		--------
		Provide access to xAI Collections for grouping uploaded documents
		and reusing them across Responses-based interactions.

		This class manages collection metadata and membership only.
		Collections are referenced by ID in other APIs (e.g. Responses).

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	
	"""
	client: Optional[ Client ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	number: Optional[ int ]
	content: Optional[ str ]
	name: Optional[ str ]
	file_path: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	store_ids: Optional[ List[ str ] ]
	store_id: Optional[ str ]
	documents: Optional[ Dict[ str, str ] ]
	collections: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.client = None
		self.model = None
		self.content = None
		self.response = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.file_path = None
		self.file_name = None
		self.store_id = None
		self.collections = \
		{
				'Federal Financial Regulations': 'collection_9195d847-03a1-443c-9240-294c64dd01e2',
				'Federal Financial Data': 'collection_e28cdcc2-a9e5-430a-bdf5-94fbaf44b6a4',
				'Explanatory Statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93',
				'Public Laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee'
		}
		self.documents = \
		{
				'Outlays.csv': 'file_b0a448b3-904a-40c7-bae1-64df657fde1c',
				'Authority.csv': 'file_c6ad236f-0c52-45f4-8883-d3be032d07c2',
				'Balances.csv': 'file_0f63d120-406f-49e6-97e5-7855f2cb26b5'
		}
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return list of efficient file interaction models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-4', 'grok-4-0709', 'grok-4-latest', 'grok-4-1-fast',
		         'grok-4-1-fast-reasoning', 'grok-4-1-fast-reasoning-latest',
		         'grok-4-1-fast-non-reasoning', 'grok-4-1-fast-non-reasoning-latest', 'grok-4-fast',
		         'grok-4-fast-reasoning', 'grok-4-fast-reasoning-latest',
		         'grok-4-fast-non-reasoning', 'grok-4-fast-non-reasoning-latest',
		         'grok-code-fast-1', 'grok-3', 'grok-3-latest', 'grok-3-mini', 'grok-3-fast',
		         'grok-3-fast-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest' ]
	
	def create( self, name: str, model: str ) -> None:
		"""
		
			Purpose:
			--------
			Create a new collection with an initial set of files.

			Parameters:
			-----------
			name : str
			file_ids : List[str]
			description : str | None

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'name', name )
			throw_if( 'model', model )
			self.model = model
			self.file_name = name
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}',
			                              'Content-Type': 'application/json', } )
			response = self.client.collections.create( name=self.file_name, model_name=self.model )
			response.raise_for_status( )
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'create( self, name: str, model: str ) -> None'
			raise ex
	
	def list( self ) -> List[ Any ] | None:
		"""
		
			Purpose:
			--------
			List all collections accessible to the account.

			Returns:
			--------
			List[dict]
		
		"""
		try:
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json', } )
			_response = self.client.collections.list( )
			return list( _response )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'list( self ) -> List[ Any ] '
			raise ex
	
	def retrieve( self, store_id: str ) -> Any | None:
		"""
		
			Purpose:
			--------
			Retrieve metadata for a specific collection.

			Parameters:
			-----------
			collection_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'store_id', store_id )
			self.stores_id = store_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json', } )
			metadata = self.client.collections.get( collection_id=self.collection_id )
			return metadata
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'retrieve( self, stores_id: str ) -> Any '
			raise ex
	
	def search( self, prompt: str, store_id: str, model: str='grok-4-fast' ) -> str | None:
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
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_id )
			self.prompt = prompt
			self.model = model
			self.store_id = store_id
			self.vector_store_ids = [ store_id ]
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json', } )
			self.response = self.client.collections.search( query=self.prompt,
				collection_ids=[ self.store_id ],)
			return self.response.output_text
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'search( self, prompt: str, store_id: str, model: str ) -> str'
			raise ex
	
	def survey( self, prompt: str, store_ids: List[ str ], model: str='grok-4-fast' ) -> str | None:
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
			throw_if( 'prompt', prompt )
			throw_if( 'store_ids', store_ids )
			self.prompt = prompt
			self.model = model
			self.store_ids = store_ids
			self.tools = [
			{
				'text': 'file_search',
				'vector_store_ids': self.store_ids,
			} ]
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json', } )
			self.response = self.client.collections.search( query=self.prompt,
				collection_ids=self.store_ids, )
			return self.response.output_text
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'search( self, prompt: str, store_id: str, model: str ) -> str'
			raise ex
	
	def update( self, store_id: str, filepath: str, filename: str ) -> None:
		"""
		
			Purpose:
			--------
			Update collection membership by adding or removing files.

			Parameters:
			-----------
			collection_id : str
			add_file_ids : List[str] | None
			remove_file_ids : List[str] | None

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'filename', filename )
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			self.file_name = filename
			self.store_id = store_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update({ 'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json', } )
			with open( self.file_path, 'rb' ) as file:
				file_data = file.read( )
				_document = self.client.collections.upload_document( collection_id=self.store_id,
					name=self.file_name, data=file_data, content_type="text/html", )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'update( self, store_id: str, filepath: str, filename: str ) -> None'
			raise ex
			
	def delete( self, store_id: str ) -> None:
		"""
		
			Purpose:
			--------
			Delete a collection.

			Parameters:
			-----------
			collection_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			url = f'{self.base_url}/collections/{self.store_id}'
			self.client = Client( api_key=self.api_key )
			self.client.headers.update({'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json', } )
			response = self.client.delete( url, timeout=self.timeout )
			response.raise_for_status( )
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, store_id: str ) -> None'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'response',
		         'file_name',
		         'model',
		         'model_options',
		         'file_ids',
		         'store_ids',
		         'store_id',
		         'create',
		         'list',
		         'retrieve',
		         'search',
		         'update',
		         'delete',
		         'collections',
		         'documents' ]
	