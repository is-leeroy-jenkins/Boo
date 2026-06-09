'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                gemini.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="gemini.py" company="Terry D. Eppler">

	     gemini.py
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
    gemini.py
  </summary>
  ******************************************************************************************
'''
from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import Error, Logger
import json
import os
import requests
import PIL.Image
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from google import genai
from google.cloud import storage
from google.genai import types
from google.genai.pagers import Pager
from google.genai.types import (Part, GenerateContentConfig, ImageConfig, FunctionCallingConfig,
                                GenerateImagesConfig, GenerateVideosConfig, ThinkingConfig,
                                GeneratedImage, EmbedContentConfig, Content, ContentEmbedding,
                                Candidate, HttpOptions, GenerateImagesResponse, Field,
                                FileSearchStore, FileSearch,
                                GenerateContentResponse, GenerateVideosResponse, Image, File,
                                SpeakerVoiceConfig, VoiceConfig, SpeechConfig, Tool, ToolConfig,
                                GoogleSearch, UrlContext, SafetySetting, HarmCategory,
                                HarmBlockThreshold)

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

class Gemini( ):
	"""Gemini class.
	
	Purpose:
	    Defines the Gemini component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    number (Optional[int]): Stores number for the component runtime state.
	    google_api_key (Optional[str]): Stores google api key for the component runtime state.
	    gemini_api_key (Optional[str]): Stores gemini api key for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    api_version (Optional[str]): Stores api version for the component runtime state.
	    max_tokens (Optional[int]): Stores max tokens for the component runtime state.
	    temperature (Optional[float]): Stores temperature for the component runtime state.
	    top_p (Optional[float]): Stores top p for the component runtime state.
	    top_k (Optional[int]): Stores top k for the component runtime state.
	    candidate_count (Optional[int]): Stores candidate count for the component runtime state.
	    media_resolution (Optional[str]): Stores media resolution for the component runtime state.
	    response_modalities (Optional[List[str]]): Stores response modalities for the component runtime state.
	    stops (Optional[List[str]]): Stores stops for the component runtime state.
	    domains (Optional[List[str]]): Stores domains for the component runtime state.
	    frequency_penalty (Optional[float]): Stores frequency penalty for the component runtime state.
	    presence_penalty (Optional[float]): Stores presence penalty for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    content_response (Optional[GenerateContentResponse]): Stores content response for the component runtime state.
	    image_response (Optional[GenerateImagesResponse]): Stores image response for the component runtime state.
	    content_config (Optional[GenerateContentConfig]): Stores content config for the component runtime state.
	    function_config (Optional[FunctionCallingConfig]): Stores function config for the component runtime state.
	    thought_config (Optional[ThinkingConfig]): Stores thought config for the component runtime state.
	    genimg_config (Optional[GenerateImagesConfig]): Stores genimg config for the component runtime state.
	    image_config (Optional[ImageConfig]): Stores image config for the component runtime state.
	    tool_config (Optional[List[types.Tool]]): Stores tool config for the component runtime state.
	    tool_choice (Optional[str]): Stores tool choice for the component runtime state.
	    tools (Optional[List[str]]): Stores tools for the component runtime state."""
	number: Optional[ int ]
	google_api_key: Optional[ str ]
	gemini_api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	media_resolution: Optional[ str ]
	response_modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	domains: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ str ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	
	def __init__( self ):
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.model = None
		self.api_version = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.candidate_count = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.prompt = None
		self.response_format = None
		self.number = None
		self.response_modalities = [ ]
		self.stops = [ ]
		self.tools = [ ]

class Chat( Gemini ):
	"""Chat class.
	
	Purpose:
	    Defines the Chat component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    use_vertex (Optional[bool]): Stores use vertex for the component runtime state.
	    http_options (Optional[HttpOptions]): Stores http options for the component runtime state.
	    client (Optional[genai.Client]): Stores client for the component runtime state.
	    storage_client (Optional[storage.Client]): Stores storage client for the component runtime state.
	    contents (Optional[Union[str, List[str], List[Content]]]): Stores contents for the component runtime state.
	    image_uri (Optional[str]): Stores image uri for the component runtime state.
	    audio_uri (Optional[str]): Stores audio uri for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    files (Optional[List[str]]): Stores files for the component runtime state.
	    content_block (Optional[str]): Stores content block for the component runtime state.
	    context (Optional[List[Dict[str, Any]]]): Stores context for the component runtime state.
	    urls (Optional[List[str]]): Stores urls for the component runtime state.
	    max_urls (Optional[int]): Stores max urls for the component runtime state.
	    response_schema (Optional[Any]): Stores response schema for the component runtime state.
	    safety_profile (Optional[str]): Stores safety profile for the component runtime state.
	    safety_settings (Optional[List[SafetySetting]]): Stores safety settings for the component runtime state."""
	use_vertex: Optional[ bool ]
	http_options: Optional[ HttpOptions ]
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	contents: Optional[ Union[ str, List[ str ], List[ Content ] ] ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	files: Optional[ List[ str ] ]
	content_block: Optional[ str ]
	context: Optional[ List[ Dict[ str, Any ] ] ]
	urls: Optional[ List[ str ] ]
	max_urls: Optional[ int ]
	response_schema: Optional[ Any ]
	safety_profile: Optional[ str ]
	safety_settings: Optional[ List[ SafetySetting ] ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ):
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.api_version = None
		self.client = None
		self.content_config = None
		self.image_config = None
		self.function_tool_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_objects = None
		self.tools = [ ]
		self.response_modalities = [ ]
		self.files = [ ]
		self.http_options = { }
		self.number = None
		self.candidate_count = None
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.use_vertex = None
		self.instructions = None
		self.media_resolution = None
		self.tool_choice = None
		self.contents = None
		self.grounding_metadata = None
		self.content_block = None
		self.context = [ ]
		self.client = None
		self.storage_client = None
		self.content_response = None
		self.image_response = None
		self.image_uri = None
		self.audio_uri = None
		self.file_path = None
		self.stops = [ ]
		self.response_mime_type = None
		self.response_schema = None
		self.urls = [ ]
		self.max_urls = None
		self.safety_profile = None
		self.safety_settings = None
		self.file_search_store_names = [ ]
		self.include_server_side_tool_invocations = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ 'gemini-2.5-flash',
		         'gemini-2.5-flash-lite',
		         'gemini-2.5-pro',
		         'gemini-3-flash-preview',
		         'gemini-3.1-flash-lite-preview',
		         'gemini-3.1-pro-preview',
		         'gemini-2.0-flash',
		         'gemini-2.0-flash-lite' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'google_search',
		         'google_maps',
		         'url_context',
		         'file_search',
		         'code_execution' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Media options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'auto', 'any', 'none', 'validated' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ '', 'text', 'image', 'audio' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
		"""Get supported tools.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'model', model )
			self.model_name = str( model ).strip( ).lower( )
			self.options = [ 'google_search', 'url_context', 'file_search', 'code_execution' ]
			
			if self.supports_google_maps( self.model_name ):
				self.options.append( 'google_maps' )
			
			return self.options
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_supported_tools( self, model: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def supports_google_maps( self, model: str ) -> bool:
		"""Supports google maps.
		
		Purpose:
		    Performs the Chat.supports_google_maps workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'model', model )
			self.model_name = model.strip( ).lower( )
			self.maps_models = {
					'gemini-3.1-pro-preview',
					'gemini-3.1-flash-lite-preview',
					'gemini-3-flash-preview',
					'gemini-2.5-pro',
					'gemini-2.5-flash',
					'gemini-2.5-flash-lite',
					'gemini-2.0-flash'
			}
			return self.model_name in self.maps_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'supports_google_maps( self, model: str=None ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_urls( self, urls: List[ str ], max_urls: int = 10 ) -> List[ str ]:
		"""Build urls.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    urls (List[str]): Urls value used by the operation.
		    max_urls (int): Max urls value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'max_urls', max_urls )
			self.urls = urls if urls is not None else [ ]
			for url in urls:
				if url is None:
					continue
				
				self.url = url.strip( )
				if not self.url:
					continue
				
				self.urls.append( self.url )
			
			self.max_urls = max_urls
			if self.max_urls is not None:
				self.urls = self.urls[ : self.max_urls ]
			
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_urls( self, urls: List[ str ]=None, max_urls: int=None )'
			Logger( ).write( exception )
			raise exception
	
	def append_urls_to_content( self, content: str, urls: List[ str ] ) -> str | None:
		"""Append urls to content.
		
		Purpose:
		    Performs the Chat.append_urls_to_content workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    content (str): Content value used by the operation.
		    urls (List[str]): Urls value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.content_blocks = [ ]
			self.content_blocks.append( content.strip( ) )
			self.urls = urls
			if len( self.urls ) > 0:
				self.content_blocks.append( 'Reference URLs:\n' + '\n'.join( self.urls ) )
			
			return '\n\n'.join( self.content_blocks ) if len( self.content_blocks ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('append_urls_to_content( self, **kwargs ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def build_modalities( self, modalities: List[ str ] ) -> List[ str ] | None:
		"""Build modalities.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    modalities (List[str]): Modalities value used by the operation.
		
		Returns:
		    List[str] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.modalities = [ ]
			
			for modality in (modalities or [ ]):
				if modality is None:
					continue
				
				self.modality = str( modality ).strip( ).upper( )
				if self.modality in [ 'TEXT', 'IMAGE', 'AUDIO' ]:
					self.modalities.append( self.modality )
			
			return self.modalities if len( self.modalities ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_modalities( self, modalities: List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def build_reasoning( self, reasoning: str ) -> ThinkingConfig | None:
		"""Build reasoning.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    reasoning (str): Reasoning value used by the operation.
		
		Returns:
		    ThinkingConfig | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.reasoning = str( reasoning or '' ).strip( ).upper( )
			if not self.reasoning:
				return None
			
			if self.reasoning == 'THINKING_LEVEL_UNSPECIFIED':
				return None
			
			if self.reasoning not in [ 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH' ]:
				return None
			
			return ThinkingConfig( thinking_level=self.reasoning )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_reasoning( self, reasoning: str ) -> ThinkingConfig | None'
			Logger( ).write( exception )
			raise exception
	
	def build_safety_settings( self, safety_profile: str ) -> List[ SafetySetting ] | None:
		"""Build safety settings.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    safety_profile (str): Safety profile value used by the operation.
		
		Returns:
		    List[SafetySetting] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.safety_profile = str( safety_profile or '' ).strip( ).upper( )
			if not self.safety_profile:
				return None
			
			self.threshold = getattr( HarmBlockThreshold, self.safety_profile, None )
			if self.threshold is None:
				return None
			
			self.categories = [ ]
			for name in [
					'HARM_CATEGORY_HATE_SPEECH',
					'HARM_CATEGORY_HARASSMENT',
					'HARM_CATEGORY_SEXUALLY_EXPLICIT',
					'HARM_CATEGORY_DANGEROUS_CONTENT',
					'HARM_CATEGORY_CIVIC_INTEGRITY' ]:
				self.category = getattr( HarmCategory, name, None )
				if self.category is not None:
					self.categories.append( self.category )
			
			if len( self.categories ) == 0:
				return None
			
			return [ SafetySetting( category=category, threshold=self.threshold )
			         for category in self.categories ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_safety_settings( self, safety_profile: str )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
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
			if self.content_response is None:
				return None
			
			self.text = getattr( self.content_response, 'text', None )
			if isinstance( self.text, str ) and self.text.strip( ):
				return self.text.strip( )
			
			self.parts = getattr( self.content_response, 'parts', None )
			if self.parts:
				self.output = [ ]
				for part in self.parts:
					self.part_text = getattr( part, 'text', None )
					if isinstance( self.part_text, str ) and self.part_text.strip( ):
						self.output.append( self.part_text.strip( ) )
				
				if len( self.output ) > 0:
					return '\n'.join( self.output ).strip( )
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if self.candidates:
				self.output = [ ]
				for candidate in self.candidates:
					self.content = getattr( candidate, 'content', None )
					if self.content is None:
						continue
					
					for part in getattr( self.content, 'parts', None ) or [ ]:
						self.part_text = getattr( part, 'text', None )
						if isinstance( self.part_text, str ) and self.part_text.strip( ):
							self.output.append( self.part_text.strip( ) )
				
				if len( self.output ) > 0:
					return '\n'.join( self.output ).strip( )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def parse_response_schema( self, response_schema: Any ) -> Any:
		"""Parse response schema.
		
		Purpose:
		    Performs the Chat.parse_response_schema workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    response_schema (Any): Response schema value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if response_schema is None:
				return None
			
			if isinstance( response_schema, dict ):
				return response_schema
			
			if not isinstance( response_schema, str ):
				return response_schema
			
			self.schema_text = response_schema.strip( )
			if not self.schema_text:
				return None
			
			return json.loads( self.schema_text )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'parse_response_schema( self, response_schema: Any )'
			Logger( ).write( exception )
			raise exception
	
	def build_contents( self, prompt: str, content: str, context: List[ Any ] = None ) -> List[
		Content ]:
		"""Build contents.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    content (str): Content value used by the operation.
		    context (List[str]): Context value used by the operation.
		
		Returns:
		    List[Content]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = str( prompt ).strip( )
			self.context = context if context is not None else [ ]
			self.content_block = str( content or '' ).strip( )
			self.contents = [ ]
			
			for item in self.context:
				if item is None:
					continue
				
				if isinstance( item, Content ):
					self.contents.append( item )
					continue
				
				if not isinstance( item, dict ):
					continue
				
				role = str( item.get( 'role', 'user' ) or 'user' ).strip( )
				text = item.get( 'content', None )
				if text is None:
					continue
				
				text = str( text ).strip( )
				if not text:
					continue
				
				if role == 'assistant':
					self.contents.append( Content( role='model',
						parts=[ Part.from_text( text=text ) ] ) )
				else:
					self.contents.append( Content( role='user',
						parts=[ Part.from_text( text=text ) ] ) )
			
			self.user_text = self.prompt
			if self.content_block:
				self.user_text = f'{self.content_block}\n\n{self.user_text}'
			
			self.contents.append( Content( role='user',
				parts=[ Part.from_text( text=self.user_text ) ] ) )
			
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_contents( self, prompt: str, content: str, context: List[ Any ]=None )'
			Logger( ).write( exception )
			raise exception
	
	def capture_grounding_metadata( self ) -> None:
		"""Capture grounding metadata.
		
		Purpose:
		    Performs the Chat.capture_grounding_metadata workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.grounding_metadata = None
			
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if not self.candidates:
				return
			
			for candidate in self.candidates:
				self.metadata = getattr( candidate, 'grounding_metadata', None )
				if self.metadata is None:
					self.metadata = getattr( candidate, 'groundingMetadata', None )
				
				if self.metadata is not None:
					self.grounding_metadata = self.metadata
					return
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'capture_grounding_metadata( self )'
			Logger( ).write( exception )
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, str ] ]:
		"""Get grounding sources.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[Dict[str, str]]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.sources = [ ]
			
			if self.grounding_metadata is None:
				return self.sources
			
			self.chunks = getattr( self.grounding_metadata, 'grounding_chunks', None )
			if self.chunks is None:
				self.chunks = getattr( self.grounding_metadata, 'groundingChunks', None )
			
			if not self.chunks:
				return self.sources
			
			for chunk in self.chunks:
				self.web = getattr( chunk, 'web', None )
				if self.web is None and isinstance( chunk, dict ):
					self.web = chunk.get( 'web' )
				
				if self.web is None:
					continue
				
				if isinstance( self.web, dict ):
					self.uri = self.web.get( 'uri' ) or self.web.get( 'url' )
					self.title = self.web.get( 'title' ) or self.uri
				else:
					self.uri = getattr( self.web, 'uri', None )
					if self.uri is None:
						self.uri = getattr( self.web, 'url', None )
					
					self.title = getattr( self.web, 'title', None ) or self.uri
				
				if self.uri:
					self.sources.append(
						{
								'title': str( self.title or self.uri ),
								'url': str( self.uri ),
								'snippet': ''
						} )
			
			return self.sources
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_grounding_sources( self ) -> List[ Dict[ str, str ] ]'
			Logger( ).write( exception )
			raise exception
	
	def get_structured_history( self ) -> List[ Content ] | None:
		"""Get structured history.
		
		Purpose:
		    Returns normalized information for the Chat component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[Content] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.history = [ ]
			
			if self.contents is not None and isinstance( self.contents, list ):
				for item in self.contents:
					if isinstance( item, Content ):
						self.history.append( item )
			
			if self.content_response is not None:
				self.candidates = getattr( self.content_response, 'candidates', None )
				if self.candidates:
					for candidate in self.candidates:
						self.response_content = getattr( candidate, 'content', None )
						if isinstance( self.response_content, Content ):
							self.history.append( self.response_content )
							break
			
			return self.history if len( self.history ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_structured_history( self ) -> List[ Content ] | None'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: List[ str ], file_search_store_names: List[ str ] = None ) -> \
	List[ Tool ] | None:
		"""Build tools.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    tools (List[str]): Tools value used by the operation.
		    file_search_store_names (List[str]): File search store names value used by the operation.
		
		Returns:
		    List[Tool] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.tools = [
					str( tool ).strip( )
					for tool in (tools or [ ])
					if str( tool ).strip( )
			]
			
			self.file_search_store_names = file_search_store_names or [ ]
			self.tool_objects = [ ]
			
			if len( self.tools ) == 0:
				return None
			
			if 'google_search' in self.tools:
				self.tool_objects.append( Tool( google_search=GoogleSearch( ) ) )
			
			if 'url_context' in self.tools:
				self.tool_objects.append( Tool( url_context=UrlContext( ) ) )
			
			if 'file_search' in self.tools:
				throw_if( 'file_search_store_names', file_search_store_names )
				self.file_search_store_names = file_search_store_names
				self.tool_objects.append(
					Tool(
						file_search=FileSearch(
							file_search_store_names=self.file_search_store_names
						)
					)
				)
			
			if 'code_execution' in self.tools:
				self.tool_objects.append(
					Tool( code_execution=types.ToolCodeExecution( ) )
				)
			
			return self.tool_objects if len( self.tool_objects ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, tools, file_search_store_names )'
			Logger( ).write( exception )
			raise exception
	
	def build_tool_config( self, tool_choice: str, tools: List[ Tool ] ) -> ToolConfig | None:
		"""Build tool config.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    tool_choice (str): Tool choice value used by the operation.
		    tools (List[Tool]): Tools value used by the operation.
		
		Returns:
		    ToolConfig | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.tool_choice = str( tool_choice or '' ).strip( ).upper( )
			self.tool_objects = tools if tools is not None else [ ]
			
			if not self.tool_choice:
				return None
			
			if self.tool_choice == 'AUTO':
				return None
			
			if self.tool_choice not in [ 'ANY', 'NONE', 'VALIDATED' ]:
				return None
			
			if len( self.tool_objects ) == 0:
				raise ValueError( 'Gemini tool configuration requires at least one tool.' )
			
			return ToolConfig(
				function_calling_config=FunctionCallingConfig(
					mode=self.tool_choice
				)
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_tool_config( self, tool_choice, tools )'
			Logger( ).write( exception )
			raise exception
	
	def build_config( self, model: str = 'gemini-2.5-flash-lite', number: int = None,
			temperature: float = None, top_p: float = None, top_k: int = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			response_schema: Any = None, safety_profile: str = None,
			file_search_store_names: List[ str ] = None ) -> GenerateContentConfig:
		"""Build config.
		
		Purpose:
		    Builds the normalized data structure required by the Chat workflow. The function converts caller
		    input, session state, or provider-specific options into a stable shape that downstream API calls
		    and rendering code can consume safely.
		
		Args:
		    model (str): Model value used by the operation.
		    number (int): Number value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    top_k (int): Top k value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    response_format (str): Response format value used by the operation.
		    tools (List[str]): Tools value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    reasoning (str): Reasoning value used by the operation.
		    modalities (List[str]): Modalities value used by the operation.
		    media_resolution (str): Media resolution value used by the operation.
		    response_schema (Any): Response schema value used by the operation.
		    safety_profile (str): Safety profile value used by the operation.
		    file_search_store_names (List[str]): File search store names value used by the operation.
		
		Returns:
		    GenerateContentConfig: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'model', model )
			self.model = str( model ).strip( )
			
			self.number = number
			self.candidate_count = int( self.number or 0 )
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = int( top_k or 0 )
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = int( max_tokens or 0 )
			self.stops = stops if stops is not None else [ ]
			self.instructions = instruct
			self.response_mime_type = str( response_format or '' ).strip( )
			self.response_schema = self.parse_response_schema( response_schema )
			self.safety_settings = self.build_safety_settings( safety_profile )
			self.tool_choice = tool_choice
			self.media_resolution = str( media_resolution ).strip( ) if media_resolution else None
			self.file_search_store_names = file_search_store_names or [ ]
			
			self.tool_objects = self.build_tools(
				tools=tools,
				file_search_store_names=self.file_search_store_names
			)
			
			self.function_tool_config = self.build_tool_config(
				tool_choice=self.tool_choice,
				tools=self.tool_objects
			)
			
			self.response_modalities = self.build_modalities( modalities=modalities )
			self.thought_config = self.build_reasoning( reasoning )
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None and float( self.top_p ) > 0:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.config_kwargs[ 'top_k' ] = self.top_k
			
			if self.max_tokens > 0:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.candidate_count > 0:
				self.config_kwargs[ 'candidate_count' ] = self.candidate_count
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.frequency_penalty is not None:
				self.config_kwargs[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None:
				self.config_kwargs[ 'presence_penalty' ] = self.presence_penalty
			
			if self.stops is not None and len( self.stops ) > 0:
				self.config_kwargs[ 'stop_sequences' ] = self.stops
			
			if self.response_mime_type:
				self.config_kwargs[ 'response_mime_type' ] = self.response_mime_type
			
			if self.response_schema is not None:
				if isinstance( self.response_schema, dict ):
					self.config_kwargs[ 'response_json_schema' ] = self.response_schema
				else:
					self.config_kwargs[ 'response_schema' ] = self.response_schema
			
			if self.media_resolution is not None:
				self.config_kwargs[ 'media_resolution' ] = self.media_resolution
			
			if self.tool_objects is not None and len( self.tool_objects ) > 0:
				self.config_kwargs[ 'tools' ] = self.tool_objects
			
			if self.function_tool_config is not None:
				self.config_kwargs[ 'tool_config' ] = self.function_tool_config
			
			if self.safety_settings is not None and len( self.safety_settings ) > 0:
				self.config_kwargs[ 'safety_settings' ] = self.safety_settings
			
			if self.response_modalities is not None and len( self.response_modalities ) > 0:
				self.config_kwargs[ 'response_modalities' ] = self.response_modalities
			
			if self.thought_config is not None:
				self.config_kwargs[ 'thinking_config' ] = self.thought_config
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_config( self, model ) -> GenerateContentConfig'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			number: int = None, temperature: float = None, top_p: float = None,
			top_k: int = None, frequency: float = None, presence: float = None,
			max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			context: List[ Dict[ str, Any ] ] = None, content: str = None,
			urls: List[ str ] = None, max_urls: int = None, response_schema: Any = None,
			safety_profile: str = None, file_search_store_names: List[ str ] = None,
			stream: bool = False, stream_handler: Any = None ) -> str | None:
		"""Generate text.
		
		Purpose:
		    Generates provider output for the Chat workflow using validated model settings and request
		    inputs. The method coordinates request construction, provider execution, response capture, and
		    logged exception handling.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    number (int): Number value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    top_k (int): Top k value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    response_format (str): Response format value used by the operation.
		    tools (List[str]): Tools value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    reasoning (str): Reasoning value used by the operation.
		    modalities (List[str]): Modalities value used by the operation.
		    media_resolution (str): Media resolution value used by the operation.
		    context (List[Dict[str, Any]]): Context value used by the operation.
		    content (str): Content value used by the operation.
		    urls (List[str]): Urls value used by the operation.
		    max_urls (int): Max urls value used by the operation.
		    response_schema (Any): Response schema value used by the operation.
		    safety_profile (str): Safety profile value used by the operation.
		    file_search_store_names (List[str]): File search store names value used by the operation.
		    stream (bool): Stream value used by the operation.
		    stream_handler (Any): Stream handler value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.model = str( model or self.model or 'gemini-2.5-flash-lite' ).strip( )
			throw_if( 'model', self.model )
			
			self.gemini_api_key = cfg.GEMINI_API_KEY
			self.stream = bool( stream )
			self.urls = self.build_urls( urls=urls, max_urls=max_urls )
			self.content_block = self.append_urls_to_content( content=content, urls=self.urls )
			self.contents = self.build_contents( prompt=prompt, context=context,
				content=self.content_block )
			self.content_config = self.build_config( model=self.model, number=number,
				temperature=temperature, top_p=top_p, top_k=top_k, frequency=frequency,
				presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct,
				response_format=response_format, tools=tools, tool_choice=tool_choice,
				reasoning=reasoning, modalities=modalities, media_resolution=media_resolution,
				response_schema=response_schema, safety_profile=safety_profile,
				file_search_store_names=file_search_store_names )
			
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			if self.stream:
				self.stream_response = self.client.models.generate_content_stream(
					model=self.model, contents=self.contents, config=self.content_config )
				
				if stream_handler is not None:
					self.text_blocks = [ ]
					for chunk in self.stream_response:
						if chunk is None:
							continue
						
						self.chunk_text = getattr( chunk, 'text', None )
						if self.chunk_text is None or not str( self.chunk_text ):
							continue
						
						self.text_blocks.append( str( self.chunk_text ) )
						stream_handler( str( self.chunk_text ) )
					
					self.output_text = ''.join( self.text_blocks ).strip( )
					return self.output_text if self.output_text else None
				
				return self.stream_response
			
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			self.capture_grounding_metadata( )
			
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception

class Images( Gemini ):
	"""Images class.
	
	Purpose:
	    Defines the Images component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    client (Optional[genai.Client]): Stores client for the component runtime state.
	    aspect_ratio (Optional[str]): Stores aspect ratio for the component runtime state.
	    use_vertex (Optional[bool]): Stores use vertex for the component runtime state.
	    resolution (Optional[str]): Stores resolution for the component runtime state.
	    size (Optional[str]): Stores size for the component runtime state."""
	client: Optional[ genai.Client ]
	aspect_ratio: Optional[ str ]
	use_vertex: Optional[ bool ]
	resolution: Optional[ str ]
	size: Optional[ str ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-image' ):
		super( ).__init__( )
		self.number = None
		self.model = model
		self.client = None
		self.instructions = None
		self.image_config = None
		self.function_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_config = None
		self.response_modalities = [ ]
		self.tools = [ ]
		self.stops = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.temperature = None
		self.size = None
		self.top_p = None
		self.top_k = None
		self.aspect_ratio = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.candidate_count = None
		self.max_output_tokens = None
		self.use_vertex = None
		self.media_resolution = None
		self.tool_choice = None
		self.content_response = None
		self.response = None
		self.grounding_metadata = None
		self.output_mime_type = None
		self.response_mode = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'gemini-2.5-flash-image',
		         'gemini-3.1-flash-image-preview' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		"""Aspect options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9' ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Media options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'text', 'image', 'text_and_image' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'unspecified', 'minimal',
		         'low', 'medium', 'high' ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Size options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ '1K', '2K', '4K' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'google_search', 'image_search' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'auto', 'any', 'none', 'validated' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'image/jpeg',
		         'image/png',
		         'image/webp' ]
	
	@property
	def resolution_options( self ) -> List[ str ] | None:
		"""Resolution options.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ '1K', '2K', '4K' ]
	
	def supports_image_size( self, model: str = 'gemini-2.5-flash-image' ) -> bool:
		"""Supports image size.
		
		Purpose:
		    Performs the Images.supports_image_size workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			self.image_size_models = [ 'gemini-3.1-flash-image-preview',
			                           'gemini-3-pro-image-preview' ]
			return self.model_name in self.image_size_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_size( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_search_grounding( self, model: str = 'gemini-2.5-flash-image' ) -> bool:
		"""Supports search grounding.
		
		Purpose:
		    Performs the Images.supports_search_grounding workflow using the inputs supplied by the caller
		    and the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			self.search_grounding_models = [ 'gemini-3.1-flash-image-preview',
			                                 'gemini-3-pro-image-preview' ]
			return self.model_name in self.search_grounding_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_search_grounding( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_image_search( self, model: str = 'gemini-2.5-flash-image' ) -> bool:
		"""Supports image search.
		
		Purpose:
		    Performs the Images.supports_image_search workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			return self.model_name == 'gemini-3.1-flash-image-preview'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_search( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response_modalities( self, response_modalities: Optional[ str ],
			image_only: bool = False ) -> List[ str ]:
		"""Normalize response modalities.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    response_modalities (Optional[str]): Response modalities value used by the operation.
		    image_only (bool): Image only value used by the operation.
		
		Returns:
		    List[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.mode_name = str( response_modalities or '' ).strip( ).upper( )
			if self.mode_name == 'TEXT_AND_IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			if self.mode_name == 'TEXT':
				return [ 'TEXT' ]
			
			if self.mode_name == 'IMAGE':
				return [ 'IMAGE' ]
			
			if self.mode_name == 'TEXT,IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			if self.mode_name == 'TEXT, IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			return [ 'IMAGE' ] if image_only else [ 'TEXT' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = (
					'normalize_response_modalities( self, response_modalities: Optional[str], '
					'image_only: bool=False ) -> List[str]')
			Logger( ).write( exception )
			raise exception
	
	def build_grounding_tool( self, image_search: bool = False ) -> Optional[ Tool ]:
		"""Build grounding tool.
		
		Purpose:
		    Builds the normalized data structure required by the Images workflow. The function converts
		    caller input, session state, or provider-specific options into a stable shape that downstream
		    API calls and rendering code can consume safely.
		
		Args:
		    image_search (bool): Image search value used by the operation.
		
		Returns:
		    Optional[Tool]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if not self.supports_search_grounding( self.model ):
				return None
			
			self.use_image_search = bool( image_search )
			self.model_name = str( self.model or '' ).strip( ).lower( )
			if self.use_image_search and self.supports_image_search( self.model_name ):
				return Tool( google_search=types.GoogleSearch( search_types=types.SearchTypes(
					web_search=types.WebSearch( ), image_search=types.ImageSearch( ) ) ) )
			
			return Tool( google_search=types.GoogleSearch( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'build_grounding_tool( self, image_search: bool=False ) -> Optional[Tool]'
			Logger( ).write( exception )
			raise exception
	
	def get_content_config( self, response_modalities: Optional[ str ], image_only: bool = False,
			image_search: bool = False, grounded: bool = False,
			output_mime_type: Optional[ str ] = None ) -> GenerateContentConfig:
		"""Get content config.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Args:
		    response_modalities (Optional[str]): Response modalities value used by the operation.
		    image_only (bool): Image only value used by the operation.
		    image_search (bool): Image search value used by the operation.
		    grounded (bool): Grounded value used by the operation.
		    output_mime_type (Optional[str]): Output mime type value used by the operation.
		
		Returns:
		    GenerateContentConfig: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.image_only = image_only
			self.image_config = None
			self.tool_config = None
			self.grounding_metadata = None
			self.output_mime_type = str( output_mime_type or '' ).strip( ) or None
			self.image_kwargs = { }
			self.aspect_value = str( self.aspect_ratio or '' ).strip( )
			if self.aspect_value:
				self.image_kwargs[ 'aspect_ratio' ] = self.aspect_value
			
			self.size_value = str( self.size or '' ).strip( )
			if self.size_value and self.supports_image_size( self.model ):
				self.image_kwargs[ 'image_size' ] = self.size_value
			
			if len( self.image_kwargs ) > 0:
				self.image_config = types.ImageConfig( **self.image_kwargs )
			
			if grounded:
				self.grounding_tool = self.build_grounding_tool( image_search=image_search )
				if self.grounding_tool is not None:
					self.tool_config = [ self.grounding_tool ]
			
			self.response_modalities = self.normalize_response_modalities(
				response_modalities=response_modalities, image_only=image_only )
			
			self.config_kwargs = { 'response_modalities': self.response_modalities }
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.number is not None and int( self.number or 0 ) > 0:
				self.config_kwargs[ 'candidate_count' ] = int( self.number )
			
			if self.max_output_tokens is not None and int( self.max_output_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_output_tokens )
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.image_config is not None:
				self.config_kwargs[ 'image_config' ] = self.image_config
			
			if self.tool_config is not None and len( self.tool_config ) > 0:
				self.config_kwargs[ 'tools' ] = self.tool_config
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_content_config( self, **kwargs ) -> GenerateContentConfig'
			Logger( ).write( exception )
			raise exception
	
	def open_image( self, path: str ) -> PIL.Image.Image:
		"""Open image.
		
		Purpose:
		    Performs the Images.open_image workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		
		Returns:
		    PIL.Image.Image: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'path', path )
			with PIL.Image.open( path ) as source:
				return source.copy( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'open_image( self, path ) -> PIL.Image.Image'
			Logger( ).write( exception )
			raise exception
	
	def capture_metadata( self ) -> None:
		"""Capture metadata.
		
		Purpose:
		    Performs the Images.capture_metadata workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    None: This function performs its work through side effects and does not return a value.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.grounding_metadata = None
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if self.candidates:
				for candidate in self.candidates:
					self.metadata = getattr( candidate, 'grounding_metadata', None )
					if self.metadata is None:
						self.metadata = getattr( candidate, 'groundingMetadata', None )
					
					if self.metadata is not None:
						self.grounding_metadata = self.metadata
						return
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'capture_metadata( self )'
			Logger( ).write( exception )
			raise exception
	
	def get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""Get first image.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Optional[PIL.Image.Image]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if self.content_response is None:
				return None
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				for part in parts:
					try:
						if getattr( part, 'inline_data', None ) is not None:
							return part.as_image( )
					except Exception:
						continue
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					candidate_parts = getattr( content, 'parts', None ) or [ ]
					for part in candidate_parts:
						try:
							if getattr( part, 'inline_data', None ) is not None:
								return part.as_image( )
						except Exception:
							continue
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_first_image( self ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
		"""Get output text.
		
		Purpose:
		    Returns normalized information for the Images component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if self.content_response is None:
				return None
			
			text = getattr( self.content_response, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				return text
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				output = [ ]
				for part in parts:
					part_text = getattr( part, 'text', None )
					if isinstance( part_text, str ) and part_text.strip( ):
						output.append( part_text.strip( ) )
				
				if output:
					return '\n'.join( output )
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					output = [ ]
					for part in getattr( content, 'parts', None ) or [ ]:
						part_text = getattr( part, 'text', None )
						if isinstance( part_text, str ) and part_text.strip( ):
							output.append( part_text.strip( ) )
					
					if output:
						return '\n'.join( output )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_output_text( self ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False, size: str = None,
			quality: str = None, style: str = None, fmt: str = None,
			mime_type: str = None, compression: float = None, background: str = None,
			aspect_ratio: str = None, **kwargs: Any ) -> Optional[ PIL.Image.Image ]:
		"""Generate.
		
		Purpose:
		    Generates provider output for the Images workflow using validated model settings and request
		    inputs. The method coordinates request construction, provider execution, response capture, and
		    logged exception handling.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    aspect (str): Aspect value used by the operation.
		    number (int): Number value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    resolution (str): Resolution value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    output_mime_type (str): Output mime type value used by the operation.
		    response_modalities (str): Response modalities value used by the operation.
		    grounded (bool): Grounded value used by the operation.
		    image_search (bool): Image search value used by the operation.
		    size (str): Size value used by the operation.
		    quality (str): Quality value used by the operation.
		    style (str): Style value used by the operation.
		    fmt (str): Fmt value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    compression (float): Compression value used by the operation.
		    background (str): Background value used by the operation.
		    aspect_ratio (str): Aspect ratio value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[PIL.Image.Image]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			throw_if( 'model', model )
			self.model = model
			self.gemini_api_key = cfg.GEMINI_API_KEY
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.number = number
			self.aspect_ratio = aspect or aspect_ratio
			self.media_resolution = resolution or size
			self.size = self.media_resolution
			self.quality = quality
			self.style = style
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type or mime_type or fmt
			self.compression = compression
			self.background = background
			self.response_mode = response_modalities or 'image'
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self.get_content_config( image_only=True,
				grounded=grounded, image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt, model ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False, image_path: str = None,
			detail: str = None, **kwargs: Any ) -> Optional[ str ]:
		"""Analyze.
		
		Purpose:
		    Performs the Images.analyze workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    path (str): Path value used by the operation.
		    model (str): Model value used by the operation.
		    aspect (str): Aspect value used by the operation.
		    number (int): Number value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    resolution (str): Resolution value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    output_mime_type (str): Output mime type value used by the operation.
		    response_modalities (str): Response modalities value used by the operation.
		    grounded (bool): Grounded value used by the operation.
		    image_search (bool): Image search value used by the operation.
		    image_path (str): Image path value used by the operation.
		    detail (str): Detail value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			throw_if( 'path', path )
			self.file_path = path
			throw_if( 'model', model )
			self.model = model
			self.gemini_api_key = cfg.GEMINI_API_KEY
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.number = number
			self.aspect_ratio = aspect
			self.media_resolution = resolution
			self.detail = detail
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities or 'text'
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self.get_content_config( image_only=False, grounded=grounded,
				image_search=image_search, response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.image_input = self.open_image( self.file_path )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.image_input ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'analyze( self, prompt, path, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False, image_path: str = None,
			mask_path: str = None, mask: str = None, size: str = None,
			quality: str = None, style: str = None, fmt: str = None,
			mime_type: str = None, compression: float = None, background: str = None,
			aspect_ratio: str = None, **kwargs: Any ) -> Optional[ PIL.Image.Image ]:
		"""Edit.
		
		Purpose:
		    Performs the Images.edit workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    path (str): Path value used by the operation.
		    model (str): Model value used by the operation.
		    aspect (str): Aspect value used by the operation.
		    number (int): Number value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    resolution (str): Resolution value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    output_mime_type (str): Output mime type value used by the operation.
		    response_modalities (str): Response modalities value used by the operation.
		    grounded (bool): Grounded value used by the operation.
		    image_search (bool): Image search value used by the operation.
		    image_path (str): Image path value used by the operation.
		    mask_path (str): Mask path value used by the operation.
		    mask (str): Mask value used by the operation.
		    size (str): Size value used by the operation.
		    quality (str): Quality value used by the operation.
		    style (str): Style value used by the operation.
		    fmt (str): Fmt value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    compression (float): Compression value used by the operation.
		    background (str): Background value used by the operation.
		    aspect_ratio (str): Aspect ratio value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[PIL.Image.Image]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			
			throw_if( 'path', path )
			self.file_path = path
			
			throw_if( 'model', model )
			self.model = model
			
			self.gemini_api_key = cfg.GEMINI_API_KEY
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.number = number
			self.aspect_ratio = aspect or aspect_ratio
			self.media_resolution = resolution or size
			self.size = self.media_resolution
			self.quality = quality
			self.style = style
			self.mask_path = mask_path or mask
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type or mime_type or fmt
			self.compression = compression
			self.background = background
			self.response_mode = response_modalities or 'image'
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self.get_content_config(
				image_only=True,
				grounded=grounded,
				image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type
			)
			self.image_input = self.open_image( self.file_path )
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self.image_input ],
				config=self.content_config
			)
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'edit( self, prompt, path, model ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception

class Embeddings( Gemini ):
	"""Embeddings class.
	
	Purpose:
	    Defines the Embeddings component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    client (Optional[genai.Client]): Stores client for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    embedding (Optional[List[float] | List[List[float]]]): Stores embedding for the component runtime state.
	    encoding_format (Optional[str]): Stores encoding format for the component runtime state.
	    dimensions (Optional[int]): Stores dimensions for the component runtime state.
	    task_type (Optional[str]): Stores task type for the component runtime state.
	    title (Optional[str]): Stores title for the component runtime state.
	    embedding_config (Optional[types.EmbedContentConfig]): Stores embedding config for the component runtime state.
	    contents (Optional[str | List[str]]): Stores contents for the component runtime state.
	    input_text (Optional[str | List[str]]): Stores input text for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    response_modalities (Optional[str]): Stores response modalities for the component runtime state."""
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] | List[ List[ float ] ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	task_type: Optional[ str ]
	title: Optional[ str ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ str | List[ str ] ]
	input_text: Optional[ str | List[ str ] ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, model: str = 'gemini-embedding-001' ):
		super( ).__init__( )
		self.model = model
		self.client = None
		self.embedding = None
		self.embeddings = None
		self.response = None
		self.encoding_format = None
		self.input_text = None
		self.contents = None
		self.file_path = None
		self.dimensions = None
		self.task_type = None
		self.title = None
		self.response_modalities = None
		self.embedding_config = None
		self.content_config = None
		self.api_key = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'gemini-embedding-001',
		         'gemini-embedding-2',
		         'gemini-embedding-2-preview',
		         'text-embedding-004',
		         'text-multilingual-embedding-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Encoding options.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ 'float', 'base64' ]
	
	@property
	def task_options( self ) -> List[ str ]:
		"""Task options.
		
		Purpose:
		    Returns normalized information for the Embeddings component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ '',
		         'RETRIEVAL_QUERY',
		         'RETRIEVAL_DOCUMENT',
		         'SEMANTIC_SIMILARITY',
		         'CLASSIFICATION',
		         'CLUSTERING',
		         'QUESTION_ANSWERING',
		         'FACT_VERIFICATION',
		         'CODE_RETRIEVAL_QUERY' ]
	
	def normalize_dimensions( self, dimensions: int ) -> int | None:
		"""Normalize dimensions.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    dimensions (int): Dimensions value used by the operation.
		
		Returns:
		    Optional[int]: Return value produced by the operation."""
		try:
			throw_if( 'dimensions', dimensions )
			self.dimensions = dimensions
			if self.dimensions <= 0:
				return None
			
			return self.dimensions
		except Exception:
			return None
	
	def normalize_contents( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Normalize contents.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		
		Returns:
		    str | List[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'text', text )
			if isinstance( text, list ):
				self.contents = [ ]
				for item in text:
					if item is None:
						continue
					
					self.item = str( item ).strip( )
					if self.item:
						self.contents.append( self.item )
				
				throw_if( 'text', self.contents )
				return self.contents
			
			self.contents = str( text ).strip( )
			throw_if( 'text', self.contents )
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'normalize_contents( self, text: str | List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def extract_embeddings( self ) -> List[ float ] | List[ List[ float ] ] | None:
		"""Extract embeddings.
		
		Purpose:
		    Extracts structured information from a provider response, uploaded file, or application data
		    object. The function normalizes provider-specific shapes into values that can be rendered,
		    stored, or passed to later processing steps.
		
		Returns:
		    List[float] | List[List[float]] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			if self.response is None:
				return None
			
			if not hasattr( self.response, 'embeddings' ):
				return None
			
			self.embeddings = [ ]
			for item in self.response.embeddings:
				if item is None:
					continue
				
				if hasattr( item, 'values' ) and item.values is not None:
					self.embeddings.append( list( item.values ) )
			
			if len( self.embeddings ) == 0:
				return None
			
			if len( self.embeddings ) == 1 and isinstance( self.input_text, str ):
				self.embedding = self.embeddings[ 0 ]
				return self.embedding
			
			self.embedding = self.embeddings
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'extract_embeddings( self )'
			Logger( ).write( exception )
			raise exception
	
	def build_embedding_config( self, model: str = 'gemini-embedding-001', dimensions: int = None,
			task_type: str = None, title: str = None ) -> EmbedContentConfig:
		"""Build embedding config.
		
		Purpose:
		    Builds the normalized data structure required by the Embeddings workflow. The function converts
		    caller input, session state, or provider-specific options into a stable shape that downstream
		    API calls and rendering code can consume safely.
		
		Args:
		    model (str): Model value used by the operation.
		    dimensions (int): Dimensions value used by the operation.
		    task_type (str): Task type value used by the operation.
		    title (str): Title value used by the operation.
		
		Returns:
		    EmbedContentConfig: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'model', model )
			self.model = model
			self.dimensions = dimensions
			self.task_type = task_type
			self.title = title
			self.normalized_dimensions = self.normalize_dimensions( self.dimensions )
			self.config_kwargs = { }
			
			if self.normalized_dimensions is not None:
				self.config_kwargs[ 'output_dimensionality' ] = self.normalized_dimensions
			
			if self.task_type is not None and str( self.task_type ).strip( ):
				self.task_type = str( self.task_type ).strip( ).upper( )
				self.config_kwargs[ 'task_type' ] = self.task_type
			
			if self.title is not None and str( self.title ).strip( ) \
					and self.task_type == 'RETRIEVAL_DOCUMENT':
				self.title = str( self.title ).strip( )
				self.config_kwargs[ 'title' ] = self.title
			
			self.embedding_config = EmbedContentConfig( **self.config_kwargs )
			return self.embedding_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'build_embedding_config( self, model, dimensions, task_type, title )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str = 'gemini-embedding-001',
			dimensions: int = None, task_type: str = None, title: str = None,
			encoding_format: str = 'float' ) -> List[ float ] | List[ List[ float ] ] | None:
		"""Create.
		
		Purpose:
		    Performs the Embeddings.create workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    text (str | List[str]): Text value used by the operation.
		    model (str): Model value used by the operation.
		    dimensions (int): Dimensions value used by the operation.
		    task_type (str): Task type value used by the operation.
		    title (str): Title value used by the operation.
		    encoding_format (str): Encoding format value used by the operation.
		
		Returns:
		    List[float] | List[List[float]] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'text', text )
			self.input_text = text
			
			throw_if( 'model', model )
			self.model = model
			
			self.dimensions = dimensions
			self.task_type = task_type
			self.title = title
			self.encoding_format = encoding_format
			self.gemini_api_key = cfg.GEMINI_API_KEY
			
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			if self.model not in self.model_options:
				raise ValueError( f'Unsupported Gemini embedding model: {self.model}' )
			
			if self.encoding_format not in self.encoding_options:
				self.encoding_format = 'float'
			
			self.contents = self.normalize_contents( self.input_text )
			if self.contents is None:
				raise ValueError( 'The Gemini embedding contents cannot be None.' )
			
			if isinstance( self.contents, str ) and not self.contents.strip( ):
				raise ValueError( 'The Gemini embedding contents cannot be empty.' )
			
			if isinstance( self.contents, list ) and len( self.contents ) == 0:
				raise ValueError( 'The Gemini embedding contents list cannot be empty.' )
			
			self.embedding_config = self.build_embedding_config(
				model=self.model,
				dimensions=self.dimensions,
				task_type=self.task_type,
				title=self.title
			)
			
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.models.embed_content(
				model=self.model,
				contents=self.contents,
				config=self.embedding_config
			)
			
			return self.extract_embeddings( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'create( self, text, model, dimensions, task_type, title, encoding_format )'
			Logger( ).write( exception )
			raise exception

class TTS( Gemini ):
	"""TTS class.
	
	Purpose:
	    Defines the TTS component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    speed (Optional[float]): Stores speed for the component runtime state.
	    voice (Optional[str]): Stores voice for the component runtime state.
	    response (Optional[GenerateContentResponse]): Stores response for the component runtime state.
	    voice_config (Optional[VoiceConfig]): Stores voice config for the component runtime state.
	    speech_config (Optional[SpeechConfig]): Stores speech config for the component runtime state.
	    client (Optional[genai.Client]): Stores client for the component runtime state.
	    audio_path (Optional[str]): Stores audio path for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    input_text (Optional[str]): Stores input text for the component runtime state.
	    audio_bytes (Optional[bytes]): Stores audio bytes for the component runtime state."""
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	voice_config: Optional[ VoiceConfig ]
	speech_config: Optional[ SpeechConfig ]
	client: Optional[ genai.Client ]
	audio_path: Optional[ str ]
	response_format: Optional[ str ]
	input_text: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-preview-tts' ):
		super( ).__init__( )
		self.number = None
		self.model = model
		self.temperature = None
		self.top_p = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.voice_config = None
		self.speech_config = None
		self.content_config = None
		self.client = None
		self.voice = None
		self.speed = None
		self.response = None
		self.response_format = None
		self.audio_path = None
		self.input_text = None
		self.audio_bytes = None
		self.response_modalities = [ ]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts',
		         'gemini-2.5-pro-preview-tts' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the TTS component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'audio/wav' ]
	
	def to_wave_bytes( self, pcm_data: bytes, rate: int = 24000, channels: int = 1,
			sample_width: int = 2 ) -> bytes:
		"""To wave bytes.
		
		Purpose:
		    Performs the TTS.to_wave_bytes workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    pcm_data (bytes): Pcm data value used by the operation.
		    rate (int): Rate value used by the operation.
		    channels (int): Channels value used by the operation.
		    sample_width (int): Sample width value used by the operation.
		
		Returns:
		    bytes: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			import io
			import wave
			
			throw_if( 'pcm_data', pcm_data )
			with io.BytesIO( ) as buffer:
				with wave.open( buffer, 'wb' ) as wf:
					wf.setnchannels( channels )
					wf.setsampwidth( sample_width )
					wf.setframerate( rate )
					wf.writeframes( pcm_data )
				
				return buffer.getvalue( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'to_wave_bytes( self, **kwargs) -> bytes'
			Logger( ).write( exception )
			raise exception
	
	def normalize_voice( self, voice: Optional[ str ] = None ) -> str:
		"""Normalize voice.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    voice (Optional[str]): Voice value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.voice_name = str( voice or '' ).strip( )
			self.valid_voices = set( self.voice_options or [ ] )
			if self.voice_name in self.valid_voices:
				return self.voice_name
			
			return 'Kore'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'normalize_voice( self, voice: Optional[str]=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_tts_prompt( self, text: str, speed: Optional[ float ] = None,
			instruct: Optional[ str ] = None ) -> str:
		"""Normalize tts prompt.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    text (str): Text value used by the operation.
		    speed (Optional[float]): Speed value used by the operation.
		    instruct (Optional[str]): Instruct value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'text', text )
			self.prompt_parts = [ ]
			
			if instruct is not None and str( instruct ).strip( ):
				self.prompt_parts.append( str( instruct ).strip( ) )
			
			if speed is not None:
				self.speed_value = float( speed )
				if self.speed_value < 0.85:
					self.prompt_parts.append( 'Read the following text at a slow, clear pace.' )
				elif self.speed_value > 1.15:
					self.prompt_parts.append(
						'Read the following text at a faster, energetic pace.' )
			
			self.prompt_parts.append( str( text ).strip( ) )
			return '\n\n'.join( self.prompt_parts )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'normalize_tts_prompt( self, **kwargs ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, filepath: str = None,
			model: str = 'gemini-3.1-flash-tts-preview', format: str = 'audio/wav',
			speed: float = None, voice: str = None, frequency: float = None,
			presense: float = None, presence: float = None, max_tokens: int = None,
			instruct: str = None, temperature: float = None, top_p: float = None,
			file_path: str = None, language: str = None, sample_rate: int = None,
			bit_rate: int = None, store: bool = None, stream: bool = None,
			background: bool = None, **kwargs: Any ) -> bytes | None:
		"""Create speech.
		
		Purpose:
		    Creates the requested resource, connection, schema object, or user interface artifact using
		    validated inputs. The function encapsulates setup details so callers can rely on a consistent
		    resource lifecycle.
		
		Args:
		    text (str): Text value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    model (str): Model value used by the operation.
		    format (str): Format value used by the operation.
		    speed (float): Speed value used by the operation.
		    voice (str): Voice value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presense (float): Presense value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    file_path (str): File path value used by the operation.
		    language (str): Language value used by the operation.
		    sample_rate (int): Sample rate value used by the operation.
		    bit_rate (int): Bit rate value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    background (bool): Background value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    bytes | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'text', text )
			self.raw_text = text
			throw_if( 'model', model )
			self.model = model
			self.gemini_api_key = cfg.GEMINI_API_KEY
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.audio_path = filepath or file_path
			self.response_format = str( format or 'audio/wav' ).strip( )
			self.speed = speed
			self.voice = self.normalize_voice( voice )
			self.frequency_penalty = frequency
			self.presence_penalty = presence if presence is not None else presense
			self.max_tokens = max_tokens
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			self.language = language
			self.sample_rate = sample_rate
			self.bit_rate = bit_rate
			self.store = store
			self.stream = stream
			self.background = background
			self.response_modalities = [ 'AUDIO' ]
			if self.response_format != 'audio/wav':
				raise ValueError( 'Gemini TTS wrapper currently supports local WAV output only.' )
			
			if self.model not in self.model_options:
				raise ValueError( f'Unsupported Gemini TTS model: {self.model}' )
			
			self.input_text = self.normalize_tts_prompt( text=self.raw_text, speed=self.speed,
				instruct=self.instructions )
			
			if self.input_text is None or not str( self.input_text ).strip( ):
				raise ValueError( 'The Gemini TTS prompt cannot be empty.' )
			
			self.voice_config = VoiceConfig( prebuilt_voice_config=types.PrebuiltVoiceConfig(
				voice_name=self.voice ) )
			self.speech_config = SpeechConfig( voice_config=self.voice_config )
			self.config_kwargs = {
					'response_modalities': self.response_modalities,
					'speech_config': self.speech_config,
			}
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and int( self.max_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_tokens )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.models.generate_content( model=self.model,
				contents=self.input_text, config=self.content_config )
			
			self.audio_bytes = None
			self.candidates = getattr( self.response, 'candidates', None )
			if self.candidates:
				for candidate in self.candidates:
					content = getattr( candidate, 'content', None )
					parts = getattr( content, 'parts', None ) if content is not None else [ ]
					
					for part in parts or [ ]:
						inline_data = getattr( part, 'inline_data', None )
						if inline_data is not None and inline_data.data:
							self.audio_bytes = self.to_wave_bytes( inline_data.data )
							break
					
					if self.audio_bytes is not None:
						break
			
			if self.audio_bytes is None or len( self.audio_bytes ) == 0:
				raise ValueError( 'No audio bytes were returned by Gemini TTS.' )
			
			if self.audio_path is not None and str( self.audio_path ).strip( ):
				with open( self.audio_path, 'wb' ) as f:
					f.write( self.audio_bytes )
			
			return self.audio_bytes
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, text, filepath, model, format, speed, voice )'
			Logger( ).write( exception )
			raise exception

class Transcription( Gemini ):
	"""Transcription class.
	
	Purpose:
	    Defines the Transcription component used by the Boo application. The class groups related
	    provider configuration, runtime state, helper methods, and API-facing behavior so Streamlit
	    workflows can call a consistent interface.
	
	Attributes:
	    client (Optional[genai.Client]): Stores client for the component runtime state.
	    transcript (Optional[str]): Stores transcript for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    response (Optional[GenerateContentResponse]): Stores response for the component runtime state."""
	client: Optional[ genai.Client ]
	transcript: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000, instruct: str = None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
		self.transcript = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'Auto',
		         'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Transcription component. The method provides a stable
		    view of provider capabilities, stored state, or response metadata so UI controls and downstream
		    logic can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'audio/wav',
				'audio/mp3',
				'audio/aiff',
				'audio/aac',
				'audio/ogg',
				'audio/flac'
		]
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
		"""Normalize mime type.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    path (str): Path value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			import mimetypes
			
			self.raw_mime_type = str( mime_type or '' ).strip( )
			if not self.raw_mime_type:
				self.raw_mime_type = mimetypes.guess_type( path )[ 0 ] or ''
			
			self.mime_aliases = {
					'audio/mpeg': 'audio/mp3',
					'audio/x-mp3': 'audio/mp3',
					'audio/x-wav': 'audio/wav',
					'audio/wave': 'audio/wav',
					'audio/x-m4a': 'audio/aac',
					'audio/m4a': 'audio/aac',
					'audio/mp4': 'audio/aac',
					'audio/x-aiff': 'audio/aiff',
					'audio/aif': 'audio/aiff',
					'audio/x-flac': 'audio/flac'
			}
			self.mime_type = self.mime_aliases.get( self.raw_mime_type, self.raw_mime_type )
			
			if self.mime_type in self.format_options:
				return self.mime_type
			
			self.suffix = str( Path( path ).suffix or '' ).strip( ).lower( )
			self.extension_map = {
					'.wav': 'audio/wav',
					'.mp3': 'audio/mp3',
					'.aiff': 'audio/aiff',
					'.aif': 'audio/aiff',
					'.aac': 'audio/aac',
					'.m4a': 'audio/aac',
					'.ogg': 'audio/ogg',
					'.flac': 'audio/flac'
			}
			
			if self.suffix in self.extension_map:
				return self.extension_map[ self.suffix ]
			
			return 'audio/wav'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = 'normalize_mime_type( self, path: str, mime_type: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def build_prompt( self, language: str = None, start_time: float = None,
			end_time: float = None ) -> str:
		"""Build prompt.
		
		Purpose:
		    Builds the normalized data structure required by the Transcription workflow. The function
		    converts caller input, session state, or provider-specific options into a stable shape that
		    downstream API calls and rendering code can consume safely.
		
		Args:
		    language (str): Language value used by the operation.
		    start_time (float): Start time value used by the operation.
		    end_time (float): End time value used by the operation.
		
		Returns:
		    str: Return value produced by the operation."""
		self.prompt_parts = [ 'Generate a verbatim transcript of the speech.' ]
		
		if language is not None and str( language ).strip( ) and str( language ).strip( ) != 'Auto':
			self.prompt_parts.append(
				f'The expected spoken language is {str( language ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only transcribe the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the transcript text.' )
		return ' '.join( self.prompt_parts )
	
	def transcribe( self, path: str = None, model: str = 'gemini-3-flash-preview',
			language: str = None, mime_type: str = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, start_time: float = None, end_time: float = None,
			instruct: str = None, prompt: str = None, response_format: str = None,
			include: List[ str ] = None, top_k: int = None, store: bool = None,
			stream: bool = None, background: bool = None,
			allowed_domains: List[ str ] = None, **kwargs: Any ) -> Optional[ str ]:
		"""Transcribe.
		
		Purpose:
		    Performs the Transcription.transcribe workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		    model (str): Model value used by the operation.
		    language (str): Language value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    start_time (float): Start time value used by the operation.
		    end_time (float): End time value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    response_format (str): Response format value used by the operation.
		    include (List[str]): Include value used by the operation.
		    top_k (int): Top k value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    background (bool): Background value used by the operation.
		    allowed_domains (List[str]): Allowed domains value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.api_key = cfg.GEMINI_API_KEY
			throw_if( 'api_key', self.api_key )
			
			self.file_path = path or kwargs.get( 'filepath' ) or kwargs.get( 'file_path' )
			throw_if( 'path', self.file_path )
			
			self.model = str( model or self.model or 'gemini-3-flash-preview' ).strip( )
			throw_if( 'model', self.model )
			
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.top_k = top_k
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.response_format = response_format
			self.include = include if include is not None else [ ]
			self.store = store
			self.stream = stream
			self.background = background
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.mime_type = self.normalize_mime_type(
				path=self.file_path,
				mime_type=mime_type
			)
			
			self.prompt = self.build_prompt(
				language=language,
				start_time=start_time,
				end_time=end_time
			)
			
			if prompt is not None and str( prompt ).strip( ):
				self.prompt = f'{str( prompt ).strip( )}\n\n{self.prompt}'
			
			throw_if( 'prompt', self.prompt )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.top_k is not None and int( self.top_k or 0 ) > 0:
				self.config_kwargs[ 'top_k' ] = int( self.top_k )
			
			if self.max_tokens is not None and int( self.max_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_tokens )
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=self.api_key )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			throw_if( 'uploaded_file', self.uploaded_file )
			
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self.uploaded_file ],
				config=self.content_config
			)
			
			self.transcript = getattr( self.response, 'text', None )
			if isinstance( self.transcript, str ) and self.transcript.strip( ):
				return self.transcript.strip( )
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path: str=None, **kwargs ) -> Optional[ str ]'
			Logger( ).write( ex )
			raise ex

class Translation( Gemini ):
	"""Translation class.
	
	Purpose:
	    Defines the Translation component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    client (Optional[genai.Client]): Stores client for the component runtime state.
	    target_language (Optional[str]): Stores target language for the component runtime state.
	    source_language (Optional[str]): Stores source language for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    response (Optional[GenerateContentResponse]): Stores response for the component runtime state."""
	client: Optional[ genai.Client ]
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000,
			instruct: str = None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
		self.target_language = None
		self.source_language = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'audio/wav',
				'audio/mp3',
				'audio/aiff',
				'audio/aac',
				'audio/ogg',
				'audio/flac'
		]
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
		"""Normalize mime type.
		
		Purpose:
		    Normalizes incoming values into a predictable representation for application processing. The
		    function reduces provider, user-input, or serialization differences before values are stored or
		    displayed.
		
		Args:
		    path (str): Path value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		
		Returns:
		    str: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			import mimetypes
			
			self.raw_mime_type = str( mime_type or '' ).strip( )
			if not self.raw_mime_type:
				self.raw_mime_type = mimetypes.guess_type( path )[ 0 ] or ''
			
			self.mime_aliases = {
					'audio/mpeg': 'audio/mp3',
					'audio/x-mp3': 'audio/mp3',
					'audio/x-wav': 'audio/wav',
					'audio/wave': 'audio/wav',
					'audio/x-m4a': 'audio/aac',
					'audio/m4a': 'audio/aac',
					'audio/mp4': 'audio/aac',
					'audio/x-aiff': 'audio/aiff',
					'audio/aif': 'audio/aiff',
					'audio/x-flac': 'audio/flac'
			}
			self.mime_type = self.mime_aliases.get( self.raw_mime_type, self.raw_mime_type )
			
			if self.mime_type in self.format_options:
				return self.mime_type
			
			self.suffix = str( Path( path ).suffix or '' ).strip( ).lower( )
			self.extension_map = {
					'.wav': 'audio/wav',
					'.mp3': 'audio/mp3',
					'.aiff': 'audio/aiff',
					'.aif': 'audio/aiff',
					'.aac': 'audio/aac',
					'.m4a': 'audio/aac',
					'.ogg': 'audio/ogg',
					'.flac': 'audio/flac'
			}
			
			if self.suffix in self.extension_map:
				return self.extension_map[ self.suffix ]
			
			return 'audio/wav'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = 'normalize_mime_type( self, path: str, mime_type: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
		    Returns normalized information for the Translation component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	def build_prompt( self, target: str, source: str = 'Auto', start_time: float = None,
			end_time: float = None ) -> str:
		"""Build prompt.
		
		Purpose:
		    Builds the normalized data structure required by the Translation workflow. The function converts
		    caller input, session state, or provider-specific options into a stable shape that downstream
		    API calls and rendering code can consume safely.
		
		Args:
		    target (str): Target value used by the operation.
		    source (str): Source value used by the operation.
		    start_time (float): Start time value used by the operation.
		    end_time (float): End time value used by the operation.
		
		Returns:
		    str: Return value produced by the operation."""
		self.prompt_parts = [ f'Translate the spoken audio into {target}.' ]
		if source is not None and str( source ).strip( ) and str( source ).strip( ) != 'Auto':
			self.prompt_parts.append(
				f'The expected source language is {str( source ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only translate the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the translated text.' )
		return ' '.join( self.prompt_parts )
	
	def translate( self, path: str, model: str = 'gemini-3-flash-preview',
			language: str = 'English', source: str = 'Auto', mime_type: str = None,
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, start_time: float = None,
			end_time: float = None, instruct: str = None, prompt: str = None,
			response_format: str = None, include: List[ str ] = None, top_k: int = None,
			store: bool = None, stream: bool = None, background: bool = None,
			allowed_domains: List[ str ] = None, **kwargs: Any ) -> Optional[ str ]:
		"""Translate.
		
		Purpose:
		    Performs the Translation.translate workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		    model (str): Model value used by the operation.
		    language (str): Language value used by the operation.
		    source (str): Source value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    start_time (float): Start time value used by the operation.
		    end_time (float): End time value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    prompt (str): Prompt value used by the operation.
		    response_format (str): Response format value used by the operation.
		    include (List[str]): Include value used by the operation.
		    top_k (int): Top k value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    background (bool): Background value used by the operation.
		    allowed_domains (List[str]): Allowed domains value used by the operation.
		    **kwargs (Any): Additional keyword arguments retained for compatibility with caller workflows.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'path', path )
			self.file_path = path
			
			throw_if( 'model', model )
			self.model = model
			
			throw_if( 'language', language )
			self.target_language = language
			
			self.gemini_api_key = cfg.GEMINI_API_KEY
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			if self.model not in self.model_options:
				raise ValueError( f'Unsupported Gemini translation model: {self.model}' )
			
			self.source_language = str( source or 'Auto' ).strip( )
			self.mime_type = self.normalize_mime_type(
				path=self.file_path,
				mime_type=mime_type
			)
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.top_k = top_k
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.response_format = response_format
			self.include = include if include is not None else [ ]
			self.store = store
			self.stream = stream
			self.background = background
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.prompt = self.build_prompt( target=self.target_language,
				source=self.source_language,
				start_time=start_time, end_time=end_time )
			
			if prompt is not None and str( prompt ).strip( ):
				self.prompt = f'{str( prompt ).strip( )}\n\n{self.prompt}'
			
			if self.prompt is None or not str( self.prompt ).strip( ):
				raise ValueError( 'The Gemini translation prompt cannot be empty.' )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.top_k is not None and int( self.top_k or 0 ) > 0:
				self.config_kwargs[ 'top_k' ] = int( self.top_k )
			
			if self.max_tokens is not None and int( self.max_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_tokens )
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			
			if self.uploaded_file is None:
				raise ValueError( 'The audio file could not be uploaded to Gemini.' )
			
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self.uploaded_file ],
				config=self.content_config
			)
			
			self.translation = getattr( self.response, 'text', None )
			if isinstance( self.translation, str ) and self.translation.strip( ):
				return self.translation.strip( )
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Translation'
			ex.method = 'translate( self, path, model, language ) -> Optional[ str ]'
			Logger( ).write( ex )
			raise ex

class Files( Gemini ):
	"""Files class.
	
	Purpose:
	    Defines the Files component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    api_version (Optional[str]): Stores api version for the component runtime state.
	    google_api_key (Optional[str]): Stores google api key for the component runtime state.
	    storage_client (Optional[storage.Client]): Stores storage client for the component runtime state.
	    project_id (Optional[str]): Stores project id for the component runtime state.
	    project_location (Optional[str]): Stores project location for the component runtime state.
	    file_id (Optional[str]): Stores file id for the component runtime state.
	    bucket_id (Optional[str]): Stores bucket id for the component runtime state.
	    display_name (Optional[str]): Stores display name for the component runtime state.
	    mime_type (Optional[str]): Stores mime type for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    file_list (Optional[List[File]]): Stores file list for the component runtime state.
	    file_paths (Optional[List[str]]): Stores file paths for the component runtime state.
	    file_lists (Optional[List[File]]): Stores file lists for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    use_vertex (Optional[bool]): Stores use vertex for the component runtime state.
	    collections (Optional[Dict[str, str]]): Stores collections for the component runtime state.
	    documents (Optional[Dict[str, str]]): Stores documents for the component runtime state."""
	api_version: Optional[ str ]
	google_api_key: Optional[ str ]
	storage_client: Optional[ storage.Client ]
	project_id: Optional[ str ]
	project_location: Optional[ str ]
	file_id: Optional[ str ]
	bucket_id: Optional[ str ]
	display_name: Optional[ str ]
	mime_type: Optional[ str ]
	file_path: Optional[ str ]
	file_list: Optional[ List[ File ] ]
	file_paths: Optional[ List[ str ] ]
	file_lists: Optional[ List[ File ] ]
	response: Optional[ Any ]
	use_vertex: Optional[ bool ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self, model: str = 'gemini-2.0-flash' ):
		super( ).__init__( )
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.project_location = cfg.GOOGLE_CLOUD_LOCATION
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.tool_choice = None
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.storage_client = None
		self.bucket_id = None
		self.file_id = None
		self.display_name = None
		self.media_resolution = None
		self.mime_type = None
		self.file_path = None
		self.file_list = [ ]
		self.response = None
		self.collections = { }
		self.documents = { }
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""File options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return self.files
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'gemini-3.5-flash',
		         'gemini-3.5 flash-lite',
		         'gemini-3.0-flash',
		         'gemini-3.0-flash-lite' ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Media options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'AUTO',
		         'ANY',
		         'NONE',
		         'VALIDATED' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'google_search',
		         'google_maps',
		         'file_search',
		         'url_context',
		         'code_execution',
		         'computer_use' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns normalized information for the Files component. The method provides a stable view of
		    provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [ 'MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'AUDIO' ]
	
	def upload( self, filepath: str, name: str = None, display_name: str = None,
			filename: str = None, mime_type: str = None, purpose: str = None ) -> File | None:
		"""Upload.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    filepath (str): Filepath value used by the operation.
		    name (str): Name value used by the operation.
		    display_name (str): Display name value used by the operation.
		    filename (str): Filename value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    purpose (str): Purpose value used by the operation.
		
		Returns:
		    File | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			
			if not os.path.exists( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			
			self.display_name = name or display_name or filename or Path( self.file_path ).name
			self.mime_type = mime_type
			self.purpose = purpose
			self.gemini_api_key = cfg.GEMINI_API_KEY
			
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.files.upload(
				path=self.file_path,
				config={ 'display_name': self.display_name }
			)
			
			self.file_id = getattr( self.response, 'name', None )
			if self.file_id is None:
				self.file_id = getattr( self.response, 'uri', None )
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'upload( self, filepath, name, display_name, filename, mime_type, purpose )'
			Logger( ).write( ex )
			raise ex
	
	def upload_file( self, path: str, file_path: str = None, filepath: str = None,
			name: str = None, display_name: str = None, filename: str = None,
			mime_type: str = None, purpose: str = None ) -> File | None:
		"""Upload file.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    path (str): Path value used by the operation.
		    file_path (str): File path value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    name (str): Name value used by the operation.
		    display_name (str): Display name value used by the operation.
		    filename (str): Filename value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    purpose (str): Purpose value used by the operation.
		
		Returns:
		    File | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'path', path )
			self.file_path = path
			
			return self.upload(
				filepath=self.file_path,
				name=name,
				display_name=display_name,
				filename=filename,
				mime_type=mime_type,
				purpose=purpose
			)
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'upload_file( self, path, file_path, filepath, name, display_name, filename )'
			Logger( ).write( ex )
			raise ex
	
	def files_upload( self, path: str, file_path: str = None, filepath: str = None,
			name: str = None, display_name: str = None, filename: str = None,
			mime_type: str = None, purpose: str = None ) -> File | None:
		"""Files upload.
		
		Purpose:
		    Performs the Files.files_upload workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		    file_path (str): File path value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    name (str): Name value used by the operation.
		    display_name (str): Display name value used by the operation.
		    filename (str): Filename value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		    purpose (str): Purpose value used by the operation.
		
		Returns:
		    File | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.file_paths = file_path
			self.filepath = filepath
			return self.upload( filepath=self.file_path, name=name, display_name=display_name,
				filename=filename, mime_type=mime_type, purpose=purpose )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'files_upload( self, path, file_path, filepath, name, display_name, filename )'
			Logger( ).write( ex )
			raise ex
	
	def list( self, model: str = 'gemini-2.0-flash', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None ) -> List[ File ]:
		"""List.
		
		Purpose:
		    Performs the Files.list workflow using the inputs supplied by the caller and the current runtime
		    configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		
		Returns:
		    List[File]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.model = model
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.gemini_api_key = cfg.GEMINI_API_KEY
			
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.file_list = list( self.client.files.list( ) )
			return self.file_list
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list( self, model, temperature, top_p, frequency, presence, max_tokens, stops )'
			Logger( ).write( ex )
			raise ex
	
	def list_files( self ) -> List[ File ]:
		"""List files.
		
		Purpose:
		    Performs the Files.list_files workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[File]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.list( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list_files( self ) -> List[ File ]'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, file_id: str ) -> Optional[ File ]:
		"""Retrieve.
		
		Purpose:
		    Performs the Files.retrieve workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		
		Returns:
		    Optional[File]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.gemini_api_key = cfg.GEMINI_API_KEY
			
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.files.get( name=self.file_id )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id ) -> Optional[ File ]'
			Logger( ).write( ex )
			raise ex
	
	def extract( self, file_id: str, format: str = None,
			page_number: int = None ) -> Dict[ str, Any ] | None:
		"""Extract.
		
		Purpose:
		    Performs the Files.extract workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    format (str): Format value used by the operation.
		    page_number (int): Page number value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.format = format
			self.page_number = page_number
			self.response = self.retrieve( self.file_id )
			
			if self.response is None:
				return None
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'name': getattr( self.response, 'name', self.file_id ),
					'display_name': getattr( self.response, 'display_name', None ),
					'mime_type': getattr( self.response, 'mime_type', None ),
					'uri': getattr( self.response, 'uri', None ),
					'state': getattr( self.response, 'state', None ),
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'extract( self, file_id, format, page_number )'
			Logger( ).write( ex )
			raise ex
	
	def download( self, file_id: str, format: str = None,
			page_number: int = None ) -> Dict[ str, Any ] | None:
		"""Download.
		
		Purpose:
		    Performs the Files.download workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    format (str): Format value used by the operation.
		    page_number (int): Page number value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			return self.extract(
				file_id=self.file_id,
				format=format,
				page_number=page_number
			)
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'download( self, file_id, format, page_number )'
			Logger( ).write( ex )
			raise ex
	
	def content( self, file_id: str, format: str = None,
			page_number: int = None ) -> Dict[ str, Any ] | None:
		"""Content.
		
		Purpose:
		    Performs the Files.content workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    file_id (str): File id value used by the operation.
		    format (str): Format value used by the operation.
		    page_number (int): Page number value used by the operation.
		
		Returns:
		    Dict[str, Any] | None: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			return self.extract(
				file_id=self.file_id,
				format=format,
				page_number=page_number
			)
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'content( self, file_id, format, page_number )'
			Logger( ).write( ex )
			raise ex
	
	def delete( self, file_id: str = None, id: str = None, name: str = None ) -> bool | None:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    name (str): Name value used by the operation.
		
		Returns:
		    Optional[bool]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = file_id or id or name
			throw_if( 'file_id', value )
			self.file_id = str( value ).strip( )
			self.gemini_api_key = cfg.GEMINI_API_KEY
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.client.files.delete( name=self.file_id )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'delete( self, file_id: str=None, id: str=None, name: str=None ) -> bool'
			Logger( ).write( ex )
			raise ex
	
	def summarize( self, prompt: str, filepath: str = None, file_id: str = None,
			id: str = None, model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
			instruct: str = None, tools: List[ str ] = None, tool_choice: str = None,
			include: List[ str ] = None, store: bool = None, stream: bool = None,
			previous_id: str = None, conversation_id: str = None ) -> str | None:
		"""Summarize.
		
		Purpose:
		    Performs the Files.summarize workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    tools (List[str]): Tools value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    include (List[str]): Include value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			
			throw_if( 'model', model )
			self.model = model
			
			self.file_id = file_id or id
			self.file_path = filepath
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tools = tools if tools is not None else [ ]
			self.tool_choice = tool_choice
			self.include = include if include is not None else [ ]
			self.store = store
			self.stream = stream
			self.previous_id = previous_id
			self.conversation_id = conversation_id
			self.gemini_api_key = cfg.GEMINI_API_KEY
			
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			if not self.file_id and not self.file_path:
				raise ValueError( 'A Gemini file ID or local filepath is required.' )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and int( self.max_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_tokens )
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.stops is not None and len( self.stops ) > 0:
				self.config_kwargs[ 'stop_sequences' ] = self.stops
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=self.gemini_api_key )
			
			if self.file_id:
				self.uploaded_file = self.client.files.get( name=self.file_id )
			else:
				self.uploaded_file = self.client.files.upload( path=self.file_path )
			
			if self.uploaded_file is None:
				raise ValueError( 'The Gemini file could not be resolved.' )
			
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.uploaded_file, self.prompt ],
				config=self.content_config
			)
			
			self.output_text = getattr( self.response, 'text', None )
			return self.output_text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'summarize( self, prompt, filepath, file_id, id, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def search( self, prompt: str, filepath: str = None, file_id: str = None,
			id: str = None, model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
			instruct: str = None, tools: List[ str ] = None, tool_choice: str = None,
			include: List[ str ] = None, store: bool = None, stream: bool = None,
			previous_id: str = None, conversation_id: str = None ) -> str | None:
		"""Search.
		
		Purpose:
		    Performs the Files.search workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    filepath (str): Filepath value used by the operation.
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    tools (List[str]): Tools value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    include (List[str]): Include value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			
			return self.summarize( prompt=self.prompt, filepath=filepath, file_id=file_id,
				id=id, model=model, temperature=temperature, top_p=top_p, frequency=frequency,
				presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct,
				tools=tools, tool_choice=tool_choice, include=include, store=store,
				stream=stream, previous_id=previous_id, conversation_id=conversation_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'search( self, prompt, filepath, file_id, id, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def survey( self, prompt: str, filepaths: List[ str ] = None,
			file_id: str = None, id: str = None, model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
			instruct: str = None, tools: List[ str ] = None, tool_choice: str = None,
			include: List[ str ] = None, store: bool = None, stream: bool = None,
			previous_id: str = None, conversation_id: str = None ) -> str | None:
		"""Survey.
		
		Purpose:
		    Performs the Files.survey workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    filepaths (List[str]): Filepaths value used by the operation.
		    file_id (str): File id value used by the operation.
		    id (str): Id value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    instruct (str): Instruct value used by the operation.
		    tools (List[str]): Tools value used by the operation.
		    tool_choice (str): Tool choice value used by the operation.
		    include (List[str]): Include value used by the operation.
		    store (bool): Store value used by the operation.
		    stream (bool): Stream value used by the operation.
		    previous_id (str): Previous id value used by the operation.
		    conversation_id (str): Conversation id value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			
			if file_id or id:
				return self.summarize( prompt=self.prompt, file_id=file_id, id=id,
					model=model, temperature=temperature, top_p=top_p, frequency=frequency,
					presence=presence, max_tokens=max_tokens, stops=stops,
					instruct=instruct, tools=tools, tool_choice=tool_choice,
					include=include, store=store, stream=stream, previous_id=previous_id,
					conversation_id=conversation_id )
			
			throw_if( 'filepaths', filepaths )
			self.file_paths = filepaths
			self.outputs = [ ]
			for filepath in self.file_paths:
				self.outputs.append(
					self.summarize( prompt=self.prompt, filepath=filepath, model=model,
						temperature=temperature, top_p=top_p, frequency=frequency,
						presence=presence, max_tokens=max_tokens, stops=stops,
						instruct=instruct, tools=tools, tool_choice=tool_choice, include=include,
						store=store, stream=stream, previous_id=previous_id,
						conversation_id=conversation_id ) )
			
			return '\n\n'.join( [ item for item in self.outputs if item ] )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'survey( self, prompt, filepaths, file_id, id, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
		"""Web search.
		
		Purpose:
		    Performs the Files.web_search workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    instruct (str): Instruct value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation."""
		try:
			throw_if( 'prompt', prompt )
			self.contents = prompt;
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt, model ) -> Optional[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
			instruct: str = None ) -> str | None:
		"""Search maps.
		
		Purpose:
		    Performs the Files.search_maps workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    prompt (str): Prompt value used by the operation.
		    model (str): Model value used by the operation.
		    temperature (float): Temperature value used by the operation.
		    top_p (float): Top p value used by the operation.
		    frequency (float): Frequency value used by the operation.
		    presence (float): Presence value used by the operation.
		    max_tokens (int): Max tokens value used by the operation.
		    stops (List[str]): Stops value used by the operation.
		    instruct (str): Instruct value used by the operation.
		
		Returns:
		    Optional[str]: Return value produced by the operation."""
		try:
			throw_if( 'prompt', prompt )
			self.contents = f"Using Google Search and Maps data, answer: {prompt}"
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'search_maps( self, prompt, model ) -> Optional[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, file_id: str ) -> bool | None:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    file_id (str): File id value used by the operation.
		
		Returns:
		    Optional[bool]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.client.files.delete( name=self.file_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'FileStore'
			ex.method = 'delete( self, file_id: str ) -> bool'
			Logger( ).write( ex )
			raise ex

class FileSearch( Gemini ):
	"""FileSearch class.
	
	Purpose:
	    Defines the FileSearch component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    client (Optional[genai.Client]): Stores client for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    store_id (Optional[str]): Stores store id for the component runtime state.
	    store_name (Optional[str]): Stores store name for the component runtime state.
	    collections (Optional[Dict[str, str]]): Stores collections for the component runtime state.
	    stores (Optional[List[FileSearchStore]]): Stores stores for the component runtime state."""
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	store_id: Optional[ str ]
	store_name: Optional[ str ]
	collections: Optional[ Dict[ str, str ] ]
	stores: Optional[ List[ FileSearchStore ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.client = None
		self.response = None
		self.store_id = None
		self.store_name = None
		self.collections = { }
		self.stores = [ ]
		self.refresh_collections( )
	
	def refresh_collections( self ) -> Dict[ str, str ]:
		"""Refresh collections.
		
		Purpose:
		    Performs the FileSearch.refresh_collections workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Returns:
		    Dict[str, str]: Return value produced by the operation."""
		try:
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.collections = { }
			self.stores = [ ]
			for store in self.client.file_search_stores.list( ):
				self.stores.append( store )
				self.display_name = getattr( store, 'display_name', None )
				self.resource_name = getattr( store, 'name', None )
				
				if self.resource_name is None:
					continue
				
				self.label = str( self.display_name ).strip( ) if self.display_name else str(
					self.resource_name ).strip( )
				self.collections[ self.label ] = str( self.resource_name ).strip( )
			
			return self.collections
		except Exception:
			self.collections = { }
			self.stores = [ ]
			return self.collections
	
	def create( self, name: str ) -> FileSearchStore | Any:
		"""Create.
		
		Purpose:
		    Performs the FileSearch.create workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    name (str): Name value used by the operation.
		
		Returns:
		    FileSearchStore | Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'name', name )
			self.store_name = str( name ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.file_search_stores.create(
				config={ 'display_name': self.store_name } )
			self.refresh_collections( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'create( self, name: str ) -> FileSearchStore | Any'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> FileSearchStore | Any:
		"""Retrieve.
		
		Purpose:
		    Performs the FileSearch.retrieve workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    FileSearchStore | Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.file_search_stores.get( name=self.store_id )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'retrieve( self, store_id: str ) -> FileSearchStore | Any'
			Logger( ).write( exception )
			raise exception
	
	def list( self ) -> List[ FileSearchStore ] | Any:
		"""List.
		
		Purpose:
		    Performs the FileSearch.list workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Returns:
		    List[FileSearchStore] | Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.refresh_collections( )
			return self.stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'list( self ) -> List[ FileSearchStore ] | Any'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str, force: bool = True ) -> bool | Any:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    store_id (str): Store id value used by the operation.
		    force (bool): Force value used by the operation.
		
		Returns:
		    bool | Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.client.file_search_stores.delete( name=self.store_id,
				config={ 'force': bool( force ) } )
			self.refresh_collections( )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'delete( self, store_id: str, force: bool=True ) -> bool | Any'
			Logger( ).write( exception )
			raise exception
	
	def upload_file( self, path: str, store_id: str, file_path: str = None, id: str = None,
			display_name: str = None, mime_type: str = None ) -> Any:
		"""Upload file.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    path (str): Path value used by the operation.
		    store_id (str): Store id value used by the operation.
		    file_path (str): File path value used by the operation.
		    id (str): Id value used by the operation.
		    display_name (str): Display name value used by the operation.
		    mime_type (str): Mime type value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_path = path or file_path
			throw_if( 'file_path', self.file_path )
			
			self.store_id = store_id or id
			throw_if( 'store_id', self.store_id )
			
			self.file_path = str( self.file_path ).strip( )
			self.store_id = str( self.store_id ).strip( )
			self.display_name = str( display_name or Path( self.file_path ).name ).strip( )
			self.mime_type = mime_type
			self.gemini_api_key = cfg.GEMINI_API_KEY
			
			if self.gemini_api_key is None or not str( self.gemini_api_key ).strip( ):
				raise ValueError( 'GEMINI_API_KEY is required.' )
			
			if not os.path.exists( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			
			self.client = genai.Client( api_key=self.gemini_api_key )
			
			if self.mime_type:
				self.file_config = {
						'display_name': self.display_name,
						'mime_type': self.mime_type,
				}
			else:
				self.file_config = {
						'display_name': self.display_name,
				}
			
			self.uploaded_file = self.client.files.upload( file=self.file_path,
				config=self.file_config )
			
			if self.uploaded_file is None:
				raise ValueError( 'The file could not be uploaded to Gemini Files.' )
			
			self.file_name = getattr( self.uploaded_file, 'name', None )
			if self.file_name is None or not str( self.file_name ).strip( ):
				raise ValueError( 'Gemini Files upload did not return a file resource name.' )
			
			self.response = self.client.file_search_stores.import_file(
				file_search_store_name=self.store_id,
				file_name=self.file_name
			)
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'upload_file( self, path, store_id, file_path, id, display_name, mime_type )'
			Logger( ).write( exception )
			raise exception

class CloudBuckets( Gemini ):
	"""CloudBuckets class.
	
	Purpose:
	    Defines the CloudBuckets component used by the Boo application. The class groups related
	    provider configuration, runtime state, helper methods, and API-facing behavior so Streamlit
	    workflows can call a consistent interface.
	
	Attributes:
	    project_id (Optional[str]): Stores project id for the component runtime state.
	    bucket_name (Optional[str]): Stores bucket name for the component runtime state.
	    object_name (Optional[str]): Stores object name for the component runtime state.
	    file_path (Optional[str]): Stores file path for the component runtime state.
	    file_ids (Optional[List[str]]): Stores file ids for the component runtime state.
	    store_ids (Optional[List[str]]): Stores store ids for the component runtime state.
	    client (Optional[storage.Client]): Stores client for the component runtime state.
	    bucket (Optional[storage.Bucket]): Stores bucket for the component runtime state.
	    response (Optional[Any]): Stores response for the component runtime state.
	    collections (Optional[Dict[str, str]]): Stores collections for the component runtime state.
	    documents (Optional[Dict[str, str]]): Stores documents for the component runtime state."""
	project_id: Optional[ str ]
	bucket_name: Optional[ str ]
	object_name: Optional[ str ]
	file_path: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	store_ids: Optional[ List[ str ] ]
	client: Optional[ storage.Client ]
	bucket: Optional[ storage.Bucket ]
	response: Optional[ Any ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the CloudBuckets object with its default configuration, runtime state, provider
		    settings, and compatibility fields. This constructor prepares the instance for later method
		    calls without performing external work beyond local attribute assignment."""
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.client = storage.Client( project=self.project_id )
		self.bucket_name = None
		self.object_name = None
		self.file_path = None
		self.media_resolution = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.bucket = None
		self.response = None
		self.collections = {
				'Federal Financial Data': 'jeni-financial/data',
				'Federal Financial Regulations': 'jeni-financial/regulations',
				'DoW Financial Data': 'jeni-dow/budget/data',
				'DoW Financial Regulations': 'jeni-dow/budget/regulations',
				'DoA Financial Data': 'jenni-doa/Financial Data',
		}
		self.documents = {
				'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
				'SF133.csv': 'file-32s641QK1Xb5QUatY3zfWF',
				'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
				'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns normalized information for the CloudBuckets component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str] | None: Return value produced by the operation."""
		return [
				'gemini-2.5-flash',
				'gemini-2.5-flash-image',
				'gemini-2.5-flash-tts',
				'gemini-2.5-flash-lite',
				'gemini-2.0-flash',
				'gemini-2.0-flash-lite',
		]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Media options.
		
		Purpose:
		    Returns normalized information for the CloudBuckets component. The method provides a stable view
		    of provider capabilities, stored state, or response metadata so UI controls and downstream logic
		    can consume it consistently.
		
		Returns:
		    List[str]: Return value produced by the operation."""
		return [
				'media_resolution_high',
				'media_resolution_medium',
				'media_resolution_low',
		]
	
	def create( self, bucket: str = None, name: str = None,
			bucket_name: str = None ) -> storage.Bucket:
		"""Create.
		
		Purpose:
		    Performs the CloudBuckets.create workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		
		Returns:
		    storage.Bucket: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			value = bucket or bucket_name or name
			throw_if( 'bucket', value )
			self.bucket_name = str( value ).strip( )
			self.bucket = self.client.bucket( self.bucket_name )
			
			if not self.bucket.exists( ):
				self.bucket = self.client.create_bucket( self.bucket_name )
			
			self.response = self.bucket
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'create( self, bucket: str=None, name: str=None, bucket_name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def create_bucket( self, bucket: str = None, name: str = None,
			bucket_name: str = None ) -> storage.Bucket:
		"""Create bucket.
		
		Purpose:
		    Creates the requested resource, connection, schema object, or user interface artifact using
		    validated inputs. The function encapsulates setup details so callers can rely on a consistent
		    resource lifecycle.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		
		Returns:
		    storage.Bucket: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.create( bucket=bucket, name=name, bucket_name=bucket_name )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'create_bucket( self, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def upload_file( self, path: str = None, bucket: str = None, name: str = None,
			file_path: str = None, bucket_name: str = None, id: str = None,
			store_id: str = None ) -> storage.Blob:
		"""Upload file.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    path (str): Path value used by the operation.
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    file_path (str): File path value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    storage.Blob: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.file_path = path or file_path
			throw_if( 'file_path', self.file_path )
			
			self.bucket_name = bucket or bucket_name or store_id or id
			throw_if( 'bucket', self.bucket_name )
			
			self.file_path = str( self.file_path ).strip( )
			self.bucket_name = str( self.bucket_name ).strip( )
			self.object_name = str( name or Path( self.file_path ).name ).strip( )
			self.bucket = self.client.bucket( self.bucket_name )
			
			if not self.bucket.exists( ):
				raise ValueError( f'Google Cloud Storage bucket not found: {self.bucket_name}' )
			
			self.response = self.bucket.blob( self.object_name )
			self.response.upload_from_filename( self.file_path )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'upload_file( self, path: str=None, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def upload( self, path: str = None, bucket: str = None, name: str = None,
			file_path: str = None, bucket_name: str = None, id: str = None,
			store_id: str = None ) -> storage.Blob:
		"""Upload.
		
		Purpose:
		    Persists or stages input data so it can be used by later provider or application workflows. The
		    function standardizes file handling and returns a stable reference for downstream processing.
		
		Args:
		    path (str): Path value used by the operation.
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    file_path (str): File path value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    storage.Blob: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.upload_file( path=path, bucket=bucket, name=name, file_path=file_path,
				bucket_name=bucket_name, id=id, store_id=store_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'upload( self, path: str=None, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def files_upload( self, path: str = None, bucket: str = None, name: str = None,
			file_path: str = None, bucket_name: str = None, id: str = None,
			store_id: str = None ) -> storage.Blob:
		"""Files upload.
		
		Purpose:
		    Performs the CloudBuckets.files_upload workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    path (str): Path value used by the operation.
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    file_path (str): File path value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    storage.Blob: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.upload_file( path=path, bucket=bucket, name=name, file_path=file_path,
				bucket_name=bucket_name, id=id, store_id=store_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'files_upload( self, path: str=None, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, bucket: str = None, name: str = None,
			bucket_name: str = None, id: str = None, store_id: str = None ) -> Any:
		"""Retrieve.
		
		Purpose:
		    Performs the CloudBuckets.retrieve workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.bucket_name = bucket or bucket_name or store_id or id or name
			throw_if( 'bucket', self.bucket_name )
			self.bucket_name = str( self.bucket_name ).strip( )
			self.bucket = self.client.bucket( self.bucket_name )
			
			if not self.bucket.exists( ):
				raise ValueError( f'Google Cloud Storage bucket not found: {self.bucket_name}' )
			
			if name and name != self.bucket_name:
				self.object_name = str( name ).strip( )
				self.response = self.bucket.get_blob( self.object_name )
				return self.response
			
			self.bucket.reload( )
			self.response = self.bucket
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'retrieve( self, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def retrieve_bucket( self, bucket: str = None, name: str = None,
			bucket_name: str = None, id: str = None, store_id: str = None ) -> Any:
		"""Retrieve bucket.
		
		Purpose:
		    Performs the CloudBuckets.retrieve_bucket workflow using the inputs supplied by the caller and
		    the current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.retrieve( bucket=bucket, name=name, bucket_name=bucket_name,
				id=id, store_id=store_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'retrieve_bucket( self, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def get( self, bucket: str = None, name: str = None,
			bucket_name: str = None, id: str = None, store_id: str = None ) -> Any:
		"""Get.
		
		Purpose:
		    Performs the CloudBuckets.get workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    Any: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.retrieve( bucket=bucket, name=name, bucket_name=bucket_name,
				id=id, store_id=store_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'get( self, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def list( self, bucket: str = None, bucket_name: str = None,
			id: str = None, store_id: str = None ) -> List[ storage.Blob ]:
		"""List.
		
		Purpose:
		    Performs the CloudBuckets.list workflow using the inputs supplied by the caller and the current
		    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
		    data-processing paths can call it consistently.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    List[storage.Blob]: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.bucket_name = bucket or bucket_name or store_id or id
			throw_if( 'bucket', self.bucket_name )
			self.bucket_name = str( self.bucket_name ).strip( )
			self.bucket = self.client.bucket( self.bucket_name )
			if not self.bucket.exists( ):
				raise ValueError( f'Google Cloud Storage bucket not found: {self.bucket_name}' )
			
			self.response = list( self.bucket.list_blobs( ) )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'list( self, bucket: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def delete( self, bucket: str = None, name: str = None,
			bucket_name: str = None, id: str = None, store_id: str = None ) -> bool:
		"""Delete.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			self.bucket_name = bucket or bucket_name or store_id or id or name
			throw_if( 'bucket', self.bucket_name )
			self.bucket_name = str( self.bucket_name ).strip( )
			self.bucket = self.client.bucket( self.bucket_name )
			
			if not self.bucket.exists( ):
				raise ValueError( f'Google Cloud Storage bucket not found: {self.bucket_name}' )
			
			if name and name != self.bucket_name:
				self.object_name = str( name ).strip( )
				self.response = self.bucket.blob( self.object_name )
				self.response.delete( )
				return True
			
			self.bucket.delete( )
			self.response = True
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'delete( self, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def delete_bucket( self, bucket: str = None, name: str = None,
			bucket_name: str = None, id: str = None, store_id: str = None ) -> bool:
		"""Delete bucket.
		
		Purpose:
		    Removes or resets the requested application state or provider resource in a controlled manner.
		    The function keeps cleanup behavior centralized so callers do not duplicate lifecycle logic.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.delete( bucket=bucket, name=name, bucket_name=bucket_name,
				id=id, store_id=store_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'delete_bucket( self, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
	
	def remove( self, bucket: str = None, name: str = None,
			bucket_name: str = None, id: str = None, store_id: str = None ) -> bool:
		"""Remove.
		
		Purpose:
		    Performs the CloudBuckets.remove workflow using the inputs supplied by the caller and the
		    current runtime configuration. The function keeps this behavior isolated so related UI,
		    provider, and data-processing paths can call it consistently.
		
		Args:
		    bucket (str): Bucket value used by the operation.
		    name (str): Name value used by the operation.
		    bucket_name (str): Bucket name value used by the operation.
		    id (str): Id value used by the operation.
		    store_id (str): Store id value used by the operation.
		
		Returns:
		    bool: Return value produced by the operation.
		
		Raises:
		    Exception: Re-raises exceptions after recording them with the application logger."""
		try:
			return self.delete( bucket=bucket, name=name, bucket_name=bucket_name,
				id=id, store_id=store_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'remove( self, bucket: str=None, name: str=None )'
			Logger( ).write( ex )
			raise ex
