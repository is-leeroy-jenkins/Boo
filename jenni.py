'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                jenni.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="jenni.py" company="Terry D. Eppler">

	     jenni.py
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
    jenni.py
  </summary>
  ******************************************************************************************
'''
import os

from app import temperature
from boogr import ErrorDialog, Error
import config as cfg
import google
from google import genai
from google.genai import types
from pathlib import Path
from PIL import Image
from requests import Response
from typing import Any, List, Optional, Dict

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Gemini( ):
	'''
	
		Purpose:
		-------
		Base class for Gemma AI Functionality
		
	'''
	project_id: Optional[ str ]
	api_key: Optional[ str ]
	cloud_location: Optional[ str ]
	instructions: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	
	def __init__( self ):
		self.api_key = cfg.GOOGLE_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT
		self.cloud_location = cfg.GOOGLE_CLOUD_LOCATION
		self.model = None
		self.api_version = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.frequency_penalty = None
		self.presence_penalty = None

class Chat( Gemini ):
	'''

	    Purpose:
	    _______
	    Class containing lists of OpenAI models by generation

    '''
	use_vertex: Optional[ bool ]
	http_options: Optional[ Dict[ str, Any ] ]
	config: Optional[ types.GenerateContentConfig ]
	client: Optional[ genai.Client ]
	contents: Optional[ List[ str ] ]
	response: Optional[ Response ]
	image: Optional[ Image ]
	
	def __init__( self, model: str='gemini-2.5-flash', version: str='v1alpha',
			use_ai: bool=True, instruct: str=None, contents: List[ str ]=None ):
		super( ).__init__( )
		self.model = model
		self.api_version = version
		self.use_vertex = use_ai
		self.http_options = { }
		self.contents = contents
		self.instructions = instruct
		self.client = None
		self.config = None
		self.response = None
		self.image = None
		
	@property
	def model_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			_______
			List[ str ] - list of available models

		'''
		return [ 'gemini-3-flash-preview',
		         'gemini-2.5-flash',
		         'gemini-2.0-flash-lite',
		         'gemini-2.5-flash-image',
		         'gemini-2.5-flash-native-audio-preview-12-2025',
		         'gemini-2.5-flash-preview-tts' ]

	@property
	def version_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			--------
			List[ str ] - list of available api versions
			
		'''
		return [ 'v1', 'v1alpha' ]
		
	def generate_text( self, prompt: str ) -> str | None:
		pass

	def generate_image( self, prompt: str ) -> str | None:
		pass

	def analyze_image( self, prompt: str ) -> str | None:
		pass
	
	def summarize_document( self, prompt: str, filepath: str ) -> str | None:
		pass