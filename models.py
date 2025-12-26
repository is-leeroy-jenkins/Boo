'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                models.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="models.py" company="Terry D. Eppler">

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
    models.py
  </summary>
  ******************************************************************************************
  '''
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Prompt( BaseModel ):
	'''

		Purpose:
		--------
		Class for the user's location

	'''
	instructions: Optional[ str ]
	context: Optional[ str ]
	output_indicator: Optional[ str ]
	input_data: Optional[ str ]
	id: Optional[ str ]
	version: Optional[ str ]
	format: Optional[ str ]
	variables: Optional[ List[ str ] ]
	question: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class Text( BaseModel ):
	'''

		Purpose:
		--------
		A class used to generate text responses.

	'''
	type: Optional[ str ]
	value: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class File( BaseModel ):
	'''

		Purpose:
		--------
		A class used to represent GptFile objects.

	'''
	filename: Optional[ str ]
	bytes: Optional[ int ]
	created_at: Optional[ int ]
	expires_at: Optional[ int ]
	id: Optional[ str ]
	object: Optional[ str ]
	purpose: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class Error( BaseModel ):
	'''

		Purpose:
		--------
		A class for exceptions

	'''
	code: Optional[ str ]
	message: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class JsonSchema( BaseModel ):
	'''

		Purpose:
		--------
		A class used to generate json schema responses.

	'''
	type: Optional[ str ]
	name: Optional[ str ]
	description: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class JsonObject( BaseModel ):
	'''

		Purpose:
		--------
		A class used to generate json schema responses.

	'''
	type: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		
class Reasoning( BaseModel ):
	'''

		Purpose:
		--------
		Class providing reasoning functionality.

	'''
	effort: Optional[ str ]
	summary: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class Math( BaseModel ):
	class Step( BaseModel ):
		explanation: Optional[ str ]
		output: Optional[ str ]

	steps: Optional[ List[ Step ] ]
	final_answer: Optional[ str ]


class Document( BaseModel ):
	invented_year: Optional[ int ]
	summary: Optional[ str ]
	inventors: Optional[ List[ str ] ]
	description: Optional[ str ]

	class Concept( BaseModel ):
		title: Optional[ str ]
		description: Optional[ str ]

	concepts: Optional[ List[ Concept ] ]

class Response( BaseModel ):
	'''

		Purpose:
		--------
		Base class for GPT responses.

	'''
	id: Optional[ str ]
	object: Optional[ object ]
	input: Optional[ List[ Dict[ str, Any ] ] ]
	model: Optional[ str ]
	include: Optional[ List[ str ] ]
	instructions: Optional[ str ]
	max_output_tokens: Optional[ int ]
	previous_response_id: Optional[ int ]
	reasoning: Optional[ str ]
	role: Optional[ str ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	parallel_tool_calls: Optional[ bool ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	truncation: Optional[ str ]
	text: Optional[ Text ]
	status: Optional[ str ]
	created: Optional[ str ]
	data: Optional[ Dict[ str, Any ] ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class ResponseFormat( BaseModel ):
	'''

		Purpose:
		--------
		A class for objects specifying the format that the model must output.

	'''
	text: Optional[ Text ]
	json_schema: Optional[ JsonSchema ]
	json_object: Optional[ JsonObject ]

	class Config:
		arbitrary_types_allowed = True

class FileSearch( BaseModel ):
	'''

		Purpose:
		--------
		A tool that searches for relevant content from uploaded file

	'''
	type: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	max_num_results: Optional[ int ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class WebSearch( BaseModel ):
	'''

		Purpose:
		--------
		Class for a tool that searches the web
		for relevant results to use in a response.

	'''
	type: Optional[ str ]
	search_context_size: Optional[ str ]
	user_location: Optional[ object ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class ComputerUse( BaseModel ):
	'''

		Purpose:
		--------
		A class for a tool that controls a virtual computer

	'''
	type: Optional[ str ]
	display_height: Optional[ int ]
	display_width: Optional[ int ]
	environment: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class WxForecast( BaseModel ):
	'''

		Purpose:
		--------
		A class for a tool that forecasts weather

	'''
	type: Optional[ str ]
	temperature: Optional[ int ]
	precipitation: Optional[ int ]
	sky_conditions: Optional[ str ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class Directions( BaseModel ):
	'''

		Purpose:
		--------
		A class for a tool that provides directons

	'''
	type: Optional[ str ]
	route: Optional[ int ]
	
	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class Function( BaseModel ):
	'''

		Class for a function the model can choose to call

	'''
	name: Optional[ str ]
	type: Optional[ str ]
	description: Optional[ str ]
	parameters: Optional[ List[ str ] ]
	strict: Optional[ bool ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class Message( BaseModel ):
	'''

		Purpose:
		--------
		Class representing the system message

	'''
	content: str
	role: str
	type: Optional[ str ]
	instructions: Optional[ str ]
	data: Optional[ Dict ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class Location( BaseModel ):
	'''

		Purpose:
		--------
		Class for the user's location

	'''
	type: Optional[ str ]
	city: Optional[ str ]
	country: Optional[ str ]
	region: Optional[ str ]
	timezone: Optional[ str ]
	
	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class GeoCoordinates( BaseModel ):
	'''

		Purpose:
		--------
		Class for the user's location

	'''
	type: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]
	
	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class ValidationStatus( BaseModel ):
	is_valid: bool
	syntax_errors: List[ str ]
	
	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class SQLQueryGeneration( BaseModel ):
	query: str
	query_type: str
	tables_used: List[ str ]
	estimated_complexity: str
	execution_notes: List[ str ]
	validation_status: ValidationStatus
	
	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'

class SkyCoordinates( BaseModel ):
	'''

		Purpose:
		--------
		Class for the user's location

	'''
	type: Optional[ str ]
	declination: Optional[ float ]
	right_ascension: Optional[ float ]
	
	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
