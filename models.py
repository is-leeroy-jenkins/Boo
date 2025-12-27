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
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Prompt( BaseModel ):
	'''

		Purpose:
		--------
		Represents a structured “system prompt” or instruction bundle used to steer an LLM call.
		This model is intended to capture the canonical components you pass into Boo when you
		want to track prompts as first-class objects (versioning, variables, and provenance).

		Attributes:
		----------
		instructions: Optional[ str ]
			The primary instruction block (typically the system message content).

		context: Optional[ str ]
			Optional background context provided to the model (policies, references, etc.).

		output_indicator: Optional[ str ]
			A short indicator describing the desired output style/format (e.g., "json", "table").

		input_data: Optional[ str ]
			Optional data payload embedded into the prompt (small inputs, examples, etc.).

		id: Optional[ str ]
			Optional identifier for tracking prompts (e.g., GUID, hash, or friendly name).

		version: Optional[ str ]
			Optional version string for prompt management and experimentation.

		format: Optional[ str ]
			Optional format label describing the prompt template type (e.g., "chat", "completion").

		variables: Optional[ List[ str ] ]
			Optional list of placeholder variables referenced by the prompt template.

		question: Optional[ str ]
			Optional question or user query associated with the prompt.

	'''
	instructions: Optional[ str ] = None
	context: Optional[ str ] = None
	output_indicator: Optional[ str ] = None
	input_data: Optional[ str ] = None
	id: Optional[ str ] = None
	version: Optional[ str ] = None
	format: Optional[ str ] = None
	variables: Optional[ List[ str ] ] = None
	question: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Text( BaseModel ):
	'''

		Purpose:
		--------
		Represents a simple text-bearing payload used in structured output. In OpenAI-style
		responses, text can appear as a typed object rather than a raw string. Boo uses this
		model to normalize "text output" as a consistent shape.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator when provided by the upstream API (commonly "text").

		value: Optional[ str ]
			The text content.

	'''
	type: Optional[ str ] = None
	value: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class File( BaseModel ):
	'''

		Purpose:
		--------
		Represents a file-like object returned by an API (uploaded artifacts, generated files,
		or tool outputs). This is intentionally permissive: Boo only needs the common metadata.

		Attributes:
		----------
		filename: Optional[ str ]
			The original or assigned filename.

		bytes: Optional[ int ]
			The size of the file in bytes, if provided.

		created_at: Optional[ int ]
			Unix timestamp of creation (seconds), if provided.

		expires_at: Optional[ int ]
			Unix timestamp when the file expires, if the upstream supports expiring artifacts.

		id: Optional[ str ]
			Unique file identifier in the upstream system.

		object: Optional[ str ]
			Object discriminator from the upstream API (e.g., "file").

		purpose: Optional[ str ]
			Intended purpose for the file in the upstream system (e.g., "assistants", "fine-tune").

	'''
	filename: Optional[ str ] = None
	bytes: Optional[ int ] = None
	created_at: Optional[ int ] = None
	expires_at: Optional[ int ] = None
	id: Optional[ str ] = None
	object: Optional[ str ] = None
	purpose: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Error( BaseModel ):
	'''

		Purpose:
		--------
		Represents an error object returned by an upstream API. Boo stores errors in structured
		form so UI and logging layers can surface details without brittle string parsing.

		Attributes:
		----------
		code: Optional[ str ]
			A short error code when available (e.g., "invalid_request_error").

		message: Optional[ str ]
			Human-readable error message.

	'''
	code: Optional[ str ] = None
	message: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class JsonSchema( BaseModel ):
	'''

		Purpose:
		--------
		Represents a JSON Schema-based response format definition. This is most commonly used
		when you request “structured outputs” constrained by a JSON schema.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the response format (commonly "json_schema").

		name: Optional[ str ]
			A friendly name for the schema/format.

		description: Optional[ str ]
			A description of the schema’s intent.

		schema: Optional[ Dict[ str, Any ] ]
			The JSON Schema definition object (draft-07 compatible patterns are common).

		strict: Optional[ bool ]
			Whether the upstream should enforce strict compliance with the schema.

	'''
	type: Optional[ str ] = None
	name: Optional[ str ] = None
	description: Optional[ str ] = None
	schema: Optional[ Dict[ str, Any ] ] = None
	strict: Optional[ bool ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class JsonObject( BaseModel ):
	'''

		Purpose:
		--------
		Represents a “JSON object” response format definition. This is typically used when you
		want JSON output without providing a full JSON Schema.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the response format (commonly "json_object").

	'''
	type: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Reasoning( BaseModel ):
	'''

		Purpose:
		--------
		Represents reasoning configuration and/or summary data surfaced by an upstream model.
		Boo keeps this structured so callers can persist “reasoning metadata” without coupling
		to a specific vendor response shape.

		Attributes:
		----------
		effort: Optional[ str ]
			A label describing reasoning effort (when supported), e.g., "low", "medium", "high".

		summary: Optional[ str ]
			A short summary of reasoning (when the upstream provides it).

	'''
	effort: Optional[ str ] = None
	summary: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Math( BaseModel ):
	'''

		Purpose:
		--------
		Represents a structured “math solution” style payload (steps + final answer). Boo uses
		this for workflows where a tool or model returns stepwise math output.

		Attributes:
		----------
		steps: Optional[ List[ Step ] ]
			Ordered list of intermediate steps with explanations and outputs.

		final_answer: Optional[ str ]
			Final answer string.

	'''
	class Step( BaseModel ):
		'''

			Purpose:
			--------
			Represents an individual intermediate computation or explanation step.

			Attributes:
			----------
			explanation: Optional[ str ]
				Natural language explanation of the step.

			output: Optional[ str ]
				The computed output (may be numeric or symbolic).

		'''
		explanation: Optional[ str ] = None
		output: Optional[ str ] = None

		class Config:
			arbitrary_types_allowed = True
			extra = 'ignore'

	steps: Optional[ List[ Step ] ] = None
	final_answer: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Document( BaseModel ):
	'''

		Purpose:
		--------
		Represents a generic “document-like” structured output. Boo uses this for demos and
		for workflows where the model produces a multi-field narrative artifact with concepts.

		Attributes:
		----------
		invented_year: Optional[ int ]
			Example field used in some structured-output prompts; can be repurposed.

		summary: Optional[ str ]
			High-level summary of the document.

		inventors: Optional[ List[ str ] ]
			Example list field used in structured-output prompts.

		description: Optional[ str ]
			Long-form description.

		concepts: Optional[ List[ Concept ] ]
			List of extracted or described concepts.

	'''
	invented_year: Optional[ int ] = None
	summary: Optional[ str ] = None
	inventors: Optional[ List[ str ] ] = None
	description: Optional[ str ] = None

	class Concept( BaseModel ):
		'''

			Purpose:
			--------
			Represents a named concept extracted from or associated with a document.

			Attributes:
			----------
			title: Optional[ str ]
				Concept title/name.

			description: Optional[ str ]
				Concept description.

		'''
		title: Optional[ str ] = None
		description: Optional[ str ] = None

		class Config:
			arbitrary_types_allowed = True
			extra = 'ignore'

	concepts: Optional[ List[ Concept ] ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class ResponseFormat( BaseModel ):
	'''

		Purpose:
		--------
		Represents a response formatting directive. This mirrors common “response_format”
		shapes where you can request plain text, JSON object output, or JSON schema-constrained
		structured output.

		Attributes:
		----------
		text: Optional[ Text ]
			Text response format information, when applicable.

		json_schema: Optional[ JsonSchema ]
			Schema definition for structured output.

		json_object: Optional[ JsonObject ]
			JSON object format directive.

	'''
	text: Optional[ Text ] = None
	json_schema: Optional[ JsonSchema ] = None
	json_object: Optional[ JsonObject ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Function( BaseModel ):
	'''

		Purpose:
		--------
		Represents a tool/function descriptor for tool-calling. Boo uses this to keep “tool
		specification” structured (name, description, JSON schema parameters, strictness).

		Attributes:
		----------
		name: Optional[ str ]
			Function name as exposed to the model.

		type: Optional[ str ]
			Type discriminator (commonly "function") when used in tool lists.

		description: Optional[ str ]
			Human-readable description of the tool/function.

		parameters: Optional[ Dict[ str, Any ] ]
			JSON Schema-like parameters object describing accepted inputs.

		strict: Optional[ bool ]
			Whether the upstream should strictly validate arguments against the schema.

	'''
	name: Optional[ str ] = None
	type: Optional[ str ] = None
	description: Optional[ str ] = None
	parameters: Optional[ Dict[ str, Any ] ] = None
	strict: Optional[ bool ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Message( BaseModel ):
	'''

		Purpose:
		--------
		Represents a chat message-like object used by Boo to normalize conversational state.
		This is intentionally general to support both “input messages” and “output messages”.

		Attributes:
		----------
		content: str
			Message content payload. Boo treats this as required for operational messages.

		role: str
			Message role (e.g., "system", "user", "assistant", "tool").

		type: Optional[ str ]
			Optional discriminator if an upstream system emits typed message objects.

		instructions: Optional[ str ]
			Optional per-message instruction string (used in some orchestration patterns).

		data: Optional[ Dict ]
			Optional message metadata or additional structured payload.

	'''
	content: str
	role: str
	type: Optional[ str ] = None
	instructions: Optional[ str ] = None
	data: Optional[ Dict ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Location( BaseModel ):
	'''

		Purpose:
		--------
		Represents a high-level user location descriptor used by web search or other tools.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for location objects.

		city: Optional[ str ]
			City name.

		country: Optional[ str ]
			Country name or code.

		region: Optional[ str ]
			State/province/region.

		timezone: Optional[ str ]
		IANA timezone string when known.

	'''
	type: Optional[ str ] = None
	city: Optional[ str ] = None
	country: Optional[ str ] = None
	region: Optional[ str ] = None
	timezone: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class GeoCoordinates( BaseModel ):
	'''

		Purpose:
		--------
		Represents a latitude/longitude coordinate pair, optionally with a timezone. This is
		useful for tools like web search, maps, or proximity-based retrieval.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for geocoordinate objects.

		latitude: Optional[ float ]
			Latitude in decimal degrees.

		longitude: Optional[ float ]
			Longitude in decimal degrees.

		timezone: Optional[ str ]
			IANA timezone string when known.

	'''
	type: Optional[ str ] = None
	latitude: Optional[ float ] = None
	longitude: Optional[ float ] = None
	timezone: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class FileSearch( BaseModel ):
	'''

		Purpose:
		--------
		Represents configuration for a file-search tool invocation. Boo uses this to keep tool
		config structured and to support serialization/rehydration of tool configurations.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "file_search").

		vector_store_ids: Optional[ List[ str ] ]
			Vector store identifiers available to the search tool.

		max_num_results: Optional[ int ]
			Maximum number of results to return.

		filters: Optional[ Dict[ str, Any ] ]
			Optional filter object (metadata filters, namespace filters, etc.).

	'''
	type: Optional[ str ] = None
	vector_store_ids: Optional[ List[ str ] ] = None
	max_num_results: Optional[ int ] = None
	filters: Optional[ Dict[ str, Any ] ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class WebSearch( BaseModel ):
	'''

		Purpose:
		--------
		Represents configuration for a web-search tool invocation.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "web_search").

		search_context_size: Optional[ str ]
			Desired context size (vendor-specific; common values are "low", "medium", "high").

		user_location: Optional[ Any ]
			Optional location descriptor to bias search results. This may be a Location,
			GeoCoordinates, or a vendor-specific object.

	'''
	type: Optional[ str ] = None
	search_context_size: Optional[ str ] = None
	user_location: Optional[ Any ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class ComputerUse( BaseModel ):
	'''

		Purpose:
		--------
		Represents configuration for a computer-use tool invocation (UI automation / virtual
		display sessions).

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "computer_use").

		display_height: Optional[ int ]
			Height (pixels) of the virtual display.

		display_width: Optional[ int ]
			Width (pixels) of the virtual display.

		environment: Optional[ str ]
			Environment label (e.g., "browser", "desktop") when supported by the tool provider.

	'''
	type: Optional[ str ] = None
	display_height: Optional[ int ] = None
	display_width: Optional[ int ] = None
	environment: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class WxForecast( BaseModel ):
	'''

		Purpose:
		--------
		Represents a simplified weather forecast payload returned by a tool or model.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		temperature: Optional[ int ]
			Temperature value (units depend on the tool/provider).

		precipitation: Optional[ int ]
			Precipitation percentage or amount (provider-specific).

		sky_conditions: Optional[ str ]
			Text description such as "clear", "cloudy", "rain".

	'''
	type: Optional[ str ] = None
	temperature: Optional[ int ] = None
	precipitation: Optional[ int ] = None
	sky_conditions: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Directions( BaseModel ):
	'''

		Purpose:
		--------
		Represents a simplified directions/route payload returned by a mapping tool.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		route: Optional[ Any ]
			Route representation (provider-specific). Frequently this is a string/polyline or
			a structured list of steps.

	'''
	type: Optional[ str ] = None
	route: Optional[ Any ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class ValidationStatus( BaseModel ):
	'''

		Purpose:
		--------
		Represents validation results produced by Boo (or by a tool/model) when generating
		structured artifacts (SQL, JSON, schemas, etc.).

		Attributes:
		----------
		is_valid: Optional[ bool ]
			True if the artifact passed validation; False otherwise.

		syntax_errors: Optional[ List[ str ] ]
			List of syntax/validation error messages. Empty or None when valid.

	'''
	is_valid: Optional[ bool ] = None
	syntax_errors: Optional[ List[ str ] ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class SqlStatement( BaseModel ):
	'''

		Purpose:
		--------
		Represents a canonical set of SQL commands associated with a table. This model is
		useful when a model/tool generates multiple related statements (DDL + DML patterns).

		Attributes:
		----------
		table_name: Optional[ str ]
			Name of the table associated with the commands.

		create_command: Optional[ str ]
			CREATE TABLE or equivalent DDL.

		select_command: Optional[ str ]
			SELECT statement template or example query.

		insert_command: Optional[ str ]
			INSERT statement template.

		delete_command: Optional[ str ]
			DELETE statement template.

		update_command: Optional[ str ]
			UPDATE statement template.

	'''
	table_name: Optional[ str ] = None
	create_command: Optional[ str ] = None
	select_command: Optional[ str ] = None
	insert_command: Optional[ str ] = None
	delete_command: Optional[ str ] = None
	update_command: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class SQLQueryGeneration( BaseModel ):
	'''

		Purpose:
		--------
		Represents a structured SQL query generation result. This is intended for Boo’s
		SQL-oriented tooling where you want the SQL plus metadata and validation.

		Attributes:
		----------
		query: str
			The generated SQL query.

		query_type: str
			Query type label (e.g., "SELECT", "INSERT", "UPDATE", "DDL").

		tables_used: List[ str ]
			List of table names referenced by the query.

		estimated_complexity: str
			Complexity label (e.g., "simple", "moderate", "complex").

		execution_notes: List[ str ]
			Notes about execution assumptions, indexes, filters, or constraints.

		validation_status: ValidationStatus
			Validation status object describing whether the SQL is syntactically/semantically valid.

	'''
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
		Represents right ascension / declination coordinate pairs used in astronomy-oriented
		structured outputs.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		declination: Optional[ float ]
			Declination in decimal degrees.

		right_ascension: Optional[ float ]
			Right ascension in decimal degrees or hours (provider-specific).

	'''
	type: Optional[ str ] = None
	declination: Optional[ float ] = None
	right_ascension: Optional[ float ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Response( BaseModel ):
	'''

		Purpose:
		--------
		Represents a general-purpose “OpenAI-style” response envelope suitable for Boo’s
		structured output handling. This model intentionally supports both request parameters
		(as echoed back) and response payload fields, while remaining permissive to vendor
		evolution via `extra='ignore'`.

		Attributes:
		----------
		id: Optional[ str ]
			Response identifier produced by the upstream API.

		object: Optional[ str ]
			Object discriminator (e.g., "response"). Kept as string for typical API payloads.

		created: Optional[ int ]
			Unix timestamp (seconds) when the response was created.

		model: Optional[ str ]
			Model identifier used to produce the response.

		status: Optional[ str ]
			Status indicator (e.g., "completed", "in_progress", "failed") when provided.

		error: Optional[ Error ]
			Error details when the response failed.

		input: Optional[ List[ Dict[ str, Any ] ] ]
			The input items or messages sent to the upstream API, when returned/recorded.

		output: Optional[ List[ Dict[ str, Any ] ] ]
			The output items returned by the upstream API (messages, tool calls, etc.).
			Kept generic because output item schemas can vary by provider and feature set.

		usage: Optional[ Dict[ str, Any ] ]
			Token usage or billing-related metadata when provided.

		metadata: Optional[ Dict[ str, Any ] ]
			Arbitrary metadata bag for Boo or vendor-provided fields.

		previous_response_id: Optional[ str ]
			Identifier of the previous response in a conversation chain, if applicable.

		reasoning: Optional[ Reasoning ]
			Reasoning configuration/summary when supported.

		response_format: Optional[ ResponseFormat ]
			The response format settings used or returned by the upstream system.

		temperature: Optional[ float ]
			Sampling temperature used for generation.

		top_p: Optional[ float ]
			Nucleus sampling parameter.

		max_output_tokens: Optional[ int ]
			Max output tokens cap.

		store: Optional[ bool ]
			Whether the upstream should store the response (provider-specific).

		stream: Optional[ bool ]
			Whether streaming was requested.

		parallel_tool_calls: Optional[ bool ]
			Whether parallel tool calls were enabled.

		tool_choice: Optional[ Any ]
			Tool choice directive (string or structured object, depending on API).

		tools: Optional[ List[ Function ] ]
			Tool definitions available for tool calling.

		instructions: Optional[ str ]
			System instructions echoed or stored with the response.

		include: Optional[ List[ str ] ]
			Optional include directives (fields to expand/include in responses).

		truncation: Optional[ str ]
			Truncation strategy label when supported by the upstream.

		role: Optional[ str ]
			High-level role label if the provider surfaces it at the envelope level.

		text: Optional[ Text ]
			Convenience field used by Boo when a response is reducible to a simple text object.

		data: Optional[ Dict[ str, Any ] ]
			Generic structured data payload when a tool/model emits a single JSON object.

	'''
	id: Optional[ str ] = None
	object: Optional[ str ] = None
	created: Optional[ int ] = None
	model: Optional[ str ] = None
	status: Optional[ str ] = None
	error: Optional[ Error ] = None

	input: Optional[ List[ Dict[ str, Any ] ] ] = None
	output: Optional[ List[ Dict[ str, Any ] ] ] = None
	usage: Optional[ Dict[ str, Any ] ] = None
	metadata: Optional[ Dict[ str, Any ] ] = None

	previous_response_id: Optional[ str ] = None
	reasoning: Optional[ Reasoning ] = None
	response_format: Optional[ ResponseFormat ] = None

	temperature: Optional[ float ] = None
	top_p: Optional[ float ] = None
	max_output_tokens: Optional[ int ] = None

	store: Optional[ bool ] = None
	stream: Optional[ bool ] = None
	parallel_tool_calls: Optional[ bool ] = None

	tool_choice: Optional[ Any ] = None
	tools: Optional[ List[ Function ] ] = None

	instructions: Optional[ str ] = None
	include: Optional[ List[ str ] ] = None
	truncation: Optional[ str ] = None
	role: Optional[ str ] = None

	text: Optional[ Text ] = None
	data: Optional[ Dict[ str, Any ] ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
