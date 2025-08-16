'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                Goo.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="Goo.py" company="Terry D. Eppler">

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
    Goo.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations
import os
from langchain.agents.agent import AgentExecutor
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
	TextLoader, PyPDFLoader, CSVLoader, UnstructuredHTMLLoader
)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import Tool
from typing import List, Optional
from boogr import Error, ErrorDialog

class Fetch( ):
	"""
		Purpose:
			Provides a unified interface for querying structured (SQL database) and
			unstructured (documents) data sources through a conversational agent
			powered by OpenAI models. The class initializes an agent with memory and
			multiple tools (SQL, document retrieval, API integrations) to handle
			both structured queries and natural language Q&A.

		Parameters:
			db_uri (str):
				URI string for the SQLite database connection. Example:
				'sqlite:///./finance.db'.
			doc_paths (List[str]):
				List of file paths to documents (txt, pdf, csv, html/htm) to be
				ingested into a vector store for retrieval-augmented generation.
			model (str, optional):
				OpenAI model identifier (default: 'gpt-4o-mini').
			temperature (float, optional):
				Sampling temperature for the LLM, controlling creativity
				(default: 0.8).

		Attributes:
			llm (ChatOpenAI):
				The language model instance used for reasoning and responses.
			memory (ConversationBufferMemory):
				Memory object that tracks prior conversation context.
			sql_tool (Tool or List[Tool]):
				Tool(s) for interacting with the SQL database.
			doc_tool (Tool or None):
				Tool for document-based Q&A, or None if no documents provided.
			api_tools (List[Tool]):
				Optional additional API tools (currently placeholder).
			agent (AgentExecutor):
				LangChain agent initialized with the SQL, document, and API tools.

		Methods:
			query(prompt: str) -> str:
				Runs a natural language query against the agent, which may choose to
				use the SQL tool, document tool, or LLM reasoning directly.
			chat_history() -> List[str]:
				Returns prior conversation history as a list of messages.

		Usage Example:
			>>> from foo_2 import Fetch
			>>> import os
			>>> os.environ["OPENAI_API_KEY"] = "<your_api_key>"

			>>> fetch = Fetch(
			...     db_uri="sqlite:///./finance.db",
			...     doc_paths=["./docs/Policy.pdf", "./docs/Notes.txt"],
			...     model="gpt-4o-mini",
			...     temperature=0.3
			... )

			# Querying the database
			>>> print(fetch.query("List top 5 vendors by obligations."))

			# Asking a document-grounded question
			>>> print(fetch.query("What approvals are required for travel?"))

			# Inspect chat history
			>>> print(fetch.chat_history())
	"""
	db_uri: str | None
	doc_path: str | None
	model: str | None
	temperature: float
	sql_tool: Optional[ Tool ]
	doc_tool: Optional[ Tool ]
	database: Optional[ SQLDatabase ]
	db_toolkit: Optional[ SQLDatabaseToolkit ]
	api_tools: Optional[ List[ Tool ] ]
	extension: Optional[ str ]
	agent: Optional[ AgentExecutor ]
	pdf_loader: Optional[ PyPDFLoader ]
	csv_loader: Optional[ CSVLoader ]
	text_loader: Optional[ TextLoader ]
	html_loader: Optional[ UnstructuredHTMLLoader ]

	def __init__( self, db_uri: str, doc_paths: List[ str ], model: str='gpt-4o-mini',
	              temperature: float=0.8 ):
		"""

			Purpose:
				Initializes the Fetch system, including SQL and Document tools, and conversation memory.
			Parameters:
				db_uri (str):
					URI for the SQLite database.
				doc_paths (List[str]):
					List of file paths to documents (txt, pdf, csv, html/htm).
				model (str):
					OpenAI small model to use (default: gpt-4o-mini).
				temperature (float):
					LLM temperature setting for creativity.
			Returns:
				None

		"""
		# --- Core LLM & config ---
		self.model = model
		self.temperature = temperature
		self.llm = ChatOpenAI( model = self.model, temperature=self.temperature,
			streaming = True )
		self.db_uri = db_uri
		self.doc_paths = doc_paths
		self.memory = ConversationBufferMemory( memory_key = 'chat_history',
			return_messages = True )

		# --- Defaults BEFORE initialization ---
		self.documents = [ ]
		self.db_toolkit = None
		self.database = None
		self.loader = None
		self.tool = None
		self.extension = None

		# --- Initialize tools AFTER defaults ---
		self.sql_tool = self._init_sql_tool( )
		self.doc_tool = self._init_doc_tool( )
		self.api_tools = self._init_api_tools( )

		# --- Build the agent ---
		self.__tools = [ t for t in [ self.sql_tool, self.doc_tool ] + self.api_tools if
		                 t is not None ]
		self.agent = initialize_agent(
			tools = self.__tools,
			llm = self.llm,
			memory = self.memory,
			agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
			verbose = True )

	def _init_sql_tool( self ) -> Tool | None:
		"""

			Purpose:
			---------
				Sets up SQL querying tools using LangChain's SQLDatabaseToolkit.

			Parameters:
			----------
				None

			Returns:
			---------
				list[Tool] | None:
					A list of SQL-related tools (query, schema, list tables), or
					None if initialization fails.

		"""
		try:
			self.database = SQLDatabase.from_uri( self.db_uri )
			self.db_toolkit = SQLDatabaseToolkit( db=self.database, llm=self.llm )
			self.tool=self.db_toolkit.get_tools( )[ 0 ]
			return Tool( name = 'SQLDatabase', func=self.tool.func,
				description='Use this to query structured data like employee count, or payroll' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Fetch'
			exception.method = '_init_sql_tool( self ) -> Tool'
			error = ErrorDialog( exception )
			error.show( )

	def _load_documents( self ):
		"""

			Purpose:
			---------
				Loads and parses documents from the provided file paths. Supports
				PDF, CSV, HTML, and TXT formats.

			Parameters:
			---------
				None

			Returns:
			---------
				list:
					A list of loaded Document objects for downstream chunking and
					embedding.

		"""
		try:
			for _path in self.doc_paths:
				self.extension = os.path.splitext( _path )[ -1 ].lower( )
				if self.extension == '.pdf':
					self.loader = PyPDFLoader( _path )
				elif self.extension == '.csv':
					self.loader = CSVLoader( _path )
				elif self.extension in [ '.html', '.htm' ]:
					self.loader = UnstructuredHTMLLoader( _path )
				else:
					self.loader = TextLoader( _path )
				_docs = self.loader.load( )
				self.documents.extend( _docs )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Fetch'
			exception.method = '_load_documents( self )'
			error = ErrorDialog( exception )
			error.show( )

	def _init_doc_tool( self ) -> Tool | None:
		"""

			Purpose:
			--------
			Creates a document-retrieval tool backed by FAISS and OpenAI
			embeddings. Allows natural language queries to be answered based on
			the uploaded documents.

			Parameters:
			-----------
			None

			Returns:
			-----------
			Tool | None:
			A Tool configured for document-based Q&A, or None if no
			documents are provided or processing fails.

		"""
		try:
			raw_docs = self._load_documents( )
			chunks = (RecursiveCharacterTextSplitter( chunk_size = 500, chunk_overlap = 50 )
			          .split_documents( raw_docs ))
			embeddings = OpenAIEmbeddings( )
			vectordb = FAISS.from_documents( chunks, embeddings )
			retriever = vectordb.as_retriever( )
			qa_chain = RetrievalQA.from_chain_type( llm = self.llm, retriever = retriever,
				chain_type = 'map_reduce' )
			name = 'DocumentQA'
			func = qa_chain.run
			description = 'Use this to answer questions from uploaded documents'
			return Tool( name = name, func = func, description = description )
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Fetch'
			exception.method = '_init_doc_tool( self ) -> Tool:'
			error = ErrorDialog( exception )
			error.show( )

	def _init_api_tools( self ) -> List[ Tool ] | None:
		"""

			Purpose:
			--------
			Placeholder for initializing additional API tools. Can be extended
			to support custom services or endpoints.

			Parameters:
			--------
			None

			Returns:
			--------
			list[Tool]:
			A list of additional API tools (currently empty by default).

		"""
		try:
			return [ ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Fetch'
			exception.method = '_init_api_tools( self ) -> List[ Tool ]'
			error = ErrorDialog( exception )
			error.show( )

	def query( self, prompt: str ) -> str | None:
		"""

			Purpose:
			--------
			Executes a query through the initialized agent. The agent decides
			whether to use the SQL tool, document retrieval, or LLM reasoning
			to generate the answer.

			Parameters:
			--------
			prompt (str):
			The natural language query string.

			Returns:
			--------
			str:
			The agent's response to the query, which may be grounded in
			database results, documents, or general LLM reasoning.

		"""
		try:
			if prompt is None:
				raise Exception( 'The object prompt does not exist or is None!' )
			else:
				return self.agent.run( prompt )
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Fetch'
			exception.method = 'query( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )

	def chat_history( self ) -> List[ str ] | None:
		"""

			Purpose:
			_______
			Retrieves the prior conversation history maintained in memory.

			Parameters:
			_______
			None

			Returns:
			_______
			list[str]:
			A list of prior conversation messages, including both user
			prompts and agent responses.

		"""
		try:
			return [ f'{msg.type.upper( )}: {msg.content}' for msg in
			         self.memory.chat_memory.messages ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Fetch'
			exception.method = 'chat_history( self ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
