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
	        Provides a unified conversational system with explicit methods for
	        querying structured data (SQL), unstructured documents, or free-form
	        chat with an OpenAI LLM. Each method is deterministic and isolates
	        a specific capability.

	    Parameters:
	        db_uri (str):
	            URI string for the SQLite database connection
	            (e.g., 'sqlite:///./finance.db').
	        doc_paths (list[str]):
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
	            Tracks prior conversation context across turns.
	        sql_tool (Tool or None):
	            SQL database query tool initialized for structured data access.
	        doc_tool (Tool or None):
	            Document retrieval tool initialized for unstructured data access.
	        api_tools (list[Tool]):
	            Additional API tools, if configured.
	        agent (AgentExecutor):
	            LangChain agent configured with all available tools, memory, and LLM.
	        __tools (list[Tool]):
	            Internal list of active tools, filtered to exclude None.

	    Methods:
	        query_sql(question: str) -> str:
	            Routes the request exclusively to the SQL database tool, returning
	            answers derived from database queries.

	        query_docs(question: str, with_sources: bool = False) -> str:
	            Routes the request exclusively to the document retriever.
	            Optionally returns cited sources if supported by the retriever.

	        query_chat(prompt: str) -> str:
	            Sends a free-form chat prompt directly to the LLM, bypassing SQL
	            and document retrieval tools.

	    Usage Example:
	        >>> from foo import Fetch
	        >>> import os
	        >>> os.environ["OPENAI_API_KEY"] = "<your_api_key>"

	        >>> fetch = Fetch(
	        ...     db_uri="sqlite:///./finance.db",
	        ...     doc_paths=["./docs/Policy.pdf", "./docs/Notes.txt"],
	        ...     model="gpt-4o-mini",
	        ...     temperature=0.3
	        ... )

	        # Structured query
	        >>> print(fetch.query_sql("List top 5 vendors by obligations."))

	        # Document-grounded question
	        >>> print(fetch.query_docs("What approvals are required for travel?", with_sources=True))

	        # Free-form conversational query
	        >>> print(fetch.query_chat("Summarize the key challenges in federal budgeting."))
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

	# ---------- Explicit entrypoints ----------
	def query_sql( self, question: str ) -> str:
		"""
		Purpose:
			Answer a question using ONLY the SQL tool.

		Parameters:
			question (str):
				Natural language question that maps to a SQL database query.

		Returns:
			str:
				Answer string derived from database queries.
		"""
		try:
			if not question:
				raise Exception( 'Argument "question" cannot be empty' )
			return self.sql_tool.func( question )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Fetch'
			exception.cause = 'query_sql'
			exception.method = 'query_sql(self, question)'
			ErrorDialog( exception ).show( )

	def query_docs( self, question: str, *, with_sources: bool = False ) -> str | None:
		"""

			Purpose:
				Answer a question using ONLY the document retriever.

			Parameters:
				question (str):
					Natural language question to search within the ingested documents.
				with_sources (bool, optional):
					Whether to return document sources with the answer
					(default: False).

			Returns:
				str:
					Answer string based on document retrieval, optionally
					including sources.

		"""
		try:
			if not question:
				raise Exception( 'Argument "question" cannot be empty' )
			if self.doc_tool is None:
				raise Exception( 'No document tool is configured.' )

			if with_sources and hasattr( self, "doc_chain_with_sources" ):
				out = self.doc_chain_with_sources( { "question": question } )
				ans = out.get( "answer", "" )
				src = out.get( "sources", "" )
				return f"{ans}\n\nSOURCES:\n{src}" if src else ans

			return self.doc_tool.func( question )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Fetch'
			exception.cause = 'query_docs'
			exception.method = 'query_docs(self, question, with_sources)'
			ErrorDialog( exception ).show( )

	def query_chat( self, prompt: str ) -> str | None:
		"""

			Purpose:
				Engage in free-form chat with the LLM, bypassing SQL and
				document retrieval tools.

			Parameters:
				prompt (str):
					Free-form conversational input for the LLM.

			Returns:
				str:
					LLM-generated response string.

		"""
		try:
			if not prompt:
				raise Exception( 'Argument "prompt" cannot be empty' )
			return self.llm.invoke( prompt ).content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Fetch'
			exception.cause = 'query_chat'
			exception.method = 'query_chat(self, prompt)'
			ErrorDialog( exception ).show( )

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
