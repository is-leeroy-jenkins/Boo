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

	     Boo is a df analysis tool integrating various Generative AI, Text-Processing, and
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
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader, UnstructuredHTMLLoader
from langchain.memory import ConversationBufferMemory
from typing import List

from langchain_core.tools import Tool
from boogr import Error, ErrorDialog


class Fetch:
	'''
	
		Class providing:
			- OpenAI LLM initialization
			- SQL database agent (via LangChain’s SQL toolkit)
			- Document retriever using FAISS + RetrievalQA
			- Unified agent with a .query() method
			
	'''
	
	
	def __init__( self, db_uri: str, doc_paths: List[ str ], model: str='gpt-4o-mini',
	              temperature: float=0.8 ):
		'''

			Initializes the Fetch system.
			:param db_uri: URI for the SQLite database
			:param doc_paths: List of file paths to documents (txt, pdf, csv, raw_html)
			:param model: OpenAI small_model to use (default: openai-4)
			:param temperature: LLM temperature setting for creativity

		'''
		self.llm = ChatOpenAI( model=model, temperature=temperature, streaming=True )
		self.db_uri = db_uri
		self.doc_paths = doc_paths
		self.memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True )
		self.sql_tool = self._init_sql_tool( )
		self.doc_tool = self._init_doc_tool( )
		self.api_tools = self._init_api_tools( )
		self.documents = None
		self.db_toolkit = None
		self.database = None
		self.loader = None
		self.tool = None
		self.extension = None
		self.agent = initialize_agent( tools=[ self.sql_tool, self.doc_tool ] + self.api_tools,
			llm=self.llm,
			memory=self.memory,
			agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
			verbose=True )
	
	
	def _init_sql_tool( self ) -> Tool | None:
		'''

			Purpose:
				Sets up SQL querying tool using LangChain's SQLDatabaseToolkit.

			Return:
				Tool configured for SQL querying

		'''
		try:
			self.database = SQLDatabase.from_uri( self.db_uri )
			self.db_toolkit = SQLDatabaseToolkit( db=self.database, llm=self.llm )
			self.tool = self.db_toolkit.get_tools( )[ 0 ]
			return Tool(
				name='SQLDatabase',
				func=self.tool.func,
				description='Use this to query structured target_values like employee count, or payroll'
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'Goo'
			exception.cause = 'Fetch'
			exception.method = '_init_sql_tool( self ) -> Tool'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def _load_documents( self ):
		'''

			Purpose:
			_______
			Detects and loads supported
			documents (TXT, PDF, CSV, HTML) from paths.
			:return: List of loaded documents

		'''
		try:
			for _path in self.doc_paths:
				self.extension = os.path.splitext( _path )[ -1 ].lower( )
				if self.extension == '.pdf':
					self.loader = PyPDFLoader( _path )
				elif self.extension == '.csv':
					self.loader = CSVLoader( _path )
				elif self.extension in [ '.raw_html', '.htm' ]:
					self.loader = UnstructuredHTMLLoader( _path )
				else:
					self.loader = TextLoader( _path )
				
				_docs = self.loader.load( )
				self.documents.extend( _docs )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Goo'
			exception.cause = 'Fetch'
			exception.method = '_load_documents( self )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def _init_doc_tool( self ) -> Tool | None:
		'''

			Purpose:
				Creates document retrieval
				tool using FAISS + OpenAI Embeddings.

			Returns:
				Tool configured for document-based Q&A
			
		'''
		try:
			raw_docs = self._load_documents( )
			chunks = (RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=50 )
			          .split_documents( raw_docs ) )
			embeddings = OpenAIEmbeddings( )
			vectordb = FAISS.from_documents( chunks, embeddings )
			retriever = vectordb.as_retriever( )
			qa_chain = RetrievalQA.from_chain_type( llm=self.llm, retriever=retriever )
			name = 'DocumentQA'
			func = qa_chain.run
			description = 'Use this to answer questions from uploaded documents'
			return Tool( name, func, description )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Goo'
			exception.cause = 'Fetch'
			exception.method = '_init_doc_tool( self ) -> Tool:'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def _init_api_tools( self ) -> List[ Tool ] | None:
		'''
		
			Placeholder for future external
			API-based tools like SerpAPI, weather, or finance.
			Returns: List of external tool integrations (currently empty)
			
		'''
		try:
			return [ ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'Goo'
			exception.cause = 'Fetch'
			exception.method = '_init_api_tools( self ) -> List[ Tool ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def query( self, prompt: str ) -> str | None:
		'''

			Purpose:
			_______
			Passes a natural language
			prompt to the LangChain agent.

			Param:
			_____
			prompt: User query

			Return:
			______
			str: Answer generated by the appropriate tool
			
		'''
		try:
			if prompt is None:
				raise Exception( 'The object prompt does not exist or is None!' )
			else:
				return self.agent.run( prompt )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Goo'
			exception.cause = 'Fetch'
			exception.method = 'query( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def chat_history( self ) -> List[ str ] | None:
		'''
		
			Returns formatted conversation history.
			:return: List of strings showing previous user and AI messages
			
		'''
		try:
			return [ f'{msg.type.upper( )}: {msg.content}' for msg in
			         self.memory.chat_memory.messages ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'Goo'
			exception.cause = 'Fetch'
			exception.method = 'chat_history( self ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
