'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                chonk.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="chonk.py" company="Terry D. Eppler">

     This is a Budget Execution and Data Analysis Application for Federal Analysts
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
  '''
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
from booger import Error, ErrorDialog


class Fetch:
	'''
		Class providing lang chain functionality below:
			- OpenAI LLM initialization
			- SQL database agent (via LangChain’s SQL toolkit)
			- Document retriever using FAISS + RetrievalQA
			- Unified agent with a .query() method
	'''
	
	
	def __init__( self, db_uri: str, doc_paths: List[ str ], model: str = 'gpt-4o',
	              temperature: float = 0.8 ):
		'''
			Initializes the Fetch system.
			:param db_uri: URI for the SQLite database
			:param doc_paths: List of file paths to documents (txt, pdf, csv, html)
			:param model: OpenAI model to use (default: gpt-4)
			:param temperature: LLM temperature setting for creativity
		'''
		self.llm = ChatOpenAI( model=model, temperature=temperature, streaming=True )
		self.db_uri = db_uri
		self.doc_paths = doc_paths
		self.memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True )
		self.sql_tool = self._init_sql_tool( )
		self.doc_tool = self._init_doc_tool( )
		self.api_tools = self._init_api_tools( )
		self.agent = initialize_agent(
			tools=[ self.sql_tool, self.doc_tool ] + self.api_tools,
			llm=self.llm,
			memory=self.memory,
			agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
			verbose=True
		)
	
	
	def _init_sql_tool( self ) -> Tool:
		'''
			Sets up SQL querying tool using LangChain's SQLDatabaseToolkit.
			:return: Tool configured for SQL querying
		'''
		try:
			db = SQLDatabase.from_uri( self.db_uri )
			toolkit = SQLDatabaseToolkit( db=db, llm=self.llm )
			base_tool = toolkit.get_tools( )[ 0 ]
			return Tool(
				name='SQLDatabase',
				func=base_tool.func,
				description='Use this to query structured data like employee records, or payroll'
			)
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Chaingle'
			_exc.cause = 'Fetch'
			_exc.method = '_init_sql_tool( self ) -> Tool'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _load_documents( self ):
		'''
			Detects and loads supported documents (TXT, PDF, CSV, HTML) from paths.
			:return: List of loaded documents
		'''
		try:
			all_docs = [ ]
			for path in self.doc_paths:
				ext = os.path.splitext( path )[ -1 ].lower( )
				if ext == '.pdf':
					loader = PyPDFLoader( path )
				elif ext == '.csv':
					loader = CSVLoader( path )
				elif ext in [ '.html', '.htm' ]:
					loader = UnstructuredHTMLLoader( path )
				else:
					loader = TextLoader( path )
				docs = loader.load( )
				all_docs.extend( docs )
			return all_docs
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Chaingle'
			_exc.cause = 'Fetch'
			_exc.method = '_load_documents( self )'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _init_doc_tool( self ) -> Tool:
		'''
			Creates document retrieval tool using FAISS + OpenAI Embeddings.
			:return: Tool configured for document-based Q&A
		'''
		try:
			raw_docs = self._load_documents( )
			chunks = (RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=50 )
			          .split_documents( raw_docs ))
			embeddings = OpenAIEmbeddings( )
			vectordb = FAISS.from_documents( chunks, embeddings )
			retriever = vectordb.as_retriever( )
			qa_chain = RetrievalQA.from_chain_type( llm=self.llm, retriever=retriever )
			name = 'DocumentQA'
			func = qa_chain.run
			description = 'Use this to answer questions from uploaded documents'
			return Tool( name, func, description )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Chaingle'
			_exc.cause = 'Fetch'
			_exc.method = '_init_doc_tool( self ) -> Tool:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _init_api_tools( self ) -> List[ Tool ]:
		'''
			Placeholder for future external API-based tools like SerpAPI, weather, or finance.
			:return: List of external tool integrations (currently empty)
		'''
		try:
			return [ ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Chaingle'
			_exc.cause = 'Fetch'
			_exc.method = '_init_api_tools( self ) -> List[ Tool ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def query( self, question: str ) -> str:
		'''
			Passes a natural language question to the LangChain agent.
			:param question: User query
			:return: Answer generated by the appropriate tool
		'''
		try:
			if question is None:
				_msg = "The object 'question' does not exist or is None!"
				raise Exception( _msg )
			else:
				return self.agent.run( question )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Chaingle'
			_exc.cause = 'Fetch'
			_exc.method = 'query( self, question: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chat_history( self ) -> List[ str ]:
		'''
			Returns formatted conversation history.
			:return: List of strings showing previous user and AI messages
		'''
		try:
			return [ f'{msg.type.upper( )}: {msg.content}' for msg in
			         self.memory.chat_memory.messages ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Chaingle'
			_exc.cause = 'Fetch'
			_exc.method = 'chat_history( self ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
