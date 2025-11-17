'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                data.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="data.py" company="Terry D. Eppler">

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
    data.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations
from .boogr import Error, ErrorDialog
import chromadb
from chromadb import Settings
import fitz
import os
import re
import json
import numpy as np
import pandas as pd
import re
import string
import spacy
import openpyxl
from openai import OpenAI
from pathlib import Path
import requests
import sqlite3
from sqlite3 import Connection, Cursor
import tiktoken
from typing import Any, List, Tuple, Optional, Dict

def throw_if( name: str, value: object ):
	"""
	
		Parameters:
		-----------
		:param name:
		:param value:
		
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class SQLite( ):
	"""
	
		Class providing CRUD
		operations for a SQLite database.
	
		Methods:
			- create_table: Creates a df with specified schema.
			- insert: Inserts a record into a df.
			- fetch_all: Fetches all rows from a df.
			- fetch_one: Fetches a single record matching the query.
			- update: Updates rows that match a given condition.
			- delete: Deletes rows that match a given condition.
			- close: Closes the database connection.
		
	"""
	db_path: Optional[ str ]
	connection: Optional[ Connection ]
	cursor: Optional[ Cursor ]
	file_path: Optional[ str ]
	where: Optional[ str ]
	file_name: Optional[ str ]
	table_name: Optional[ str ]
	placeholders: Optional[ List[ str ] ]
	columns: Optional[ List[ str ] ]
	params: Optional[ Tuple ]
	tables: Optional[ List ]
	
	def __init__( self ):
		"""
			
			Pupose:
			Initializes the connection to the SQLite database.
			
			Args:
				db_name (str): The name of the database file.
			
		"""
		self.db_path = r'stores\sqlite\datamodels\Data.db'
		self.conn = sqlite3.connect( self.db_path )
		self.cursor = self.conn.cursor( )
		self.file_path = None
		self.where = None
		self.pairs = None
		self.sql = None
		self.file_name = None
		self.table_name = None
		self.placeholders = [ str ]
		self.columns = [ str ]
		self.params = ( )
		self.column_names = [ str ]
		self.tables = [ ]
	
	def __dir__( self ):
		return [ 'db_path',
		         'conn',
		         'cursor',
		         'path',
		         'where',
		         'pairs',
		         'sql',
		         'file_name',
		         'table_name',
		         'placeholders',
		         'columns',
		         'params',
		         'column_names',
		         'tables',
		         'close',
		         'import_excel',
		         'delete',
		         'update',
		         'insert',
		         'create_table',
		         'fetch_one',
		         'fetch_all' ]
	
	def create( self ) -> None:
		"""

			Purpose:
			Creates the 'embeddings' table with appropriate schema if it does not already exist.

			Returns:
			None

		"""
		try:
			self.cursor.execute( """
                                 CREATE TABLE IF NOT EXISTS embeddings
                                 (
                                     id          INTEGER PRIMARY KEY AUTOINCREMENT,
                                     source_file TEXT    NOT NULL,
                                     chunk_index INTEGER NOT NULL,
                                     chunk_text  TEXT    NOT NULL,
                                     embedding   TEXT    NOT NULL,
                                     created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                                 )""" )
			
			self.conn.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def create_table( self, sql: str ) -> None:
		"""
			
			Purpose:
			Creates a df using a provided SQL statement.
	
			Args:
				sql (str): The CREATE TABLE SQL statement.
		"""
		try:
			throw_if( 'sql', sql )
			self.sql = sql
			self.cursor.execute( self.sql )
			self.conn.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'create_table( self, sql: str ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def insert( self, table: str, columns: List[ str ], values: Tuple[ Any, ... ] ) -> None:
		"""
			
			Purpose:
			--------
			Inserts a new record into a df.
	
			Parameter:
			--------
			table (str): The name of the df.
			columns (List[str]): Column names.
			values (Tuple): Corresponding target_values.
			
		"""
		try:
			throw_if( 'table', table )
			throw_if( 'columns', columns )
			throw_if( 'values', values )
			self.placeholders = ', '.join( '?' for _ in values )
			col_names = ', '.join( columns )
			sql = f'INSERT INTO {table} ({col_names}) VALUES ({self.placeholders})'
			self.cursor.execute( sql, values )
			self.conn.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = ('insert( self, df: str, columns: List[ str ], '
			                    'target_values: Tuple[ Any, ... ] ) -> None')
			error = ErrorDialog( exception )
			error.show( )

	def insert_many( self, source_file: str, chunks: List[ str ], vectors: np.ndarray ) -> None:
		"""
	
			Purpose:
			--------
			Batch inserts multiple chunks and their embeddings into the database.
	
			Parameters:
			--------
			source_file (str): Name or path of the source document.
			chunks (List[str]): List of cleaned text chunks.
			vectors (np.ndarray): Matrix of embedding vectors.
	
			Returns:
			--------
				None
	
		"""
		try:
			records = [ (source_file, i, chunks[ i ], json.dumps( vectors[ i ].tolist( ) ))
			            for i in range( len( chunks ) ) ]
			self.cursor.executemany(
				f''' INSERT INTO {self.table_name} ({self.file_name}, chunk_index,
					chunk_text, embedding)
	                VALUES (?, ?, ?, ?)
				''', records )
			self.conn.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'insert_many'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_all( self, table: str ) -> List[ Tuple ] | None:
		"""
		
			Purpose:
			--------
			Retrieves all rows from a df.
	
			Parameters:
			--------
			table (str): The name of the df.
			
			Returns:
			--------
			List[Tuple]: List of rows.
			
		"""
		try:
			throw_if( 'table', table )
			self.cursor.execute( f'SELECT * FROM {table}' )
			return self.cursor.fetchall( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'fetch_all( self, df: str ) -> List[ Tuple ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_one( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> Tuple | None:
		"""
		
			Purpose:
			--------
			Retrieves a single row matching a WHERE clause.
	
			Parameters:
			--------
			table (str): Table name.
			where (str): WHERE clause (excluding 'WHERE').
			params (Tuple): Parameters for the clause.
			
			Returns:
			--------
			Optional[Tuple]: The fetched row or None.
			
		"""
		try:
			throw_if( 'params', params )
			throw_if( 'where', where )
			throw_if( 'table', table )
			self.table_name = table
			self.where = where
			self.sql = f'SELECT * FROM {self.table_name} WHERE {self.where} LIMIT 1'
			self.cursor.execute( self.sql, self.params )
			return self.cursor.fetchone( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = (
					'fetch_one( self, df: str, where: str, params: Tuple[ Any, ... ] ) -> '
					'Optional[ Tuple ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def update( self, table: str, pairs: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""
		
			Purpose:
			--------
			Updates rows in a df.
	
			Parameters:
			--------
			table (str): Table name.
			pairs (str): SET clause with placeholders.
			where (str): WHERE clause with placeholders.
			params (Tuple): Parameters for both clauses.
			
		"""
		try:
			throw_if( 'pairs', pairs )
			throw_if( 'params', params )
			throw_if( 'where', where )
			throw_if( 'table', table )
			self.table_name = table
			self.where = where
			self.params = params
			self.sql = f'UPDATE {self.table_name} SET {pairs} WHERE {self.where}'
			self.cursor.execute( self.sql, params )
			self.conn.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = (
					'update( self, df: str, pairs: str, where: str, params: Tuple[ Any, '
					'... ] ) -> None')
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""
		
			Purpose:
			--------
			Deletes row matching the given WHERE clause.
	
			Parameters:
			--------
			table (str): Table name.
			where (str): WHERE clause (excluding 'WHERE').
			params (Tuple): Parameters for clause.
				
		"""
		try:
			throw_if( 'where', where )
			throw_if( 'table', table )
			throw_if( 'params', params )
			self.table_name = table
			self.where = where
			self.params = params
			self.sql = f"DELETE FROM {self.table_name} WHERE {self.where}"
			self.cursor.execute( self.sql, self.params )
			self.conn.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'delete( self, df: str, where: str, params: Tuple[ Any] )->None'
			error = ErrorDialog( exception )
			error.show( )
	
	def import_excel( self, path: str ) -> None:
		"""
		
			Purpose:
			--------
			Reads all worksheets from an Excel file into pandas DataFrames and
			stores each as a df in the SQLite database.
		
			Parameters:
			--------
			path (str): Path to the Excel workbook.
			
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.file_name = os.path.basename( self.file_path )
			_excel = pd.ExcelFile( path )
			for _sheet in _excel.sheet_names:
				_df = _excel.parse( _sheet )
				_df.to_sql( _sheet, self.conn, if_exists='replace', index=False )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'import_excel( self, path: str ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def close( self ) -> None:
		"""
	
			Purpose:
			--------
			Closes the database connection.
	
		"""
		try:
			self.conn.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'close( self ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class Chroma( ):
	'''

		Purpose:
		_______
		Provides persistent storage and retrieval of sentence-level embeddings using ChromaDB.
		Supports adding documents, semantic querying, deletion by ID, and disk persistence.

		Parameters:
		__________
		persist_path (str):  Filesystem path for storing ChromaDB collections.
		collection_name (str):  Logical name of the vector store collection.

		Attributes:
		__________
		client (chromadb.Client): Instantiated Chroma client.
		collection (chromadb.Collection): Vector collection used for insert and query.

	'''
	client: chromadb.Client
	collection: chromadb.Collection
	
	def __init__( self, path: str='./chroma', collection: str='embeddings' ) -> None:
		'''

			Purpose:
			__________
			Initializes a persistent Chroma vector database collection.

			Parameters:
			___________
			persist_path (str): Directory to persist collection to disk.
			collection_name (str): Identifier for the Chroma collection.

			Returns:
			_______
			None

		'''
		self.client = chromadb.Client( Settings( persist_directory=path, anonymized_telemetry=False ) )
		self.collection = self.client.get_or_create_collection( name=collection )
	
	def add( self, ids: List[ str ], texts: List[ str ], embeddings: List[ List[ float ] ],
	         metadatas: Optional[ List[ dict ] ] = None ) -> None:
		'''

			Purpose:
			________
			Adds documents, embeddings, and optional metadata to the vector store.

			Parameters:
			___________
			ids (List[str]): Unique identifiers for each record.
			texts (List[str]): Corresponding document strings.
			embeddings (List[List[float]]): Vector representations of documents.
			metadatas (Optional[List[dict]]): Optional metadata for filtering or tagging.

			Returns:
			________
			None

		'''
		try:
			self.collection.add( documents=texts, embeddings=embeddings, ids=ids,
				metadatas=metadatas )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'Chroma'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def query( self, text: List[ str ], num: int=5, where: Optional[ Dict ]=None ) -> List[str]:
		'''

			Purpose:
			________
			Performs similarity-based vector search using provided queries.

			Parameters:
			___________
			query_texts (List[str]): List of queries to run.
			n_results (int): Number of top matches to return.
			where (Optional[dict]): Optional metadata filter to apply.

			Returns:
			________
			List[str]: Most relevant documents based on vector similarity.

		'''
		try:
			result = self.collection.query( query_texts=text, n_results=num, where=where or { } )
			return result.get( 'documents', [ ] )[ 0 ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'Chroma'
			exception.method = 'query'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, ids: List[ str ] ) -> None:
		'''

			Purpose:
			_________
			Deletes one or more records from the collection by document ID.

			Parameters:
			___________
			ids (List[str]): List of unique document IDs to delete.

			Returns:
			________
			None

		'''
		try:
			self.collection.delete( ids=ids )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	def count( self ) -> int | None:
		'''

			Purpose:
			________
			Returns the total number of records in the collection.

			Returns:
			________
			int: Row count of stored vectors.

		'''
		try:
			return self.collection.count( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'Chroma'
			exception.method = 'count'
			error = ErrorDialog( exception )
			error.show( )
	
	def clear( self ) -> None:
		'''

			Purpose:
			-------
			Deletes all documents from the collection.

			Parameters:
			----------
			None

			Returns:
			--------
			None

		'''
		try:
			self.collection.delete( where={ } )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'Chroma'
			exception.method = 'clear'
			error = ErrorDialog( exception )
			error.show( )
	
	def persist( self ) -> None:
		'''

			Purpose:
			---------
			Saves the current state of the collection to disk.

			Returns:
			------
			None

		'''
		try:
			self.client.persist( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'Chroma'
			exception.method = 'persist'
			error = ErrorDialog( exception )
			error.show( )

class GoogleSearchTool( ):
	"""

		Purpose:
			Provides a reusable interface for querying the Google Custom Search API
			and integrates with OpenAI's Tool API if desired.

		Attributes:
			api_key (str): Google API key for Custom Search.
			cse_id (str): Custom Search Engine ID.
			client (OpenAI | None): OpenAI API client instance.

	"""
	api_key: Optional[ str ]
	cse_id: Optional[ str ]
	params: Optional[ Dict[ str, str ] ]
	url: Optional[ str ]
	
	def __init__( self ):
		"""

			Parameters:
			api_key (str): Google API key.
			cse_id (str): Custom Search Engine ID.
			openai_api_key (str | None): Optional OpenAI API key for tool usage.

		"""
		self.api_key = os.getenv( 'GOOGLE_API_KEY' )
		self.cse_id = os.getenv( 'GOOGLE_CSE_ID' )
		self.client = OpenAI( api_key=self.api_key )
		self.url = 'https://cse.google.com/cse?cx=' + self.cse_id
	
	def search( self, query: str, num: int=5 ) -> List[ Dict ]:
		"""

			Purpose:
				Perform a search query using Google's Custom Search JSON API.

			Parameters:
				query (str): Search string.
				num (int): Number of results (max 10 per request).

			Returns:
				list[dict]: Search results with title, link, and snippet.

		"""
		self.params = \
		{
			'q': query,
			'key': self.api_key,
			'cx': self.cse_id,
			'num': num
		}
		
		response = requests.get( self.url, params=self.params )
		response.raise_for_status( )
		results = response.json( )
		return [ { 'title': item[ 'title' ], 'link': item[ 'link' ], 'snippet': item[ 'snippet' ] }
		         for item in results.get( 'items', [ ] ) ]
	
	def get_tool_schema( self ) -> dict:
		"""

			Purpose:
			Return the OpenAI Tool API schema for this search function.

			Returns:
			dict: JSON schema for the tool definition.

		"""
		return \
		{
			'name': 'google_search',
			'description': 'Search Google Custom Search Engine and return top results.',
			'parameters':
			{
				'type': 'object',
				'properties':
				{
						'query':
						{
							'type': 'string',
							'description': 'Search query string'
						}
				},
				'required': [ 'query' ],
			},
		}
	
	def run_with_openai( self, user_message: str, model: str='gpt-5-nana' ) -> str | None:
		"""

			Purpose:
				Run a chat completion request with OpenAI using the Google Search tool.

			Parameters:
				user_message (str): The message to send to the model.
				model (str): OpenAI model to use.

			Returns:
				dict: OpenAI API response.

		"""
		if self.client is None:
			raise RuntimeError( 'OpenAI client not initialized. Provide openai_api_key.' )
		
		return self.client.chat.completions.create(
			model=model,
			messages=[ { 'role': 'user', 'content': user_message } ],
			tools=[ self.get_tool_schema( ) ],
			tool_choice={ 'type': 'function', 'function': { 'name': 'google_search' } },
		).choices[ 0 ].message.content
