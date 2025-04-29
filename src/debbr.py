'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                embbr.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="tiggr.py" company="Terry D. Eppler">

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
  <summary>
  '''
import os
import re
import json
import pandas as pd
import re
import fitz
import string
import spacy
import openpyxl
from openai import OpenAI
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
import sqlite3
from typing import Any, List, Tuple, Optional


class SQLite( ):
	"""
	
		Class providing CRUD
		operations for a SQLite database.
	
		Methods:
			- create_table: Creates a table with specified schema.
			- insert: Inserts a record into a table.
			- fetch_all: Fetches all records from a table.
			- fetch_one: Fetches a single record matching the query.
			- update: Updates records that match a given condition.
			- delete: Deletes records that match a given condition.
			- close: Closes the database connection.
		
	"""
	
	
	def __init__( self  ):
		"""
			
			Pupose:
			Initializes the connection to the SQLite database.
			
			Args:
				db_name (str): The name of the database file.
			
		"""
		self.db_path = r'C:\Users\terry\source\repos\Boo\data\sqlite\datamodels\Data.db'
		self.conn = sqlite3.connect( db_path )
		self.cursor = self.conn.cursor( )
		self.file_path = None
		self.where = None
		self.pairs = None
		self.sql = None
		self.file_name = None
		self.table_name = None
		self.placeholders = [ ]
		self.columns = [ ]
		self.params = ( )
		self.column_names = [ ]
		self.tables = [ ]
	
	
	def __dir__( self ):
		'''

			Purpose:
			Returns a list of members

		'''
		return [ 'db_path', 'conn', 'cursor', 'file_path', 'where',
		         'pairs', 'sql', 'file_name', 'table_name', 'placeholders',
		         'columns', 'params', 'column_names', 'tables',
				 'close', 'import_excel', 'delete', 'update',
		         'insert', 'create_table', 'fetch_one', 'fetch_all' ]
	
	
	def create_table( self, sql: str ) -> None:
		"""
			
			Purpose:
			Creates a table using a provided SQL statement.
	
			Args:
				sql (str): The CREATE TABLE SQL statement.
		"""
		try:
			if sql is None:
				raise Exception( 'The input "sql" cannot be None' )
			else:
				self.sql = sql
				self.cursor.execute( self.sql )
				self.conn.commit( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = 'create_table( self, sql: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )

	
	def insert( self, table: str, columns: List[ str ], values: Tuple[ Any, ... ] ) -> None:
		"""
			
			Purpose:
			Inserts a new record into a table.
	
			Args:
				table (str): The name of the table.
				columns (List[str]): Column names.
				values (Tuple): Corresponding values.
			
		"""
		try:
			if table is None:
				raise Exception( 'The input "table" cannot be None' )
			elif columns is None:
				raise Exception( 'The input "columns" cannot be None' )
			elif values is None:
				raise Exception( 'The input "values" cannot be None' )
			else:
				self.placeholders = ', '.join( '?' for _ in values )
				col_names = ', '.join( columns )
				sql = f'INSERT INTO {table} ({col_names}) VALUES ({placeholders})'
				self.cursor.execute( sql, values )
				self.conn.commit( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = ('insert( self, table: str, columns: List[ str ], values: Tuple[ Any, '
			               '... ] ) -> None')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def fetch_all( self, table: str ) -> List[ Tuple ]:
		"""
		
			Purpose:
			Retrieves all records from a table.
	
			Args:
				table (str): The name of the table.
			
			Returns:
				List[Tuple]: List of rows.
			
		"""
		try:
			self.cursor.execute( f"SELECT * FROM {table}" )
			return self.cursor.fetchall( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = 'fetch_all( self, table: str ) -> List[ Tuple ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def fetch_one( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> Tuple:
		"""
		
			Purpose:
			Retrieves a single row matching a WHERE clause.
	
			Args:
				table (str): Table name.
				where (str): WHERE clause (excluding 'WHERE').
				params (Tuple): Parameters for the clause.
			
			Returns:
				Optional[Tuple]: The fetched row or None.
			
		"""
		try:
			if table is None:
				raise Exception( 'The input "table" cannot be None' )
			elif where is None:
				raise Exception( 'The input "where" cannot be None' )
			elif params is None:
				raise Exception( 'The input "params" cannot be None' )
			else:
				self.table_name = table
				self.where = where
				self.sql = f'SELECT * FROM {self.table_name} WHERE {self.where} LIMIT 1'
				self.cursor.execute( self.sql, self.params )
				return self.cursor.fetchone( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = ('fetch_one( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> '
			               'Optional[ Tuple ]')
			_err = ErrorDialog( _exc )
			_err.show( )

	
	def update( self, table: str, pairs: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""
		
			Purpose:
			Updates records in a table.
	
			Args:
				table (str): Table name.
				pairs (str): SET clause with placeholders.
				where (str): WHERE clause with placeholders.
				params (Tuple): Parameters for both clauses.
			
		"""
		try:
			if table is None:
				raise Exception( 'The input "table" cannot be None' )
			elif where is None:
				raise Exception( 'The input "where" cannot be None' )
			elif params is None:
				raise Exception( 'The input "params" cannot be None' )
			elif pairs is None:
				raise Exception( 'The input "pairs" cannot be None' )
			else:
				self.table_name = table
				self.where = where
				self.params = params
				self.sql = f'UPDATE {self.table_name} SET {pairs} WHERE {self.where}'
				self.cursor.execute( sql, params )
				self.conn.commit( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = ('update( self, table: str, pairs: str, where: str, params: Tuple[ Any, '
			               '... ] ) -> None')
			_err = ErrorDialog( _exc )
			_err.show( )


	def delete( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""
		
			Purpose:
			Deletes records matching the given WHERE clause.
	
			Args:
				table (str): Table name.
				where (str): WHERE clause (excluding 'WHERE').
				params (Tuple): Parameters for clause.
				
		"""
		try:
			if table is None:
				raise Exception( 'The input "table" cannot be None' )
			elif where is None:
				raise Exception( 'The input "where" cannot be None' )
			elif params is None:
				raise Exception( 'The input "params" cannot be None' )
			else:
				self.table_name = table
				self.where = where
				self.params = params
				self.sql = f"DELETE FROM {self.table_name} WHERE {self.where}"
				self.cursor.execute( sql, self.params )
				self.conn.commit( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = 'delete( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )


	def import_excel( self, path: str ) -> None:
		"""
		
			Purpose:
			Reads all worksheets from an Excel file into pandas DataFrames and
			stores each as a table in the SQLite database.
		
			Args:
				path (str): Path to the Excel workbook.
			
		"""
		try:
			if path is None:
				raise Exception( 'The input "path" cannot be None' )
			else:
				self.file_path = path
				self.file_name = os.path.basename( self.file_path )
				xls = pd.ExcelFile( path )
				for sheet_name in xls.sheet_names:
					df = xls.parse( sheet_name )
					df.to_sql( sheet_name, self.conn, if_exists='replace', index=False )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = 'import_excel( self, path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )


	def close( self ) -> None:
		"""

			Closes the database connection.

		"""
		try:
			self.conn.close( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'debbr'
			_exc.cause = 'SQLite'
			_exc.method = 'close( self ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )