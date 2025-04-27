'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                embbr.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="embrr.py" company="Terry D. Eppler">

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
from openai import OpenAI
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
from typing import List, Optional


class Vector( ):
	"""
		
		Vector
		---------
		A class for generating OpenAI embeddings, performing normalization, computing similarity,
		and interacting with OpenAI Vector Stores via the OpenAI API. Includes local
		export/import,
		vector diagnostics, and bulk querying functionality.
	
	"""
	
	
	def __init__( self ):
		"""
			
			Initialize the Vector object with
			OpenAI API credentials and embedding small_model.
	
			Parameters:
			- api_key (Optional[str]): OpenAI API key (uses global config if None)
			- small_model (str): OpenAI embedding small_model to use
		
		"""
		self.small_model = 'text-embedding-3-small'
		self.large_model = 'text-embedding-3-large'
		self.ada_model = 'text-embedding-3-ada'
		self.client = OpenAI( )
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
		self.cache = { }
		self.results = { }
		self.response = None
		self.vector_stores = [ ]
		self.store_ids = [ ]
		self.file_ids = [ ]
		self.vectors = [ ]
		self.batches = [ List[ str ] ]
		self.tables = List[ pd.DataFrame ]
	
	
	def __dir__( self ):
		'''
		
			Purpose:
			Returns a list of class members
		
		'''
		return [ 'small_model', 'large_model', 'ada_model',
		         'client', 'cache', 'results',
		         'response', 'vector_stores', 'file_ids',
		         'batches', 'tables', 'vectors', 'embedd',
		         'most_similar', 'bulk_similar', 'similarity_heatmap',
		         'export_jsonl', 'import_jsonl', 'create_vector_store',
		         'list_vector_stores', 'upload_vector_store',
		         'query_vector_store', 'delete_vector_store' ]
	
	
	def embed( self, texts: List[ str ], batch: int = 10, max: int = 3,
	           time: float = 2.0 ) -> pd.DataFrame:
		"""
		
			Generate and normalize embeddings for a list of input texts.
	
			Parameters:
			- texts (List[str]): List of input pages strings
			- batch (int): Number of texts per API request batch
			- max (int): Number of retries on API failure
			- time (float): Seconds to wait between retries
	
			Returns:
			- pd.DataFrame: DataFrame containing original pages, raw embeddings,
			and normalized embeddings
			
		"""
		try:
			if texts is None:
				raise Exception( 'Input "texts" cannot be None' )
			else:
				self.batches = self._batch_chunks( texts, batch )
				for index, batch in enumerate( self.batches ):
					for attempt in range( max ):
						try:
							self.response = self.client.embeddings.create( input=batch,
								model=self.small_model )
							_vectors = [ record.embedding for record in self.response.data ]
							self.vectors.extend( _vectors )
							break
						except Exception as e:
							print( f'[Batch {index + 1}] Retry {attempt + 1}/{max}: {e}' )
							time.sleep( time )
					else:
						raise RuntimeError( f'Failed after {max} attempts on batch {index + 1}' )
				
				_embeddings = np.array( self.vectors )
				_normed = self._normalize( _embeddings )
				_data = \
					{
						'pages': texts,
						'embedding': list( _embeddings ),
						'normed_embedding': list( _normed )
					}
				
				return pd.DataFrame( _data )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = ('embed( self, texts: List[ str ], batch: int=10, max: int=3, '
			               'time: float=2.0 ) -> pd.DataFrame')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _batch_chunks( self, texts: List[ str ], size: int ) -> List[ List[ str ] ]:
		"""
		
			Split a list of texts into batches of specified size.
	
			Parameters:
			- texts (List[str]): Full list of input strings
			- size (int): Desired batch size
	
			Returns:
			- List of pages batches
		
		"""
		try:
			if texts is None:
				raise Exception( 'Input "texts" cannot be None' )
			elif size is None:
				raise Exception( 'Input "size" cannot be None' )
			else:
				return [ texts[ i:i + size ] for i in range( 0, len( texts ), size ) ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = (' _batch_chunks( self, texts: List[ str ], size: int ) -> [ List[ str '
			               '] ]')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _normalize( self, vector: np.ndarray ) -> np.ndarray:
		"""
		
			Normalize a matrix of vector using L2 norm.
	
			Parameters:
			- vector (np.ndarray): Matrix of vector
	
			Returns:
			- np.ndarray: Normalized vector
			
		"""
		try:
			if vector is None:
				raise Exception( 'Input "vector" cannot be None' )
			else:
				_norms = np.linalg.norm( vector, axis=1, dims=True )
				return vector / np.clip( _norms, 1e-10, None )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = '_normalize( self, vector: np.ndarray ) -> np.ndarray'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _cosine_similarity_matrix( self, vector: np.ndarray, matrix: np.ndarray ) -> np.ndarray:
		"""
		
			Purpose:
			Compute cosine similarity between a query vector and a matrix of vector.
	
			Parameters:
			- vector (np.ndarray): A single normalized vector
			- matrix (np.ndarray): A matrix of normalized vector
	
			Returns:
			- np.ndarray: Cosine similarity scores
			
		"""
		try:
			if vector is None:
				raise Exception( 'Input "vector" cannot be None' )
			elif matrix is None:
				raise Exception( 'Input "matrix" cannot be None' )
			else:
				_query = vector / np.linalg.norm( vector )
				_matrix = matrix / np.linalg.norm( matrix, axis=1, dims=True )
				return np.dot( _matrix, _query )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = (
				'_cosine_similarity_matrix( self, vector: np.ndarray, matrix: np.ndarray '
				') -> np.ndarray')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def most_similar( self, query: str, table: pd.DataFrame, top: int = 5 ) -> pd.DataFrame:
		"""
		
			Purpose:
			Compute most similar rows in a DataFrame using cosine similarity.
	
			Parameters:
			- query (str): Query string to compare
			- table (pd.DataFrame): DataFrame with 'normed_embedding'
			- toptop_k (int): Number of top matches to return
	
			Returns:
			- pd.DataFrame: Top-k results sorted by similarity
			
		"""
		try:
			if query is None:
				raise Exception( 'Input "query" cannot be None' )
			elif table is None:
				raise Exception( 'Input "table" cannot be None' )
			else:
				_embd = self.embed( [ query ] )[ 'normed_embedding' ].iloc[ 0 ]
				_scores = self._cosine_similarity_matrix( _embd,
					np.vstack( table[ 'normed_embedding' ] ) )
				_copy = table.copy( )
				_copy[ 'similarity' ] = _scores
				return _copy.sort_values( 'similarity', ascending=False ).head( top )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = ('most_similar( self, query: str, table: pd.DataFrame, top: int = 5 ) '
			               '-> '
			               'pd.DataFrame')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def bulk_similar( self, queries: List[ str ], dataframe: pd.DataFrame, top: int = 5 ) -> { }:
		"""
		
			Purpose:
			Perform most_similar for a list of queries.
	
			Parameters:
			- queries (List[str]): List of query strings
			- table (pd.DataFrame): DataFrame to search
			- toptop_k (int): Number of top results per query
	
			Returns:
			- Dict[str, pd.DataFrame]: Dictionary of query to top-k results
			
		"""
		try:
			if queries is None:
				raise Exception( 'Input "queries" cannot be None' )
			elif dataframe is None:
				raise Exception( 'Input "dataframe" cannot be None' )
			else:
				_results = { }
				for query in queries:
					_results[ query ] = self.most_similar( query, dataframe, top )
				return _results
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = ('bulk_similar( self, queries: List[ str ], dataframe: pd.DataFrame, '
			               'top: int = 5 ) -> { }')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def similarity_heatmap( self, dataframe: pd.DataFrame ) -> pd.DataFrame:
		"""
		
			Purpose:
			Compute full pairwise cosine similarity heatmap from normed embeddings.
	
			Parameters:
			- table (pd.DataFrame): DataFrame with 'normed_embedding' column
	
			Returns:
			- pd.DataFrame: Pairwise cosine similarity heatmap
			
		"""
		try:
			if dataframe is None:
				raise Exception( 'Input "dataframe" cannot be None' )
			else:
				_matrix = np.vstack( dataframe[ 'normed_embedding' ] )
				_similarity = np.dot( _matrix, _matrix.T )
				return pd.DataFrame( _similarity, index=dataframe[ 'pages' ],
					columns=dataframe[ 'pages' ] )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'similarity_heatmap( self, dataframe: pd.DataFrame ) -> pd.DataFrame'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def export_jsonl( self, dataframe: pd.DataFrame, path: str ) -> None:
		"""
		
			Purpose:
			Export DataFrame of pages and embeddings to a JSONL file.
	
			Parameters:
			- table (pd.DataFrame): DataFrame with 'pages' and 'embedding'
			- path (str): Output path for .jsonl file
		
		"""
		try:
			if dataframe is None:
				raise Exception( 'Input "dataframe" is required.' )
			elif path is None:
				raise Exception( 'Output "path" is required.' )
			else:
				with open( path, 'w', encoding='utf-8' ) as f:
					for _, row in dataframe.iterrows( ):
						record = { 'pages': row[ 'pages' ], 'embedding': row[ 'embedding' ] }
						f.write( json.dumps( record ) + '\n' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'export_jsonl( self, dataframe: pd.DataFrame, path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def import_jsonl( self, path: str ) -> pd.DataFrame:
		"""
		
			Import pages and embeddings
			from a JSONL file into a DataFrame.
	
			Parameters:
			- path (str): Path to the .jsonl file
	
			Returns:
			- pd.DataFrame: DataFrame with normalized embeddings
			
		"""
		try:
			if path is None:
				raise Exception( 'Input "path" must be provided.' )
			else:
				texts, embeddings = [ ], [ ]
				with open( path, 'r', encoding='utf-8' ) as f:
					for line in f:
						_record = json.loads( line.strip( ) )
						texts.append( _record[ 'pages' ] )
						embeddings.append( _record[ 'embedding' ] )
				_normed = self._normalize( np.array( embeddings ) )
				_data = \
				{
					'pages': texts,
					'embedding': embeddings,
					'normed_embedding': list( _normed )
				}
				
				return pd.DataFrame( _data )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'import_jsonl( self, path: str ) -> pd.DataFrame'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_vector_store( self, name: str ) -> str:
		"""
		
			Create a new
			OpenAI vector store.
	
			Parameters:
			- name (str): Name for the vector store
	
			Returns:
			- str: ID of the created vector store
			
		"""
		try:
			if name is None:
				raise Exception( 'Input "name" is required' )
			else:
				self.response = self.client.beta.vector_stores.create( name=name )
				return self.response[ 'id' ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'create_vector_store( self, name: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def list_vector_stores( self ) -> List[ str ]:
		"""
		
			List all available
			OpenAI vector vector_stores.
	
			Returns:
			- List[str]: List of vector store IDs
			
		"""
		try:
			self.response = self.client.beta.vector_stores.list( )
			return [ item[ 'id' ] for item in self.response.get( 'data', [ ] ) ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'list_vector_stores( self ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def upload_vector_store( self, dataframe: pd.DataFrame, ids: str ) -> None:
		"""
		
			Upload documents to a
			 given OpenAI vector store.
	
			Parameters:
			- table (pd.DataFrame): DataFrame with 'pages' column
			- ids (str): OpenAI vector store ID
			
		"""
		try:
			if dataframe is None:
				raise Exception( 'Input "dataframe" cannot be None' )
			elif ids is None:
				raise Exception( 'Input "ids" cannot be None' )
			else:
				documents = [
					{ 'content': row[ 'pages' ], 'metadata': { 'source': f'row_{i}' } }
					for i, row in dataframe.iterrows( )
				]
				self.client.beta.vector_stores.file_batches.create( store_id=ids,
					documents=documents )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'upload_vector_store( self, dataframe: pd.DataFrame, ids: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def query_vector_store( self, id: str, query: str, top: int = 5 ) -> List[ dict ]:
		"""
		
			Purpose:
			Query a vector store using a natural language string.
	
			Parameters:
			- ids (str): OpenAI vector store ID
			- query (str): Search query
			- top (int): Number of results to return
	
			Returns:
			- List[dict]: List of matching documents and similarity scores
			
		"""
		try:
			if id is None:
				raise Exception( 'Input "id" must be provided' )
			elif query is None:
				raise Exception( 'Input "query" must be provided' )
			else:
				self.response = self.client.beta.vector_stores.query( store_id=id, query=query,
					top_k=top )
				return [
					{ 'pages': result[ 'document' ], 'score': result[ 'score' ] }
					for result in self.response.get( 'data', [ ] )
				]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = ('query_vector_store( self, id: str, query: str, top: int = 5 ) -> '
			               'List[ '
			               'dict ]')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def delete_vector_store( self, storeid: str, ids: List[ str ] ) -> None:
		"""
		
			Purpose:
			Delete specific documents from a vector store.
	
			Parameters:
			- ids (str): OpenAI vector store ID
			- ids (List[str]): List of document IDs to delete
			
		"""
		try:
			if storeid is None:
				raise Exception( 'Input "storeid" cannot be None' )
			elif ids is None:
				raise Exception( 'Input "ids" cannot be None' )
			else:
				self.client.beta.vector_stores.documents.delete( store_id=storeid,
					document_ids=ids )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'delete_vector_store( self, storeid: str, ids: List[ str ] ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )


class Xtractor( ):
	"""
	
		Xtractor
		----------------
		A utility class for extracting clean pages from PDF files into a list of strings.
		Handles nuances such as layout artifacts, page separation, optional filtering,
		and includes table detection capabilities.
		
	"""
	
	
	def __init__( self, headers: bool = False, length: int = 10, tables: bool = True ):
		"""
		
			Purpose:
			Initialize the PDF pages extractor with configurable settings.
	
			Parameters:
			- headers (bool): If True, attempts to strip recurring headers/footers.
			- length (int): Minimum number of characters for a line to be included.
			- tables (bool): If True, extract pages from detected tables using block
			grouping.
			
		"""
		self.strip_headers = headers
		self.minimum_length = length
		self.extract_tables = tables
		self.file_path = None
		self.page = None
		self.pages = [ ]
		self.lines = [ ]
		self.clean_lines = [ ]
		self.extracted_lines = [ ]
		self.extracted_tables = [ ]
		self.extracted_pages = [ ]
	
	
	def __dir__( self ):
		'''
			
			Purpose:
			Returns a list of class member names.
		
		'''
		return [ 'strip_headers', 'minimum_length', 'extract_tables',
		         'file_path', 'page', 'pages', 'lines', 'clean_lines', 'extracted_lines',
		         'extracted_tables', 'extracted_pages', 'extract_lines',
		         'extract_text', 'extract_tables', 'export_csv',
		         'export_text', 'export_excel' ]
	
	
	def extract_lines( self, path: str, max: Optional[ int ] = None ) -> List[ str ]:
		"""
			
			Extract lines of pages from a PDF,
			optionally limiting to the first N pages.
	
			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Max number of pages to process (None for all pages)
	
			Returns:
			- List[str]: Cleaned list of non-empty lines
			
		"""
		try:
			if path is None:
				raise Exception( 'Input "path" must be specified' )
			elif max is None:
				raise Exception( 'Input "max" must be specified' )
			else:
				self.file_path = path
				with fitz.open( self.file_path ) as doc:
					for i, page in enumerate( doc ):
						if max is not None and i >= max:
							break
						if self.extract_tables:
							self.extracted_lines = self._extract_table_blocks( page )
						else:
							_text = page.get_text( 'pages' )
							self.lines = _text.splitlines( )
						self.clean_lines = self._filter_lines( self.lines )
						self.extracted_lines.extend( self.clean_lines )
				return self.extracted_lines
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = ('extract_lines( self, path: str, max: Optional[ int ] = None ) -> '
			               'List[ '
			               'str ]')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _extract_table_blocks( self, page ) -> List[ str ]:
		"""
			
			Attempt to extract structured blocks
			such as tables using spatial grouping.
	
			Parameters:
			- page: PyMuPDF page object
	
			Returns:
			- List[str]: Grouped blocks including potential tables
			
		"""
		try:
			if page is None:
				raise Exception( 'Input "page" cannot be None' )
			else:
				_blocks = page.get_text( 'blocks' )
				_sorted = sorted( _blocks, key=lambda b: (round( b[ 1 ], 1 ), round( b[ 0 ], 1 )) )
				self.lines = [ b[ 4 ].strip( ) for b in _sorted if b[ 4 ].strip( ) ]
				return self.lines
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = '_extract_table_blocks( self, page ) -> List[ str ]:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _filter_lines( self, lines: List[ str ] ) -> List[ str ]:
		"""
		
			Filter and clean lines
			 from a page of pages.
	
			Parameters:
			- lines (List[str]): Raw lines of pages
	
			Returns:
			- List[str]: Filtered, non-trivial lines
			
		"""
		try:
			if line is None:
				raise Exception( 'Input "line" is None' )
			else:
				self.lines = lines
				for line in self.lines:
					_line = line.strip( )
					if len( _line ) < self.minimum_length:
						continue
					if self.strip_headers and self._is_repeated_header_or_footer( _line ):
						continue
					self.clean_lines.append( _line )
				return self.clean_lines
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = '_filter_lines( self, lines: List[ str ] ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _is_repeated_header_or_footer( self, line: str ) -> bool:
		"""
			
			Heuristic to detect common
			headers/footers (basic implementation).
	
			Parameters:
			- line (str): A line of pages
	
			Returns:
			- bool: True if line is likely a header or footer
		
		"""
		try:
			if line is None:
				raise Exception( 'Input "line" is None' )
			else:
				_keywords = [ 'page', 'public law', 'u.s. government', 'united states' ]
				return any( kw in line.lower( ) for kw in _keywords )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = '_is_repeated_header_or_footer( self, line: str ) -> bool'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def extract_text( self, path: str, max: Optional[ int ]=None ) -> str:
		"""
		
			Extract the entire pages from a
			PDF into one continuous string.
	
			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum number of pages to process
	
			Returns:
			- str: Full concatenated pages
		
		"""
		try:
			if path is None:
				raise Exception( 'Input "path" must be specified' )
			elif max is None:
				raise Exception( 'Input "max" must be specified' )
			else:
				self.file_path = path
				self.lines = self.extract_lines( self.file_path, max=max )
				return "\n".join( self.lines )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'extract_text( self, path: str, max: Optional[ int ] = None ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def extract_tables( self, path: str, max: Optional[ int ]=None ) -> List[ pd.DataFrame ]:
		"""
			
			Extract tables from the PDF
			and return them as a list of DataFrames.
	
			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum number of pages to process
	
			Returns:
			- List[pd.DataFrame]: List of DataFrames representing detected tables
			
		"""
		try:
			if path is None:
				raise Exception( 'Input "path" must be specified' )
			elif max is None:
				raise Exception( 'Input "max" must be specified' )
			else:
				self.file_path = path
				with fitz.open( self.file_path ) as _doc:
					for i, page in enumerate( _doc ):
						if max is not None and i >= max:
							break
						_blocks = page.find_tables( )
						for _tb in _blocks.tables:
							_df = pd.DataFrame( _tb.extract( ) )
							self.tables.append( _df )
				return self.tables
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = (
				'extract_tables( self, path: str, max: Optional[ int ] = None ) -> List[ '
				'pd.DataFrame ]')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None:
		"""
			
			Export a list of DataFrames (tables)
			to individual CSV files.
	
			Parameters:
			- tables (List[pd.DataFrame]): List of tables to export
			- filename (str): Prefix for output filenames (e.g., 'output_table')
		
		"""
		try:
			if tables is None:
				raise Exception( 'Input "tables" must not be None' )
			elif filename is None:
				raise Exception( 'Input "filename" must not be None' )
			else:
				self.tables = tables
				for i, df in enumerate( self.tables ):
					df.to_csv( f'{filename}_{i + 1}.csv', index=False )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def export_text( self, lines: List[ str ], path: str ) -> None:
		"""
			
			Export extracted lines of
			pages to a plain pages file.
	
			Parameters:
			- lines (List[str]): List of pages lines
			- path (str): Path to output pages file
		
		"""
		try:
			if lines is None:
				raise Exception( 'Input "lines" must be provided.' )
			elif path is None:
				raise Exception( 'Input "path" must be provided.' )
			else:
				self.file_path = path
				self.lines = lines
				with open( self.file_path, 'w', encoding='utf-8' ) as f:
					for line in self.lines:
						f.write( line + "\n" )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'export_text( self, lines: List[ str ], path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None:
		"""
			
			Export all extracted tables into a single
			Excel workbook with one sheet per table.
	
			Parameters:
			- tables (List[pd.DataFrame]): List of tables to export
			- path (str): Path to the output Excel file
		
		"""
		try:
			if tables is None:
				raise Exception( 'Input "tables" must not be None' )
			elif path is None:
				raise Exception( 'Input "path" must not be None' )
			else:
				self.tables = tables
				self.file_path = path
				with pd.ExcelWriter( self.file_path, engine='xlsxwriter' ) as _writer:
					for i, df in enumerate( self.tables ):
						_sheet = f'Table_{i + 1}'
						df.to_excel( writer, sheet_name=_sheet, index=False )
					_writer.save( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
