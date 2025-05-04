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
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve
from openai import OpenAI
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
from typing import Any, List, Tuple, Optional
import textwrap as tr


class Vector( ):
	"""
		
		Vector
		---------
		A class for generating OpenAI vectors, performing normalization, computing similarity,
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
		self.ada_model = 'text-embedding-ada-002'
		self.client = OpenAI( )
		self.raw_text = None
		self.file_path = None
		self.file_name = None
		self.file_ids = None
		self.directory = None
		self.cache = { }
		self.results = { }
		self.stats = { }
		self.data = { }
		self.id = None
		self.response = None
		self.dataframe = None
		self.vector_stores = [ str ]
		self.store_ids = [ ]
		self.file_ids = [ ]
		self.files = [ str ]
		self.tokens = List[ str ]
		self.array = List[ float ]
		self.vectors = List[ List[ float ] ]
		self.batches = List[ List[ str ] ]
		self.tables = List[ pd.DataFrame ]
	
	
	def __dir__( self ):
		'''
		
			Purpose:
			Returns a list of class members
		
		'''
		return [ 'small_model', 'large_model', 'ada_model',
		         'id', 'files', 'tokens', 'array', 'store_ids',
		         'client', 'cache', 'results', 'directory', 'stats'
		         'response', 'vector_stores', 'file_ids', 'data',
		         'batches', 'tables', 'vectors', 'create', 'dataframe',
		         'most_similar', 'bulk_similar', 'similarity_heatmap',
		         'export_jsonl', 'import_jsonl', 'create_vector_store',
		         'list_vector_stores', 'upload_vector_store',
		         'query_vector_store', 'delete_vector_store',
		         'upload_document', 'upload_documents' ]
	
	
	def create( self, tokens: List[ str ], batch: int=10, max: int=3,
	            time: float=2.0 ) -> pd.DataFrame:
		"""
		
			Generate and normalize
			vectors for a list of input tokens.
	
			Parameters:
			- tokens (List[str]): List of input pages strings
			- batch (int): Number of tokens per API request batch
			- max (int): Number of retries on API failure
			- time (float): Seconds to wait between retries
	
			Returns:
			- pd.DataFrame: DataFrame containin
			g original pages, raw vectors,
			and normalized vectors
			
		"""
		try:
			if tokens is None:
				raise Exception( 'Input "tokens" cannot be None' )
			else:
				self.tokens = tokens
				self.batches = self._batch_chunks( self.tokens, batch )
				self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
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
				
				_embeddings = np.array( self.array )
				_normed = self._normalize( _embeddings )
				self.data = \
				{
					'pages': tokens,
					'embedding': list( _embeddings ),
					'normed_embedding': list( _normed )
				}
				
				return pd.DataFrame( self.data )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = ('create_small_embedding( self, tokens: List[ str ], batch: int=10, max: int=3, '
			               'time: float=2.0 ) -> pd.DataFrame')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _batch_chunks( self, texts: List[ str ], size: int ) -> List[ List[ str ] ]:
		"""
		
			Split a list of tokens
			into batches of specified size.
	
			Parameters:
			- tokens (List[str]): Full list of input strings
			- size (int): Desired batch size
	
			Returns:
			- List of pages batches
		
		"""
		try:
			if texts is None:
				raise Exception( 'Input "tokens" cannot be None' )
			elif size is None:
				raise Exception( 'Input "size" cannot be None' )
			else:
				return [ texts[ i:i + size ] for i in range( 0, len( texts ), size ) ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = (' _batch_chunks( self, tokens: List[ str ], size: int ) -> [ List[ str '
			               '] ]')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def get_purpose_options( self ) -> List[ str ]:
		'''
		
			Returns a list of string representing the purpose of the file
		
		'''
		return [ 'assistants', 'assistants_output', 'batch',
		         'batch_output', 'fine-tune', 'fine-tune-results',
		         'vision' ]
	
	
	def get_type_options( self ) -> { }:
		'''
			
			Returns a dictionary of file formats and types
			
		'''
		return \
			{
				'.c': 'text/x-c',
				'.cpp': 'text/x-c++',
				'.cs': 'text/x-csharp',
				'.css': 'text/css',
				'.doc': 'application/msword',
				'.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
				'.go': 'text/x-golang',
				'.html': 'text/html',
				'.java': 'text/x-java',
				'.js': 'text/javascript',
				'.json': 'application/json',
				'.md': 'text/markdown',
				'.pdf': 'application/pdf',
				'.php': 'text/x-php',
				'.pptx': 'application/vnd.openxmlformats-officedocument.presentationml'
				         '.presentation',
				'.py': 'text/x-python',
				'.py': 'text/x-script.python',
				'.rb': 'text/x-ruby',
				'.sh': 'application/x-sh',
				'.tex': 'text/x-tex',
				'.ts': 'application/typescript',
				'.txt': 'text/plain'
			}
	
	
	def _normalize( self, vector: np.ndarray ) -> np.ndarray:
		"""
		
			Normalize a matrix
			of vector using L2 norm.
	
			Parameters:
			- vector (np.ndarray): Matrix of vector
	
			Returns:
			- np.ndarray: Normalized vector
			
		"""
		try:
			if vector is None:
				raise Exception( 'Input "vector" cannot be None' )
			else:
				self.array = vector
				_norms = np.linalg.norm( self.array, axis=1, dims=True )
				return self.array / np.clip( _norms, 1e-10, None )
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
				self.array = vector
				_query = self.array / np.linalg.norm( self.array )
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
	
	
	def most_similar( self, query: str, df: pd.DataFrame, top: int=5 ) -> pd.DataFrame:
		"""
		
			Purpose:
			Compute most similar rows in a DataFrame using cosine similarity.
	
			Parameters:
			- query (str): Query string to compare
			- df (pd.DataFrame): DataFrame with 'normed_embedding'
			- toptop_k (int): Number of top matches to return
	
			Returns:
			- pd.DataFrame: Top-k results sorted by similarity
			
		"""
		try:
			if query is None:
				raise Exception( 'Input "query" cannot be None' )
			elif df is None:
				self.dataframe = df
				raise Exception( 'Input "df" cannot be None' )
			else:
				_embd = self.create( [ query ] )[ 'normed_embedding' ].iloc[ 0 ]
				_series = np.vstack( self.dataframe[ 'normed_embedding' ] )
				_scores = self._cosine_similarity_matrix( _embd,
					np.vstack( self.dataframe[ 'normed_embedding' ] ) )
				_copy = self.dataframe.copy( )
				_copy[ 'similarity' ] = _scores
				return _copy.sort_values( 'similarity', ascending=False ).head( top )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = ('most_similar( self, query: str, df: pd.DataFrame, top: int = 5 ) '
			               '-> '
			               'pd.DataFrame')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def bulk_similar( self, queries: List[ str ], df: pd.DataFrame, top: int=5 ) -> { }:
		"""
		
			Purpose:
			Perform most_similar for a list of queries.
	
			Parameters:
			- queries (List[str]): List of query strings
			- df (pd.DataFrame): DataFrame to search
			- toptop_k (int): Number of top results per query
	
			Returns:
			- Dict[str, pd.DataFrame]: Dictionary of query to top-k results
			
		"""
		try:
			if queries is None:
				raise Exception( 'Input "queries" cannot be None' )
			elif df is None:
				raise Exception( 'Input "df" cannot be None' )
			else:
				self.dataframe = df
				for query in queries:
					self.results[ query ] = self.most_similar( query, self.dataframe, top )
				return self.results
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = ('bulk_similar( self, queries: List[ str ], df: pd.DataFrame, '
			               'top: int = 5 ) -> { }')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def similarity_heatmap( self, df: pd.DataFrame ) -> pd.DataFrame:
		"""
		
			Purpose:
			Compute full pairwise cosine similarity heatmap from normed vectors.
	
			Parameters:
			- df (pd.DataFrame): DataFrame with 'normed_embedding' column
	
			Returns:
			- pd.DataFrame: Pairwise cosine similarity heatmap
			
		"""
		try:
			if df is None:
				raise Exception( 'Input "df" cannot be None' )
			else:
				self.dataframe = df
				_matrix = np.vstack( self.dataframe[ 'normed_embedding' ] )
				_similarity = np.dot( _matrix, _matrix.T )
				return pd.DataFrame( _similarity, index=self.dataframe[ 'pages' ],
					columns=self.dataframe[ 'pages' ] )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'similarity_heatmap( self, df: pd.DataFrame ) -> pd.DataFrame'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def export_jsonl( self, df: pd.DataFrame, path: str ) -> None:
		"""
		
			Purpose:
			Export DataFrame of pages and vectors to a JSONL file.
	
			Parameters:
			- df (pd.DataFrame): DataFrame with 'pages' and 'embedding'
			- path (str): Output path for .jsonl file
		
		"""
		try:
			if df is None:
				raise Exception( 'Input "df" is required.' )
			elif path is None:
				raise Exception( 'Output "path" is required.' )
			else:
				self.dataframe = df
				self.file_path = path
				self.file_name = os.path.basename( self.file_path )
				self.directory = os.path.dirname( self.file_path )
				with open( path, 'w', encoding='utf-8' ) as f:
					for _, row in self.dataframe.iterrows( ):
						_record = { 'pages': row[ 'pages' ], 'embedding': row[ 'embedding' ] }
						f.write( json.dumps( record ) + '\n' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'export_jsonl( self, df: pd.DataFrame, path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def import_jsonl( self, path: str ) -> pd.DataFrame:
		"""
		
			Purpose:
			Import pages and vectors
			from a JSONL file into a DataFrame.
	
			Parameters:
			- path (str): Path to the .jsonl file
	
			Returns:
			- pd.DataFrame: DataFrame with normalized vectors
			
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
				self.data = \
				{
					'pages': texts,
					'embedding': embeddings,
					'normed_embedding': list( _normed )
				}
				
				return pd.DataFrame( self.data )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'import_jsonl( self, path: str ) -> pd.DataFrame'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_vector_store( self, name: str ) -> str:
		"""
		
			Purpose:
			Creates a new
			OpenAI vector store given a name.
	
			Parameters:
			- name (str): Name for the vector store
	
			Returns:
			- str: ID of the created vector store
			
		"""
		try:
			if name is None:
				raise Exception( 'Input "name" is required' )
			else:
				self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
				self.response = self.client.beta.vector_stores.create_small_embedding( name=name )
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
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.response = self.client.beta.vector_stores.list( )
			return [ item[ 'id' ] for item in self.response.get( 'data', [ ] ) ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'list_vector_stores( self ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def upload_vector_store( self, df: pd.DataFrame, id: str ) -> None:
		"""
		
			Upload documents to a
			 given OpenAI vector store.
	
			Parameters:
			- df (pd.DataFrame): DataFrame with 'pages' column
			- ids (str): OpenAI vector store ID
			
		"""
		try:
			if df is None:
				raise Exception( 'Input "df" cannot be None' )
			elif id is None:
				raise Exception( 'Input "ids" cannot be None' )
			else:
				self.dataframe = df
				self.id = id
				documents = [
					{ 'content': row[ 'pages' ], 'metadata': { 'source': f'row_{i}' } }
					for i, row in self.dataframe.iterrows( ) ]
				
				self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
				self.client.beta.vector_stores.file_batches.create_small_embedding( store_id=self.id,
					documents=documents )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'upload_vector_store( self, df: pd.DataFrame, ids: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def query_vector_store( self, id: str, query: str, top: int=5 ) -> List[ dict ]:
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
				self.id = id
				self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
				self.response = self.client.beta.vector_stores.query( store_id=self.id,
					query=query,
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
			- storeid (str): OpenAI vector store ID
			- ids (List[str]): List of document IDs to delete
			
		"""
		try:
			if storeid is None:
				raise Exception( 'Input "storeid" cannot be None' )
			elif ids is None:
				raise Exception( 'Input "ids" cannot be None' )
			else:
				self.file_ids = ids
				self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
				self.client.beta.vector_stores.documents.delete( store_id=storeid,
					document_ids=self.file_ids )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'delete_vector_store( self, storeid: str, ids: List[ str ] ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def upload_document( self, path: str, id: str ) -> None:
		"""
		
			Purpose:
			Uploads document to vector store given path and id.
	
			Parameters:
			- path (str):  local path to the document
	
			Returns:
			- str:  ID of the  vector store
			
		"""
		try:
			if path is None:
				raise Exception( 'Input "path" cannot be None' )
			elif id is None:
				raise Exception( 'Input "id" cannot be None' )
			else:
				self.file_path = path
				self.file_name = os.path.basename( self.file_path )
				self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
				self.response = self.client.files.create( file=open( self.file_path, 'rb' ),
					purpose="assistants" )
				attach_response = self.client.vector_stores.files.create( vector_store_id=id,
					file_id=self.response.id )
				return { 'file': self.file_name, 'status': 'success' }
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'upload_document( self, path: str, id: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def upload_documents( self, path: str, id: str ) -> None:
		"""
		
			Purpose:
			Uploads documents to vector store given path and id.
	
			Parameters:
			- path (str):  local path to the document
	
			Returns:
			- str:  ID of the  vector store
			
		"""
		try:
			if path is None:
				raise Exception( 'Input "path" cannot be None' )
			elif id is None:
				raise Exception( 'Input "id" cannot be None' )
			else:
				self.file_path = path
				self.id = id
				self.file_name = os.path.basename( self.file_path )
				self.directory = os.path.dirname( self.file_path )
				self.files = [ os.path.join( self.directory, f ) for f in os.listdir( self.directory
				) ]
				self.stats = \
				{
					'total_files': len( self.files ),
					'successful_uploads': 0,
					'failed_uploads': 0,
					'errors': [ ]
				}
				
				with concurrent.futures.ThreadPoolExecutor( max_workers=10 ) as thread:
					_futures = {
						thread.submit( self.upload_document, self.file_path, self.id ): self.file_path
						for self.file_path in self.files }
					for future in tqdm( concurrent.futures.as_completed( _futures ),
							total=len( self.files ) ):
						result = future.result( )
						if result[ 'status' ] == 'success':
							self.stats[ 'successful_uploads' ] += 1
						else:
							self.stats[ 'failed_uploads' ] += 1
							self.stats[ 'errors' ].append( result )
				
				return self.stats
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Vector'
			_exc.method = 'upload_documents( self, path: str, id: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )


class Xtractor( ):
	'''
	
		Purpose:
		Class providing Feature Extraction functionality
	
	'''
	def __init__( self ):
		self.client = OpenAI( )
		self.small_model = 'text-embedding-3-small'
		self.large_model = 'text-embedding-3-large'
		self.ada_model = 'text-embedding-3-ada'
		self.response = None
		self.raw_input = None
		self.tokens = [ str ]
		self.lines = [ str ]
		self.labels = [ str ]
		self.distances = [ float ]
		self.distance_metrics = [ float ]
		self.n_classes = None
		self.data = [ float ]
		self.precision = { }
		self.aeverage_precision = { }
		self.recall = None
	
	
	def create_small_embedding( self, text: str ) -> List[ float ]:
		"""
		
			Purpose:
			Create embeddings using the small model from OpenAI.
	
			Parameters:
			- text (str):  the text to be embedded
	
			Returns:
			- List[ float ]:  embedded embeddings
			
		"""
		try:
			self.raw_input = text.replace( '\n', ' ' )
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.response = self.client.embeddings.create( input=[ self.raw_input ],
				model=self.small_model )
			
			return self.response.data[ 0 ].embedding
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_small_embedding( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_small_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]:
		"""
		
			Purpose:
			Create embeddings using the small model from OpenAI.
	
			Parameters:
			- tokens List[ str ]:  the list of strings (ie., tokens) to be embedded
	
			Returns:
			- List[ List[ float ] ]:  embedded embeddings
			
		"""
		try:
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.tokens = [ t.replace( '\n', ' ' ) for t in tokens ]
			self.data = self.client.embeddings.create( input=self.tokens,
				model=self.small_model ).data
			return [ d.embedding for d in self.data ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_small_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_large_embedding( self, text: str ) -> List[ float ]:
		"""
		
			Purpose:
			Create embeddings using the large model from OpenAI.
	
			Parameters:
			- text (str):  the string (ie, token) to be embedded
	
			Returns:
			- List[ List[ float ] ]:  embedded embeddings
			
		"""
		try:
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.raw_input = text.replace( '\n', ' ' )
			self.response = self.client.embeddings.create( input=[ self.raw_input ],
				model=self.large_model)
			
			return self.response.data[ 0 ].embedding
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_large_embedding( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_large_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]:
		"""
		
			Purpose:
			Create embeddings using the large model from OpenAI.
	
			Parameters:
			- tokens List[ str ]:  the list of strings (ie., tokens) to be embedded
	
			Returns:
			- List[ List[ float ] ]:  embedded embeddings
			
		"""
		try:
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.tokens = [ t.replace( '\n', ' ' ) for t in tokens ]
			self.data = self.client.embeddings.create( input=self.tokens,
				model=self.large_model ).data
			return [ d.embedding for d in self.data ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_large_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]'
			_err = ErrorDialog( _exc )
			_err.show( )
			
			
	def create_ada_embedding( self, text: str ) -> List[ float ]:
		"""
		
			Purpose:
			Create embeddings using the ada model from OpenAI.
	
			Parameters:
			- text (str) :  the text (ie., token) to be embedded
	
			Returns:
			- List[ float ] :  embedded embeddings
			
		"""
		try:
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.raw_input = text.replace( '\n', ' ' )
			self.response = self.client.embeddings.create( input=[ self.raw_input ],
				model=self.ada_model )
			
			return self.response.data[ 0 ].embedding
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_ada_embedding( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_ada_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]:
		"""
		
			Purpose:
			Create embeddings using the ada model from OpenAI.
	
			Parameters:
			- tokens List[ str ]:  the list of strings (ie., tokens) to be embedded
	
			Returns:
			- List[ List[ float ] ]:  embedded embeddings
			
		"""
		try:
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.tokens = [ t.replace( '\n', ' ' ) for t in tokens ]
			self.data = self.client.embeddings.create( input=self.tokens,
				model=self.ada_model ).data
			return [ d.embedding for d in self.data ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_ada_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	async def create_small_async( self, text: str ) -> List[ float ]:
		"""
		
			Purpose:
			Asynchronously creates embeddings using the small model from OpenAI.
	
			Parameters:
			- text (str):  the text to be embedded
	
			Returns:
			- List[ float ]:  embedded embeddings
			
		"""
		try:
			self.raw_input = text.replace( '\n', ' ' )
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			
			return (
				await self.client.embeddings.create( input=[ self.raw_input ], model=self.small_model ))
			[ 'data' ][ 0 ][ 'embedding' ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'ccreate_small_async( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	async def create_large_async( self, text: str ) -> List[ float ]:
		"""
		
			Purpose:
			Asynchronously creates embeddings using the large model from OpenAI.
	
			Parameters:
			- text (str):  the text to be embedded
	
			Returns:
			- List[ float ]:  embedded embeddings
			
		"""
		try:
			self.raw_input = text.replace( '\n', ' ' )
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			
			return (
				await self.client.embeddings.create( input=[ self.raw_input ],
					model=self.large_model ))
			[ 'data' ][ 0 ][ 'embedding' ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_large_async( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	async def create_ada_async( self, text: str ) -> List[ float ]:
		"""
		
			Purpose:
			Asynchronously creates embeddings using the ada model from OpenAI.
	
			Parameters:
			- text (str):  the text to be embedded
	
			Returns:
			- List[ float ]:  embedded embeddings
			
		"""
		try:
			self.raw_input = text.replace( '\n', ' ' )
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			
			return (
				await self.client.embeddings.create( input=[ self.raw_input ],
					model=self.ada_model ))
			[ 'data' ][ 0 ][ 'embedding' ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_ada_async( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def calculate_cosine_similarity( self, a: List[ float ], b: List[ float ] ):
		"""
		
			Purpose:
			Calculates cosine similarity between two vectors 'a' and 'b'.
	
			Parameters:
			- a List[ float ]:  vector 'a',
			- b List[ float ]:  vector 'b'
	
			Returns:
			- List[ float ]:  embedded embeddings
			
		"""
		try:
			return np.dot( a, b ) / (np.linalg.norm( a ) * np.linalg.norm( b ))
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'c calculate_cosine_similarity( self, a, b )'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def plot_multiclass_precision( self, y_score, y_original, classes, classifier ):
		"""
		
			Purpose:
			Calculates cosine similarity between two vectors 'a' and 'b'.
	
			Parameters:
			- a List[ float ]:  vector 'a',
			- b List[ float ]:  vector 'b'
	
			Returns:
			- List[ float ]:  embedded embeddings
			
		"""
		try:
			self.n_classes = len( classes )
			_data = [ (y_original == classes[ i ]) for i in range( self.n_classes ) ]
			y_true = pd.concat( _data, axis=1 ).values
			
			self.precision = dict( )
			self.recall = dict( )
			self.average_precision = dict( )
			for i in range( self.n_classes ):
				self.precision[ i ], self.recall[ i ], _ = precision_recall_curve( y_true[ :, i ],
					y_score[ :, i ] )
				self.average_precision[ i ] = average_precision_score( y_true[ :, i ],
					y_score[ :, i ] )
			
			precision_micro, recall_micro, _ = precision_recall_curve( y_true.ravel( ),
				y_score.ravel( ) )
			self.average_precision = average_precision_score( y_true, y_score, average='micro' )
			print( str( classifier )
			       + ' - Average precision score over all classes: {0:0.2f}'.format(
				self.average_precision
			)
			       )
			
			plt.figure( figsize=( 9, 6 ) )
			f_scores = np.linspace( 0.2, 0.8, num=4 )
			self.lines = [ ]
			self.labels = [ ]
			for f_score in f_scores:
				x = np.linspace( 0.01, 1 )
				y = f_score * x / (2 * x - f_score)
				(l,) = plt.plot( x[ y >= 0 ], y[ y >= 0 ], color='gray', alpha=0.2 )
				plt.annotate( 'f1={0:0.1f}'.format( f_score ), xy=(0.9, y[ 45 ] + 0.02) )
			
			self.lines.append( l )
			self.labels.append( 'iso-f1 curves' )
			(l,) = plt.plot( recall_micro, precision_micro, color="gold", lw=2 )
			self.lines.append( l )
			self.labels.append(
				'average Precision-recall (auprc = {0:0.2f})' ''.format( average_precision_micro )
			)
			
			for i in range( self.n_classes ):
				(l,) = plt.plot( self.recall[ i ], self.precision[ i ], lw=2 )
				self.lines.append( l )
				self.labels.append(
					"Precision-recall for class `{0}` (auprc = {1:0.2f})"
					"".format( classes[ i ], self.average_precision[ i ] ) )
			
			fig = plt.gcf( )
			fig.subplots_adjust( bottom=0.25 )
			plt.xlim( [ 0.0, 1.0 ] )
			plt.ylim( [ 0.0, 1.05 ] )
			plt.xlabel( 'Recall' )
			plt.ylabel( 'Precision' )
			plt.title( f'{classifier}: Precision-Recall curve for each class' )
			plt.legend( self.lines, self.labels )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'plot_multiclass_precision( self, y_score, y_original, classes, classifier )'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def calculate_distances( self, query: List[ float ], embeddings: List[ List[ float ] ],
	                         distance_metric='cosine' ) -> List[ List[ float ] ]:
		"""
		
			Purpose:
			Calculates cosine similarity between two vectors 'a' and 'b'.
	
			Parameters:
			- a List[ float ]:  vector 'a',
			- b List[ float ]:  vector 'b'
	
			Returns:
			- List[ float ]:  embedded embeddings
			
		"""
		try:
			self.distance_metrics = \
			{
				'cosine': spatial.distance.cosine,
				'L1': spatial.distance.cityblock,
				'L2': spatial.distance.euclidean,
				'Linf': spatial.distance.chebyshev,
			}
			
			self.distances = [
				self.distance_metrics[ distance_metric ]( query, embedding )
				for embedding in embeddings ]
			return self.distances
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = ('calculate_distances( self, query: List[ float ], embeddings: '
			               'List[ List[ float ] ],  distance_metric=')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def calculate_nearest_neighbor( self, distances: List[ float ] ) -> np.ndarray:
		'''

			purpose:

		'''
		try:
			self.distances = distances
			return np.argsort( self.distances )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'calculate_nearest_neighbor( self, distances: List[ float ] ) -> np.ndarray'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_pca_components( self, vectors: List[ List[ float ] ], num=2 ) -> np.ndarray:
		"""
			
			Purpose:
			Return the PCA components of a list of vectors.
		
		"""
		try:
			self.vectors = vectors
			pca = PCA( n_components=num )
			array_of_embeddings = np.array( self.vectors )
			return pca.fit_transform( array_of_embeddings )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_pca_components( self, vectors: List[ List[ float ] ], num=2 ) -> np.ndarray'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_tsne_components( self, vectors: List[ List[ float ] ], num=2 ) -> np.ndarray:
		'''

			purpose:

		'''
		try:
			self.vectors = vectors
			tsne = TSNE( n_components=num )
			array_of_embeddings = np.array( self.vectors )
			return tsne.fit_transform( array_of_embeddings )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_tsne_components( self, vectors: List[ List[ float ] ], num=2 ) -> np.ndarray'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_chart( self, components: np.ndarray,
	                  labels: Optional[ List[ str ] ]=None,
	                  strings: Optional[ List[ str ] ]=None,
	                  x_title='Component-0',
	                  y_title='Component-1',
	                  mark_size=5 ) -> None:
		'''

			purpose:

		'''
		try:
			empty_list = [ "" for _ in components ]
			data = pd.DataFrame(
				{
					x_title: components[ :, 0 ],
					y_title: components[ :, 1 ],
					'label': labels if labels else empty_list,
					'string': [ '<br>'.join( tr.wrap( s, width=30 ) ) for s in strings ]
					if strings
					else empty_list,
				} )
			
			chart = px.scatter(
				data,
				x=x_title,
				y=y_title,
				color='label' if labels else None,
				symbol='label' if labels else None,
				hover_data=[ 'string' ] if strings else None
			).update_traces( marker=dict( size=mark_size ) )
			return chart
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = "('create_chart( self, components: np.ndarray  mark_size=5 ) -> None')"
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def creat_3dchart( self,
	                   components: np.ndarray,
	                   labels: Optional[ List[ str ] ]=None,
	                   strings: Optional[ List[ str ] ]=None,
	                   x_title: str='Component-0',
	                   y_title: str='Component-1',
	                   z_title: str='Compontent-2',
	                   mark_size: int=5 ):
		'''

			purpose:

		'''
		try:
			empty_list = [ "" for _ in components ]
			_contents = \
				{
					x_title: components[ :, 0 ],
					y_title: components[ :, 1 ],
					z_title: components[ :, 2 ],
					'label': labels if labels else empty_list,
					'string': [ '<br>'.join( tr.wrap( s, width=30 ) ) for s in strings ]
					if strings
					else empty_list,
				}
			
			data = pd.DataFrame( _contents )
			chart = px.scatter_3d(
				data,
				x=x_title,
				y=y_title,
				z=z_title,
				color='label' if labels else None,
				symbol='label' if labels else None,
				hover_data=[ 'string' ] if strings else None ).update_traces(
				marker=dict( size=mark_size ) )
			return chart
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Xtractor'
			_exc.method = 'create_vector_store( self, name: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
