'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                eddy.py
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
import re
import json
import pandas as pd
import re
import string
import spacy
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
import string


class Embedding( ):
	"""
	Embedding
	---------
	A class for generating OpenAI embeddings, performing normalization, computing similarity,
	and interacting with OpenAI Vector Stores via the OpenAI API.
	"""
	
	
	def __init__( self, api_key: Optional[ str ] = None ):
		"""
		Initialize the Embedding object with OpenAI API credentials and embedding model.

		Parameters:
		- api_key (Optional[str]): OpenAI API key (uses global config if None)
		- model (str): OpenAI embedding model to use
		"""
		self.model = "text-embedding-3-small"
		if api_key:
			openai.api_key = api_key
	
	
	def embed( self, texts: List[ str ], batches: int = 10, retries: int = 3,
	           time: float = 2.0 ) -> pd.DataFrame:
		"""
		
			Generate and normalize embeddings for a list of input texts.
	
			Parameters:
			- texts (List[str]): List of input text strings
			- batches (int): Number of texts per API request batches
			- retries (int): Number of retries on API failure
			- time (float): Seconds to wait between retries
	
			Returns:
			- pd.DataFrame: DataFrame containing original text, raw embeddings, and normalized
			embeddings
			
		"""
		all_embeddings = [ ]
		batches = self._batch_chunks( texts, batches )
		for idx, batches in enumerate( batches ):
			for attempt in range( retries ):
				try:
					response = openai.embeddings.create( input=batches, model=self.model )
					vectors = [ record.embedding for record in response.data ]
					all_embeddings.extend( vectors )
					break
				except Exception as e:
					print( f'[Batch {idx + 1}] Retry {attempt + 1}/{retries}: {e}' )
					time.sleep( time )
			else:
				raise RuntimeError( f'Failed after {retries} attempts on batches {idx + 1}' )
		
		embeddings_np = np.array( all_embeddings )
		normed = self._normalize( embeddings_np )
		
		return pd.DataFrame( {
			'text': texts,
			'embedding': list( embeddings_np ),
			'normed_embedding': list( normed )
		} )
	
	
	def _batch_chunks( self, texts: List[ str ], batch_size: int ) -> List[ List[ str ] ]:
		"""
		
			Break the list of texts into batches of specified size.
	
			Parameters:
			- texts (List[str]): The complete list of input strings
			- batches (int): Size of each batches
	
			Returns:
			- List[List[str]]: Batches of texts
			
		"""
		return [ texts[ i:i + batch_size ] for i in range( 0, len( texts ), batch_size ) ]
	
	
	def _normalize( self, vectors: np.ndarray ) -> np.ndarray:
		"""
		
			Normalize a numpy array of vectors to unit length (L2 norm).
	
			Parameters:
			- vectors (np.ndarray): Array of vectors
	
			Returns:
			- np.ndarray: Normalized vectors
			
		"""
		norms = np.linalg.norm( vectors, axis=1, keepdims=True )
		return vectors / np.clip( norms, 1e-10, None )
	
	
	def _compute_cosine_similarity( self, vec: np.ndarray, matrix: np.ndarray ) -> np.ndarray:
		"""
		
			Compute cosine similarity between a query vector and a matrix of vectors.
	
			Parameters:
			- vec (np.ndarray): Single normalized query vector
			- matrix (np.ndarray): Matrix of normalized vectors
	
			Returns:
			- np.ndarray: Array of cosine similarity scores
			
		"""
		query_norm = vec / np.linalg.norm( vec )
		matrix_norm = matrix / np.linalg.norm( matrix, axis=1, keepdims=True )
		return np.dot( matrix_norm, query_norm )
	
	
	def most_similar( self, query: str, df: pd.DataFrame, top_k: int = 5 ) -> pd.DataFrame:
		"""
		
			Find the top-k most similar entries in a DataFrame given a natural language query.
	
			Parameters:
			- query (str): Natural language query string
			- df (pd.DataFrame): DataFrame with 'normed_embedding' and 'text' columns
			- top_k (int): Number of most similar entries to return
	
			Returns:
			- pd.DataFrame: Top-k results sorted by cosine similarity
			
		"""
		query_embedding = self.embed( [ query ] )[ 'normed_embedding' ].iloc[ 0 ]
		similarity_scores = self._compute_cosine_similarity( query_embedding,
			np.vstack( df[ 'normed_embedding' ] ) )
		df_copy = df.copy( )
		df_copy[ 'similarity' ] = similarity_scores
		return df_copy.sort_values( 'similarity', ascending=False ).head( top_k )
	

	def upload_vector_store( self, df: pd.DataFrame, store_id: str ) -> None:
		"""
		
			Upload a DataFrame of documents and embeddings to an OpenAI vector store.
	
			Parameters:
			- df (pd.DataFrame): DataFrame containing 'text' column to upload
			- store_id (str): Unique ID for the vector store
		
		"""
		documents = [
			{ 'content': row[ 'text' ], 'metadata': { 'source': f'row_{i}' } }
			for i, row in df.iterrows( )
		]
		openai.beta.vector_stores.file_batches.create( store_id=store_id, documents=documents )
	
	
	def query_vector_store( self, store_id: str, query: str, top_k: int = 5 ) -> List[
		dict ]:
		"""
		
			Perform a semantic search in an OpenAI vector store.
	
			Parameters:
			- store_id (str): ID of the OpenAI vector store
			- query (str): The query string to match
			- top_k (int): Number of top results to retrieve
	
			Returns:
			- List[dict]: Matching documents and their similarity scores
			
		"""
		response = openai.beta.vector_stores.query( store_id=store_id, query=query, top_k=top_k )
		return [
			{
				'text': result[ 'document' ],
				'score': result[ 'score' ]
			}
			
			for result in response.get( 'data', [ ] )
		]
