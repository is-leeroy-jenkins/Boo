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
from openai import OpenAI
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
from typing import List, Optional


class Embedding( ):
	"""
		
		Embedding
		---------
		A class for generating OpenAI embeddings, performing normalization, computing similarity,
		and interacting with OpenAI Vector Stores via the OpenAI API. Includes local export/import,
		vector diagnostics, and bulk querying functionality.
	
	"""
	
	
	def __init__( self ):
		"""
			
			Initialize the Embedding object with OpenAI API credentials and embedding model.
	
			Parameters:
			- api_key (Optional[str]): OpenAI API key (uses global config if None)
			- model (str): OpenAI embedding model to use
		
		"""
		self.model = 'text-embedding-3-small'
		self.client = OpenAI( )
		self.cache = { }
		self.response = None
		self.vector_stores = List[ str ]
		self.store_ids = List[ str ]
		self.file_ids = List[ str ]
		self.vectors = [ ]
		self.batches = List[ List[ str ] ]
		self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
	
	
	def embed( self, texts: List[ str ], batch: int = 10, max: int = 3,
	           time: float = 2.0 ) -> pd.DataFrame:
		"""
		
			Generate and normalize embeddings for a list of input texts.
	
			Parameters:
			- texts (List[str]): List of input text strings
			- batch (int): Number of texts per API request batch
			- max (int): Number of retries on API failure
			- time (float): Seconds to wait between retries
	
			Returns:
			- pd.DataFrame: DataFrame containing original text, raw embeddings,
			and normalized embeddings
			
		"""
		self.vectors = [ ]
		self.batches = self._batch_chunks( texts, batch )
		for idx, batch in enumerate( self.batches ):
			for attempt in range( max ):
				try:
					self.response = self.client.embeddings.create( input=batch, model=self.model )
					self.vectors = [ record.embedding for record in self.response.data ]
					self.vectors.extend( self.vectors )
					break
				except Exception as e:
					print( f'[Batch {idx + 1}] Retry {attempt + 1}/{max}: {e}' )
					time.sleep( time )
			else:
				raise RuntimeError( f'Failed after {max} attempts on batch {idx + 1}' )
		
		embeddings_np = np.array( self.vectors )
		normed = self._normalize( embeddings_np )
		
		return pd.DataFrame( {
			'text': texts,
			'embedding': list( embeddings_np ),
			'normed_embedding': list( normed )
		} )
	
	
	def _batch_chunks( self, texts: List[ str ], size: int ) -> List[ List[ str ] ]:
		"""
		
			Split a list of texts into batches of specified size.
	
			Parameters:
			- texts (List[str]): Full list of input strings
			- batch (int): Desired batch size
	
			Returns:
			- List of text batches
		
		"""
		return [ texts[ i:i + size ] for i in range( 0, len( texts ), size ) ]
	
	
	def _normalize( self, vector: np.ndarray ) -> np.ndarray:
		"""
		
			Normalize a matrix of vector using L2 norm.
	
			Parameters:
			- vector (np.ndarray): Matrix of vector
	
			Returns:
			- np.ndarray: Normalized vector
			
		"""
		norms = np.linalg.norm( vector, axis=1, dims=True )
		return vector / np.clip( norms, 1e-10, None )
	
	
	def _cosine_similarity_matrix( self, vector: np.ndarray, matrix: np.ndarray ) -> np.ndarray:
		"""
		
			Compute cosine similarity between a query vector and a matrix of vector.
	
			Parameters:
			- vector (np.ndarray): A single normalized vector
			- matrix (np.ndarray): A matrix of normalized vector
	
			Returns:
			- np.ndarray: Cosine similarity scores
			
		"""
		query_norm = vector / np.linalg.norm( vector )
		matrix_norm = matrix / np.linalg.norm( matrix, axis=1, dims=True )
		return np.dot( matrix_norm, query_norm )
	
	
	def most_similar( self, query: str, dataframe: pd.DataFrame, top: int = 5 ) -> pd.DataFrame:
		"""
		
			Compute most similar rows in a DataFrame using cosine similarity.
	
			Parameters:
			- query (str): Query string to compare
			- dataframe (pd.DataFrame): DataFrame with 'normed_embedding'
			- toptop_k (int): Number of top matches to return
	
			Returns:
			- pd.DataFrame: Top-k results sorted by similarity
			
		"""
		query_embedding = self.embed( [ query ] )[ 'normed_embedding' ].iloc[ 0 ]
		similarity_scores = self._cosine_similarity_matrix( query_embedding,
			np.vstack( dataframe[ 'normed_embedding' ] ) )
		df_copy = dataframe.copy( )
		df_copy[ 'similarity' ] = similarity_scores
		return df_copy.sort_values( 'similarity', ascending=False ).head( top )
	
	
	def bulk_similar( self, queries: List[ str ], dataframe: pd.DataFrame, top: int = 5 ) -> { }:
		"""
		
			Perform most_similar for a list of queries.
	
			Parameters:
			- queries (List[str]): List of query strings
			- dataframe (pd.DataFrame): DataFrame to search
			- toptop_k (int): Number of top results per query
	
			Returns:
			- Dict[str, pd.DataFrame]: Dictionary of query to top-k results
			
		"""
		results = { }
		for query in queries:
			results[ query ] = self.most_similar( query, dataframe, top )
		return results
	
	
	def similarity_heatmap( self, dataframe: pd.DataFrame ) -> pd.DataFrame:
		"""
		
			Compute full pairwise cosine similarity heatmap from normed embeddings.
	
			Parameters:
			- dataframe (pd.DataFrame): DataFrame with 'normed_embedding' column
	
			Returns:
			- pd.DataFrame: Pairwise cosine similarity heatmap
			
		"""
		matrix = np.vstack( dataframe[ 'normed_embedding' ] )
		similarity_matrix = np.dot( matrix, matrix.T )
		return pd.DataFrame( similarity_matrix, index=dataframe[ 'text' ],
			columns=dataframe[ 'text' ] )
	
	
	def export_jsonl( self, dataframe: pd.DataFrame, path: str ) -> None:
		"""
			
			Export DataFrame of text and embeddings to a JSONL file.
	
			Parameters:
			- dataframe (pd.DataFrame): DataFrame with 'text' and 'embedding'
			- path (str): Output path for .jsonl file
		
		"""
		with open( path, 'w', encoding='utf-8' ) as f:
			for _, row in dataframe.iterrows( ):
				record = { 'text': row[ 'text' ], 'embedding': row[ 'embedding' ] }
				f.write( json.dumps( record ) + '\n' )
	
	
	def import_jsonl( self, path: str ) -> pd.DataFrame:
		"""
		
			Import text and embeddings from a JSONL file into a DataFrame.
	
			Parameters:
			- path (str): Path to the .jsonl file
	
			Returns:
			- pd.DataFrame: DataFrame with normalized embeddings
			
		"""
		texts, embeddings = [ ], [ ]
		with open( path, 'r', encoding='utf-8' ) as f:
			for line in f:
				record = json.loads( line.strip( ) )
				texts.append( record[ 'text' ] )
				embeddings.append( record[ 'embedding' ] )
		normed = self._normalize( np.array( embeddings ) )
		return pd.DataFrame(
			{ 'text': texts, 'embedding': embeddings, 'normed_embedding': list( normed ) } )
	
	
	def create_vector_store( self, name: str ) -> str:
		"""
		
			Create a new OpenAI vector store.
	
			Parameters:
			- name (str): Name for the vector store
	
			Returns:
			- str: ID of the created vector store
			
		"""
		self.response = self.client.beta.vector_stores.create( name=name )
		return self.response[ 'id' ]
	
	
	def list_vector_stores( self ) -> List[ str ]:
		"""
		
			List all available OpenAI vector vector_stores.
	
			Returns:
			- List[str]: List of vector store IDs
			
		"""
		self.response = self.client.beta.vector_stores.list( )
		return [ item[ 'id' ] for item in self.response.get( 'data', [ ] ) ]
	
	
	def upload_vector_store( self, dataframe: pd.DataFrame, storeids: str ) -> None:
		"""
		
			Upload documents to a given OpenAI vector store.
	
			Parameters:
			- dataframe (pd.DataFrame): DataFrame with 'text' column
			- storeids (str): OpenAI vector store ID
			
		"""
		documents = [
			{ 'content': row[ 'text' ], 'metadata': { 'source': f'row_{i}' } }
			for i, row in dataframe.iterrows( )
		]
		self.client.beta.vector_stores.file_batches.create( store_id=storeids,
			documents=documents )
	
	
	def query_vector_store( self, id: str, query: str, top: int = 5 ) -> List[ dict ]:
		"""
		
			Query a vector store using a natural language string.
	
			Parameters:
			- storeids (str): OpenAI vector store ID
			- query (str): Search query
			- top (int): Number of results to return
	
			Returns:
			- List[dict]: List of matching documents and similarity scores
			
		"""
		self.response = self.client.beta.vector_stores.query( store_id=id, query=query, top_k=top )
		return [
			{ 'text': result[ 'document' ], 'score': result[ 'score' ] }
			for result in self.response.get( 'data', [ ] )
		]
	
	
	def delete_vector_store( self, storeid: str, docids
	: List[ str ] ) -> None:
		"""
		
			Delete specific documents from a vector store.
	
			Parameters:
			- storeids (str): OpenAI vector store ID
			- docids (List[str]): List of document IDs to delete
			
		"""
		self.client.beta.vector_stores.documents.delete( store_id=storeid, document_ids=docids )


class Extractor( ):
	"""
	
		Extractor
		----------------
		A utility class for extracting clean text from PDF files into a list of strings.
		Handles nuances such as layout artifacts, page separation, optional filtering,
		and includes table detection capabilities.
		
	"""
	
	
	def __init__( self, headers: bool = False, length: int = 10,
	              tables: bool = True ):
		"""
			
			Initialize the PDF text extractor with configurable settings.
	
			Parameters:
			- headers (bool): If True, attempts to strip recurring headers/footers.
			- length (int): Minimum number of characters for a line to be included.
			- tables (bool): If True, extract text from detected tables using block
			grouping.
			
		"""
		self.strip_headers = headers
		self.minimum_length = length
		self.extract_tables = tables
		self.lines = [ ]
		self.clean_lines = [ ]
	
	
	def extract_lines( self, path: str, max: Optional[ int ] = None ) -> List[ str ]:
		"""
			
			Extract lines of text from a PDF, optionally limiting to the first N pages.
	
			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Max number of pages to process (None for all pages)
	
			Returns:
			- List[str]: Cleaned list of non-empty lines
			
		"""
		with fitz.open( path ) as doc:
			for i, page in enumerate( doc ):
				if max is not None and i >= max:
					break
				if self.extract_tables:
					page_lines = self._extract_table_blocks( page )
				else:
					text = page.get_text( "text" )
					page_lines = text.splitlines( )
				self.clean_lines = self._filter_lines( page_lines )
				self.lines.extend( self.clean_lines )
		return self.lines
	
	
	def _extract_table_blocks( self, page ) -> List[ str ]:
		"""
			
			Attempt to extract structured blocks such as tables using spatial grouping.
	
			Parameters:
			- page: PyMuPDF page object
	
			Returns:
			- List[str]: Grouped blocks including potential tables
			
		"""
		blocks = page.get_text( "blocks" )
		sorted_blocks = sorted( blocks, key=lambda b: (round( b[ 1 ], 1 ), round( b[ 0 ], 1 )) )
		self.lines = [ b[ 4 ].strip( ) for b in sorted_blocks if b[ 4 ].strip( ) ]
		return self.lines
	
	
	def _filter_lines( self, lines: List[ str ] ) -> List[ str ]:
		"""
		
			Filter and clean lines from a page of text.
	
			Parameters:
			- lines (List[str]): Raw lines of text
	
			Returns:
			- List[str]: Filtered, non-trivial lines
			
		"""
		self.lines = lines
		for line in self.lines:
			line = line.strip( )
			if len( line ) < self.minimum_length:
				continue
			if self.strip_headers and self._is_repeated_header_or_footer( line ):
				continue
			self.clean_lines.append( line )
		return self.clean_lines
	
	
	def _is_repeated_header_or_footer( self, line: str ) -> bool:
		"""
			
			Heuristic to detect common headers/footers (basic implementation).
	
			Parameters:
			- line (str): A line of text
	
			Returns:
			- bool: True if line is likely a header or footer
		
		"""
		keywords = [ "page", "public law", "u.s. government", "united states" ]
		return any( kw in line.lower( ) for kw in keywords )
	
	
	def extract_text( self, path: str, max: Optional[ int ] = None ) -> str:
		"""
		
			Extract the entire text from a PDF into one continuous string.
	
			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum number of pages to process
	
			Returns:
			- str: Full concatenated text
		
		"""
		self.lines = self.extract_lines( path, max=max )
		return "\n".join( self.lines )
	
	
	def extract_dataframes( self, path: str, max: Optional[ int ] = None ) -> \
	List[ pd.DataFrame ]:
		"""
			
			Extract tables from the PDF and return them as a list of DataFrames.
	
			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum number of pages to process
	
			Returns:
			- List[pd.DataFrame]: List of DataFrames representing detected tables
			
		"""
		tables = [ ]
		with fitz.open( path ) as doc:
			for i, page in enumerate( doc ):
				if max is not None and i >= max:
					break
				table_blocks = page.find_tables( )
				for tb in table_blocks.tables:
					df = pd.DataFrame( tb.extract( ) )
					tables.append( df )
		return tables
	
	
	def export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None:
		"""
			
			Export a list of DataFrames (tables) to individual CSV files.
	
			Parameters:
			- tables (List[pd.DataFrame]): List of tables to export
			- filename (str): Prefix for output filenames (e.g., 'output_table')
		
		"""
		for i, df in enumerate( tables ):
			df.to_csv( f"{filename}_{i + 1}.csv", index=False )
	
	
	def export_text( self, lines: List[ str ], path: str ) -> None:
		"""
			
			Export extracted lines of text to a plain text file.
	
			Parameters:
			- lines (List[str]): List of text lines
			- path (str): Path to output text file
		
		"""
		with open( path, "w", encoding="utf-8" ) as f:
			for line in lines:
				f.write( line + "\n" )
	
	
	def export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None:
		"""
			
			Export all extracted tables into a single Excel workbook with one sheet per table.
	
			Parameters:
			- tables (List[pd.DataFrame]): List of tables to export
			- path (str): Path to the output Excel file
		
		"""
		with pd.ExcelWriter( path, engine="xlsxwriter" ) as writer:
			for i, df in enumerate( tables ):
				sheet_name = f"Table_{i + 1}"
				df.to_excel( writer, sheet_name=sheet_name, index=False )
			writer.save( )
