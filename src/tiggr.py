'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                tiggr.py
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
from bs4 import BeautifulSoup
from lxml.xsltext import self_node
import spacy
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter
from gensim.models import Word2Vec



class Text:
	'''
	
		Class providing documents preprocessing functionality
		
		@params: tokenize: bool
		
	'''
	def __init__( self ):
		self.raw_input = None
		self.cleaned = [ str ]
		self.removed = [ str ]
		self.normalized = None
		self.translator = None
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.tokenizer = None
		self.lemmatized = None
		self.tokenized = None
		self.corrected = None
		self.words = [ str ]
		self.tokens = [ str ]
		self.lines = [ str ]
		self.chunks = [ ]
		self.raw_html = None
		self.stop_words = [ ]
		self.filtered = [ ]
	
	
	def __dir__( self ):
		'''
			returns a list[ str ] of members
		'''
		return [ 'raw_input', 'cleaned', 'finished',
		         'lowercase', 'normalized', 'unicoded',
		         'translator', 'lemmatizer', 'tokenizer',
		         'stemmer', 'lemmatized', 'lemmatized_tokens',
		         'cleaned_tokens', 'raw_html', 'cleaned_html',
		         'stop_words', 'filtered', 'chunks' ]
	
	
	def remove_whitespace( self, text: str ) -> str:
		"""
			Removes extra spaces and blank lines from the input text.

			Parameters:
			-----------
			text : str
				The raw input text string to be cleaned.

			Returns:
			--------
			str
				A cleaned text string with:
					- Consecutive whitespace reduced to a single space
					- Leading/trailing spaces removed
					- Blank lines removed
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.words = re.sub( r'[ \t]+', ' ', self.raw_input )
				self.lines = [ line.strip( ) for line in self.words.splitlines( ) ]
				self.cleaned = [ line for line in self.lines if line ]
				self.removed = '\n'.join( self.cleaned )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_whitespace( self, text: str ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_punctuation( self, text: str ) -> str:
		"""

			Removes all punctuation characters from the input text string.

			Parameters:
			-----------
			text : str
				The input text string to be cleaned.

			Returns:
			--------
			str
				The text string with all punctuation removed.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.translator = str.maketrans( '', '', string.punctuation )
				self.removed = self.raw_input.translate( self.translator )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_punctuation( self, text: str ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
			
			
	def remove_special( self, text: str ) -> str:
		"""

			Removes special characters from the input text string.

			This function:
			  - Retains only alphanumeric characters and whitespace
			  - Removes symbols like @, #, $, %, &, etc.
			  - Preserves letters, numbers, and spaces

			Parameters:
			-----------
			text : str
				The raw input text string potentially containing special characters.

			Returns:
			--------
			str
				A cleaned string containing only letters, numbers, and spaces.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.removed = re.sub( r'[^A-Za-z0-9\s]', '', self.raw_input )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_special( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_html( self, text: str ) -> str:
		"""

			Removes HTML tags from the input text string.

			This function:
			  - Parses the text as HTML
			  - Extracts and returns only the visible content without tags

			Parameters:
			-----------
			text : str
				The input text containing HTML tags.

			Returns:
			--------
			str
				A cleaned string with all HTML tags removed.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_html = text
				self.cleaned_html = BeautifulSoup( self.raw_html, "raw_html.parser" )
				self.removed = self.cleaned_html.get_text( separator=' ', strip=True )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_html( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_errors( self, text: str ) -> str:
		"""

			Removes misspelled or non-English words from the input text.

			This function:
			  - Converts text to lowercase
			  - Tokenizes the text into words
			  - Filters out words not recognized as valid English using TextBlob
			  - Returns a string with only correctly spelled words

			Parameters:
			-----------
			text : str
				The input text to clean.

			Returns:
			--------
			str
				A cleaned string containing only valid English words.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ w for w in self.tokens if Word( w ).spellcheck( )[ 0 ][ 1 ] > 0.9 ]
				self.removed = ' '.join( self.words )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_errors( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def correct_errors( self, text: str ) -> str:
		"""

			Corrects misspelled words in the input text string.

			This function:
			  - Converts text to lowercase
			  - Tokenizes the text into words
			  - Applies spelling correction using TextBlob
			  - Reconstructs and returns the corrected text

			Parameters:
			-----------
			text : str
				The input text string with potential spelling mistakes.

			Returns:
			--------
			str
				A corrected version of the input string with proper English words.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ str( Word( w ).correct( ) ) for w in self.tokens ]
				self.corrected = ' '.join( self.words )
				return self.corrected
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'correct_errors( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_html( self, text: str ) -> str:
		"""

			Removes HTML  from text.

			This function:
			  - Strips HTML tags

			Parameters:
			-----------
			text : str
				The formatted input text.

			Returns:
			--------
			str
				A cleaned version of the text with formatting removed.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.removed = (BeautifulSoup( self.raw_input, "raw_html.parser" )
				                    .get_text( separator=' ', strip=True ))
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_html( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_markdown( self, text: str ) -> str:
		"""

			Removes Markdown
			
			This function:
			  - Removes Markdown syntax (e.g., *, #, [], etc.)

			Parameters:
			-----------
			text : str
				The formatted input text.

			Returns:
			--------
			str
				A cleaned version of the text with formatting removed.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.cleaned = re.sub( r'\[.*?\]\(.*?\)', '', self.raw_input )
				self.corrected = re.sub( r'[`_*#~>-]', '', self.cleaned )
				self.removed = re.sub( r'!\[.*?\]\(.*?\)', '', self.corrected )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_markdown( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_stopwords( self, text: str ) -> str:
		"""

			Removes English stopwords from the input text string.

			This function:
			  - Tokenizes the input text
			  - Removes common stopwords (e.g., "the", "is", "and", etc.)
			  - Returns the text with only meaningful words

			Parameters:
			-----------
			text : str
				The input text string.

			Returns:
			--------
			str
				A cleaned version of the input text without stopwords.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.stopwords = set( stopwords.words( 'english' ) )
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ w for w in self.tokens if w.isalnum( ) and w not in self.stopwords ]
				self.removed = ' '.join( self.words )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_stopwords( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def normalize( self, text: str ) -> str:
		"""

			Performs normalization on the input text string.

			This function:
			  - Converts text to lowercase

			Parameters:
			-----------
			text : str
				The input text string to be lemmatized.

			Returns:
			--------
			str
				A string with all words lemmatized.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.normalized = self.raw_input.lower( )
				return self.normalized
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'normalize_text( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def lemmatize( self, text: str ) -> str:
		"""

			Performs lemmatization on the input text string.

			This function:
			  - Converts text to lowercase
			  - Tokenizes the text into words
			  - Lemmatizes each token using WordNetLemmatizer
			  - Reconstructs the lemmatized tokens into a single string

			Parameters:
			-----------
			text : str
				The input text string to be lemmatized.

			Returns:
			--------
			str
				A string with all words lemmatized.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ self.lemmatizer.lemmatize( token ) for token in self.tokens ]
				self.lemmatized = ' '.join( self.words )
				return self.lemmatized
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize_words( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize( self, text: str ) -> list[ str ]:
		'''

			Splits the raw input.
			removes non-words and returns tokens
			Args:
				cleaned_line: (str) - clean documents.

			Returns:
				list: Cleaned and normalized documents.

		'''
		try:
			if text is None:
				raise Exception( 'The input argument "text" was None' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ w for w in self.tokens.split( ' ' ) ]
				self.tokens = [ re.sub( r'[^\w"-]', '', word ) for word in self.words if
				                word.strip( ) ]
				return self.tokens
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize( self, text: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk( self, text: str, size: int = 50 ) -> list[ list[ str ] ]:
		"""

			Tokenizes cleaned text and breaks it
			into chunks for downstream embeddings.

			This function:
			  - Converts text to lowercase
			  - Tokenizes text using NLTK's word_tokenize
			  - Breaks tokens into chunks of a specified size
			  - Optionally joins tokens into strings (for transformer models)

			Parameters:
			-----------
			text : str
				The cleaned input text to be tokenized and chunked.

			chunk_size : int, optional (default=50)
				Number of tokens per chunk.

			return_string : bool, optional (default=True)
				If True, returns each chunk as a string; otherwise, returns a list of tokens.

			Returns:
			--------
			list
				A list of token chunks. Each chunk is either a list of tokens or a string.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.chunks = [ self.tokens[ i: i + size ] for i in
				                range( 0, len( self.tokens ), size ) ]
				return self.chunks
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = ('chunk( self, text: str, chunk_size: int=50, '
			               'return_string: bool=True ) -> list')
			_err = ErrorDialog( _exc )
			_err.show( )


class Token:
	'''
	
	'''	
	def __init__( self, model: str = 'text-embedding-ada-002' ):
		"""
		
			Initialize the PreTokenizer with a specific OpenAI model.
	
			Args:
				model (str): The name of the OpenAI model used for tokenization.
			
		"""
		self.model = model
		self.encoding = tiktoken.encoding_for_model( model )
	
	
	def load_file( self, path: str ) -> str:
		"""
		
			Load the content of a document's file.
	
			Args:
				path (str): The path to the documents file.
	
			Returns:
				str: The contents of the file as a string.
				
		"""
		try:
			if path is None:
				raise Exception( 'Input parameter "path" is required.' )
			else:
				return Path( path ).read_text( encoding='utf-8' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'load_file( self, path: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def parse_hierarchy( self, text: str ) -> list:
		"""
		
			Parse the cleaned documents into a structured hierarchy of sections, subsections,
			and paragraphs.
		
			Args:
				text (str): Cleaned legal or structured document documents.
		
			Returns:
				list: A list of dictionaries, each representing a structural unit with section 
				markers
				and documents.
			
		"""
		try:
			if text is None:
				raise Exception( 'Input parameter "text" is required.' )
			else:
				self.section_pattern = re.compile( r'(SEC\.\s*\d+[A-Z]*\.)' )
				self.subsection_pattern = re.compile( r'\([a-z]\)' )
				self.paragraph_pattern = re.compile( r'\(\d+\)' )
				
				self.structured = [ ]
				self.current_section = None
				self.current_subsection = None
				self.current_paragraph = None
				self.buffer = ""
				
				for line in text.split( '\n' ):
					line = line.strip( )
					
					sec_match = self.section_pattern.match( line )
					if sec_match:
						if self.buffer:
							self.structured.append( {
								'section': self.current_section,
								'subsection': self.current_subsection,
								'paragraph': self.current_paragraph,
								'text': self.buffer.strip( )
							} )
							self.buffer = ""
						self.current_section = sec_match.group( 1 )
						self.current_subsection = None
						self.current_paragraph = None
						continue
					
					sub_match = self.subsection_pattern.match( line )
					if sub_match and len( line ) < 30:
						if self.buffer:
							self.structured.append( {
								'section': self.current_section,
								'subsection': self.current_subsection,
								'paragraph': self.current_paragraph,
								'text': self.buffer.strip( )
							} )
							self.buffer = ""
						self.current_subsection = sub_match.group( 0 )
						self.current_paragraph = None
						continue
					
					para_match = self.paragraph_pattern.match( line )
					if para_match:
						if self.buffer:
							self.structured.append( {
								'section': self.current_section,
								'subsection': self.current_subsection,
								'paragraph': self.current_paragraph,
								'documents': self.buffer.strip( )
							} )
							self.buffer = ""
						self.current_paragraph = para_match.group( 0 )
						continue
					
					self.buffer += " " + line
				
				if self.buffer:
					self.structured.append( {
						'section': self.current_section,
						'subsection': self.current_subsection,
						'paragraph': self.current_paragraph,
						'documents': self.buffer.strip( )
					} )
				
				return self.structured
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'parse_hierarchy( self, documents: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize( self, text: str ) -> list:
		"""
			Purpose:
				Tokenize a block of documents using the OpenAI model tokenizer.
		
			Args:
				text (str): Text to tokenize.
		
			Returns:
				list: A list of token IDs.
				
		"""
		try:
			if text is None:
				raise Exception( 'Input parameter "documents" is required.' )
			else:
				return self.encoding.encode( text )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'tokenize( self, documents: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk_tokens( self, tokens: list, max_tokens: int = 800, overlap: int = 50 ) -> list:
		"""
			Purpose:
				Split a list of tokens into overlapping chunks based on token limits.
	
			Args:
				tokens (list): Tokenized input documents.
				max_tokens (int): Max token size per chunk.
				overlap (int): Overlapping token count between chunks.
		
			Returns:
				list: A list of token chunks.
		"""
		try:
			if tokens is None:
				raise Exception( 'Input parameter "tokens" is required.' )
			else:
				chunks = [ ]
				start = 0
				while start < len( tokens ):
					end = start + max_tokens
					chunk = tokens[ start:end ]
					chunks.append( chunk )
					start += max_tokens - overlap
				
				return chunks
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = ('chunk_tokens( self, tokens: list, '
			               'max_tokens: int = 800, overlap: int = 50 ) -> list')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def decode_tokens( self, tokens: list ) -> str:
		"""
			
			Purpose:
				Decode a list of token IDs back to string documents.
		
			Args:
				tokens (list): A list of token IDs.
		
			Returns:
				str: Decoded string.
				
		"""
		try:
			if tokens is None:
				raise Exception( 'Input parameter "tokens" is required.' )
			else:
				return self.encoding.decode( tokens )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'decode_tokens( self, tokens: list ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk_text_for_embedding( self, text: str, max_tokens: int = 800,
	                              overlap: int = 50 ) -> list:
		"""
		
			Chunk documents into strings suitable for embedding under the token limit.
		
			Args:
				text (str): Raw or cleaned input documents.
				max_tokens (int): Max tokens per chunk for embedding model.
				overlap (int): Overlap between consecutive chunks.
		
			Returns:
				list: List of decoded documents chunks.
				
		"""
		try:
			if (text is None):
				_msg = 'Input parameter "documents" is required.'
				raise Exception( _msg )
			else:
				tokens = self.tokenize( text )
				token_chunks = self.chunk_tokens( tokens, max_tokens, overlap )
				return [ self.decode_tokens( chunk ) for chunk in token_chunks ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = ('chunk_text_for_embedding( self, text: str, max_tokens: int = '
			               '800, '
			               'overlap: int = 50 ) -> list')
			_err = ErrorDialog( _exc )
			_err.show( )
