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
import nltk
from lxml.xsltext import self_node
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
import string


class Text:
	'''
		Class providing documents preprocessing functionality
		@params: tokenize: bool
	'''
	def __init__( self, tokenize: bool = False ):
		self.tokenize = tokenize
		self.raw_text = None
		self.cleaned_lines = [ str ]
		self.cleaned_text = None
		self.lower_case = None
		self.normalized_text = None
		self.unicode_text = None
		self.translator = None
		self.lemmatizer = None
		self.tokenizer = None
		self.lemmatized_text = None
		self.lemmatized_tokens = None
		self.tokenized_text = None
		self.tokens = [ str ]
		self.lines = [ ]
		self.all_tokens = [ ]
		self.cleaned_tokens = [ str ]
		self.chunks = [ ]
		self.html = None
		self.cleaned_html = None
		self.stop_words = [ ]
		self.filtered = [ ]
	
	def __dir__(self):
		'''
			returns a list[ str ] of members
		'''
		return [ 'tokenize', 'raw_text', 'cleaned_lines', 'cleaned_text',
		         'lower_case', 'normalized_text', 'unicode_text',
		         'translator', 'lemmatizer', 'tokenizer', 'lemmatized_text',
		         'lemmatized_tokens', 'cleaned_tokens', 'html', 'cleaned_html',
		         'stop_words', 'filtered', 'chunks' ]
	
	def clean_whitespace( self, text: str ) -> str:
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
				self.raw_text = text
				self.raw_text = re.sub( r'[ \t]+', ' ', self.raw_text )
				self.lines = [ line.strip( ) for line in self.raw_text.splitlines( ) ]
				self.cleaned_lines = [ line for line in self.lines if line ]
				self.cleaned_text = '\n'.join( self.cleaned_lines )
				
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'clean_whitespace( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def normalize( self, text: str ) -> str:
		"""

			Normalizes the input text string.

			This function:
			  - Converts text to lowercase
			  - Removes accented characters (e.g., é -> e)
			  - Removes leading/trailing spaces
			  - Collapses multiple whitespace characters into a single space

			Parameters:
			-----------
			text : str
				The raw input text string to be normalized.

			Returns:
			--------
			str
				A normalized, cleaned version of the input string.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_text = text
				self.lower_case = self.raw_text.lower( )
				self.unicode_text = (unicodedata.normalize( 'NFKD', self.lower_case )
				        .encode( 'ascii', 'ignore' )
				        .decode( 'utf-8' ) )
				self.normalized_text = re.sub( r'\s+', ' ', self.unicode_text ).strip( )
				
				return self.normalized_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'normalize( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def trim_whitespace( self, text: str ) -> str:
		"""

			Trims whitespace from the input text string.

			This function:
			  - Removes leading and trailing whitespace
			  - Replaces multiple internal spaces with a single space

			Parameters:
			-----------
			text : str
				The raw input string with potential extra whitespace.

			Returns:
			--------
			str
				The cleaned string with trimmed and normalized whitespace.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_text = text
				self.raw_text = self.raw_text.strip( )
				self.cleaned_text = re.sub( r'\s+', ' ', self.raw_text )
				
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'trim_whitespace( self, text: str ) -> str'
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
				self.raw_text = text
				self.translator = str.maketrans( '', '', string.punctuation )
				self.cleaned_text = self.raw_text.translate( self.translator )
				
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_punctuation( self, text: str ) -> str'
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
				self.raw_text = text
				self.lower_case = self.raw_text.lower( )
				self.lemmatizer = WordNetLemmatizer( )
				self.tokens = word_tokenize( self.lower_case )
				self.lemmatized_tokens = [ self.lemmatizer.lemmatize( token ) for token in self.tokens ]
				self.lemmatized_text = ' '.join( self.lemmatized_tokens )
				
				return self.lemmatized_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'lemmatize( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize( self, text: str ) -> list:
		"""

			Tokenizes the input text string into individual word tokens.

			This function:
			  - Converts text to lowercase
			  - Uses NLTK's word_tokenize to split the text into words and punctuation tokens

			Parameters:
			-----------
			text : str
				The raw input text string to be tokenized.

			Returns:
			--------
			list
				A list of tokens (words and punctuation) extracted from the text.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_text = text
				self.lower_case = self.raw_text.lower( )
				self.tokenized_text = word_tokenize( self.lower_case )
				
				return self.tokenized_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize( self, text: str ) -> list'
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
				self.raw_text = text
				self.cleaned_text = re.sub( r'[^A-Za-z0-9\s]', '', self.raw_text )
				
				return self.cleaned_text
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
				self.raw_text = text
				_soup = BeautifulSoup( self.raw_text, "html.parser" )
				self.cleaned_text = _soup.get_text( separator=' ', strip=True )
				
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_html( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk_tokens( self, text: list, chunk_size: int = 50 ) -> list:
		"""

			Breaks a list of cleaned, tokenized strings
			into chunks of a specified number of tokens.

			This function:
			  - Flattens the input list of tokenized strings (i.e., list of lists)
			  - Groups tokens into chunks of length `chunk_size`
			  - Returns a list of token chunks, each as a list of tokens

			Parameters:
			-----------
			text : list of tokenizd words
				The input list where each element is a list of tokens (words).

			chunk_size : int, optional (default=50)
				Number of tokens per chunk.

			Returns:
			--------
			list
				A list of token chunks. Each chunk is a list of tokens.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_text = text
				self.tokenized_text = [ token for sublist in self.raw_text for token in sublist ]
				self.chunks = [
					self.tokenized_text[ i:i + chunk_size ]
					for i in range( 0, len( all_tokens ), chunk_size )
				]
				
				return self.chunks
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'chunk_tokens( self, text: list, chunk_size: int = 50 ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk_text( self, text: str, chunk_size: int=50, return_string: bool=True ) -> list:
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
				self.raw_text = text
				self.tokens = word_tokenize( self.raw_text.lower( ) )
				self.chunks = [
					self.tokens[ i:i + chunk_size ]
					for i in range( 0, len( self.tokens ), chunk_size )
				]
				
				if return_srting:
					return [ ' '.join( chunk ) for chunk in self.chunks ]
				else:
					return self.chunks
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = ( 'chunk_text( self, text: str, chunk_size: int=50, '
			               'return_string: bool=True ) -> list' )
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
				self.raw_text = text
				self.tokens = word_tokenize( self.raw_text.lower( ) )
				self.cleaned_tokens = \
					[ word for word in self.tokens if Word( word ).spellcheck( )[ 0 ][ 1 ] > 0.9 ]
				
				return ' '.join( self.cleaned_tokens )
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
				self.raw_text = text
				self.tokens = word_tokenize( self.raw_text.lower( ) )
				self.corrected_tokens = [ str( Word( word ).correct( ) ) for word in self.tokens ]
				self.corrected_text = ' '.join( self.corrected_tokens )
				
				return self.corrected_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'correct_errors( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_headers( self, text: str ) -> str:
		"""
		
			Removes common headers and footers from a text document.
	
			This function:
			  - Assumes repeated lines at the top or bottom (like titles, page numbers)
			  - Removes lines that are common across multiple pages (heuristic)
			  - Returns cleaned text with main body content only
	
			Parameters:
			-----------
			text : str
				The input text potentially containing headers/footers.
	
			Returns:
			--------
			str
				The cleaned text with headers and footers removed.
			
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_text = text
				self.lines = self.raw_text.splitlines( )
				self.cleaned_lines = [ line.strip( ) for line in self.lines if line.strip( ) ]
				self.line_counts = Counter( self.cleaned_lines )
				_threshold = max( 1, int( len( self.cleaned_lines ) * 0.01 ) )
				_repeated_lines = { line for line, count in self.line_counts.items( ) if count > _threshold }
				_body_lines = [ line for line in self.lines if line not in _repeated_lines ]
				self.cleaned_text = '\n'.join( _body_lines )
				
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_headers( self, text: str ) -> str:'
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
				self.raw_text = text
				self.html = BeautifulSoup( self.raw_text, "html.parser" ).get_text( separator=' ', strip=True )
				return self.html
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
				self.raw_text = text
				_mark = re.sub( r'\[.*?\]\(.*?\)', '', self.raw_text )  # Markdown links
				_chars = re.sub( r'[`_*#~>-]', '', _mark )  # Markdown chars
				self.cleaned_text = re.sub( r'!\[.*?\]\(.*?\)', '', _chars )  # Markdown images
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_markdown( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_spaces( self, text: str ) -> str:
		"""

			Removes extra spaces from text.

			This function:
			  - Collapses whitespace (newlines, tabs)

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
				self.raw_text = text
				self.cleaned_text = re.sub( r'\s+', ' ', self.raw_text ).strip( )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_spaces( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def collapse_lines( self, text: str ) -> str:
		"""

			Removes extra spaces from text.

			This function:
			  - Collapses whitespace (newlines, tabs)

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
				self.raw_text = text
				self.cleaned_text = re.sub( r'[\r\n\t]+', ' ', self.raw_text )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'collapse_lines( self, text: str ) -> str'
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
				self.raw_text = text
				nltk.download( 'punkt', quiet=True )
				_stopwords = set( stopwords.words( 'english' ) )
				self.tokens = word_tokenize( self.raw_text.lower( ) )
				self.filtered = [ word for word in self.tokens if word.isalnum( ) and word not in _stopwords ]
				self.cleaned_text = ' '.join( self.filtered )
				
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_stopwords( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def convert_file( self, input: str, output: str ):
		'''

			Reads the unprocessed 'input' file and writes a cleaned 'cleaned_lines' file

			Args:
				input (str): path to pre-processed file.
				output (str): path to processed file.
			Returns:
				str: Cleaned and normalized documents.

		'''
		try:
			if not os.path.exists( input ):
				raise FileNotFoundError( f'Input file not found: {input}' )
			
			if output is None:
				raise FileNotFoundError( f'Output file not found: {output}' )
			
			self.lines = [ ]
			with open( input, 'r', encoding='utf-8' ) as infile:
				for line in infile:
					_cleaned = self.clean_line( line )
					if _cleaned:
						self.lines.append( _cleaned )
			
			with open( output, 'w', encoding='utf-8' ) as outfile:
				for _line in self.lines:
					if self.tokenize:
						self.tokens = self.tokenize( _line )
						json_obj = { 'tokens': self.tokens }
					else:
						json_obj = { 'line': self.cleaned_line }
					json_str = json.dumps( json_obj, ensure_ascii=False )
					outfile.write( json_str + '\n' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'convert( self, input: str, cleaned_lines: str )'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def clean_text( self, text: str ) -> str:
		'''
			
			Clean the raw documents extracted from a PDF for preprocessing.
			This includes removing headers, footers, page numbers, hyphenations,
			fixing line breaks, collapsing whitespace, and normalizing section markers.
	
			Args:
				text (str): Raw extracted documents.
	
			Returns:
				str: Cleaned and normalized documents.
		
		'''
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_text = text
				self.normal = self.raw_text.replace( '\r\n', '\n' ).replace( '\r', '\n' )
				self.headers = re.sub( r'PUBLIC.*?\n', '', self.normal, flags=re.IGNORECASE )
				self.footers = re.sub( r'\n\s*\d+\s*\n', '\n', self.headers )
				self.hyphens = re.sub( r'(\w+)-\n(\w+)', r'\1\2', self.footers )
				self.merge = re.sub( r'(?<!\n)\n(?![\n])', ' ', self.hyphens )
				self.whitespace = re.sub( r'\s+', ' ', self.merge )
				
				return self.whitespace.strip( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'clean_text( self, line: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def clean_line( self, line: str ) -> str:
		"""
			
			Clean the raw documents extracted from a PDF for preprocessing.
	
			This includes removing headers, footers, page numbers, hyphenations,
			fixing line breaks, collapsing whitespace, and normalizing section markers.
	
			Args:
				documents (str): Raw extracted documents.
	
			Returns:
				str: Cleaned and normalized documents.
			
		"""
		try:
			if line is None:
				_msg = 'The input argument "line" is None'
				raise Exception( _msg )
			else:
				self.raw_text = line
				self.normalized = self.raw_text.replace( '\r\n', '\n' ).replace( '\r', '\n' )
				self.headers = re.sub( r'PUBLIC.*?\n', '', self.normalized, flags=re.IGNORECASE )
				self.footers = re.sub( r'\n\s*\d+\s*\n', '\n', self.headers )
				self.hyphens = re.sub( r'(\w+)-\n(\w+)', r'\1\2', self.footers )
				self.merge = re.sub( r'(?<!\n)\n(?![\n])', ' ', self.hyphens )
				self.whitespace = re.sub( r'\s+', ' ', self.merge )
				self.normalized = re.sub( r'(SEC\.\s*\d+[A-Z]*\.)', r'\n\n\1', self.whitespace )
				
				return self.normalized.strip( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'clean_line( self, line: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def clean_string( self, text: str, stem='None' ) -> str:
		'''
		
			Clean the raw documents extracted for preprocessing.
			This includes removing headers, footers, page numbers, hyphenations,
			fixing line breaks, collapsing whitespace, and normalizing section markers.
	
			Args:
				text (str): Raw extracted documents.
	
			Returns:
				str: Cleaned and normalized documents.
				
		'''
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required' )
			else:
				self.raw_text = text
				self.final_string = ''
				self.lower = self.raw_text.lower( )
				self.lines = re.sub( r'\n', '', self.lower )
				self.translator = str.maketrans( '', '', string.punctuation )
				self.text = self.lines.translate( self.translator )
				self.split = self.text.split( )
				self.text_filtered = [ re.sub( r'\w*\d\w*', '', w ) for w in self.split ]
				
				if stem == 'Stem':
					self.stemmer = PorterStemmer( )
					self.text_stemmed = [ self.stemmer.stem( y ) for y in self.text_filtered ]
				elif stem == 'Lem':
					self.lemmer = WordNetLemmatizer( )
					self.text_stemmed = [ self.lemmer.lemmatize( y ) for y in self.text_filtered ]
				elif stem == 'Spacy':
					self.text_filtered = nlp( ' '.join( self.text_filtered ) )
					self.text_stemmed = [ y.lemma_ for y in self.text_filtered ]
				else:
					self.text_stemmed = self.text_filtered
				
				self.final_string = ' '.join( self.text_stemmed )
				
				return self.final_string
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'clean_string( self, documents: str, stem="None" ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize( self, text: str ) -> list:
		'''
		
			Clean the raw documents extracted for preprocessing.
			This includes removing headers, footers, page numbers, hyphenations,
			fixing line breaks, collapsing whitespace, and normalizing section markers.
	
			Args:
				cleaned_line: (str) - clean documents.
	
			Returns:
				list: Cleaned and normalized documents.
				
		'''
		try:
			if text is None:
				raise Exception( 'The input argument "text" was None' )
			else:
				self.raw_text = text
				_words = self.raw_text.split( )
				self.tokens = [ re.sub( r'[^\w"-]', '', word ) for word in _words if word.strip( ) ]
				return tokens
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize( self, cleaned_line: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def clean_html( self, html: str ) -> list:
		'''
		Method that cleans html given as input
		:param html: str
		:return: list
		'''
		try:
			if html is None:
				raise Exception( 'The input argument "html" was None' )
			else:
				soup = BeautifulSoup( markup=html, features='html.parser' )
				for data in soup( [ 'style', 'script', 'code', 'a' ] ):
					data.decompose( )
				
				return ' '.join( soup.stripped_strings )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			
			_exc.cause = 'Text'
			_exc.method = 'clean_html( self, html: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )


class Token:
	'''
	
	'''
	def __init__( self, model: str = "documents-embedding-ada-002" ):
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
				list: A list of dictionaries, each representing a structural unit with section markers
				and documents.
			
		"""
		try:
			if text is None:
				raise Exception( 'Input parameter "documents" is required.' )
			else:
				self.section_pattern = re.compile( r"(SEC\.\s*\d+[A-Z]*\.)" )
				self.subsection_pattern = re.compile( r"\([a-z]\)" )
				self.paragraph_pattern = re.compile( r"\(\d+\)" )
				
				self.structured = [ ]
				self.current_section = None
				self.current_subsection = None
				self.current_paragraph = None
				self.buffer = ""
				
				for line in text.split( "\n" ):
					line = line.strip( )
					
					sec_match = self.section_pattern.match( line )
					if sec_match:
						if self.buffer:
							self.structured.append( {
								"section": self.current_section,
								"subsection": self.current_subsection,
								"paragraph": self.current_paragraph,
								"documents": self.buffer.strip( )
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
								"section": self.current_section,
								"subsection": self.current_subsection,
								"paragraph": self.current_paragraph,
								"documents": self.buffer.strip( )
							} )
							self.buffer = ""
						self.current_subsection = sub_match.group( 0 )
						self.current_paragraph = None
						continue
					
					para_match = self.paragraph_pattern.match( line )
					if para_match:
						if self.buffer:
							self.structured.append( {
								"section": self.current_section,
								"subsection": self.current_subsection,
								"paragraph": self.current_paragraph,
								"documents": self.buffer.strip( )
							} )
							self.buffer = ""
						self.current_paragraph = para_match.group( 0 )
						continue
					
					self.buffer += " " + line
				
				if self.buffer:
					self.structured.append( {
						"section": self.current_section,
						"subsection": self.current_subsection,
						"paragraph": self.current_paragraph,
						"documents": self.buffer.strip( )
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
	
	
	def chunk_tokens( self, tokens: list, max_tokens: int=800, overlap: int=50 ) -> list:
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
			if(text is None):
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
			_exc.method = ( 'chunk_text_for_embedding( self, documents: str, max_tokens: int = 800, '
			               'overlap: int = 50 ) -> list' )
			_err = ErrorDialog( _exc )
			_err.show( )
