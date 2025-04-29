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
from nltk import pos_tag, FreqDist, ConditionalFreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter, defaultdict
from gensim.models import Word2Vec
from collections import defaultdict
from typing import Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer


class Text:
	'''
	
		Class providing document preprocessing functionality
		
	    Methods:
	    --------
	    collapse_whitespace( self, text: str ) -> str
	    remove_punctuation( self, text: str ) -> str:
		remove_special( self, text: str ) -> str:
		remove_html( self, text: str ) -> str
		remove_errors( self, text: str ) -> str
		correct_errors( self, text: str ) -> str:
		remove_markdown( self, text: str ) -> str
	    normalize( text: str ) -> str
	    tokenize_sentences( text: str ) -> str
	    tokenize_words( text: str ) -> get_list
	    load_file( url: str ) -> li
	    lemmatize( tokens: get_list ) -> str
	    bag_of_words( tokens: get_list ) -> dict
	    train_word2vec( sentences: get_list, vector_size=100, window=5, min_count=1 ) -> Word2Vec
	    compute_tfidf( lines: get_list, max_features=1000, prep=True ) -> tuple
	    
	'''
	
	
	def __init__( self ):
		'''
			Constructor for 'Text' objects
		'''
		self.file_path = None
		self.raw_input = None
		self.normalized = None
		self.lemmatized = None
		self.tokenized = None
		self.corrected = None
		self.cleaned_text = None
		self.words = List[ str ]
		self.tokens = List[ str ]
		self.lines = List[ str ]
		self.pages = List[ str ]
		self.paragraphs = List[ str ]
		self.chunks = List[ str ]
		self.chunk_size = 0
		self.vocabulary = List[ str ]
		self.cleaned_lines = List[ str ]
		self.cleaned_pages = List[ str ]
		self.removed = List[ str ]
		self.raw_pages = List[ str ]
		self.frequency_distribution = { }
		self.conditional_distribution = { }
		self.lowercase = None
		self.translator = None
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.tokenizer = None
		self.vectorizer = None
	
	
	def __dir__( self ):
		'''
		
			Purpose:
			Returns a list of strings
			representing class members.
			
		'''
		return [ 'path', 'raw_input', 'raw_pages', 'normalized', 'lemmatized',
		         'tokenized', 'corrected', 'cleaned_text', 'words', 'paragraphs',
		         'tokens', 'lines', 'pages', 'chunks', 'chunk_size', 'cleaned_pages',
		         'stop_words', 'cleaned_lines', 'removed', 'lowercase',
		         'translator', 'lemmatizer', 'stemmer', 'tokenizer', 'vectorizer',
		         'load_text', 'split_lines', 'split_pages', 'collapse_whitespace',
		         'remove_punctuation', 'remove_special', 'remove_html', 'remove_errors',
		         'remove_markdown', 'remove_stopwords', 'remove_headers',
		         'normalize_text', 'lemmatize', 'tokenize_text', 'tokenize_words',
		         'tokenize_sentences', 'chunk_text', 'chunk_tokens',
		         'bag_of_words', 'train_word2vec', 'compute_tfidf' ]
	
	
	def load_text( self, path: str ) -> str:
		try:
			if path is None:
				raise Exception( 'The input argument "path" is required' )
			else:
				self.file_path = path
				self.raw_input = Path( self.file_path ).read_text( encoding='utf-8' )
				return self.raw_input
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'load_text( self, path: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def split_lines( self, text: str ) -> List[ str ]:
		"""
		
			Splits the input text into lines

			Parameters:
			-----------
			text : str

			Returns:
			--------
			list[ str ]
			
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required' )
			else:
				self.raw_input = text
				with open( self.raw_input, 'r', encoding='utf-8' ) as f:
					self.lines = f.readlines( ).strip( ).splitlines( )
					return self.lines
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'split_lines( self, text: str ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def split_pages( self, path: str, delimit: str='\f' ) -> List[ str ]:
		"""

			Reads text from a file, splits it into lines,
			and groups them into text.

			Args:
				path (str): Path to the text file.
				delimiter (str): Page separator string
				(default is '\f' for form feed).

			Returns:
				list[ str ]  where each element
				is the text.

		"""
		try:
			if path is None:
				raise Exception( 'The input argument "path" is required' )
			else:
				self.file_path = path
				with open( self.file_path, 'r', encoding='utf-8' ) as _file:
					_content = _file.read( )
				self.raw_pages = _content.split( delimit )
				
				for _page in self.raw_pages:
					self.lines = _page.strip( ).splitlines( )
					self.cleaned_text = '\n'.join(
						[ line.strip( ) for line in lines if line.strip( ) ] )
					self.pages.append( self.cleaned_text )
				
				return self.pages
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'split_pages( self, path: str, delimit: str="\f" ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def collapse_whitespace( self, text: str ) -> str:
		"""
		
			Removes extra spaces and
			blank lines from the input text.

			Parameters:
			-----------
			text : str

			Returns:
			--------
			
				A cleaned_lines text string with:
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
				self.cleaned_lines = [ line.strip( ) for line in self.words.splitlines( ) ]
				self.lines = [ line for line in self.cleaned_lines if line ]
				self.cleaned_text = '\n'.join( self.lines )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'collapse_whitespace( self, text: str ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_punctuation( self, text: str ) -> str:
		"""

			Removes all punctuation characters
			 from the input text string.

			Parameters:
			-----------
			pages : str
				The input text string to be cleaned_lines.

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
				self.cleaned_text = self.raw_input.translate( self.translator )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_punctuation( self, text: str ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_special( self, text: str, keep_spaces: bool=True ) -> str:
		"""

			Removes special characters
			from the input text string.

			This function:
			  - Retains only alphanumeric characters and whitespace
			  - Removes symbols like @, #, $, %, &, etc.
			  - Preserves letters, numbers, and spaces

			Parameters:
			-----------
			pages : str
				The raw input text string potentially
				containing special characters.

			Returns:
			--------
			str
				A cleaned_lines string containing
				only letters, numbers, and spaces.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			elif keep_spaces:
				self.raw_input = text
				self.cleaned_text = re.sub( r'[^a-zA-Z0-9\s]', '', self.raw_input )
				return self.cleaned_text
			else:
				self.raw_input = text
				self.cleaned_text = re.sub( r'[^a-zA-Z0-9]', '', self.raw_input )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_special( self, text: str, keep_spaces: bool = True ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )


	def remove_html( self, text: str ) -> str:
		"""
	
			Removes HTML tags
			from the input text string.
	
			This function:
			  - Parses the text as HTML
			  - Extracts and returns only the visible content without tags
	
			Parameters:
			-----------
			pages : str
				The input text containing HTML tags.
	
			Returns:
			--------
			str
				A cleaned_lines string with all HTML tags removed.
	
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_html = text
				self.cleaned_html = BeautifulSoup( self.raw_html, "raw_html.parser" )
				self.cleaned_text = self.cleaned_html.get_text( separator=' ', strip=True )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_html( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_errors( self, text: str ) -> str:
		"""
	
			Removes misspelled or non-English
			words from the input text.
	
			This function:
			  - Converts text to lowercase
			  - Tokenizes the text into words
			  - Filters out words not recognized as valid English using TextBlob
			  - Returns a string with only correctly spelled words
	
			Parameters:
			-----------
			pages : str
				The input pages to clean.
	
			Returns:
			--------
			str
				A cleaned_lines string containing only valid English words.
	
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ w for w in self.tokens if Word( w ).spellcheck( )[ 0 ][ 1 ] > 0.9 ]
				self.cleaned_text = ' '.join( self.words )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_errors( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def correct_errors( self, text: str ) -> str:
		"""
	
			Corrects misspelled words
			in the input text string.
	
			This function:
			  - Converts text to lowercase
			  - Tokenizes the text into words
			  - Applies spelling correction using TextBlob
			  - Reconstructs and returns the corrected text
	
			Parameters:
			-----------
			pages : str
				The input pages string with potential spelling mistakes.
	
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
	
			
	
			This function:
			  - Removes HTML from pages.
	
			Parameters:
			-----------
			pages : str
				The formatted input pages.
	
			Returns:
			--------
			str
				A cleaned_lines version of the pages with formatting removed.
	
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.cleaned_text = (BeautifulSoup( self.raw_input, "raw_html.parser" )
				                .get_text( separator=' ', strip=True ) )
				return self.cleaned_text
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
			pages : str
				The formatted input pages.
	
			Returns:
			--------
			str
				A cleaned_lines version of the pages with formatting removed.
	
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.cleaned_text = re.sub( r'\[.*?\]\(.*?\)', '', self.raw_input )
				self.corrected = re.sub( r'[`_*#~>-]', '', self.cleaned_text )
				self.cleaned_text = re.sub( r'!\[.*?\]\(.*?\)', '', self.corrected )
				return self.cleaned_text
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_markdown( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_stopwords( self, text: str ) -> str:
		"""
	
			Removes English stopwords
			from the input pages string.
	
			This function:
			  - Tokenizes the input pages
			  - Removes common stopwords (e.g., "the", "is", "and", etc.)
			  - Returns the pages with only meaningful words
	
			Parameters:
			-----------
			pages : str
				The input pages string.
	
			Returns:
			--------
			str
				A cleaned_lines version of the input pages without stopwords.
	
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
	
	
	def remove_headers( self, pages: List[ str ], min: int=3 ) -> List[ str ]:
		"""
			
			Removes repetitive headers and footers
			across a list of pages by frequency analysis.
		
			Args:
				pages (list of str): A list where each
				element is the full text of one page.
				min (int): Minimum number of times
				a line must appear at the top/bottom to
				be considered a header/footer.
		
			Returns:
				list of str: List of cleaned_lines page
				texts without detected headers/footers.
			
		"""
		try:
			if pages is None:
				raise Exception( 'The argument "pages" is required.' )
			else:
				_headers = defaultdict( int )
				_footers = defaultdict( int )
				self.pages = pages
				
				# First pass: collect frequency of top/bottom lines
				for _page in self.pages:
					self.lines = _page.strip( ).splitlines( )
					if not self.lines:
						_headers[ self.lines[ 0 ].strip( ) ] += 1
						_footers[ self.lines[ -1 ].strip( ) ] += 1
				
				# Identify candidates for removal
				_head = { line for line, count in _headers.items( ) if
				          count >= min }
				_foot = { line for line, count in _footers.items( ) if
				          count >= min }
				
				# Second pass: clean pages
				for _page in self.pages:
					self.lines = _page.strip( ).splitlines( )
					if not self.lines:
						self.cleaned_pages.append( _page )
						continue
					
					# Remove header
					if self.lines[ 0 ].strip( ) in _head:
						self.lines = self_node.lines[ 1: ]
					
					# Remove footer
					if self.lines and self.lines[ -1 ].strip( ) in _foot:
						self.lines = self_node.lines[ :-1 ]
					
					self.cleaned_pages.append( "\n".join( lines ) )
				
				return self.cleaned_pages
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_headers( self, pages: List[ str ], min: int=3 ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def normalize_text( self, text: str ) -> str:
		"""
	
			Performs normalization on the input pages string.
	
			This function:
			  - Converts pages to lowercase
	
			Parameters:
			-----------
			pages : str
				The input pages string to be lemmatized.
	
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
				self.normalized = self.raw_input.lower( ).translate( str.maketrans( '', '', string.punctuation ) )
				return self.normalized
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'normalize_text( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
		
		
		def get_wordnet_pos( tag: str ) -> Any | None:
			if tag is None:
				raise Exception( 'The argument "tag" is required.' )
			else:
				try:
					if tag.startswith( 'J' ):
						return wordnet.ADJ
					elif tag.startswith( 'V' ):
						return wordnet.VERB
					elif tag.startswith( 'N' ):
						return wordnet.NOUN
					elif tag.startswith( 'R' ):
						return wordnet.ADV
					else:
						return wordnet.NOUN
				except Exception as e:
					_exc = Error( e )
					_exc.module = 'Tiggr'
					_exc.cause = 'Text'
					_exc.method = 'normalize_text( self, text: str ) -> str'
					_err = ErrorDialog( _exc )
					_err.show( )
	
	def lemmatize_tokens( self, tokens: List[ str ]) -> List[ str ]:
		"""
	
			Performs lemmatization on the input List[ str ] into a string
			of word-tokens.
	
			This function:
			  - Converts pages to lowercase
			  - Tokenizes the lowercased pages into words
			  - Lemmatizes each token using WordNetLemmatizer
			  - Reconstructs the lemmatized tokens into a single string
	
			Parameters:
			-----------
			pages : str
				The input pages string to be lemmatized.
	
			Returns:
			--------
			str
				A string with all words lemmatized.
	
		"""
		
		try:
			if tokens is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.tokens = tokens
				pos_tags = pos_tag( self.tokens )
				self.lemmatized = [ self.lemmatizer.lemmatize( word, get_wordnet_pos( tag ) ) for
				               word, tag in pos_tags ]
				return self.lemmatized
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize_words( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_text( self, text: str ) -> List[ str ]:
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
				self.tokens.clear( )
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
			_exc.method = 'tokenize_text( self, text: str ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_words( self, text: str ) -> List[ str ]:
		"""
		
			Tokenize a sentence or
			paragraph into word tokens.
	
			Args:
				text (str): Input pages.
	
			Returns:
				list: List of word tokens.
				
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" was None' )
			else:
				self.raw_input = text
				self.tokens = word_tokenize( self.raw_input )
				return self.tokens
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize_words( self, text: str ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_sentences( self, text: str ) -> List[ str ]:
		"""
		
			Tokenize a paragraph or
			document into a list[ str ] of sentence strings.
	
			Args:
				text (str): Input pages.
	
			Returns:
				list: List of sentence strings.
				
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.tokens = sent_tokenize( self.raw_input )
				return self.tokens
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize_sentences( self, text: str ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk_text( self, text: str, max: int=800 ) -> List[ str ]:
		'''

			Simple chunking by words
			 assuming ~1.3 words per token

			Parameters:
			-----------
			pages : str
				The input pages string to be chunked

			Returns:
			--------
			list[ str ]
				A list with all words chunked.

		'''
		try:
			if (text is None):
				raise Exception( 'The input argument "text" is required.' )
			else:
				self.raw_input = text
				self.lines = self.raw_input.split( )
				self.chunk_size = int( max * 1.3 )
				self.chunks = [ ' '.join( words[ i:i + chunk_size ] ) for i in
				                range( 0, len( words ), chunk_size ) ]
				return self.chunks
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'chunk_text( self, text: str, max: int = 512 ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )

	
	def chunk_tokens( self, tokens: List[ str ], max: int=800, over: int=50 ) -> List[ str ]:
		"""
			Purpose:
				Split a get_list of tokens into overlapping chunks based on token limits.
	
			Args:
				tokens (list): Tokenized input documents.
				max (int): Max token size per chunk_tokens.
				over (int): Overlapping token count between chunks.
	
			Returns:
				list: A get_list of token chunks.
		"""
		try:
			if tokens is None:
				raise Exception( 'Input parameter "tokens" is required.' )
			else:
				_start = 0
				while _start < len( tokens ):
					_end = _start + max
					self.tokens = tokens[ start:end ]
					self.chunks.append( self.tokens )
					_start += max - over
				
				return self.chunks
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = (
				'chunk_tokens( self, tokens: list[ str ], max: int=800, over: int=50 ) -> list[ '
				'str ]')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	def split_paragraphs( self, path: str ) -> List[ str ]:
		"""
		
			Reads text from a file and
			splits it into paragraphs. A paragraph is defined as a block
			of text separated by one or more empty lines.
	
			Args:
				path (str): Path to the text file.
	
			Returns:
				list of str: List of paragraph strings.
				
		"""
		try:
			if path is None:
				raise Exception( 'The input argument "path" is required.' )
			else:
				self.file_path = path
				with open( self.file_path, 'r', encoding='utf-8' ) as _file:
					self.raw_input = _file.read( )
			
					# Normalize line breaks and split on multiple newlines
					self.paragraphs = [ para.strip( ) for para in self.raw_input.split( '\n\n' ) if para.strip( ) ]
					return paragraphs
		except UnicodeDecodeError:
			with open( self.file_path, 'r', encoding='latin1' ) as _file:
				self.raw_input = _file.read( )
				self.paragraphs = [ para.strip( ) for para in self.raw_input.split( '\n\n' ) if para.strip( ) ]
				return paragraphs
	
	
	def compute_frequency_distribution( self, lines: List[ str ], process: bool=True ) -> FreqDist:
		"""
		
			Creates a word frequency freq_dist
			from a list of documents.
	
			Args:
				documents (list): List of raw or preprocessed text documents.
				process (bool): If True, applies normalization,
				tokenization, stopword removal, and lemmatization.
	
			Returns:
				dict: Dictionary of words and their corresponding frequencies.
				
		"""
		try:
			if lines is None:
				raise Exception( 'The input argument "lines" is required.' )
			else:
				self.lines = lines
				for _line in self.lines:
					if process:
						self.normalized = self.normalize_text( _line )
						self.words = self.tokenize_words( self.normalized )
						self.tokens = self.lemmatize_tokens( self.words )
					else:
						self.words = self.tokenize_words( _line )
						self.tokens.append( self.words )
				self.frequency_distribution = dict( Counter( self.tokens ) )
				return self.frequency_distribution
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'compute_frequency_distribution( self, documents: list, process: bool=True) -> FreqDist'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def compute_conditional_distribution( self, lines: List[ str ], condition=None, process: bool=True ) -> ConditionalFreqDist:
		"""

			Computes a Conditional Frequency Distribution (CFD)
			 over a collection of documents.

			Args:
				documents (list):
				A list of text sections (pages, paragraphs, etc.).

				condition (function):
				A function to determine the condition/grouping. If None, uses document index.

				process (bool):
				If True, applies normalization, tokenization,
				stopword removal, and lemmatization.

			Returns:
				ConditionalFreqDist:
				An NLTK ConditionalFreqDist object mapping conditions to word frequencies.

		"""
		try:
			if lines is None:
				raise Exception( 'The input argument "lines" is required.' )
			else:
				self.lines = lines
				self.conditional_distribution = ConditionalFreqDist( )
				
				for idx, _line in enumerate( self.lines ):
					condition = condition( _line ) if condition else f'Doc_{idx}'
					
					if process:
						self.normalized = self.normalize_text( _line )
						self.words = self.tokenize_words( self.normalized )
						self.tokens = self.lemmatize_tokens( self.words )
					else:
						self.tokens = self.tokenize_words( _line )
					
					for _token in self.tokens:
						self.conditional_distribution[ condition ][ _token ] += 1
				
				return self.conditional_distribution
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'compute_conditional_distribution( self, lines: List[ str ], condition=None, process: bool=True ) -> ConditionalFreqDist'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_vocabulary( self, freq_dist: dict, min: int=1 ) -> List[ str ]:
		"""
		
			Builds a vocabulary list from a frequency
			distribution by applying a minimum frequency threshold.
	
			Args:
				freq_dist (dict):
				A dictionary mapping words to their frequencies.
				min (int): Minimum number
				of occurrences required for a word to be included.
	
			Returns:
				list: Sorted list of unique vocabulary words.
				
		"""
		try:
			if freq_dist is None:
				raise Exception( 'The input argument "freq_dist" is required.' )
			else:
				self.frequency_distribution = freq_dist
				self.words = [ word for word, freq in self.frequency_distribution.items( ) if freq >= min ]
				self.vocabulary = sorted( self.words )
				return self.vocabulary
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'create_vocabulary( self, freq_dist: dict, min: int=1 ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def bag_of_words( self, tokens: List[ str ]) -> dict:
		"""
		
			Construct a Bag-of-Words (BoW)
			frequency dictionary from tokens.
	
			Args:
				tokens (list): List of tokens from a document.
	
			Returns:
				dict: Word frequency dictionary.
				
		"""
		try:
			if tokens is None:
				raise Exception( 'The input argument "tokens" is required.' )
			else:
				self.tokens = tokens
				return dict( Counter( self.tokens ) )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'bag_of_words( self, tokens: List[ str ] ) -> dict'
			_err = ErrorDialog( _exc )
			_err.show( )

	
	def train_word2vec( self, tokens: List[ str ], size=100, window=5, min=1 ) -> Word2Vec:
		"""
			Purpose:
				Train a Word2Vec embedding small_model from tokenized sentences.
	
			Args:
				sentences (get_list of get_list of str): List of tokenized sentences.
				vector_size (int): Dimensionality of word vec.
				window (int): Max distance between current and predicted word.
				min_count (int): Minimum frequency for inclusion in vocabulary.
	
			Returns:
				Word2Vec: Trained Gensim Word2Vec small_model.
		"""
		try:
			if tokens is None:
				raise Exception( 'The input argument "tokens" is required.' )
			else:
				self.tokens = tokens
				return Word2Vec( sentences=self.tokens, vector_size=size,
					window=window, min_count=min )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = ('train_word2vec( self, tokens: list, '
			               'size=100, window=5, min=1 ) -> Word2Vec')
			_err = ErrorDialog( _exc )
			_err.show( )


	def compute_tfidf( self, lines: List[ str ], max: int=1000, prep: bool=True ) -> Tuple:
		"""
			Purpose:
			Compute TF-IDF matrix with optional full preprocessing pipeline.
	
			Args:
				lines (list): List of raw or preprocessed pages documents.
				max_features (int): Max number of terms to include (vocabulary size).
				prep (bool): If True, normalize, tokenize_text, clean, and lemmatize input.
	
			Returns:
				tuple:
					- tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF feature matrix.
					- feature_names (get_list): Vocabulary terms.
					- vectorizer (TfidfVectorizer): Fitted vectorizer instance.
	
		"""
		try:
			if lines is None:
				raise Exception( 'The input argument "lines" is required.' )
			elif prep:
				self.lines = lines
				for _doc in self.lines:
					self.normalized = self.normalize( _doc )
					self.tokens = self.tokenize_words( self.normalized )
					self.words = [ self.lemmatize( token ) for token in self.tokens ]
					self.cleaned_text = " ".join( self.words )
					self.cleaned_lines.append( cleaned_text )
					
				self.vectorizer = TfidfVectorizer( max_features=max, stop_words='english' )
				_matrix = self.vectorizer.fit_transform( self.cleaned_lines )
				return ( _matrix, self.vectorizer.get_feature_names_out( ).tolist( ),
				        self.vectorizer)
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = ('compute_tfidf( self, lines: list, max: int=1000, prep: bool=True ) -> '
			               'Tuple')
			_err = ErrorDialog( _exc )
			_err.show( )
