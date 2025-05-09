'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                tigrr.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="tigrr.py" company="Terry D. Eppler">

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
from collections import defaultdict
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
from lxml.xsltext import self_node
from booggr import Error, ErrorDialog
from pathlib import Path
import nltk
from nltk import pos_tag, FreqDist, ConditionalFreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter, defaultdict
from gensim.models import Word2Vec
import string
import spacy
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from pymupdf import Page, Document
import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import textwrap as tr
from typing import Any, List, Tuple, Optional, Union, Dict

# Ensure punkt tokenizer is available for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Text:
	'''
	
		Purpose:
		Class providing text preprocessing functionality
		
	    Methods:
	    --------
	    load_text( url: str ) -> str
	    split_lines( self, text: str ) -> list
	    split_pages( self, path: str, delimit: str ) -> list
	    collapse_whitespace( self, text: str ) -> str
	    remove_punctuation( self, text: str ) -> str:
		remove_special( self, text: str, keep_spaces: bool ) -> str:
		remove_html( self, text: str ) -> str
		remove_errors( self, text: str ) -> str
		correct_errors( self, text: str ) -> str:
		remove_markdown( self, text: str ) -> str
		remove_stopwords( self, text: str ) -> str
		remove_headers( self, pages, min: int=3 ) -> str
	    normalize_text( text: str ) -> str
	    lemmatize_tokens( tokens: List[ str ] ) -> str
	    tokenize_text( text: str ) -> str
	    tokenize_words( text: str ) -> List[ str ]
	    tokenize_sentences( text: str ) -> str
	    chunk_text( self, text: str, max: int=800 ) -> List[ str ]
	    chunk_tokens( self, text: str, max: int=800, over: int=50 ) -> List[ str ]
	    split_paragraphs( self, path: str ) -> List[ str ]
	    compute_frequency_distribution( self, lines: List[ str ], proc: bool=True ) -> List[ str ]
	    compute_conditional_distribution( self, lines: List[ str ], condition: str=None,
	    proc: bool=True ) -> List[ str ]
	    create_vocabulary( self, frequency, min: int=1 ) -> List[ str ]
	    create_wordbag( tokens: List[ str ] ) -> dict
	    create_word2vec( sentences: List[ str ], vector_size=100, window=5, min_count=1 ) -> Word2Vec
	    create_tfidf( tokens: List[ str ], max_features=1000, prep=True ) -> tuple
	    
	'''
	
	
	def __init__( self ):
		'''
			Constructor for 'Text' objects
		'''
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.words = List[ str ]
		self.tokens = List[ str ]
		self.lines = List[ str ]
		self.pages = List[ str ]
		self.ids = List[ int ]
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
		self.file_path = None
		self.raw_input = None
		self.normalized = None
		self.lemmatized = None
		self.tokenized = None
		self.corrected = None
		self.cleaned_text = None
		self.lowercase = None
		self.translator = None
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
		         'tokens', 'tokens', 'pages', 'chunks', 'chunk_size', 'cleaned_pages',
		         'stop_words', 'cleaned_lines', 'removed', 'lowercase',
		         'translator', 'lemmatizer', 'stemmer', 'tokenizer', 'vectorizer',
		         'load_text', 'split_lines', 'split_pages', 'collapse_whitespace',
		         'remove_punctuation', 'remove_special', 'remove_html', 'remove_errors',
		         'remove_markdown', 'remove_stopwords', 'remove_headers',
		         'normalize_text', 'lemmatize', 'tokenize_text', 'tokenize_words',
		         'tokenize_sentences', 'chunk_text', 'chunk_tokens',
		         'create_wordbag', 'create_word2vec', 'create_tfidf' ]
	
	
	def load_text( self, path: str ) -> str:
		try:
			if path is None:
				raise Exception( 'The text argument "path" is required' )
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
		
			Splits the text
			into tokens

			Parameters:
			-----------
			text : str

			Returns:
			--------
			list[ str ]
			
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required' )
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

			Reads text from a file, splits it into tokens,
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
				raise Exception( 'The text argument "path" is required' )
			else:
				self.file_path = path
				with open( self.file_path, 'r', encoding='utf-8' ) as _file:
					_content = _file.read( )
				self.raw_pages = _content.split( delimit )
				for _page in self.raw_pages:
					self.lines = _page.strip( ).splitlines( )
					self.cleaned_text = '\n'.join( [ line.strip( )
					                                 for line in self.lines if line.strip( ) ] )
					self.cleaned_pages.append( self.cleaned_text )
					_retval = self.cleaned_pages
				return _retval
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
			blank tokens from the text text.

			Parameters:
			-----------
			text : str

			Returns:
			--------
			
				A cleaned_lines text string with:
					- Consecutive whitespace reduced to a single space
					- Leading/trailing spaces removed
					- Blank tokens removed
					
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.words = re.sub( r'[ \t]+', ' ', self.raw_input )
				self.cleaned_lines = [ line.strip( ) for line in self.words.splitlines( ) ]
				self.lines = [ line for line in self.cleaned_lines if line ]
				_retval = '\n'.join( self.lines )
				return _retval
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
			 from the text text string.

			Parameters:
			-----------
			pages : str
				The text text string to be cleaned_lines.

			Returns:
			--------
			str
				The text string with all punctuation removed.

		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.translator = str.maketrans( '', '', string.punctuation )
				_retval = self.raw_input.translate( self.translator )
				return _retval
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
			from the text text string.

			This function:
			  - Retains only alphanumeric characters and whitespace
			  - Removes symbols like @, #, $, %, &, etc.
			  - Preserves letters, numbers, and spaces

			Parameters:
			-----------
			pages : str
				The raw text text string potentially
				containing special characters.

			Returns:
			--------
			str
				A cleaned_lines string containing
				only letters, numbers, and spaces.

		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			elif keep_spaces:
				self.raw_input = text
				_retval = re.sub( r'[^a-zA-Z0-9\s]', '', self.raw_input )
				return _retval
			else:
				self.raw_input = text
				_retval = re.sub( r'[^a-zA-Z0-9]', '', self.raw_input )
				return _retval
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
			from the text text string.
	
			This function:
			  - Parses the text as HTML
			  - Extracts and returns only the visible content without tags
	
			Parameters:
			-----------
			pages : str
				The text text containing HTML tags.
	
			Returns:
			--------
			str
				A cleaned_lines string with all HTML tags removed.
	
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_html = text
				self.cleaned_html = BeautifulSoup( self.raw_html, "raw_html.parser" )
				_retval = self.cleaned_html.get_text( separator=' ', strip=True )
				return _retval
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
			words from the text text.
	
			This function:
			  - Converts text to lowercase
			  - Tokenizes the text into words
			  - Filters out words not recognized as valid English using TextBlob
			  - Returns a string with only correctly spelled words
	
			Parameters:
			-----------
			pages : str
				The text pages to clean.
	
			Returns:
			--------
			str
				A cleaned_lines string containing only valid English words.
	
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ w for w in self.tokens if Word( w ).spellcheck( )[ 0 ][ 1 ] > 0.9 ]
				_retval = ' '.join( self.words )
				return _retval
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
			in the text text string.
	
			This function:
			  - Converts text to lowercase
			  - Tokenizes the text into words
			  - Applies spelling correction using TextBlob
			  - Reconstructs and returns the corrected text
	
			Parameters:
			-----------
			pages : str
				The text pages string with potential spelling mistakes.
	
			Returns:
			--------
			str
				A corrected version of the text string with proper English words.
	
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ str( Word( w ).correct( ) ) for w in self.tokens ]
				_retval = ' '.join( self.words )
				return _retval
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
				The formatted text pages.
	
			Returns:
			--------
			str
				A cleaned_lines version of the pages with formatting removed.
	
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				_retval = (BeautifulSoup( self.raw_input, "raw_html.parser" )
				                     .get_text( separator=' ', strip=True ))
				return _retval
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
				The formatted text pages.
	
			Returns:
			--------
			str
				A cleaned_lines version of the pages with formatting removed.
	
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.cleaned_text = re.sub( r'\[.*?\]\(.*?\)', '', self.raw_input )
				self.corrected = re.sub( r'[`_*#~>-]', '', self.cleaned_text )
				_retval = re.sub( r'!\[.*?\]\(.*?\)', '', self.corrected )
				return _retval
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
			from the text pages string.
	
			This function:
			  - Tokenizes the text pages
			  - Removes common stopwords (e.g., "the", "is", "and", etc.)
			  - Returns the pages with only meaningful words
	
			Parameters:
			-----------
			pages : str
				The text pages string.
	
			Returns:
			--------
			str
				A cleaned_lines version of the text pages without stopwords.
	
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.stopwords = set( stopwords.words( 'english' ) )
				self.lowercase = self.raw_input.lower( )
				self.tokens = word_tokenize( self.lowercase )
				self.words = [ w for w in self.tokens if w.isalnum( ) and w not in self.stopwords ]
				_retval = ' '.join( self.words )
				return _retval
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_stopwords( self, text: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_headers( self, pages: List[ str ], min: int = 3 ) -> List[ str ]:
		"""
			
			Removes repetitive headers and footers
			across a list of pages by frequency analysis.
		
			Args:
				pages (list of str): A list where each
				element is the full text of one page.
				min (int): Minimum num of times
				a line must appear at the top/bottom to
				be considered a header/footer.
		
			Returns:
				list of str: List of cleaned_lines page
				tokens without detected headers/footers.
			
		"""
		try:
			if pages is None:
				raise Exception( 'The argument "pages" is required.' )
			else:
				_headers = defaultdict( int )
				_footers = defaultdict( int )
				self.pages = pages
				
				# First pass: collect frequency of top/bottom tokens
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
					
					self.cleaned_pages.append( "\n".join( self.lines ) )
				_retval = self.cleaned_pages
				return _retval
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_headers( self, pages: List[ str ], min: int=3 ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def normalize_text( self, text: str ) -> str:
		"""
	
			Performs normalization on the text pages string.
	
			This function:
			  - Converts pages to lowercase
	
			Parameters:
			-----------
			pages : str
				The text pages string to be lemmatized.
	
			Returns:
			--------
			str
				A string with all words lemmatized.
	
		"""
		try:
			if text is None:
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.normalized = self.raw_input.lower( ).translate(
					str.maketrans( '', '', string.punctuation ) )
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
	
	
	def lemmatize_tokens( self, tokens: List[ str ] ) -> List[ str ]:
		"""
	
			Performs lemmatization on the text List[ str ] into a string
			of word-tokens.
	
			This function:
			  - Converts pages to lowercase
			  - Tokenizes the lowercased pages into words
			  - Lemmatizes each token using WordNetLemmatizer
			  - Reconstructs the lemmatized tokens into a single string
	
			Parameters:
			-----------
			pages : str
				The text pages string to be lemmatized.
	
			Returns:
			--------
			str
				A string with all words lemmatized.
	
		"""
		
		try:
			if tokens is None:
				raise Exception( 'The text argument "text" is required.' )
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
	
			Splits the raw text.
			removes non-words and returns tokens
			Args:
				cleaned_line: (str) - clean documents.
	
			Returns:
				list: Cleaned and normalized documents.
	
		'''
		try:
			if text is None:
				raise Exception( 'The text argument "text" was None' )
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
				raise Exception( 'The text argument "text" was None' )
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
				raise Exception( 'The text argument "text" is required.' )
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
	
	
	def chunk_text( self, text: str, max: int = 800 ) -> List[ str ]:
		'''

			Simple chunking by words
			 assuming ~1.3 words per token

			Parameters:
			-----------
			text : str
				The text text to be chunked

			Returns:
			--------
			list[ str ]
				A list with all words chunked.

		'''
		try:
			if (text is None):
				raise Exception( 'The text argument "text" is required.' )
			else:
				self.raw_input = text
				self.lines = self.raw_input.split( )
				self.chunk_size = int( max * 1.3 )
				_retval = [ ' '.join( self.words[ i:i + chunk_size ] ) for i in
				                range( 0, len( self.words ), chunk_size ) ]
				return _retval
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'chunk_text( self, text: str, max: int=512 ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk_tokens( self, tokens: List[ str ], max: int = 800, over: int = 50 ) -> List[ str ]:
		"""
		
			Purpose:
				Split a list of strings into
				overlapping chunks based on token limits.
	
			Args:
				tokens (list): Tokenized text documents.
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
				
				_retval = self.chunks
				return _retval
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
		
			Reads  a file and
			splits it into paragraphs. A paragraph is defined as a block
			of text separated by one or more empty tokens.
	
			Args:
				path (str): Path to the text file.
	
			Returns:
				list of str: List of paragraph strings.
				
		"""
		try:
			if path is None:
				raise Exception( 'The text argument "path" is required.' )
			else:
				self.file_path = path
				with open( self.file_path, 'r', encoding='utf-8' ) as _file:
					self.raw_input = _file.read( )
					
					# Normalize line breaks and split on multiple newlines
					self.paragraphs = [ para.strip( ) for para in self.raw_input.split( '\n\n' ) if
					                    para.strip( ) ]
					return paragraphs
		except UnicodeDecodeError:
			with open( self.file_path, 'r', encoding='latin1' ) as _file:
				self.raw_input = _file.read( )
				self.paragraphs = [ para.strip( ) for para in self.raw_input.split( '\n\n' ) if
				                    para.strip( ) ]
				return paragraphs
	
	
	def compute_frequency_distribution( self, lines: List[ str ],
	                                    process: bool = True ) -> FreqDist:
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
				raise Exception( 'The text argument "tokens" is required.' )
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
			_exc.method = ('compute_frequency_distribution( self, documents: list, process: '
			               'bool=True) -> FreqDist')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def compute_conditional_distribution( self, lines: List[ str ], condition=None,
	                                      process: bool = True ) -> ConditionalFreqDist:
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
				raise Exception( 'The text argument "tokens" is required.' )
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
			_exc.method = ('compute_conditional_distribution( self, tokens: List[ str ], '
			               'condition=None, process: bool=True ) -> ConditionalFreqDist')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_vocabulary( self, freq_dist: dict, min: int = 1 ) -> List[ str ]:
		"""
		
			Builds a vocabulary list from a frequency
			distribution by applying a minimum frequency threshold.
	
			Args:
				freq_dist (dict):
				A dictionary mapping words to their frequencies.
				min (int): Minimum num
				of occurrences required for a word to be included.
	
			Returns:
				list: Sorted list of unique vocabulary words.
				
		"""
		try:
			if freq_dist is None:
				raise Exception( 'The text argument "freq_dist" is required.' )
			else:
				self.frequency_distribution = freq_dist
				self.words = [ word for word, freq in self.frequency_distribution.items( ) if
				               freq >= min ]
				self.vocabulary = sorted( self.words )
				return self.vocabulary
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'create_vocabulary( self, freq_dist: dict, min: int=1 ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_wordbag( self, tokens: List[ str ] ) -> dict:
		"""
			
			Purpose:
			Construct a Bag-of-Words (BoW)
			frequency dictionary from a list of strings.
	
			Args:
				tokens (list): List of tokens from a document.
	
			Returns:
				dict: Word frequency dictionary.
				
		"""
		try:
			if tokens is None:
				raise Exception( 'The text argument "tokens" is required.' )
			else:
				self.tokens = tokens
				return dict( Counter( self.tokens ) )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'create_wordbag( self, tokens: List[ str ] ) -> dict'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_word2vec( self, tokens: List[ str ], size=100, window=5, min=1 ) -> Word2Vec:
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
				raise Exception( 'The text argument "tokens" is required.' )
			else:
				self.tokens = tokens
				return Word2Vec( sentences=self.tokens, vector_size=size,
					window=window, min_count=min )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = ('create_word2vec( self, tokens: list, '
			               'size=100, window=5, min=1 ) -> Word2Vec')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_tfidf( self, lines: List[ str ], max: int=1000, prep: bool=True ) -> Tuple:
		"""
		
			Purpose:
			
			Compute TF-IDF matrix with optional full preprocessing pipeline.
	
			Args:
				lines (list): List of raw or preprocessed pages documents.
				max (int): Max num of terms to include (vocabulary size).
				prep (bool): If True, normalize, tokenize_text, clean, and lemmatize text.
	
			Returns:
				tuple:
					- tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF feature matrix.
					- feature_names (list): Vocabulary terms.
					- vectorizer (TfidfVectorizer): Fitted vectorizer instance.
	
		"""
		try:
			if lines is None:
				raise Exception( 'The text argument "tokens" is required.' )
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
				return (_matrix, self.vectorizer.get_feature_names_out( ).tolist( ),
				        self.vectorizer)
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = ('create_tfidf( self, tokens: list, max: int=1000, prep: bool=True ) '
			               '-> '
			               'Tuple')
			_err = ErrorDialog( _exc )
			_err.show( )


class PDF( ):
	"""

		PDF
		----------------
		A utility class for extracting clean pages from PDF files into a list of strings.
		Handles nuances such as layout artifacts, page separation, optional filtering,
		and includes df detection capabilities.
		
		
	    Methods:
	    --------
	    extract_lines( self, path, max: int=None) -> List[ str ]
	    extract_text( self, path, max: int=None) -> str
	    export_csv( self, tables: List[ pd.DataFrame ], filename: str=None ) -> None
	    export_text( self, lines: List[ str ], path: str=None ) -> None
	    export_excel( self, tables: List[ pd.DataFrame ], path: str=None ) -> None

	"""
	
	
	def __init__( self, headers: bool=False, min: int=10, tables: bool=True ):
		"""

			Purpose:
			Initialize the PDF pages extractor with configurable settings.

			Parameters:
			- headers (bool): If True, attempts to strip recurring headers/footers.
			- min (int): Minimum num of characters for a line to be included.
			- tables (bool): If True, extract pages from detected tables using block
			grouping.

		"""
		self.strip_headers = headers
		self.minimum_length = min
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
		         'path', 'page', 'pages', 'tokens', 'clean_lines', 'extracted_lines',
		         'extracted_tables', 'extracted_pages', 'extract_lines',
		         'extract_text', 'extract_tables', 'export_csv',
		         'export_text', 'export_excel' ]
	
	
	def extract_lines( self, path: str, max: Optional[ int ]=None ) -> List[ str ]:
		"""

			Extract tokens of pages from a PDF,
			optionally limiting to the first N pages.

			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Max num of pages to process (None for all pages)

			Returns:
			- List[str]: Cleaned list of non-empty tokens

		"""
		try:
			if path is None:
				raise Exception( 'Input "path" must be specified' )
			else:
				self.file_path = path
				with fitz.open( self.file_path ) as doc:
					for i, page in enumerate( doc ):
						if max is not None and i >= max:
							break
						if self.extract_tables:
							self.extracted_lines = self._extract_tables( page )
						else:
							_text = page.get_text( 'pages' )
							self.lines = _text.splitlines( )
						self.clean_lines = self._filter_lines( self.lines )
						self.extracted_lines.extend( self.clean_lines )
				return self.extracted_lines
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = ('extract_lines( self, path: str, max: Optional[ int ]=None ) -> '
			               'List[ str ]')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _extract_tables( self, page: Page ) -> List[ str ]:
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
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = '_extract_tables( self, page ) -> List[ str ]:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _filter_lines( self, lines: List[ str ] ) -> List[ str ]:
		"""

			Filter and clean tokens
			 from a page of pages.

			Parameters:
			- tokens (List[str]): Raw tokens of pages

			Returns:
			- List[str]: Filtered, non-trivial tokens

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
					if self.strip_headers and self._has_repeating_header( _line ):
						continue
					self.clean_lines.append( _line )
				return self.clean_lines
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = '_filter_lines( self, tokens: List[ str ] ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _has_repeating_header( self, line: str ) -> bool:
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
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = '_has_repeating_header( self, line: str ) -> bool'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def extract_text( self, path: str, max: Optional[ int ]=None ) -> str:
		"""

			Extract the entire pages from a
			PDF into one continuous string.

			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum num of pages to process

			Returns:
			- str: Full concatenated pages

		"""
		try:
			if path is None:
				raise Exception( 'Input "path" must be specified' )
			else:
				if max is not None and max > 0:
					self.file_path = path
					self.lines = self.extract_lines( self.file_path, max=max )
					return '\n'.join( self.lines )
				elif max is None or max <= 0:
					self.file_path = path
					self.lines = self.extract_lines( self.file_path )
					return '\n'.join( self.lines )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = 'extract_text( self, path: str, max: Optional[ int ]=None ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def extract_tables( self, path: str, max: Optional[ int ]=None ) -> List[ pd.DataFrame ]:
		"""

			Extract tables from the PDF
			and return them as a list of DataFrames.

			Parameters:
			- path (str): Path to the PDF file
			- max (Optional[int]): Maximum num of pages to process

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
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
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
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = 'export_csv( self, tables: List[ pd.DataFrame ], filename: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def export_text( self, lines: List[ str ], path: str ) -> None:
		"""

			Export extracted tokens of
			pages to a plain pages file.

			Parameters:
			- tokens (List[str]): List of pages tokens
			- path (str): Path to output pages file

		"""
		try:
			if lines is None:
				raise Exception( 'Input "tokens" must be provided.' )
			elif path is None:
				raise Exception( 'Input "path" must be provided.' )
			else:
				self.file_path = path
				self.lines = lines
				with open( self.file_path, 'w', encoding='utf-8' ) as f:
					for line in self.lines:
						f.write( line + '\n' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = 'export_text( self, tokens: List[ str ], path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None:
		"""

			Export all extracted tables into a single
			Excel workbook with one sheet per df.

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
			_exc.module = 'tiggr'
			_exc.cause = 'PDF'
			_exc.method = 'export_excel( self, tables: List[ pd.DataFrame ], path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )


class Token( ):
	'''
	
	
		Purpose:
		________
		
		Wrapper for Hugging Face tokenizers using the `transformers` library.
	    Includes sentence-level segmentation, tokenization, encoding, and decoding.
	    
	    Methods:
	    _______
	    
	    tokenize_sentences( self, text: str ) -> List[str]
	    tokenize( self, text: str ) -> List[str]
	    encode( self, text: str ) -> List[str]
	    batch_encode( self, text: str ) -> List[str]
	    decode( self, ids: List[ str ], skip: bool=True ) -> List[str]
	    convert_tokens( self, tokens: List[str] ) -> List[str]
	    convert_ids( self, ids: List[str] ) -> List[str]
	    get_vocab( self ) -> List[str]
	    save_tokenizer( self, path: str ) -> None
	    load_tokenizer( self, path: str ) -> None
	
	'''
	
	
	def __init__( self ):
		'''
		
		
			Purpose:
	        Initializes the tokenizer wrapper using a pre-trained small_model from Hugging Face.
	
	        Args:
	            model_name (str): The name of the pre-trained small_model (e.g., "bert-base-uncased").
        '''
		self.model_name = 'google-bert/bert-base-uncased'
		self.tokenizer = AutoTokenizer.from_pretrained( self.model_name, trust_remote_code=True )
		self.raw_input = None
	
	
	def get_vocab( self ) -> Dict[ str, int ]:
		"""
			
			Retrieves the
			tokenizer's vocabulary.
	
			Returns:
				Dict[str, int]: Mapping of token string to token ID.
			
		"""
		try:
			return self.tokenizer.get_vocab( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'get_vocab( self ) -> Dict[ str, int ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def load_tokenizer( self, path: str ) -> None:
		"""
		
			Loads a tokenizer from
			 a specified directory path.
	
			Args:
				path (str): Path to the tokenizer config and vocab files.
			
		"""
		try:
			if path is None:
				raise Exception( 'Input "path" must be provided.' )
			else:
				self.tokenizer = AutoTokenizer.from_pretrained( path )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'load_tokenizer( self, path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def save_tokenizer( self, path: str ) -> None:
		"""
			
			Saves the tokenizer
			to a directory.
	
			Args:
				path (str): Target path to save tokenizer config and vocab.
			
		"""
		try:
			if path is None:
				raise Exception( 'The target "path" must be provided.' )
			else:
				self.tokenizer.save_pretrained( path )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'save_tokenizer( self, path: str ) -> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_sentences( self, text: str ) -> List[ str ]:
		"""
		
			Segments the text
			text into individual sentences.
	
			Args:
				text (str): The text document as a string.
	
			Returns:
				List[str]: List of sentence strings.
			
		"""
		try:
			if text is None:
				raise Exception( 'Input "text" must be provided.' )
			else:
				self.raw_input = text
				return nltk.sent_tokenize( self.raw_input )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'tokenize_sentences( self, text: str ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize( self, text: str ) -> List[ str ]:
		"""
		
			Tokenizes text into subword
			tokens using the tokenizer's vocabulary.
	
			Args:
				text (str): The raw text text.
	
			Returns:
				List[str]: Tokenized list of word-pieces/subwords.
			
		"""
		try:
			if text is None:
				raise Exception( 'Input "text" must be provided.' )
			else:
				return self.tokenizer.tokenize( text )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'tokenize( self, text: str ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def encode( self, text: str, max: int=512, trunc: bool=True,
	            padd: Union[ bool, str ]=False, tensors: str=None ) -> Dict[
		str, Union[ List[ int ], any ] ]:
		"""
		
			Encodes a single string of text
			into small_model-ready text IDs and attention masks.
	
			Args:
				text (str): Input string.
				max (int): Max length of token sequence.
				trunc (bool): If True, trunc sequences over max.
				padd (bool | str): If True or 'max', pad to max length.
				tensors (str): One of 'pt', 'tf', or 'np'.
	
			Returns:
				Dict[str, any]: Dictionary with input_ids, attention_mask, etc.
			
		"""
		try:
			if text is None:
				raise Exception( 'Input "text" must be provided.' )
			else:
				return self.tokenizer( text, truncation=trunc, padding=padd,
					max_length=max, return_tensors=tensors )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'encode( self, path: str ) -> Dict[ str, Union[ List[ int ], any ] ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def batch_encode( self, texts: List[ str ], max: int=512, trunc: bool=True,
	                  pad: Union[ bool, str ]='max', tensors: str=None ) -> Dict[ str, any ]:
		"""
			
			Encodes a list of
			text inputs as a batch.
	
			Args:
				texts (List[str]): A list of text samples.
				max (int): Max length for truncate.
				trunc (bool): Whether to truncate.
				pad (bool | str): Padding mode.
				tensors (str): Output tensor type.
	
			Returns:
				Dict[str, any]: Tokenized batch with text IDs, masks, etc.
			
		"""
		try:
			if texts is None:
				raise Exception( 'Input "texts" must be provided.' )
			else:
				return self.tokenizer( texts, truncation=trunc, adding=pad,
					max_length=max, return_tensors=tensors )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'batch_encode( self, text: List[ str ] ) -> Dict[ str, any ]-> None'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def decode( self, ids: List[ int ], skip: bool=True ) -> str:
		"""
			
			Converts a list of
			token IDs back to a string.
	
			Args:
				ids (List[int]): Encoded token IDs.
				skip (bool): Exclude special tokens from output.
	
			Returns:
				str: Human-readable decoded string.
				
		"""
		try:
			if ids is None:
				raise Exception( 'The text "ids" must be provided.' )
			else:
				return self.tokenizer.decode( ids, skip_special_tokens=skip )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'decode( self, ids: List[ int ], skip: bool=True ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def convert_tokens( self, tokens: List[ str ] ) -> List[ int ]:
		"""
			
			Converts tokens into
			their corresponding vocabulary IDs.
	
			Args:
				tokens (List[str]): List of subword tokens.
	
			Returns:
				List[int]: Token IDs.
			
		"""
		try:
			if tokens is None:
				raise Exception( 'The text "tokens" must be provided.' )
			else:
				return self.tokenizer.convert_tokens_to_ids( tokens )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'convert_tokens( self, tokens: List[ str ] ) -> List[ int ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def convert_ids( self, ids: List[ int ] ) -> List[ str ]:
		"""
		
			Converts token IDs
			back to subword tokens.
	
			Args:
				ids (List[int]): List of token IDs.
	
			Returns:
				List[str]: List of token strings.
			
		"""
		try:
			if ids is None:
				raise Exception( 'The text "ids" must be provided.' )
			else:
				return self.tokenizer.convert_ids_to_tokens( ids )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'tiggr'
			_exc.cause = 'Token'
			_exc.method = 'convert_ids( self, ids: List[ int ] ) -> List[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )


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
		                                                    'response', 'vector_stores', 'file_ids',
		         'data', 'batches', 'tables', 'vectors', 'create_small_embedding', 'dataframe',
		         'most_similar', 'bulk_similar', 'similarity_heatmap',
		         'export_jsonl', 'import_jsonl', 'create_vector_store',
		         'list_vector_stores', 'upload_vector_store',
		         'query_vector_store', 'delete_vector_store',
		         'upload_document', 'upload_documents' ]
	
	
	def create( self, tokens: List[ str ], batch: int=10, max: int=3,
	            time: float=2.0 ) -> pd.DataFrame:
		"""

			Generate and normalize
			vectors for a list of text tokens.

			Parameters:
			- tokens (List[str]): List of text pages strings
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
			_exc.method = (
				'create_small_embedding( self, tokens: List[ str ], batch: int=10, max: int=3, '
				'time: float=2.0 ) -> pd.DataFrame')
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def _batch_chunks( self, texts: List[ str ], size: int ) -> List[ List[ str ] ]:
		"""

			Split a list of tokens
			into batches of specified size.

			Parameters:
			- tokens (List[str]): Full list of text strings
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
	
	
	def bulk_similar( self, queries: List[ str ], df: pd.DataFrame, top: int=5 ) -> Dict:
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
				self.client.beta.vector_stores.file_batches.create_small_embedding(
					store_id=self.id,
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
				
				with (concurrent.futures.ThreadPoolExecutor( max_workers=10 ) as thread):
					_futures = \
						{
							thread.submit( self.upload_document, self.file_path,
								self.id ): self.file_path
							for self.file_path in self.files
						}
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


class Embeddy( ):
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
			Create embeddings using the small small_model from OpenAI.

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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_small_embedding( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_small_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]:
		"""

			Purpose:
			Create embeddings using the small small_model from OpenAI.

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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_small_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_large_embedding( self, text: str ) -> List[ float ]:
		"""

			Purpose:
			Create embeddings using the large small_model from OpenAI.

			Parameters:
			- text (str):  the string (ie, token) to be embedded

			Returns:
			- List[ List[ float ] ]:  embedded embeddings

		"""
		try:
			self.client.api_key = os.getenv( 'OPENAI_API_KEY' )
			self.raw_input = text.replace( '\n', ' ' )
			self.response = self.client.embeddings.create( input=[ self.raw_input ],
				model=self.large_model )
			
			return self.response.data[ 0 ].embedding
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Embeddy'
			_exc.method = 'create_large_embedding( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_large_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]:
		"""

			Purpose:
			Create embeddings using the large small_model from OpenAI.

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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_large_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_ada_embedding( self, text: str ) -> List[ float ]:
		"""

			Purpose:
			Create embeddings using the ada small_model from OpenAI.

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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_ada_embedding( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_ada_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]:
		"""

			Purpose:
			Create embeddings using the ada small_model from OpenAI.

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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_ada_embeddings( self, tokens: List[ str ] ) -> List[ List[ float ] ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	async def create_small_async( self, text: str ) -> List[ float ]:
		"""

			Purpose:
			Asynchronously creates embeddings using the small small_model from OpenAI.

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
					model=self.small_model ))
			[ 'data' ][ 0 ][ 'embedding' ]
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'embbr'
			_exc.cause = 'Embeddy'
			_exc.method = 'ccreate_small_async( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	async def create_large_async( self, text: str ) -> List[ float ]:
		"""

			Purpose:
			Asynchronously creates embeddings using the large small_model from OpenAI.

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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_large_async( self, text: str ) -> List[ float ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	async def create_ada_async( self, text: str ) -> List[ float ]:
		"""

			Purpose:
			Asynchronously creates embeddings using the ada small_model from OpenAI.

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
			_exc.cause = 'Embeddy'
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
			_exc.cause = 'Embeddy'
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
			
			plt.figure( figsize=(9, 6) )
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
			_exc.cause = 'Embeddy'
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
			_exc.cause = 'Embeddy'
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
			_exc.cause = 'Embeddy'
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
			_exc.cause = 'Embeddy'
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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_tsne_components( self, vectors: List[ List[ float ] ], num=2 ) -> np.ndarray'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def create_chart( self, components: np.ndarray,
	                  labels: Optional[ List[ str ] ] = None,
	                  strings: Optional[ List[ str ] ] = None,
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
			_exc.cause = 'Embeddy'
			_exc.method = "('create_chart( self, components: np.ndarray  mark_size=5 ) -> None')"
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def creat_3dchart( self,
	                   components: np.ndarray,
	                   labels: Optional[ List[ str ] ] = None,
	                   strings: Optional[ List[ str ] ] = None,
	                   x_title: str = 'Component-0',
	                   y_title: str = 'Component-1',
	                   z_title: str = 'Compontent-2',
	                   mark_size: int = 5 ):
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
			_exc.cause = 'Embeddy'
			_exc.method = 'create_vector_store( self, name: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
