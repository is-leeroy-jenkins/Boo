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
from collections import defaultdict


class Text:
	'''
	
		Class providing documents preprocessing functionality
		
	    Methods:
	    --------
	    collapse_whitespace( self, pages: str ) -> str
	    remove_punctuation( self, pages: str ) -> str:
		remove_special( self, pages: str ) -> str:
		remove_html( self, pages: str ) -> str
		remove_errors( self, pages: str ) -> str
		correct_errors( self, pages: str ) -> str:
		remove_markdown( self, pages: str ) -> str
	    normalize(pages: str) -> str
	    tokenize_sentences(pages: str) -> str
	    tokenize_words(pages: str) -> get_list
	    load_file( url: str) -> li
	    lemmatize(tokens: get_list) -> str
	    bag_of_words(tokens: get_list) -> dict
	    train_word2vec(sentences: get_list, vector_size=100, window=5, min_count=1) -> Word2Vec
	    compute_tfidf(corpus: get_list, max_features=1000, prep=True) -> tuple
	    
	'''
	
	
	def __init__( self ):
		'''
			Constructor for creating Text objects
		'''
		self.raw_input = None
		self.raw_pages = None
		self.normalized = None
		self.lemmatized = None
		self.tokenized = None
		self.corrected = None
		self.cleaned_text = None
		self.words = [ str ]
		self.tokens = [ str ]
		self.lines = [ str ]
		self.pages = [ str ]
		self.chunks = [ str ]
		self.chunk_size = 0
		self.stop_words = [ str ]
		self.cleaned_lines = [ str ]
		self.removed = [ str ]
		self.lowercase = None
		self.translator = None
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.tokenizer = None
		self.vectorizer = None
	
	
	def load_text( self, path: str ) -> str:
		try:
			if path is None:
				raise Exception( 'The input argument "path" is required' )
			else:
				self.raw_input = path
				return Path( path ).read_text( encoding='utf-8' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'load_text( self, path: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def split_lines( self, text: str ) -> list[ str ]:
		"""
		
			Splits the input pages into lines

			Parameters:
			-----------
			pages : str

			Returns:
			--------
			list[ str ]
			
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "pages" is required' )
			else:
				with open( text, 'r', encoding='utf-8' ) as f:
					self.lines = f.readlines( )
					return self.lines
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'split_lines( self, text: str ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def split_pages( self, path: str, delimiter: str = '\f' ) -> list[ str ]:
		"""
		
			Reads text from a file, splits it into lines,
			and groups them into pages.
	
			Args:
				path (str): Path to the text file.
				delimiter (str): Page separator string
				(default is '\f' for form feed).
	
			Returns:
				list of str: List where each element
				is the text of one page.
				
		"""
		try:
			if path is None:
				raise Exception( 'The input argument "path" is required' )
			else:
				with open( path, 'r', encoding='utf-8' ) as f:
					self.raw_input = f.read( )
		except UnicodeDecodeError:
			with open( path, 'r', encoding='latin1' ) as f:
				self.raw_input = f.read( )
			
			# Split by pages
			self.raw_pages = self.raw_input.split( delimiter )
			
			# Clean and consolidate lines per page
			for page in self.raw_pages:
				self.lines = page.strip( ).splitlines( )
				self.cleaned_text = "\n".join( [ line.strip( ) for line in self.lines if line.strip( ) ] )
				self.pages.append( self.cleaned_text )
			
			return self.pages
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'split_pages( self, path: str, delimiter: str = "/f" ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def collapse_whitespace( self, text: str ) -> str:
		"""
			Removes extra spaces and
			blank lines from the input pages.

			Parameters:
			-----------
			pages : str

			Returns:
			--------
			
				A cleaned_lines pages string with:
					- Consecutive whitespace reduced to a single space
					- Leading/trailing spaces removed
					- Blank lines removed
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "pages" is required.' )
			else:
				self.raw_input = text
				self.words = re.sub( r'[ \t]+', ' ', self.raw_input )
				self.cleaned_lines = [ line.strip( ) for line in self.words.splitlines( ) ]
				self.lines = [ line for line in self.cleaned_lines if line ]
				self.removed = '\n'.join( self.lines )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'collapse_whitespace( self, pages: str ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_punctuation( self, text: str ) -> str:
		"""

			Removes all punctuation characters
			 from the input pages string.

			Parameters:
			-----------
			pages : str
				The input pages string to be cleaned_lines.

			Returns:
			--------
			str
				The pages string with all punctuation removed.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "pages" is required.' )
			else:
				self.raw_input = text
				self.translator = str.maketrans( '', '', string.punctuation )
				self.removed = self.raw_input.translate( self.translator )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_punctuation( self, pages: str ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_special( self, text: str, keep_spaces=True ) -> str:
		"""

			Removes special characters
			from the input pages string.

			This function:
			  - Retains only alphanumeric characters and whitespace
			  - Removes symbols like @, #, $, %, &, etc.
			  - Preserves letters, numbers, and spaces

			Parameters:
			-----------
			pages : str
				The raw input pages string potentially
				containing special characters.

			Returns:
			--------
			str
				A cleaned_lines string containing only letters, numbers, and spaces.

		"""
		try:
			if text is None:
				raise Exception( 'The input argument "pages" is required.' )
			elif keep_spaces:
				self.raw_input = text
				return re.sub( r'[^a-zA-Z0-9\s]', '', text )
			else:
				return re.sub( r'[^a-zA-Z0-9]', '', text )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_punctuation( self, pages: str ) -> str:'
			_err = ErrorDialog( _exc )
			_err.show( )


	def remove_html( self, text: str ) -> str:
		"""
	
			Removes HTML tags from the input pages string.
	
			This function:
			  - Parses the pages as HTML
			  - Extracts and returns only the visible content without tags
	
			Parameters:
			-----------
			pages : str
				The input pages containing HTML tags.
	
			Returns:
			--------
			str
				A cleaned_lines string with all HTML tags removed.
	
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "pages" is required.' )
			else:
				self.raw_html = text
				self.cleaned_html = BeautifulSoup( self.raw_html, "raw_html.parser" )
				self.removed = self.cleaned_html.get_text( separator=' ', strip=True )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_html( self, pages: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_errors( self, text: str ) -> str:
		"""
	
			Removes misspelled or non-English
			words from the input pages.
	
			This function:
			  - Converts pages to lowercase
			  - Tokenizes the pages into words
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
				raise Exception( 'The input argument "pages" is required.' )
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
			_exc.method = 'remove_errors( self, pages: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def correct_errors( self, text: str ) -> str:
		"""
	
			Corrects misspelled words in the input pages string.
	
			This function:
			  - Converts pages to lowercase
			  - Tokenizes the pages into words
			  - Applies spelling correction using TextBlob
			  - Reconstructs and returns the corrected pages
	
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
				raise Exception( 'The input argument "pages" is required.' )
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
			_exc.method = 'correct_errors( self, pages: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_html( self, text: str ) -> str:
		"""
	
			Removes HTML  from pages.
	
			This function:
			  - Strips HTML tags
	
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
				raise Exception( 'The input argument "pages" is required.' )
			else:
				self.raw_input = text
				self.removed = (BeautifulSoup( self.raw_input, "raw_html.parser" )
				                .get_text( separator=' ', strip=True ))
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_html( self, pages: str ) -> str'
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
				raise Exception( 'The input argument "pages" is required.' )
			else:
				self.raw_input = text
				self.cleaned_text = re.sub( r'\[.*?\]\(.*?\)', '', self.raw_input )
				self.corrected = re.sub( r'[`_*#~>-]', '', self.cleaned_text )
				self.removed = re.sub( r'!\[.*?\]\(.*?\)', '', self.corrected )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'remove_markdown( self, pages: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_stopwords( self, text: str ) -> str:
		"""
	
			Removes English stopwords from the input pages string.
	
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
				raise Exception( 'The input argument "pages" is required.' )
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
			_exc.method = 'remove_stopwords( self, pages: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def remove_headers( self, pages: List[ str ], min_occurrences: int=3 ) -> List[ str ]:
		"""
			
			Removes repetitive headers and footers
			across a list of pages by frequency analysis.
		
			Args:
				pages (list of str): A list where each
				element is the full text of one page.
				min_occurrences (int): Minimum number of times
				a line must appear at the top/bottom to
				be considered a header/footer.
		
			Returns:
				list of str: List of cleaned_lines page
				texts without detected headers/footers.
			
		"""
		header_counts = defaultdict( int )
		footer_counts = defaultdict( int )
		
		# First pass: collect frequency of top/bottom lines
		for page in pages:
			lines = page.strip( ).splitlines( )
			if lines:
				header_counts[ lines[ 0 ].strip( ) ] += 1
				footer_counts[ lines[ -1 ].strip( ) ] += 1
		
		# Identify candidates for removal
		headers_to_remove = { line for line, count in header_counts.items( ) if
		                      count >= min_occurrences }
		footers_to_remove = { line for line, count in footer_counts.items( ) if
		                      count >= min_occurrences }
		
		# Second pass: clean pages
		cleaned_pages = [ ]
		for page in pages:
			lines = page.strip( ).splitlines( )
			if not lines:
				cleaned_pages.append( page )
				continue
			
			# Remove header
			if lines[ 0 ].strip( ) in headers_to_remove:
				lines = lines[ 1: ]
			
			# Remove footer
			if lines and lines[ -1 ].strip( ) in footers_to_remove:
				lines = lines[ :-1 ]
			
			cleaned_pages.append( "\n".join( lines ) )
		
		return cleaned_pages
	
	
	def normalize( self, text: str ) -> str:
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
				raise Exception( 'The input argument "pages" is required.' )
			else:
				self.raw_input = text
				self.normalized = self.raw_input.lower( )
				return self.normalized
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'normalize_text( self, pages: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def lemmatize( self, text: str ) -> str:
		"""
	
			Performs lemmatization on the input pages string.
	
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
			if text is None:
				raise Exception( 'The input argument "pages" is required.' )
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
			_exc.method = 'tokenize_words( self, pages: str ) -> str'
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
				raise Exception( 'The input argument "pages" was None' )
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
			_exc.method = 'tokenize_text( self, pages: str ) -> get_list'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_words( self, text: str ) -> List[ str ]:
		"""
		
			Tokenize a sentence or paragraph into word tokens.
	
			Args:
				text (str): Input pages.
	
			Returns:
				list: List of word tokens.
				
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "pages" was None' )
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
			_exc.method = 'tokenize_words( self, pages: str ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_sentences( self, text: str ) -> List[ str ]:
		"""
		
			Tokenize a paragraph or document into a get_list of sentence strings.
	
			Args:
				text (str): Input pages.
	
			Returns:
				list: List of sentence strings.
				
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "pages" is required.' )
			else:
				self.tokens = sent_tokenize( text )
				return self.tokens
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize_sentences( self, pages: str ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def chunk_text( self, text: str, max: int=800 ) -> List[ str ]:
		'''

			Simple chunking by words assuming ~1.3 words per token

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
				raise Exception( 'The input argument "pages" is required.' )
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
			_exc.method = 'chunk_text( self, pages: str, max: int = 512 ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )

	
	def chunk_tokens( self, tokens: [ str ], max: int=800, over: int=50 ) -> List[ str ]:
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
	
	
	def bag_of_words( self, tokens: List[ str ] ) -> dict:
		"""
		
			Construct a Bag-of-Words (BoW) frequency dictionary from tokens.
	
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
			_exc.method = 'bag_of_words( self, tokens: get_list ) -> dict'
			_err = ErrorDialog( _exc )
			_err.show( )

	
	def train_word2vec( self, tokens: List[ str ], size=100, window=5, min=1 ) -> Word2Vec:
		"""
			Purpose:
				Train a Word2Vec embedding model from tokenized sentences.
	
			Args:
				sentences (get_list of get_list of str): List of tokenized sentences.
				vector_size (int): Dimensionality of word vec.
				window (int): Max distance between current and predicted word.
				min_count (int): Minimum frequency for inclusion in vocabulary.
	
			Returns:
				Word2Vec: Trained Gensim Word2Vec model.
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


	def compute_tfidf( self, corpus: [ str ], max: int=1000, prep: bool=True ) -> Tuple:
		"""
			Purpose:
			Compute TF-IDF matrix with optional full preprocessing pipeline.
	
			Args:
				corpus (list): List of raw or preprocessed pages documents.
				max_features (int): Max number of terms to include (vocabulary size).
				prep (bool): If True, normalize, tokenize_text, clean, and lemmatize input.
	
			Returns:
				tuple:
					- tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF feature matrix.
					- feature_names (get_list): Vocabulary terms.
					- vectorizer (TfidfVectorizer): Fitted vectorizer instance.
	
		"""
		try:
			if corpus is None:
				raise Exception( 'The input argument "corpus" is required.' )
			elif prep:
				for doc in corpus:
					self.normalized = self.normalize( doc )
					self.tokens = self.tokenize_words( self.normalized )
					self.lines = [ self.lemmatize( token ) for token in self.tokens ]
					self.cleaned_text = " ".join( self.lines )
					self.cleaned_lines.append( cleaned_text )
			
			self.vectorizer = TfidfVectorizer( max_features=max, stop_words='english' )
			_matrix = self.vectorizer.fit_transform( self.cleaned_lines )
			return ( _matrix, self.vectorizer.get_feature_names_out( ).tolist( ),
			        self.vectorizer)
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = ('compute_tfidf( self, corpus: list, max: int=1000, prep: bool=True ) -> '
			               'tuple')
			_err = ErrorDialog( _exc )
			_err.show( )
