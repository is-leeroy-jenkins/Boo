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
		
	    Methods:
	    --------
	    collapse_whitespace( self, text: str ) -> str
	    remove_punctuation( self, text: str ) -> str:
		remove_special( self, text: str ) -> str:
		remove_html( self, text: str ) -> str
		remove_errors( self, text: str ) -> str
		correct_errors( self, text: str ) -> str:
		remove_markdown( self, text: str ) -> str
	    normalize(text: str) -> str
	    tokenize_sentences(text: str) -> str
	    tokenize_words(text: str) -> list
	    load_file( url: str) -> li
	    lemmatize(tokens: list) -> str
	    bag_of_words(tokens: list) -> dict
	    train_word2vec(sentences: list, vector_size=100, window=5, min_count=1) -> Word2Vec
	    compute_tfidf(corpus: list, max_features=1000, prep=True) -> tuple
	    
	'''
	def __init__( self ):
		self.raw_input = None
		self.cleaned = [ str ]
		self.removed = [ str ]
		self.normalized = None
		self.lowercase = None
		self.translator = None
		self.lemmatizer = WordNetLemmatizer( )
		self.stemmer = PorterStemmer( )
		self.tokenizer = None
		self.lemmatized = None
		self.tokenized = None
		self.corrected = None
		self.vectorizer = None
		self.words = [ str ]
		self.tokens = [ str ]
		self.lines = [ str ]
		self.chunks = [ str ]
		self.stop_words = [ str ]
	
	
	def __dir__( self ):
		'''
			returns a list[ str ] of members
		'''
		return [ 'raw_input', 'cleaned', 'removed',
		         'lowercase', 'normalized', 'translator',
		         'lemmatizer', 'tokenizer', 'vectorizer',
		         'stemmer', 'lemmatized', 'tokens',
		         'tokenized', 'corrected', 'words',
		         'stop_words', 'lines', 'chunks' ]
	
	
	def collapse_whitespace( self, text: str ) -> str:
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
				self.cleaned = [ line.strip( ) for line in self.words.splitlines( ) ]
				self.lines = [ line for line in self.cleaned if line ]
				self.removed = '\n'.join( self.lines )
				return self.removed
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'collapse_whitespace( self, text: str ) -> str:'
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
			  - Tokenizes the lowercased text into words
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
	
	
	def chunk( self, tokens: list, max: int=800, over: int=50 ) -> list[ str ]:
		"""
			Purpose:
				Split a list of tokens into overlapping chunks based on token limits.

			Args:
				tokens (list): Tokenized input documents.
				max (int): Max token size per chunk.
				over (int): Overlapping token count between chunks.

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
					end = start + max
					chunk = tokens[ start:end ]
					chunks.append( chunk )
					start += max - over
				
				return chunks
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'chunk( self, tokens: list, max: int=800, over: int=50 ) -> list[ str ]'
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
			_exc.method = 'tokenize( self, text: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_words( self, text: str ) -> list[ str ]:
		"""
		
			Tokenize a sentence or paragraph into word tokens.

			Args:
				text (str): Input text.

			Returns:
				list: List of word tokens.
				
		"""
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
			_exc.method = 'tokenize_words( self, text: str ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize_sentences( self, text: str ) -> list[ str ]:
		"""
		
			Tokenize a paragraph or document into a list of sentence strings.
	
			Args:
				text (str): Input text.
	
			Returns:
				list: List of sentence strings.
				
		"""
		try:
			if text is None:
				raise Exception( 'The input argument "text" is required.' )
			else:
				return sent_tokenize( text )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'tokenize_sentences( self, text: str ) -> list[ str ]'
			_err = ErrorDialog( _exc )
			_err.show( )

	
	def bag_of_words( self, tokens: list[ str ] ) -> dict:
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
				return dict( Counter( tokens ) )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'bag_of_words( self, tokens: list ) -> dict'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def train_word2vec( self, tokens: list, size=100, window=5, min=1 ) -> Word2Vec:
		"""
			Train a Word2Vec embedding model from tokenized sentences.
	
			Args:
				sentences (list of list of str): List of tokenized sentences.
				vector_size (int): Dimensionality of word vectors.
				window (int): Max distance between current and predicted word.
				min_count (int): Minimum frequency for inclusion in vocabulary.
	
			Returns:
				Word2Vec: Trained Gensim Word2Vec model.
		"""
		try:
			if tokens is None:
				raise Exception( 'The input argument "sentences" is required.' )
			else:
				return Word2Vec( sentences=tokens, vector_size=size, window=window, min_count=min )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'train_word2vec( self, tokens: list, size=100, window=5, min=1 ) -> Word2Vec'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def compute_tfidf( self, corpus: list, max: int=1000, prep: bool=True ) -> tuple:
		"""
	
			Compute TF-IDF matrix with optional full preprocessing pipeline.
	
			Args:
				corpus (list): List of raw or preprocessed text documents.
				max_features (int): Max number of terms to include (vocabulary size).
				prep (bool): If True, normalize, tokenize, clean, and lemmatize input.
	
			Returns:
				tuple:
					- tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF feature matrix.
					- feature_names (list): Vocabulary terms.
					- vectorizer (TfidfVectorizer): Fitted vectorizer instance.
	
		"""
		try:
			if corpus is None:
				raise Exception( 'The input argument "corpus" is required.' )
			elif prep:
				cleaned_docs = [ ]
				for doc in corpus:
					norm = self.normalize( doc )
					self.tokens = self.tokenize_words( norm )
					_lemma = [ self.lemmatize( token ) for token in self.tokens ]
					cleaned_doc = " ".join( _lemma )
					cleaned_docs.append( cleaned_doc )
					
			self.vectorizer = TfidfVectorizer( max_features=max, stop_words='english' )
			tfidf_matrix = self.vectorizer.fit_transform( cleaned_docs )
			return tfidf_matrix, self.vectorizer.get_feature_names_out( ).tolist( ), self.vectorizer
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Token'
			_exc.method = 'compute_tfidf( self, corpus: list, max: int=1000, prep: bool=True ) -> tuple'
			_err = ErrorDialog( _exc )
			_err.show( )


	def load_file( self, path: str ) -> str:
		"""
		
			Load the content of a document's file.
	
			Args:
				path (str): The url to the documents file.
	
			Returns:
				str: The contents of the file as a string.
				
		"""
		try:
			if path is None:
				raise Exception( 'Input parameter "url" is required.' )
			else:
				self.raw_input = path
				return Path( path ).read_text( encoding='utf-8' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'load_file( self, url: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	

