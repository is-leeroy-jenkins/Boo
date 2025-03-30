'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                Tidy.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="Tidy.py" company="Terry D. Eppler">

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
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from Booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
import string


class Text:
	'''
		Class providing text preprocessing functionality
		@params: tokenize: bool
	'''
	
	
	def __init__( self, tokenize: bool = False ):
		self.tokenize = tokenize
	
	
	def convert( self, input: str, output: str ):
		'''
			
			Reads the unprocessed 'input' file and writes a cleaned 'output' file
	
			Args:
				input (str): path to pre-processed file.
				output (str): path to processed file.
			Returns:
				str: Cleaned and newlines text.
				
		'''
		try:
			if not os.path.exists( input ):
				raise FileNotFoundError( f'Input file not found: {input}' )
			
			if output is None:
				raise FileNotFoundError( f'Output file not found: {output}' )
			
			cleaned_lines = [ ]
			with open( input, 'r', encoding='utf-8' ) as infile:
				for line in infile:
					cleaned = self.clean_line( line )
					if cleaned:
						cleaned_lines.append( cleaned )
			
			with open( output, 'w', encoding='utf-8' ) as outfile:
				for cleaned_line in cleaned_lines:
					if self.tokenize:
						tokens = self.tokenize( cleaned_line )
						json_obj = { 'tokens': tokens }
					else:
						json_obj = { 'line': cleaned_line }
					json_str = json.dumps( json_obj, ensure_ascii=False )
					outfile.write( json_str + '\n' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tidy'
			_exc.cause = 'Text'
			_exc.method = 'convert( self, input: str, output: str )'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def clean_text( self, text: str ) -> str:
		'''
			
			Clean the raw text extracted from a PDF for preprocessing.
	
			This includes removing headers, footers, page numbers, hyphenations,
			fixing line breaks, collapsing whitespace, and normalizing section markers.
	
			Args:
				text (str): Raw extracted text.
	
			Returns:
				str: Cleaned and newlines text.
		
		'''
		try:
			if text is None:
				_msg = 'The input parameter "text" is required.'
				raise Exception( _msg )
			else:
				# Step 1: Normalize newlines
				self.newlines = text.replace( '\r\n', '\n' ).replace( '\r', '\n' )
				
				# Step 2: Remove page headers and footers (Public Law-specific)
				self.headers = re.sub( r'PUBLIC.*?\n', '', self.newlines, flags=re.IGNORECASE )
				self.footers = re.sub( r'\n\s*\d+\s*\n', '\n', self.headers )
				
				# Step 3: Remove hyphenation at line breaks (e.g., "appropria-\ntion")
				self.hyphens = re.sub( r'(\w+)-\n(\w+)', r'\1\2', self.footers )
				
				# Step 4: Merge broken lines where sentence continues
				self.merge = re.sub( r'(?<!\n)\n(?![\n])', ' ', self.hyphens )
				
				# Step 5: Collapse excessive whitespace
				self.whitespace = re.sub( r'\s+', ' ', self.merge )
				
				# Step 6: Normalize section markers (optional but useful)
				self.normalized = re.sub( r'(SEC\.\s*\d+[A-Z]*\.)', r'\n\n\1', self.whitespace )
				
				return self.normalized.strip( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tidy'
			_exc.cause = 'Text'
			_exc.method = 'clean_text( self, line: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def clean_line( self, line: str ) -> str:
		"""Clean the raw text extracted from a PDF for preprocessing.

		This includes removing headers, footers, page numbers, hyphenations,
		fixing line breaks, collapsing whitespace, and normalizing section markers.

		Args:
			text (str): Raw extracted text.

		Returns:
			str: Cleaned and newlines text.
		"""
		try:
			if line is None:
				_msg = 'The input parameter "line" is None'
				raise Exception( _msg )
			else:
				# Step 1: Normalize newlines
				self.newlines = line.replace( '\r\n', '\n' ).replace( '\r', '\n' )
				
				# Step 2: Remove page headers and footers (Public Law-specific)
				self.headers = re.sub( r'PUBLIC.*?\n', '', self.newlines, flags=re.IGNORECASE )
				self.footers = re.sub( r'\n\s*\d+\s*\n', '\n', self.headers )
				
				# Step 3: Remove hyphenation at line breaks (e.g., "appropria-\ntion")
				self.hyphens = re.sub( r'(\w+)-\n(\w+)', r'\1\2', self.footers )
				
				# Step 4: Merge broken lines where sentence continues
				self.merge = re.sub( r'(?<!\n)\n(?![\n])', ' ', self.hyphens )
				
				# Step 5: Collapse excessive whitespace
				self.whitespace = re.sub( r'\s+', ' ', self.merge )
				
				# Step 6: Normalize section markers (optional but useful)
				self.normalized = re.sub( r'(SEC\.\s*\d+[A-Z]*\.)', r'\n\n\1', self.whitespace )
				
				return self.normalized.strip( )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tidy'
			_exc.cause = 'Text'
			_exc.method = 'clean_line( self, line: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def clean_string( self, text: str, stem='None' ) -> str:
		'''
		
			Clean the raw text extracted for preprocessing.
			This includes removing headers, footers, page numbers, hyphenations,
			fixing line breaks, collapsing whitespace, and normalizing section markers.
	
			Args:
				text (str): Raw extracted text.
	
			Returns:
				str: Cleaned and newlines text.
				
		'''
		try:
			if text is None:
				_msg = 'The input parameter "text" is required'
				raise Exception( _msg )
			else:
				self.final_string = ''
				self.lower = text.lower( )
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
			_exc.module = 'Tidy'
			_exc.cause = 'Text'
			_exc.method = 'clean_string( self, text: str, stem="None" ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize( self, cleaned_line: str ) -> list:
		'''
		
			Clean the raw text extracted for preprocessing.
			This includes removing headers, footers, page numbers, hyphenations,
			fixing line breaks, collapsing whitespace, and normalizing section markers.
	
			Args:
				cleaned_line: (str) - clean text.
	
			Returns:
				list: Cleaned and newlines text.
				
		'''
		try:
			if cleaned_line is None:
				_msg = 'The input parameter "cleaned_line" was None'
				raise Exception( _msg )
			else:
				words = cleaned_line.split( )
				tokens = [ re.sub( r'[^\w"-]', '', word ) for word in words if word.strip( ) ]
				return tokens
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tidy'
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
				_msg = 'The input parameter "html" was None'
				raise Exception( _msg )
			else:
				soup = BeautifulSoup( html, 'html.parser' )
				for data in soup( [ 'style', 'script', 'code', 'a' ] ):
					data.decompose( )
				
				return ' '.join( soup.stripped_strings )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tidy'
			_exc.cause = 'Text'
			_exc.method = 'clean_html( self, html: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )


class Token:
	
	def __init__( self, model: str = "text-embedding-ada-002" ):
		"""Initialize the PreTokenizer with a specific OpenAI model.

		Args:
			model (str): The name of the OpenAI model used for tokenization.
		"""
		self.model = model
		self.encoding = tiktoken.encoding_for_model( model )
	
	
	def load_file( self, path: str ) -> str:
		"""Load the content of a text file.

		Args:
			path (str): The path to the text file.

		Returns:
			str: The contents of the file as a string.
		"""
		try:
			if path is None:
				_msg = 'Input parameter "text" is required.'
				raise Exception( _msg )
			else:
				return Path( path ).read_text( encoding='utf-8' )
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tidy'
			_exc.cause = 'Token'
			_exc.method = 'load_file( self, path: str ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def parse_hierarchy( self, text: str ) -> list:
		"""Parse the cleaned text into a structured hierarchy of sections, subsections,
		and paragraphs.
	
		Args:
			text (str): Cleaned legal or structured document text.
	
		Returns:
			list: A list of dictionaries, each representing a structural unit with section markers
			and text.
		"""
		try:
			if text is None:
				_msg = 'Input parameter "text" is required.'
				raise Exception( _msg )
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
								"text": self.buffer.strip( )
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
								"text": self.buffer.strip( )
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
								"text": self.buffer.strip( )
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
						"text": self.buffer.strip( )
					} )
				
				return self.structured
		except Exception as e:
			_exc = Error( e )
			_exc.module = 'Tidy'
			_exc.cause = 'Token'
			_exc.method = 'parse_hierarchy( self, text: str ) -> list'
			_err = ErrorDialog( _exc )
			_err.show( )


def tokenize( self, text: str ) -> list:
	"""
		Tokenize a block of text using the OpenAI model tokenizer.
	
		Args:
			text (str): Text to tokenize.
	
		Returns:
			list: A list of token IDs.
	"""
	try:
		if text is None:
			_msg = 'Input parameter "text" is required.'
			raise Exception( _msg )
		else:
			return self.encoding.encode( text )
	except Exception as e:
		_exc = Error( e )
		_exc.module = 'Tidy'
		_exc.cause = 'Token'
		_exc.method = 'tokenize( self, text: str ) -> list'
		_err = ErrorDialog( _exc )
		_err.show( )


def chunk_tokens( self, tokens: list, max_tokens: int = 800, overlap: int = 50 ) -> list:
	"""Split a list of tokens into overlapping chunks based on token limits.

	Args:
		tokens (list): Tokenized input text.
		max_tokens (int): Max token size per chunk.
		overlap (int): Overlapping token count between chunks.

	Returns:
		list: A list of token chunks.
	"""
	try:
		if tokens is None:
			_msg = 'Input parameter "tokens" is required.'
			raise Exception( _msg )
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
		_exc.module = 'Tidy'
		_exc.cause = 'Token'
		_exc.method = ('chunk_tokens( self, tokens: list, '
		               'max_tokens: int = 800, overlap: int = 50 ) -> list')
		_err = ErrorDialog( _exc )
		_err.show( )


def decode_tokens( self, tokens: list ) -> str:
	"""
		Decode a list of token IDs back to string text.
	
		Args:
			tokens (list): A list of token IDs.
	
		Returns:
			str: Decoded string.
	"""
	try:
		if tokens is None:
			_msg = 'Input parameter "tokens" is required.'
			raise Exception( _msg )
		return self.encoding.decode( tokens )
	except Exception as e:
		_exc = Error( e )
		_exc.module = 'Tidy'
		_exc.cause = 'Token'
		_exc.method = 'decode_tokens( self, tokens: list ) -> str'
		_err = ErrorDialog( _exc )
		_err.show( )


def chunk_text_for_embedding( self, text: str, max_tokens: int = 800,
                              overlap: int = 50 ) -> list:
	"""
		Chunk text into strings suitable for embedding under the token limit.
	
		Args:
			text (str): Raw or cleaned input text.
			max_tokens (int): Max tokens per chunk for embedding model.
			overlap (int): Overlap between consecutive chunks.
	
		Returns:
			list: List of decoded text chunks.
	"""
	try:
		if(text is None):
			_msg = 'Input parameter "text" is required.'
			raise Exception( _msg )
		else:
			tokens = self.tokenize( text )
			token_chunks = self.chunk_tokens( tokens, max_tokens, overlap )
			return [ self.decode_tokens( chunk ) for chunk in token_chunks ]
	except Exception as e:
		_exc = Error( e )
		_exc.module = 'Tidy'
		_exc.cause = 'Token'
		_exc.method = ( 'chunk_text_for_embedding( self, text: str, max_tokens: int = 800, '
		               'overlap: int = 50 ) -> list' )
		_err = ErrorDialog( _exc )
		_err.show( )
