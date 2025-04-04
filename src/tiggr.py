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
				raise Exception( 'The input parameter "text" is required.' )
			else:
				# Replace multiple spaces or tabs with a single space
				text = re.sub( r'[ \t]+', ' ', text )
				
				# Remove leading/trailing spaces from each line
				lines = [ line.strip( ) for line in text.splitlines( ) ]
				
				# Remove empty lines
				cleaned_lines = [ line for line in lines if line ]
				
				# Join lines back into a single string
				cleaned_text = '\n'.join( cleaned_lines )
				
				return cleaned_text
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
				raise Exception( 'The input parameter "text" is required.' )
			else:
				# Convert to lowercase
				_lower = text.lower( )
				
				# Remove accented characters using Unicode normalization
				_unicode = (unicodedata.normalize( 'NFKD', _lower )
				        .encode( 'ascii', 'ignore' )
				        .decode( 'utf-8' ) )
				
				# Trim leading/trailing spaces and collapse internal whitespace
				_normalized = re.sub( r'\s+', ' ', _unicode ).strip( )
				
				return _normalized
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
		# Strip leading and trailing whitespace
		text = text.strip( )
		
		# Replace multiple whitespace characters (spaces, tabs, etc.) with a single space
		cleaned_text = re.sub( r'\s+', ' ', text )
		
		return cleaned_text
	
	
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
		# Create a translation table that maps punctuation to None
		translator = str.maketrans( '', '', string.punctuation )
		
		# Apply the translation to the text
		cleaned_text = text.translate( translator )
		
		return cleaned_text
	
	
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
		# Initialize lemmatizer
		lemmatizer = WordNetLemmatizer( )
		
		lower_case = text.lower( )
		# Convert to lowercase and tokenize
		tokens = word_tokenize( lower_case )
		
		# Lemmatize each token
		lemmatized_tokens = [ lemmatizer.lemmatize( token ) for token in tokens ]
		
		# Join tokens back to a string
		lemmatized_text = ' '.join( lemmatized_tokens )
		
		return lemmatized_text
	
	
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
		# Convert to lowercase
		_lower = text.lower( )
		
		# Tokenize
		tokens = word_tokenize( _lower )
		
		return tokens
	
	
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
		# Uses regex to replace all non-alphanumeric characters with empty strings
		cleaned_text = re.sub( r'[^A-Za-z0-9\s]', '', text )
		
		return cleaned_text
	
	
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
		# Parse HTML and extract text
		soup = BeautifulSoup( text, "html.parser" )
		cleaned_text = soup.get_text( separator=' ', strip=True )
		
		return cleaned_text
	
	
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
		# Flatten the list of token lists into a single list
		all_tokens = [ token for sublist in text for token in sublist ]
		
		# Create chunks of tokens
		chunks = [
			all_tokens[ i:i + chunk_size ]
			for i in range( 0, len( all_tokens ), chunk_size )
		]
		
		return chunks
	
	
	def chunk_text( self, text: str, chunk_size: int = 50, return_as_string: bool = True ) -> list:
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

			return_as_string : bool, optional (default=True)
				If True, returns each chunk as a string; otherwise, returns a list of tokens.

			Returns:
			--------
			list
				A list of token chunks. Each chunk is either a list of tokens or a string.

		"""
		# Tokenize the text into words
		tokens = word_tokenize( text.lower( ) )
		
		# Create chunks of specified token length
		token_chunks = [
			tokens[ i:i + chunk_size ]
			for i in range( 0, len( tokens ), chunk_size )
		]
		
		# Optionally join tokens into strings
		if return_as_string:
			return [ ' '.join( chunk ) for chunk in token_chunks ]
		else:
			return token_chunks


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
		# Convert to lowercase and tokenize
		tokens = word_tokenize( text.lower( ) )
		
		# Keep only correctly spelled words (as per Word dictionary in TextBlob)
		cleaned_tokens = [ word for word in tokens if Word( word ).spellcheck( )[ 0 ][ 1 ] > 0.9 ]
		
		# Return cleaned string
		return ' '.join( cleaned_tokens )
	
	
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
		# Convert to lowercase and tokenize
		tokens = word_tokenize( text.lower( ) )
		
		# Apply spelling correction to each token
		corrected_tokens = [ str( Word( word ).correct( ) ) for word in tokens ]
		
		# Join the corrected words into a single string
		corrected_text = ' '.join( corrected_tokens )
		
		return corrected_text
	
	
	def remove_headers_footers( self, text: str ) -> str:
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
		# Split the text into lines
		lines = text.splitlines( )
		
		# Remove empty lines and trim whitespace
		lines = [ line.strip( ) for line in lines if line.strip( ) ]
		
		# Count line frequencies to identify repeated headers/footers
		line_counts = Counter( lines )
		
		# Identify frequent lines (appear in >1% of total lines)
		threshold = max( 1, int( len( lines ) * 0.01 ) )
		repeated_lines = { line for line, count in line_counts.items( ) if count > threshold }
		
		# Remove lines that are likely headers or footers
		body_lines = [ line for line in lines if line not in repeated_lines ]
		
		# Reconstruct the cleaned text
		cleaned_text = '\n'.join( body_lines )
		
		return cleaned_text
	
	
	def remove_formatting( self, text: str ) -> str:
		"""

			Removes formatting artifacts (Markdown, HTML, control characters) from text.

			This function:
			  - Strips HTML tags
			  - Removes Markdown syntax (e.g., *, #, [], etc.)
			  - Collapses whitespace (newlines, tabs)
			  - Optionally removes special characters for clean unformatted text

			Parameters:
			-----------
			text : str
				The formatted input text.

			Returns:
			--------
			str
				A cleaned version of the text with formatting removed.

		"""
		# Remove HTML tags
		text = BeautifulSoup( text, "html.parser" ).get_text( separator=' ', strip=True )
		
		# Remove Markdown syntax
		text = re.sub( r'\[.*?\]\(.*?\)', '', text )  # Markdown links
		text = re.sub( r'[`_*#~>-]', '', text )  # Markdown chars
		text = re.sub( r'!\[.*?\]\(.*?\)', '', text )  # Markdown images
		
		# Remove control characters and normalize whitespace
		text = re.sub( r'[\r\n\t]+', ' ', text )  # Newlines, tabs
		text = re.sub( r'\s+', ' ', text ).strip( )  # Collapse multiple spaces
		
		return text
	
	
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
		# Download required NLTK resources (only once)
		nltk.download( 'punkt', quiet=True )
		
		# Define English stopword set
		stop_words = set( stopwords.words( 'english' ) )
		
		# Tokenize and lowercase
		tokens = word_tokenize( text.lower( ) )
		
		# Remove stopwords
		filtered_tokens = [ word for word in tokens if word.isalnum( ) and word not in stop_words ]
		
		# Join tokens back into a string
		cleaned_text = ' '.join( filtered_tokens )
		
		return cleaned_text
	
	
	def convert_file( self, input: str, output: str ):
		'''

			Reads the unprocessed 'input' file and writes a cleaned 'output' file

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
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'convert( self, input: str, output: str )'
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
				raise Exception( 'The input parameter "text" is required.' )
			else:
				# Step 1: Normalize normalized
				self.normal = text.replace( '\r\n', '\n' ).replace( '\r', '\n' )
				
				# Step 2: Remove page headers and footers (Public Law-specific)
				self.headers = re.sub( r'PUBLIC.*?\n', '', self.normal, flags=re.IGNORECASE )
				self.footers = re.sub( r'\n\s*\d+\s*\n', '\n', self.headers )
				
				# Step 3: Remove hyphenation at line breaks (e.g., "appropria-\ntion")
				self.hyphens = re.sub( r'(\w+)-\n(\w+)', r'\1\2', self.footers )
				
				# Step 4: Merge broken lines where sentence continues
				self.merge = re.sub( r'(?<!\n)\n(?![\n])', ' ', self.hyphens )
				
				# Step 5: Collapse excessive whitespace
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
				_msg = 'The input parameter "line" is None'
				raise Exception( _msg )
			else:
				# Step 1: Normalize normalized
				self.normalized = line.replace( '\r\n', '\n' ).replace( '\r', '\n' )
				
				# Step 2: Remove page headers and footers (Public Law-specific)
				self.headers = re.sub( r'PUBLIC.*?\n', '', self.normalized, flags=re.IGNORECASE )
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
				raise Exception( 'The input parameter "text" is required' )
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
			_exc.module = 'Tiggr'
			_exc.cause = 'Text'
			_exc.method = 'clean_string( self, documents: str, stem="None" ) -> str'
			_err = ErrorDialog( _exc )
			_err.show( )
	
	
	def tokenize( self, line: str ) -> list:
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
			if line is None:
				raise Exception( 'The input parameter "line" was None' )
			else:
				words = line.split( )
				tokens = [ re.sub( r'[^\w"-]', '', word ) for word in words if word.strip( ) ]
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
				raise Exception( 'The input parameter "html" was None' )
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
		"""Initialize the PreTokenizer with a specific OpenAI model.

		Args:
			model (str): The name of the OpenAI model used for tokenization.
		"""
		self.model = model
		self.encoding = tiktoken.encoding_for_model( model )
	
	
	def load_file( self, path: str ) -> str:
		"""
			Load the content of a documents file.
	
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
