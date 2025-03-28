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
import spacy
from Booger import Error, ErrorDialog

class Text:
    '''
        Class providing text preprocessing functionality
        @params: tokenize: bool
    '''
    def __init__( self, tokenize: bool=False ):
        self.tokenize = tokenize


    def convert( self, input: str, output: str ):
        try:
            if not os.path.exists( input ):
                raise FileNotFoundError( f'Input file not found: {input}' )
            
            if output is None:
                raise FileNotFoundError( f'Output file not found: {output}' )
    
            cleaned_lines = []
            with open( input, 'r', encoding='utf-8' ) as infile:
                for line in infile:
                    cleaned = self.clean_line( line )
                    if cleaned:
                        cleaned_lines.append( cleaned )
    
            with open( output, 'w', encoding='utf-8' ) as outfile:
                for cleaned_line in cleaned_lines:
                    if self.tokenize:
                        tokens = self.tokenize( cleaned_line )
                        json_obj = {'tokens': tokens}
                    else:
                        json_obj = {'line': cleaned_line}
                    json_str = json.dumps( json_obj, ensure_ascii=False )
                    outfile.write( json_str + '\n' )
        except Exception as e:
            _exc = Error( e )
            _exc.module = 'Tidy'
            _exc.cause = 'Text'
            _exc.method = 'convert( self, input: str, output: str )'
            _err = ErrorDialog( _exc )
            _err.show( )


    def clean_line( self, line: str ) -> str:
        '''
            Method that cleans a line of text given as input
            :param line: str
        '''
        try:
            if not line.strip( ):
                return ''
    
            line = line.strip( )
            line = re.sub( r'<.*?>', '', line )
            line = line.lower( )
            line = re.sub( r'[^\x20-\x7E]', '', line )
            line = re.sub( r'\d', '', line )
            line = re.sub( r'[^a-zA-Z\s\.,!\?"\-]', '', line )
            line = re.sub( r'\s+', ' ', line )
            return line.strip( )
        except Exception as e:
            _exc = Error( e )
            _exc.module = 'Tidy'
            _exc.cause = 'Text'
            _exc.method = 'clean_line( self, line: str ) -> str'
            _err = ErrorDialog( _exc )
            _err.show( )


    def tokenize( self, cleaned_line: str ) -> list:
        '''
        Method that tokenizes a line of text given as input
        :param cleaned_line: str
        :return: list
        '''
        try:
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


    def clean_html( self,  html: str ) -> list:
        '''
        Method that cleans html given as input
        :param html: str
        :return: list
        '''
        try:
            soup = BeautifulSoup( html, 'html.parser' )
            for data in soup( [ 'style', 'script', 'code', 'a' ] ):
                data.decompose( )
        
            return ' '.join( soup.stripped_strings )
        except Exception as e:
            _exc = Error( e )
            _exc.module = 'Tidy'
            _exc.cause = 'Text'
            _exc.method = 'clean_html( self,  html: str )'
            _err = ErrorDialog( _exc )
            _err.show( )
    
    
    def clean_string( self, text: str, stem='None' ) -> str:
        try:
            final_string = ''
            text = text.lower( )
            text = re.sub( r'\n', '', text )
            translator = str.maketrans( '', '', string.punctuation )
            text = text.translate( translator )
            text = text.split( )
            useless_words = nltk.corpus.stopwords.words( 'english' )
            useless_words = useless_words + [ 'hi', 'im' ]
            text_filtered = [ word for word in text if not word in useless_words ]
            text_filtered = [ re.sub( r'\w*\d\w*', '', w ) for w in text_filtered ]
            
            if stem == 'Stem':
                stemmer = PorterStemmer( )
                text_stemmed = [ stemmer.stem( y ) for y in text_filtered ]
            elif stem == 'Lem':
                lem = WordNetLemmatizer( )
                text_stemmed = [ lem.lemmatize( y ) for y in text_filtered ]
            elif stem == 'Spacy':
                text_filtered = nlp( ' '.join( text_filtered ) )
                text_stemmed = [ y.lemma_ for y in text_filtered ]
            else:
                text_stemmed = text_filtered
            
            final_string = ' '.join( text_stemmed )
            
            return final_string
        except Exception as e:
            _exc = Error( e )
            _exc.module = 'Tidy'
            _exc.cause = 'Text'
            _exc.method = 'clean_string( self, text: str, stem="None" ) -> str)'
            _err = ErrorDialog( _exc )
            _err.show( )