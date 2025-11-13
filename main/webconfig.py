'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                Config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="webconfig.py" company="Terry D. Eppler">

	     Boo is a df analysis tool integrating GenAI, GptText Processing, and Machine-Learning
	     algorithms for federal analysts.
	     Copyright ©  2022  Terry Eppler

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
    webconfig.py
  </summary>
  ******************************************************************************************
  '''
import os
from typing import Optional, List, Dict

class WebConfig:
	# Keys
	SECRET_KEY: Optional[ bytes ]
	MAIL_SERVER: Optional[ str ]
	MAIL_PORT: Optional[ str ]
	MAIL_USE_TLS : Optional[ str ]
	MAIL_USERNAME: Optional[ str ]
	MAIL_PASSWORD: Optional[ str ]
	FLASKY_MAIL_SUBJECT_PREFIX: Optional[ str ]
	FLASKY_MAIL_SENDER: Optional[ str ]
	FLASKY_ADMIN: Optional[ str ]
	SQLALCHEMY_TRACK_MODIFICATIONS: Optional[ int ]
	FalseAPPLICATION_WIDTH: Optional[ int ]
	THEME: Optional[ str ]
	OUTPUT_FILE_NAME: Optional[ str ]
	SAMPLE_RATE: Optional[ int ]
	MODELS: Optional[ List[ str ] ]
	DEFAULT_MODEL: Optional[ str ]
	SQLALCHEMY_DATABASE_URI: Optional[ str ]

	def __int__( self ):
		self.SECRET_KEY = os.urandom( 32 )
		self.MAIL_SERVER = os.environ.get( 'MAIL_SERVER', 'smtp.googlemail.com' )
		self.MAIL_PORT = int( os.environ.get( 'MAIL_PORT', '587' ) )
		self.MAIL_USE_TLS = os.environ.get( 'MAIL_USE_TLS', 'true' ).lower( ) in [ 'true', 'on', '1' ]
		self.MAIL_USERNAME = os.environ.get( 'MAIL_USERNAME' )
		self.MAIL_PASSWORD = os.environ.get( 'MAIL_PASSWORD' )
		self.FLASKY_MAIL_SUBJECT_PREFIX = '[Flasky]'
		self.FLASKY_MAIL_SENDER = 'Flasky Admin <flasky@example.com>'
		self.FLASKY_ADMIN = os.environ.get( 'FLASKY_ADMIN' )
		self.FalseAPPLICATION_WIDTH = 750
		self.SQLALCHEMY_TRACK_MODIFICATIONS = 750
		self.THEME = "DarkGray12"
		self.OUTPUT_FILE_NAME = "boo.wav"
		self.SAMPLE_RATE = 48000
		self.MODELS = [ 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo' ]
		self.DEFAULT_MODEL = self.MODELS[ 0 ]
		self.SQLALCHEMY_DATABASE_URI = r'C:\Users\terry\source\repos\Boo\stores\sqlite\datamodels\Data.db'

class DevelopmentConfig( WebConfig ):
	DEBUG: Optional[ bool ]
	SQLALCHEMY_DATABASE_URI: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.DEBUG = True
		self.SQLALCHEMY_DATABASE_URI = os.environ.get( 'DEV_DATABASE_URL' )

class TestingConfig( WebConfig ):
	TESTING: Optional[ bool ]
	SQLALCHEMY_DATABASE_URI: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.TESTING = True
		self.SQLALCHEMY_DATABASE_URI = os.environ.get( 'TEST_DATABASE_URL' )
 
class ProductionConfig( WebConfig ):
	SQLALCHEMY_DATABASE_URI: Optional[ str ]
	config: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.SQLALCHEMY_DATABASE_URI = os.environ.get( 'DATABASE_URL' )
		self.config = \
		{
			 'development': DevelopmentConfig,
			 'testing': TestingConfig,
			 'production': ProductionConfig,
			 'default': DevelopmentConfig
		 }