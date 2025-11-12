'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                Config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

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
    config.py
  </summary>
  ******************************************************************************************
  '''
import os

# Constants
APPLICATION_WIDTH = 750
THEME = "DarkGray12"
OUTPUT_FILE_NAME = "boo.wav"
SAMPLE_RATE = 48000
MODELS = [ 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo' ]
DEFAULT_MODEL = MODELS[ 0 ]
SQLALCHEMY_DATABASE_URI = r'C:\Users\terry\source\repos\Boo\stores\sqlite\datamodels\Data.db'

# Keys
GEOAPIFY_API_KEY = os.getenv( 'GEOAPIFY_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GROQ_API_KEY = os.getenv( 'GROQ_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
HUGGINGFACE_API_KEY = os.getenv( 'HUGGINGFACE_API_KEY' )
IPINFO_API_KEY = os.getenv( 'IPINFO_API_KEY' )
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
PINECONE_API_KEY = os.getenv( 'PINECONE_API_KEY' )
LANGSMITH_API_KEY = os.getenv( 'LANGSMITH_API_KEY' )
LLAMAINDEX_API_KEY = os.getenv( 'LLAMAINDEX_API_KEY' )
LLAMACLOUD_API_KEY = os.getenv( 'LLAMACLOUD_API_KEY' )
MISTRAL_API_KEY = os.getenv( 'MISTRAL_API_KEY' )
NASA_API_KEY = os.getenv( 'NASA_API_KEY' )
NEWSAPI_API_KEY = os.getenv( 'NEWSAPI_API_KEY' )
WEATHERAPI_API_KEY = os.getenv( 'WEATHERAPI_API_KEY' )
WEAVIEATE_API_KEY = os.getenv( 'WEAVIEATE_API_KEY' )
QDRANT_API_KEY = os.getenv( 'QDRANT_API_KEY' )
SINGLESTORE_API_KEY = os.getenv( 'SINGLESTORE_API_KEY' )
SECRET_KEY = os.urandom( 32 )

def set_environment( ):
	"""
		
		Purpose:
		--------
		Gets availible environment vaariables for configuration
		
		
	"""
	variable_dict = globals( ).items( )
	for key, value in variable_dict:
		if 'API' in key or 'ID' in key:
			os.environ[ key ] = value

