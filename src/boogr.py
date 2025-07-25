'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                Boogr.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="boogr.py" company="Terry D. Eppler">

	     Boo is a df analysis tool integrating GenAI, Text Processing, and Machine-Learning
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
    Boogr.py
  </summary>
  ******************************************************************************************
  '''
import base64
from enum import Enum
import FreeSimpleGUI as sg
import fitz
from googlesearch import search
import random
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.figure
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
import numpy as np
import os
from pandas import read_csv as CsvReader
from pandas import read_excel as ExcelReader
from PIL import Image, ImageTk, ImageSequence
from src.static import EXT, Client
from sys import exit, exc_info
from src.minion import App
import traceback
import urllib.request
import webbrowser
from typing import Dict, List, Tuple


class Error( Exception ):
	'''

        Purpose:
        ---------
		Class wrapping error used as the path argument for ErrorDialog class

        Constructor:
		----------
        Error( error: Exception, heading: str=None, cause: str=None,
                method: str=None, module: str=None )

    '''
	
	
	def __init__( self, error: Exception, heading: str = None, cause: str = None,
	              method: str = None, module: str = None ):
		super( ).__init__( )
		self.heading = heading
		self.cause = cause
		self.method = method
		self.module = module
		self.type = exc_info( )[ 0 ]
		self.trace = traceback.format_exc( )
		self.info = str( exc_info( )[ 0 ] ) + ': \r\n \r\n' + traceback.format_exc( )
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.info is not None:
			return self.info
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'message', 'cause',
		         'method', 'module', 'scaler',
		         'stack_trace', 'info' ]


class ButtonIcon( ):
	'''

        Constructor:
        -----------
		ButtonIcon( png: Enum )

        Pupose:
		---------
		Class representing form images

    '''
	
	
	def __init__( self, png: Enum ):
		self.name = png.name
		self.button = r'C:\Users\terry\source\repos\Boo\resources\img\button'
		self.file_path = self.button + r'\\' + self.name + '.png'
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		return self.file_path
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'button', 'name', 'file_path' ]


class TitleIcon( ):
	'''

	    Construcotr:
	    -----------
		TitleIcon( ico )

	    Purpose:
		--------
		Class used to define the TitleIcon used on the GUI

	'''
	
	
	def __init__( self, ico ):
		self.name = ico.name
		self.folder = r'C:\Users\terry\source\repos\Boo\resources\ico'
		self.file_path = self.folder + r'\\' + self.name + r'.ico'
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.file_path is not None:
			return self.file_path
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'folder', 'name', 'authority_filepath' ]


class Dark( ):
	'''

        Constructor:
		-----------
        Dark( )

        Pupose:
		-------
		Class representing the theme

    '''

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		self.form_size = (400, 200)
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color' ]


class FileDialog( Dark ):
	'''

	    Construcotr: 
	    ------------
	    FileDialog( )

	    Purpose: 
	    -------
	    Class that creates dialog to get path

	'''
	
	
	def __init__( self, extension=EXT.XLSX ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11 )
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (500, 240)
		self.selected_item = None
		self.message = 'Grab File'
		self.extension = extension
		self.excel = (('Excel', '*.xlsx'),)
		self.csv = (('CSV', '*.csv'),)
		self.pdf = (('PDF', '*.pdf'),)
		self.sql = (('SQL', '*.sql',),)
		self.text = (('Text', '*.txt'),)
		self.access = (('Access', '*.accdb'),)
		self.sqlite = (('SQLite', '*.db'),)
		self.sqlserver = (('MSSQL', '*.mdf', '*.ldf', '*.sdf'),)
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.selected_item is not None:
			return self.selected_item
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'original', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'original', 'selected_item', 'show',
		         'message', 'extension', 'excel', 'csv', 'pdf', 'sql',
		         'pages', 'access', 'sqlite', 'sqlserver' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( ) ],
			            [ sg.Text( self.message, font=('Roboto', 11) ) ],
			            [ sg.Text( ) ],
			            [ sg.Input( key='-PATH-' ), sg.FileBrowse( size=(15, 1) ) ],
			            [ sg.Text( ) ],
			            [ sg.Text( ) ],
			            [ sg.OK( size=(8, 1), ), sg.Cancel( size=(10, 1) ) ] ]
			
			_window = sg.Window( ' Booger', _layout,
				font=self.theme_font,
				size=self.form_size,
				icon=self.icon_path )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Cancel'):
					break
				elif _event == 'OK':
					self.selected_item = _values[ '-PATH-' ]
					_window.close( )
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'FileDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class FolderDialog( Dark ):
	'''

		Purpose:
		----------
		Class defining dialog used to select a directory url

		Construcotr:
		-----------
		FolderDialog( )

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (500, 250)
		self.selected_item = None
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if isinstance( self.selected_item, str ):
			return self.selected_item
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		return [ 'form_size', 'settings_path', 'original', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'original', 'selected_item', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( ) ],
			            [ sg.Text( 'Search for Directory' ) ],
			            [ sg.Text( ) ],
			            [ sg.Input( key='-PATH-' ), sg.FolderBrowse( size=(15, 1) ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.OK( size=(8, 1) ), sg.Cancel( size=(10, 1) ) ] ]
			
			_window = sg.Window( '  Booger', _layout,
				font=self.theme_font,
				size=self.form_size,
				icon=self.icon_path )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Cancel'):
					break
				elif _event == 'OK':
					self.selected_item = _values[ '-PATH-' ]
					sg.popup_ok( self.selected_item,
						title='Results',
						icon=self.icon_path,
						font=self.theme_font )
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.cause = 'FolderDialog'
			exception.method = 'show( self )'
			_error = ErrorDialog( exception )
			_error.show( )


class SaveFileDialog( Dark ):
	'''

	    Constructor:
	    ---------------
	    SaveFileDialog( url = '' ):

        Purpose:
        --------
        Class define object that provides a dialog to locate file destinations

    '''
	
	
	def __init__( self, path='' ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		self.file_name = None
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'\resources\theme' )
		self.form_size = (400, 250)
		self.original = path
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.file_name is not None:
			return self.file_name
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'original', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'original', 'file_name', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_username = os.environ.get( 'USERNAME' )
			_filename = sg.popup_get_file( 'Select Location / Enter File Name',
				title='  Booger',
				font=self.theme_font,
				icon=self.icon_path,
				save_as=True )
			
			self.file_name = _filename
			
			if os.path.exists( self.original ):
				_src = io.open( self.original ).read( )
				_dest = io.open( _filename, 'w+' ).write( _src )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'SaveFileDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class GoogleDialog( Dark ):
	'''

	    Constructor:
	    -----------
	    GoogleDialog(  )

	    Purpose:
	    --------
	    Class that renames a folder


	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		self.results = None
		self.querytext = None
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = ( 500, 235 )
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\app\web\google.png'
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			Returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if isinstance( self.results, list ):
			return self.results[ 0 ]
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'original', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'image', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( ) ],
			            [ sg.Image( source=self.image ) ],
			            [ sg.Text( size=(10, 1) ),
			              sg.Input( key='-QUERY-', size=(40, 2) ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( size=(10, 1) ), sg.Submit( size=(15, 1) ),
			              sg.Text( size=(5, 1) ), sg.Cancel( size=(15, 1) ) ] ]
			
			_window = sg.Window( '  Booger', _layout,
				icon=self.icon_path,
				font=self.theme_font,
				size=self.form_size )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_X_EVENT, sg.WIN_CLOSED, 'Cancel'):
					break
				elif _event == 'Submit':
					self.querytext = _values[ '-QUERY-' ]
					_google = search( term=self.querytext, num_results=5 )
					_app = App( Client.Edge )
					for result in list( _google ):
						self.results.append( result )
						_app.run_args( result )
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'GoogleDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class EmailDialog( Dark ):
	'''

	    Purpose:
	    --------
	    Class providing form used to send email messages.

	    Construcotr:
	    ------------ 
	    EmailDialog( sender: str=None, receiver: str=None,
			    subject: str=None, heading: str=None )


    '''
	
	
	def __init__( self, sender: str = None, receiver: list[ str ] = None,
	              subject: str = None, message: list[ str ] = None ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\app\web\outlook.png'
		self.form_size = (570, 550)
		self.sender = sender
		self.receiver = receiver
		self.subject = subject
		self.message = message
	
	
	def __str__( self ) -> List[ str ] | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.message is not None:
			return self.message
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'progressbar_color',
		         'sender', 'reciever', 'message',
		         'subject', 'others', 'password',
		         'username', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_btn = (20, 1)
			_input = (35, 1)
			_spc = (5, 1)
			_img = (50, 22)
			_clr = '#69B1EF'
			_layout = [ [ sg.Text( ' ', size=_spc ), ],
			            [ sg.Text( ' ', size=_spc ), ],
			            [ sg.Text( ' ', size=_spc ),
			              sg.Text( 'From:', size=_btn, text_color=_clr ),
			              sg.Input( key='-EMAIL FROM-', size=_input ) ],
			            [ sg.Text( ' ', size=_spc ), sg.Text( 'To:', size=_btn,
				            text_color=_clr ),
			              sg.Input( key='-EMAIL TO-', size=_input ) ],
			            [ sg.Text( ' ', size=_spc ),
			              sg.Text( 'Subject:', size=_btn, text_color=_clr ),
			              sg.Input( key='-EMAIL SUBJECT-', size=_input ) ],
			            [ sg.Text( ' ', size=_spc ), sg.Text( ) ],
			            [ sg.Text( ' ', size=_spc ),
			              sg.Text( 'Username:', size=_btn, text_color=_clr ),
			              sg.Input( key='-USER-', size=_input ) ],
			            [ sg.Text( ' ', size=_spc ),
			              sg.Text( 'Password:', size=_btn, text_color=_clr ),
			              sg.Input( password_char='*', key='-PASSWORD-', size=_input ) ],
			            [ sg.Text( ' ', size=_spc ) ],
			            [ sg.Text( ' ', size=_spc ),
			              sg.Multiline( 'Type your message here', size=(65, 10),
				              key='-EMAIL TEXT-' ) ],
			            [ sg.Text( ' ', size=(100, 1) ) ],
			            [ sg.Text( ' ', size=_spc ), sg.Button( 'Send', size=_btn ),
			              sg.Text( ' ', size=_btn ), sg.Button( 'Cancel', size=_btn ) ] ]
			
			_window = sg.Window( '  Send Message', _layout,
				icon=self.icon_path,
				size=self.form_size )
			
			while True:  # Event Loop
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, 'Cancel', 'Exit'):
					break
				if _event == 'Send':
					sg.popup_quick_message( 'Sending...this will take a moment...',
						background_color='red' )
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'EmailDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class MessageDialog( Dark ):
	'''

	    Purpose:
	    ---------
	    Class that provides form used to display informational messages

	    Construcotr:  MessageDialog( documents = '' )

    '''
	
	# Fields
	text: str = None
	
	
	def __init__( self, text: str = None ):
		self.text = text
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (450, 250)
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.text is not None:
			return self.text
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'original', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'image', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_txtsz = (100, 1)
			_btnsz = (10, 1)
			_layout = [ [ sg.Text( size=_txtsz ) ],
			            [ sg.Text( size=_txtsz ) ],
			            [ sg.Text( size=(5, 1) ),
			              sg.Text( self.text,
				              font=('Roboto', 11),
				              enable_events=True,
				              key='-TEXT-',
				              text_color='#69B1EF',
				              size=(80, 1) ) ],
			            [ sg.Text( size=_txtsz ) ],
			            [ sg.Text( size=_txtsz ) ],
			            [ sg.Text( size=_txtsz ) ],
			            [ sg.Text( size=(5, 1) ), sg.Ok( size=_btnsz ),
			              sg.Text( size=(15, 1) ), sg.Cancel( size=_btnsz ) ] ]
			
			_window = sg.Window( r'  Booger', _layout,
				icon=self.icon_path,
				font=self.theme_font,
				size=self.form_size )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Ok', 'Cancel'):
					break
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'MessageDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ErrorDialog( Dark ):
	'''

	    Construcotr:  ErrorDialog( error )

	    Purpose:  Class that displays excetption target_values that accepts
            a single, optional argument 'error' of scaler Error

    '''
	
	# Fields
	error: Exception = None
	heading: str = None
	module: str = None
	info: str = None
	cause: str = None
	method: str = None
	
	
	def __init__( self, error: Error ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = ( 500, 300 )
		self.error = error
		self.heading = error.heading
		self.module = error.module
		self.info = error.trace
		self.cause = error.cause
		self.method = error.method
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if isinstance( self.info, str ):
			return self.info
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'progressbar_color',
		         'info', 'cause', 'method', 'error', 'heading'
		                                             'module', 'scaler', 'message' 'show' ]
	
	
	def show( self ) -> object:
		'''

            Purpose:
            --------
            

            Parameters:
            ----------
            

            Returns:
            ---------
            

		'''
		_msg = self.heading if isinstance( self.heading, str ) else None
		_info = f'Module:\t{self.module}\r\nClass:\t{self.cause}\r\n' \
		        f'Method:\t{self.method}\r\n \r\n{self.info}'
		_red = '#F70202'
		_font = ('Roboto', 10)
		_padsz = (3, 3)
		_layout = [ [ sg.Text( ) ],
		            [ sg.Text( f'{_msg}', size=(100, 1), key='-MSG-', text_color=_red,
			            font=_font ) ],
		            [ sg.Text( size=(150, 1) ) ],
		            [ sg.Multiline( f'{_info}', key='-INFO-', size=(80, 7), pad=_padsz ) ],
		            [ sg.Text( ) ],
		            [ sg.Text( size=(20, 1) ), sg.Cancel( size=(15, 1), key='-CANCEL-' ),
		              sg.Text( size=(10, 1) ), sg.Ok( size=(15, 1), key='-OK-' ) ] ]
		
		_window = sg.Window( r' Booger', _layout,
			icon=self.icon_path,
			font=self.theme_font,
			size=self.form_size )
		
		while True:
			_event, _values = _window.read( )
			if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Canel', '-OK-'):
				break
		
		_window.close( )


class InputDialog( Dark ):
	'''

	    Construcotr:  Input( prompt )

	    Purpose:  class that produces a contact path form

	'''
	# Fields
	theme_background: str = None
	response: str = None
	
	
	def __init__( self, question: str = None ):
		super( ).__init__( )
		self.question = question
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = ( 500, 250 )
		self.selected_item = None
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if isinstance( self.response, str ):
			return self.response
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( ) ],
			            [ sg.Text( self.question, font=('Roboto', 9, 'bold') ) ],
			            [ sg.Text( ) ],
			            [ sg.Text( 'Enter:', size=(10, 2) ),
			              sg.InputText( key='-INPUT-', size=(40, 2) ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( size=(10, 1) ),
			              sg.Submit( size=(15, 1), key='-SUBMIT-' ),
			              sg.Text( size=(5, 1) ),
			              sg.Cancel( size=(15, 1), key='-CANCEL-' ) ] ]
			
			_window = sg.Window( '  Booger', _layout,
				icon=self.icon_path,
				font=self.theme_font,
				size=self.form_size )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_X_EVENT, sg.WIN_CLOSED, '-CANCEL-', 'Exit'):
					break
				
				self.response = _values[ '-INPUT-' ]
				sg.popup( _event, _values, self.response,
					text_color=sg.theme_text_color( ),
					font=self.theme_font,
					icon=self.icon )
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'InputDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ScrollingDialog( Dark ):
	'''

        'Construcotr:

            ScrollingDialog( documents = '' )

        Purpose:

            Provides form for multiline path/cleaned_lines

	'''
	
	
	def __init__( self, text='' ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = ( 700, 600 )
		self.text = text if isinstance( text, str ) and text != '' else None
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if isinstance( self.text, str ):
			return self.text
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_line = (100, 1)
			_space = (5, 1)
			_btnsize = (25, 1)
			_arrow = self.arrowcolor
			_back = super( ).button_backcolor
			_padsz = (3, 3, 3, 3)
			_layout = [ [ sg.Text( ' ', size=_line ) ],
			            [ sg.Text( ' ', size=_line ) ],
			            [ sg.Text( size=_space ),
			              sg.Multiline( size=(70, 20), key='-TEXT-', pad=_padsz ),
			              sg.Text( size=_space ) ],
			            [ sg.Text( ' ', size=_line ) ],
			            [ sg.Text( ' ', size=_space ), sg.Input( k='-IN-', size=(70, 20) ),
			              sg.Text( size=_space ) ],
			            [ sg.Text( ' ', size=_line ) ],
			            [ sg.Text( size=_space ), sg.Button( 'Submit', size=_btnsize ),
			              sg.Text( size=(15, 1) ), sg.Button( 'Exit', size=_btnsize ),
			              sg.Text( size=_space ), ] ]
			
			_window = sg.Window( '  Booger', _layout,
				icon=self.icon_path,
				size=self.form_size,
				font=self.theme_font,
				resizable=True )
			
			while True:
				event, values = _window.read( )
				self.text = values[ '-TEXT-' ]
				if event in (sg.WIN_CLOSED, 'Exit'):
					break
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ScrollingDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ContactForm( Dark ):
	'''

        Construcotr: ContactForm( contact )

        Purpose:  class that produces a contact path form

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\app\web\outlook.png'
		self.form_size = (500, 300)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( r'Enter Contact Details' ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( 'Name', size=(10, 1) ),
			              sg.InputText( '1', size=(80, 1), key='-NAME-' ) ],
			            [ sg.Text( 'Address', size=(10, 1) ),
			              sg.InputText( '2', size=(80, 1), key='-ADDRESS-' ) ],
			            [ sg.Text( 'Phone', size=(10, 1) ),
			              sg.InputText( '3', size=(80, 1), key='-PHONE-' ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( size=(100, 1) ) ],
			            [ sg.Text( size=(10, 1) ), sg.Submit( size=(10, 1) ),
			              sg.Text( size=(20, 1) ), sg.Cancel( size=(10, 1) ) ] ]
			
			_window = sg.Window( '  Booger', _layout,
				icon=self.icon_path,
				font=self.theme_font,
				size=self.form_size )
			
			while True:
				_event, _values = _window.read( )
				sg.popup( 'Results', _values, _values[ '-NAME-' ],
					_values[ '-ADDRESS-' ],
					_values[ '-PHONE-' ],
					text_color=self.theme_textcolor,
					font=self.theme_font,
					icon=self.icon )
				
				if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Cancel'):
					break
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ContactForm'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class GridForm( Dark ):
	'''

        Construcotr: GridForm( )
        
        Purpose:  object providing form that simulates a datagrid

	'''
	
	
	def __init__( self, rows=30, columns=10 ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.width = (17, 1)
		self.rows = rows
		self.columns = columns
		self.form_size = (1250, 650)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'theme_background', 'theme_textcolor',
		         'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'progressbar_color',
		         'field_width', 'rows', 'columns', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_black = self.theme_background
			_columns = self.columns
			_headings = [ f'HEADER-{i + 1}' for i in range( _columns ) ]
			_space = [ [ sg.Text( size=(10, 1) ) ], [ sg.Text( size=(10, 1) ) ],
			           [ sg.Text( size=(10, 1) ) ] ]
			_header = [
				[ sg.Text( h, size=(16, 1), justification='left' ) for h in _headings ] ]
			_records = [ [ [ sg.Input( size=self.width, pad=(0, 0), font=self.theme_font )
			                 for c in range( len( _headings ) ) ] for r in range( self.rows )
			               ], ]
			_buttons = [ [ sg.Text( size=(35, 1) ), sg.Text( size=(10, 1) ), ],
			             [ sg.Text( size=(100, 1) ), sg.Text( size=(100, 1) ),
			               sg.Ok( size=(35, 2) ) ],
			             [ sg.Sizegrip( background_color=_black ) ] ]
			# noinspection PyTypeChecker
			_layout = _space + _header + _records + _buttons
			
			_window = sg.Window( '  Booger', _layout,
				finalize=True,
				size=self.form_size,
				icon=self.icon_path,
				font=self.theme_font,
				resizable=True )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, '-CANCEL-'):
					break
				
				_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'GridForm'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class LoadingPanel( Dark ):
	'''

        Construcotr:  LoadingPanel( )
        
        Purpose:  object providing form loading behavior

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\loaders\loading.gif'
		self.form_size = (800, 600)
		self.timeout = 6000
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text(
				background_color='#000000',
				text_color='#FFF000',
				justification='c',
				key='-T-',
				font=('Bodoni MT', 40) ) ], [ sg.Image( key='-IMAGE-' ) ] ]
			
			_window = sg.Window( '  Loading...', _layout,
				icon=self.icon_path,
				element_justification='c',
				margins=(0, 0),
				size=(800, 600),
				element_padding=(0, 0), finalize=True )
			
			_window[ '-T-' ].expand( True, True )
			_interframe_duration = Image.open( self.image ).info[ 'duration' ]
			
			while True:
				for frame in ImageSequence.Iterator( Image.open( self.image ) ):
					_event, _values = _window.read( timeout=_interframe_duration )
					if _event == sg.WIN_CLOSED or _event == sg.WIN_X_EVENT:
						exit( 0 )
					_window[ '-IMAGE-' ].update( data=ImageTk.PhotoImage( frame ) )
					_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'LoadingPanel'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class WaitingPanel( Dark ):
	'''

        Construcotr:  WaitingPanel( )
        
        Purpose:  object providing form loader behavior

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\loaders\loader.gif'
		self.theme_font = ('Roboto', 11)
		self.form_size = (800, 600)
		self.timeout = 6000
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]


	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text(
				background_color='#000000',
				text_color='#FFF000',
				justification='c',
				key='-T-',
				font=('Bodoni MT', 40) ) ], [ sg.Image( key='-IMAGE-' ) ] ]
			
			_window = sg.Window( '  Waiting...', _layout,
				icon=self.icon_path,
				element_justification='c',
				margins=(0, 0),
				element_padding=(0, 0),
				size=(800, 600),
				finalize=True )
			
			_window[ '-T-' ].expand( True, True )
			_interframe_duration = Image.open( self.image ).info[ 'duration' ]
			
			while True:
				for frame in ImageSequence.Iterator( Image.open( self.image ) ):
					_event, _values = _window.read( timeout=_interframe_duration )
					if _event == sg.WIN_CLOSED:
						exit( 0 )
					_window[ '-IMAGE-' ].update( data=ImageTk.PhotoImage( frame ) )
					_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'WaitingPanel'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ProcessingPanel( Dark ):
	'''

        Construcotr:  ProcessingPanel( )

        Purpose:  object providing form processing behavior

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\loaders\processing.gif'
		self.form_size = (800, 600)
		self.timeout = None
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text(
				background_color='#000000',
				text_color='#FFF000',
				justification='c',
				key='-T-',
				font=('Bodoni MT', 40) ) ], [ sg.Image( key='-IMAGE-' ) ] ]
			
			_window = sg.Window( '  Processing...', _layout,
				element_justification='c',
				icon=self.icon_path,
				margins=(0, 0),
				size=(800, 600),
				element_padding=(0, 0),
				finalize=True )
			
			_window[ '-T-' ].expand( True, True )
			
			_interframe_duration = Image.open( self.image ).info[ 'duration' ]
			self.timeout = _interframe_duration
			
			while True:
				for frame in ImageSequence.Iterator( Image.open( self.image ) ):
					_event, _values = _window.read( timeout=self.timeout,
						timeout_key='-TIMEOUT-' )
					if _event == sg.WIN_CLOSED or _event == sg.WIN_X_EVENT:
						exit( 0 )
					
					_window[ '-IMAGE-' ].update( data=ImageTk.PhotoImage( frame ) )
					_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ProcessingPanel'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class SplashPanel( Dark ):
	'''

        Construcotr:  SplashPanel( )

        Purpose:  Class providing splash dialog behavior

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\BudgetEx.png'
		self.form_size = (800, 600)
		self.timeout = 6000
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_img = self.image
			_imgsize = (500, 400)
			_line = (100, 2)
			_space = (15, 1)
			_layout = [ [ sg.Text( size=_space ), sg.Text( size=_line ) ],
			            [ sg.Text( size=_space ), sg.Text( size=_line ) ],
			            [ sg.Text( size=_space ),
			              sg.Image( filename=self.image, size=_imgsize ) ] ]
			_window = sg.Window( '  Booger', _layout,
				no_titlebar=True,
				keep_on_top=True,
				grab_anywhere=True,
				size=self.form_size )
			while True:
				_event, _values = _window.read( timeout=self.timeout, close=True )
				if _event in (sg.WIN_CLOSED, 'Exit'):
					break
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'SplashPanel'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class Notification( Dark ):
	'''

        Purpose:
        ----------
        object providing form processing behavior

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (800, 600)
		self.success = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAA3NCSVQICAjb4U' \
		               b'/gAAAACXBIWXMAAAEKAAABCgEWpLzLAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5r' \
		               b'c2NhcGUub3Jnm+48GgAAAHJQTFRF////ZsxmbbZJYL9gZrtVar9VZsJcbMRYaM' \
		               b'ZVasFYaL9XbMFbasRZaMFZacRXa8NYasFaasJaasFZasJaasNZasNYasJYasJZ' \
		               b'asJZasJZasJZasJZasJYasJZasJZasJZasJZasJaasJZasJZasJZasJZ2IAizQ' \
		               b'AAACV0Uk5TAAUHCA8YGRobHSwtPEJJUVtghJeYrbDByNjZ2tvj6vLz9fb3/CyrN0oAAA' \
		               b'DnSURBVDjLjZPbWoUgFIQnbNPBIgNKiwwo5v1fsQvMvUXI5oqPf4DFOgCrhLKjC8GNV' \
		               b'gnsJY3nKm9kgTsduVHU3SU/TdxpOp15P7OiuV/PVzk5L3d0ExuachyaTWkAkLFtiBKAq' \
		               b'ZHPh/yuAYSv8R7XE0l6AVXnwBNJUsE2+GMOzWL8k3OEW7a/q5wOIS9e7t5qnGExvF5Bvl' \
		               b'c4w/LEM4Abt+d0S5BpAHD7seMcf7+ZHfclp10TlYZc2y2nOqc6OwruxUWx0rDjNJtyp6' \
		               b'HkUW4bJn0VWdf/a7nDpj1u++PBOR694+Ftj/8PKNdnDLn/V8YAAAAASUVORK5CYII='
		self.fail = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAA3NCSVQICAjb4U' \
		            b'/gAAAACXBIWXMAAADlAAAA5QGP5Zs8AAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm' \
		            b'+48GgAAAIpQTFRF////20lt30Bg30pg4FJc409g4FBe4E9f4U9f4U9g4U9f4E9g31Bf4E9f4E9f' \
		            b'4E9f4E9f4E9f4FFh4Vdm4lhn42Bv5GNx5W575nJ' \
		            b'/6HqH6HyI6YCM6YGM6YGN6oaR8Kev9MPI9cb' \
		            b'M9snO9s3R+Nfb+dzg+d/i++vt/O7v/fb3/vj5//z8//7' \
		            b'+////KofnuQAAABF0Uk5TAAcIGBktSY' \
		            b'SXmMHI2uPy8/XVqDFbAAAA8UlEQVQ4y4VT15LCMBBTQkgPYem9d9D' \
		            b'//x4P2I7vILN68kj2WtsAh' \
		            b'yDO8rKuyzyLA3wjSnvi0Eujf3KY9OUP+kno651CvlB0Gr1byQ9UXff' \
		            b'+py5SmRhhIS0oPj4SaUUC' \
		            b'AJHxP9+tLb/ezU0uEYDUsCc+l5' \
		            b'/T8smTIVMgsPXZkvepiMj0Tm5txQLENu7gSF7HIuMreRxYNkb' \
		            b'mHI0u5Hk4PJOXkSMz5I3nyY08HMjbpOFylF5WswdJPmYeVaL28968yNfGZ2r9gvqFalJNUy2UW' \
		            b'mq1Wa7di/3Kxl3tF1671YHRR04dWn3s9cXRV09f3vb1fwPD7z9j1WgeRgAAAABJRU5ErkJggg=='
		self.ninja = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAnCAYAAABuf0pMAAABhWlDQ1BJQ0MgUHJvZmlsZQA' \
		             b'AeJx9kT1Iw0AcxV9bS1WqDnYo4pChOlkQFRFcpIpFsFDaCq06mFz6BU0akhQXR8G14ODHYtXB' \
		             b'xVlXB1dBEPwAcXRyUnSREv+XFFrEeHDcj3f3HnfvAG' \
		             b'+jwhSjaxxQVFNPxWNCNrcqBF7hRz96E' \
		             b'MasyAwtkV7MwHV83cPD17soz3I/9+fok/MGAzwC8RzTdJN4g3h609Q47xOHWEmUic' \
		             b'+Jx3S6I' \
		             b'PEj1yWH3zgXbfbyzJCeSc0Th4iFYgdLHcxKukI8RRyRFZXyvVmHZc5bnJVKjbXuyV8YzKsr' \
		             b'aa7THEYcS0ggCQESaiijAhNRWlVSDKRoP+biH7L9SXJJ5CqDkWMBVSgQbT/4H/zu1ihMTjh' \
		             b'JwRjgf7GsjxEgsAs065b1fWxZzRPA9wxcqW1/tQHMfJJeb2uRI2BgG7i4bmvSHnC5A4SfNF' \
		             b'EXbclH01soAO9n9E05YPAW6F1zemvt4/QByFBXyzfAwSEwWqTsdZd3d3f29u+ZVn8/pE' \
		             b'Fyu/Q7rYsAAAbASURBVHicvZd/bJVXGcc/55z3vvdHuf3BbaFldGyDbQhSJsGNlSC66S' \
		             b'gM/hDYxhJLRIcsbs7IRBONiTEi0RmDJltUthlykegYCT+EyUKZcZBABGSzU34NKpcC7S' \
		             b'1tb2/f3h/v+57jH6Vd6S+gbXyS88853+d5vuf7nuc85xWMhVXWrgbWAAuBU8B24DUS8a5' \
		             b'buYpRJq4Bfg5UDbLaDLxMIr4N4P3tmyLBoB357uZdFWkncP6fJw9lRkUgWF7zW19F13ky' \
		             b'NCRmnKV5sabkaM38ioiBKs/39fZ9Z+Qfj4rf5S9tex7AGklyu/zJZYHcx+ssqwRlleCpK' \
		             b'L6wAZgQ8lk4XbGq5h7KxkfIZvPzUp0ZxhcV0NGZlasWz2hxDu5ueutGLDkSAoHcpbVCO2g' \
		             b'ZxlWFvckBHrrPJxyL8dKvz5DJ5ABwulyuJjs5eOwC44tC79ydPzu5B3/nClTWRkTq0CLI' \
		             b'o2UEgQYMLyyfzhe/MJei4jCHD5+gtfEqUkqUkgSDkt3vNXP6cisLKs8ejSn18i+KS8P' \
		             b'fa2/J3DGBSPbCHKE7bIRizlTBN55bwaxZDyKl4Oy58xw4cJz3/v4fFswIEw7ZHDp6gSMft' \
		             b'HDgfAGfKbdIvH1sabll1QOPAftu+xDGYjGSyaRdGJu5eO1Xl+x66qkVTJ02DcdxOH' \
		             b'GynncP/oMtf7nYiy8JaIqCgsspB+k7eIHxlNiae13FOq/hz1P0paNPNDVuvi0FtNbCGD' \
		             b'PbGLOxufHEJMuySKfT1NW9zxtbd3PoVIrualC9Pm2upM2FymiEq2mQOkdbPsh1YVFsVT7' \
		             b'9nO/th8Zbl2FrW9tdGF7yPO9bnueFHafr3N69e+/XydOUlpfhtLUjlaCwIISlJJ6vSTtZ' \
		             b'XNdn2oyZdF2/wjMb6zEotAxiRC/Jk8C8QRVQSpFMJudms7n1zU3JpzsdR9t2IB4KhTZXL' \
		             b'fhmTnWePL3ha0tFkeuSzuZZ9MTjZJINXEk6VEyIUFx+H/sPvEsm08Uv45fxVHSwNHOAH' \
		             b'w5QoOX69QVdXZmfdKQ6Pt/RmW4BXgVeq573SHMPpqB4+p5IwFv27JLZLP5cFRcbW3lz10' \
		             b'VOJKNUFki+vXwCD02PUXesiZ/taR1O4LabCDQ0/Hd5KtWx08lkEmBeAfF69byHM/29gh' \
		             b'O/NDWQ/fgEVmERQgESX0XJ2hWYO7taNvQS+PBf9YA46DjOW8aYP1Q/+og7nGekdF611J3' \
		             b'7kcEiEPhyHJlg5bDZBLqHoAN8h0R8Sy+BU6c+FEKK0OyqWQN2PJTZ5UsetPz2VwRmmVYF' \
		             b'ZAPlGARg6N9mlM4Q9FpM3irb4cnQ90nEGxiAGoEFK55caXmtO4wM4aoijLDwZLhf8mxL' \
		             b'wE/FtQz9Jn9lT0PftRE1o74mdWamMB7C70TKMDk1bgDGl6Fav3HHXwf1Hy0BLUOHDdKA' \
		             b'RvlpAn4aYfz+sPVD+Y/6EwDYFctqLL/9DV9FJ+Ws2JAwEvEBB3vUCgDkreI6hDJGDPtF5' \
		             b'w82OToClbUhAIGOCe3edQt045gRkJOfLaWytg5oobJ2o+U7VUaANC7K3KzyphfnA6RIx' \
		             b'M+NGQHbu75JYB4DCoAfuCq6ptpNpSf5DqABWFFdyOs/XsTKZQt5Xqf2DRVrRIcwPPHx1a5' \
		             b'VvNWTke4gxufu7HlmG03UKqLCZFBRi/uXzqX8nikEH5ieql2/bda1M/FE/1gjugdygbJ3' \
		             b'gm6L8e2wMAiMUFyxK7hmXPJWCQvcFOdyUTbc+wA76v7NgV8d18DDwAACIy7DgrJH610rNj' \
		             b'NvlfTOKZNDC4sVuascscvwIiGSGQPwdRLxNweLM4oqENdstwlLf9I6tAi0hgx7pnlN1Pg' \
		             b'dPckN8PZQUUZMQMvwTiMsZJ9Tb5AbVnvXUkV2IVNxeqaPkIh3jDmBrD1xixH2cWF8hPG1' \
		             b'1Ll222s/Dd5KVxWyy+ptzYeHizOqq1hOXlVoe6lPeaogLf2ujzwV9QM6rfLW+BttGYC' \
		             b'VJOI7h4oxqm6oL/+pIwvHAILli/Jg7JwVw9Jd9JQoQ9yAvZsYDYG+pnT2b9x48fZJDvD' \
		             b'B/4WAr8b9Pugm6T70pme6mUR82BfWmBHIXd2301WF9QE/jaVzH0njbwVm3spv1C+iHgu' \
		             b'WL1pjdObTvopkfBmqHq70+trYKFD5FSG99vW+jKBlKAysvV3XnlqRQBCwgQDdyki6f/b' \
		             b'kDVx/sobu1mfCpdVfllJszthT0J/8eu0CtpCI778VgUnAhEES3LZFYp99QQj5jFbRcC5' \
		             b'QKrUI9F3+KYn4j4YjAN07D3GzAoqbFRB98Kbf8PsM98bIAVl6HghD2P8Avm6w' \
		             b'ywIVvIgAAAAASUVORK5CYII='
		self.message = '\r\nThe action you have performed \
                          has been successful!'
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.message is not None:
			return self.message
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> int | None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    int | None

		'''
		try:
			return sg.popup_notify( self.message,
				title='Booger Notification',
				icon=self.ninja,
				display_duration_in_ms=10000,
				fade_in_duration=5000,
				alpha=1 )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'Notification'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ImageSizeEncoder( Dark ):
	'''

        Construcotr:
        ------------
        ImageSizeEncoder( )

        Purpose:
        ----------
        Class resizing image and encoding behavior

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (800, 600)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		version = '1.3.1'
		__version__ = version.split( )[ 0 ]
		
		
		def resize( input_file, size, output_file=None, encode_format='PNG' ):
			_image = Image.open( input_file )
			_width, _height = _image.size
			_newwidth, _newheight = size
			if _newwidth != _width or _newheight != _height:
				_scale = min( _newheight / _height, _newwidth / _width )
				_resizedimage = _image.resize( (int( _width * _scale ), int( _height * _scale )),
					Image.ANTIALIAS )
			else:
				_resizedimage = _image
			
			if output_file is not None:
				_resizedimage.save( output_file )
			
			with io.BytesIO( ) as bio:
				_resizedimage.save( bio, format=encode_format )
				_contents = bio.getvalue( )
				_encoded = base64.b64encode( _contents )
			return _encoded
		
		
		def update_outfilename( ):
			_infile = _values[ '-IN-' ]
			if os.path.isfile( _infile ):
				_image = Image.open( _infile )
				_width, _height = _image.size
				_window[ '-ORIG WIDTH-' ].update( _image.size[ 0 ] )
				if not _values[ '-WIDTH-' ]:
					_window[ '-WIDTH-' ].update( _image.size[ 0 ] )
				if not _values[ '-HEIGHT-' ]:
					_window[ '-HEIGHT-' ].update( _image.size[ 1 ] )
				_window[ '-ORIG HEIGHT-' ].update( _image.size[ 1 ] )
				
				_infilename = os.path.basename( _infile )
				_infilenameonly, _infileext = os.path.splitext( _infilename )
				if _values[ '-NEW FORMAT-' ]:
					outfileext = _values[ '-NEW FORMAT-' ].lower( )
					if outfileext == 'jpeg':
						outfileext = 'jpg'
				else:
					outfileext = _infileext[ 1: ]  # strip off the .
				outfile = f'{_infilenameonly}{_width}x{_height}.{outfileext}'
				_outfullname = os.path.join( os.path.dirname( _infile ), outfile )
				
				if _values[ '-DO NOT SAVE-' ]:
					_window[ '-NEW FILENAME-' ].update( '' )
					_window[ '-BASE64-' ].update( True )
				else:
					_window[ '-NEW FILENAME-' ].update( _outfullname )
			else:
				_window[ '-NEW FILENAME-' ].update( '' )
				_window[ '-ORIG WIDTH-' ].update( '' )
				# _window['-WIDTH-'].update('')
				_window[ '-ORIG HEIGHT-' ].update( '' )
				# _window['-HEIGHT-'].update('')
				_window[ '-NEW FILENAME-' ].update( )
		
		
		_formatlist = ('', 'PNG', 'JPEG', 'BMP', 'ICO', 'GIF', 'TIFF')
		_newformat = [
			[ sg.Combo( _formatlist,
				default_value=sg.user_settings_get_entry( '-new format-', '' ),
				readonly=True, enable_events=True, key='-NEW FORMAT-' ) ] ]
		
		_layout = [ [ sg.Text( 'Image Resizer' ) ],
		            [ sg.Frame( 'Input Filename', [
			            [ sg.Input( key='-IN-', enable_events=True, s=80 ),
			              sg.FileBrowse( ), ],
			            [ sg.T( 'Original size' ), sg.T( k='-ORIG WIDTH-' ), sg.T( 'target_values' ),
			              sg.T( k='-ORIG HEIGHT-' ) ] ] ) ],
		            [ sg.Frame( 'Output Filename',
			            [ [ sg.In( k='-NEW FILENAME-', s=80 ), sg.FileBrowse( ), ],
			              [ sg.In( default_text=sg.user_settings_get_entry( '-_width-', '' ),
				              s=4,
				              k='-WIDTH-' ), sg.T( 'target_values' ),
			                sg.In( default_text=sg.user_settings_get_entry( '-_height-', '' ),
				                s=4, k='-HEIGHT-' ) ] ] ) ],
		            [ sg.Frame( 'Convert To New Format', _newformat ) ],
		            [ sg.CBox( 'Encode to Base64 and leave on Clipboard', k='-BASE64-',
			            default=sg.user_settings_get_entry( '-base64-', True ) ) ],
		            [ sg.CBox( 'Do not save file - Only convert and Base64 Encode',
			            k='-DO NOT SAVE-', enable_events=True,
			            default=sg.user_settings_get_entry( '-do not save-', False ) ) ],
		            [ sg.CBox( 'Autoclose Immediately When Done',
			            default=sg.user_settings_get_entry( '-autoclose-',
				            True if sg.running_windows( ) else False ),
			            k='-AUTOCLOSE-' ) ],
		            [ sg.Button( 'Resize', bind_return_key=True ), sg.Button( 'Exit' ) ],
		            [ sg.T(
			            'Note - on some systems, autoclose cannot be used because the clipboard '
			            'is '
			            'cleared by tkinter' ) ],
		            [ sg.T( 'Your settings are automatically saved between runs' ) ],
		            [ sg.T( f'Version {version}' ),
		              sg.T( 'Go to psgresizer GitHub Repo', font='_ 8', enable_events=True,
			              k='-PSGRESIZER-' ),
		              sg.T( 'A PySimpleGUI Application - Go to PySimpleGUI home', font='_ 8',
			              enable_events=True, k='-PYSIMPLEGUI-' ) ],
		            ]
		
		_window = sg.Window( 'Resize Image', _layout,
			icon=self.icon_path,
			right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_LOC_EXIT,
			enable_close_attempted_event=True,
			finalize=True )
		_window[ '-PSGRESIZER-' ].set_cursor( 'hand1' )
		_window[ '-PYSIMPLEGUI-' ].set_cursor( 'hand1' )
		while True:
			_event, _values = _window.read( )
			# print(_event, _values)
			if _event in (sg.WIN_CLOSED, sg.WIN_CLOSE_ATTEMPTED_EVENT, 'Exit'):
				break
			_infile = _values[ '-IN-' ]
			update_outfilename( )
			
			if _event == '-DO NOT SAVE-':
				if _values[ '-DO NOT SAVE-' ]:
					_window[ '-NEW FILENAME-' ].update( '' )
					_window[ '-BASE64-' ].update( True )
			if _event == 'Resize':
				try:
					if os.path.isfile( _infile ):
						update_outfilename( )
						infilename = os.path.basename( _infile )
						infilenameonly, infileext = os.path.splitext( infilename )
						if _values[ '-NEW FORMAT-' ]:
							encode_format = _values[ '-NEW FORMAT-' ].upper( )
						else:
							encode_format = infileext[ 1: ].upper( )  # strip off the .
						if encode_format == 'JPG':
							encode_format = 'JPEG'
						outfullfilename = _values[ '-NEW FILENAME-' ]
						width, height = int( _values[ '-WIDTH-' ] ), int( _values[ '-HEIGHT-' ] )
						if _values[ '-DO NOT SAVE-' ]:
							encoded = resize( input_file=_infile, size=(width, height),
								encode_format=encode_format )
						else:
							encoded = resize( input_file=_infile, size=(width, height),
								output_file=outfullfilename, encode_format=encode_format )
						
						if _values[ '-BASE64-' ]:
							sg.clipboard_set( encoded )
						
						sg.popup_quick_message( 'DONE!', font='_ 40', background_color='red',
							text_color='white' )
				
				except Exception as e:
					sg.popup_error_with_traceback( 'Error resizing or converting',
						'Error encountered during the resize or Base64 encoding', e )
				if _values[ '-AUTOCLOSE-' ]:
					break
			elif _event == 'Version':
				sg.popup_scrolled( sg.get_versions( ), non_blocking=True )
			elif _event == 'Edit Me':
				sg.execute_editor( __file__ )
			elif _event == 'File Location':
				sg.popup_scrolled( 'This Python file is:', __file__ )
			elif _event == '-PYSIMPLEGUI-':
				webbrowser.open_new_tab( r'http://www.PySimpleGUI.com' )
			elif _event == '-PSGRESIZER-':
				webbrowser.open_new_tab( r'https://github.com/PySimpleGUI/psgresizer' )
		
		if _event != sg.WIN_CLOSED:
			sg.user_settings_set_entry( '-autoclose-', _values[ '-AUTOCLOSE-' ] )
			sg.user_settings_set_entry( '-new format-', _values[ '-NEW FORMAT-' ] )
			sg.user_settings_set_entry( '-do not save-', _values[ '-DO NOT SAVE-' ] )
			sg.user_settings_set_entry( '-base64-', _values[ '-BASE64-' ] )
			sg.user_settings_set_entry( '-_width-', _values[ '-WIDTH-' ] )
			sg.user_settings_set_entry( '-_height-', _values[ '-HEIGHT-' ] )
		_window.close( )


class PdfForm( Dark ):
	'''

        Construcotr:
            PdfForm( )

        Purpose:
            Creates form to view a PDF

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (600, 800)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	

	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			fname = sg.popup_get_file( 'PDF Browser', 'PDF file to open', 
				file_types=(( 'PDF Files', '*.pdf' ),) )
			if fname is None:
				sg.popup_cancel( 'Cancelling' )
				exit(0)
			else:
				doc = fitz.open( fname )
				page_count = len( doc )
				
				# storage for page display lists
				dlist_tab = [ None ] * page_count
				title = 'PyMuPDF display of "%s", pages: %i' % (fname, page_count)
				
				
				def get_page( pno, zoom=0 ):
					"""
					
						Return a PNG image for a document page num.
						If zoom is other than 0, one of the 4 page quadrants
						are zoomed-in instead and the corresponding clip returned.

					"""
					dlist = dlist_tab[ pno ]  # get display get_list
					if not dlist:  # create_small_embedding if not yet there
						dlist_tab[ pno ] = doc[ pno ].get_displaylist( )
						dlist = dlist_tab[ pno ]
					r = dlist.rect  # page rectangle
					mp = r.tl + (r.br - r.tl) * 0.5  # rect middle point
					mt = r.tl + (r.tr - r.tl) * 0.5  # middle of top edge
					ml = r.tl + (r.bl - r.tl) * 0.5  # middle of left edge
					mr = r.tr + (r.br - r.tr) * 0.5  # middle of right egde
					mb = r.bl + (r.br - r.bl) * 0.5  # middle of bottom edge
					mat = fitz.Matrix( 2, 2 )  # zoom matrix
					if zoom == 1:  # top-left quadrant
						clip = fitz.Rect( r.tl, mp )
					elif zoom == 4:  # bot-right quadrant
						clip = fitz.Rect( mp, r.br )
					elif zoom == 2:  # top-right
						clip = fitz.Rect( mt, mr )
					elif zoom == 3:  # bot-left
						clip = fitz.Rect( ml, mb )
					if zoom == 0:  # total page
						pix = dlist.get_pixmap( alpha=False )
					else:
						pix = dlist.get_pixmap( alpha=False, matrix=mat, clip=clip )
					return pix.tobytes( )  # return the PNG image
				
				
				cur_page = 0
				data = get_page( cur_page )  # show page 1 for start
				image_elem = sg.Image( data=data )
				goto = sg.InputText( str( cur_page + 1 ), size=(5, 1) )
				layout = [
					[
						sg.Button( 'Prev' ),
						sg.Button( 'Next' ),
						sg.Text( 'Page:' ),
						goto,
					],
					[
						sg.Text( "Zoom:" ),
						sg.Button( 'Top-L' ),
						sg.Button( 'Top-R' ),
						sg.Button( 'Bot-L' ),
						sg.Button( 'Bot-R' ),
					],
					[ image_elem ],
				]
				my_keys = ('Next', 'Next:34', 'Prev', 'Prior:33', 'Top-L', 'Top-R',
				           'Bot-L', 'Bot-R', 'MouseWheel:Down', 'MouseWheel:Up')
				zoom_buttons = ('Top-L', 'Top-R', 'Bot-L', 'Bot-R')
				
				window = sg.Window( title, layout,
					return_keyboard_events=True, use_default_focus=False )
				
				old_page = 0
				old_zoom = 0  # used for zoom on/off
				# the zoom buttons work in on/off mode.
				while True:
					event, values = window.read( timeout=100 )
					zoom = 0
					force_page = False
					if event == sg.WIN_CLOSED:
						break
					
					if event in ("Escape:27",):  # this spares me a 'Quit' button!
						break
					if event[ 0 ] == chr( 13 ):  # surprise: this is 'Enter'!
						try:
							cur_page = int( values[ 0 ] ) - 1  # check if valid
							while cur_page < 0:
								cur_page += page_count
						except:
							cur_page = 0  # this guy's trying to fool me
						goto.update( str( cur_page + 1 ) )
					# goto.TKStringVar.pairs(str(cur_page + 1))
					
					elif event in ('Next', 'Next:34', 'MouseWheel:Down'):
						cur_page += 1
					elif event in ('Prev', 'Prior:33', 'MouseWheel:Up'):
						cur_page -= 1
					elif event == 'Top-L':
						zoom = 1
					elif event == 'Top-R':
						zoom = 2
					elif event == 'Bot-L':
						zoom = 3
					elif event == 'Bot-R':
						zoom = 4
					
					# sanitize page num
					if cur_page >= page_count:  # wrap around
						cur_page = 0
					while cur_page < 0:  # we show conventional page numbers
						cur_page += page_count
					
					# prevent creating same df again
					if cur_page != old_page:
						zoom = old_zoom = 0
						force_page = True
					
					if event in zoom_buttons:
						if 0 < zoom == old_zoom:
							zoom = 0
							force_page = True
						
						if zoom != old_zoom:
							force_page = True
					
					if force_page:
						data = get_page( cur_page, zoom )
						image_elem.update( data=data )
						old_page = cur_page
					old_zoom = zoom
					
					# update page num field
					if event in my_keys or not values[ 0 ]:
						goto.update( str( cur_page + 1 ) )
					# goto.TKStringVar.pairs(str(cur_page + 1))
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'PdfForm'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class CalendarDialog( Dark ):
	'''

        Construcotr:
        ------------
        CalendarDialog( )

        Purpose:
        ---------
        class creates form providing today selection behavior

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		self.selected_item = None
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (500, 250)
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if isinstance( self.selected_item, tuple ):
			year = str( self.selected_item[ 2 ] )
			month = str( self.selected_item[ 0 ] ).zfill( 2 )
			day = str( self.selected_item[ 1 ] ).zfill( 2 )
			date = f'{year}/{month}/{day}'
			return date
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_btnsize = (20, 1)
			_calendar = (250, 250)
			
			_months = [ 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
			            'AUG', 'SEP', 'OCT', 'NOV', 'DEC' ]
			
			_days = [ 'SUN', 'MON', 'TUE', 'WEC', 'THU', 'FRI', 'SAT' ]
			
			_cal = sg.popup_get_date( title='Calendar',
				no_titlebar=False,
				icon=self.icon_path,
				month_names=_months,
				day_abbreviations=_days,
				close_when_chosen=True )
			
			self.selected_item = _cal
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'CalendarDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ComboBoxDialog( Dark ):
	'''

        Construcotr:
            ComboBoxDialog( target_values: get_list = None )

        Purpose:
            Logger object provides form for log printing

	'''
	
	
	def __init__( self, data: list = None ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (400, 150)
		self.items = data
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.selected_item is not None:
			return self.selected_item
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_btnsz = (10, 1)
			_spc = (5, 1)
			if self.items is None:
				self.items = [ f'Item {x} ' for x in range( 30 ) ]
				_values = self.items
			
			_layout = [ [ sg.Text( size=_spc ), sg.Text( size=_spc ) ],
			            [ sg.Text( size=_spc ), sg.Text( 'Select Item' ) ],
			            [ sg.Text( size=_spc ),
			              sg.DropDown( self.items, key='-ITEM-', size=(35, 1) ) ],
			            [ sg.Text( size=_spc ), sg.Text( size=_spc ) ],
			            [ sg.Text( size=_spc ), sg.OK( size=_btnsz ), sg.Text( size=(8,
			                                                                         1) ),
			              sg.Cancel( size=_btnsz ) ] ]
			
			_window = sg.Window( '  Booger', _layout,
				icon=self.icon_path,
				size=self.form_size )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, 'Exit', 'Cancel'):
					break
				
				self.selected_item = _values[ '-ITEM-' ]
				sg.popup( _event, _values, self.selected_item,
					text_color=self.theme_textcolor,
					font=self.theme_font,
					icon=self.icon )
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ComboBoxDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ListBoxDialog( Dark ):
	'''

        Construcotr:
            ListBox( target_values: get_list = None )

        Purpose:
            List search and selection

    '''
	
	
	def __init__( self, data: list[ str ] = None ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (400, 250)
		self.image = os.getcwd( ) + r'\resources\img\app\dialog\lookup.png'
		self.items = data
	
	
	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.selected_item is not None:
			return self.selected_item
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_btnsize = (10, 1)
			_space = (10, 1)
			_line = (100, 1)
			_txtsz = (25, 1)
			_inpsz = (25, 1)
			_lstsz = (25, 5)
			_names = [ ]
			
			if isinstance( self.items, list ):
				_names = [ src for src in self.items ]
			else:
				_names = [ f'Item - {i}' for i in range( 40 ) ]
			
			_layout = [ [ sg.Text( size=_space ), sg.Text( size=_line ) ],
			            [ sg.Text( size=_space ), sg.Text( r'Search:' ) ],
			            [ sg.Text( size=_space ),
			              sg.Input( size=_inpsz, enable_events=True, key='-INPUT-' ) ],
			            [ sg.Text( size=_space ), sg.Text( size=_line ) ],
			            [ sg.Text( size=_space ),
			              sg.Listbox( _names, size=_lstsz, key='-ITEM-',
				              font=self.theme_font ) ],
			            [ sg.Text( size=_space ), sg.Text( size=_line ) ],
			            [ sg.Text( size=_space ),
			              sg.Button( 'Select', size=_btnsize, enable_events=True ),
			              sg.Text( size=(3, 1) ), sg.Button( 'Exit', size=_btnsize ) ] ]
			
			_window = sg.Window( '  Booger', _layout,
				size=self.form_size,
				font=self.theme_font )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, 'Exit'):
					break
				self.selected_item = str( _values[ '-ITEM-' ][ 0 ] )
				if _event == 'Selected':
					self.selected_item = str( _values[ '-ITEM-' ][ 0 ] )
					sg.popup( 'Results', self.selected_item,
						font=self.theme_font,
						icon=self.icon )
					_window.close( )
				
				if _values[ '-INPUT-' ] != '':
					_search = _values[ '-INPUT-' ]
					_newvalues = [ x for x in _names if _search in x ]
					_window[ '-ITEM-' ].update( _newvalues )
				else:
					_window[ '-ITEM-' ].update( _names )
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ListBoxDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ColorDialog( Dark ):
	'''

        Construcotr:

            ColorDialog( )

        Purpose:

            class provides a form to select colors returning path target_values

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (450, 450)
		self.rgb = None
		self.hex = None
		self.argb = None
		self.html = None
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_colormap = {
				'alice blue': '#F0F8FF',
				'AliceBlue': '#F0F8FF',
				'antique white': '#FAEBD7',
				'AntiqueWhite': '#FAEBD7',
				'AntiqueWhite1': '#FFEFDB',
				'AntiqueWhite2': '#EEDFCC',
				'AntiqueWhite3': '#CDC0B0',
				'AntiqueWhite4': '#8B8378',
				'aquamarine': '#7FFFD4',
				'aquamarine1': '#7FFFD4',
				'aquamarine2': '#76EEC6',
				'aquamarine3': '#66CDAA',
				'aquamarine4': '#458B74',
				'azure': '#F0FFFF',
				'azure1': '#F0FFFF',
				'azure2': '#E0EEEE',
				'azure3': '#C1CDCD',
				'azure4': '#838B8B',
				'beige': '#F5F5DC',
				'bisque': '#FFE4C4',
				'bisque1': '#FFE4C4',
				'bisque2': '#EED5B7',
				'bisque3': '#CDB79E',
				'bisque4': '#8B7D6B',
				'black': '#000000',
				'blanched almond': '#FFEBCD',
				'BlanchedAlmond': '#FFEBCD',
				'blue': '#0000FF',
				'blue violet': '#8A2BE2',
				'blue1': '#0000FF',
				'blue2': '#0000EE',
				'blue3': '#0000CD',
				'blue4': '#00008B',
				'BlueViolet': '#8A2BE2',
				'brown': '#A52A2A',
				'brown1': '#FF4040',
				'brown2': '#EE3B3B',
				'brown3': '#CD3333',
				'brown4': '#8B2323',
				'burlywood': '#DEB887',
				'burlywood1': '#FFD39B',
				'burlywood2': '#EEC591',
				'burlywood3': '#CDAA7D',
				'burlywood4': '#8B7355',
				'cadet blue': '#5F9EA0',
				'CadetBlue': '#5F9EA0',
				'CadetBlue1': '#98F5FF',
				'CadetBlue2': '#8EE5EE',
				'CadetBlue3': '#7AC5CD',
				'CadetBlue4': '#53868B',
				'chartreuse': '#7FFF00',
				'chartreuse1': '#7FFF00',
				'chartreuse2': '#76EE00',
				'chartreuse3': '#66CD00',
				'chartreuse4': '#458B00',
				'chocolate': '#D2691E',
				'chocolate1': '#FF7F24',
				'chocolate2': '#EE7621',
				'chocolate3': '#CD661D',
				'chocolate4': '#8B4513',
				'coral': '#FF7F50',
				'coral1': '#FF7256',
				'coral2': '#EE6A50',
				'coral3': '#CD5B45',
				'coral4': '#8B3E2F',
				'cornflower blue': '#6495ED',
				'CornflowerBlue': '#6495ED',
				'cornsilk': '#FFF8DC',
				'cornsilk1': '#FFF8DC',
				'cornsilk2': '#EEE8CD',
				'cornsilk3': '#CDC8B1',
				'cornsilk4': '#8B8878',
				'cyan': '#00FFFF',
				'cyan1': '#00FFFF',
				'cyan2': '#00EEEE',
				'cyan3': '#00CDCD',
				'cyan4': '#008B8B',
				'dark blue': '#00008B',
				'dark cyan': '#008B8B',
				'dark goldenrod': '#B8860B',
				'dark gray': '#A9A9A9',
				'dark green': '#006400',
				'dark grey': '#A9A9A9',
				'dark khaki': '#BDB76B',
				'dark magenta': '#8B008B',
				'dark olive green': '#556B2F',
				'dark orange': '#FF8C00',
				'dark orchid': '#9932CC',
				'dark red': '#8B0000',
				'dark salmon': '#E9967A',
				'dark sea green': '#8FBC8F',
				'dark slate blue': '#483D8B',
				'dark slate gray': '#2F4F4F',
				'dark slate grey': '#2F4F4F',
				'dark turquoise': '#00CED1',
				'dark violet': '#9400D3',
				'DarkBlue': '#00008B',
				'DarkCyan': '#008B8B',
				'DarkGoldenrod': '#B8860B',
				'DarkGoldenrod1': '#FFB90F',
				'DarkGoldenrod2': '#EEAD0E',
				'DarkGoldenrod3': '#CD950C',
				'DarkGoldenrod4': '#8B6508',
				'DarkGray': '#A9A9A9',
				'DarkGreen': '#006400',
				'DarkGrey': '#A9A9A9',
				'DarkKhaki': '#BDB76B',
				'DarkMagenta': '#8B008B',
				'DarkOliveGreen': '#556B2F',
				'DarkOliveGreen1': '#CAFF70',
				'DarkOliveGreen2': '#BCEE68',
				'DarkOliveGreen3': '#A2CD5A',
				'DarkOliveGreen4': '#6E8B3D',
				'DarkOrange': '#FF8C00',
				'DarkOrange1': '#FF7F00',
				'DarkOrange2': '#EE7600',
				'DarkOrange3': '#CD6600',
				'DarkOrange4': '#8B4500',
				'DarkOrchid': '#9932CC',
				'DarkOrchid1': '#BF3EFF',
				'DarkOrchid2': '#B23AEE',
				'DarkOrchid3': '#9A32CD',
				'DarkOrchid4': '#68228B',
				'DarkRed': '#8B0000',
				'DarkSalmon': '#E9967A',
				'DarkSeaGreen': '#8FBC8F',
				'DarkSeaGreen1': '#C1FFC1',
				'DarkSeaGreen2': '#B4EEB4',
				'DarkSeaGreen3': '#9BCD9B',
				'DarkSeaGreen4': '#698B69',
				'DarkSlateBlue': '#483D8B',
				'DarkSlateGray': '#2F4F4F',
				'DarkSlateGray1': '#97FFFF',
				'DarkSlateGray2': '#8DEEEE',
				'DarkSlateGray3': '#79CDCD',
				'DarkSlateGray4': '#528B8B',
				'DarkSlateGrey': '#2F4F4F',
				'DarkTurquoise': '#00CED1',
				'DarkViolet': '#9400D3',
				'deep pink': '#FF1493',
				'deep sky blue': '#00BFFF',
				'DeepPink': '#FF1493',
				'DeepPink1': '#FF1493',
				'DeepPink2': '#EE1289',
				'DeepPink3': '#CD1076',
				'DeepPink4': '#8B0A50',
				'DeepSkyBlue': '#00BFFF',
				'DeepSkyBlue1': '#00BFFF',
				'DeepSkyBlue2': '#00B2EE',
				'DeepSkyBlue3': '#009ACD',
				'DeepSkyBlue4': '#00688B',
				'dim gray': '#696969',
				'dim grey': '#696969',
				'DimGray': '#696969',
				'DimGrey': '#696969',
				'dodger blue': '#1E90FF',
				'DodgerBlue': '#1E90FF',
				'DodgerBlue1': '#1E90FF',
				'DodgerBlue2': '#1C86EE',
				'DodgerBlue3': '#1874CD',
				'DodgerBlue4': '#104E8B',
				'firebrick': '#B22222',
				'firebrick1': '#FF3030',
				'firebrick2': '#EE2C2C',
				'firebrick3': '#CD2626',
				'firebrick4': '#8B1A1A',
				'floral white': '#FFFAF0',
				'FloralWhite': '#FFFAF0',
				'forest green': '#228B22',
				'ForestGreen': '#228B22',
				'gainsboro': '#DCDCDC',
				'ghost white': '#F8F8FF',
				'GhostWhite': '#F8F8FF',
				'gold': '#FFD700',
				'gold1': '#FFD700',
				'gold2': '#EEC900',
				'gold3': '#CDAD00',
				'gold4': '#8B7500',
				'goldenrod': '#DAA520',
				'goldenrod1': '#FFC125',
				'goldenrod2': '#EEB422',
				'goldenrod3': '#CD9B1D',
				'goldenrod4': '#8B6914',
				'green': '#00FF00',
				'green yellow': '#ADFF2F',
				'green1': '#00FF00',
				'green2': '#00EE00',
				'green3': '#00CD00',
				'green4': '#008B00',
				'GreenYellow': '#ADFF2F',
				'grey': '#BEBEBE',
				'grey0': '#000000',
				'grey1': '#030303',
				'grey2': '#050505',
				'grey3': '#080808',
				'grey4': '#0A0A0A',
				'grey5': '#0D0D0D',
				'grey6': '#0F0F0F',
				'grey7': '#121212',
				'grey8': '#141414',
				'grey9': '#171717',
				'grey10': '#1A1A1A',
				'grey11': '#1C1C1C',
				'grey12': '#1F1F1F',
				'grey13': '#212121',
				'grey14': '#242424',
				'grey15': '#262626',
				'grey16': '#292929',
				'grey17': '#2B2B2B',
				'grey18': '#2E2E2E',
				'grey19': '#303030',
				'grey20': '#333333',
				'grey21': '#363636',
				'grey22': '#383838',
				'grey23': '#3B3B3B',
				'grey24': '#3D3D3D',
				'grey25': '#404040',
				'grey26': '#424242',
				'grey27': '#454545',
				'grey28': '#474747',
				'grey29': '#4A4A4A',
				'grey30': '#4D4D4D',
				'grey31': '#4F4F4F',
				'grey32': '#525252',
				'grey33': '#545454',
				'grey34': '#575757',
				'grey35': '#595959',
				'grey36': '#5C5C5C',
				'grey37': '#5E5E5E',
				'grey38': '#616161',
				'grey39': '#636363',
				'grey40': '#666666',
				'grey41': '#696969',
				'grey42': '#6B6B6B',
				'grey43': '#6E6E6E',
				'grey44': '#707070',
				'grey45': '#737373',
				'grey46': '#757575',
				'grey47': '#787878',
				'grey48': '#7A7A7A',
				'grey49': '#7D7D7D',
				'grey50': '#7F7F7F',
				'grey51': '#828282',
				'grey52': '#858585',
				'grey53': '#878787',
				'grey54': '#8A8A8A',
				'grey55': '#8C8C8C',
				'grey56': '#8F8F8F',
				'grey57': '#919191',
				'grey58': '#949494',
				'grey59': '#969696',
				'grey60': '#999999',
				'grey61': '#9C9C9C',
				'grey62': '#9E9E9E',
				'grey63': '#A1A1A1',
				'grey64': '#A3A3A3',
				'grey65': '#A6A6A6',
				'grey66': '#A8A8A8',
				'grey67': '#ABABAB',
				'grey68': '#ADADAD',
				'grey69': '#B0B0B0',
				'grey70': '#B3B3B3',
				'grey71': '#B5B5B5',
				'grey72': '#B8B8B8',
				'grey73': '#BABABA',
				'grey74': '#BDBDBD',
				'grey75': '#BFBFBF',
				'grey76': '#C2C2C2',
				'grey77': '#C4C4C4',
				'grey78': '#C7C7C7',
				'grey79': '#C9C9C9',
				'grey80': '#CCCCCC',
				'grey81': '#CFCFCF',
				'grey82': '#D1D1D1',
				'grey83': '#D4D4D4',
				'grey84': '#D6D6D6',
				'grey85': '#D9D9D9',
				'grey86': '#DBDBDB',
				'grey87': '#DEDEDE',
				'grey88': '#E0E0E0',
				'grey89': '#E3E3E3',
				'grey90': '#E5E5E5',
				'grey91': '#E8E8E8',
				'grey92': '#EBEBEB',
				'grey93': '#EDEDED',
				'grey94': '#F0F0F0',
				'grey95': '#F2F2F2',
				'grey96': '#F5F5F5',
				'grey97': '#F7F7F7',
				'grey98': '#FAFAFA',
				'grey99': '#FCFCFC',
				'grey100': '#FFFFFF',
				'honeydew': '#F0FFF0',
				'honeydew1': '#F0FFF0',
				'honeydew2': '#E0EEE0',
				'honeydew3': '#C1CDC1',
				'honeydew4': '#838B83',
				'hot pink': '#FF69B4',
				'HotPink': '#FF69B4',
				'HotPink1': '#FF6EB4',
				'HotPink2': '#EE6AA7',
				'HotPink3': '#CD6090',
				'HotPink4': '#8B3A62',
				'indian red': '#CD5C5C',
				'IndianRed': '#CD5C5C',
				'IndianRed1': '#FF6A6A',
				'IndianRed2': '#EE6363',
				'IndianRed3': '#CD5555',
				'IndianRed4': '#8B3A3A',
				'ivory': '#FFFFF0',
				'ivory1': '#FFFFF0',
				'ivory2': '#EEEEE0',
				'ivory3': '#CDCDC1',
				'ivory4': '#8B8B83',
				'khaki': '#F0E68C',
				'khaki1': '#FFF68F',
				'khaki2': '#EEE685',
				'khaki3': '#CDC673',
				'khaki4': '#8B864E',
				'lavender': '#E6E6FA',
				'lavender blush': '#FFF0F5',
				'LavenderBlush': '#FFF0F5',
				'LavenderBlush1': '#FFF0F5',
				'LavenderBlush2': '#EEE0E5',
				'LavenderBlush3': '#CDC1C5',
				'LavenderBlush4': '#8B8386',
				'lawn green': '#7CFC00',
				'LawnGreen': '#7CFC00',
				'lemon chiffon': '#FFFACD',
				'LemonChiffon': '#FFFACD',
				'LemonChiffon1': '#FFFACD',
				'LemonChiffon2': '#EEE9BF',
				'LemonChiffon3': '#CDC9A5',
				'LemonChiffon4': '#8B8970',
				'light blue': '#ADD8E6',
				'light coral': '#F08080',
				'light cyan': '#E0FFFF',
				'light goldenrod': '#EEDD82',
				'light goldenrod yellow': '#FAFAD2',
				'light gray': '#D3D3D3',
				'light green': '#90EE90',
				'light grey': '#D3D3D3',
				'light pink': '#FFB6C1',
				'light salmon': '#FFA07A',
				'light sea green': '#20B2AA',
				'light sky blue': '#87CEFA',
				'light slate blue': '#8470FF',
				'light slate gray': '#778899',
				'light slate grey': '#778899',
				'light steel blue': '#B0C4DE',
				'light yellow': '#FFFFE0',
				'LightBlue': '#ADD8E6',
				'LightBlue1': '#BFEFFF',
				'LightBlue2': '#B2DFEE',
				'LightBlue3': '#9AC0CD',
				'LightBlue4': '#68838B',
				'LightCoral': '#F08080',
				'LightCyan': '#E0FFFF',
				'LightCyan1': '#E0FFFF',
				'LightCyan2': '#D1EEEE',
				'LightCyan3': '#B4CDCD',
				'LightCyan4': '#7A8B8B',
				'LightGoldenrod': '#EEDD82',
				'LightGoldenrod1': '#FFEC8B',
				'LightGoldenrod2': '#EEDC82',
				'LightGoldenrod3': '#CDBE70',
				'LightGoldenrod4': '#8B814C',
				'LightGoldenrodYellow': '#FAFAD2',
				'LightGray': '#D3D3D3',
				'LightGreen': '#90EE90',
				'LightGrey': '#D3D3D3',
				'LightPink': '#FFB6C1',
				'LightPink1': '#FFAEB9',
				'LightPink2': '#EEA2AD',
				'LightPink3': '#CD8C95',
				'LightPink4': '#8B5F65',
				'LightSalmon': '#FFA07A',
				'LightSalmon1': '#FFA07A',
				'LightSalmon2': '#EE9572',
				'LightSalmon3': '#CD8162',
				'LightSalmon4': '#8B5742',
				'LightSeaGreen': '#20B2AA',
				'LightSkyBlue': '#87CEFA',
				'LightSkyBlue1': '#B0E2FF',
				'LightSkyBlue2': '#A4D3EE',
				'LightSkyBlue3': '#8DB6CD',
				'LightSkyBlue4': '#607B8B',
				'LightSlateBlue': '#8470FF',
				'LightSlateGray': '#778899',
				'LightSlateGrey': '#778899',
				'LightSteelBlue': '#B0C4DE',
				'LightSteelBlue1': '#CAE1FF',
				'LightSteelBlue2': '#BCD2EE',
				'LightSteelBlue3': '#A2B5CD',
				'LightSteelBlue4': '#6E7B8B',
				'LightYellow': '#FFFFE0',
				'LightYellow1': '#FFFFE0',
				'LightYellow2': '#EEEED1',
				'LightYellow3': '#CDCDB4',
				'LightYellow4': '#8B8B7A',
				'lime green': '#32CD32',
				'LimeGreen': '#32CD32',
				'linen': '#FAF0E6',
				'magenta': '#FF00FF',
				'magenta1': '#FF00FF',
				'magenta2': '#EE00EE',
				'magenta3': '#CD00CD',
				'magenta4': '#8B008B',
				'maroon': '#B03060',
				'maroon1': '#FF34B3',
				'maroon2': '#EE30A7',
				'maroon3': '#CD2990',
				'maroon4': '#8B1C62',
				'medium aquamarine': '#66CDAA',
				'medium blue': '#0000CD',
				'medium orchid': '#BA55D3',
				'medium purple': '#9370DB',
				'medium sea green': '#3CB371',
				'medium slate blue': '#7B68EE',
				'medium spring green': '#00FA9A',
				'medium turquoise': '#48D1CC',
				'medium violet red': '#C71585',
				'MediumAquamarine': '#66CDAA',
				'MediumBlue': '#0000CD',
				'MediumOrchid': '#BA55D3',
				'MediumOrchid1': '#E066FF',
				'MediumOrchid2': '#D15FEE',
				'MediumOrchid3': '#B452CD',
				'MediumOrchid4': '#7A378B',
				'MediumPurple': '#9370DB',
				'MediumPurple1': '#AB82FF',
				'MediumPurple2': '#9F79EE',
				'MediumPurple3': '#8968CD',
				'MediumPurple4': '#5D478B',
				'MediumSeaGreen': '#3CB371',
				'MediumSlateBlue': '#7B68EE',
				'MediumSpringGreen': '#00FA9A',
				'MediumTurquoise': '#48D1CC',
				'MediumVioletRed': '#C71585',
				'midnight blue': '#191970',
				'MidnightBlue': '#191970',
				'mint cream': '#F5FFFA',
				'MintCream': '#F5FFFA',
				'misty rose': '#FFE4E1',
				'MistyRose': '#FFE4E1',
				'MistyRose1': '#FFE4E1',
				'MistyRose2': '#EED5D2',
				'MistyRose3': '#CDB7B5',
				'MistyRose4': '#8B7D7B',
				'moccasin': '#FFE4B5',
				'navajo white': '#FFDEAD',
				'NavajoWhite': '#FFDEAD',
				'NavajoWhite1': '#FFDEAD',
				'NavajoWhite2': '#EECFA1',
				'NavajoWhite3': '#CDB38B',
				'NavajoWhite4': '#8B795E',
				'navy': '#000080',
				'navy blue': '#000080',
				'NavyBlue': '#000080',
				'old lace': '#FDF5E6',
				'OldLace': '#FDF5E6',
				'olive drab': '#6B8E23',
				'OliveDrab': '#6B8E23',
				'OliveDrab1': '#C0FF3E',
				'OliveDrab2': '#B3EE3A',
				'OliveDrab3': '#9ACD32',
				'OliveDrab4': '#698B22',
				'orange': '#FFA500',
				'orange red': '#FF4500',
				'orange1': '#FFA500',
				'orange2': '#EE9A00',
				'orange3': '#CD8500',
				'orange4': '#8B5A00',
				'OrangeRed': '#FF4500',
				'OrangeRed1': '#FF4500',
				'OrangeRed2': '#EE4000',
				'OrangeRed3': '#CD3700',
				'OrangeRed4': '#8B2500',
				'orchid': '#DA70D6',
				'orchid1': '#FF83FA',
				'orchid2': '#EE7AE9',
				'orchid3': '#CD69C9',
				'orchid4': '#8B4789',
				'pale goldenrod': '#EEE8AA',
				'pale green': '#98FB98',
				'pale turquoise': '#AFEEEE',
				'pale violet red': '#DB7093',
				'PaleGoldenrod': '#EEE8AA',
				'PaleGreen': '#98FB98',
				'PaleGreen1': '#9AFF9A',
				'PaleGreen2': '#90EE90',
				'PaleGreen3': '#7CCD7C',
				'PaleGreen4': '#548B54',
				'PaleTurquoise': '#AFEEEE',
				'PaleTurquoise1': '#BBFFFF',
				'PaleTurquoise2': '#AEEEEE',
				'PaleTurquoise3': '#96CDCD',
				'PaleTurquoise4': '#668B8B',
				'PaleVioletRed': '#DB7093',
				'PaleVioletRed1': '#FF82AB',
				'PaleVioletRed2': '#EE799F',
				'PaleVioletRed3': '#CD687F',
				'PaleVioletRed4': '#8B475D',
				'papaya whip': '#FFEFD5',
				'PapayaWhip': '#FFEFD5',
				'peach puff': '#FFDAB9',
				'PeachPuff': '#FFDAB9',
				'PeachPuff1': '#FFDAB9',
				'PeachPuff2': '#EECBAD',
				'PeachPuff3': '#CDAF95',
				'PeachPuff4': '#8B7765',
				'peru': '#CD853F',
				'pink': '#FFC0CB',
				'pink1': '#FFB5C5',
				'pink2': '#EEA9B8',
				'pink3': '#CD919E',
				'pink4': '#8B636C',
				'plum': '#DDA0DD',
				'plum1': '#FFBBFF',
				'plum2': '#EEAEEE',
				'plum3': '#CD96CD',
				'plum4': '#8B668B',
				'powder blue': '#B0E0E6',
				'PowderBlue': '#B0E0E6',
				'purple': '#A020F0',
				'purple1': '#9B30FF',
				'purple2': '#912CEE',
				'purple3': '#7D26CD',
				'purple4': '#551A8B',
				'red': '#FF0000',
				'red1': '#FF0000',
				'red2': '#EE0000',
				'red3': '#CD0000',
				'red4': '#8B0000',
				'rosy brown': '#BC8F8F',
				'RosyBrown': '#BC8F8F',
				'RosyBrown1': '#FFC1C1',
				'RosyBrown2': '#EEB4B4',
				'RosyBrown3': '#CD9B9B',
				'RosyBrown4': '#8B6969',
				'royal blue': '#4169E1',
				'RoyalBlue': '#4169E1',
				'RoyalBlue1': '#4876FF',
				'RoyalBlue2': '#436EEE',
				'RoyalBlue3': '#3A5FCD',
				'RoyalBlue4': '#27408B',
				'saddle brown': '#8B4513',
				'SaddleBrown': '#8B4513',
				'salmon': '#FA8072',
				'salmon1': '#FF8C69',
				'salmon2': '#EE8262',
				'salmon3': '#CD7054',
				'salmon4': '#8B4C39',
				'sandy brown': '#F4A460',
				'SandyBrown': '#F4A460',
				'sea green': '#2E8B57',
				'SeaGreen': '#2E8B57',
				'SeaGreen1': '#54FF9F',
				'SeaGreen2': '#4EEE94',
				'SeaGreen3': '#43CD80',
				'SeaGreen4': '#2E8B57',
				'seashell': '#FFF5EE',
				'seashell1': '#FFF5EE',
				'seashell2': '#EEE5DE',
				'seashell3': '#CDC5BF',
				'seashell4': '#8B8682',
				'sienna': '#A0522D',
				'sienna1': '#FF8247',
				'sienna2': '#EE7942',
				'sienna3': '#CD6839',
				'sienna4': '#8B4726',
				'sky blue': '#87CEEB',
				'SkyBlue': '#87CEEB',
				'SkyBlue1': '#87CEFF',
				'SkyBlue2': '#7EC0EE',
				'SkyBlue3': '#6CA6CD',
				'SkyBlue4': '#4A708B',
				'slate blue': '#6A5ACD',
				'slate gray': '#708090',
				'slate grey': '#708090',
				'SlateBlue': '#6A5ACD',
				'SlateBlue1': '#836FFF',
				'SlateBlue2': '#7A67EE',
				'SlateBlue3': '#6959CD',
				'SlateBlue4': '#473C8B',
				'SlateGray': '#708090',
				'SlateGray1': '#C6E2FF',
				'SlateGray2': '#B9D3EE',
				'SlateGray3': '#9FB6CD',
				'SlateGray4': '#6C7B8B',
				'SlateGrey': '#708090',
				'snow': '#FFFAFA',
				'snow1': '#FFFAFA',
				'snow2': '#EEE9E9',
				'snow3': '#CDC9C9',
				'snow4': '#8B8989',
				'spring green': '#00FF7F',
				'SpringGreen': '#00FF7F',
				'SpringGreen1': '#00FF7F',
				'SpringGreen2': '#00EE76',
				'SpringGreen3': '#00CD66',
				'SpringGreen4': '#008B45',
				'steel blue': '#4682B4',
				'SteelBlue': '#4682B4',
				'SteelBlue1': '#63B8FF',
				'SteelBlue2': '#5CACEE',
				'SteelBlue3': '#4F94CD',
				'SteelBlue4': '#36648B',
				'tan': '#D2B48C',
				'tan1': '#FFA54F',
				'tan2': '#EE9A49',
				'tan3': '#CD853F',
				'tan4': '#8B5A2B',
				'thistle': '#D8BFD8',
				'thistle1': '#FFE1FF',
				'thistle2': '#EED2EE',
				'thistle3': '#CDB5CD',
				'thistle4': '#8B7B8B',
				'tomato': '#FF6347',
				'tomato1': '#FF6347',
				'tomato2': '#EE5C42',
				'tomato3': '#CD4F39',
				'tomato4': '#8B3626',
				'turquoise': '#40E0D0',
				'turquoise1': '#00F5FF',
				'turquoise2': '#00E5EE',
				'turquoise3': '#00C5CD',
				'turquoise4': '#00868B',
				'violet': '#EE82EE',
				'violet red': '#D02090',
				'VioletRed': '#D02090',
				'VioletRed1': '#FF3E96',
				'VioletRed2': '#EE3A8C',
				'VioletRed3': '#CD3278',
				'VioletRed4': '#8B2252',
				'wheat': '#F5DEB3',
				'wheat1': '#FFE7BA',
				'wheat2': '#EED8AE',
				'wheat3': '#CDBA96',
				'wheat4': '#8B7E66',
				'white': '#FFFFFF',
				'white smoke': '#F5F5F5',
				'WhiteSmoke': '#F5F5F5',
				'yellow': '#FFFF00',
				'yellow green': '#9ACD32',
				'yellow1': '#FFFF00',
				'yellow2': '#EEEE00',
				'yellow3': '#CDCD00',
				'yellow4': '#8B8B00',
				'YellowGreen': '#9ACD32' }
			_hextocolor = { v: k for k, v in _colormap.items( ) }
			_colorlist = list( _colormap.keys( ) )
			COLORS_PER_ROW = 40
			_fontsize = 9
			
			
			def make_window( ):
				_layout = [ [ sg.Text( ), ],
				            [ sg.Text( f'{len( _colorlist )} Colors', font=self.theme_font ), ],
				            [ sg.Text( size=(5, 1) ), ] ]
				
				for rows in range( len( _colorlist ) // COLORS_PER_ROW + 1 ):
					_row = [ ]
					
					for i in range( COLORS_PER_ROW ):
						try:
							color = _colorlist[ rows * COLORS_PER_ROW + i ]
							_row.append(
								sg.Text( ' ', s=1, background_color=color, text_color=color,
									font=self.theme_font,
									right_click_menu=[ '_', _colormap[ color ] ],
									tooltip=color, enable_events=True,
									key=(color, _colormap[ color ]) ) )
						except IndexError as e:
							break
						except Exception as e:
							sg.popup_error( f'Error while creating _color _window....', e,
								f'rows = {rows}  i = {i}' )
							break
					_layout.append( _row )
				_layout.append( [ sg.Text( ' ', size=(10, 1) ), ] )
				_layout.append( [ sg.Text( ' ', size=(10, 1) ), ] )
				_layout.append( [ sg.Text( ' ', size=(50, 1) ), sg.Cancel( size=(20, 1) ), ] )
				
				return sg.Window( ' Booger', _layout,
					font=self.theme_font,
					size=self.form_size,
					element_padding=(1, 1),
					border_depth=0,
					icon=self.icon_path,
					right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_EXIT,
					use_ttk_buttons=True )
			
			
			_window = make_window( )
			
			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, 'Cancel', 'Exit'):
					break
				if _event == 'Edit me':
					sg.execute_editor( __file__ )
					continue
				elif isinstance( _event, tuple ):
					_color, _colorhex = _event[ 0 ], _event[ 1 ]
				else:
					_color, _colorhex = _hextocolor[ _event ], _event
				
				_layout2 = [ [ sg.Text( _colorhex + ' on clipboard' ) ],
				             [ sg.DummyButton( _color, button_color=self.button_color,
					             tooltip=_colorhex ),
				               sg.DummyButton( _color, button_color=self.button_color,
					               tooltip=_colorhex ) ] ]
				
				_window2 = sg.Window( 'Buttons with white and black documents', _layout2,
					keep_on_top=True,
					finalize=True,
					size=self.form_size,
					icon=self.icon )
				
				sg.clipboard_set( _colorhex )
			
			_window.close( )
			
			sg.popup_quick_message( 'Building _window... one moment please...',
				background_color=self.theme_background,
				icon=self.icon_path,
				text_color=self.theme_textcolor,
				font=self.theme_font )
			
			sg.set_options( button_element_size=(12, 1),
				element_padding=(0, 0),
				auto_size_buttons=False,
				border_width=1,
				tooltip_time=100 )
		except Exception as e:
			_exception = Error( e )
			_exception.module = 'Booger'
			_exception.cause = 'ColorDialog'
			_exception.method = 'show( self )'
			_error = ErrorDialog( _exception )
			_error.show( )


class BudgetForm( Dark ):
	'''

        Constructor:

            BudgetForm( )

        Purpose:

            Class defining basic dashboard for the application

    '''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.image = r'C:\Users\terry\source\repos\Boo\resources\img\app\Application.png'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (1200, 650)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'progressbar_color',
		         'title_items', 'header_items', 'first_items',
		         'second_items', 'third_items', 'form_size',
		         'image', 'show', 'create_title', 'create_header',
		         'create_first', 'create_second', 'create_third',
		         'create_fourth', 'set_layout', 'show' ]
	
	
	def create_title( self, items: list ) -> list:
		'''

            Purpose:

            Parameters:

            Returns:

		'''
		if items is not None:
			try:
				_blu = '#051F3D'
				_blk = '#101010'
				_mblk = '#1E1E1E'
				BPAD_TOP = ((5, 5), (5, 5))
				BPAD_LEFT = ((5, 5), (5, 5))
				BPAD_LEFT_INSIDE = (5, (3, 5))
				BPAD_RIGHT = ((5, 10), (3, 3))
				_font = 'Roboto 20'
				_form = (450, 150)
				_hdrsz = (920, 100)
				_title = [
					[ sg.Text( f'{items[ 0 ]}', font=_font, background_color=_mblk,
						enable_events=True, grab=False ),
					  sg.Push( background_color=_mblk ),
					  sg.Text( f'{items[ 1 ]}', font=_font, background_color=_mblk ) ],
				]
				self.__titlelayout = _title
				return _title
			except Exception as e:
				exception = Error( e )
				exception.module = 'Booger'
				exception.cause = 'BudgetForm'
				exception.method = 'create_title( self, items )'
				error = ErrorDialog( exception )
				error.show( )
	
	
	def create_header( self, items: list ) -> list:
		'''

            Purpose:

            Parameters:

            Returns:

		'''
		if items is not None:
			try:
				_blu = '#051F3D'
				_blk = '#101010'
				_mblk = '#1E1E1E'
				BPAD_TOP = ((5, 5), (5, 5))
				BPAD_LEFT = ((5, 5), (5, 5))
				BPAD_LEFT_INSIDE = (5, (3, 5))
				BPAD_RIGHT = ((5, 10), (3, 3))
				_hdr = 'Roboto 20'
				_frasz = (450, 150)
				_hdrsz = (920, 100)
				_header = [ [ sg.Push( ), sg.Text( f'{items[ 0 ]}', font=_hdr ), sg.Push( ) ],
				            [ sg.Text( f'{items[ 1 ]}' ) ],
				            [ sg.Text( f'{items[ 2 ]}' ) ] ]
				self.__headerlayout = _header
				return _header
			except Exception as e:
				exception = Error( e )
				exception.module = 'Booger'
				exception.cause = 'BudgetForm'
				exception.method = 'create_header( self, items )'
				error = ErrorDialog( exception )
				error.show( )
	
	
	def create_first( self, items: list ) -> list:
		'''

            Purpose:

            Parameters:

            Returns:

		'''
		if items is not None:
			try:
				_blu = '#051F3D'
				_blk = '#101010'
				_mblk = '#1E1E1E'
				BPAD_TOP = ((5, 5), (5, 5))
				BPAD_LEFT = ((5, 5), (5, 5))
				BPAD_LEFT_INSIDE = (5, (3, 5))
				BPAD_RIGHT = ((5, 10), (3, 3))
				_hdr = 'Roboto 20'
				_frasz = (450, 150)
				_hdrsz = (920, 100)
				_first = [ [ sg.Push( ), sg.Text( 'Block 1 Header', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 1 line 1', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 1 line 2', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 1 line 3', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 1 line 4', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 1 line 5', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 1 line 6', font=_hdr ), sg.Push( ) ] ]
				self.__firstlayout = _first
				return _first
			except Exception as e:
				exception = Error( e )
				exception.module = 'Booger'
				exception.cause = 'BudgetForm'
				exception.method = 'create_first( self, items: get_list ) -> get_list'
				error = ErrorDialog( exception )
				error.show( )
	
	
	def create_second( self, items: list ) -> list:
		'''

            Purpose:

            Parameters:

            Returns:

		'''
		if items is not None:
			try:
				_blu = '#051F3D'
				_blk = '#101010'
				_mblk = '#1E1E1E'
				BPAD_TOP = ((5, 5), (5, 5))
				BPAD_LEFT = ((5, 5), (5, 5))
				BPAD_LEFT_INSIDE = (5, (3, 5))
				BPAD_RIGHT = ((5, 10), (3, 3))
				_hdr = 'Roboto 20'
				_frasz = (450, 150)
				_hdrsz = (920, 100)
				_second = [ [ sg.Push( ), sg.Text( 'Block 2 Header', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 2 line 1', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 2 line 2', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 2 line 3', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 2 line 4', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 2 line 5', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 2 line 6', font=_hdr ), sg.Push( ) ] ]
				self.__secondlayout = _second
				return _second
			except Exception as e:
				exception = Error( e )
				exception.module = 'Booger'
				exception.cause = 'BudgetForm'
				exception.method = 'create_second( self, items )'
				error = ErrorDialog( exception )
				error.show( )
	
	
	def create_third( self, items: list ) -> list:
		'''

            Purpose:

            Parameters:

            Returns:

		'''
		if items is not None:
			try:
				_blu = '#051F3D'
				_blk = '#101010'
				_mblk = '#1E1E1E'
				BPAD_TOP = ((5, 5), (5, 5))
				BPAD_LEFT = ((5, 5), (5, 5))
				BPAD_LEFT_INSIDE = (5, (3, 5))
				BPAD_RIGHT = ((5, 10), (3, 3))
				_hdr = 'Roboto 20'
				_frasz = (450, 150)
				_hdrsz = (920, 100)
				_third = [ [ sg.Push( ), sg.Text( 'Block 3 Header', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 3 line 1', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 3 line 2', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 3 line 3', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 3 line 4', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 3 line 5', font=_hdr ), sg.Push( ) ],
				           [ sg.Push( ), sg.Text( 'Block 3 line 6', font=_hdr ), sg.Push( ) ] ]
				self.__thirdlayout = _third
				return _third
			except Exception as e:
				exception = Error( e )
				exception.module = 'Booger'
				exception.cause = 'BudgetForm'
				exception.method = 'create_third( self, items: get_list )'
				error = ErrorDialog( exception )
				error.show( )
	
	
	def create_fourth( self, items: list ) -> list:
		'''

            Purpose:

            Parameters:

            Returns:

		'''
		if items is not None:
			try:
				_blu = '#051F3D'
				_blk = '#101010'
				_mblk = '#1E1E1E'
				BPAD_TOP = ((5, 5), (5, 5))
				BPAD_LEFT = ((5, 5), (5, 5))
				BPAD_LEFT_INSIDE = (5, (3, 5))
				BPAD_RIGHT = ((5, 10), (3, 3))
				_hdr = 'Roboto 20'
				_frasz = (450, 150)
				_hdrsz = (920, 100)
				_fourth = [ [ sg.Push( ), sg.Text( 'Block 4 Header', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 4 line 1', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 4 line 2', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 4 line 3', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 4 line 4', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 4 line 5', font=_hdr ), sg.Push( ) ],
				            [ sg.Push( ), sg.Text( 'Block 4 line 6', font=_hdr ), sg.Push( ) ] ]
				self.__fourthlayout = _fourth
				return _fourth
			except Exception as e:
				exception = Error( e )
				exception.module = 'Booger'
				exception.cause = 'BudgetForm'
				exception.method = 'create_fourth( self, items: get_list )'
				error = ErrorDialog( exception )
				error.show( )
	
	
	def set_layout( self ) -> list:
		'''

            Purpose:

            Parameters:

            Returns:

		'''
		try:
			_blu = '#051F3D'
			_blk = '#101010'
			_mblk = '#1E1E1E'
			BPAD_TOP = ((5, 5), (5, 5))
			BPAD_LEFT = ((5, 5), (5, 5))
			BPAD_LEFT_INSIDE = (5, (5, 5))
			BPAD_RIGHT = ((5, 5), (5, 5))
			_hdr = 'Roboto 20'
			_li = 'Roboto 11'
			_frasz = (450, 150)
			_hdrsz = (920, 100)
			_layout = [
				[ sg.Frame( '', self.__titlelayout, pad=(0, 0), background_color=_mblk,
					expand_x=True,
					border_width=0, grab=True ) ],
				[ sg.Frame( '', self.__headerlayout, size=_hdrsz, pad=BPAD_TOP,
					expand_x=True,
					relief=sg.RELIEF_FLAT, border_width=0 ) ],
				[ sg.Frame( '',
					[ [ sg.Frame( '', self.__firstlayout, size=_frasz, pad=
					BPAD_LEFT_INSIDE,
						border_width=0, expand_x=True, expand_y=True, ) ],
					  [ sg.Frame( '', self.__thirdlayout, size=_frasz, pad=
					  BPAD_LEFT_INSIDE,
						  border_width=0, expand_x=True, expand_y=True ) ] ],
					pad=BPAD_LEFT, background_color=_blk, border_width=0,
					expand_x=True, expand_y=True ),
				  sg.Frame( '',
					  [ [ sg.Frame( '', self.__secondlayout, size=_frasz,
						  pad=BPAD_LEFT_INSIDE,
						  border_width=0, expand_x=True, expand_y=True ) ],
					    [ sg.Frame( '', self.__fourthlayout, size=_frasz,
						    pad=BPAD_LEFT_INSIDE,
						    border_width=0, expand_x=True, expand_y=True ) ] ],
					  pad=BPAD_LEFT, background_color=_blk, border_width=0,
					  expand_x=True, expand_y=True ), ],
				[ sg.Sizegrip( background_color=_mblk ) ] ]
			self.__formlayout = _layout
			return _layout
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'BudgetForm'
			exception.method = 'set_layout( self, items )'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_blu = '#051F3D'
			_blk = '#101010'
			_mblk = '#1E1E1E'
			BPAD_TOP = ((5, 5), (5, 5))
			BPAD_LEFT = ((5, 5), (5, 5))
			BPAD_LEFT_INSIDE = (5, (5, 5))
			BPAD_RIGHT = ((5, 5), (5, 5))
			_hdr = 'Roboto 20'
			_li = 'Roboto 11'
			_frasz = (450, 150)
			_hdrsz = (920, 100)
			self.__titlelayout = [
				[ sg.Text( 'Booger', font=_hdr, background_color=_mblk,
					enable_events=True, grab=False ), sg.Push( background_color=_mblk ),
				  sg.Text( 'Wednesday 27 Oct 2021', font=_hdr, background_color=_mblk ) ],
			]
			self.__headerlayout = [ [ sg.Push( ), sg.Text( 'Top Header', font=_hdr ), sg.Push(
			) ],
			                        [ sg.Image( source=self.image, subsample=3,
				                        enable_events=True ), sg.Push( ) ],
			                        [ sg.Text( 'Top Header line 2' ), sg.Push( ) ] ]
			self.__firstlayout = [
				[ sg.Push( ), sg.Text( 'Block 1 Header', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 1 line 1', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 1 line 2', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 1 line 3', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 1 line 4', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 1 line 5', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 1 line 6', font=_hdr ), sg.Push( ) ] ]
			self.__secondlayout = [
				[ sg.Push( ), sg.Text( 'Block 2 Header', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 2 line 1', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 2 line 2', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 2 line 3', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 2 line 4', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 2 line 5', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 2 line 6', font=_hdr ), sg.Push( ) ] ]
			self.__thirdlayout = [
				[ sg.Push( ), sg.Text( 'Block 3 Header', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 3 line 1', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 3 line 2', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 3 line 3', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 3 line 4', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 3 line 5', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 3 line 6', font=_hdr ), sg.Push( ) ] ]
			self.__fourthlayout = [
				[ sg.Push( ), sg.Text( 'Block 4 Header', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 4 line 1', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 4 line 2', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 4 line 3', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 4 line 4', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 4 line 5', font=_hdr ), sg.Push( ) ],
				[ sg.Push( ), sg.Text( 'Block 4 line 6', font=_hdr ), sg.Push( ) ] ]
			self.__formlayout = [
				[ sg.Frame( '', self.__titlelayout, pad=(0, 0), background_color=_mblk,
					expand_x=True, border_width=0, grab=True ) ],
				[ sg.Frame( '',
					[ [ sg.Frame( '', self.__headerlayout, size=_frasz, pad=BPAD_TOP,
						expand_x=True,
						relief=sg.RELIEF_FLAT, border_width=0 ) ] ], pad=BPAD_LEFT,
					background_color=_blu, border_width=0, expand_x=True ), ],
				[ sg.Frame( '',
					[ [ sg.Frame( '', self.__firstlayout, size=_frasz, pad=
					BPAD_LEFT_INSIDE,
						border_width=0, expand_x=True, expand_y=True, ) ],
					  [ sg.Frame( '', self.__thirdlayout, size=_frasz, pad=
					  BPAD_LEFT_INSIDE,
						  border_width=0, expand_x=True, expand_y=True ) ] ],
					pad=BPAD_LEFT, background_color=_blu, border_width=0,
					expand_x=True, expand_y=True ),
				  sg.Frame( '',
					  [ [ sg.Frame( '', self.__secondlayout, size=_frasz,
						  pad=BPAD_LEFT_INSIDE,
						  border_width=0, expand_x=True, expand_y=True ) ],
					    [ sg.Frame( '', self.__fourthlayout, size=_frasz,
						    pad=BPAD_LEFT_INSIDE,
						    border_width=0, expand_x=True, expand_y=True ) ] ],
					  pad=BPAD_LEFT, background_color=_blu, border_width=0,
					  expand_x=True, expand_y=True ), ],
				[ sg.Sizegrip( background_color=_mblk ) ] ]
			_window = sg.Window( '  Booger', self.__formlayout,
				size=self.form_size,
				margins=(0, 0),
				background_color=_blk,
				grab_anywhere=True,
				no_titlebar=True,
				resizable=True,
				right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_LOC_EXIT )
			while True:
				_event, _values = _window.read( )
				print( _event, _values )
				if _event == sg.WIN_CLOSED or _event == 'Exit':
					break
				elif _event == 'Edit Me':
					sg.execute_editor( __file__ )
				elif _event == 'Version':
					sg.popup_scrolled( sg.get_versions( ), keep_on_top=True )
				elif _event == 'File Location':
					sg.popup_scrolled( 'This Python file is:', __file__ )
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'BudgetForm'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class ChartPanel( Dark ):
	'''

        Constructor:
        ChartPanel( )

        Purpose:
        Provides form with a bar chart

    '''
	
	
	@property
	def header( self ) -> str:
		if self.__header is not None:
			return self.__header
	
	
	@header.setter
	def header( self, value: str ):
		if value is not None:
			self.__header = value
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (750, 650)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_sm = (10, 1)
			_md = (15, 1)
			_lg = (20, 1)
			_xl = (100, 1)
			_width = 50
			_space = 75
			_offset = 3
			_graphsz = _datasz = (500, 500)
			_black = sg.theme_background_color( )
			
			_layout = [ [ sg.Text( size=_sm ), sg.Text( size=_xl ) ],
			            [ sg.Text( size=_sm ),
			              sg.Graph( _graphsz, (0, 0), _datasz, k='-GRAPH-' ) ],
			            [ sg.Text( size=_sm ), sg.Text( size=_xl ) ],
			            [ sg.Text( size=_lg ), sg.Button( 'Next', size=_md ),
			              sg.Text( size=_lg ), sg.Exit( size=_md ) ],
			            [ sg.Sizegrip( background_color=_black ) ] ]
			
			_window = sg.Window( 'Booger', _layout,
				finalize=True,
				resizable=True,
				icon=self.icon_path,
				font=self.theme_font,
				size=self.form_size )
			
			_graph = _window[ '-GRAPH-' ]
			
			while True:
				_graph.erase( )
				for i in range( 7 ):
					_item = random.randint( 0, _graphsz[ 1 ] )
					_graph.draw_rectangle( top_left=(i * _space + _offset, _item),
						bottom_right=(i * _space + _offset + _width, 0),
						fill_color=sg.theme_button_color_background( ),
						line_color=sg.theme_button_color_text( ) )
					
					_graph.draw_text( text=_item, color='#FFFFFF',
						location=(i * _space + _offset + 25, _item + 10) )
				
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, 'Exit'):
					break
			
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ChartForm'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class CsvForm( Dark ):
	'''

        Construcotr:
        CsvForm( )

        Purpose:
        Provides form that reads CSV file with pandas

	'''
	
	
	@property
	def header( self ) -> str:
		if self.__header is not None:
			return self.__header
	
	
	@header.setter
	def header( self, value: str ):
		if value is not None:
			self.__header = value
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (800, 600)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_sm = (3, 1)
			_med = (15, 1)
			_spc = (25, 1)
			_dialog = FileDialog( )
			_dialog.show( )
			_path = _dialog.selected_path
			
			if _path == '':
				_msg = MessageDialog( 'No file url was provided!' )
				_msg.show( )
				return
			
			_data = [ ]
			_header = [ ]
			
			_button = sg.popup_yes_no( 'Does file have column column_names?',
				icon=self.icon_path,
				font=self.theme_font )
			
			if _path is not None:
				try:
					_frame = CsvReader( _path, sep=',', engine='python', header=None )
					_data = _frame.values.tolist( )
					if _button == 'Yes':
						_header = _frame.iloc[ 0 ].tolist( )
						_data = _frame[ 1: ].column_values.tolist( )
					elif _button == 'No':
						_header = [ 'Column' + str( x ) for x in range( len( _data[ 0 ] ) ) ]
				except Exception:
					sg.popup_error( 'Error reading file' )
					return
			
			_left = [ [ sg.Text( size=_sm ), ] ]
			_right = [ [ sg.Text( size=_sm ), ] ]
			_datagrid = [ [ sg.Table( values=_data, headings=_header, justification='center',
				row_height=18, display_row_numbers=True, vertical_scroll_only=False,
				header_background_color='#1B262E', header_relief=sg.RELIEF_FLAT,
				header_border_width=1, selected_row_colors=('#FFFFFF', '#4682B4'),
				header_text_color='#FFFFFF', header_font=('Roboto', 8, 'bold'),
				font=('Roboto', 8), background_color='#EDF3F8',
				alternating_row_color='#EDF3F8', border_width=1, text_color='#000000',
				expand_x=True, expand_y=True, sbar_relief=sg.RELIEF_FLAT,
				num_rows=min( 26, len( _data ) ) ), ], ]
			_window = sg.Window( '  Booger', _datagrid, icon=self.icon_path,
				font=self.theme_font, resizable=True )
			_event, _values = _window.read( )
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'CsvForm'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ExcelForm( Dark ):
	'''

        Construcotr:
        ExcelForm( )

        Purpose:
        Provides form that reads CSV file with pandas

	'''
	
	
	@property
	def header( self ) -> str:
		if self.__header is not None:
			return self.__header
	
	
	@header.setter
	def header( self, value: str ):
		if value is not None:
			self.__header = value
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (1250, 700)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_small = (3, 1)
			_med = (15, 1)
			_spc = (25, 1)
			_dialog = FileDialog( )
			_dialog.show( )
			_filename = _dialog.selected_item
			
			if _filename == '':
				_msg = MessageDialog( 'No file was provided!' )
				_msg.show( )
				return
			
			_data = [ ]
			_header = [ ]
			
			_button = sg.popup_yes_no( 'First Row Has Headers?',
				title='Headers?',
				icon=self.icon_path,
				font=('Roboto', 10) )
			if _filename is not None:
				try:
					_dataframe = ExcelReader( _filename, index_col=0 )
					_data = _dataframe.values.tolist( )
					if _button == 'Yes':
						_header = [ f'{i} ' for i in _dataframe.columns ]
					elif _button == 'No':
						_header = [ 'Column-' + str( x ) for x in range( len( _data[ 0 ] ) ) ]
				except:
					sg.popup_error( 'Error reading file' )
					return
			_left = [ [ sg.Text( size=_small ), ] ]
			_right = [ [ sg.Text( size=_small ), ] ]
			_datagrid = [ [ sg.Table( values=_data, headings=_header, justification='center',
				row_height=18, display_row_numbers=True, vertical_scroll_only=False,
				header_background_color='#1B262E', header_relief=sg.RELIEF_FLAT,
				header_border_width=1, selected_row_colors=('#FFFFFF', '#4682B4'),
				header_text_color='#FFFFFF', header_font=('Roboto', 10, 'bold'),
				font=('Roboto', 8), background_color='#EDF3F8',
				alternating_row_color='#EDF3F8', border_width=1, text_color='#000000',
				expand_x=False, expand_y=True, sbar_relief=sg.RELIEF_FLAT,
				num_rows=min( 26, len( _data ) ) ), ], ]
			_layout = [ [ sg.Text( size=(3, 3) ) ],
			            [ sg.Column( _left, expand_x=True ),
			              sg.Column( _datagrid, expand_x=True, expand_y=True ),
			              sg.Column( _right, expand_x=True ) ],
			            [ sg.Text( size=_small ) ],
			            [ sg.Text( size=(10, 1) ), sg.Button( 'Open', size=_med,
				            key='-OPEN-' ),
			              sg.Text( size=_spc ), sg.Button( 'Export', size=_med,
				            key='-EXPORT-' ),
			              sg.Text( size=_spc ), sg.Button( 'Save', size=_med, key='-SAVE-' ),
			              sg.Text( size=_spc ), sg.Button( 'Close', size=_med, key='-CLOSE-'
			            ) ],
			            [ sg.Sizegrip( ) ], ]
			_window = sg.Window( ' Booger', _layout,
				size=self.form_size,
				grab_anywhere=True,
				icon=self.icon_path,
				font=self.theme_font,
				resizable=True,
				right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_SETTINGS_EXIT )
			_event, _values = _window.read( )
			if _event in (sg.WIN_X_EVENT, '-CLOSE-'):
				_window.close( )
			elif _event in ('-OPEN-', '-EXPORT-', '-SAVE-', 'Save'):
				_info = 'Not Yet Implemented!'
				_msg = MessageDialog( _info )
				_msg.show( )
				_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ExcelForm'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class GraphForm( Dark ):
	'''

        Construcotr:
        GraphForm( )

        Purpose:
        Provides form that reads CSV file with pandas

	'''
	
	
	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (800, 600)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'theme_background',
		         'theme_textcolor', 'element_forecolor', 'element_backcolor',
		         'text_backcolor', 'text_forecolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		def create_axis_grid( ):
			plt.close( 'all' )
			
			
			def get_demo_image( ):
				_delta = 0.5
				_extent = (-3, 4, -4, 3)
				_xaxis = np.arange( -3.0, 4.001, _delta )
				_yaxis = np.arange( -4.0, 3.001, _delta )
				_X, _Y = np.meshgrid( _xaxis, _yaxis )
				_Z1 = np.exp( -_X ** 2 - _Y ** 2 )
				_Z2 = np.exp( -(_X - 1) ** 2 - (_Y - 1) ** 2 )
				_Z = (_Z1 - _Z2) * 2
				
				return _Z, _extent
			
			
			def get_rgb( ):
				_Z, _extent = get_demo_image( )
				_Z[ _Z < 0 ] = 0.
				_Z = _Z / _Z.max( )
				_red = _Z[ :13, :13 ]
				_green = _Z[ 2:, 2: ]
				_blue = _Z[ :13, 2: ]
				return _red, _green, _blue
			
			
			_figure = plt.figure( 1 )
			_axis = RGBAxes( _figure, [ 0.1, 0.1, 0.8, 0.8 ] )
			_r, _g, _b = get_rgb( )
			_kwargs = dict( origin="lower", interpolation="nearest" )
			_axis.imshow_rgb( _r, _g, _b, **_kwargs )
			_axis.RGB.set_xlim( 0., 9.5 )
			_axis.RGB.set_ylim( 0.9, 10.6 )
			plt.draw( )
			return plt.gcf( )
		
		
		def create_figure( ):
			_figure = matplotlib.figure.Figure( figsize=(5, 4), dpi=100 )
			_data = np.arange( 0, 3, .01 )
			_figure.add_subplot( 111 ).plot( _data, 2 * np.sin( 2 * np.pi * _data ) )
			return _figure
		
		
		def create_subplot_3d( ):
			_figure = plt.figure( )
			_axis = _figure.add_subplot( 1, 2, 1, projection='3d' )
			_x = np.arange( -5, 5, 0.25 )
			_y = np.arange( -5, 5, 0.25 )
			_x, _y = np.meshgrid( _x, _y )
			_r = np.sqrt( _x ** 2 + _y ** 2 )
			_z = np.sin( _r )
			surf = _axis.plot_surface( _x, _y, _z, rstride=1, cstride=1, cmap=cm,
				linewidth=0, antialiased=False )
			
			_axis.set_zlim3d( -1.01, 1.01 )
			_figure.colorbar( surf, shrink=0.5, aspect=5 )
			_axis = _figure.add_subplot( 1, 2, 2, projection='3d' )
			_x, _y, _z = get_test_data( )
			_axis.plot_wireframe( _x, _y, _z, rstride=10, cstride=10 )
			return _figure
		
		
		def create_pyplot_scales( ):
			plt.close( 'all' )
			np.random.seed( 19680801 )
			
			_y = np.random.normal( loc=0.5, scale=0.4, size=1000 )
			_y = _y[ (_y > 0) & (_y < 1) ]
			_y.sort( )
			_x = np.arange( len( _y ) )
			
			# create_graph with various axes scales
			plt.figure( 1 )
			
			# linear
			plt.subplot( 221 )
			plt.plot( _x, _y )
			plt.yscale( 'linear' )
			plt.title( 'linear' )
			plt.grid( True )
			
			# log
			plt.subplot( 222 )
			plt.plot( _x, _y )
			plt.yscale( 'log' )
			plt.title( 'log' )
			plt.grid( True )
			
			# symmetric log
			plt.subplot( 223 )
			plt.plot( _x, _y - _y.mean( ) )
			plt.yscale( 'symlog', linthreshy=0.01 )
			plt.title( 'symlog' )
			plt.grid( True )
			
			# logit
			plt.subplot( 224 )
			plt.plot( _x, _y )
			plt.yscale( 'logit' )
			plt.title( 'logit' )
			plt.grid( True )
			plt.gca( ).yaxis.set_minor_formatter( NullFormatter( ) )
			plt.subplots_adjust( top=0.92, bottom=0.08, left=0.10, right=0.95,
				hspace=0.25,
				wspace=0.35 )
			
			return plt.gcf( )
		
		
		def draw_figure( element, figure ):
			plt.close( 'all' )
			_canvas = FigureCanvasAgg( figure )
			_buffer = io.BytesIO( )
			_canvas.print_figure( _buffer, format='png' )
			if _buffer is None:
				return None
			_buffer.seek( 0 )
			element.update( data=_buffer.read( ) )
			return _canvas
		
		
		_figures = {
			'Axis Grid': create_axis_grid,
			'Subplot 3D': create_subplot_3d,
			'Scales': create_pyplot_scales,
			'Basic Figure': create_figure }
		
		
		def create_window( ):
			_leftcolumn = [ [ sg.T( 'Charts' ) ],
			                [ sg.Listbox( list( _figures ),
				                default_values=[ list( _figures )[ 0 ] ], size=(15, 5),
				                key='-LB-' ) ],
			                [ sg.T( 'Styles' ) ],
			                [ sg.Combo( plt.style.available, size=(15, 10), key='-STYLE-' ) ],
			                [ sg.T( 'Themes' ) ],
			                [ sg.Combo( sg.theme_list( ),
				                default_value=sg.theme( ),
				                size=(15, 10),
				                key='-THEME-' ) ] ]
			
			_layout = [ [ sg.T( 'Budget Chart', font=('Roboto', 10) ) ],
			            [ sg.Col( _leftcolumn ), sg.Image( key='-IMAGE-' ) ],
			            [ sg.B( 'Draw' ), sg.B( 'Exit' ) ] ]
			
			_window = sg.Window( 'Booger', _layout, finalize=True )
			
			return _window
		
		
		_window = create_window( )
		
		while True:
			_event, _values = _window.read( )
			print( _event, _values )
			if _event == 'Exit' or _event == sg.WIN_CLOSED:
				break
			if _event == 'Draw':
				if _values[ '-THEME-' ] != sg.theme( ):
					_window.close( )
					sg.theme( _values[ '-THEME-' ] )
					_window = create_window( )
				
				if _values[ '-LB-' ]:
					_func = _figures[ _values[ '-LB-' ][ 0 ] ]
					if _values[ '-STYLE-' ]:
						plt.style.use( _values[ '-STYLE-' ] )
					
					draw_figure( _window[ '-IMAGE-' ], _func( ) )
		
		_window.close( )


class FileBrowser( ):
	'''
        File Chooser - with clearable history
        This is a design pattern that is very useful for programs that you run often that requires
        a filename be entered.  You've got 4 options to use to get your filename with this pattern:
        1. Copy and paste a filename into the combo element
        2. Use the last used item which will be visible when you generate_text the window
        3. Choose an item from the get_list of previously used items
        4. Browse for a new name
        To clear the get_list of previous entries, click the "Clear History" button.
        The history is stored in a json file using the PySimpleGUI User Settings APIs
        The code is as sparse as possible to enable easy integration into your code.
        Copyright 2021-2023 PySimpleSoft, Inc. and/or its licensors. All rights reserved.
        Redistribution, modification, or any other use of PySimpleGUI or any portion thereof
        is subject to the terms of the PySimpleGUI License Agreement
        available at https://eula.pysimplegui.com.
        You may not redistribute, modify or otherwise use PySimpleGUI or its contents except
         pursuant to the PySimpleGUI License Agreement.
    '''
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (400, 200)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		layout = [ [ sg.Combo( sorted( sg.user_settings_get_entry( '-filenames-', [ ] ) ),
			default_value=sg.user_settings_get_entry( '-last filename-', '' ),
			size=(50, 1), key='-FILENAME-' ), sg.FileBrowse( ), sg.B( 'Clear History' ) ],
		           [ sg.Button( 'Ok', bind_return_key=True ), sg.Button( 'Cancel' ) ] ]
		
		window = sg.Window( 'Browser File System', layout )
		while True:
			event, values = window.read( )
			
			if event in (sg.WIN_CLOSED, 'Cancel'):
				break
			if event == 'Ok':
				sg.user_settings_set_entry( '-filenames-', list( set(
					sg.user_settings_get_entry( '-filenames-', [ ] ) + [
						values[ '-FILENAME-' ], ] ) ) )
				sg.user_settings_set_entry( '-last filename-', values[ '-FILENAME-' ] )
				break
			elif event == 'Clear History':
				sg.user_settings_set_entry( '-filenames-', [ ] )
				sg.user_settings_set_entry( '-last filename-', '' )
				window[ '-FILENAME-' ].update( values=[ ], value='' )
		
		window.close( )


class ChatWindow( ):
	'''

	    Function to generate a chat window

	'''
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (800, 600)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( 'Your query will go here', size=(40, 1) ) ],
			            [ sg.Output( size=(110, 20), font=('Roboto 11') ) ],
			            [ sg.Multiline( size=(70, 5), enter_submits=True, key='-QUERY-',
				            do_not_clear=False ),
			              sg.Button( 'SEND', button_color=(sg.YELLOWS[ 0 ], sg.BLUES[ 0 ]),
				              bind_return_key=True ),
			              sg.Button( 'EXIT', button_color=(sg.YELLOWS[ 0 ], sg.GREENS[ 0 ]) ) ] ]
			
			window = sg.Window( 'Chat Window', _layout,
				font=('Roboto', ' 11'),
				keep_on_top = True,
				default_button_element_size=(8, 2),
				use_default_focus=False,
				size=self.form_size )
			
			# The Event Loop
			while True:
				event, values = window.read( )
				# quit if exit button or target_values
				if event in (sg.WIN_CLOSED, 'EXIT'):
					break
				if event == 'SEND':
					query = values[ '-QUERY-' ].rstrip( )
					print( 'The query you entered was {}'.format( query ), flush=True )
			
			window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ChatWindow'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class ChatBot( ):
	
	# -------  Make a new Window  ------- #
	# give our form a spiffy pairs of colors
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (800, 600)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] comprised of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			layout = [ [ sg.Text( 'Your query will go here', size=(40, 1) ) ],
			           [ sg.Output( size=(127, 30), font=('Rooboto 11') ) ],
			           [ sg.Text( 'Command History' ),
			             sg.Text( '', size=(20, 3), key='-HISTORY-' ) ],
			           [ sg.ML( size=(85, 5), enter_submits=True, key='-QUERY-',
				           do_not_clear=False ),
			             sg.Button( 'SEND',
				             button_color=(sg.YELLOWS[ 0 ], sg.BLUES[ 0 ]),
				             bind_return_key=True ),
			             sg.Button( 'EXIT',
				             button_color=(sg.YELLOWS[ 0 ], sg.GREENS[ 0 ]) ) ] ]
			
			window = sg.Window( 'Chat window with history', layout,
				default_element_size=(30, 2),
				font=('Roboto', ' 11'),
				default_button_element_size=(8, 2),
				return_keyboard_events=True,
				size=self.form_size )
			
			# ---===--- Loop taking in user path and using it  --- #
			command_history = [ ]
			history_offset = 0
			
			while True:
				event, value = window.read( )
				
				if event == 'SEND':
					query = value[ '-QUERY-' ].rstrip( )
					# EXECUTE YOUR COMMAND HERE
					print( 'The command you entered was {}'.format( query ) )
					command_history.append( query )
					history_offset = len( command_history ) - 1
					# manually clear path because keyboard events blocks clear
					window[ '-QUERY-' ].update( '' )
					window[ '-HISTORY-' ].update( '\n'.join( command_history[ -3: ] ) )
				
				elif event in (sg.WIN_CLOSED, 'EXIT'):  # quit if exit event or target_values
					break
				
				elif 'Up' in event and len( command_history ):
					command = command_history[ history_offset ]
					# decrement is not zero
					history_offset -= 1 * (history_offset > 0)
					window[ '-QUERY-' ].update( command )
				
				elif 'Down' in event and len( command_history ):
					# increment up to end of get_list
					history_offset += 1 * (history_offset < len( command_history ) - 1)
					command = command_history[ history_offset ]
					window[ '-QUERY-' ].update( command )
				
				elif 'Escape' in event:
					window[ '-QUERY-' ].update( '' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ChatBot'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class InputWindow( ):
	"""
	    Demo sg.Columns and sg.Frames
	    Demonstrates using mixture of sg.Column and sg.Frame elements to generate_text a nice window
	    layout.
	    A couple of the concepts shown here include:
	    * Using sg.Columns and sg.Frames with specific sizes on them
	    * Buttons that have the same documents on them that arew differentiated using explicit keys
	    * One way to hard-code the size of a Frame is to hard-code the size of a Column inside the
	    frame

	    CAUTION:
	        Using explicit sizes on Column and Frame elements may not have the same effect on
	        all computers.  Hard coding parts of layouts can sometimes not have the same result on
	        all computers.

	    There are 3 sg.Columns.  Two are side by side at the top_p and the third is along the
	    bottom

	    When there are multiple Columns on a row, be aware that the default is for those Columns
	    to be
	    aligned along their center.  If you want them to be top_p-aligned, then you need to use the
	    vtop helper function to make that happen.

	    Copyright 2021-2023 PySimpleSoft, Inc. and/or its licensors. All rights reserved.

	    Redistribution, modification, or any other use of PySimpleGUI or any portion thereof is
	    subject to the terms of the PySimpleGUI License Agreement available at
	    https://eula.pysimplegui.com.

	    You may not redistribute, modify or otherwise use PySimpleGUI or its contents except
	    pursuant to the PySimpleGUI License Agreement.
	"""
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (520, 550)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] comprised of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			col2 = sg.Column( [ [ sg.Frame( 'Accounts:',
				[ [ sg.Column( [ [ sg.Listbox( [ 'Account ' + str( i ) for i in range( 1, 16 ) ],
					key='-ACCT-LIST-', size=(15, 20) ), ] ], size=(150, 400) ) ] ] ) ] ],
				pad=(0, 0) )
			
			col1 = sg.Column( [
				# Categories sg.Frame
				[ sg.Frame( 'Categories:', [ [ sg.Radio( 'Websites', 'radio1', default=True,
					key='-WEBSITES-', size=(10, 1) ),
				                               sg.Radio( 'Software', 'radio1',
					                               key='-SOFTWARE-',
					                               size=(10, 1) ) ] ], ) ],
				# Information sg.Frame
				[ sg.Frame( 'Information:',
					[ [ sg.Text( ), sg.Column( [ [ sg.Text( 'Account:' ) ],
					                             [ sg.Input(
						                             key='-ACCOUNT-IN-',
						                             size=(19, 1) ) ],
					                             [ sg.Text( 'User Id:' ) ],
					                             [ sg.Input(
						                             key='-USERID-IN-',
						                             size=(19, 1) ),
						                             sg.Button( 'Copy',
							                             key='-USERID-' ) ],
					                             [ sg.Text( 'Password:' ) ],
					                             [ sg.Input( key='-PW-IN-',
						                             size=(19, 1) ),
					                               sg.Button( 'Copy',
						                               key='-PASS-' ) ],
					                             [ sg.Text( 'Location:' ) ],
					                             [ sg.Input( key='-LOC-IN-',
						                             size=(19, 1) ),
					                               sg.Button( 'Copy',
						                               key='-LOC-' ) ],
					                             [ sg.Text( 'Notes:' ) ],
					                             [ sg.Multiline(
						                             key='-NOTES-',
						                             size=(25, 3) ) ],
					                             ], size=(235, 350),
						pad=(0, 0) ) ] ] ) ], ], pad=(0, 0) )
			
			col3 = sg.Column( [ [ sg.Frame( 'Actions:',
				[ [ sg.Column(
					[ [ sg.Button( 'Save' ), sg.Button( 'Clear' ), sg.Button( 'Delete' ), ] ],
					size=(450, 45), pad=(0, 0) ) ] ] ) ] ], pad=(0, 0) )
			
			# The final layout is a simple one
			layout = [ [ col1, col2 ],
			           [ col3 ] ]
			
			# A perhaps better layout would have been to use the vtop layout helpful function.
			# This would allow the col2 column to have a different height and still be top_p
			# aligned
			# layout = [sg.vtop([col1, col2]),
			#           [col3]]
			window = sg.Window( 'Columns and Frames', layout,
				size=self.form_size )
			
			while True:
				event, values = window.read( )
				print( event, values )
				if event == sg.WIN_CLOSED:
					break
				
				window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'InputWindow'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class Executable( ):
	'''
	    Make a "Windows os" executable with PyInstaller
	    Copyright 2023 PySimpleSoft, Inc. and/or its licensors.
	    All rights reserved.
	    Redistribution, modification, or any other use of PySimpleGUI or any
	    portion thereof is subject to the terms of the PySimpleGUI
	    License Agreement available at https://eula.pysimplegui.com.
	    You may not redistribute, modify or otherwise use PySimpleGUI or
	    its contents except pursuant to the PySimpleGUI License Agreement.
	'''
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (600, 600)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] comprised of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			layout = [ [ sg.Text( 'PyInstaller EXE Creator', font='Any 15' ) ],
			           [ sg.Text( 'Source Python File' ),
			             sg.Input( key='-sourcefile-', size=(45, 1) ),
			             sg.FileBrowse( file_types=(("Python Files", "*.py"),) ) ],
			           [ sg.Text( 'Icon File' ), sg.Input( key='-iconfile-', size=(45, 1) ),
			             sg.FileBrowse( file_types=(("Icon Files", "*.ico"),) ) ],
			           [ sg.Frame( 'Output', font='Any 15', layout=[
				           [ sg.Output( size=(65, 15), font='Courier 10' ) ] ] ) ],
			           [ sg.Button( 'Make EXE', bind_return_key=True ),
			             sg.Button( 'Quit', button_color=('white', 'firebrick3') ) ],
			           [ sg.Text( 'Made with PySimpleGUI (www.PySimpleGUI.org)',
				           auto_size_text=True,
				           font='Courier 8' ) ] ]
			
			window = sg.Window( 'PySimpleGUI EXE Maker', layout,
				size=self.form_size,
				auto_size_text=False,
				auto_size_buttons=False,
				default_element_size=(20, 1),
				text_justification='right' )
			# ---===--- Loop taking in user path --- #
			while True:
				event, values = window.read( )
				if event in ('Exit', 'Quit', None):
					break
				
				source_file = values[ '-sourcefile-' ]
				icon_file = values[ '-iconfile-' ]
				icon_option = '-i "{}"'.format( icon_file ) if icon_file else ''
				source_path, source_filename = os.path.split( source_file )
				workpath_option = '--workpath "{}"'.format( source_path )
				dispath_option = '--distpath "{}"'.format( source_path )
				specpath_option = '--specpath "{}"'.format( source_path )
				folder_to_remove = os.path.join( source_path, source_filename[ :-3 ] )
				file_to_remove = os.path.join( source_path, source_filename[ :-3 ] + '.spec' )
				command_line = 'pyinstaller -wF --clean "{}" {} {} {} {}'.format( source_file,
					icon_option, workpath_option, dispath_option, specpath_option )
				
				if event == 'Make EXE':
					try:
						print( command_line )
						print( 'Making EXE...the program has NOT locked up...' )
						window.refresh( )
						# print('Running command {}'.format(command_line))
						out, err = runCommand( command_line, window=window )
						shutil.rmtree( folder_to_remove )
						os.remove( file_to_remove )
						print( '**** DONE ****' )
					except:
						sg.PopupError( 'Something went wrong',
							'close this window and copy command line from documents printed out '
							'in '
							'main window',
							'Here is the cleaned_lines from the run', out )
						print(
							'Copy and paste this line into the command prompt to manually run '
							'PyInstaller:\n\n',
							command_line )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'Executable'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def run_command( cmd, timeout=None, window=None ):
		"""

			Purpose:
			--------
            run shell command

			Parameters:
			-----------
            @param cmd: command to execute
            @param timeout: timeout for command execution

			Returns:
			--------
            @return: (return code from command, command cleaned_lines)

		"""
		try:
			p = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT )
			output = ''
			for line in p.stdout:
				line = line.decode( errors='replace' if (sys.version_info) < (3, 5)
				else 'backslashreplace' ).rstrip( )
				output += line
				print( line )
				if window:
					window.Refresh( )
			
			retval = p.wait( timeout )
			
			return (retval, output)
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'Executable'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class ThemeSelector( ):
	'''
	    Purpose:
	    --------

	    Parameters:
	    ----------

	    Returns:
	    ---------

	'''
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = (300, 400)
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] comprised of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			layout = [ [ sg.Text( 'UI Theme Browser' ) ],
			           [ sg.Text( 'Click a look and feel color to see demo window' ) ],
			           [ sg.Listbox( values=sg.theme_list( ),
				           size=(20, 20), key='-LIST-', enable_events=True ) ],
			           [ sg.Button( 'Exit' ) ] ]
			
			window = sg.Window( 'Look and Feel Browser', layout,
				size=self.form_size )
			
			# Event Loop
			while True:
				event, values = window.read( )
				if event in (sg.WIN_CLOSED, 'Exit'):
					break
				sg.theme( values[ '-LIST-' ][ 0 ] )
				sg.popup_get_text( 'This is {}'.format( values[ '-LIST-' ][ 0 ] ),
					default_text=values[ '-LIST-' ][ 0 ] )
			
			window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'ThemeSelector'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class UrlImageViewer( ):
	'''
	    Purpose:
	    --------

	    Parameters:
	    ----------

	    Returns:
	    ---------

	'''
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.form_size = ( 800, 600 )
	
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] comprised of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size', 'settings_path', 'theme_background',
		         'theme_textcolor', 'element_backcolor', 'element_forecolor',
		         'text_forecolor', 'text_backcolor', 'input_backcolor',
		         'input_forecolor', 'button_color', 'button_backcolor',
		         'button_forecolor', 'icon_path', 'theme_font',
		         'scrollbar_color', 'input_text', 'show' ]
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			image_URL = (r'https://upload.wikimedia.org/wikipedia/commons/4/47'
			             r'/PNG_transparency_demonstration_1.png')
			layout = [ [ sg.Image( urllib.request.urlopen( image_URL ).read( ) ) ] ]
			window = sg.Window( 'Image From URL', layout,
				size=self.form_size )
			while True:
				event, values = window.read( )
				if event == sg.WIN_CLOSED or event == 'Exit':
					break
			
			window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'UrlImageViewer'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class AutoComplete( ):
	"""


		Purpose:
		--------
	    Autocomplete path
	
	    There are 3 keyboard characters to be aware of:
	    * Arrow up - Change selected item in get_list
	    * Arrow down - Change selected item in get_list
	    * Escape - Erase the path and start over
	    * Return/Enter - use the current item selected from the get_list
	
	    You can easily remove the ignore case option by searching for the "Irnore Case" Check box
	    key:
	        '-IGNORE CASE-'
	
	    The variable "choices" holds the get_list of strings your program will match against.
	    Even though the listbox of choices doesn't have a scrollbar visible, the get_list is longer
	    than shown
	        and using your keyboard more of it will br shown as you scroll down with the arrow keys
	    The selection wraps around from the end to the start (and vicea versa). You can change
	    this behavior to
	        make it stay at the beignning or the end
	"""
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		self.choices = None
		self.input_width = None
		self.num_items_to_show = None
		self.list_element = None
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			self.choices = sorted( [ elem.__name__ for elem in sg.Element.__subclasses__( ) ] )
			
			self.input_width = 20
			self.num_items_to_show = 4
			
			self.layout = [
				[ sg.CB( 'Ignore Case', k='-IGNORE CASE-' ) ],
				[ sg.Text( 'Input PySimpleGUI Element Name:' ) ],
				[ sg.Input( size=(self.input_width, 1), enable_events=True, key='-IN-' ) ],
				[ sg.pin( sg.Col( [ [ sg.Listbox( values=[ ], size=(self.input_width,
				                                                    self.num_items_to_show),
					enable_events=True, key='-BOX-',
					select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, no_scrollbar=True ) ] ],
					key='-BOX-CONTAINER-', pad=(0, 0), visible=False ) ) ]
			]
			
			self.window = sg.Window( 'AutoComplete', self.layout,
				return_keyboard_events=True,
				finalize=True,
				font=('Roboto', 11) )
			
			self.list_element = self.window.Element( '-BOX-' )
			self.prediction_list, self.input_text, self.selected_item = [ ], "", 0
			
			# Event Loop
			while True:
				event, values = self.window.read( )
				# print(event, target_values)
				if event == sg.WINDOW_CLOSED:
					break
				# pressing down arrow will trigger event -IN- then aftewards event Down:40
				elif event.startswith( 'Escape' ):
					self.window[ '-IN-' ].update( '' )
					self.window[ '-BOX-CONTAINER-' ].update( visible=False )
				elif event.startswith( 'Down' ) and len( self.prediction_list ):
					self.selected_item = (self.selected_item + 1) % len( self.prediction_list )
					self.list_element.update( set_to_index=self.selected_item,
						scroll_to_index=self.selected_item )
				elif event.startswith( 'Up' ) and len( self.prediction_list ):
					self.selected_item = (self.selected_item + (
								len( self.prediction_list ) - 1)) % len( self.prediction_list )
					self.list_element.update( set_to_index=self.selected_item,
						scroll_to_index=self.selected_item )
				elif event == '\r':
					if len( values[ '-BOX-' ] ) > 0:
						self.window[ '-IN-' ].update( value=values[ '-BOX-' ] )
						self.window[ '-BOX-CONTAINER-' ].update( visible=False )
				elif event == '-IN-':
					text = values[ '-IN-' ] if not values[ '-IGNORE CASE-' ] else values[
						'-IN-' ].lower( )
					if text == self.input_text:
						continue
					else:
						self.input_text = text
					self.prediction_list = [ ]
					if text:
						if values[ '-IGNORE CASE-' ]:
							self.prediction_list = [ item for item in self.choices if
							                         item.lower( ).startswith( text ) ]
						else:
							self.prediction_list = [ item for item in self.choices if
							                         item.startswith( text ) ]
					
					self.list_element.update( values=self.prediction_list )
					self.selected_item = 0
					self.list_element.update( set_to_index=self.selected_item )
					
					if len( self.prediction_list ) > 0:
						self.window[ '-BOX-CONTAINER-' ].update( visible=True )
					else:
						self.window[ '-BOX-CONTAINER-' ].update( visible=False )
				elif event == '-BOX-':
					self.window[ '-IN-' ].update( value=values[ '-BOX-' ] )
					self.window[ '-BOX-CONTAINER-' ].update( visible=False )
			
			self.window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'UrlImageViewer'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )


class CheckBox( ):
	"""

		Purpose:
		----------
	    The Base64 Image encoding feature of PySimpleGUI makes it possible to create_small_embedding beautiful GUIs
	    very simply

	    These 2 checkboxes required 3 extra words of code than a normal checkbox.
	    1. Keep track of the current value using the Image Element's Metadata
	    2. Changle / Update the image when clicked
	    3. The Base64 image definition

	    Enable the event on the Image with the checkbox so that you can take action (flip the
	    value)

	"""
	
	
	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
		self.checked = \
			(
				b'iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAKMGlDQ1BJQ0MgUHJvZmlsZQAAeJydlndUVNcWh8'
				b'+9d3qhzTAUKUPvvQ0gvTep0kRhmBlgKAMOMzSxIaICEUVEBBVBgiIGjIYisSKKhYBgwR6QIKDEYBRRUXkzslZ05eW9l5ffH2d9a5+99z1n733WugCQvP25vHRYCoA0noAf4uVKj4yKpmP7AQzwAAPMAGCyMjMCQj3DgEg+Hm70TJET+CIIgDd3xCsAN428g+h08P9JmpXBF4jSBInYgs3JZIm4UMSp2YIMsX1GxNT4FDHDKDHzRQcUsbyYExfZ8LPPIjuLmZ3GY4tYfOYMdhpbzD0i3pol5IgY8RdxURaXky3iWyLWTBWmcUX8VhybxmFmAoAiie0CDitJxKYiJvHDQtxEvBQAHCnxK47/igWcHIH4Um7pGbl8bmKSgK7L0qOb2doy6N6c7FSOQGAUxGSlMPlsult6WgaTlwvA4p0/S0ZcW7qoyNZmttbWRubGZl8V6r9u/k2Je7tIr4I/9wyi9X2x/ZVfej0AjFlRbXZ8scXvBaBjMwDy97/YNA8CICnqW/vAV/ehieclSSDIsDMxyc7ONuZyWMbigv6h/+nwN/TV94zF6f4oD92dk8AUpgro4rqx0lPThXx6ZgaTxaEb/XmI/3HgX5/DMISTwOFzeKKIcNGUcXmJonbz2FwBN51H5/L+UxP/YdiftDjXIlEaPgFqrDGQGqAC5Nc+gKIQARJzQLQD/dE3f3w4EL+8CNWJxbn/LOjfs8Jl4iWTm/g5zi0kjM4S8rMW98TPEqABAUgCKlAAKkAD6AIjYA5sgD1wBh7AFwSCMBAFVgEWSAJpgA+yQT7YCIpACdgBdoNqUAsaQBNoASdABzgNLoDL4Dq4AW6DB2AEjIPnYAa8AfMQBGEhMkSBFCBVSAsygMwhBuQIeUD+UAgUBcVBiRAPEkL50CaoBCqHqqE6qAn6HjoFXYCuQoPQPWgUmoJ+h97DCEyCqbAyrA2bwAzYBfaDw+CVcCK8Gs6DC+HtcBVcDx+D2+EL8HX4NjwCP4dnEYAQERqihhghDMQNCUSikQSEj6xDipFKpB5pQbqQXuQmMoJMI+9QGBQFRUcZoexR3qjlKBZqNWodqhRVjTqCakf1oG6iRlEzqE9oMloJbYC2Q/ugI9GJ6Gx0EboS3YhuQ19C30aPo99gMBgaRgdjg/HGRGGSMWswpZj9mFbMecwgZgwzi8ViFbAGWAdsIJaJFWCLsHuxx7DnsEPYcexbHBGnijPHeeKicTxcAa4SdxR3FjeEm8DN46XwWng7fCCejc/Fl+Eb8F34Afw4fp4gTdAhOBDCCMmEjYQqQgvhEuEh4RWRSFQn2hKDiVziBmIV8TjxCnGU+I4kQ9InuZFiSELSdtJh0nnSPdIrMpmsTXYmR5MF5O3kJvJF8mPyWwmKhLGEjwRbYr1EjUS7xJDEC0m8pJaki+QqyTzJSsmTkgOS01J4KW0pNymm1DqpGqlTUsNSs9IUaTPpQOk06VLpo9JXpSdlsDLaMh4ybJlCmUMyF2XGKAhFg+JGYVE2URoolyjjVAxVh+pDTaaWUL+j9lNnZGVkLWXDZXNka2TPyI7QEJo2zYeWSiujnaDdob2XU5ZzkePIbZNrkRuSm5NfIu8sz5Evlm+Vvy3/XoGu4KGQorBToUPhkSJKUV8xWDFb8YDiJcXpJdQl9ktYS4qXnFhyXwlW0lcKUVqjdEipT2lWWUXZSzlDea/yReVpFZqKs0qySoXKWZUpVYqqoypXtUL1nOozuizdhZ5Kr6L30GfUlNS81YRqdWr9avPqOurL1QvUW9UfaRA0GBoJGhUa3RozmqqaAZr5ms2a97XwWgytJK09Wr1ac9o62hHaW7Q7tCd15HV8dPJ0mnUe6pJ1nXRX69br3tLD6DH0UvT2693Qh/Wt9JP0a/QHDGADawOuwX6DQUO0oa0hz7DecNiIZORilGXUbDRqTDP2Ny4w7jB+YaJpEm2y06TX5JOplWmqaYPpAzMZM1+zArMus9/N9c1Z5jXmtyzIFp4W6y06LV5aGlhyLA9Y3rWiWAVYbbHqtvpobWPNt26xnrLRtImz2WczzKAyghiljCu2aFtX2/W2p23f2VnbCexO2P1mb2SfYn/UfnKpzlLO0oalYw7qDkyHOocRR7pjnONBxxEnNSemU73TE2cNZ7Zzo/OEi55Lsssxlxeupq581zbXOTc7t7Vu590Rdy/3Yvd+DxmP5R7VHo891T0TPZs9Z7ysvNZ4nfdGe/t57/Qe9lH2Yfk0+cz42viu9e3xI/mF+lX7PfHX9+f7dwXAAb4BuwIeLtNaxlvWEQgCfQJ3BT4K0glaHfRjMCY4KLgm+GmIWUh+SG8oJTQ29GjomzDXsLKwB8t1lwuXd4dLhseEN4XPRbhHlEeMRJpEro28HqUYxY3qjMZGh0c3Rs+u8Fixe8V4jFVMUcydlTorc1ZeXaW4KnXVmVjJWGbsyTh0XETc0bgPzEBmPXM23id+X/wMy421h/Wc7cyuYE9xHDjlnIkEh4TyhMlEh8RdiVNJTkmVSdNcN24192Wyd3Jt8lxKYMrhlIXUiNTWNFxaXNopngwvhdeTrpKekz6YYZBRlDGy2m717tUzfD9+YyaUuTKzU0AV/Uz1CXWFm4WjWY5ZNVlvs8OzT+ZI5/By+nL1c7flTuR55n27BrWGtaY7Xy1/Y/7oWpe1deugdfHrutdrrC9cP77Ba8ORjYSNKRt/KjAtKC94vSliU1ehcuGGwrHNXpubiySK+EXDW+y31G5FbeVu7d9msW3vtk/F7OJrJaYllSUfSlml174x+6bqm4XtCdv7y6zLDuzA7ODtuLPTaeeRcunyvPKxXQG72ivoFcUVr3fH7r5aaVlZu4ewR7hnpMq/qnOv5t4dez9UJ1XfrnGtad2ntG/bvrn97P1DB5wPtNQq15bUvj/IPXi3zquuvV67vvIQ5lDWoacN4Q293zK+bWpUbCxp/HiYd3jkSMiRniabpqajSkfLmuFmYfPUsZhjN75z/66zxailrpXWWnIcHBcef/Z93Pd3Tvid6D7JONnyg9YP+9oobcXtUHtu+0xHUsdIZ1Tn4CnfU91d9l1tPxr/ePi02umaM7Jnys4SzhaeXTiXd272fMb56QuJF8a6Y7sfXIy8eKsnuKf/kt+lK5c9L1/sdek9d8XhyumrdldPXWNc67hufb29z6qv7Sern9r6rfvbB2wGOm/Y3ugaXDp4dshp6MJN95uXb/ncun572e3BO8vv3B2OGR65y747eS/13sv7WffnH2x4iH5Y/EjqUeVjpcf1P+v93DpiPXJm1H2070nokwdjrLHnv2T+8mG88Cn5aeWE6kTTpPnk6SnPqRvPVjwbf57xfH666FfpX/e90H3xw2/Ov/XNRM6Mv+S/XPi99JXCq8OvLV93zwbNPn6T9mZ+rvitwtsj7xjvet9HvJ+Yz/6A/VD1Ue9j1ye/Tw8X0hYW/gUDmPP8uaxzGQAAAp1JREFUeJzFlk1rE1EUhp9z5iat9kMlVXGhKH4uXEo1CoIKrnSnoHs3unLnxpW7ipuCv0BwoRv/gCBY2/gLxI2gBcHGT9KmmmTmHBeTlLRJGquT+jJ3djPPfV/OPefK1UfvD0hIHotpsf7jm4mq4k6mEsEtsfz2gpr4rGpyPYjGjyUMFy1peNg5odkSV0nNDNFwxhv2JAhR0ZKGA0JiIAPCpgTczaVhRa1//2qoprhBQdv/LSKNasVUVAcZb/c9/A9oSwMDq6Rr08DSXNW68TN2pAc8U3CLsVQ3bpwocHb/CEs16+o8ZAoVWKwZNycLXD62DYDyUszbLzW2BMHa+lIm4Fa8lZpx6+QEl46OA1CaX+ZjpUFeV0MzAbecdoPen1lABHKRdHThdcECiNCx27XQxTXQufllHrxaIFKItBMK6xSXCCSeFsoKZO2m6AUtE0lvaE+wCPyKna055erx7SSWul7pes1Xpd4Z74OZhfQMrwOFLlELYAbjeeXuud0cKQyxZyzHw9efGQ6KStrve8WrCpHSd7J2gL1Jjx0qvxIALh4aIxJhulRmKBKWY+8Zbz+nLXWNWgXqsXPvxSfm5qsAXDg4yu3iLn7Gzq3Jv4t3XceQxpSLQFWZelnmztldnN43wvmDoxyeGGLvtlyb0z+Pt69jSItJBfJBmHpZXnG+Gtq/ejcMhtSBCuQjYWqmzOyHFD77oZo63WC87erbudzTGAMwXfrM2y81nr+rIGw83nb90XQyh9Ccb8/e/CAxCF3aYOZgaB4zYDSffvKvN+ANz+NefXvg4KykbmabDXU30/yOguKbyHYnNzKuwUnmhPxpF3Ok19UsM2r6BEpB6n7NpPFU6smpuLpoqCgZFdCKBDC3MDKmntNSVEuu/AYecjifoa3JogAAAABJRU5ErkJggg==')
		self.unchecked = \
			(
				b'iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAKMGlDQ1BJQ0MgUHJvZmlsZQAAeJydlndUVNcWh8'
				b'+9d3qhzTAUKUPvvQ0gvTep0kRhmBlgKAMOMzSxIaICEUVEBBVBgiIGjIYisSKKhYBgwR6QIKDEYBRRUXkzslZ05eW9l5ffH2d9a5+99z1n733WugCQvP25vHRYCoA0noAf4uVKj4yKpmP7AQzwAAPMAGCyMjMCQj3DgEg+Hm70TJET+CIIgDd3xCsAN428g+h08P9JmpXBF4jSBInYgs3JZIm4UMSp2YIMsX1GxNT4FDHDKDHzRQcUsbyYExfZ8LPPIjuLmZ3GY4tYfOYMdhpbzD0i3pol5IgY8RdxURaXky3iWyLWTBWmcUX8VhybxmFmAoAiie0CDitJxKYiJvHDQtxEvBQAHCnxK47/igWcHIH4Um7pGbl8bmKSgK7L0qOb2doy6N6c7FSOQGAUxGSlMPlsult6WgaTlwvA4p0/S0ZcW7qoyNZmttbWRubGZl8V6r9u/k2Je7tIr4I/9wyi9X2x/ZVfej0AjFlRbXZ8scXvBaBjMwDy97/YNA8CICnqW/vAV/ehieclSSDIsDMxyc7ONuZyWMbigv6h/+nwN/TV94zF6f4oD92dk8AUpgro4rqx0lPThXx6ZgaTxaEb/XmI/3HgX5/DMISTwOFzeKKIcNGUcXmJonbz2FwBN51H5/L+UxP/YdiftDjXIlEaPgFqrDGQGqAC5Nc+gKIQARJzQLQD/dE3f3w4EL+8CNWJxbn/LOjfs8Jl4iWTm/g5zi0kjM4S8rMW98TPEqABAUgCKlAAKkAD6AIjYA5sgD1wBh7AFwSCMBAFVgEWSAJpgA+yQT7YCIpACdgBdoNqUAsaQBNoASdABzgNLoDL4Dq4AW6DB2AEjIPnYAa8AfMQBGEhMkSBFCBVSAsygMwhBuQIeUD+UAgUBcVBiRAPEkL50CaoBCqHqqE6qAn6HjoFXYCuQoPQPWgUmoJ+h97DCEyCqbAyrA2bwAzYBfaDw+CVcCK8Gs6DC+HtcBVcDx+D2+EL8HX4NjwCP4dnEYAQERqihhghDMQNCUSikQSEj6xDipFKpB5pQbqQXuQmMoJMI+9QGBQFRUcZoexR3qjlKBZqNWodqhRVjTqCakf1oG6iRlEzqE9oMloJbYC2Q/ugI9GJ6Gx0EboS3YhuQ19C30aPo99gMBgaRgdjg/HGRGGSMWswpZj9mFbMecwgZgwzi8ViFbAGWAdsIJaJFWCLsHuxx7DnsEPYcexbHBGnijPHeeKicTxcAa4SdxR3FjeEm8DN46XwWng7fCCejc/Fl+Eb8F34Afw4fp4gTdAhOBDCCMmEjYQqQgvhEuEh4RWRSFQn2hKDiVziBmIV8TjxCnGU+I4kQ9InuZFiSELSdtJh0nnSPdIrMpmsTXYmR5MF5O3kJvJF8mPyWwmKhLGEjwRbYr1EjUS7xJDEC0m8pJaki+QqyTzJSsmTkgOS01J4KW0pNymm1DqpGqlTUsNSs9IUaTPpQOk06VLpo9JXpSdlsDLaMh4ybJlCmUMyF2XGKAhFg+JGYVE2URoolyjjVAxVh+pDTaaWUL+j9lNnZGVkLWXDZXNka2TPyI7QEJo2zYeWSiujnaDdob2XU5ZzkePIbZNrkRuSm5NfIu8sz5Evlm+Vvy3/XoGu4KGQorBToUPhkSJKUV8xWDFb8YDiJcXpJdQl9ktYS4qXnFhyXwlW0lcKUVqjdEipT2lWWUXZSzlDea/yReVpFZqKs0qySoXKWZUpVYqqoypXtUL1nOozuizdhZ5Kr6L30GfUlNS81YRqdWr9avPqOurL1QvUW9UfaRA0GBoJGhUa3RozmqqaAZr5ms2a97XwWgytJK09Wr1ac9o62hHaW7Q7tCd15HV8dPJ0mnUe6pJ1nXRX69br3tLD6DH0UvT2693Qh/Wt9JP0a/QHDGADawOuwX6DQUO0oa0hz7DecNiIZORilGXUbDRqTDP2Ny4w7jB+YaJpEm2y06TX5JOplWmqaYPpAzMZM1+zArMus9/N9c1Z5jXmtyzIFp4W6y06LV5aGlhyLA9Y3rWiWAVYbbHqtvpobWPNt26xnrLRtImz2WczzKAyghiljCu2aFtX2/W2p23f2VnbCexO2P1mb2SfYn/UfnKpzlLO0oalYw7qDkyHOocRR7pjnONBxxEnNSemU73TE2cNZ7Zzo/OEi55Lsssxlxeupq581zbXOTc7t7Vu590Rdy/3Yvd+DxmP5R7VHo891T0TPZs9Z7ysvNZ4nfdGe/t57/Qe9lH2Yfk0+cz42viu9e3xI/mF+lX7PfHX9+f7dwXAAb4BuwIeLtNaxlvWEQgCfQJ3BT4K0glaHfRjMCY4KLgm+GmIWUh+SG8oJTQ29GjomzDXsLKwB8t1lwuXd4dLhseEN4XPRbhHlEeMRJpEro28HqUYxY3qjMZGh0c3Rs+u8Fixe8V4jFVMUcydlTorc1ZeXaW4KnXVmVjJWGbsyTh0XETc0bgPzEBmPXM23id+X/wMy421h/Wc7cyuYE9xHDjlnIkEh4TyhMlEh8RdiVNJTkmVSdNcN24192Wyd3Jt8lxKYMrhlIXUiNTWNFxaXNopngwvhdeTrpKekz6YYZBRlDGy2m717tUzfD9+YyaUuTKzU0AV/Uz1CXWFm4WjWY5ZNVlvs8OzT+ZI5/By+nL1c7flTuR55n27BrWGtaY7Xy1/Y/7oWpe1deugdfHrutdrrC9cP77Ba8ORjYSNKRt/KjAtKC94vSliU1ehcuGGwrHNXpubiySK+EXDW+y31G5FbeVu7d9msW3vtk/F7OJrJaYllSUfSlml174x+6bqm4XtCdv7y6zLDuzA7ODtuLPTaeeRcunyvPKxXQG72ivoFcUVr3fH7r5aaVlZu4ewR7hnpMq/qnOv5t4dez9UJ1XfrnGtad2ntG/bvrn97P1DB5wPtNQq15bUvj/IPXi3zquuvV67vvIQ5lDWoacN4Q293zK+bWpUbCxp/HiYd3jkSMiRniabpqajSkfLmuFmYfPUsZhjN75z/66zxailrpXWWnIcHBcef/Z93Pd3Tvid6D7JONnyg9YP+9oobcXtUHtu+0xHUsdIZ1Tn4CnfU91d9l1tPxr/ePi02umaM7Jnys4SzhaeXTiXd272fMb56QuJF8a6Y7sfXIy8eKsnuKf/kt+lK5c9L1/sdek9d8XhyumrdldPXWNc67hufb29z6qv7Sern9r6rfvbB2wGOm/Y3ugaXDp4dshp6MJN95uXb/ncun572e3BO8vv3B2OGR65y747eS/13sv7WffnH2x4iH5Y/EjqUeVjpcf1P+v93DpiPXJm1H2070nokwdjrLHnv2T+8mG88Cn5aeWE6kTTpPnk6SnPqRvPVjwbf57xfH666FfpX/e90H3xw2/Ov/XNRM6Mv+S/XPi99JXCq8OvLV93zwbNPn6T9mZ+rvitwtsj7xjvet9HvJ+Yz/6A/VD1Ue9j1ye/Tw8X0hYW/gUDmPP8uaxzGQAAAPFJREFUeJzt101KA0EQBeD3XjpBCIoSPYC3cPQaCno9IQu9h+YauYA/KFk4k37lYhAUFBR6Iko/at1fU4uqbp5dLg+Z8pxW0z7em5IQgaIhEc6e7M5kxo2ULxK1njNtNc5dpIN9lRU/RLZBpZPofJWIUePcBQAiG+BAbC8gwsHOjdqHO0PquaHQ92eT7FZPFqUh2/v5HX4DfUuFK1zhClf4H8IstDp/DJd6Ff2dVle4wt+Gw/am0Qhbk72ZEBu0IzCe7igF8i0xOQ46wFJz6Uu1r4RFYhvnZnfNNh+tV8+GKBT+s4EAHE7TbcVYi9FLPn0F1D1glFsARrAAAAAASUVORK5CYII=')
	
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			layout = [ [ sg.Text( 'Fancy Checkboxes... Simply' ) ],
			           [ sg.Image( self.checked, key=('-IMAGE-', 1), metadata=True,
				           enable_events=True ),
			             sg.Text( True, enable_events=True, k=('-TEXT-', 1) ) ],
			           [ sg.Image( self.unchecked, key=('-IMAGE-', 2), metadata=False,
				           enable_events=True ),
			             sg.Text( False, enable_events=True, k=('-TEXT-', 2) ) ],
			           [ sg.Button( 'Go' ), sg.Button( 'Exit' ) ] ]
			
			window = sg.Window( 'Custom Checkboxes', layout, font="_ 14" )
			while True:
				event, values = window.read( )
				print( event, values )
				if event == sg.WIN_CLOSED or event == 'Exit':
					break
				# if a checkbox is clicked, flip the vale and the image
				if event[ 0 ] in ('-IMAGE-', '-TEXT-'):
					cbox_key = ('-IMAGE-', event[ 1 ])
					text_key = ('-TEXT-', event[ 1 ])
					window[ cbox_key ].metadata = not window[ cbox_key ].metadata
					window[ cbox_key ].update(
						self.checked if window[ cbox_key ].metadata else self.unchecked )
					# Update the path next to the checkbox
					window[ text_key ].update( window[ cbox_key ].metadata )
			
			window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'UrlImageViewer'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )



class MachineLearningWindow( ):
	'''

	    Purpose:
	    --------


	    Parameters:
	    ----------



	    Returns:
	    ---------


	'''
	def __init__(self):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = r'C:\Users\terry\source\repos\Boo\resources\ico\ninja.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Boo', r'C:\Users\terry\source\repos\Boo\resources\theme' )
	
	def __build_window( self ):
		'''

		    Purpose:
		    --------


		    Parameters:
		    ----------



		    Returns:
		    ---------


		'''
		try:
			sg.set_options( text_justification='right' )
			
			flags = [ [ sg.CB( 'Normalize', size=(12, 1), default=True ),
			            sg.CB( 'Verbose', size=(20, 1) ) ],
			          [ sg.CB( 'BaseCluster', size=(12, 1) ), sg.CB(
				          'Flush Output', size=(20, 1), default=True ) ],
			          [ sg.CB( 'Write Results', size=(12, 1) ), sg.CB(
				          'Keep Intermediate Data', size=(20, 1) ) ],
			          [ sg.CB( 'Normalize', size=(12, 1), default=True ),
			            sg.CB( 'Verbose', size=(20, 1) ) ],
			          [ sg.CB( 'BaseCluster', size=(12, 1) ), sg.CB(
				          'Flush Output', size=(20, 1), default=True ) ],
			          [ sg.CB( 'Write Results', size=(12, 1) ),
			            sg.CB( 'Keep Intermediate Data', size=(20, 1) ) ], ]
			
			loss_functions = [ [ sg.Rad( 'Cross-Entropy', 'loss', size=(12, 1) ),
			                     sg.Rad( 'Logistic', 'loss', default=True, size=(12, 1) ) ],
			                   [ sg.Rad( 'Hinge', 'loss', size=(12, 1) ),
			                     sg.Rad( 'Huber', 'loss', size=(12, 1) ) ],
			                   [ sg.Rad( 'Kullerback', 'loss', size=(12, 1) ),
			                     sg.Rad( 'MAE(L1)', 'loss', size=(12, 1) ) ],
			                   [ sg.Rad( 'MSE(L2)', 'loss', size=(12, 1) ),
			                     sg.Rad( 'MB(L0)', 'loss', size=(12, 1) ) ], ]
			
			command_line_parms = [ [ sg.Text( 'Passes', size=(8, 1) ),
			                         sg.Spin( values=[ i for i in range( 1, 1000 ) ],
				                         initial_value=20, size=(6, 1) ),
			                         sg.Text( 'Steps', size=(8, 1), pad=((7, 3)) ),
			                         sg.Spin( values=[ i for i in range( 1, 1000 ) ],
				                         initial_value=20, size=(6, 1) ) ],
			                       [ sg.Text( 'ooa', size=(8, 1) ),
			                         sg.Input( default_text='6', size=(8, 1) ),
			                         sg.Text( 'nn', size=(8, 1) ),
			                         sg.Input( default_text='10', size=(10, 1) ) ],
			                       [ sg.Text( 'q', size=(8, 1) ),
			                         sg.Input( default_text='ff', size=(8, 1) ),
			                         sg.Text( 'ngram', size=(8, 1) ),
			                         sg.Input( default_text='5', size=(10, 1) ) ],
			                       [ sg.Text( 'l', size=(8, 1) ),
			                         sg.Input( default_text='0.4', size=(8, 1) ),
			                         sg.Text( 'Layers', size=(8, 1) ),
			                         sg.Drop( values=('BatchNorm', 'other') ) ], ]
			
			layout = [ [ sg.Frame( 'Command Line Parameteres', command_line_parms,
				title_color='green', font='Any 12' ) ],
			           [ sg.Frame( 'Flags', flags, font='Any 12', title_color='blue' ) ],
			           [ sg.Frame( 'Loss Functions', loss_functions,
				           font='Any 12', title_color='red' ) ],
			           [ sg.Submit( ), sg.Cancel( ) ] ]
			
			sg.set_options( text_justification='left' )
			
			window = sg.Window( 'Machine Learning',
				layout, font=("Helvetica", 12) )
			window.read( )
			window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'MachineLearningWindow'
			exception.method = '__build_window( self)'
			error = ErrorDialog( exception )
			error.show( )
	
	def __custom_meter( self ):
		'''

		    Purpose:
		    --------


		    Parameters:
		    ----------



		    Returns:
		    ---------


		'''
		try:
			# layout the form
			layout = [ [ sg.Text( 'A custom progress meter' ) ],
			           [ sg.ProgressBar( 1000, orientation='h',
				           size=(20, 20), key='progress' ) ],
			           [ sg.Cancel( ) ] ]
			
			# create_small_embedding the form`
			window = sg.Window( 'Custom Progress Meter', layout )
			progress_bar = window[ 'progress' ]
			# loop that would normally do something useful
			for i in range( 1000 ):
				# check to see if the cancel button was clicked and exit loop if clicked
				event, values = window.read( timeout=0, timeout_key='timeout' )
				if event == 'Cancel' or event == None:
					break
				# update bar with loop value +1 so that bar eventually reaches the maximum
				progress_bar.update_bar( i + 1 )
			# done with loop... need to destroy the window as it's still open
			window.CloseNonBlocking( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'MachineLearningWindow'
			exception.method = '__custom_meter( self)'
			error = ErrorDialog( exception )
			error.show( )
	
	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			self.__custom_meter( )
			self.__build_window( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Booger'
			exception.cause = 'MachineLearningWindow'
			exception.method = 'show( self)'
			error = ErrorDialog( exception )
			error.show( )

