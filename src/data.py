'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                target_values.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="target_values.py" company="Terry D. Eppler">

     This is a Federal Budget, Finance, and Accounting application.
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

     You can contact me at: terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    target_values.py
  </summary>
  ******************************************************************************************
  '''
import sqlite3 as sqlite
from pandas import DataFrame
from pandas import read_sql as sqlreader
import pyodbc as db
import os
from static import Source, Provider, SQL, ParamStyle
from boogr import Error, ErrorDialog

class Pascal( ):
	'''
	
		Constructor:
		Pascal( path: str )
	
		Purpose:
		Class splits path 'path' argument into Pascal Casing

	'''


	def __init__( self, input: str=None ):
		self.input = input
		self.output = input if input.istitle( ) else self.join( )


	def __str__( self ) -> str:
		if self.output is not None:
			return self.output


	def __dir__( self ) -> list[ str ]:
		'''
		
			Retunes a get_list[ str ] of member names.
		
		'''
		return [ 'path', 'cleaned_lines', 'split', 'join' ]


	def split( self ) -> str:
		'''
			
			Purpose:
	
			Parameters:
	
			Returns:
			
		'''

		try:
			_buffer = [ str( c ) for c in self.output ]
			_output = [ ]
			_retval = ''
			for char in _buffer:
				if char.islower( ):
					_output.append( char )
				elif char.isupper( ) and _buffer.index( char ) == 0.00:
					_output.append( char )
				elif char.isupper( ) and _buffer.index( char ) > 0.00:
					_output.append( ' ' )
					_output.append( char )
			for o in _output:
				_retval += f'{o}'
			return _retval
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'Pascal'
			exception.method = 'join( self )'
			error = ErrorDialog( exception )
			error.show( )


	def join( self ) -> str:
		'''
			
			Purpose:
	
			Parameters:
	
			Returns:
			
		'''

		try:
			if self.input.count( ' ' ) > 0:
				_buffer = [ str( c ) for c in self.input ]
				_output = [ ]
				_retval = ''
				for char in _buffer:
					if char != ' ':
						_output.append( char )
					elif char == ' ':
						_index = _buffer.index( char )
						_next = str( _buffer[ _index + 1 ] )
						if _next.islower( ):
							_cap = _next.upper( )
							_buffer.remove( _next )
							_buffer.insert( _index + 1, _cap )
						_buffer.remove( char )
						_output.append( _next.upper( ) )

			for o in _output:
				_retval += f'{o}'
			return _retval.replace( 'AH', 'Ah' ).replace( 'BOC', 'Boc' ) \
				.replace( 'RPIO', 'Rpio' ).replace( 'RC', 'Rc' ) \
				.replace( 'PRC', 'Prc' ).replace( 'ID', 'Id' ) \
				.replace( 'OMB', 'Omb' ).replace( 'NPM', 'Npm' ) \
				.replace( 'FOC', 'Foc' ).replace( 'ORG', 'Org' ) \
				.replace( 'THE', 'The' ).replace( 'OR', 'Or' ) \
				.replace( 'AND', 'And' ).replace( 'BUT', 'But' ) \
				.replace( 'OF', 'Of' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'Pascal'
			exception.method = 'join( self )'
			error = ErrorDialog( exception )
			error.show( )


class SqlPath( ):
	'''
	
		Constructor:
			SqlPath( )
	
		Purpose:
			Class providing relative_path paths to the
			folders containing sqlstatement files and driverinfo
			paths used in the application

	'''


	def __init__( self ):
		self.sqlite_driver = 'sqlite3'
		self.sqlite_path = r'data/sqlite/Boo.target_values'
		self.access_driver = r'DRIVER={ Microsoft Access Driver (*.mdb, *.accdb) };DBQ='
		self.access_path = os.getcwd( ) + r'\db\access\Boo.accdb'


	def __dir__( self ) -> list[ str ]:
		'''
		Retunes a get_list[ str ] of member names.
		'''
		return [ 'sqlite_driver', 'sqlite_database',
		         'access_driver', 'access_path' ]


class SqlFile( ):
	'''
	
		Constructor:
	
			SqlFile( source: Source=None, provider: Provider  = Provider.SQLite,
					command: SQL=SQL.SELECTALL )
	
		Purpose:
	
			Class providing access to sqlstatement sub-folders in the application provided
			optional arguments source, provider, and command.

	'''


	def __init__( self, source: Source=None, provider: Provider=Provider.SQLite,
	              commandtype: SQL = SQL.SELECTALL ):
		self.command_type = commandtype
		self.source = source
		self.provider = provider
		self.data = [ 'Appropriations',
		              'BudgetaryResourceExecution',
		              'BudgetFunctions',
		              'BudgetObjectClasses',
		              'FederalHolidays',
		              'FiscalYears',
		              'MainAccounts',
		              'ProductCategories',
		              'Regulations',
		              'ResourceLines',
		              'TreasurySymbols' ]


	def __dir__( self ) -> list[ str ]:
		'''
		
			Retunes a get_list[ str ] of member names.
			
		'''
		return [ 'source', 'provider', 'command_type', 'get_file_path',
		         'get_folder_path', 'get_command_text' ]


	def get_file_path( self ) -> str:
		'''

			Purpose:
	
			Parameters:
	
			Returns:

		'''

		try:
			_sqlpath = SqlPath( )
			_data = self.data
			_provider = self.provider.name
			_tablename = self.source.name
			_command = self.command_type.name
			_current = os.getcwd( )
			_filepath = ''
			if _provider == 'SQLite' and _tablename in _data:
				_filepath = f'{_sqlpath.sqlite_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
			elif _provider == 'ACCDB' and _tablename in _data:
				_filepath = f'{_sqlpath.access_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
			elif _provider == 'SqlServer' and _tablename in _data:
				_filepath = f'{_sqlpath.sqlserver_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
			else:
				_filepath = f'{_sqlpath.sqlite_database}\\{_command}\\{_tablename}.sql'
				return os.path.join( _current, _filepath )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlFile'
			exception.method = 'get_file_path( self )'
			error = ErrorDialog( exception )
			error.show( )


	def get_folder_path( self ) -> str:
		'''
		
			Purpose:
	
			Parameters:
	
			Returns:
			
		'''

		try:
			_sqlpath = SqlPath( )
			_data = self.data
			_source = self.source.name
			_provider = self.provider.name
			_command = self.command_type.name
			_current = os.getcwd( )
			_folder = ''
			if _provider == 'SQLite' and _source in _data:
				_folder = f'{_sqlpath.sqlite_database}\\{_command}'
				return os.path.join( _current, _folder )
			elif _provider == 'ACCDB' and _source in _data:
				_folder = f'{_sqlpath.access_database}\\{_command}'
				return os.path.join( _current, _folder )
			elif _provider == 'SqlServer' and _source in _data:
				_folder = f'{_sqlpath.sqlserver_database}\\{_command}'
				return os.path.join( _current, _folder )
			else:
				_folder = f'{_sqlpath.sqlite_database}\\{_command}'
				return os.path.join( _current, _folder )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlFile'
			exception.method = 'get_folder_path( self )'
			error = ErrorDialog( exception )
			error.show( )


	def get_command_text( self ) -> str:
		'''
			
			Purpose:
	
			Parameters:
	
			Returns:
			
		'''

		try:
			_source = self.source.name
			_paths = self.get_file_path( )
			_folder = self.get_folder_path( )
			_sql = ''
			for name in os.listdir( _folder ):
				if name.endswith( '.sql' ) \
						and os.path.splitext( name )[ 0 ] == self.source.name:
					_path = os.path.join( _folder, name )
					_query = open( _path )
					_sql = _query.read( )
				if _sql is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
				else:
					return _sql
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlFile'
			exception.method = 'get_command_text( self, other )'
			error = ErrorDialog( exception )
			error.show( )


class DbConfig( ):
	'''
	
		Constructor:
			DbConfig( source: Source, provider: Provider=Provider.SQLite )
	
		Purpose:
			Class provides get_list of Budget Execution tables across two databases

	'''


	def __init__( self, src: Source, pro: Provider=Provider.SQLite ):
		self.provider = pro
		self.source = src
		self.table_name = src.name
		self.sqlite_path = os.getcwd( ) + r'\target_values\sqlite\Boo.target_values'
		self.access_driver = r'DRIVER={ Microsoft Access Driver (*.mdb, *.accdb) };DBQ='
		self.access_path = os.getcwd( ) + r'\db\access\Boo.accdb'
		self.sqlserver_driver = r'DRIVER={ ODBC Driver 17 for SQL Server };SERVER=.\SQLExpress;'
		self.sqlserver_path = os.getcwd( ) + r'\db\sqlserver\Boo.mdf'
		self.data = [ 'BudgetaryResourceExecution',
		              'BudgetAuthority',
		              'BudgetOutlays'
		              'Appropriations',
		              'BudgetFunctions',
		              'BudgetObjectClasses',
		              'FederalHolidays',
		              'FiscalYears',
		              'MainAccounts',
		              'Partitions',
		              'ProductCategories',
		              'Regulations',
		              'ResourceLines',
		              'TreasurySymbols' ]


	def __str__( self ) -> str:
		if self.table_name is not None:
			return self.table_name


	def __dir__( self ) -> list[ str ]:
		'''
		Retunes a get_list[ str ] of member names.
		'''
		return [ 'source', 'provider', 'table_name', 'get_driver_info',
		         'sqlite_path', 'access_driver', 'access_path',
		         'sqlserver_driver', 'sqlserver_path',
		         'get_data_path', 'get_connection_string' ]


	def get_driver_info( self ) -> str:
		'''
	
			Purpose:
				Returns a path defining the driverinfo being used
	
			Parameters:  None
	
			Returns:  str

		'''
		try:
			if self.provider.name == 'SQLite':
				return self.sqlite_path
			elif self.provider.name == 'Access':
				return self.access_driver
			elif self.provider.name == 'SqlServer':
				return self.sqlserver_driver
			else:
				return self.sqlite_driver
		except Exception as e:
			exception = Error( e )
			exception.cause = 'DbConfig Class'
			exception.method = 'getdriver_info( self )'
			error = ErrorDialog( exception )
			error.show( )


	def get_data_path( self ) -> str:
		'''
	
			Purpose:
	
			Parameters:
	
			Returns:

		'''

		try:
			if self.provider.name == 'SQLite':
				return self.sqlite_path
			elif self.provider.name == 'Access':
				return self.access_path
			elif self.provider.name == 'SqlServer':
				return self.sqlserver_path
			else:
				return self.sqlite_path
		except Exception as e:
			exception = Error( e )
			exception.cause = 'DbConfig Class'
			exception.method = 'get_data_path( self )'
			error = ErrorDialog( exception )
			error.show( )


	def get_connection_string( self ) -> str:
		'''
	
			Purpose:
	
			Parameters:
	
			Returns:

		'''

		try:
			_path = self.get_data_path( )
			if self.provider.name == Provider.Access.name:
				return self.get_driver_info( ) + _path
			elif self.provider.name == Provider.SqlServer.name:
				return r'DRIVER={ ODBC Driver 17 for SQL Server };Server=.\SQLExpress;' \
					+ f'AttachDBFileName={self.source.name}' \
					+ f'DATABASE={_path}Trusted_Connection=yes;'
			else:
				return f'{_path} '
		except Exception as e:
			exception = Error( e )
			exception.cause = 'DbConfig Class'
			exception.method = 'get_connection_string( self )'
			error = ErrorDialog( exception )
			error.show( )


class Connection( DbConfig ):
	'''
	
		Constructor:
			Connection( source: Source, provider: Provider=Provider.SQLite )
	
		Purpose:
			Class providing object used to connect to the databases

	'''

	def __init__( self, src: Source, pro: Provider=Provider.SQLite ):
		super( ).__init__( src, pro )
		self.source = super( ).source
		self.provider = super( ).provider
		self.data_path = super( ).get_data_path( )
		self.driver = super( ).get_driver_info( )
		self.dsn = super( ).table_name + ';'
		self.connection_string = super( ).get_connection_string( )


	def __dir__( self ) -> list[ str ]:
		'''
		
			Retunes a get_list[ str ] of member names.
			
		'''
		return [ 'source', 'provider', 'table_name', 'getdriver_info',
		         'get_data_path', 'get_connection_string',
		         'driver_info', 'data_path',
		         'connection_string', 'connect' ]


	def connect( self ):
		'''
			Purpose:
				Establishes a target_values connections using the connecdtion
				path.
	
			Parameters:
				self
	
			Returns:
				None
		'''

		try:
			if self.provider.name == Provider.Access.name:
				return db.connect( self.connection_string )
			elif self.provider.name == Provider.SqlServer.name:
				return db.connect( self.connection_string )
			else:
				return sqlite.connect( self.connection_string )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'Connection'
			exception.method = 'connect( self )'
			error = ErrorDialog( exception )
			error.show( )


class SqlConfig( ):
	'''

	 Constructor:

		 SqlConfig( commandtype: SQL=SQL.SELECTALL, columnnames: get_list = None,
					columnvalues: tuple=None, paramstyle: ParamStyle = None )

	 Purpose:

		 Class provides database interaction behavior

	 '''

	def __init__( self, cmd: SQL = SQL.SELECTALL, names: list[ str ]=None,
	              values: tuple=None, paramstyle: ParamStyle = None ):
		self.command_type = cmd
		self.column_names = names
		self.column_values = values
		self.parameter_style = paramstyle
		self.criteria = dict( zip( names, list( values ) ) ) \
			if names is not None and values is not None else None


	def __dir__( self ) -> list[ str ]:
		'''

		Returns a get_list[ str ] of member names.

		'''
		return [ 'command_type', 'column_names', 'column_values',
		         'parameter_style', 'pair_dump', 'where_dump',
		         'set_dump', 'column_dump', 'value_dump' ]


	def pair_dump( self ) -> str:
		'''
		Purpose:

		Parameters:

		Returns:
		'''

		try:
			if self.column_names is not None and self.column_values is not None:
				_pairs = ''
				_kvp = zip( self.column_names, self.column_values )
				for k, v in _kvp:
					_pairs += f'{k} = \'{v}\' AND '
				_criteria = _pairs.rstrip( ' AND ' )
				if _criteria is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _criteria
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlConfig'
			exception.method = 'pair_dump( self )'
			error = ErrorDialog( exception )
			error.show( )


	def where_dump( self ) -> str:
		'''
		Purpose:

		Parameters:

		Returns:
		'''

		try:
			if (isinstance( self.column_names, list ) and
					isinstance( self.column_values, tuple )):
				pairs = ''
				for k, v in zip( self.column_names, self.column_values ):
					pairs += f'{k} = \'{v}\' AND '
				_criteria = 'WHERE ' + pairs.rstrip( ' AND ' )
				if _criteria is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _criteria
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlConfig'
			exception.method = 'where_dump( self )'
			error = ErrorDialog( exception )
			error.show( )


	def set_dump( self ) -> str:
		'''

		Purpose:

		Parameters:

		Returns:

		'''

		try:
			if self.column_names is not None and self.column_values is not None:
				_pairs = ''
				_criteria = ''
				for k, v in zip( self.column_names, self.column_values ):
					_pairs += f'{k} = \'{v}\', '
				_criteria = 'SET ' + _pairs.rstrip( ', ' )
				if _criteria is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _criteria
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlConfig'
			exception.method = 'set_dump( self )'
			error = ErrorDialog( exception )
			error.show( )


	def column_dump( self ) -> str:
		'''
		Purpose:

		Parameters:

		Returns:
		'''

		try:
			if self.column_names is not None:
				_colnames = ''
				for n in self.column_names:
					_colnames += f'{n}, '
				_columns = '(' + _colnames.rstrip( ', ' ) + ')'
				if _columsn is None:
					_msg = 'INVALID INPUT!'
					raise ValueError( _msg )
			else:
				return _columns
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlConfig'
			exception.method = 'column_dump( self )'
			error = ErrorDialog( exception )
			error.show( )


	def value_dump( self ) -> str:
		'''
		Purpose:

		Parameters:

		Returns:
		'''

		try:
			if self.column_values is not None:
				_vals = ''
				for v in self.column_values:
					_vals += f'{v}, '
					_values = 'VALUES (' + _vals.rstrip( ', ' ) + ')'
					if _values is None:
						_msg = 'INVALID INPUT!'
						raise ValueError( _msg )
			else:
				return _values
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'SqlConfig'
			exception.method = 'value_dump( self )'
			error = ErrorDialog( exception )
			error.show( )


class DataGenerator( ):
	'''
	
		Constructor:
	
			DataGenerator( source: Source )
	
		Purpose:
	
			Class containing factory method for providing
			pandas dataframes.

	'''


	def __init__( self, src: Source ):
		self.source = src
		self.table_name = src.name
		self.data_path = DbConfig( src ).get_data_path( )
		self.command_text = f'SELECT * FROM {src.name};'


	def __dir__( self ) -> list[ str ]:
		'''
			Returns a get_list[ str ] of member names
		'''
		return [ 'source', 'data_path', 'table_name',
		         'command_text', 'create_frame', 'create_tuples' ]


	def create_frame( self ) -> DataFrame:
		'''

			Purpose:
	
			Parameters:
	
			Returns:

		'''

		try:
			_connection = sqlite.connect( self.data_path )
			_sql = f'SELECT * FROM {self.source.name};'
			_frame = sqlreader( _sql, _connection )
			if _frame is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _frame
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'DataGenerator'
			exception.method = 'create_frame( self )'
			error = ErrorDialog( exception )
			error.show( )


	def create_tuples( self ) -> list[ tuple ]:
		'''
	
			Purpose:
	
			Parameters:
	
			Returns:

		'''

		try:
			_connection = sqlite.connect( self.data_path )
			_sql = f'SELECT * FROM {self.source.name};'
			_frame = sqlreader( _sql, _connection )
			_data = [ tuple( i ) for i in _frame.iterrows( ) ]
			if _data is None:
				_msg = "INVALID INPUT!"
				raise ValueError( _msg )
			else:
				return _data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Data'
			exception.cause = 'DataGenerator'
			exception.method = 'create_tuples( self )'
			error = ErrorDialog( exception )
			error.show( )
