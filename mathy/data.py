'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Data.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Data.py" company="Terry D. Eppler">

     Mathy Data

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
	Data.py
</summary>
******************************************************************************************
'''
from argparse import ArgumentError
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from Static import Scaler
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple


class Model( BaseModel ):
	"""

		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.

	"""

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		self.pipeline = None


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the linerar_model to the training df.

			Parameters:
			-----------
				X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
				y (np.ndarray): Target vector w/shape ( n_samples, ).

			Returns:
			--------
				None

		"""
		raise NotImplementedError


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Generate predictions from  the trained linerar_model.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).

			Returns:
			-----------
				np.ndarray: Predicted target_values or class labels.

		"""
		raise NotImplementedError


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Compute the core metric (e.g., R²) of the model on test df.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
				float: Score value (e.g., R² for regressors).

		"""
		raise NotImplementedError


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			---------
			Evaluate the model using multiple performance metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				dict: Dictionary containing multiple evaluation metrics.

		"""
		raise NotImplementedError


class Metric( BaseModel ):
	"""

		Purpose:
		---------
		Base interface for all preprocessors. Provides standard `fit`, `transform`, and
	    `fit_transform` methods.

	"""

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True

	def __init__( self ):
		self.pipeline = None
		self.transformed_data = [ ]
		self.transformed_values = [ ]


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
		"""

			Purpose:
			---------
			Fits the preprocessor to the text df.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (Optional[np.ndarray]): Optional target array.

		"""
		raise NotImplementedError


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the text df using the fitted preprocessor.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Purpose:
			---------
			Fits the preprocessor and then transforms the text df.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (Optional[np.ndarray]): Optional target array.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			self.fit( X, y )
			return self.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Metric'
			exception.method = ('fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray '
			                    ']=None'
			                    ') -> np.ndarray')
			error = ErrorDialog( exception )
			error.show( )


class Dataset( Metric ):
	"""

		Purpose:
		-----------
		Utility class for preparing machine rate datasets from a pandas DataFrame.

		Members:
		------------
		dataframe: pd.DataFrame
		data: np.ndarray
		rows: int
		columns: int
		target: str
		test_size: float
		random_state: int
		features: list
		target_values
		numeric_columns
		text_columns: list
		training_data: pd.DataFrame
		training_values
		testing_data
		testing_values

	"""

	def __init__( self, df: pd.DataFrame, target: str, size: float=0.2, rando: int=42 ):
		"""

			Purpose:
			-----------
			Initialize and split the dataset.

			Parameters:
			-----------
			df (pd.DataFrame): Matrix text vector.
			target List[ str ]: Name of the target columns.
			size (float): Proportion of df to use as test set.
			rando (int): Seed for reproducibility.

		"""
		super( ).__init__( )
		self.dataframe = df
		self.data = df[ 1:, : ]
		self.rows = len( df )
		self.columns = len( df.columns )
		self.target = target
		self.test_size = size
		self.random_state = rando
		self.features = [ column for column in df.columns ]
		self.target_values = [ value for value in df[ 1:, target ] ]
		self.numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
		self.text_columns = df.select_dtypes( include=[ 'object', 'category' ] ).columns.tolist( )
		self.training_data = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 0 ]
		self.testing_data = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 1 ]
		self.training_values = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 2 ]
		self.testing_values = \
		train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 3 ]
		self.transtuple = [ ]


	def __dir__( self ):
		'''

			Purpose:
			-----------
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'rows', 'columns', 'target', 'split_data',
		         'features', 'test_size', 'rando', 'df', 'scale_data',
		         'numeric_columns', 'text_columns', 'scaler', 'transtuple', 'create_testing_data',
		         'calculate_statistics', 'create_training_data',
		         'target_values', 'training_data', 'testing_data', 'training_values',
		         'testing_values', 'transform_columns' ]


	def transform_columns( self, name: str, encoder: object, columns: List[ str ] ) -> None:
		"""

			Purpose:
			-----------
				Scale numeric features using selected scaler.

			Paramters:
			-----------
				name - the name of the encoder
				encoder - the encoder object to transform the df.
				columns - the list of column names to apply the transformation to.

		"""
		try:
			if name is None:
				raise Exception( 'Arguent "name" cannot be None' )
			elif encoder is None:
				raise Exception( 'Arguent "encoder" cannot be None' )
			elif columns is None:
				raise Exception( 'Arguent "columns" cannot be None' )
			else:
				_tuple = (name, encoder, columns)
				self.transtuple.append( _tuple )
				self.column_transformer = ColumnTransformer( self.transtuple )
				self.column_transformer.fit_transform( self.data )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'transform_columns( self, name: str, encoder: object, columns: List[ str ] )'
			error = ErrorDialog( exception )
			error.show( )


	def calculate_statistics( self ) -> Dict:
		"""

			Purpose:
			-----------
			Method calculating descriptive statistics.

			Returns:
			-----------
			Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			statistics = self.dataframe.describe( )
			return statistics
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'caluclate_statistics( ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_training_data( self ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""

			Purpose:
			-----------
			Return the training features and labels.

			Returns:
			-----------
			Tuple[ np.ndarray, np.ndarray ]: ( training_data, training_values )

		"""
		return tuple( self.training_data, self.training_values )


	def create_testing_data( self ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""

			Purpose:
			-----------
			Return the test features and labels.

			Returns:
			-----------
			Tuple[ np.ndarray, np.ndarray ]: testing_data, testing_values

		"""
		return tuple( self.testing_data, self.testing_values )