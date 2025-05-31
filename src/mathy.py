'''
******************************************************************************************
  Assembly:                Boo
  Filename:                mathy.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2023

  Last Modified By:        Terry D. Eppler
  Last Modified On:        06-01-2023
******************************************************************************************
<copyright file="mathy.py" company="Terry D. Eppler">

 Bobo is a target_values analysis tool for EPA Analysts.
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
boo.py
</summary>
******************************************************************************************
'''
from argparse import ArgumentError

import numpy as np
from typing import Optional, List, Dict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (
	LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
	BayesianRidge, SGDClassifier, SGDRegressor, Perceptron
)
from sklearn.ensemble import (
	RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
	BaggingClassifier, VotingClassifier, StackingClassifier,
	RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
	BaggingRegressor, VotingRegressor, StackingRegressor
)
from sklearn.metrics import (
	r2_score, mean_squared_error, mean_absolute_error,
	explained_variance_score, median_absolute_error
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import (
	StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
	OneHotEncoder, OrdinalEncoder
)

from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
import pandas as pd
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple


class Model( BaseModel ):
	"""
	
		Abstract base class
		that defines the interface for all linerar_model wrappers.
	
	"""
	pipeline: Optional[ Pipeline ]


	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True


	def __init__( self ):
		self.pipeline = None


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the linerar_model to
			the training data.
	
			Parameters:
				X (np.ndarray): Feature vector w/shape ( n_samples, n_features ).
				y (np.ndarray): Target vector w/shape ( n_samples, ).
	
			Returns:
				None
			
		"""
		raise NotImplementedError


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Generate predictions from
			 the trained linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
	
			Returns:
				np.ndarray: Predicted target_values or class labels.
			
		"""
		raise NotImplementedError


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the core metric
			(e.g., R²) of the model on test data.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target target_values.
	
			Returns:
				float: Score value (e.g., R² for regressors).
			
		"""
		raise NotImplementedError


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluate the model using
			 multiple performance metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				dict: Dictionary containing multiple evaluation metrics.
			
		"""
		raise NotImplementedError


class Metric( BaseModel ):
	"""

		Base interface for all
		preprocessors. Provides standard `fit`, `transform`, and
		`fit_transform` methods.

	"""
	pipeline: Optional[ Pipeline ]
	scaled_values: Optional[ np.ndarray ]


	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True


	def __init__( self ):
		self.pipeline = None
		self.scaled_values = None


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
		"""

			Fits the preprocessor
			to the text data.

			Args:
				X (pd.DataFrame): Feature matrix.
				y (Optional[np.ndarray]): Optional target array.

		"""
		raise NotImplementedError


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the text
			data using the fitted preprocessor.

			Args:
				X (pd.DataFrame): Feature matrix.

			Returns:
				np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""

			Fits the preprocessor and
			then transforms the text data.

			Args:
				X (pd.DataFrame): Feature matrix.
				y (Optional[np.ndarray]): Optional target array.

			Returns:
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
		Utility class for preparing machine rate
		datasets from a pandas DataFrame.

	"""
	data: np.ndarray
	target: List[ str ]
	test_size: float
	random_state: int
	scaler: str
	dataframe: Optional[ pd.DataFrame ]
	row_count: Optional[ int ]
	column_count: Otional[ int ]
	feature_names: Optional[ List[ str ] ]
	target_values: np.ndarray
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	numeric_features: Optional[ List[ str ] ]
	categorical_features: Optional[ List[ str ] ]


	def __init__( self, data: np.ndarray, target: List[ str ], size: float=0.2,
	              rando: int=42, scaler: str= 'standard' ):
		"""

			Purpose:
			Initialize and split the dataset.

			Parameters:
				data (np.ndarray): Matrix text vector.
				target List[ str ]: Name of the target columns.
				size (float): Proportion of data to use as test set.
				random_state (int): Seed for reproducibility.

		"""
		self.data = data
		self.dataframe = pd.DataFrame( data = self.data, columns = self.data[ 0, : ],
			index = self.data[ :, 0 ] )
		self.row_count = len( self.dataframe )
		self.column_count = len( self.dataframe.columns )
		self.target = target
		self.feature_names = [ column for column in self.dataframe.columns ]
		self.target_values = [ value for value in self.dataframe[ 1:, [ target ] ] ]
		self.test_size = size
		self.random_state = state
		self.scaler = scaler
		self.numeric_features = self.dataframe.select_dtypes(
			include = [ 'number' ] ).columns.tolist( )
		self.categorical_features = self.dataframe.select_dtypes(
			include = [ 'object', 'category' ] ).columns.tolist( )
		self.X_train = train_test_split( self.dataframe, self.target,
			test_size = self.test_size, random_state = self.random_state )[ 0 ]
		self.X_test = train_test_split( self.dataframe, self.target,
			test_size = self.test_size, random_state = self.random_state )[ 1 ]
		self.y_train = train_test_split( self.dataframe, self.target,
			test_size = self.test_size, random_state = self.random_state )[ 2 ]
		self.y_test = train_test_split( self.dataframe, self.target,
			test_size = self.test_size, random_state = self.random_state )[ 3 ]


	def __dir__( self ):
		'''

			Purpose:
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'row_count', 'column_count', 'target', 'split_data',
		         'feature_names', 'test_size', 'random_state', 'data', 'scale_data',
		         'numeric_features', 'categorical_features', 'scaler',
		         'create_testing_data', 'calculate_statistics', 'create_training_data',
		         'target_values', 'X_train', 'X_test', 'y_train', 'y_test' ]


	def scale_data( self, type: str='standard' ) -> None:
		"""

			Purpose:
				Scale numeric features using selected scaler.

			Raises:
				ValueError: If scaler scaler is not 'standard' or 'minmax'.

		"""
		try:
			if type is None:
				raise Exception( 'The text argument "scaler" is None' )
			elif self.scaler == 'standard':
				_standard = StandardScaler( )
				self.scaled_values = _standard.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns=self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			elif self.scaler == 'minmax':
				_minmax = MinMaxScaler( )
				self.scaled_values = _minmax.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns = self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			elif self.scaler == 'simple':
				_simple = SimpleImputer( )
				self.scaled_values = _minmax.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns = self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			elif self.scaler == 'neighbor':
				_nearest = NeighborImputer( )
				self.scaled_values = _minmax.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns = self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			elif self.scaler == 'robust':
				_robust = RobustScaler( )
				self.scaled_values = _robust.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns = self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			elif self.scaler == 'normal':
				_normal = Normalizer( )
				self.scaled_values = _minmax.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns = self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			elif self.scaler == 'onehot':
				_onehot = OneHotEncoder( )
				self.scaled_values = _minmax.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns = self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			elif self.scaler == 'ordinal':
				_ordinal = OrdinalEncoder( )
				self.scaled_values = _ordinal.fit_transform( self.data[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns = self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
			else:
				_standard = StandardScaler( )
				self.scaled_values = _standard.fit_transform( self.X[ self.numeric_features ] )
				_df = pd.DataFrame( data=self.scaled_values, columns=self.numeric_features,
					index=self.data.index )
				self.dataframe = pd.concat( [ _df, self.data[ self.categorical_features ] ],
					axis=1 )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'scale_data( )'
			error = ErrorDialog( exception )
			error.show( )


	def split_data( self, data: np.ndarray, target: np.ndarray, size: float=0.20,
	                rando: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			Split the dataset into training and test sets.

			Returns:
				Tuple[ pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray ]

		"""
		try:
			if data is None:
				raise ArgumentError( 'data is not provided.' )
			elif target is None:
				raise ArgumentError( 'target is not provided.' )
			else:
				self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( data,
					target,
					size, rando )
				return tuple( self.X_train, self.X_test, self.y_train, self.y_test )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'split_data( self, data: np.ndarray, target: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )


	def calculate_statistics( self ) -> Dict:
		"""

			Purpose:
			Split the dataset into training and test sets.

			Returns:
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
				Return the training feature_names and labels.
	
			Returns:
				Tuple[ np.ndarray, np.ndarray ]: ( X_train, y_train )
				
		"""
		return tuple( self.X_train, self.y_train )


	def create_testing_data( self ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			Return the test feature_names and labels.
	
			Returns:
				Tuple[ np.ndarray, np.ndarray ]: X_test, y_test
				
		"""
		return tuple( self.X_test, self.y_test )


class StandardScaler( Metric ):
	"""

		Standardizes feature_names by
		removing the mean and scaling to unit variance.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = StandardScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

			Fits the standard_scaler
			to the data.

			Args:
				X (np.ndarray): Input data.
				y (Optional[np.ndarray]): Ignored.

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.pipeline = self.standard_scaler.fit( X )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the data
			using the fitted StandardScaler.

			Args:
				X (np.ndarray): Input data.

			Returns:
				np.ndarray: Scaled data.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.scaled_values = self.standard_scaler.transform( X )
				return self.scaled_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class MinMaxScaler( Metric ):
	"""

		Scales feature_names to
		a given range (default is [0, 1]).

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.minmax_scaler = MinMaxScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

			Fits the standard_scaler
			to the data.

			Args:
				X (np.ndarray): Input data.
				y (Optional[np.ndarray]): Ignored.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.minmax_scaler.fit( X )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the data
			using the fitted MinMaxScaler.

			Args:
				X (np.ndarray): Input data.

			Returns:
				np.ndarray: Scaled data.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.scaled_values = self.minmax_scaler.transform( X )
				return self.scaled_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class RobustScaler( Metric ):
	"""

		Scales feature_names using statistics
		that are robust to outliers.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.robust_scaler = RobustScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

				Fits the standard_scaler
				to the data.

				Args:
					X (np.ndarray): Input data.
					y (Optional[np.ndarray]): Ignored.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.robust_scaler.fit( X )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RobustScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the data
			using the fitted RobustScaler.

			Args:
				X (np.ndarray): Input data.

			Returns:
				np.ndarray: Scaled data.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.robust_scaler.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class Normalizer( Metric ):
	"""

		Scales text vectors individually to unit norm.

	"""


	def __init__( self, norm: str = 'l2' ) -> None:
		super( ).__init__( )
		self.normal_scaler = Normalizer( norm = norm )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

			Fits the normalizer
			(no-op for Normalizer).

			Args:
				X (np.ndarray): Input data.
				y (Optional[np.ndarray]): Ignored.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.normal_scaler.fit( X )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Normalizer'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Applies normalization
			to each sample.

			Args:
				X (np.ndarray): Input data.

			Returns:
				np.ndarray: Normalized data.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.normal_scaler.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Normalizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class OneHotEncoder( Metric ):
	"""

		Encodes categorical feature_names
		 as a one-hot numeric array.

	"""


	def __init__( self, handle_unknown: str = 'ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = OneHotEncoder( sparse = False, handle_unknown = handle_unknown )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

			Fits the hot_encoder
			to the categorical data.

			Args:
				X (np.ndarray): Categorical text data.
				y (Optional[np.ndarray]): Ignored.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.hot_encoder.fit( X )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the text
			data into a one-hot encoded format.

			Args:
				X (np.ndarray): Categorical text data.

			Returns:
				np.ndarray: One-hot encoded matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.hot_encoder.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OneHotEncoder'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class OrdinalEncoder( Metric ):
	"""

		Encodes categorical
		feature_names as ordinal integers.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = OrdinalEncoder( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

			Purpose:
			________
			Fits the ordial_encoder
			to the categorical data.

			Args:
			_____
				X (np.ndarray): Categorical text data.
				y (Optional[np.ndarray]): Ignored.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.ordinal_encoder.fit( X )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the text
			data into ordinal-encoded format.

			Args:
				X (np.ndarray): Categorical text data.

			Returns:
				np.ndarray: Ordinal-encoded matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.ordinal_encoder.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class SimpleImputer( Metric ):
	"""

		Fills missing target_values
		using a specified strategy.

	"""


	def __init__( self, strategy: str = 'mean' ) -> None:
		super( ).__init__( )
		self.simple_imputer = SimpleImputer( strategy = strategy )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

			Fits the simple_imputer
			to the data.

			Args:
				X (np.ndarray): Input data with missing target_values.
				y (Optional[np.ndarray]): Ignored.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.simple_imputer.fit( X )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SimpleImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the text
			data by filling in missing target_values.

			Args:
				X (np.ndarray): Input data with missing target_values.

			Returns:
				np.ndarray: Imputed data.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.simple_imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SimpleImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class NeighborImputer( Metric ):
	"""

		Fills missing target_values
		using k-nearest neighbors.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.knn_imputer = KNNImputer( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
		"""

			Purpose:
			________
			Fits the simple_imputer
			to the data.

			Args:
			_____
			X (np.ndarray): Input data with missing target_values.
			y (Optional[np.ndarray]): Ignored.

		"""
		try:
			self.pipeline = self.knn_imputer.fit( X )
			return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NeighborImputer'
			exception.method = 'fit( self, X: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_________
			
			Transforms the text data by imputing missing target_values.

			Args:
				X (np.ndarray): Input data
				with missing target_values.

			Returns:
				np.ndarray: Imputed data.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.knn_imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NeighborImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class PerceptronClassifier( Model ):
	"""

		PerceptronClassifier classifier.

	"""
	score: Optional[ float ]
	perceptron_classifier: PerceptronClassifier
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, alpha: float=0.0001, max: int=1000, mix: bool=True ) -> None:
		"""

			Initialize the
			PerceptronClassifier linerar_model.

			Attributes:
				linerar_model (Perceptron): Internal linear binary classifier.
					Parameters:
						max_iter (int): Maximum number of iterations.
						Default is 1000.

		"""
		super( ).__init__( )
		self.perceptron_classifier = PerceptronClassifier( alpha=alpha, max_iter=max, shuffle=mix )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Fit the
			PerceptronClassifier linerar_model.

			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Binary class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.perceptron_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Predict binary class
			labels using the PerceptronClassifier.

			Parameters:
				X (pd.DataFrame): Feature matrix.

			Returns:
				np.ndarray: Predicted binary labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.perceptron_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute accuracy of the
			PerceptronClassifier classifier.

			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): True class labels.

			Returns:
				float: Accuracy score.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.perceptron_classifier.predict( X )
				self.accuracy = accuracy_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Evaluate classifier performance
			using standard classification metrics.

			Parameters:
				X (np.ndarray): Input feature_names of shape (n_samples, n_features).
				y (np.ndarray): Ground truth class labels.

			Returns:
				dict: Dictionary of evaluation metrics including:
					- Accuracy (float)
					- Precision (float)
					- Recall (float)
					- F1 Score (float)
					- ROC AUC (float)
					- Matthews Corrcoef (float)
					- Confusion Matrix (List[List[int]])

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.perceptron_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )



class MultilayerRegressor( Model ):
	"""

		Chains multiple preprocessing
		steps into a pipeline.

	"""
	pipeline: Optional[ Pipeline ]
	multilayer_perceptron_regressor: MLPRegressor
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, steps: List[ Tuple[ str, Metric ] ] ) -> None:
		super( ).__init__( )
		self.multilayer_perceptron_regressor = MLPRegressor( )
		self.pipeline = Pipeline( steps )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
		"""

			Fits all pipeline
			steps to the text data.

			Args:
				X (np.ndarray): Input feature matrix.
				y (Optional[np.ndarray]): Optional target array.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.multilayer_perceptron_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Applies all transformations
			in the pipeline to the text data.

			Args:
				X (np.ndarray): Input feature matrix.

			Returns:
				np.ndarray: Transformed feature matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.pipeline.transform( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""

			Fits and transforms all
			pipeline steps on the text data.

			Args:
				X (np.ndarray): Input feature matrix.
				y (Optional[np.ndarray]): Optional target array.

			Returns:
				np.ndarray: Transformed feature matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.multilayer_perceptron_regressor.fit_transform( X, y )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = ('fit_transform( self, X: np.ndarray, y: '
			                    'Optional[ np.ndarray ]=None ) -> np.ndarray')
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
				Compute the R^2 score of the model on the given test data.

			Parameters:
				X (np.ndarray): Test features.
				y (np.ndarray): True values.

			Returns:
				float: R-squared score.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.multilayer_perceptron_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Evaluate the model using
			multiple regression metrics.

			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.

			Returns:
				dict: Dictionary of MAE, MSE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
				Plot actual vs predicted target_values.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'MLP: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LinearRegressor( Model ):
	"""
	
		Ordinary Least Squares Regression.
	
	"""
	linear_regressor: LinearRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self ) -> None:
		"""

			Purpose:
				Initialize the Linear Regression linerar_model.
	
			Attributes:
				linerar_model (LinearRegression): Internal OLS linerar_model using least squares.
					Parameters:
						fit_intercept (bool): Whether to include an intercept term. Default is
						True.
						copy_X (bool): Whether to copy the feature matrix. Default is True.
					
		"""
		super( ).__init__( )
		self.linerar_regressor = LinearRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Purpose:
				Fit the OLS regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.linerar_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict target target_values
			using the OLS linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.linerar_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the R-squared
			score of the OLS model.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): True target target_values.
	
			Returns:
				float: R-squared score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.linerar_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
		
			Evaluate the model using
			multiple regression metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				dict: Dictionary of MAE, MSE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
			
			Purpose:
				Plot actual vs predicted target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Least-Squares Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LeastSquaresRegressor( Model ):
	"""

		Purpose:
		-------
		Ordinary Least Squares Regression fits a linear model with coefficients w = (w1, …, wp)
		to minimize the residual sum of squares between the observed targets in the dataset,
		and the targets predicted by the linear approximation.

	"""
	ols_regressor: LinearRegressor = None
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self ) -> None:
		"""

			Purpose:
				Initialize an instance of OLSModel.

			Returns:
				None
		"""
		super( ).__init__( )
		self.ols_regressor = LinearRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
				Fit the OLS model using training data.

			Parameters:
				X (np.ndarray): Training features of shape (n_samples, n_features).
				y (np.ndarray): Training targets of shape (n_samples,).

			Returns:
				None

		"""
		self.ols_regressor.fit( X, y )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
				Predict using the fitted OLS model.

			Parameters:
				X (np.ndarray): Input feature matrix of shape (n_samples, n_features).

			Returns:
				np.ndarray: Predicted values of shape (n_samples,).

		"""
		return self.ols_regressor.predict( X )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
				Compute the R^2 score of the model on the given test data.

			Parameters:
				X (np.ndarray): Test features.
				y (np.ndarray): True values.

			Returns:
				float: R-squared score.

		"""
		self.prediction = self.ols_regressor.predict( X )
		return r2_score( y, self.prediction )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> dict:
		"""

			Purpose:
				Evaluate the OLS model using multiple regression metrics.

			Parameters:
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): True target values of shape (n_samples,).

			Returns:
				dict: Dictionary containing regression evaluation metrics.

		"""
		self.prediction = self.ols_regressor.predict( X )
		return \
		{
			"MAE": mean_absolute_error( y, self.prediction ),
			"MSE": mean_squared_error( y, self.prediction ),
			"RMSE": mean_squared_error( y, self.prediction, squared=False ),
			"R2": r2_score( y, self.prediction ),
			"Explained Variance": explained_variance_score( y, self.prediction ),
			"Median Absolute Error": median_absolute_error( y, self.prediction )
		}


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
				Plot actual vs predicted values for visual inspection.

			Parameters:
				X (np.ndarray): Input feature matrix.
				y (np.ndarray): True target values.

			Returns:
				None

		"""
		self.prediction = self.ols_regressor.predict( X )
		plt.scatter( y, self.prediction )
		plt.xlabel( 'Observed' )
		plt.ylabel( 'Projected' )
		plt.title( 'OLS Regressor: Observed vs Projected' )
		plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
		plt.grid( True )
		plt.show( )


class RidgeRegressor( Model ):
	"""
		
		Purpose:
		--------
		Solves a regression model where the loss function is the linear least squares function and
		regularization is given by the l2-norm. Also known as Ridge Regression
		or Tikhonov regularization. This estimator has built-in support for
		multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets))
		
	"""
	alpha: float
	solver: str
	max_iter: int
	random_state: int
	ridge_regressor: RidgeRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, alpha: float=1.0, solver: str='auto', max: int=1000, rando: int=42 ) -> None:
		"""
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.alpha = alpha
		self.solver = solver
		self.max_iter = max
		self.random_state = rando
		self.ridge_regressor = RidgeRegressor( alpha=self.alpha, solver=self.solver,
			max_iter=self.max_iter, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the RidgeRegressor
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.ridge_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Project target target_values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.ridge_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the R-squared
			score for the Ridge model.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ridge_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
				Evaluates the Ridge model
				using multiple metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Evaluation metrics including MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Ridge Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LassoRegressor( Model ):
	"""
		
		Purpose:
		--------
		Linear Model trained with L1 for the regularizer. Regularization improves the
		conditioning of the problem and reduces the variance of the estimates. Larger values
		specify stronger regularization.
		
	"""
	alpha: float
	max_iter: int
	random_state: int
	lasso_regressor: LassoRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]


	def __init__( self, alph: float=1.0, max: int=500, rando: int=42,
	              mix='random' ) -> None:
		"""
		
			Initialize the
			LassoRegressor linerar_model.
	
			Attributes:
				linerar_model (Lasso): Internal LassoRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						max_iter (int): Maximum number of iterations. Default is 1000.
					
		"""
		super( ).__init__( )
		self.alpha = alph
		self.max_iter = max
		self.random_state = rando
		self.lasso_regressor = LassoRegressor( alpha=self.alpha, max_iter=self.max_iter,
			random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Purpose:
			--------
			Fit the LassoRegressor
			regression linerar_model.
	
			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
			---------
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.lasso_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict target target_values
			using the LassoRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.lasso_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the Lasso model.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				float: R^2 score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.lasso_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegressor'
			exception.method = 'score(self, X: np.ndarray, y: np.ndarray) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
		
			Evaluate the Lasso model
			using multiple regression metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Dictionary of MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot actual vs.
			predicted target_values.
	
			Parameters:
				X (np.ndarray): Input feature matrix.
				y (np.ndarray): Ground truth target_values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Lasso Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class ElasticNetRegressor( Model ):
	"""
	
		Purpose:
		--------
		A Linear regression with combined L1 and L2 for regularization.
	
	"""
	alpha: float
	ratio: float
	random_state: int
	selection: str
	elasticnet_regressor: ElasticNetRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, alpha: float=1.0, ratio: float=0.5,
	              rando: int=42, select: str='random' ) -> None:
		"""

			Purpose:
				Initialize the
				ElasticNetRegressor linerar_model.
	
			Attributes:
				linerar_model (ElasticNet): Internal ElasticNetRegressor regression linerar_model.
					Parameters:
						hyper (float): Overall regularization strength. Default is 1.0.
						ratio (float): Mixing parameter (0 = RidgeRegressor,
						1 = LassoRegressor). Default is 0.5.
					
		"""
		super( ).__init__( )
		self.alpha = alpha
		self.ratio = ratio
		self.random_state = rando
		self.selection = select
		self.elasticnet_regressor = ElasticNetRegressor( alpha=self.alpha, l1_ratio=self.ratio,
			random_state=self.random_state, selection=self.selection )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
		
			Fit the ElasticNetRegressor
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.elasticnet_model.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict target target_values
			using the ElasticNetRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.elasticnet_model.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			on the test set.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				float: R^2 score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.elasticnet_model.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
		
			Evaluate model performance
			using regression metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				dict: Evaluation metrics.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot actual vs. predicted
			regression output.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True target target_values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'ElasticNet Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LogisticRegressor( Model ):
	"""
	
		Purpose:
		--------
		This class implements regularized logistic regression using the ‘liblinear’ library,
		‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is
		applied by default. It can handle both dense and sparse input. Use C-ordered arrays or
		CSR matrices containing 64-bit floats for optimal performance;
		any other input format will be converted (and copied). The ‘newton-cg’, ‘sag’, and
		‘lbfgs’ solvers support only L2 regularization with primal formulation, or no
		regularization. The ‘liblinear’ solver supports both L1 and L2 regularization,
		with a dual formulation only for the L2 regularization. The Elastic-Net regularization
		is only supported by the ‘saga’ solver.
	
	"""
	solver: str
	penalty: str
	logistic_regressor: LogisticRegressor
	max_iter: int
	alpha: float
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, c: float=1.0, penalty: str='l2', max: int=1000, solver: str='lbfgs' ) -> None:
		"""

			Purpose:
			--------
			Initialize the Logistic
			Regression linerar_model.

			Parameters:
			-----------
				max (int): Maximum number of iterations. Default is 1000.
				solver (str): Algorithm to use in optimization. Default is 'lbfgs'.
					
		"""
		super( ).__init__( )
		self.alpha = c
		self.penalty = penalty
		self.max_iter = max
		self.solver = solver
		self.logistic_regressor = LogisticRegressor( C=self.alpha, max_iter=self.max_iter,
			solver=self.solver, penalty=self.penalty )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the logistic
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target class labels.
	
			Returns:
				None
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.logistic_regressor.fit( X, y )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict class labels using
			the logistic regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.logistic_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
			
			Compute classification
			accuracy.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): True class labels.
	
			Returns:
				float: Accuracy score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.logistic_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluate the classifier
			using multiple classification metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names of shape (n_samples, n_features).
				y (np.ndarray): True labels of shape (n_samples,).
	
			Returns:
				dict: Dictionary containing:
					- Accuracy (float)
					- Precision (float)
					- Recall (float)
					- F1 Score (float)
					- ROC AUC (float)
					- Matthews Corrcoef (float)
					- Confusion Matrix (List[List[int]])
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot confusion matrix
			for classifier predictions.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.logistic_regressor.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Logistic Regression Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BayesianRidgeRegressor( Model ):
	"""
	
		Purpose:
		--------
		Bayesian regression techniques can be used to include regularization parameters in the
		estimation procedure: the regularization parameter is not set in a hard sense
		but tuned to the data at hand. This can be done by introducing uninformative priors over
		the hyper parameters of the model. The regularization used in Ridge regression and
		classification is equivalent to finding a maximum a posteriori estimation under a
		Gaussian prior over the coefficients with precision . Instead of setting lambda manually,
		it is possible to treat it as a random variable to be estimated from the data.
	
	"""
	max_iter: int
	shape_alpha: int
	scale_alpha: int
	shape_lambda: int
	scale_lambda: int
	bayesian_ridge_regressor: BayesianRidgeRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, max: int=300, shape_alpha=1e-06,
	              scale_alpha=1e-06, shape_lambda=1e-06, scale_lambda=1e-06 ) -> None:
		"""

			Purpose:
				Initializes the BayesianRidgeRegressor.
					
		"""
		super( ).__init__( )
		self.max_iter = max
		self.shape_alpha = shape_alpha
		self.scale_alpha = scale_alpha
		self.shape_lambda = shape_lambda
		self.scale_lambda = scale_lambda
		self.bayesian_ridge_regressor = BayesianRidgeRegressor( n_iter=self.max_iter,
			alpha_1=self.shape_alpha, alpha_2=self.scale_alpha,
			lambda_1=self.shape_lambda, lambda_2=self.scale_lambda )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Purpose:
				Fit the Bayesian RidgeRegressor
				regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.bayesian_ridge_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
				Predicts target target_values
				using the Bayesian linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.bayesian_ridge_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
				Compute the R^2 score
				of the model on test data.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): True target_values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bayesian_ridge_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluate the Bayesian model
			with regression metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target target_values.
	
			Returns:
				dict: Dictionary of evaluation metrics.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs.
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True target target_values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bayesian_ridge_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Bayesian-Ridge Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRidgeRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StochasticGradientClassifier( Model ):
	"""
	
		Purpose:
		--------
		Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative
		rate of linear classifiers under convex loss functions such as
		(linear) Support Vector Machines and Logistic Regression.
	
	"""
	loss: str
	regularization: str
	max_iter: str
	score: Optional[ float ]
	stochastic_gradient_classifier: SGDClassifier
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, loss: str='hinge', max: int=5, reg: str='l2' ) -> None:
		"""
		
			Initialize the
			SGDClassifier linerar_model.
	
			Attributes:
				linerar_model (SGDClassifier): Internal linear classifier trained via SGD.
					Parameters:
						reg (str): Loss function to use. Default is 'log_loss'.
						max (int): Maximum number of passes over the data. Default is 1000.
					
		"""
		super( ).__init__( )
		self.loss = loss
		self.max_iter = max
		self.regularization = reg
		self.stochastic_gradient_classifier = SGDClassifier( loss=self.loss,
			max_iter=self.max_iter, penalty=self.regularization )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the SGD
			classifier linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.stochastic_gradient_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
				Predict class labels
				using the SGD classifier.
		
				Parameters:
					X (pd.DataFrame): Feature matrix.
		
				Returns:
					np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stochastic_gradient_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the SGDRegressor.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""
		 
			Evaluate the classifier
			using standard metrics.
	
			Parameters:
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): True class labels of shape (n_samples,).
	
			Returns:
				dict: Dictionary containing:
					- Accuracy (float)
					- Precision (float)
					- Recall (float)
					- F1 Score (float)
					- ROC AUC (float)
					- Matthews Corrcoef (float)
					- Confusion Matrix (List[List[int]])
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_gradient_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StochasticGradientRegressor( Model ):
	"""
	
		Purpose:
		--------
		Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative
		rate of linear classifiers under convex loss functions such as
		(linear) Support Vector Machines and Logistic Regression. Even though SGD has been around
		in the machine rate community for a long time, it has received a considerable amount
		of attention just recently in the context of large-scale rate.

		SGD has been successfully applied to large-scale and sparse machine rate problems
		often encountered in text classification and natural language processing.
		Given that the data is sparse, the classifiers in this module easily scale to problems
		with more than 10^5 training examples and more than 10^5 features.
	
	"""
	loss: str
	alpha: int
	penalty: int
	max_iter: int
	regularization: str
	stochastic_gradient_regressor: SGDRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, loss: str='hinge', max: int=5, reg: str='l2' ) -> None:
		"""
		
		
			Initialize the
			SGDRegressor linerar_model.
	
			Attributes:
				linerar_model (SGDRegressor): Internal linear regressor trained via SGD.
					Parameters:
						alpha (float)" Regulation
						regularization (str): Regularization term. Default is 'l2'.
						max_iter (int): Maximum number of passes. Default is 1000.
					
		"""
		super( ).__init__( )
		self.loss = loss
		self.max_iter = max
		self.penalty = reg
		self.stochastic_gradient_regressor = SGDRegressor( loss=self.loss,
			max_iter=self.max_iter, penalty=self.penalty )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the SGD
			regressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.stochastic_gradient_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict target_values using
			the SGD regressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stochastic_gradient_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientRegressor'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute R^2 score
			for the SGDRegressor.

			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.

			Returns:
				float: R^2 score.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_gradient_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluate regression model
			performance.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True target target_values.
	
			Returns:
				dict: Evaluation metrics dictionary.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientRegressor'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot predicted vs.
			actual target_values.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bayesian_ridge_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Stochastic Gradient Regression: Observed vs Projected' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticGradientRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class NearestNeighborClassifier( Model ):
	"""
	
		Purpose:
		--------
		The principle behind nearest neighbor methods is to find a predefined number of
		training samples closest in distance to the new point, and predict the label from these.
		The number of samples can be a user-defined constant (k-nearest neighbor rate),
		or vary based on the local density of points (radius-based neighbor rate).
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training data
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).
	
	"""
	n_neighbors: int
	score: Optional[ float ]
	nearest_neighbor_classifier: KNeighborsClassifier
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, num: int=5 ) -> None:
		"""
			
			Initialize the KNeighborsClassifier l
			inerar_model.
	
			Attributes:
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.
					
		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.nearest_neighbor_classifier = KNeighborsClassifier( n_neighbors=self.n_neighbors )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Fit the KNN
			classifier linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.nearest_neighbor_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict class labels
			using the KNN classifier.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.nearest_neighbor_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
			
			Compute classification
			accuracy for k-NN.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth labels.
	
			Returns:
				float: Accuracy score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return accuracy_score( y, self.nearest_neighbor_classifier.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
		
			Evaluate classification performance
			using various metrics.
	
			Parameters:
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): True class labels of shape (n_samples,).
	
			Returns:
				dict: Dictionary containing:
					- Accuracy (float)
					- Precision (float)
					- Recall (float)
					- F1 Score (float)
					- ROC AUC (float)
					- Matthews Corrcoef (float)
					- Confusion Matrix (List[List[int]])
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.mean_absolute_error = mean_absolute_error( y, self.prediction )
				self.mean_squared_error = mean_squared_error( y, self.prediction )
				self.r_mean_squared_error = mean_squared_error( y, self.prediction,
					squared = False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared = False )
				return \
				{
					'MAE': self.mean_absolute_error,
					'MSE': self.mean_squared_error,
					'RMSE': self.r_mean_squared_error,
					'R2': self.r2_score,
					'Explained Variance': self.explained_variance_score,
					'Median Absolute Error': self.median_absolute_error,
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborClassifier'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.nearest_neighbor_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class NearestNeighborRegressor( Model ):
	"""
	
		Purpose:
		--------
		The principle behind nearest neighbor methods is to find a predefined number of
		training samples closest in distance to the new point, and predict the label from these.
		The number of samples can be a user-defined constant (k-nearest neighbor rate),
		or vary based on the local density of points (radius-based neighbor rate).
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training data
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).
	
	"""
	n_neighbors: int
	nearest_neighbor_regressor: KNeighborsRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, num: int=5 ) -> None:
		"""
		
			Initialize the
			KNeighborsRegressor linerar_model.
	
			Attributes:
				linerar_model (KNeighborsRegressor): Internal non-parametric regressor.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.
					
		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.nearest_neighbor_regressor = KNeighborsRegressor( n_neighbors=self.n_neighbors )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the KNN
			regressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.nearest_neighbor_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict target_values using
			the KNN regressor.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.nearest_neighbor_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
			
			Compute R^2 score
			for k-NN regressor.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.nearest_neighbor_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""
			
			Evaluate k-NN regression
			performance with multiple metrics.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): True target target_values.
	
			Returns:
				dict: Dictionary of evaluation scores.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.nearest_neighbor_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Nearest-Neighbor Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RandomForestClassifier( Model ):
	"""
	
		Purpose:
		--------
		In random forests, each tree in the ensemble is built from a sample drawn with replacement
		(i.e., a bootstrap sample) from the training set.

		Furthermore, when splitting each node during the construction of a tree,
		the best split is found either from all input features or a random subset of
		size max_features.

		The purpose of these two sources of randomness is to decrease the variance
		of the forest estimator. Indeed, individual decision trees typically exhibit high variance
		and tend to overfit. The injected randomness in forests yield decision trees with
		somewhat decoupled prediction errors. By taking an average of those predictions,
		some errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.
	
	"""
	n_estimators: int
	criterion: str
	max_depth: int
	random_state: int
	random_forest_classifier: RandomForestClassifier
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]


	def __init__( self, est: int=10, crit: str='gini', max: int=3, rando: int=42 ) -> None:
		"""
		
			Initializes the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.n_estimators = est
		self.criterion = crit
		self.max_depth = max
		self.random_state = rando
		self.random_forest_classifier = RandomForestClassifier( n_estimators=self.n_estimators,
			criterion=self.criterion, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
		
			Fit the classifier.
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_classifier.fit( X, y )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

				Purpose:
				-------
				Predict class labels
				using the SGD classifier.
		
				Parameters:
				----------
					X (pd.DataFrame): Feature matrix.
		
				Returns:
				---------
					np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.random_forest_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the SGDRegressor.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_classifier.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""
		
			Evaluate the Lasso model
			using multiple regression metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Dictionary of MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RandomForestRegressor( Model ):
	"""

		Purpose:
		--------
		In random forests, each tree in the ensemble is built from a sample drawn with replacement
		(i.e., a bootstrap sample) from the training set.

		Furthermore, when splitting each node during the construction of a tree,
		the best split is found either from all input features or a random subset of
		size max_features.

		The purpose of these two sources of randomness is to decrease the variance
		of the forest estimator. Indeed, individual decision trees typically exhibit high variance
		and tend to overfit. The injected randomness in forests yield decision trees with
		somewhat decoupled prediction errors. By taking an average of those predictions,
		some errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.

	"""
	n_estimators: int
	criterion: str
	max_depth: int
	random_state: int
	random_forest_regressor: RandomForestRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, est: int=10, crit: str='gini', max: int=3, rando: int=42 ) -> None:
		"""
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.n_estimators = est
		self.criterion = crit
		self.max_depth = max
		self.random_state = rando
		self.random_forest_regressor = RandomForestRegressor( n_estimators=self.n_estimators,
			criterion=self.criterion, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the RidgeRegressor
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
				
		"""
		try:
			if X is None:
				raise ArgumentError( 'The argument "X" is required!' )
			elif y is None:
				raise ArgumentError( 'The argument "y" is required!' )
			else:
				self.pipeline = self.random_forest_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Project target target_values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.random_forest_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the R-squared
			score for the Ridge model.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluates the Ridge model
			using multiple metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Evaluation metrics including MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.random_forest_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Random Forest Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class GradientBoostingClassifier( Model ):
	"""

		Purpose:
		--------
		Gradient Boosting builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

	"""
	loss: str
	learning_rate: int
	n_estimators: int
	max_depth: int
	gradient_boost_classifier: Optional[ GradientBoostingRegressor ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, loss: str='deviance', rate: int=0.1,
	              est: int=100, max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			________
			Initialize the GradientBoostingClassifier.

			Parameters:
			___________
			**kwargs: Parameters for GradientBoostingClassifier.

		"""
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = rate
		self.n_estimators = est
		self.max_depth = max
		self.random_state = rando
		self.gradient_boost_classifier = GradientBoostingClassifier( loss=self.loss,
			learning_rate=self.learning_rate, n_estimators=self.n_estimators,
			max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Purpose:
			________
			Fit the model to the training data.

			Parameters:
			__________
				X (np.ndarray): Feature matrix.
				y (np.ndarray): Class labels.

		"""
		self.pipeline = self.gradient_boost_classifier.fit( X, y )
		return self.pipeline


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			________
			Predict class labels.

			Parameters:
			__________
				X (np.ndarray): Feature matrix.

			Returns:
			________
				np.ndarray: Predicted labels.

		"""
		self.prediction = self.gradient_boost_classifier.predict( X )
		return self.prediction


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			_______
			Compute classification accuracy.

			Parameters:
			__________
				X (np.ndarray): Features.
				y (np.ndarray): Ground truth labels.

			Returns:
			________
				float: Accuracy score.

		"""
		self.prediction = self.gradient_boost_classifier.predict( X )
		self.accuracy = accuracy_score( y, self.prediction )
		return self.accuracy


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Purpose:
			--------
				Evaluate classifier using multiple metrics.

			Parameters:
			-----------
				X (np.ndarray): Feature matrix.
				y (np.ndarray): Ground truth labels.

			Returns:
			--------
				Dict[str, float]: Evaluation scores.

		"""
		self.prediction = self.gradient_boost_classifier.predict( X )
		return \
		{
			"Accuracy": accuracy_score( y, self.prediction ),
			"Precision": precision_score( y, self.prediction, average='binary' ),
			"Recall": recall_score( y, self.prediction, average='binary' ),
			"F1 Score": f1_score( y, self.prediction, average='binary' ),
			"ROC AUC": roc_auc_score( y, self.prediction ),
			"Matthews Corrcoef": matthews_corrcoef( y, self.prediction )
		}


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Display the confusion matrix.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth labels.

		"""
		self.prediction = self.gradient_boost_classifier.predict( X )
		cm = confusion_matrix( y, self.prediction )
		ConfusionMatrixDisplay( confusion_matrix=cm ).plot( cmap='Oranges' )
		plt.title( "Gradient Boosting Classifier Confusion Matrix" )
		plt.grid( False )
		plt.show( )


class GradientBoostingRegressor( Model ):
	"""

		Purpose:
		--------
		Gradient Boosting builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

	"""
	loss: str
	learning_rate: float
	n_estimators: int
	max_depth: int
	random_state: int
	gradient_boost_regressor: Optional[ GradientBoostingRegressor ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, loss: str='deviance', rate: int=0.1,
	              est: int=100, max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			_______
				Initialize the GradientBoostingRegressor.

			Parameters:
			__________
				**kwargs: Keyword arguments passed to the constructor.

		"""
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = learning
		self.n_estimators = est
		self.max_depth = max
		self.gradient_boost_regressor = GradientBoostingRegressor( loss=self.loss,
			learning_rate=self.learning_rate, n_estimators=self.n_estimators,
			max_depth=self.max_depth )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Purpose:
			_______
			Fit the gradient boosting model.

			Parameters:
			___________
				X (np.ndarray): Training features.
				y (np.ndarray): Training targets.

		"""
		self.pipeline = self.gradient_boost_regressor.fit( X, y )
		return self.pipeline


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_________
			Predict regression targets.

			Parameters:
			_________
				X (np.ndarray): Feature matrix.

			Returns:
			________
				np.ndarray: Predicted target values.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		return self.prediction


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			________
			Compute the coefficient of determination R².

			Parameters:
			___________
				X (np.ndarray): Test features.
				y (np.ndarray): True target values.

			Returns:
			_______
				float: R² score.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		return r2_score( y, self.prediction )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Purpose:
			_______
			Evaluate performance using standard regression metrics.

			Parameters:
			__________
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.

			Returns:
			________
				Dict[str, float]: Evaluation metrics.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		return \
		{
			'MAE': mean_absolute_error( y, self.prediction ),
			'MSE': mean_squared_error( y, self.prediction ),
			'RMSE': mean_squared_error( y, self.prediction, squared=False ),
			'R2': r2_score( y, self.prediction ),
			'Explained Variance': explained_variance_score( y, self.prediction ),
			'Median Absolute Error': median_absolute_error( y, self.prediction )
		}


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			________
			Plot predicted vs actual target values.

			Parameters:
			___________
				X (np.ndarray): Feature matrix.
				y (np.ndarray): Ground truth target values.

		"""
		self.prediction = self.gradient_boost_regressor.predict( X )
		plt.scatter( y, self.prediction, alpha=0.6 )
		plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
		plt.xlabel( 'Observed' )
		plt.ylabel( 'Projected' )
		plt.title( 'Gradient-Boosting Regression: Observed vs Projected' )
		plt.grid( True )
		plt.show( )


class AdaBoostClassifier( Model ):
	"""

		Purpose:
		---------
		The core principle of AdaBoost is to fit a sequence of weak learners
		(i.e., models that are only slightly better than random guessing,
		such as small decision trees) on repeatedly modified versions of the data.
		The predictions from all of them are then combined through a weighted
		majority vote (or sum) to produce the final prediction.

	"""
	n_estimators: int
	max_depth: int
	ada_boost_classifier: Optional[ AdaBoostClassifier ]
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]


	def __init__( self, estimators: int=100, max: int=3 ) -> None:
		"""
		
			Initialize the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.max_depth = max
		self.n_estimators = estimators
		self.ada_boost_classifier = AdaBoostClassifier( n_estimators=self.estimators,
			max_depth=self.max_depth )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0
		self.scaler_type = None


	def scale( self ) -> None:
		"""
			
			Purpose:
				Scale numeric feature_names using selected scaler.
	
			Raises:
				ValueError: If scaler scaler is not 'standard' or 'minmax'.
				
		"""
		if self.scaler_type is None:
			return

		scaler =\
		{
			'standard': StandardScaler( ),
			'minmax': MinMaxScaler( )
		}.get( self.scaler_type )

		if scaler is None:
			raise ValueError( 'Scaler must be standard or minmax.' )

		scaled_array = scaler.fit_transform( self.X[ self.numeric_features ] )
		scaled_df = pd.DataFrame( scaled_array, columns = self.numeric_features,
			index = self.X.index )

		# Combine scaled numeric with untouched categorical
		self.X_scaled = pd.concat( [ scaled_df, self.X[ self.categorical_features ] ], axis=1 )


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Purpose:
			_______
				Fit the classifier.

			Parameters:
			_________
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.ada_boost_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
				Predict class labels
				using the SGD classifier.
		
				Parameters:
					X (pd.DataFrame): Feature matrix.
		
				Returns:
					np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.ada_boost_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the SGDRegressor.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_classifier.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""
		
			Evaluate the Lasso model
			using multiple regression metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Dictionary of MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'ADA Boost Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AdaBoostRegressor( Model ):
	"""

		Purpose:
		---------
		The core principle of AdaBoost is to fit a sequence of weak learners
		(i.e., models that are only slightly better than random guessing,
		such as small decision trees) on repeatedly modified versions of the data.
		The predictions from all of them are then combined through a weighted
		majority vote (or sum) to produce the final prediction.
		
	"""
	n_estimators: int
	max_depth: int
	ada_boost_regressor: Optional[ AdaBoostRegressor ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, estimators: int=100, max: int=3 ) -> None:
		"""
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.max_depth = max
		self.n_estimators = estimators
		self.ada_boost_regressor = AdaBoostRegressor( n_estimators=self.n_estimators,
			max_depth=self.max_depth )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the RidgeRegressor
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.ada_boost_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Project target target_values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.ada_boost_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			--------
			Compute the R-squared
			score for the Ridge model.
	
			Parameters:
			----------
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
			--------
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluates the Ridge model
			using multiple metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Evaluation metrics including MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ada_boost_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'ADA Boost Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BaggingClassifier( Model ):
	"""

		Purpose:
		--------
		 Bagging methods form a class of algorithms which build several instances of a black-box
		 estimator on random subsets of the original training set and then aggregate their
		 individual predictions to form a final prediction. These methods are used as a way
		 to reduce the variance of a base estimator (e.g., a decision tree), by introducing
		 randomization into its construction procedure and then making an ensemble out of it.
		 In many cases, bagging methods constitute a very simple way to improve with respect
		 to a single model, without making it necessary to adapt the underlying base algorithm.
		 As they provide a way to reduce overfitting, bagging methods work best with strong and
		 complex models (e.g., fully developed decision trees), in contrast with boosting methods
		 which usually work best with weak models (e.g., shallow decision trees).

	"""
	base_estimator: object
	n_estimators: int
	max_features: int
	random_state: int
	bagging_classifier: BaggingClassifier
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]


	def __init__( self, base: object=None, num: int=10, max: int=1, rando: int=42 ) -> None:
		"""
		
			Initialize the BaggingClassifier.
			
		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = max
		self.random_state = rando
		self.bagging_classifier = BaggingClassifier( base_estimator=self.base_estimator,
			n_estimators=self.n_estimators, max_features=self.max_features,
			random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
		
			Fit the classifier.
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.bagging_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
				Predict class labels
				using the SGD classifier.
		
				Parameters:
					X (pd.DataFrame): Feature matrix.
		
				Returns:
					np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.bagging_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the SGDRegressor.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bagging_classifier.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""
		
			Evaluate the Lasso model
			using multiple regression metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Dictionary of MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bagging_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Bagging Classifier Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BaggingRegressor( Model ):
	"""

		Purpose:
		--------
		 Bagging methods form a class of algorithms which build several instances of a black-box
		 estimator on random subsets of the original training set and then aggregate their
		 individual predictions to form a final prediction. These methods are used as a way
		 to reduce the variance of a base estimator (e.g., a decision tree), by introducing
		 randomization into its construction procedure and then making an ensemble out of it.
		 In many cases, bagging methods constitute a very simple way to improve with respect
		 to a single model, without making it necessary to adapt the underlying base algorithm.
		 As they provide a way to reduce overfitting, bagging methods work best with strong and
		 complex models (e.g., fully developed decision trees), in contrast with boosting methods
		 which usually work best with weak models (e.g., shallow decision trees).
		
	"""
	base_estimator: object
	n_estimators: int
	max_features: int
	random_state: int
	bagging_regressor: BaggingRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, base: object=None, num: int=10, max: int=1, rando: int=42 ) -> None:
		"""
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.base_estimator = base
		self.n_estimators = num
		self.max_features = max
		self.random_state = rando
		self.bagging_regressor = BaggingRegressor( base_estimator=self.base_estimator,
			n_estimator=self.n_estimators, max_features=self.max_features,
			random_state=self.random_state )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the RidgeRegressor
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.bagging_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Project target target_values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.bagging_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the R-squared
			score for the Ridge model.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bagging_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluates the Ridge model
			using multiple metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Evaluation metrics including MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.bagging_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Bagging Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class VotingClassifier( Model ):
	"""
	
		Purpose:
		--------
		The idea behind the VotingClassifier is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class labels. Such a classifier can be useful for a set of equally well
		performing model in order to balance out their individual weaknesses.
	
	"""
	estimators: List[ ( str, object ) ]
	voting_classifier: VotingClassifier
	voting: Optional[ str ]
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]


	def __init__( self, estimators: List[ ( str, object ) ], voting='hard' ) -> None:
		"""
		
			Initialize the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.estimators = estimators
		self.voting = voting
		self.voting_classifier = VotingClassifier( estimators=self.estimators,
			voting = self.voting )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
		
			Fit the classifier.
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.voting_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
				Predict class labels
				using the SGD classifier.
		
				Parameters:
					X (pd.DataFrame): Feature matrix.
		
				Returns:
					np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.voting_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the SGDRegressor.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.voting_classifier.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""
		
			Evaluate the Lasso model
			using multiple regression metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Dictionary of MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.voting_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Voting Classifer Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class VotingRegressor( Model ):
	"""

		Purpose:
		--------
		The idea behind the VotingClassifier is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class labels. Such a classifier can be useful for a set of equally well
		performing model in order to balance out their individual weaknesses.

	"""
	estimators: List[ ( str, object ) ]
	voting: Optional[ str ]
	voting_regressor: VotingRegressor
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, estimators: List[ ( str, object ) ], voting='hard' ) -> None:
		"""
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.estimators = estimators
		self.voting = voting
		self.voting_regressor = VotingRegressor( estimators=self.estimators, voting=self.voting )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the RidgeRegressor
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.voting_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Project target target_values
			using the RidgeRegressor linerar_model.
	
			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
	
			Returns:
			-------
			np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.voting_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the R-squared
			score for the Ridge model.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.voting_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluates the Ridge model
			using multiple metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Evaluation metrics including MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.voting_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Voting Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StackingClassifier( Model ):
	"""
	
		Wrapper for scikit-learn VotingClassifier.
	
	"""
	estimators: Optional[ List[ Tuple[ str, ClassifierMixin ] ] ]
	final_estimator: Optional[ ClassifierMixin ]
	stacking_classifier: StackingClassifier
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]


	def __init__( self, estimators: List[ Tuple[ str, ClassifierMixin ] ],
	              final_estimator: Optional[ ClassifierMixin ]=None ) -> None:
		"""
		
			Initialize the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.estimators = estimators
		self.final_estimator = final_estimator
		self.stacking_classifier = StackingClassifier( estimators=self.estimators,
			final_estimator=self.final_estimator )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""

			Purpose:
			---------
			Fit the classifier.

			Parameters:
			----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.stacking_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
				Predict class labels
				using the SGD classifier.
		
				Parameters:
					X (pd.DataFrame): Feature matrix.
		
				Returns:
					np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stacking_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the SGDRegressor.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stacking_classifier.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""
		
			Evaluate the Lasso model
			using multiple regression metrics.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Dictionary of MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): True class labels.

			Returns:
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stacking_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Stacking Classifer Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StackingRegressor( Model ):
	"""
		
		RidgeRegressor Regression
		(L2 regularization).
		
	"""
	estimators: Optional[ List[ Tuple[ str, ClassifierMixin ] ] ]
	final_estimator: Optional[ ClassifierMixin ]
	stacking_regressor: StackingRegressor
	estimators = List[ ( str, object ) ]
	final_estimator = object
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	median_absolute_error: Optional[ float ]


	def __init__( self, estimators: List[ ( str, object ) ], final: object=None ) -> None:
		"""
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.estimators = estimators
		self.final_estimator = final
		self.stacking_regressor = StackingRegressor( estimators=self.estimators,
			final=self.final_estimator )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score = 0.0
		self.correlation_coefficient = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the RidgeRegressor
			regression linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.stacking_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Project target target_values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target target_values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stacking_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the R-squared
			score for the Ridge model.
	
			Parameters:
				X (np.ndarray): Test feature_names.
				y (np.ndarray): Ground truth target_values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stacking_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluates the Ridge model
			using multiple metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				dict: Evaluation metrics including MAE, RMSE, R², etc.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average='binary' )
				self.recall = mean_squared_error( y, self.prediction, average='binary' )
				self.f1_score = f1_score( y, self.prediction, average='binary' )
				self.roc_auc_score = roc_auc_score( y, self.prediction )
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction )
				return \
				{
					'Accuracy': self.accuracy,
					'Precision': self.precision,
					'Recall': self.recall,
					'F1 Score': self.f1_score,
					'ROC AUC': self.roc_auc_score,
					'Correlation Coeff': self.correlation_coefficient
				}
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs
			actual target_values.
	
			Parameters:
				X (np.ndarray): Input feature_names.
				y (np.ndarray): Ground truth target target_values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stacking_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Stacking Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
