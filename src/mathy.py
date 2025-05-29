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

 Bobo is a values analysis tool for EPA Analysts.
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
from sklearn.preprocessing import (
	StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
	OneHotEncoder, OrdinalEncoder
)

from sklearn.model_selection import train_test_split
from sklearn.base import  ClassifierMixin
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
				np.ndarray: Predicted values or class labels.
			
		"""
		raise NotImplementedError
	
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the core metric
			(e.g., R²) of the model on test data.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target values.
	
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
				y (np.ndarray): Ground truth values.
	
			Returns:
				dict: Dictionary containing multiple evaluation metrics.
			
		"""
		raise NotImplementedError


class Metric( BaseModel):
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
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> None:
		"""

			Fits the preprocessor
			to the input data.

			Args:
				X (pd.DataFrame): Feature matrix.
				y (Optional[np.ndarray]): Optional target array.

		"""
		raise NotImplementedError
	
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the input
			data using the fitted preprocessor.

			Args:
				X (pd.DataFrame): Feature matrix.

			Returns:
				np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError
	
	
	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Fits the preprocessor and
			then transforms the input data.

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
			exception.method = ('fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None'
			                    ') -> np.ndarray')
			error = ErrorDialog( exception )
			error.show( )


class Dataset( ):
	"""

		Purpose:
		Utility class for preparing machine learning
		datasets from a pandas DataFrame.

	"""
	data: np.ndarray
	target: np.ndarray
	dataframe: Optional[ pd.DataFrame ]
	records: Optional[ int ]
	fields: Otional[ int ]
	features: Optional[ List[ str ] ]
	size: float
	random_state: int
	values: np.ndarray
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	
	def __init__( self, data: np.ndarray, target: str=None, size: float=0.2, state: int=42 ):
		"""

			Purpose:
			Initialize and split the dataset.

			Parameters:
				data (np.ndarray): Matrix input vector.
				target (str): Name of the target column.
				size (float): Proportion of data to use as test set.
				random_state (int): Seed for reproducibility.

		"""
		self.dataframe = pd.DataFrame( data=data, columns=data[ 0, : ], index=data[ :, 0 ] )
		self.records = len( self.dataframe )
		self.fields = len( self.dataframe.columns )
		self.target = target
		self.features = [ column for column in self.dataframe.columns ]
		self.values = self.data[ 1:, target ]
		self.size = size
		self.random_state = state
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	
	def __dir__( self ):
		'''

			Purpose:
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'records', 'fields', 'target', 'split_data',
		         'features', 'size', 'random_state', 'data', 'calculate_metrics',
		         'get_training_data', 'get_testing_data',
		         'values', 'X_train', 'X_test', 'y_train', 'y_test' ]
	
	
	def split_data( self ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""

			Purpose:
			Split the dataset into training and test sets.

			Returns:
				Tuple[ pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray ]

		"""
		try:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.data,
				self.values, self.size, self.random_state )
			return (self.X_train, self.X_test, self.y_train, self.y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ('split_data( self ) -> Tuple[ pd.DataFrame, np.ndarray, pd.DataFrame,'
			                    ' np.ndarray ]')
			error = ErrorDialog( exception )
			error.show( )
	
	
	def calculate_metrics( self ) -> Dict:
		"""

			Purpose:
			Split the dataset into training and test sets.

			Returns:
				Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]

		"""
		try:
			df = pd.DataFrame( data=self.data, columns=self.features )
			return df.describe( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = 'caluclate_metrics( ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def get_training_data( self ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
				Return the training features and labels.
	
			Returns:
				Tuple[pd.DataFrame, np.ndarray]: X_train, y_train
		"""
		return self.X_train, self.y_train
	
	
	def get_testing_data( self ) -> Tuple[ np.ndarray, np.ndarray ]:
		"""
		
			Purpose:
			Return the test features and labels.
	
			Returns:
				Tuple[ np.ndarray, np.ndarray ]: X_test, y_test
				
		"""
		return self.X_test, self.y_test


class StandardScaler( Metric ):
	"""

		Standardizes features by
		removing the mean and scaling to unit variance.

	"""
	
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = StandardScaler( )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
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

		Scales features to
		a given range (default is [0, 1]).

	"""
	
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.minmax_scaler = MinMaxScaler( )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
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

		Scales features using statistics
		that are robust to outliers.

	"""
	
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.robust_scaler = RobustScaler( )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class Normalizer( Metric ):
	"""

		Scales input vectors individually to unit norm.

	"""
	
	
	def __init__( self, norm: str = 'l2' ) -> None:
		super( ).__init__( )
		self.normal_scaler = Normalizer( norm=norm )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
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
			exception.cause = ''
			exception.method = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class OneHotEncoder( Metric ):
	"""

		Encodes categorical features
		 as a one-hot numeric array.

	"""
	
	
	def __init__( self, handle_unknown: str='ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = OneHotEncoder( sparse=False, handle_unknown=handle_unknown )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
		"""

			Fits the hot_encoder
			to the categorical data.

			Args:
				X (np.ndarray): Categorical input data.
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
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the input
			data into a one-hot encoded format.

			Args:
				X (np.ndarray): Categorical input data.

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
		features as ordinal integers.

	"""
	
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = OrdinalEncoder( )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
		"""

			Fits the ordial_encoder
			to the categorical data.

			Args:
				X (np.ndarray): Categorical input data.
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

			Transforms the input
			data into ordinal-encoded format.

			Args:
				X (np.ndarray): Categorical input data.

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

		Fills missing values
		using a specified strategy.

	"""
	
	
	def __init__( self, strategy: str='mean' ) -> None:
		super( ).__init__( )
		self.simple_imputer = SimpleImputer( strategy=strategy )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
		"""

			Fits the simple_imputer
			to the data.

			Args:
				X (np.ndarray): Input data with missing values.
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

			Transforms the input
			data by filling in missing values.

			Args:
				X (np.ndarray): Input data with missing values.

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


class NearestNeighborImputer( Metric ):
	"""

		Fills missing values
		using k-nearest neighbors.

	"""
	
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.knn_imputer = KNNImputer( )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
		"""

			Fits the simple_imputer
			to the data.

			Args:
				X (np.ndarray): Input data with missing values.
				y (Optional[np.ndarray]): Ignored.

		"""
		try:
			self.pipeline = self.knn_imputer.fit( X )
			return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_________
			
			Transforms the input data by imputing missing values.

			Args:
				X (np.ndarray): Input data
				with missing values.

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
			exception.cause = 'NearestNeighborImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class MultiLayerPerceptron( Model ):
	"""

		Chains multiple preprocessing
		steps into a pipeline.

	"""
	pipeline: Optional[ Pipeline ]
	
	
	def __init__( self, steps: List[ Tuple[ str, Metric ] ] ) -> None:
		super( ).__init__( )
		self.pipeline = Pipeline( steps )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
		"""

			Fits all pipeline
			steps to the input data.

			Args:
				X (np.ndarray): Input feature matrix.
				y (Optional[np.ndarray]): Optional target array.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.pipeline.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Applies all transformations
			in the pipeline to the input data.

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
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""

			Fits and transforms all
			pipeline steps on the input data.

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
				self.prediction = self.pipeline.fit_transform( X, y )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = ('fit_transform( self, X: np.ndarray, y: '
			                    'Optional[ np.ndarray ]=None ) -> np.ndarray')
			error = ErrorDialog( exception )
			error.show( )


class LinearRegressor( Model ):
	"""
	
		Ordinary Least Squares Regression.
	
	"""
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
		
			Initialize the Linear
			Regression linerar_model.
	
			Attributes:
				linerar_model (LinearRegression): Internal OLS linerar_model using least squares.
					Parameters:
						fit_intercept (bool): Whether to include an intercept term. Default is
						True.
						copy_X (bool): Whether to copy the feature matrix. Default is True.
					
		"""
		super( ).__init__( )
		self.linerar_model = LinearRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
		
			Fit the OLS
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
			else:
				self.pipeline = self.linerar_model.fit( X, y )
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
			
			Predict target values
			using the OLS linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.linerar_model.predict( X )
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
				X (np.ndarray): Test features.
				y (np.ndarray): True target values.
	
			Returns:
				float: R-squared score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.linerar_model.predict( X )
				return r2_score( y, self.prediction )
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
				y (np.ndarray): Ground truth values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
				Plot actual vs predicted values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): True target values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'OLS: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RidgeRegressor( Model ):
	"""
		
		RidgeRegressor Regression
		(L2 regularization).
		
	"""
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
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.ridge_model = RidgeRegressor( alpha=1.0 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
				self.pipeline = self.ridge_model.fit( X, y )
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
			
			Project target values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.ridge_model.predict( X )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.ridge_model.predict( X )
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
			
			Evaluates the Ridge model
			using multiple metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			actual values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'Ridge: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
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
		
		Wrapper for LassoRegressor Regression (L1 regularization).
		
	"""
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	
	
	def __init__( self ) -> None:
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
		self.lasso_model = LassoRegressor( alpha=1.0 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
		
			Fit the LassoRegressor
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
				self.pipeline = self.lasso_model.fit( X, y )
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
		
			Predict target values
			using the LassoRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.lasso_model.predict( X )
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth values.
	
			Returns:
				float: R^2 score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return r2_score( y, self.predict( X ) )
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			predicted values.
	
			Parameters:
				X (np.ndarray): Input feature matrix.
				y (np.ndarray): Ground truth values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'Lasso: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
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
	
		Wrapper for ElasticNetRegressor Regression (L1 + L2 regularization).
	
	"""
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
			
			Initialize the
			ElasticNetRegressor linerar_model.
	
			Attributes:
				linerar_model (ElasticNet): Internal ElasticNetRegressor regression linerar_model.
					Parameters:
						alpha (float): Overall regularization strength. Default is 1.0.
						l1_ratio (float): Mixing parameter (0 = RidgeRegressor,
						1 = LassoRegressor). Default is 0.5.
					
		"""
		super( ).__init__( )
		self.elasticnet_model = ElasticNetRegressor( alpha=1.0, l1_ratio=0.5 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
		
			Predict target values
			using the ElasticNetRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
				
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target values.
	
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
				X (np.ndarray): Input features.
				y (np.ndarray): True target values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'ElasticNet: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
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
	
		Wrapper for a Logistic Regression.
	
	"""
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
		
			Initialize the Logistic
			Regression linerar_model.
	
			Attributes:
				linerar_model (LogisticRegression): Internal logistic regression classifier.
					Parameters:
						max_iter (int): Maximum number of iterations. Default is 1000.
						solver (str): Algorithm to use in optimization. Default is 'lbfgs'.
					
		"""
		super( ).__init__( )
		self.logistic_model = LogisticRegressor( max_iter=1000 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
				self.prediction = self.logistic_model.fit( X, y )
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
				self.prediction = self.logistic_model.predict( X )
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
				X (np.ndarray): Test features.
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
				return accuracy_score( y, self.logistic_model.predict( X ) )
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
				X (np.ndarray): Input features of shape (n_samples, n_features).
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
	
	
	def create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot confusion matrix
			for classifier predictions.
	
			Parameters:
				X (np.ndarray): Input features.
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
				self.prediction = self.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).plot( )
				plt.title( 'Logistic Regression Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LogisticRegressor'
			exception.method = 'create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BayesianRegressor( Model ):
	"""
	
		Wrapper for Bayesian RidgeRegressor Regression.
	
	"""
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
		
			Initialize the
			BayesianRegressor linerar_model.
	
			Attributes:
				linerar_model (BayesianRidge): Internal probabilistic linear regression
				linerar_model.
					Parameters:
						compute_score (bool): If True, compute marginal
						log-likelihood. Default is False.
					
		"""
		super( ).__init__( )
		self.bayesian_model = BayesianRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
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
				self.pipeline = self.bayesian_model.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predicts target values
			using the Bayesian linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.bayesian_model.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRegressor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute the R^2 score
			of the model on test data.
	
			Parameters:
				X (np.ndarray): Test features.
				y (np.ndarray): True values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return r2_score( y, self.bayesian_model.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluate the Bayesian model
			with regression metrics.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.cause = 'BayesianRegressor'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Plot predicted vs.
			actual values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): True target values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'Bayesian Ridge: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BayesianRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class SgdClassifier( Model ):
	"""
	
		SGD-based linear classifiers.
	
	"""
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	
	
	def __init__( self ) -> None:
		"""
		
			Initialize the
			SGDClassifier linerar_model.
	
			Attributes:
				linerar_model (SGDClassifier): Internal linear classifier trained via SGD.
					Parameters:
						loss (str): Loss function to use. Default is 'log_loss'.
						max_iter (int): Maximum number of passes over the data. Default is 1000.
					
		"""
		super( ).__init__( )
		self.sgd_classification_model = SGDClassifier( loss='log_loss', max_iter=1000 )
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
				self.pipeline = self.sgd_classification_model.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SgdClassifier'
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
				self.prediction = self.sgd_classification_model.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SgdClassifier'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute R^2 score
			for the SGDRegressor.
	
			Parameters:
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target values.
	
			Returns:
				float: R^2 score.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return r2_score( y, self.sgd_classification_model.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SgdClassifier'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> dict:
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
				self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return {
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
			exception.cause = 'SgdClassifier'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class SgdRegressor( Model ):
	"""
	
		Wrapper for SGD-based linear regressors.
	
	"""
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
		
		
			Initialize the
			SGDRegressor linerar_model.
	
			Attributes:
				linerar_model (SGDRegressor): Internal linear regressor trained via SGD.
					Parameters:
						penalty (str): Regularization term. Default is 'l2'.
						max_iter (int): Maximum number of passes. Default is 1000.
					
		"""
		super( ).__init__( )
		self.sgd_regression_model = SGDRegressor( max_iter=1000 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the SGD
			regressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.sgd_regression_model.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SgdRegressor'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict values using
			the SGD regressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.sgd_regression_model.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SgdRegressor'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
			
			Evaluate regression model
			performance.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): True target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.cause = 'SgdRegressor'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class Perceptron( Model ):
	"""
	
		Perceptron classifier.
	
	"""
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	
	
	def __init__( self ) -> None:
		"""
		
			Initialize the
			Perceptron linerar_model.
	
			Attributes:
				linerar_model (Perceptron): Internal linear binary classifier.
					Parameters:
						max_iter (int): Maximum number of iterations.
						Default is 1000.
					
		"""
		super( ).__init__( )
		self.perceptron_model = Perceptron( max_iter=1000 )
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
			Perceptron linerar_model.
	
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
				self.pipeline = self.perceptron_model.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Perceptron'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def project( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict binary class
			labels using the Perceptron.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted binary labels.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.perceptron_model.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Perceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""
		
			Compute accuracy of the
			Perceptron classifier.
	
			Parameters:
				X (np.ndarray): Test features.
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
				return accuracy_score( y, self.perceptron_model.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Perceptron'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""
		
			Evaluate classifier performance
			using standard classification metrics.
	
			Parameters:
				X (np.ndarray): Input features of shape (n_samples, n_features).
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
				self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return {
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
			exception.cause = 'Perceptron'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


class NearestNeighborClassifier( Model ):
	"""
	
		Wrapper for k-Nearest Neighbors Classifier.
	
	"""
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	r_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	
	
	def __init__( self ) -> None:
		"""
			
			Initialize the KNeighborsClassifier l
			inerar_model.
	
			Attributes:
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.
					
		"""
		super( ).__init__( )
		self.knn_classification_model = KNeighborsClassifier( n_neighbors=5 )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	
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
				self.pipeline = self.knn_classification_model.fit( X, y )
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
				self.prediction = self.knn_classification_model.predict( X )
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
				X (np.ndarray): Test features.
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
				return accuracy_score( y, self.knn_classification_model.predict( X ) )
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
				self.r_mean_squared_error = mean_squared_error( y, self.prediction, squared=False )
				self.r2_score = r2_score( y, self.prediction )
				self.explained_variance_score = explained_variance_score( y, self.prediction )
				self.median_absolute_error = median_absolute_error( y, self.prediction,
					squared=False )
				return {
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


class NearestNeighborRegressor( Model ):
	"""
	
		Wrapper for k-Nearest Neighbors Regressor.
	
	"""
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
		
			Initialize the
			KNeighborsRegressor linerar_model.
	
			Attributes:
				linerar_model (KNeighborsRegressor): Internal non-parametric regressor.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.
					
		"""
		super( ).__init__( )
		self.knn_regression_model = KNeighborsRegressor( n_neighbors=5 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline:
		"""
			
			Fit the KNN
			regressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target values.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.knn_regression_model.fit( X, y )
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
		
			Predict values using
			the KNN regressor.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.knn_regression_model.predict( X )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth values.
	
			Returns:
				float: R-squared score.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return r2_score( y, self.knn_regression_model, predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> dict:
		"""
			
			Evaluate k-NN regression
			performance with multiple metrics.
	
			Parameters:
				X (np.ndarray): Test features.
				y (np.ndarray): True target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> dict'
			error = ErrorDialog( exception )
			error.show( )


class RandomForestClassifier( Model ):
	"""
	
		Wrapper for scikit-learn RandomForestClassifier.
	
	"""
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	
	
	def __init__( self ) -> None:
		"""
		
			Initialize the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.random_forest_classifier = RandomForestClassifier( )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target values.
	
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input features.
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
				self.prediction = self.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).plot( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassifier'
			exception.method = 'create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
			
			
class RandomForestRegressor( Model ):
	"""
		
		RidgeRegressor Regression
		(L2 regularization).
		
	"""
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
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.random_forest_regressor = RandomForestRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
			
			Project target values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth values.
	
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
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			actual values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'Random Forest: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
			
			
class AdaBoostClassifier( Model ):
	"""
	
		Wrapper for scikit-learn RandomForestClassifier.
	
	"""
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	
	
	def __init__( self ) -> None:
		"""
		
			Initialize the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.ada_boost_classifier = AdaBoostClassifier( )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target values.
	
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
				return r2_score( y, self.prediction )
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input features.
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
				self.prediction = self.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).plot( )
				plt.title( 'ADA Boost Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AdaBoostClassifier'
			exception.method = 'create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
			
			
			
class AdaBoostRegressor( Model ):
	"""
		
		RidgeRegressor Regression
		(L2 regularization).
		
	"""
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
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.ada_boost_regressor = AdaBoostRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
			
			Project target values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
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
		
			Compute the R-squared
			score for the Ridge model.
	
			Parameters:
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth values.
	
			Returns:
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
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			actual values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'ADA Boost: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
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
	
		Wrapper for scikit-learn BaggingClassifier.
	
	"""
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	
	
	def __init__( self ) -> None:
		"""
		
			Initialize the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.bagging_classifier = BaggingClassifier( )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target values.
	
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
				return r2_score( y, self.prediction )
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input features.
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
				self.prediction = self.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).plot( )
				plt.title( 'Bagging Classifier Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassifier'
			exception.method = 'create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
			
			
class BaggingRegressor( Model ):
	"""
		
		RidgeRegressor Regression
		(L2 regularization).
		
	"""
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
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.bagging_regressor = BaggingRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
			
			Project target values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth values.
	
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
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			actual values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'Bagging Regression: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
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
	
		Wrapper for scikit-learn VotingClassifier.
	
	"""
	score: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	precision: Optional[ float ]
	recall: Optional[ float ]
	roc_auc_score: Optional[ float ]
	f1_score: Optional[ float ]
	correlation_coefficient: Optional[ float ]
	
	
	def __init__( self, estimators=estimators, voting=voting ) -> None:
		"""
		
			Initialize the RandomForestClassifier.
			
		"""
		super( ).__init__( )
		self.estimators = estimators
		self.voting = voting
		self.voting_classifier = VotingClassifier( estimators=self.estimators, voting=self.voting )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target values.
	
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input features.
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
				self.prediction = self.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).plot( )
				plt.title( 'Voting Classifer Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'VotingClassifier'
			exception.method = 'create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
			
			
class VotingRegressor( Model ):
	"""
		
		RidgeRegressor Regression
		(L2 regularization).
		
	"""
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
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.voting_regressor = VotingRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
			
			Project target values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth values.
	
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
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			actual values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'Voting Regression: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
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
		self.stacking_classifier = StackingClassifier( )
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target values.
	
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
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
				X (np.ndarray): Input features.
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
				self.prediction = self.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).plot( )
				plt.title( 'Stacking Classifer Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingClassifier'
			exception.method = 'create_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StackinRegressor( Model ):
	"""
		
		RidgeRegressor Regression
		(L2 regularization).
		
	"""
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
		
			Initialize the
			RidgeRegressor linerar_model.
	
			Attributes:
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.
					
		"""
		super( ).__init__( )
		self.stacking_regressor = StackinRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
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
			
			Project target values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (pd.DataFrame): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
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
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth values.
	
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
				y (np.ndarray): Ground truth target values.
	
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
				self.correlation_coefficient = matthews_corrcoef( y, self.prediction  )
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
			actual values.
	
			Parameters:
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target values.
	
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
				plt.xlabel( 'Actual' )
				plt.ylabel( 'Predicted' )
				plt.title( 'Stacking Regression: Actual vs Predicted' )
				plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StackingRegressor'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

