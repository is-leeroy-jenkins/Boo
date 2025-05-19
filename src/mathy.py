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

import pandas as pd
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple


class Model( BaseModel ):
	"""
	
		Abstract base class
		that defines the interface for all linerar_model wrappers.
	
	"""
	
	
	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
			
			Fit the linerar_model to
			the training data.
	
			Parameters:
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): Target vector of shape (n_samples,).
	
			Returns:
				None
			
		"""
		raise NotImplementedError
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
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
				X (np.ndarray): Feature matrix.
				y (np.ndarray): True target values.
	
			Returns:
				float: Score value (e.g., R² for regressors).
			
		"""
		raise NotImplementedError
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
		"""
			
			Evaluate the model using
			 multiple performance metrics.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				X (np.ndarray): Feature matrix.
				y (Optional[np.ndarray]): Optional target array.

		"""
		raise NotImplementedError
	
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Transforms the input
			data using the fitted preprocessor.

			Args:
				X (np.ndarray): Feature matrix.

			Returns:
				np.ndarray: Transformed feature matrix.

		"""
		raise NotImplementedError
	
	
	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""

			Fits the preprocessor and
			then transforms the input data.

			Args:
				X (np.ndarray): Feature matrix.
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
			exception.method = ('fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None '
			                    ') -> np.ndarray')
			error = ErrorDialog( exception )
			error.show( )


class Dataset( ):
	"""

		Purpose:
		Utility class for preparing machine learning
		datasets from a pandas DataFrame.

	"""
	data: pd.DataFrame
	records: Optional[ int ]
	fields: Otional[ int ]
	target: str
	features: Otional[ List[ str ] ]
	size: float
	random_state: int
	data: pd.DataFrame
	values: pd.Series
	X_train: Optional[ pd.DataFrame ]
	X_test: Optional[ pd.DataFrame ]
	y_train: Optional[ pd.Series ]
	y_test: Optional[ pd.Series ]
	
	
	def __init__( self, df: pd.DataFrame, target: str, size: float=0.2, state: int=42 ):
		"""

			Purpose:
			Initialize and split the dataset.

			Parameters:
				df (pd.DataFrame): Input dataset as a pandas DataFrame.
				target (str): Name of the target column.
				size (float): Proportion of data to use as test set.
				random_state (int): Seed for reproducibility.

		"""
		self.dataframe = df
		self.records = len( df )
		self.fields = len( df.columns )
		self.target = target
		self.features = [ name for name in df.columns ]
		self.size = size
		self.random_state = state
		self.data = self.dataframe[ 1:, : ]
		self.values = self.dataframe[ 1:, target ]
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	
	def split_data( self ) -> Tuple[ pd.DataFrame, pd.Series, pd.DataFrame, pd.Series ]:
		"""

			Purpose:
			Split the dataset into training and test sets.

			Returns:
				Tuple[ pd.DataFrame, pd.Series, pd.DataFrame, pd.Series ]

		"""
		try:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.data,
				self.values, self.size, self.random_state )
			return (self.X_train, self.X_test, self.y_train, self.y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = ('split_data( self ) -> Tuple[ DataFrame, Series, DataFrame, '
			                    'Series ]')
			error = ErrorDialog( exception )
			error.show( )
	
	
	def calculate_metrics( self ) -> Dict:
		"""

			Purpose:
			Split the dataset into training and test sets.

			Returns:
				Tuple[ pd.DataFrame, pd.Series, pd.DataFrame, pd.Series ]

		"""
		try:
			return df.describe( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Data'
			exception.method = 'caluclate_metrics( ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def __dir__( self ):
		'''

			Purpose:
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'records', 'fields', 'target',
		         'features', 'size', 'random_state', 'data',
		         'values', 'X_train', 'X_test', 'y_train', 'y_test' ]


class StandardScaler( Metric ):
	"""

		Standardizes features by
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

		Scales features to
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

		Scales features using statistics
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
	
	
	def __init__( self, handle_unknown: str = 'ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = OneHotEncoder( sparse=False, handle_unknown=handle_unknown )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
			exception.cause = ''
			exception.method = ''
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
			exception.cause = ''
			exception.method = ''
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
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
			exception.cause = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class SimpleImputer( Metric ):
	"""

		Fills missing values
		using a specified strategy.

	"""
	
	
	def __init__( self, strategy: str = 'mean' ) -> None:
		super( ).__init__( )
		self.simple_imputer = SimpleImputer( strategy=strategy )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
			exception.cause = ''
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
			exception.cause = ''
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
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
			exception.cause = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


class MultiLayerPerceptron( Metric ):
	"""

		Chains multiple preprocessing
		steps into a pipeline.

	"""
	pipeline: Optional[ Pipeline ]
	
	
	def __init__( self, steps: List[ Tuple[ str, Metric ] ] ) -> None:
		self.pipeline = Pipeline( steps )
	
	
	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
				_pipeline: Pipeline = self.pipeline.fit( X, y )
				return _pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
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
				return self.pipeline.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
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
				return self.pipeline.fit_transform( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
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
		self.linerar_model = LinearRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Fit the OLS
			regression linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
				y (np.ndarray): Target vector.
	
			Returns:
				None
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.linerar_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		
		Predict target values
		using the OLS linerar_model.

		Parameters:
			X (np.ndarray): Feature matrix.

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
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
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
				return r2_score( y, self.linerar_model.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LinearRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
		"""
		
			Evaluate the model using
			multiple regression metrics.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
			exception.method = 'evaluate(self, X: np.ndarray, y: np.ndarray ) -> dict'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def plot( self, X: np.ndarray, y: np.ndarray ) -> None:
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
			exception.method = 'plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
		self.ridge_model = RidgeRegressor( alpha=1.0 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
			
			Fit the RidgeRegressor
			regression linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				self.ridge_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict target values
			using the RidgeRegressor linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
	
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
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
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
				return r2_score( y, self.ridge_model.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RidgeRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
		"""
			
			Evaluate the Ridge model
			using multiple metrics.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
			exception.method = 'evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def plot( self, X: np.ndarray, y: np.ndarray ) -> None:
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
				y_pred = self.predict( X )
				plt.scatter( y, y_pred )
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
			exception.method = 'plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
		self.lasso_model = LassoRegressor( alpha=1.0 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
		Fit the LassoRegressor
		regression linerar_model.

		Parameters:
			X (np.ndarray): Feature matrix.
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
				self.lasso_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegressor'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict target values
			using the LassoRegressor linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.lasso_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'LassoRegressor'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
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
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
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
			exception.method = 'evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def plot( self, X: np.ndarray, y: np.ndarray ) -> None:
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
				y_pred = self.predict( X )
				plt.scatter( y, y_pred )
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
			exception.method = 'plot( self, X: np.ndarray, y: np.ndarray) -> None'
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
		self.elasticnet_model = ElasticNetRegressor( alpha=1.0, l1_ratio=0.5 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Fit the ElasticNetRegressor
			regression linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				self.elasticnet_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'fit( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict target values
			using the ElasticNetRegressor linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
	
			Returns:
				np.ndarray: Predicted target values.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.elasticnet_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
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
				return r2_score( y, self.elasticnet_model.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'ElasticNetRegressor'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> Dict:
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
			exception.method = 'evaluate( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def plot( self, X: np.ndarray, y: np.ndarray ) -> None:
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
				y_pred = self.predict( X )
				plt.scatter( y, y_pred )
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
			exception.method = 'plot( self, X: np.ndarray, y: np.ndarray ) -> None'
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
		self.logistic_model = LogisticRegressor( max_iter=1000 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
		Fit the logistic
		regression linerar_model.

		Parameters:
			X (np.ndarray): Feature matrix.
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
				self.logistic_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		
		Predict class labels using
		the logistic regression linerar_model.

		Parameters:
			X (np.ndarray): Feature matrix.

		Returns:
			np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.logistic_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def plot_confusion_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
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
			exception.cause = ''
			exception.method = ''
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
		self.bayesian_model = BayesianRegressor( )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
		Fit the Bayesian RidgeRegressor
		regression linerar_model.

		Parameters:
			X (np.ndarray): Feature matrix.
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
				self.bayesian_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict target values
			using the Bayesian linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
	
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
			exception.cause = ''
			exception.method = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
		"""
			
			Evaluate the Bayesian model
			with regression metrics.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def plot( self, X: np.ndarray, y: np.ndarray ) -> None:
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
			exception.cause = ''
			exception.method = ''
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
		self.sgd_classification_model = SGDClassifier( loss='log_loss', max_iter=1000 )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
			
			Fit the SGD
			classifier linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				self.sgd_classification_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
			
				Predict class labels
				using the SGD classifier.
		
				Parameters:
					X (np.ndarray): Feature matrix.
		
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
			exception.cause = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
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
			exception.cause = ''
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
		self.sgd_regression_model = SGDRegressor( max_iter=1000 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
			
			Fit the SGD
			regressor linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				self.sgd_regression_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		
		Predict values using
		the SGD regressor linerar_model.

		Parameters:
			X (np.ndarray): Feature matrix.

		Returns:
			np.ndarray: Predicted values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.sgd_regression_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> Dict:
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
			exception.cause = ''
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
		self.perceptron_model = Perceptron( max_iter=1000 )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
			
			Fit the
			Perceptron linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				self.perceptron_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict binary class
			labels using the Perceptron.
	
			Parameters:
				X (np.ndarray): Feature matrix.
	
			Returns:
				np.ndarray: Predicted binary labels.
				
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.perceptron_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
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
			exception.cause = ''
			exception.method = ''
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
		self.knn_classification_model = KNeighborsClassifier( n_neighbors=5 )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.r_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
		
			Fit the KNN
			classifier linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				self.knn_classification_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
			
			Predict class labels
			using the KNN classifier.
	
			Parameters:
				X (np.ndarray): Feature matrix.
	
			Returns:
				np.ndarray: Predicted class labels.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return self.knn_classification_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
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
			exception.cause = ''
			exception.method = ''
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
		self.knn_regression_model = KNeighborsRegressor( n_neighbors=5 )
		self.prediction = None
		self.accuracy = 0.0
		self.precision = 0.0
		self.recall = 0.0
		self.f1_score = 0.0
		self.roc_auc_score= 0.0
		self.correlation_coefficient = 0.0
	
	
	def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""
			
			Fit the KNN
			regressor linerar_model.
	
			Parameters:
				X (np.ndarray): Feature matrix.
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
				self.knn_regression_model.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""
		
			Predict values using
			the KNN regressor.
	
			Parameters:
				X (np.ndarray): Feature matrix.
	
			Returns:
				np.ndarray: Predicted values.
			
		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.knn_regression_model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = ''
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
	
	
	def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
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
			exception.cause = ''
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )
