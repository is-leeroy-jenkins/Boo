'''
******************************************************************************************
  Assembly:                Boo
  Filename:                Mathy.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="mathy.py" company="Terry D. Eppler">

     Boo is a df analysis tool integrating various Generative AI, Text-Processing, and
     Machine-Learning algorithms for federal analysts.
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
	Mathy.py
</summary>
******************************************************************************************
'''
from argparse import ArgumentError

import numpy as np
from typing import Optional, List, Dict

from pandas.core.common import random_state
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (
	LinearRegression, LogisticRegression, RidgeClassifier, Ridge, Lasso, ElasticNet,
	BayesianRidge, SGDClassifier, SGDRegressor, Perceptron,
)
from sklearn.ensemble import (
	RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
	BaggingClassifier, VotingClassifier, StackingClassifier,
	RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
	BaggingRegressor, VotingRegressor, StackingRegressor )
from sklearn.metrics import (
	r2_score, mean_squared_error, mean_absolute_error,
	explained_variance_score, median_absolute_error
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import ( StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
	OneHotEncoder, OrdinalEncoder )

from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.metrics import silhouette_score
from sklearn.cluster import ( KMeans, MiniBatchKMeans, DBSCAN,  MeanShift,
                              SpectralClustering, AgglomerativeClustering,
                              Birch, OPTICS )
from static import Scaler
import pandas as pd
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple


class Model( BaseModel ):
	"""
		
		Purpose:
		---------
		Abstract base class that defines the interface for all linerar_model wrappers.
	
	"""
	pipeline: Optional[ Pipeline ]


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
	pipeline: Optional[ Pipeline ]
	transformed_values: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]


	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
		allow_mutation = True


	def __init__( self ):
		self.pipeline = None
		self.transformed_values = None


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
		self.training_data = train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 0 ]
		self.testing_data = train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 1 ]
		self.training_values = train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 2 ]
		self.testing_values = train_test_split( df[ 1:, : ], target, test_size=size, random_state=rando )[ 3 ]
		self.transtuple = [ ]


	def __dir__( self ):
		'''

			Purpose:
			-----------
			This function retuns a list of strings (members of the class)

		'''
		return [ 'dataframe', 'rows', 'columns', 'target', 'split_data',
		         'features', 'test_size', 'rando', 'df', 'scale_data',
		         'numeric_columns', 'text_columns', 'scaler', 'transtuple'
		         'create_testing_data', 'calculate_statistics', 'create_training_data',
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
				raise Exception( 'Name cannot be None' )
			elif encoder is None:
				raise Exception( 'Encoder cannot be None' )
			elif columns is None:
				raise Exception( 'Columns cannot be None' )
			else:
				_tuple = ( name, encoder, columns )
				self.transtuple.append( _tuple )
				self.column_transformer = ColumnTransformer( self.transtuple )
				self.column_transformer.fit_transform( self.data  )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'Dataset'
			exception.method = 'scale_data( )'
			error = ErrorDialog( exception )
			error.show( )


	def calculate_statistics( self ) -> Dict:
		"""

			Purpose:
			-----------
			Split the dataset into training and test sets.

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


class StandardScaler( Metric ):
	"""

		Purpose:
		--------
		Standardizes features by removing the mean and scaling to unit variance.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = StandardScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
			X (np.ndarray): Input df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.standard_scaler.fit( X )
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

			Purpose:
			---------
			Transforms the df using the fitted StandardScaler.

			Parameters:
			-----------
				X (np.ndarray): Input df.

			Returns:
			-----------
				np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.standard_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class MinMaxScaler( Metric ):
	"""

		Purpose:
		---------
		Scales features to a given range (default is [0, 1]).

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.minmax_scaler = MinMaxScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""

			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
				X (np.ndarray): Input df.
				y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.minmax_scaler.fit( X )
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

			Purpose:
			---------
			Transforms the df using the fitted MinMaxScaler.

			Parameters:
			-----------
				X (np.ndarray): Input df.

			Returns:
			-----------
				np.ndarray: Scaled df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.transformed_data = self.minmax_scaler.transform( X )
				return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class RobustScaler( Metric ):
	"""

		Purpose:
		--------
		Scales features using statistics that are robust to outliers.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.robust_scaler = RobustScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""


			Purpose:
			---------
			Fits the standard_scaler to the df.

			Parameters:
			-----------
				X (np.ndarray): Input df.
				y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.robust_scaler.fit( X )
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

			Purpose:
			---------
			Transforms the df using the fitted RobustScaler.

			Parameters:
			-----------
				X (np.ndarray): Input df.

			Returns:
			-----------
				np.ndarray: Scaled df.

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

		Purpose:
		---------
		Scales text vectors individually to unit norm.

	"""

	def __init__( self, norm: str = 'l2' ) -> None:
		super( ).__init__( )
		self.normal_scaler = Normalizer( norm = norm )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""


			Purpose:
			---------
			Fits the normalizer (no-op for Normalizer).

			Parameters:
			-----------
				X (np.ndarray): Input df.
				y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.normal_scaler.fit( X )
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


			Purpose:
			---------
			Applies normalization to each sample.

			Parameters:
			-----------
				X (np.ndarray): Input df.

			Returns:
			-----------
				np.ndarray: Normalized df.

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

		Purpose:
		---------
		Encodes categorical features as a one-hot numeric array.

	"""

	def __init__( self, unknown: str='ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = OneHotEncoder( sparse=False, handle_unknown =unknown )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""


			Purpose:
			---------
			Fits the hot_encoder to the categorical df.

			Parameters:
			-----------
				X (np.ndarray): Categorical text df.
				y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.hot_encoder.fit( X )
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


			Purpose:
			---------
			Transforms the text
			df into a one-hot encoded format.

			Parameters:
			-----------
				X (np.ndarray): Categorical text df.

			Returns:
			-----------
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


			Purpose:
			---------
		Encodes categorical
		features as ordinal integers.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = OrdinalEncoder( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""

			Purpose:
			________
			Fits the ordial encoder to the categorical df.

			Parameters:
			_____
			X (np.ndarray): Categorical text df.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.ordinal_encoder.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OrdinalEncoder'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Transforms the text df into ordinal-encoded format.


			Parameters:
			-----------
				X (np.ndarray): Categorical text df.

			Returns:
			-----------
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
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class MeanImputer( Metric ):
	"""

		Purpose:
		-----------
		Fills missing target_values using a specified strategy.

	"""

	def __init__( self, strat: str='mean' ) -> None:
		super( ).__init__( )
		self.mean_imputer = SimpleImputer( strategy=strat )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""


			Purpose:
			---------
			Fits the simple_imputer
			to the df.

			Parameters:
			-----------
				X (np.ndarray): Input df with missing target_values.
				y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.mean_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			---------
			Transforms the text
			df by filling in missing target_values.

			Parameters:
			-----------
				X (np.ndarray): Input df with missing target_values.

			Returns:
			-----------
				np.ndarray: Imputed df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.mean_imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


class NearestImputer( Metric ):
	"""

		Purpose:
		---------
		Fills missing target_values using k-nearest neighbors.

	"""


	def __init__( self ) -> None:
		super( ).__init__( )
		self.knn_imputer = KNNImputer( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""

			Purpose:
			________
			Fits the simple_imputer
			to the df.

			Parameters:
			_____
			X (np.ndarray): Input df with missing target_values.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
				Pipeline

		"""
		try:
			self.knn_imputer.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'fit( self, X: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			_________

			Transforms the text df by imputing missing target_values.

			Parameters:
			-----------
				X (np.ndarray): Input df
				with missing target_values.

			Returns:
			-----------
				np.ndarray: Imputed df.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				return self.knn_imputer.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestImputer'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )



class Cluster( ):
	"""

        Purpose:
        ---------
		Abstract base class for clustering models.
		This class defines the interface for clustering models, including methods
		for fitting, predicting, evaluating, and visualization.

	"""


	def fit( self, X: np.ndarray ) -> None:
		"""

	        Purpose:
	        ---------
			Fit the clustering model to the data.

			Parameters:
			----------
			X: The input data of shape (n_samples, n_features).

		"""
		raise NotImplementedError( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

	        Purpose:
	        ---------
			Predict the cluster labels for the input data.

			Parameters:
			----------
			X: Input features to cluster.

			Returns:
			---------
			np.ndarray

		"""
		raise NotImplementedError( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

	        Purpose:
	        ---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			----------
			X: Input features to cluster.

			Returns:
			---------
			float

		"""
		raise NotImplementedError( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

	        Purpose:
	        ---------
			Visualize clusters using a 2D scatter plot.

			Parameters:
			----------
			X: Input data of shape (n_samples, 2).

		"""
		raise NotImplementedError( )


class KMeansClustering( Cluster ):
	"""

		Purpose:
		---------
		Wrapper class for KMeans clustering.

	"""


	def __init__( self, num: int=8, rando: int=42 ) -> None:
		"""
			Purpose:
			---------
			Initialize the KMeans model.

			Parameters:
			----------
			num: Number of clusters to form.
			rando: Random seed for reproducibility.
			rando: int

		"""
		self.model: BaseEstimator=KMeans( n_clusters=num, random_state=rando )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the KMeans model on the dataset.

			Parameters:
			----------
			X: The input data.

		"""
		try:
			self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict the closest cluster each sample in X belongs to.

			Parameters:
			----------
			X: The input data.

			Returns:
			--------
			np.ndarray

		"""
		try:
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			----------
			X: The input data.
			X: np.ndarray

			Returns:
			---------
			float

		"""
		try:
			labels = self.model.predict( X )
			return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize clustering result using a scatter plot.

			Parameters:
			----------
			X: Input data of shape (n_samples, 2).

		"""
		try:
			if X is None:
				raise Exception( 'The input arguement "X" is required.' )
			else:
				labels = self.model.predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'viridis' )
				plt.title( "KMeans Cluster" )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class DbscanClustering( Cluster ):
	"""

		Wrapper class for DBSCAN clustering.

	"""


	def __init__( self, eps: float=0.5, min: int=5 ) -> None:
		"""

			Purpose:
			---------
			Initialize the DBSCAN model.

			Parameters:
			----------
			eps: float
			min: int

		"""
		self.model: BaseEstimator = DBSCAN( eps=eps, min_samples=min )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the DBSCAN model to the data.

			Parameters:
			----------
			X: Input features.

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict clusters using DBSCAN fit.

			Parameters:
			----------
			X: Input features.

			Returns:
			---------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate DBSCAN clusters with silhouette score.

			Parameters:
			----------
			X: Input features.
			X: np.ndarray

			Returns:
			---------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels ) if len( set( labels ) ) > 1 else -1.0
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize DBSCAN clusters.

			Parameters:
			----------
			X: 2D input features.

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'plasma' )
				plt.title( 'DBSCAN Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AgglomerativeClusteringModel( Cluster ):
	"""

		Purpose:
		---------
		Wrapper class for Agglomerative clustering.

	"""


	def __init__( self, num: int=2 ) -> None:
		"""

			Purpose:
			---------
			Initialize AgglomerativeClustering.

			Parameters:
			----------
			num: int

		"""
		self.model: BaseEstimator=AgglomerativeClustering( n_clusters=num )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit Agglomerative model to data.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict clusters using agglomerative clustering.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			---------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate agglomerative clustering using silhouette score.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			-------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize agglomerative clustering results.

			Parameters:
			----------
			X: 2D input data.
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'tab10' )
				plt.title( 'Agglomerative Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class SpectralClusteringModel( Cluster ):
	"""

		Purpose:
		---------
		Wrapper class for Spectral Cluster.

	"""


	def __init__( self, num: int=8 ) -> None:
		"""

			Purpose:
			---------
			Initialize the SpectralClustering model.

			Parameters:
			----------
			num: int

		"""
		self.model: BaseEstimator=SpectralClustering( n_clusters=num )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the SpectralClustering model.

			Parameters:
			----------
			X:  np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict clusters using SpectralClustering.

			Parameters:
			----------
			X: np.ndarray


			Return:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate SpectralClustering results.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize Spectral Cluster results.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Accent' )
				plt.title( 'Spectral Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class MeanShiftClustering( Cluster ):
	"""

		Purpose:
		---------
		Wrapper class for MeanShift clustering.

	"""


	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize MeanShift model.

		"""
		self.model: BaseEstimator = MeanShift( )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit MeanShift model to the data.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict clusters using MeanShift.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate MeanShift clustering.

			Parameters:
			----------
			X:  np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize MeanShift clustering.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Set1' )
				plt.title( 'MeanShift Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AffinityPropagationClustering( Cluster ):
	"""

		Purpose:
		---------
		Wrapper for AffinityPropagation clustering.

	"""


	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize AffinityPropagation model.

		"""
		self.model: BaseEstimator = AffinityPropagation( )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the model to data.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict clusters.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit( X ).labels_
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate clustering result.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit( X ).labels_
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize clustering with AffinityPropagation.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit( X ).labels_
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Paired' )
				plt.title( 'AffinityPropagation Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BirchClustering( Cluster ):
	"""

		Purpose:
		---------
		Wrapper for Birch clustering.

	"""


	def __init__( self, n_clusters: Optional[ int ] = None ) -> None:
		"""

			Purpose:
			---------
			Initialize Birch clustering.

			Parameters:
			----------
			num: Optional[int]

		"""
		self.model: BaseEstimator = Birch( n_clusters = n_clusters )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit Birch clustering model.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict clusters with Birch.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate Birch clustering.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize Birch clustering.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Dark2' )
				plt.title( 'Birch Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class OpticsClustering( Cluster ):
	"""

			Purpose:
			---------
			Wrapper for OPTICS clustering.

	"""


	def __init__( self, min: int=5 ) -> None:
		"""

			Purpose:
			---------
			Initialize OPTICS model.

			Parameters:
			----------
			min: int

		"""
		self.model: BaseEstimator = OPTICS( min_samples=min )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit OPTICS model.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict clusters with OPTICS.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> float:
		"""

			Purpose:
			---------
			Evaluate OPTICS clustering.

			Parameters:
			----------
			X: Input features.
			X: np.ndarray

			Returns:
			---------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize OPTICS clustering result.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c=labels, cmap='rainbow' )
				plt.title( 'OPTICS Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class PerceptronClassifier( Model ):
	"""


			Purpose:
			---------
			PerceptronClassifier classifier.

	"""


	def __init__( self, reg: float=0.0001, max: int=1000, mix: bool=True ) -> None:
		"""

			Purpose:
			---------
			Initialize the PerceptronClassifier linerar_model.


			Parameters:
			----------
			max_iter (int): Maximum number of iterations.
			Default is 1000.

		"""
		super( ).__init__( )
		self.perceptron_classifier: Perceptron=Perceptron( alpha=reg, max_iter=max, shuffle=mix )
		self.prediction: np.array=None
		self.mean_absolute_error: float=0.0
		self.mean_squared_error: float=0.0
		self.r_mean_squared_error: float=0.0
		self.r2_score: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the PerceptronClassifier linerar_model.

			Parameters:
			---------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Binary class labels.

			Returns:
			--------
			object

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.perceptron_classifier.fit( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'PerceptronClassifier'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray )'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			---------
			Predict binary class labels using the PerceptronClassifier.

			Parameters:
			---------
			X (pd.DataFrame): Feature matrix.

			Returns:
			---------
			np.ndarray

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


			Purpose:
			---------
			Compute accuracy of the PerceptronClassifier classifier.

			Parameters:
			---------
				X (np.ndarray): Test features.
				y (np.ndarray): True class labels.

			Returns:
			---------
			float

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
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""


			Purpose:
			-----------
			Evaluate classifier performance using standard classification metrics.

			Parameters:
			---------
				X (np.ndarray): Input features of shape (n_samples, n_features).
				y (np.ndarray): Ground truth class labels.

			Returns:
			---------
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
				self.median_absolute_error = median_absolute_error( y, self.prediction, squared=False )
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


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			---------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

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
				self.prediction = self.perceptron_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )



class MultilayerClassifier( Model ):
	"""

		Purpose:
		-----------
		This model optimizes the squared error using LBFGS or stochastic gradient descent.

		Activation function for the hidden layers:
			- ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
			- ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
			- ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
			- ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

		The solver for weight optimization:
			- ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
			- ‘sgd’ refers to stochastic gradient descent.
			- ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma and Diederik

	"""


	def __init__( self, hidden: tuple=( 100, ), activation='relu', solver='adam', alpha=0.0001,
	             learning: str='constant', rando: int=42 ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activation
		self.learning_rate = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.multilayer_classifier: MLPClassifier=MLPClassifier( hidden_layer_sizes=hidden,
			activation=activ, solver=solver, alpha=alpha, learning_rate=learning, random_state=42 )
		self.pipeline: Pipeline=Pipeline( steps )
		self.prediction:  np.array=None
		self.accuracy: float=0.0
		self.mean_absolute_error: float=0.0
		self.mean_squared_error: float=0.0
		self.r_mean_squared_error: float=0.0
		self.r2_score: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			-----------
			Fits all pipeline steps to the text df.

			Parameters:
			-----------
				X (np.ndarray): Input feature matrix.
				y (Optional[np.ndarray]): Optional target array.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.multilayer_classifier.fit( X, y )
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


			Purpose:
			-----------
			Applies all transformations in the pipeline to the text df.

			Parameters:
			-----------
				X (np.ndarray): Input feature matrix.

			Returns:
			-----------
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


			Purpose:
			-----------
			Fits and transforms all pipeline steps on the text df.

			Parameters:
			-----------
				X (np.ndarray): Input feature matrix.
				y (Optional[np.ndarray]): Optional target array.

			Returns:
			-----------
				np.ndarray: Transformed feature matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.multilayer_classifier.fit_transform( X, y )
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
			-----------
				Compute the R^2 accuracy of the model on the given test df.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True values.

			Returns:
			-----------
				float: R-squared accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.multilayer_classifier.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
			Evaluate the model using multiple regression metrics.


			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
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
			exception.cause = 'MultilayerClassifier'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			---------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

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
				self.prediction = self.multilayer_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerClassifier'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class MultilayerRegressor( Model ):
	"""

		Purpose:
		-----------
		This model optimizes the squared error using LBFGS or stochastic gradient descent.

		Activation function for the hidden layers:
			- ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
			- ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
			- ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
			- ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

		The solver for weight optimization:
			- ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
			- ‘sgd’ refers to stochastic gradient descent.
			- ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma and Diederik

	"""


	def __init__( self, hidden: tuple=( 100, ), activation='relu', solver='adam', alpha=0.0001,
	             learning: str='constant', rando: int=42 ) -> None:
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activation
		self.learning_rate = learning
		self.solver = solver
		self.alpha = alpha
		self.random_state = rando
		self.multilayer_regressor: MLPRegressor=MLPRegressor( hidden_layer_sizes=hidden,
			activation=activ, solver=solver, alpha=alpha, learning_rate=learning, random_state=42 )
		self.pipeline: Pipeline=Pipeline( steps )
		self.prediction:  np.array=None
		self.accuracy: float=0.0
		self.mean_absolute_error: float=0.0
		self.mean_squared_error: float=0.0
		self.r_mean_squared_error: float=0.0
		self.r2_score: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""

			Purpose:
			-----------
			Fits all pipeline steps to the text df.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.
			y (Optional[np.ndarray]): Optional target array.

			Returns:
			--------
			Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.pipeline = self.multilayer_regressor.fit( X, y )
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


			Purpose:
			-----------
			Applies all transformations in the pipeline to the text df.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.

			Returns:
			-----------
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


			Purpose:
			-----------
			Fits and transforms all pipeline steps on the text df.

			Parameters:
			-----------
			X (np.ndarray): Input feature matrix.
			y (Optional[np.ndarray]): Optional target array.

			Returns:
			-----------
			np.ndarray: Transformed feature matrix.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.multilayer_regressor.fit_transform( X, y )
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
			-----------
			Compute the R^2 accuracy of the model on the given test df.

			Parameters:
			-----------
			X (np.ndarray): Test features.
			y (np.ndarray): True values.

			Returns:
			-----------
			float: R-squared accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.multilayer_regressor.predict( X )
				self.accuracy = r2_score( y, self.prediction )
				return self.accuracy
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MultilayerRegressor'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
			Evaluate the model using multiple regression metrics.


			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
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
			-----------
				Plot actual vs predicted target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
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

		Purpose:
		-----------
		Ordinary Least Squares Regression fits a linear model with coefficients to minimize the
		residual sum of squares between the observed targets in the dataset, and the targets
		predicted by the linear approximation. The coefficient estimates for Ordinary Least Squares
		rely on the independence of the features.

		When features are correlated and the columns of the design matrix have an approximately
		linear dependence, the design matrix becomes close to singular and as a result,
		the least-squares estimate becomes highly sensitive to random errors in the observed target,
		producing a large variance. This situation of multicollinearity can arise, for example,
		when data are collected without an experimental design.

	"""


	def __init__( self ) -> None:
		"""

			Purpose:
			-----------
			Initialize the Linear Regression linerar_model.

			Parameters:
			-----------
			fit_intercept (bool): Whether to include an intercept term. Default is True.
			copy_X (bool): Whether to copy the feature matrix. Default is True.

		"""
		super( ).__init__( )
		self.linerar_regressor: LinearRegression=LinearRegression( fit_intercept=True, copy_X=True )
		self.prediction: np.arry=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the OLS regression linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Target vector.

			Returns:
			--------
			Pipeline

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


			Purpose:
			-----------
			Predict target target_values using the OLS linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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


			Purpose:
			-----------
			Compute the R-squared accuracy of the OLS model.

			Parameters:
			-----------
			X (np.ndarray): Test features.
			y (np.ndarray): True target target_values.

			Returns:
			-----------
			float: R-squared accuracy.

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
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float] | None:
		"""


			Purpose:
			-----------
			Evaluate the model using multiple regression metrics.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
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
			-----------
			Plot actual vs predicted target_values.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): True target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.linerar_regressor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Linear Regression: Observed vs Projected' )
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


class RidgeClassification( Model ):
	"""

		Purpose:
		--------
		This classifier first converts binary targets to {-1, 1} and then treats the problem as a
		regression task, optimizing the same objective as above. The predicted class corresponds
		to the sign of the regressor’s prediction. For multiclass classification, the problem is
		treated as multi-output regression, and the predicted class corresponds to the output
		with the highest value.

		It might seem questionable to use a (penalized) Least Squares loss to fit a classification
		model instead of the more traditional logistic or hinge losses. However, in practice,
		all those models can lead to similar cross-validation scores in terms of accuracy
		or precision/recall, while the penalized least squares loss used by the RidgeClassification
		allows for a very different choice of the numerical solvers with
		distinct computational performance profiles.

	"""


	def __init__( self, alpha: float=1.0, solver: str='auto', max: int=1000,
	              rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the Ridge Classifierlinerar_model.

			Parameters:
			-----------
			- alpha (float): Regularization strength. Default is 1.0.
			- solver (str): Solver to use. Default is 'auto'.
			- max (int): max iterations
			- rando (int): random seed

		"""
		super( ).__init__( )
		self.alpha: float=alpha
		self.solver: str=solver
		self.max_iter: int=max
		self.random_state: int=rando
		self.ridge_classifier: RidgeClassifier=RidgeClassifier( alpha=self.alpha,
			solver=self.solver, max_iter=self.max_iter, random_state=self.random_state )
		self.prediction: np.array = None
		self.accuracy: float = 0.0
		self.precision: float = 0.0
		self.recall: float = 0.0
		self.f1_score: float = 0.0
		self.roc_auc_score: float = 0.0
		self.correlation_coefficient: float = 0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""


			Purpose:
			-----------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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


			Purpose:
			-----------
			Project target target_values using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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


			Purpose:
			-----------
			Compute the R-squared accuracy for the Ridge model.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

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
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
				Evaluates the Ridge model
				using multiple metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				dict: Evaluation metrics including MAE, RMSE, R², etc.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.accuracy = accuracy_score( y, self.prediction )
				self.precision = precision_score( y, self.prediction, average = 'binary' )
				self.recall = mean_squared_error( y, self.prediction, average = 'binary' )
				self.f1_score = f1_score( y, self.prediction, average = 'binary' )
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


			Purpose:
			-----------
			Plot predicted vs actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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


class RidgeRegression( Model ):
	"""

		Purpose:
		--------
		Solves a regression model where the loss function is the linear least squares function and
		alpha is given by the l2-norm. Also known as Ridge Regression
		or Tikhonov alpha. This estimator has built-in support for
		multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets))

		The complexity parameter  controls the amount of shrinkage: the larger the value of alpha,
		the greater the amount of shrinkage and thus the coefficients become
		more robust to collinearity.

		The algorithm used to fit the model is coordinate descent. To avoid unnecessary memory
		duplication the X argument of the fit method should be directly passed as a
		Fortran-contiguous numpy array. Regularization improves the conditioning of the problem
		and reduces the variance of the estimates. Larger values specify stronger alpha.
		Alpha corresponds to 1 / (2C) in other linear models such as LogisticRegression or LinearSVC.
		If an array is passed, penalties are assumed to be specific to the targets.

	"""


	def __init__( self, alpha: float=1.0, solver: str='auto', max: int=1000, rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the RidgeRegressor linerar_model.

			Attributes:
			-----------
				linerar_model (Ridge): Internal RidgeRegressor regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						solver (str): Solver to use. Default is 'auto'.

		"""
		super( ).__init__( )
		self.alpha: float=alpha
		self.solver: str=solver
		self.max_iter: int=max
		self.random_state: int=rando
		self.ridge_regressor: Ridge=Ridge( alpha=self.alpha, solver=self.solver,
			max_iter=self.max_iter, random_state=self.random_state )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""


			Purpose:
			-----------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'RidgeRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Project target target_values using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'RidgeRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""


			Purpose:
			-----------
			Compute the R-squared accuracy for the Ridge model.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

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
			exception.cause = 'RidgeRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
				Evaluates the Ridge model
				using multiple metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'RidgeRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot predicted vs actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'RidgeRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LassoRegression( Model ):
	"""

		Purpose:
		--------
		Linear Model trained with L1 for the regularizer. Regularization improves the
		conditioning of the problem and reduces the variance of the estimates. Larger values
		specify stronger alpha. Technically the Lasso model is optimizing the same
		objective function as the Elastic Net with l1_ratio=1.0 (no L2 penalty).
		The algorithm used to fit the model is coordinate descent.

		To avoid unnecessary memory duplication the X argument of the fit method should be directly
		passed as a Fortran-contiguous numpy array. Regularization improves the conditioning of the
		problem and reduces the variance of the estimates. Larger values specify stronger
		alpha. Alpha corresponds to 1 / (2C) in other linear models such as
		LogisticRegression or LinearSVC. If an array is passed, penalties are assumed to be
		specific to the targets. Hence they must correspond in number.

	"""


	def __init__( self, alph: float=1.0, max: int=500, rando: int=42,
	              mix='random' ) -> None:
		"""


			Purpose:
			-----------
			Initialize the LassoRegression linerar_model.

			Attributes:
			-----------
				linerar_model (Lasso): Internal LassoRegression regression linerar_model.
					Parameters:
						alpha (float): Regularization strength. Default is 1.0.
						max_iter (int): Maximum number of iterations. Default is 1000.

		"""
		super( ).__init__( )
		self.alpha: float=alph
		self.max_iter: int=max
		self.random_state: int=rando
		self.lasso_regressor: Lasso=Lasso( alpha=self.alpha, max_iter=self.max_iter,
			random_state=self.random_state )
		self.prediction:  np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			--------
			Fit the LassoRegression
			regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'LassoRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Predict target target_values using the LassoRegression linerar_model.


			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'LassoRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""


			Purpose:
			-----------
			Compute R^2 accuracy for the Lasso model.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'LassoRegression'
			exception.method = 'accuracy(self, X: np.ndarray, y: np.ndarray) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""


			Purpose:
			-----------
			Evaluate the Lasso model using multiple regression metrics.


			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'LassoRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot actual vs. predicted target_values.

			Parameters:
			-----------
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
			exception.cause = 'LassoRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class ElasticNetRegression( Model ):
	"""

		Purpose:
		--------
		A Linear regression with combined L1 and L2 for alpha.

	"""


	def __init__( self, alpha: float=1.0, ratio: float=0.5, max: int=200,
	              rando: int=42, select: str='random' ) -> None:
		"""

			Purpose:
			-----------
			Initialize the ElasticNet Regressor linerar_model.


			Parameters:
			----------
			alpha (float): Overall alpha strength. Default is 1.0.
			ratio (float): Mixing parameter (0 = RidgeRegressor, 1 = LassoRegression). Default is 0.5.
			max (int): Maximum number of iterations. Default is 200.
			rando (int): Number of random iterations. Default is 42.
			select (str): selection

		"""
		super( ).__init__( )
		self.alpha: int=alpha
		self.ratio: float=ratio
		self.random_state: int=rando
		self.selection: str=select
		self.max: int=max
		self.elasticnet_regressor: ElasticNet=ElasticNet( alpha=self.alpha, l1_ratio=self.ratio,
			random_state=self.random_state, max_iter=self.max, selection=self.selection )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""


			Purpose:
			-----------
			Fit the ElasticNetRegression regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'ElasticNetRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Predict target target_values using the ElasticNetRegression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'ElasticNetRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""


			Purpose:
			-----------
			Compute R^2 accuracy on the test set.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'ElasticNetRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""


			Purpose:
			-----------
			Evaluate model performance using regression metrics.


			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
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
			exception.cause = 'ElasticNetRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot actual vs. predicted regression output.

			Parameters:
			-----------
				X (np.ndarray): Input features.
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
			exception.cause = 'ElasticNetRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class LogisticRegressor( Model ):
	"""

		Purpose:
		--------
		This class implements regularized logistic regression using the ‘liblinear’ library,
		‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that alpha is
		applied by default. It can handle both dense and sparse input. Use C-ordered arrays or
		CSR matrices containing 64-bit floats for optimal performance;
		any other input format will be converted (and copied). The ‘newton-cg’, ‘sag’, and
		‘lbfgs’ solvers support only L2 alpha with primal formulation, or no
		alpha. The ‘liblinear’ solver supports both L1 and L2 alpha,
		with a dual formulation only for the L2 alpha. The Elastic-Net alpha
		is only supported by the ‘saga’ solver.

	"""


	def __init__( self, c: float=1.0, penalty: str='l2', max: int=1000, solver: str='lbfgs' ) -> None:
		"""

			Purpose:
			--------
			Initialize the Logistic Regression linerar_model.

			Parameters:
			-----------
				max (int): Maximum number of iterations. Default is 1000.
				solver (str): Algorithm to use in optimization. Default is 'lbfgs'.

		"""
		super( ).__init__( )
		self.alpha: float=c
		self.penalty: str=penalty
		self.max_iter: int=max
		self.solver: str=solver
		self.logistic_regressor: LogisticRegression=LogisticRegression( C=self.alpha, max_iter=self.max_iter,
			solver=self.solver, penalty=self.penalty )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the logistic regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target class labels.

			Returns:
			--------
				Pipeline

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

			Purpose:
			-----------
			Predict class labels using the logistic regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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

			Purpose:
			-----------
			Compute classification accuracy.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				float: Accuracy accuracy.

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
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
			Evaluate the classifier using multiple classification metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features of shape (n_samples, n_features).
				y (np.ndarray): True labels of shape (n_samples,).

			Returns:
			-----------
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

			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
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


class BayesianRidgeRegression( Model ):
	"""

		Purpose:
		--------
		Bayesian regression techniques can be used to include alpha parameters in the
		estimation procedure: the alpha parameter is not set in a hard sense
		but tuned to the df at hand. This can be done by introducing uninformative priors over
		the hyper parameters of the model. The alpha used in Ridge regression and
		classification is equivalent to finding a maximum a posteriori estimation under a
		Gaussian prior over the coefficients with precision . Instead of setting lambda manually,
		it is possible to treat it as a random variable to be estimated from the df.

		This implementation is based on the algorithm described in Appendix A of (Tipping, 2001)
		where updates of the alpha parameters are done as suggested in (MacKay, 1992).
		Note that according to A New View of Automatic Relevance Determination
		(Wipf and Nagarajan, 2008) these update rules do not guarantee that the marginal likelihood
		is increasing between two consecutive iterations of the optimization.

	"""


	def __init__( self, max: int=300, shape_alpha: float=1e-06,
	              scale_alpha: float=1e-06, shape_lambda: float=1e-06, scale_lambda: float=1e-06 ) -> None:
		"""

			Purpose:
			-----------
				Initializes the BayesianRidgeRegression.

		"""
		super( ).__init__( )
		self.max_iter: int=max
		self.shape_alpha: float=shape_alpha
		self.scale_alpha: float=scale_alpha
		self.shape_lambda: float=shape_lambda
		self.scale_lambda: float=scale_lambda
		self.bayesian_ridge_regressor: BayesianRidge=BayesianRidge( n_iter=self.max_iter,
			alpha_1=self.shape_alpha, alpha_2=self.scale_alpha,
			lambda_1=self.shape_lambda, lambda_2=self.scale_lambda )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
				Fit the Bayesian RidgeRegressor
				regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			-----------
				Predicts target target_values
				using the Bayesian linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			-----------
				Compute the R^2 accuracy
				of the model on test df.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
			Evaluate the Bayesian model with regression metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
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
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot predicted vs. actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
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
			exception.cause = 'BayesianRidgeRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StochasticDescentClassification( Model ):
	"""

		Purpose:
		--------
		Linear classifiers (SVM, logistic regression, etc.) with SGD training. This estimator
		implements regularized linear models with stochastic gradient descent (SGD) learning:
		the gradient of the loss is estimated each sample at a time and the model is updated along
		the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch
		(online/out-of-core) learning via the partial_fit method. For best results using the
		default learning rate schedule, the data should have zero mean and unit variance.

		This implementation works with data represented as dense or sparse arrays of floating point
		 values for the features. The model it fits can be controlled with the loss parameter;
		 by default, it fits a linear support vector machine (SVM).

		The regularizer is a penalty added to the loss function that shrinks model parameters
		towards the zero vector using either the squared euclidean norm L2 or the absolute norm
		L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
		because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
		 models and achieve online feature selection.

	"""


	def __init__( self, loss: str='hinge', max: int=5, reg: str='l2' ) -> None:
		"""

			Purpose:
			-----------
			Initialize the SGDClassifier linerar_model.

			Parameters:
			-----------
			loss (str): Loss function to use. Defaults to 'hinge'.
			reg (str): Regularization function to use. Default is 'l2'.
			max (int): Maximum number of passes over the df. Default is 1000.

		"""
		super( ).__init__( )
		self.loss: str=loss
		self.max_iter: int=max
		self.regularization: str=reg
		self.stochastic_classifier: SGDClassifier=SGDClassifier( loss=self.loss,
			max_iter=self.max_iter, penalty=self.regularization )
		self.prediction: np.array=None
		self.mean_absolute_error: float=0.0
		self.mean_squared_error: float=0.0
		self.r_mean_squared_error: float=0.0
		self.r2_score: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the SGD classifier linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.stochastic_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

				Purpose:
				-----------
					Predict class labels using the SGD classifier.

				Parameters:
				-----------
					X (pd.DataFrame): Feature matrix.

				Returns:
				-----------
					np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stochastic_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for the SGDRegressor.

			Parameters:
			-----------
			X (np.ndarray): Test features.
			y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
			float: R^2 accuracy.

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
			exception.cause = ''
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Purpose:
			-----------
			Evaluate the classifier using standard metrics.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix of shape (n_samples, n_features).
			y (np.ndarray): True class labels of shape (n_samples,).

			Returns:
			-----------
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
			exception.cause = ''
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
			X (np.ndarray): Input features.
			y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StochasticDescentRegression( Model ):
	"""

		Purpose:
		--------
		Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative
		rate of linear classifiers under convex loss functions such as
		(linear) Support VectorStore Machines and Logistic Regression. Even though SGD has been around
		in the machine rate community for a long time, it has received a considerable amount
		of attention just recently in the context of large-scale rate.

		SGD has been successfully applied to large-scale and sparse machine rate problems
		often encountered in text classification and natural language processing.
		Given that the df is sparse, the classifiers in this module easily scale to problems
		with more than 10^5 training examples and more than 10^5 features.

		The regularizer is a penalty added to the loss function that shrinks model parameters
		towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1
		or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
		because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
		models and achieve online feature selection.

		This implementation works with data represented as dense numpy arrays of floating point
		values for the features.
	"""


	def __init__( self, loss: str='hinge', max: int=5, reg: str='l2' ) -> None:
		"""

			Purpose:
			-----------
			Initialize the SGDRegressor

			Parameters:
			-----------
			- alpha (float): Regulation
			- reg (str): Regularization term. Default is 'l2'.
			- max (int): Maximum number of passes. Default is 1000.

		"""
		super( ).__init__( )
		self.loss: str=loss
		self.max_iter: int=max
		self.penalty: str=reg
		self.stochastic_regressor: SGDRegressor=SGDRegressor( loss=self.loss,
			max_iter=self.max_iter, penalty=self.penalty )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the SGD regressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target target_values.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.stochastic_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticDescentRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Predict target_values using the SGD regressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.stochastic_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StochasticDescentRegression'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				float: R^2 accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
			Evaluate regression model performance.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
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
			exception.cause = 'StochasticDescentRegression'
			exception.method = ''
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot predicted vs. actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True target target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.stochastic_regressor.predict( X )
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
			exception.cause = 'StochasticDescentRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )



class NearestNeighborClassification( Model ):
	"""

		Purpose:
		--------
		The principle behind the k-nearest neighbor methods is to find a predefined number of
		training samples closest in distance to the new point, and predict the label from these.
		The number of samples can be a user-defined constant (k-nearest neighbor rate),
		or vary based on the local density of points (radius-based neighbor rate).
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training df
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

	"""


	def __init__( self, num: int=5 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

			Attributes:
			-----------
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.n_neighbors: int=num
		self.neighbor_classifier: KNeighborsClassifier=KNeighborsClassifier( n_neighbors=self.n_neighbors )
		self.prediction: np.array=None
		self.score: float=0.0
		self.mean_absolute_error: float=0.0
		self.mean_squared_error: float=0.0
		self.r_mean_squared_error: float=0.0
		self.r2_score: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Fit the KNN classifier linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Class labels.

			Returns:
			-------
			None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.neighbor_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Predict class labels using the KNN classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.neighbor_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""


			Purpose:
			-----------
			Compute classification accuracy for k-NN.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth labels.

			Returns:
			-----------
				float: Accuracy accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return accuracy_score( y, self.neighbor_classifier.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = ''
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""


			Purpose:
			-----------
			Evaluate classification performance using various metrics.


			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): True class labels of shape (n_samples,).

			Returns:
			-----------
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
			exception.cause = ''
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.neighbor_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class NearestNeighborRegression( Model ):
	"""

		Purpose:
		--------
		The principle behind k-nearest neighbor methods is to find a predefined number of
		training samples closest in distance to the new point, and predict the label from these.
		The number of samples can be a user-defined constant (k-nearest neighbor rate),
		or vary based on the local density of points (radius-based neighbor rate).
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training df
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

	"""


	def __init__( self, num: int=5 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsRegressor linerar_model.

			Parameters:
			-----------
				linerar_model (KNeighborsRegressor): Internal non-parametric regressor.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.n_neighbors = num
		self.neighbor_regressor: KNeighborsRegressor=KNeighborsRegressor( n_neighbors=num )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""


			Purpose:
			-----------
			Fit the KNN regressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target target_values.

			Returns:
			--------
				Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.neighbor_regressor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Predict target_values using the KNN regressor.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.neighbor_regressor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for k-NN regressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.neighbor_regressor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Purpose:
			-----------
			Evaluate k-NN regression performance with multiple metrics.


			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
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
			exception.cause = 'NearestNeighborRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
				Plot predicted vs actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.neighbor_regressor.predict( X )
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
			exception.cause = 'RandomForestRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class DecisionTreeClassification( Model ):
	'''

		Purpose:
		--------
		Decision Trees (DTs) are a non-parametric supervised learning method used for
		classification. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the data features.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from data
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.

	'''


	def __init__( self, criterion='gini', splitter='best',  depth=3, rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

			Attributes:
			-----------
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.criterion: str=criterion
		self.splitter: str=splitter
		self.max_depth: int=depth
		self.random_state = rando
		self.dt_classifier: DecisionTreeClassifier=DecisionTreeClassifier( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth )
		self.prediction: np.array=None
		self.score: float=0.0
		self.mean_absolute_error: float=0.0
		self.mean_squared_error: float=0.0
		self.r_mean_squared_error: float=0.0
		self.r2_score: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Fit the KNN classifier linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Class labels.

			Returns:
			-------
			None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.dt_classifier.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassification'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""


			Purpose:
			-----------
			Predict class labels using the KNN classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted class labels.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.dt_classifier.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""


			Purpose:
			-----------
			Compute classification accuracy for k-NN.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth labels.

			Returns:
			-----------
				float: Accuracy accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				return accuracy_score( y, self.dt_classifier.predict( X ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""


			Purpose:
			-----------
			Evaluate classification performance using various metrics.


			Parameters:
			-----------
				X (np.ndarray): Feature matrix of shape (n_samples, n_features).
				y (np.ndarray): True class labels of shape (n_samples,).

			Returns:
			-----------
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
			exception.cause = 'DecisionTreeClassification'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""


			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.dt_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Random Forest Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )






class DecisionTreeRegression( Model ):
	'''

		Purpose:
		--------
		Decision Trees (DTs) are a non-parametric supervised learning method used for
		regression. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the data features.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from data
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.

	'''


	def __init__( self, criterion='squared_error', splitter='best',  depth=3, rando: int=42 ) -> None:
		"""


			Purpose:
			-----------
			Initialize the KNeighborsClassifier linerar_model.

			Attributes:
			-----------
				linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
					Parameters:
						n_neighbors (int): Number of neighbors to use. Default is 5.

		"""
		super( ).__init__( )
		self.criterion: str=criterion
		self.splitter: str=splitter
		self.max_depth: int=depth
		self.random_state = rando
		self.dt_regresssor: DecisionTreeRegressor=DecisionTreeRegressor( criterion=self.criterion,
			splitter=self.splitter, max_depth=self.max_depth, random_state=rando )
		self.prediction: np.array=None
		self.score: float=0.0
		self.mean_absolute_error: float=0.0
		self.mean_squared_error: float=0.0
		self.r_mean_squared_error: float=0.0
		self.r2_score: float=0.0
		self.explained_variance_score: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""


			Purpose:
			-----------
			Fit the Decision-Tree regressor linerar_model.

			Parameters:
			-----------
			X (np.ndarray): Feature matrix.
			y (np.ndarray): Target target_values.

			Returns:
			--------
			Pipeline

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.pipeline = self.dt_regresssor.fit( X, y )
				return self.pipeline
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Predict target_values using the KNN regressor.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
				np.ndarray: Predicted target_values.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			else:
				self.prediction = self.dt_regresssor.predict( X )
				return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for k-NN regressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.dt_regresssor.predict( X )
				return r2_score( y, self.prediction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'score( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Purpose:
			-----------
			Evaluate k-NN regression performance with multiple metrics.


			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): True target target_values.

			Returns:
			-----------
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
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
				Plot predicted vs actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				None

		"""
		try:
			if X is None:
				raise Exception( 'The argument "X" is required!' )
			elif y is None:
				raise Exception( 'The argument "y" is required!' )
			else:
				self.prediction = self.dt_regresssor.predict( X )
				plt.scatter( y, self.prediction )
				plt.xlabel( 'Observed' )
				plt.ylabel( 'Projected' )
				plt.title( 'Decision Tree Regression: Observed vs Projected' )
				plt.create_graph( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'r--' )
				plt.grid( True )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DecisionTreeRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RandomForestClassification( Model ):
	"""

		Purpose:
		--------
		In random forests, each tree in the ensemble is built from a sample drawn with replacement
		(i.e., a bootstrap sample) from the training set.

		Furthermore, when splitting each node during the construction of a tree,
		the best split is found either from all input features or a random subset of
		size max_features.

		The purpose of these two sources of randomness is to decrease the variance
		of the forest estimator. Individual decision trees typically exhibit high variance
		and tend to overfit. The injected randomness in forests yield decision trees with
		decoupled prediction errors. By taking an average of those predictions,
		errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.

	"""


	def __init__( self, est: int=10, crit: str='gini', max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			-----------
			Initializes the RandomForestClassification.

		"""
		super( ).__init__( )
		self.n_estimators: int=est
		self.criterion: str=crit
		self.max_depth: int=max
		self.random_state: int=rando
		self.random_forest_classifier: RandomForestClassifier=RandomForestClassifier( n_estimators=est,
			criterion=crit, random_state=rando )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the classifier.


			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline


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
			exception.cause = 'RandomForestClassification'
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
			exception.cause = 'RandomForestClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			-----------
			Compute R^2 accuracy for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'RandomForestClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Purpose:
			-----------
			Evaluate the Lasso model using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'RandomForestClassification'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot confusion matrix for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
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
			exception.cause = 'RandomForestClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class RandomForestRegression( Model ):
	"""

		Purpose:
		--------
		In random forests, each tree in the ensemble is built from a sample drawn with replacement
		(i.e., a bootstrap sample) from the training set.

		Furthermore, when splitting each node during the construction of a tree,
		the best split is found either from all input features or a random subset of
		size max_features.

		The purpose of these two sources of randomness is to decrease the variance
		of the forest estimator. Individual decision trees typically exhibit high variance
		and tend to overfit. The injected randomness in forests yield decision trees with
		decoupled prediction errors. By taking an average of those predictions,
		some errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.

	"""


	def __init__( self, est: int=10, crit: str='gini', max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			-----------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			-----------
			alpha (float): Regularization strength. Default is 1.0.
			solver (str): Solver to use. Default is 'gini'.
			max (int): maximum iterations
			rando (int): random seed value

		"""
		super( ).__init__( )
		self.n_estimators: int=est
		self.criterion: str=crit
		self.max_depth: int=max
		self.random_state: int=rando
		self.random_forest_regressor: RandomForestRegressor=RandomForestRegressor( n_estimators=est,
			criterion=crit, random_state=rando )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			-----------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.
			y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'RandomForestRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Purpose:
			-----------
			Project target target_values using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'RandomForestRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			-----------
			Compute the R-squared accuracy for the Ridge model.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

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
			exception.cause = 'RandomForestRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Purpose:
			-----------
			Evaluates the Ridge model using multiple metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'RandomForestRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			-----------
			Plot predicted vs actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'RandomForestRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class GradientBoostingClassification( Model ):
	"""

		Purpose:
		--------
		Gradient Boosting Classifier builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

		The features are always randomly permuted at each split. Therefore, the best found split
		may vary, even with the same training data and max_features=n_features, if the improvement
		of the criterion is identical for several splits enumerated during the search of the best
		split. To obtain a deterministic behaviour during fitting, rando has to be fixed.

	"""


	def __init__( self, lss: str='deviance', rate: int=0.1,
	              est: int=100, max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			________
			Initialize the GradientBoostingClassification.

			Parameters:
			___________
			lss: str
			rate: int
			estimators: int
			max: int
			rando: int

		"""
		super( ).__init__( )
		self.loss: str=lss
		self.learning_rate: float=rate
		self.n_estimators: int=est
		self.max_depth: int=max
		self.random_state: int=rando
		self.gradient_boost_classifier: GradientBoostingClassifier=GradientBoostClassifier( loss=lss,
			learning_rate=rate, n_estimators=est, max_depth=max, random_state=rando )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			________
				Fit the model to the training df.

			Parameters:
			__________
				X (np.ndarray): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

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
				float: Accuracy accuracy.

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


class GradientBoostingRegression( Model ):
	"""

		Purpose:
		--------
		Gradient Boosting builds an additive model in a forward stage-wise fashion;
		it allows for the optimization  of arbitrary differentiable loss functions.
		In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
		or multinomial deviance loss function. Binary classification is a special case where
		only a single regression tree is induced.

	"""


	def __init__( self, lss: str='deviance', rate: float=0.1,
	              est: int=100, max: int=3, rando: int=42 ) -> None:
		"""

			Purpose:
			_______
				Initialize the GradientBoostingRegression.

			Parameters:
			___________
			lss: str
			rate: int
			estimators: int
			max: int
			rando: int

		"""
		super( ).__init__( )
		self.loss: str=lss
		self.learning_rate: float=rate
		self.n_estimators: int=est
		self.max_depth: int=max
		self.gradient_boost_regressor: GradientBoosingRegressor=GradientBoostingRegressor( loss=lss,
			learning_rate=rate, n_estimators=est, max_depth=max, random_state=rando )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			_______
				Fit the gradient boosting model.

			Parameters:
			___________
				X (np.ndarray): Training features.
				y (np.ndarray): Training targets.

			Returns:
			--------
				Pipeline

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
			float: R² accuracy.

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


class AdaBoostClassification( Model ):
	"""

		Purpose:
		---------
		An Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.

	"""


	def __init__( self, est: int=100, max: int=3 ) -> None:
		"""

			Initialize the Random Forest Classifier.

		"""
		super( ).__init__( )
		self.max_depth: int=max
		self.n_estimators: int=est
		self.ada_boost_classifier: AdaBoostClassifier=AdaBoostClassifier( n_estimators=est,
			max_depth=max )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def scale( self ) -> None:
		"""

			Purpose:
			-----------
			Scale numeric features using selected scaler.


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


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			_______
				Fit the classifier.

			Parameters:
			_________
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'AdaBoostClassification'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'AdaBoostClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'AdaBoostClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'AdaBoostClassification'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
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
			exception.cause = 'AdaBoostClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AdaBoostRegression( Model ):
	"""

		Purpose:
		---------
		The core principle of Boost Regression is to fit a sequence of weak learners
		(i.e., models that are only slightly better than random guessing,
		such as small decision trees) on repeatedly modified versions of the df.
		The predictions from all of them are then combined through a weighted
		majority vote (or sum) to produce the final prediction.

	"""


	def __init__( self, est: int=100, max: int=3 ) -> None:
		"""

			Purpose:
			--------
			Initialize the Ridge Regression Model.


			Parameters:
			----------
			estimators (int): The number of estimators used. Default is 100.
			max (int): The maximum number of iterations. Default is '3'.

		"""
		super( ).__init__( )
		self.max_depth: int=max
		self.n_estimators: int=est
		self.ada_boost_regressor: AdaBoostRegressor=AdaBoostRegressor( n_estimators=est,
			max_depth=max )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			--------
				Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			-----------
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
			exception.cause = 'AdaBoostRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Project target target_values
			using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'AdaBoostRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Purpose:
			--------
			Compute the R-squared
			accuracy for the Ridge model.

			Parameters:
			----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			--------
				float: R-squared accuracy.

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
			exception.cause = 'AdaBoostRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Evaluates the Ridge model
			using multiple metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'AdaBoostRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot predicted vs
			actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'AdaBoostRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BaggingClassification( Model ):
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


	def __init__( self, base: object=None, num: int=10, max: int=1, rando: int=42 ) -> None:
		"""

			Initialize the BaggingClassification.

		"""
		super( ).__init__( )
		self.base_estimator: object=base
		self.n_estimators: int=num
		self.max_features: int=max
		self.random_state: int=rando
		self.bagging_classifier: BaggingClassifier=BaggingClassifier( base_estimator=base,
			n_estimators=num, max_features=self.max, random_state=rando )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			--------
			 Fit the classifier.

			Parameters:
			----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			-------
				Pipeline

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
			exception.cause = 'BaggingClassification'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'BaggingClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'BaggingClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'BaggingClassification'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Plot confusion matrix
			for classifier predictions.

			Parameters:
			------------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

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
				self.prediction = self.bagging_classifier.predict( X )
				cm = confusion_matrix( y, self.prediction )
				ConfusionMatrixDisplay( confusion_matrix=cm ).create_graph( )
				plt.title( 'Bagging Classifier Confusion Matrix' )
				plt.grid( False )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BaggingClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BaggingRegression( Model ):
	"""

		Purpose:
		--------
		 Bagging methods form a class of algorithms which build several instances of a black-box
		 estimator on random subsets of the original training set and then aggregate their
		 individual predictions to form a final prediction. These methods are used as a way
		 to reduce the variance of a base estimator (e.g., a decision tree), by introducing
		 randomization into its construction procedure and then making an ensemble out of it.

		 Bagging methods constitute a very simple way to improve with respect
		 to a single model, without making it necessary to adapt the underlying base algorithm.
		 As they provide a way to reduce overfitting, bagging methods work best with strong and
		 complex models (e.g., fully developed decision trees), in contrast with boosting methods
		 which usually work best with weak models (e.g., shallow decision trees).

	"""


	def __init__( self, base: object=None, num: int=10, max: int=1, rando: int=42 ) -> None:
		"""

			Purpose:
			--------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			-----------
			alpha (float): Regularization strength. Default is 1.0.
			solver (str): Solver to use. Default is 'auto'.

		"""
		super( ).__init__( )
		self.base_estimator: object=base
		self.n_estimators: int=num
		self.max_features: int=max
		self.random_state: int=rando
		self.bagging_regressor = BaggingRegression( base_estimator=base,
			n_estimator=num, max_features=max, random_state=rando )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			--------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'BaggingRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Project target target_values
			using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'BaggingRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute the R-squared
			accuracy for the Ridge model.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

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
			exception.cause = 'BaggingRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Evaluates the Ridge model
			using multiple metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'BaggingRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot predicted vs
			actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'BaggingRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class VotingClassification( Model ):
	"""

		Purpose:
		--------
		The idea behind the VotingClassification is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class labels. Such a classifier can be useful for a set of equally well
		performing model in order to balance out their individual weaknesses.

	"""


	def __init__( self, estimators: List[ (str, object) ], vote= 'hard' ) -> None:
		"""

			Initialize the RandomForestClassification.

		"""
		super( ).__init__( )
		self.estimators: List[ ( str, object ) ]=estimators
		self.voting: str=vote
		self.voting_classifier: VotingClassifier=VotingClassifier( estimators=estimators,
			voting=vote )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
				Fit the classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'VotingClassification'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'VotingClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'VotingClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'VotingClassification'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
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
			exception.cause = 'VotingClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class VotingRegression( Model ):
	"""

		Purpose:
		--------

		Prediction voting regressor for unfitted estimators. A voting regressor is an ensemble
		meta-estimator that fits several base regressors, each on the whole dataset.
		Then it averages the individual predictions to form a final prediction.

	"""


	def __init__( self, est: List[ ( str, object ) ], vot='hard' ) -> None:
		"""

			Purpose:
			--------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			-----------
			alpha (float): Regularization strength. Default is 1.0.
			solver (str): Solver to use. Default is 'auto'.

		"""
		super( ).__init__( )
		self.estimators: List[ ( str, object ) ]=est
		self.voting: str=vot
		self.voting_regressor: VotingRegressor=VotingRegressor( estimators=est, voting=vot )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			--------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'VotingRegression'
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
			exception.cause = 'VotingRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute the R-squared
			accuracy for the Ridge model.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

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
			exception.cause = 'VotingRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Evaluates the Ridge model
			using multiple metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'VotingRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot predicted vs
			actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'VotingRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StackClassification( Model ):
	"""

		Purpose:
		-------
		Stack of estimators with a final classifier. Stacked generalization consists in stacking the
		output of individual estimator and use a classifier to compute the final prediction.
		Stacking allows to use the strength of each individual estimator by using their output
		as input of a final estimator. Note that estimators_ are fitted on the full X while
		final_estimator_ is trained using cross-validated predictions of the base
		estimators using cross_val_predict.

	"""


	def __init__( self, est: List[ Tuple[ str, ClassifierMixin ] ],
	              final: Optional[ ClassifierMixin ]=None ) -> None:
		"""

			Initialize the RandomForestClassification.

		"""
		super( ).__init__( )
		self.estimators:  List[ Tuple[ str, ClassifierMixin ] ]=est
		self.final_estimator: ClassifierMixin =final
		self.stacking_classifier: StackingClassifier=StackingClassifier( estimators=est, final_estimator=final )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
				Fit the classifier.

			Parameters:
			----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Class labels.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'StackClassification'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Predict class labels
			using the SGD classifier.

			Parameters:
			-----------
			X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'StackClassification'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute R^2 accuracy
			for the SGDRegressor.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
				float: R^2 accuracy.

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
			exception.cause = 'StackClassification'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ]:
		"""

			Evaluate the Lasso model
			using multiple regression metrics.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'StackClassification'
			exception.method = ('analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, '
			                    'float ]')
			error = ErrorDialog( exception )
			error.show( )


	def create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Plot confusion matrix
			for classifier predictions.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): True class labels.

			Returns:
			-----------
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
			exception.cause = 'StackClassification'
			exception.method = 'create_matrix( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class StackRegression( Model ):
	"""

			Purpose:
			--------
			Stack of estimators with a final regressor. Stacked generalization consists in stacking
			the output of individual estimator and use a regressor to compute the final prediction.
			Stacking allows to use the strength of each individual estimator by using
			their output as input of a final estimator. Note that estimators_ are fitted on the
			full X while final_estimator_ is trained using cross-validated predictions of
			the base estimators using cross_val_predict.

	"""


	def __init__( self, estimators: List[ Tuple[ str, ClassifierMixin ] ],
	              final: ClassifierMixin=None ) -> None:
		"""

			Purpose:
			--------
			Initialize the RidgeRegressor linerar_model.

			Parameters:
			----------
			estimators - List[ Tuple[ str, ClassifierMixin ] ]:
			Base estimators which will be stacked together.
			Each element of the list is defined as a tuple of string (i.e. name) and an estimator
			instance. An estimator can be set to ‘drop’ using set_params. The type of estimator is
			generally expected to be a classifier. However, one can pass a regressor for some
			use case (e.g. ordinal regression).

			final - ClassifierMixin, default=None
			A classifier which will be used to combine the base estimators.

		"""
		super( ).__init__( )
		self.estimators: List[ Tuple[ str, ClassifierMixin ] ]=estimators
		self.final_estimator: ClassifierMixin=final
		self.stacking_regressor: StackingRegressor=StackingRegressor( estimators=estimators,
			final=final )
		self.prediction: np.array=None
		self.accuracy: float=0.0
		self.precision: float=0.0
		self.recall: float=0.0
		self.f1_score: float=0.0
		self.roc_auc_score: float=0.0
		self.correlation_coefficient: float=0.0
		self.median_absolute_error: float=0.0


	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""

			Purpose:
			---------
			Fit the RidgeRegressor regression linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Target vector.

			Returns:
			--------
				Pipeline

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
			exception.cause = 'StackRegression'
			exception.method = 'train( self, X: np.ndarray, y: np.ndarray ) -> Pipeline'
			error = ErrorDialog( exception )
			error.show( )


	def project( self, X: np.ndarray ) -> np.ndarray:
		"""

			Project target target_values
			using the RidgeRegressor linerar_model.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.

			Returns:
			-----------
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
			exception.cause = 'StackRegression'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def score( self, X: np.ndarray, y: np.ndarray ) -> float:
		"""

			Compute the R-squared
			accuracy for the Ridge model.

			Parameters:
			-----------
				X (np.ndarray): Test features.
				y (np.ndarray): Ground truth target_values.

			Returns:
			-----------
				float: R-squared accuracy.

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
			exception.cause = 'StackRegression'
			exception.method = 'accuracy( self, X: np.ndarray, y: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict:
		"""

			Evaluates the Ridge model
			using multiple metrics.

			Parameters:
			-----------
				X (pd.DataFrame): Feature matrix.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'StackRegression'
			exception.method = 'analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict'
			error = ErrorDialog( exception )
			error.show( )


	def create_graph( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Plot predicted vs
			actual target_values.

			Parameters:
			-----------
				X (np.ndarray): Input features.
				y (np.ndarray): Ground truth target target_values.

			Returns:
			-----------
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
			exception.cause = 'StackRegression'
			exception.method = 'create_graph( self, X: np.ndarray, y: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )
