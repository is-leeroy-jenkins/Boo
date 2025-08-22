'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Processors.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Processors.py" company="Terry D. Eppler">

     Mathy Processors

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
	Processors.py
</summary>
******************************************************************************************
'''
from Data import Metric
from Booger import Error, ErrorDialog
import numpy as np
from typing import Optional
import sklearn.preprocessing as sk
from sklearn.impute import SimpleImputer, KNNImputer


class StandardScaler( Metric ):
	"""

		Purpose:
		--------
		Standardizes features by removing the mean and scaling to unit variance.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.standard_scaler = sk.StandardScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object:
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
			self

		"""
		try:
			if X is None:
				raise Exception( 'Argument "X" is None' )
			else:
				self.standard_scaler.fit( X )
				return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('fit( self, X: np.ndarray, y: Optional[np.ndarray]=None ) -> '
			                    'Pipeline')
			error = ErrorDialog( exception )
			error.show( )


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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
		self.minmax_scaler = sk.MinMaxScaler( )

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
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
			self

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


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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
		self.robust_scaler = sk.RobustScaler( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
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
			self

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


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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

	def __init__( self, norm: str='l2' ) -> None:
		super( ).__init__( )
		self.normal_scaler = sk.Normalizer( norm=norm )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
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
			self

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


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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

	def __init__( self, unknown: str = 'ignore' ) -> None:
		super( ).__init__( )
		self.hot_encoder = sk.OneHotEncoder( sparse_output=False, handle_unknown=unknown )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
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
			self

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


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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
			Encodes categorical features as ordinal integers.

	"""

	def __init__( self ) -> None:
		super( ).__init__( )
		self.ordinal_encoder = sk.OrdinalEncoder( )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
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
			self

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

	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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
		Fills missing target_values using the average.

	"""

	def __init__( self, strat: str='mean' ) -> None:
		super( ).__init__( )
		self.mean_imputer = SimpleImputer( strategy=strat )


	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ):
		"""


			Purpose:
			---------
			Fits the simple_imputer to the df.

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


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
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

	def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ):
		"""

			Purpose:
			________
			Fits the simple_imputer to the df.

			Parameters:
			_____
			X (np.ndarray): Input df with missing target_values.
			y (Optional[np.ndarray]): Ignored.

			Returns:
			--------
			self

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


	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			_________

			Transforms the text df by imputing missing target_values.

			Parameters:
			-----------
			X (np.ndarray): Input df

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

