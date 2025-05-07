'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                preppy.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="preppy.py" company="Terry D. Eppler">

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
from typing import Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    OneHotEncoder, OrdinalEncoder
)


class Preprocessor( ):
    """
    
        Base interface for all
        preprocessors. Provides standard `fit`, `transform`, and
        `fit_transform` methods.
    
    """
    def __init__( self ):
        pass

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
        self.fit( X, y )
        return self.transform( X )



class StandardScaler( Preprocessor ):
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
        _pipeline: Pipeline = self.standard_scaler.fit( X )
        return _pipeline


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
                return self.standard_scaler.transform( X )
        except Exception as e:
            exception = Error( e )
            exception.module = 'preppy'
            exception.cause = ''
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )


class MinMaxScaler( Preprocessor ):
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
        _pipeline: Pipeline = self.minmax_scaler.fit( X )
        return _pipeline

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
                return self.minmax_scaler.transform( X )
        except Exception as e:
            exception = Error( e )
            exception.module = 'preppy'
            exception.cause = ''
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )


class RobustScaler( Preprocessor ):
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
        _pipeline: Pipeline = self.robust_scaler.fit( X )
        return _pipeline


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
            exception.module = 'preppy'
            exception.cause = ''
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )


class Normalizer( Preprocessor ):
    """
    
        Scales input vectors individually to unit norm.
    
    """

    def __init__( self, norm: str='l2' ) -> None:
        super( ).__init__( )
        self.normal_scaler: Normalizer = Normalizer( norm=norm )


    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
        """
        
            Fits the normalizer
            (no-op for Normalizer).
    
            Args:
                X (np.ndarray): Input data.
                y (Optional[np.ndarray]): Ignored.
            
        """
        _pipeline: Pipeline = self.normal_scaler.fit( X )
        return _pipeline


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
            exception.module = 'preppy'
            exception.cause = ''
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )



class OneHotEncoder( Preprocessor ):
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
        _pipeline: Pipeline = self.hot_encoder.fit( X )
        return _pipeline


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
            exception.module = 'preppy'
            exception.cause = ''
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )



class OrdinalEncoder( Preprocessor ):
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
        _pipeline: Pipeline = self.ordinal_encoder.fit( X )
        return _pipeline


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
                _retval = self.ordinal_encoder.transform( X )
                return _retval
        except Exception as e:
            exception = Error( e )
            exception.module = 'preppy'
            exception.cause = ''
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )



class SimpleImputer( Preprocessor ):
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
        _pipeline: Pipeline = self.simple_imputer.fit( X )
        return _pipeline


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
                _retval: np.ndarray = self.simple_imputer.transform( X )
                return _retval
        except Exception as e:
            exception = Error( e )
            exception.module = 'preppy'
            exception.cause = ''
            exception.method = ''
            error = ErrorDialog( exception )
            error.show( )



class KnnImputer( Preprocessor ):
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
        _pipeline: Pipeline = self.knn_imputer.fit( X )
        return _pipeline


    def transform( self, X: np.ndarray ) -> np.ndarray:
        """
        
            Transforms the input
            data by imputing missing values.
    
            Args:
                X (np.ndarray): Input data
                with missing values.
    
            Returns:
                np.ndarray: Imputed data.
            
        """
        _retval: np.ndarray = self.knn_imputer.transform( X )
        return _retval


class MultiProcessor( Preprocessor ):
    """
    
        Chains multiple preprocessing
        steps into a pipeline.
    
    """

    def __init__( self, steps: List[ Tuple[ str, Preprocessor ] ] ) -> None:
        self.pipeline = Pipeline( steps )


    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Pipeline:
        """
        
            Fits all pipeline
            steps to the input data.
    
            Args:
                X (np.ndarray): Input feature matrix.
                y (Optional[np.ndarray]): Optional target array.
            
        """
        _pipeline: Pipeline = self.pipeline.fit( X, y )
        return _pipeline


    def transform( self, X: np.ndarray ) -> np.ndarray:
        """
        
            Applies all transformations
            in the pipeline to the input data.
    
            Args:
                X (np.ndarray): Input feature matrix.
    
            Returns:
                np.ndarray: Transformed feature matrix.
            
        """
        _retform: np.ndarray = self.pipeline.transform( X )
        return _retform


    def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
        """
        
            Fits and transforms all
            pipeline steps on the input data.
    
            Args:
                X (np.ndarray): Input feature matrix.
                y (Optional[np.ndarray]): Optional target array.
    
            Returns:
                np.ndarray: Transformed feature matrix.
            
        """
        _retval: np.ndarray = self.pipeline.fit_transform( X, y )
        return _retval

