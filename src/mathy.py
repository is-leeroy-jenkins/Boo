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
from typing import Optional
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


class BaseModel( ):
    """
    
	    Abstract base class
	    that defines the interface for all linerar_model wrappers.
    
    """

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


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
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


    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
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


class Preprocessor( ):
    """

        Base interface for all
        preprocessors. Provides standard `fit`, `transform`, and
        `fit_transform` methods.

    """
    
    
    def __init__( self ):
        pass
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
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
    
    
    def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
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
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
    
    
    def __init__( self, norm: str = 'l2' ) -> None:
        super( ).__init__( )
        self.normal_scaler: Normalizer = Normalizer( norm=norm )
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
    
    
    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Pipeline:
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
        _retval: np.ndarray = self.pipeline.fit_transform( X, y )
        return _retval


class LinearRegression( BaseModel ):
    """
    
	    Ordinary Least Squares Regression.
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the Linear
	        Regression linerar_model.
	
	        Attributes:
	            linerar_model (LinearRegression): Internal OLS linerar_model using least squares.
	                Parameters:
	                    fit_intercept (bool): Whether to include an intercept term. Default is True.
	                    copy_X (bool): Whether to copy the feature matrix. Default is True.
                    
        """
        self.linerar_model = LinearRegression( )


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
        self.linerar_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
        Predict target values
        using the OLS linerar_model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
            
        """
        return self.linerar_model.predict( X )


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
        return r2_score( y, self.linerar_model.predict( X ) )


    def evaluate(self, X: np.ndarray, y: np.ndarray ) -> dict:
        """
        
            Evaluate the model using
            multiple regression metrics.
    
            Parameters:
                X (np.ndarray): Feature matrix.
                y (np.ndarray): Ground truth values.
    
            Returns:
                dict: Dictionary of MAE, MSE, RMSE, R², etc.
            
        """
        y_pred = self.linerar_model.predict( X )
        return {
            'MAE': mean_absolute_error( y, y_pred ),
            'MSE': mean_squared_error( y, y_pred ),
            'RMSE': mean_squared_error( y, y_pred, squared=False ),
            'R2': r2_score( y, y_pred ),
            'Explained Variance': explained_variance_score( y, y_pred ),
            'Median Absolute Error': median_absolute_error( y, y_pred )
        }


    def plot(self, X: np.ndarray, y: np.ndarray) -> None:
        """
            
            Plot actual vs predicted values.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): True target values.
            
        """
        y_pred = self.predict(X)
        plt.scatter(y, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("OLS: Actual vs Predicted")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        plt.grid(True)
        plt.show()


class RidgeRegression( BaseModel ):
    """
        
        RidgeRegression Regression
        (L2 regularization).
        
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        RidgeRegression linerar_model.
	
	        Attributes:
	            linerar_model (Ridge): Internal RidgeRegression regression linerar_model.
	                Parameters:
	                    alpha (float): Regularization strength. Default is 1.0.
	                    solver (str): Solver to use. Default is 'auto'.
                    
        """
        self.ridge_model = RidgeRegression( alpha=1.0 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
            
            Fit the RidgeRegression
            regression linerar_model.
    
            Parameters:
                X (np.ndarray): Feature matrix.
                y (np.ndarray): Target vector.
    
            Returns:
                None
                
        """
        self.ridge_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
            
            Predict target values
            using the RidgeRegression linerar_model.
    
            Parameters:
                X (np.ndarray): Feature matrix.
    
            Returns:
                np.ndarray: Predicted target values.
            
        """
        return self.ridge_model.predict( X )


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
        return r2_score( y, self.ridge_model.predict( X ) )


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
        y_pred = self.predict(X)
        return {
            'MAE': mean_absolute_error( y, y_pred ),
            'MSE': mean_squared_error( y, y_pred ),
            'RMSE': mean_squared_error( y, y_pred, squared=False ),
            'R2': r2_score( y, y_pred ),
            'Explained Variance': explained_variance_score( y, y_pred ),
            'Median Absolute Error': median_absolute_error( y, y_pred )
        }


    def plot(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        
            Plot predicted vs
            actual values.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): Ground truth target values.
    
            Returns:
                None
            
        """
        y_pred = self.predict(X)
        plt.scatter(y, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Ridge: Actual vs Predicted")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        plt.grid(True)
        plt.show()


class LassoReression( BaseModel ):
    """
        
        Wrapper for LassoReression Regression (L1 regularization).
        
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        LassoReression linerar_model.
	
	        Attributes:
	            linerar_model (Lasso): Internal LassoReression regression linerar_model.
	                Parameters:
	                    alpha (float): Regularization strength. Default is 1.0.
	                    max_iter (int): Maximum number of iterations. Default is 1000.
                    
        """
        self.lasso_model = LassoReression( alpha=1.0 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
        Fit the LassoReression
        regression linerar_model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            None
            
        """
        self.lasso_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict target values
	        using the LassoReression linerar_model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted target values.
            
        """
        return self.lasso_model.predict( X )
    
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
            Compute R^2 score
            for the Lasso model.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): Ground truth values.
    
            Returns:
                float: R^2 score.
        """
        return r2_score(y, self.predict(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        
            Evaluate the Lasso model
            using multiple regression metrics.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): Ground truth target values.
    
            Returns:
                dict: Dictionary of MAE, RMSE, R², etc.
            
        """
        y_pred = self.predict(X)
        return {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': mean_squared_error(y, y_pred, squared=False),
            'R2': r2_score(y, y_pred),
            'Explained Variance': explained_variance_score(y, y_pred),
            'Median Absolute Error': median_absolute_error(y, y_pred)
        }

    def plot(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        
            Plot actual vs.
            predicted values.
    
            Parameters:
                X (np.ndarray): Input feature matrix.
                y (np.ndarray): Ground truth values.
                
        """
        y_pred = self.predict(X)
        plt.scatter(y, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Lasso: Actual vs Predicted")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.grid(True)
        plt.show()


class ElasticNetRegressor( BaseModel ):
    """
    
    Wrapper for ElasticNetRegressor Regression (L1 + L2 regularization).
    
    """

    def __init__( self ) -> None:
        """
	        
	        Initialize the
	        ElasticNetRegressor linerar_model.
	
	        Attributes:
	            linerar_model (ElasticNet): Internal ElasticNetRegressor regression linerar_model.
	                Parameters:
	                    alpha (float): Overall regularization strength. Default is 1.0.
	                    l1_ratio (float): Mixing parameter (0 = RidgeRegression, 1 = LassoReression). Default is 0.5.
                    
        """
        self.elasticnet_model = ElasticNetRegressor( alpha=1.0, l1_ratio=0.5 )


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
        self.elasticnet_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict target values
	        using the ElasticNetRegressor linerar_model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted target values.
	            
        """
        return self.elasticnet_model.predict( X )


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        
            Compute R^2 score
            on the test set.
    
            Parameters:
                X (np.ndarray): Test features.
                y (np.ndarray): Ground truth target values.
    
            Returns:
                float: R^2 score.
                
        """
        return r2_score(y, self.elasticnet_model.predict(X))


    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        
            Evaluate model performance
            using regression metrics.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): Ground truth values.
    
            Returns:
                dict: Evaluation metrics.
            
        """
        y_pred = self.elasticnet_model.predict(X)
        return {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': mean_squared_error(y, y_pred, squared=False),
            'R2': r2_score(y, y_pred),
            'Explained Variance': explained_variance_score(y, y_pred),
            'Median Absolute Error': median_absolute_error(y, y_pred)
        }


    def plot(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        
            Plot actual vs. predicted
            regression output.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): True target values.
                
        """
        y_pred = self.predict(X)
        plt.scatter(y, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("ElasticNet: Actual vs Predicted")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.grid(True)
        plt.show()



class LogisticRegression( BaseModel ):
    """
    
    Wrapper for Logistic Regression.
    
    """

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
        self.logistic_model = LogisticRegression( max_iter=1000 )


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
        self.logistic_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
        Predict class labels using
        the logistic regression linerar_model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
            
        """
        return self.logistic_model.predict( X )


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
        return accuracy_score( y, self.logistic_model.predict( X ) )


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
        y_pred = self.predict(X)
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average='binary'),
            "Recall": recall_score(y, y_pred, average='binary'),
            "F1 Score": f1_score(y, y_pred, average='binary'),
            "ROC AUC": roc_auc_score(y, y_pred),
            "Matthews Corrcoef": matthews_corrcoef(y, y_pred),
            "Confusion Matrix": confusion_matrix(y, y_pred).tolist()
        }


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
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title("Logistic Regression Confusion Matrix")
        plt.grid(False)
        plt.show()



class BayesianRidge( BaseModel ):
    """
    
    Wrapper for Bayesian RidgeRegression Regression.
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        BayesianRidge linerar_model.
	
	        Attributes:
	            linerar_model (BayesianRidge): Internal probabilistic linear regression linerar_model.
	                Parameters:
	                    compute_score (bool): If True, compute marginal
	                    log-likelihood. Default is False.
                    
        """
        self.bayesian_model = BayesianRidge( )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
        Fit the Bayesian RidgeRegression
        regression linerar_model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            None
            
        """
        self.bayesian_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict target values
	        using the Bayesian linerar_model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted values.
            
        """
        return self.bayesian_model.predict( X )


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
        return r2_score(y, self.bayesian_model.predict( X ) )


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
        y_pred = self.predict(X)
        return {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': mean_squared_error(y, y_pred, squared=False),
            'R2': r2_score(y, y_pred),
            'Explained Variance': explained_variance_score(y, y_pred),
            'Median Absolute Error': median_absolute_error(y, y_pred)
        }


    def plot( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
            Plot predicted vs.
            actual values.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): True target values.
                
        """
        y_pred = self.predict(X)
        plt.scatter(y, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Bayesian Ridge: Actual vs Predicted")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.grid(True)
        plt.show()


class SgdClassification( BaseModel ):
    """
    
        SGD-based linear classifiers.
    
    """

    def __init__(self) -> None:
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
        self.sgd_classification_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
	        
		        Predict class labels
		        using the SGD classifier.
		
		        Parameters:
		            X (np.ndarray): Feature matrix.
		
		        Returns:
		            np.ndarray: Predicted class labels.
            
        """
        return self.sgd_classification_model.predict( X )


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
        return r2_score( y, self.sgd_classification_model.predict( X ) )


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
        y_pred = self.predict(X)
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average='binary'),
            "Recall": recall_score(y, y_pred, average='binary'),
            "F1 Score": f1_score(y, y_pred, average='binary'),
            "ROC AUC": roc_auc_score(y, y_pred),
            "Matthews Corrcoef": matthews_corrcoef(y, y_pred),
            "Confusion Matrix": confusion_matrix(y, y_pred).tolist()
        }


class SgdRegression( BaseModel ):
    """
    
    Wrapper for SGD-based linear regressors.
    
    """

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


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
	        
	        Fit the SGD regressor linerar_model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Target values.
	
	        Returns:
	            None
            
        """
        self.sgd_regression_model.fit(X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
        Predict values using
        the SGD regressor linerar_model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted values.
            
        """
        return self.sgd_regression_model.predict( X )


    def evaluate( self, X: np.ndarray, y: np.ndarray ) -> dict:
        """
            
            Evaluate regression model
            performance.
    
            Parameters:
                X (np.ndarray): Input features.
                y (np.ndarray): True target values.
    
            Returns:
                dict: Evaluation metrics dictionary.
            
        """
        y_pred = self.predict(X)
        return {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': mean_squared_error(y, y_pred, squared=False),
            'R2': r2_score(y, y_pred),
            'Explained Variance': explained_variance_score(y, y_pred),
            'Median Absolute Error': median_absolute_error(y, y_pred)
        }



class Perceptron( BaseModel ):
    """
    
    Perceptron classifier.
    
    """

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
        self.perceptron_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
	        
	        Predict binary class
	        labels using the Perceptron.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted binary labels.
	            
        """
        return self.perceptron_model.predict( X )


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
        return accuracy_score( y, self.perceptron_model.predict( X ) )


    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
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
        y_pred = self.predict(X)
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average='binary'),
            "Recall": recall_score(y, y_pred, average='binary'),
            "F1 Score": f1_score(y, y_pred, average='binary'),
            "ROC AUC": roc_auc_score(y, y_pred),
            "Matthews Corrcoef": matthews_corrcoef(y, y_pred),
            "Confusion Matrix": confusion_matrix(y, y_pred).tolist()
        }


class KnnClassification( BaseModel ):
    """
    
    Wrapper for k-Nearest Neighbors Classifier.
    
    """

    def __init__( self ) -> None:
        """
        
        Initialize the KNeighborsClassifier linerar_model.

        Attributes:
            linerar_model (KNeighborsClassifier): Internal non-parametric classifier.
                Parameters:
                    n_neighbors (int): Number of neighbors to use. Default is 5.
                    
        """
        self.knn_classification_model = KNeighborsClassifier( n_neighbors=5 )


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
        self.knn_classification_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
	        
	        Predict class labels
	        using the KNN classifier.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted class labels.
            
        """
        return self.knn_classification_model.predict( X )


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
        return accuracy_score( y, self.knn_classification_model.predict( X ))


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
        y_pred = self.predict(X)
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average='binary'),
            "Recall": recall_score(y, y_pred, average='binary'),
            "F1 Score": f1_score(y, y_pred, average='binary'),
            "ROC AUC": roc_auc_score(y, y_pred),
            "Matthews Corrcoef": matthews_corrcoef(y, y_pred),
            "Confusion Matrix": confusion_matrix(y, y_pred).tolist()
        }

class KnnRegression( BaseModel ):
    """
    
    Wrapper for k-Nearest Neighbors Regressor.
    
    """

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
        self.knn_regression_model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict values using
	        the KNN regressor.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted values.
            
        """
        return self.knn_regression_model.predict( X )


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
        return r2_score( y, self.knn_regression_model,predict( X ) )


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
        y_pred = self.predict(X)
        return {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': mean_squared_error(y, y_pred, squared=False),
            'R2': r2_score(y, y_pred),
            'Explained Variance': explained_variance_score(y, y_pred),
            'Median Absolute Error': median_absolute_error(y, y_pred)
        }
