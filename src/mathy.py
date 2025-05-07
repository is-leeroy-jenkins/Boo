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
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDClassifier, SGDRegressor, Perceptron
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
from typing import Optional


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


class LogisticRegressor( BaseModel ):
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


class SgdClassifier( BaseModel ):
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


class SgdRegressor( BaseModel ):
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


class NearestNeighborClassifier( BaseModel ):
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


class NearestNeighborRegressor( BaseModel ):
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
