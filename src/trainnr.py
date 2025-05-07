from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDClassifier, SGDRegressor, Perceptron
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
from typing import Optional


class BaseModel( ):
    """
    
	    Abstract base class that defines the interface for all model wrappers.
    
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
	        
	        Fit the model to the training data.
	
	        Parameters:
	            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
	            y (np.ndarray): Target vector of shape (n_samples,).
	
	        Returns:
	            None
            
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
	        
	        Generate predictions from the trained model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
	
	        Returns:
	            np.ndarray: Predicted values or class labels.
            
        """
        raise NotImplementedError


class LinearRegression( BaseModel ):
    """
    
	    Wrapper for Ordinary Least Squares Regression.
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the Linear
	        Regression model.
	
	        Attributes:
	            model (LinearRegression): Internal OLS model using least squares.
	                Parameters:
	                    fit_intercept (bool): Whether to include an intercept term. Default is True.
	                    copy_X (bool): Whether to copy the feature matrix. Default is True.
                    
        """
        self.model = LinearRegression( )

    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
	        Fit the OLS
	        regression model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Target vector.
	
	        Returns:
	            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
        Predict target values
        using the OLS model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
            
        """
        return self.model.predict( X )


class RidgeRegression( BaseModel ):
    """
    
    Wrapper for RidgeRegression Regression (L2 regularization).
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        RidgeRegression model.
	
	        Attributes:
	            model (Ridge): Internal RidgeRegression regression model.
	                Parameters:
	                    alpha (float): Regularization strength. Default is 1.0.
	                    solver (str): Solver to use. Default is 'auto'.
                    
        """
        self.model = RidgeRegression( alpha=1.0 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
        Fit the RidgeRegression
        regression model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
        Predict target values
        using the RidgeRegression model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
            
        """
        return self.model.predict( X )


class LassoReression( BaseModel ):
    """
    
    Wrapper for LassoReression Regression (L1 regularization).
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        LassoReression model.
	
	        Attributes:
	            model (Lasso): Internal LassoReression regression model.
	                Parameters:
	                    alpha (float): Regularization strength. Default is 1.0.
	                    max_iter (int): Maximum number of iterations. Default is 1000.
                    
        """
        self.model = LassoReression( alpha=1.0 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
        Fit the LassoReression
        regression model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict target values
	        using the LassoReression model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted target values.
            
        """
        return self.model.predict( X )


class ElasticNetRegressor( BaseModel ):
    """
    
    Wrapper for ElasticNetRegressor Regression (L1 + L2 regularization).
    
    """

    def __init__( self ) -> None:
        """
	        
	        Initialize the
	        ElasticNetRegressor model.
	
	        Attributes:
	            model (ElasticNet): Internal ElasticNetRegressor regression model.
	                Parameters:
	                    alpha (float): Overall regularization strength. Default is 1.0.
	                    l1_ratio (float): Mixing parameter (0 = RidgeRegression, 1 = LassoReression). Default is 0.5.
                    
        """
        self.model = ElasticNetRegressor( alpha=1.0, l1_ratio=0.5 )

    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
	        Fit the ElasticNetRegressor
	        regression model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Target vector.
	
	        Returns:
	            None
            
        """
        self.model.fit( X, y )

    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict target values
	        using the ElasticNetRegressor model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted target values.
	            
        """
        return self.model.predict( X )


class LogisticRegressor( BaseModel ):
    """
    
    Wrapper for Logistic Regression.
    
    """

    def __init__( self ) -> None:
        """
        
        Initialize the Logistic
        Regression model.

        Attributes:
            model (LogisticRegression): Internal logistic regression classifier.
                Parameters:
                    max_iter (int): Maximum number of iterations. Default is 1000.
                    solver (str): Algorithm to use in optimization. Default is 'lbfgs'.
                    
        """
        self.model = LogisticRegression( max_iter=1000 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
        Fit the logistic
        regression model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target class labels.

        Returns:
            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
        Predict class labels using
        the logistic regression model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
            
        """
        return self.model.predict( X )


class BayesianRidge( BaseModel ):
    """
    
    Wrapper for Bayesian RidgeRegression Regression.
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        BayesianRidge model.
	
	        Attributes:
	            model (BayesianRidge): Internal probabilistic linear regression model.
	                Parameters:
	                    compute_score (bool): If True, compute marginal
	                    log-likelihood. Default is False.
                    
        """
        self.model = BayesianRidge( )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
        Fit the Bayesian RidgeRegression
        regression model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict target values
	        using the Bayesian model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted values.
            
        """
        return self.model.predict( X )


class SgdClassifier( BaseModel ):
    """
    
    Wrapper for SGD-based linear classifiers.
    
    """

    def __init__(self) -> None:
        """
        
	        Initialize the
	        SGDClassifier model.
	
	        Attributes:
	            model (SGDClassifier): Internal linear classifier trained via SGD.
	                Parameters:
	                    loss (str): Loss function to use. Default is 'log_loss'.
	                    max_iter (int): Maximum number of passes over the data. Default is 1000.
                    
        """
        self.model = SGDClassifier( loss='log_loss', max_iter=1000 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
	        
	        Fit the SGD
	        classifier model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Class labels.
	
	        Returns:
	            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
	        
		        Predict class labels
		        using the SGD classifier.
		
		        Parameters:
		            X (np.ndarray): Feature matrix.
		
		        Returns:
		            np.ndarray: Predicted class labels.
            
        """
        return self.model.predict( X )


class SgdRegressor( BaseModel ):
    """
    
    Wrapper for SGD-based linear regressors.
    
    """

    def __init__( self ) -> None:
        """
        
        
	        Initialize the
	        SGDRegressor model.
	
	        Attributes:
	            model (SGDRegressor): Internal linear regressor trained via SGD.
	                Parameters:
	                    penalty (str): Regularization term. Default is 'l2'.
	                    max_iter (int): Maximum number of passes. Default is 1000.
                    
        """
        self.model = SGDRegressor( max_iter=1000 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
	        
	        Fit the SGD regressor model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Target values.
	
	        Returns:
	            None
            
        """
        self.model.fit(X, y)


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
        Predict values using
        the SGD regressor model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted values.
            
        """
        return self.model.predict( X )


class Perceptron( BaseModel ):
    """
    
    Perceptron classifier.
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        Perceptron model.
	
	        Attributes:
	            model (Perceptron): Internal linear binary classifier.
	                Parameters:
	                    max_iter (int): Maximum number of iterations.
	                    Default is 1000.
                    
        """
        self.model = Perceptron( max_iter=1000 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
	        
	        Fit the
	        Perceptron model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Binary class labels.
	
	        Returns:
	            None
	            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
	        
	        Predict binary class
	        labels using the Perceptron.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted binary labels.
	            
        """
        return self.model.predict( X )


class NearestNeighborClassifier( BaseModel ):
    """
    
    Wrapper for k-Nearest Neighbors Classifier.
    
    """

    def __init__( self ) -> None:
        """
        
        Initialize the KNeighborsClassifier model.

        Attributes:
            model (KNeighborsClassifier): Internal non-parametric classifier.
                Parameters:
                    n_neighbors (int): Number of neighbors to use. Default is 5.
                    
        """
        self.model = KNeighborsClassifier( n_neighbors=5 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
        
	        Fit the KNN
	        classifier model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Class labels.
	
	        Returns:
	            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
	        
	        Predict class labels
	        using the KNN classifier.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted class labels.
            
        """
        return self.model.predict( X )


class NearestNeighborRegressor( BaseModel ):
    """
    
    Wrapper for k-Nearest Neighbors Regressor.
    
    """

    def __init__( self ) -> None:
        """
        
	        Initialize the
	        KNeighborsRegressor model.
	
	        Attributes:
	            model (KNeighborsRegressor): Internal non-parametric regressor.
	                Parameters:
	                    n_neighbors (int): Number of neighbors to use. Default is 5.
                    
        """
        self.model = KNeighborsRegressor( n_neighbors=5 )


    def fit( self, X: np.ndarray, y: np.ndarray ) -> None:
        """
	        
	        Fit the KNN
	        regressor model.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	            y (np.ndarray): Target values.
	
	        Returns:
	            None
            
        """
        self.model.fit( X, y )


    def predict( self, X: np.ndarray ) -> np.ndarray:
        """
        
	        Predict values using
	        the KNN regressor.
	
	        Parameters:
	            X (np.ndarray): Feature matrix.
	
	        Returns:
	            np.ndarray: Predicted values.
            
        """
        return self.model.predict( X )
