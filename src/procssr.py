# preprocessing_framework.py

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from typing import Optional, Union, List, Tuple
import numpy as np


class Preprocessor:
    """
    Base interface for all preprocessors. Provides standard `fit`, `transform`, and
    `fit_transform` methods.
    """

    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
        """
        Fits the preprocessor to the input data.

        Args:
            X (np.ndarray): Feature matrix.
            y (Optional[np.ndarray]): Optional target array.
        """
        raise NotImplementedError

    def transform( self, X: np.ndarray ) -> np.ndarray:
        """
        Transforms the input data using the fitted preprocessor.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Transformed feature matrix.
        """
        raise NotImplementedError

    def fit_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
        """
        Fits the preprocessor and then transforms the input data.

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
    
    Standardizes features by removing the mean and scaling to unit variance.
    
    """

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()

    def fit( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> None:
        """
        Fits the scaler to the data.

        Args:
            X (np.ndarray): Input data.
            y (Optional[np.ndarray]): Ignored.
        """
        self.scaler.fit( X )

    def transform( self, X: np.ndarray ) -> np.ndarray:
        """
        Transforms the data using the fitted StandardScaler.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled data.
        """
        return self.scaler.transform( X )


class MinMaxScaler(Preprocessor ):
    """Scales features to a given range (default is [0, 1])."""

    def __init__(self) -> None:
        # Holds the internal MinMaxScaler for min-max normalization
        self.scaler: MinMaxScaler = MinMaxScaler()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fits the scaler to the data.

        Args:
            X (np.ndarray): Input data.
            y (Optional[np.ndarray]): Ignored.
        """
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the fitted MinMaxScaler.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled data.
        """
        return self.scaler.transform(X)


class RobustScaler(Preprocessor ):
    """
    
        Scales features using statistics
        that are robust to outliers.
    
    """

    def __init__(self) -> None:
        # Holds the internal RobustScaler for IQR-based scaling
        self.scaler: RobustScaler = RobustScaler()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        
                Fits the scaler
                to the data.
        
                Args:
                    X (np.ndarray): Input data.
                    y (Optional[np.ndarray]): Ignored.
                    
        """
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the fitted RobustScaler.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled data.
        """
        return self.scaler.transform(X)


class Normalizer(Preprocessor ):
    """Scales input vectors individually to unit norm."""

    def __init__(self, norm: str = "l2") -> None:
        self.scaler: Normalizer = Normalizer(norm=norm)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fits the normalizer (no-op for Normalizer).

        Args:
            X (np.ndarray): Input data.
            y (Optional[np.ndarray]): Ignored.
        """
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies normalization to each sample.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Normalized data.
        """
        return self.scaler.transform(X)



class OneHotEncoder(Preprocessor ):
    """Encodes categorical features as a one-hot numeric array."""

    def __init__(self, handle_unknown: str = 'ignore') -> None:
        # Holds the internal OneHotEncoder instance
        self.encoder: OneHotEncoder = OneHotEncoder(sparse=False, handle_unknown=handle_unknown)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fits the encoder to the categorical data.

        Args:
            X (np.ndarray): Categorical input data.
            y (Optional[np.ndarray]): Ignored.
        """
        self.encoder.fit(X)


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data into a one-hot encoded format.

        Args:
            X (np.ndarray): Categorical input data.

        Returns:
            np.ndarray: One-hot encoded matrix.
        """
        return self.encoder.transform(X)


class OrdinalEncoder(Preprocessor ):
    """Encodes categorical features as ordinal integers."""

    def __init__(self) -> None:
        # Holds the internal OrdinalEncoder instance
        self.encoder = OrdinalEncoder()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fits the encoder to the categorical data.

        Args:
            X (np.ndarray): Categorical input data.
            y (Optional[np.ndarray]): Ignored.
        """
        self.encoder.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data into ordinal-encoded format.

        Args:
            X (np.ndarray): Categorical input data.

        Returns:
            np.ndarray: Ordinal-encoded matrix.
        """
        return self.encoder.transform(X)



class SimpleImputer(Preprocessor ):
    """Fills missing values using a specified strategy."""

    def __init__(self, strategy: str = 'mean') -> None:
        # Holds the internal SimpleImputer instance
        self.imputer: SimpleImputer = SimpleImputer(strategy=strategy)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fits the imputer to the data.

        Args:
            X (np.ndarray): Input data with missing values.
            y (Optional[np.ndarray]): Ignored.
        """
        self.imputer.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data by filling in missing values.

        Args:
            X (np.ndarray): Input data with missing values.

        Returns:
            np.ndarray: Imputed data.
        """
        return self.imputer.transform(X)


class KNNImputer(Preprocessor ):
    """Fills missing values using k-nearest neighbors."""

    def __init__(self) -> None:
        # Holds the internal KNNImputer instance
        self.imputer: KNNImputer = KNNImputer()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fits the imputer to the data.

        Args:
            X (np.ndarray): Input data with missing values.
            y (Optional[np.ndarray]): Ignored.
        """
        self.imputer.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data by imputing missing values.

        Args:
            X (np.ndarray): Input data with missing values.

        Returns:
            np.ndarray: Imputed data.
        """
        return self.imputer.transform(X)



