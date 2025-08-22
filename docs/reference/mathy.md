# mathy



## 🧠 Overview

**Mathy** is a library of clustering algorithms, preprocessing tools, 
and classification/regression models with custom wrappers, error handling, and diagnostic tools for 
machine-learning workflows. Mathy targets professionals in data science and analytics who require reusable, robust, and 
interpretable tools for building intelligent models—especially within government and research domains.
___
## 🧰 Core Modules
- **`data.py`**: Abstract base classes (`Model`, `Metric`) and the `Dataset` class for loading and splitting data.
- **`models.py`**: Wrappers for common classifiers/regressors such as:
- Perceptron, Ridge, SGD, MLP (Multilayer Perceptron)
- Decision Trees, KNN, Logistic Regression, SVMs, Ensemble Methods
- **`clusters.py`**: Unified interface for clustering techniques including:
- `KMeans`, `DBSCAN`, `Agglomerative`, and other sklearn-based clustering algorithms
- **`processors.py`**: Scalers and transformers for preprocessing:
- `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder`

## 🧠 Machine Learning
- Supports both classification and regression models.
- Model analysis using:
- Accuracy, Precision, Recall, F1 Score, ROC AUC, R², MAE, MSE, RMSE, etc.
- Seamless training, scoring, prediction (`train`, `score`, `project`, `analyze`)

## 📊 Clustering & Visualization
- Integrated support for:
  - KMeans: Centroid-based partitioning
  - DBSCAN: Density-based clustering
  - Agglomerative: Hierarchical linkage models
- 2D cluster visualizations using Matplotlib

## 🛠️ Data Preprocessing
- Scaling and normalization strategies
- Missing value imputation
- Transformation pipelines for numerical and categorical data

## 📁 Project Structure

```plaintext
mathy/
├── data.py           # Dataset handler and model interface
├── regressors.py     # Encapsulated regression models
├── classifiers.py    # Encapsulated classification models
├── clusters.py       # Clustering models (KMeans, DBSCAN, etc.)
├── preprocessors.py  # Data preprocessing tools (scalers, encoders)
├── static/           # Static constants, helpers
├── README.md         # This file
```



### 📊 Classification Models
- Classification models are used to assign data samples into discrete categories or classes. These
models are foundational for tasks such as spam detection, disease diagnosis, and image recognition.
Below is a list of supported classifiers in Mathy:
- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/classifications.py)
___
| Class Name                  | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| Model                       | Base wrapper for all classification models.                  |
| PerceptronClassifier        | Linear classifier using the perceptron learning rule.        |
| MultilayerClassifier        | Multi-layer perceptron (MLP) for non-linear classification.  |
| RidgeClassifier             | Classifier with L2 regularization to prevent overfitting.    |
| StochasticDescentClassifier | Learns model using stochastic gradient descent.         |
| NearestNeighborClassifier   | Instance-based classifier using nearest neighbors.         |
| DecisionTreeClassifier      | Splits data into decision paths using feature thresholds.    |
| RandomForestClassifier      | Ensemble of decision trees trained with bagging.             |
| GradientBoostingClassifier  | Sequential ensemble reducing bias with boosting.          |
| AdaBoostClassifier          | Boosts weak learners to correct classification errors.       |
| BaggingClassifier           | Aggregates predictions from multiple bootstrapped models.    |
| VotingClassifier            | Combines multiple models through majority voting.            |
| StackClassifier             | Meta-learner trained on outputs of base classifiers.         |
| SupportVectorClassifier     | Support Vector Machine (SVC) classifier.                     |





### 📈 Regression Models
- Regression models predict continuous numerical outcomes and are crucial in applications like
forecasting, pricing, and trend analysis. Mathy provides a range of linear and non-linear regression
models, listed below:
- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/regressions.py)

___
| Class Name                 | Description                                                        |
|----------------------------|--------------------------------------------------------------------|
| Model                      | Base model interface for regression learners.                      |
| MultilayerRegressor        | Multi-layer neural network for regression.                         |
| LinearRegressor            | Ordinary least squares regression.                                |
| RidgeRegressor             | Linear regression with L2 regularization.                         |
| LassoRegressor             | Linear regression with L1 penalty for sparsity.                   |
| ElasticNetRegressor        | Combines L1 and L2 penalties for robustness.                      |
| LogisticRegressor          | Logistic regression for binary outcomes.                          |
| BayesianRidgeRegressor     | Bayesian linear model with priors on coefficients.               |
| StochasticDescentRegressor | Optimizes regression with stochastic gradient descent.            |
| NearestNeighborRegressor   | Instance-based regression using k-nearest neighbors.              |
| DecisionTreeRegressor      | Tree-based model for continuous targets.                          |
| RandomForestRegressor      | Ensemble of trees trained on bootstrapped samples.                |
| GradientBoostingRegressor  | Boosting technique for improved predictive accuracy.              |
| AdaBoostRegressor          | Adaptive boosting for regression tasks.                           |
| BaggingRegressor           | Bagging ensemble to reduce variance.                             |
| VotingRegressor            | Aggregates predictions from multiple regressors.                  |
| StackRegressor             | Trains meta-regressor on top of base models.                      |
| SupportVectorRegressor      | Support Vector Regression (SVR) for high-dimensional data.        |




### 🔍 Clustering Models
- Clustering is an unsupervised technique used to discover natural groupings in data without labeled
outcomes. Mathy supports a variety of clustering algorithms suitable for both spherical and
irregular cluster shapes:
- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/clusters.py)

___
| Class Name                 | Description                                                     |
|----------------------------|-----------------------------------------------------------------|
| Cluster                    | Abstract base for clustering methods.                           |
| KMeansCluster              | Clusters data into k partitions via centroid minimization.      |
| DbscanCluster              | Density-based clustering that handles noise and outliers.       |
| AgglomerativeCluster       | Hierarchical clustering by iterative merging.                   |
| SpectralCluster            | Uses spectral decomposition for clustering.                     |
| MeanShiftCluster           | Clusters by finding dense regions (modes) in feature space.     |
| AffinityPropagationCluster | Message-passing clustering based on exemplar similarity.        |
| BirchCluster               | Clusters large datasets using hierarchical CF trees.            |
| OpticsCluster              | Orders points to extract density-based clusters.                |




### 📁 Data
- Encapsulates datasets and implements dimensionality reduction, correlation analysis, and feature
selection.
- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/data.py)

___
  | Class Name | Description |
  |----------------------|-------------------------------------------------------------------|
  | Metric | Base class for transformations and evaluation metrics. |
  | VarianceThreshold | Removes low-variance features from the dataset. |
  | CorrelationAnalysis | Analyzes relationships using Canonical Correlation Analysis. |
  | ComponentAnalysis | Performs PCA or similar for dimensionality reduction. |
  | Dataset | Encapsulates data loading, transformation, and partitioning. |





### 📁 Preprocessing

- This module provides a unified collection of data preprocessing tools for scaling, encoding, and
imputing. All classes extend the Metric interface, ensuring consistency in method signatures and
usage across pipelines.
- [Code](https://github.com/is-leeroy-jenkins/Mathy/blob/main/data.py)

___

| Class Name               | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Metric                   | Abstract base class for all preprocessors with methods like `fit`, `transform`, and `fit_transform`. |
| LabelBinarizer           | Binarize labels in a one-vs-all fashion.                                  |            
| TfidfTransformer         | Converts a count matrix to a normalized TF-IDF representation.              |
| TfidfVectorizer          | Converts raw documents to TF-IDF features using vocabulary learning.        |
| CountVectorizer          | Converts a collection of text documents to a matrix of token counts.        |
| HashingVectorizer        | Uses hashing trick to convert text documents into numerical feature vectors. |
| StandardScaler           | Standardizes features by removing the mean and scaling to unit variance.   |
| MinMaxScaler             | Transforms features by scaling each to a given range, usually [0, 1].       |
| RobustScaler             | Scales features using statistics robust to outliers like the median and IQR.|
| NormalScaler             | Normalizes each sample to unit norm (L1 or L2).                             |
| OneHotEncoder            | Encodes categorical features as a one-hot numeric array.                    |
| OrdinalEncoder           | Encodes categorical features as integer values based on their order.        |
| LabelEncoder             | Encodes target labels with integer values.                                  |
| PolynomialFeatures       | Generates polynomial and interaction features for a dataset.                |
| MeanImputer              | Fills missing values in a dataset using the column mean.                    |
| NearestNeighborImputer   | Imputes missing values using the nearest neighbor algorithm.                |
| IterativeImputer         | Imputes missing values by modeling them as a function of other features.    |
| SimpleImputer            | Provides basic strategies for imputing missing values (mean, median, etc.). |


## 📦 Dependencies

###### Mathy requires Python 3.9+ and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `pydantic`
- `boogr` (custom error handler module)


- To install all dependencies:

```
bash
pip install -r requirements.txt
```

## ⚡ Quickstart

#### 1. **Load and Split Data**  
- Load a dataset and define the target column.

```
python
from data import Dataset
import pandas as pd

df = pd.read_csv("data.csv")
dataset = Dataset(df, target="Label")
```

#### 2. **Preprocessing**

``` 
from processors import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(dataset.training_data)
y_train = dataset.training_values
```

#### 3. **Train and Evaluate Model**

``` 
from models import PerceptronClassifier
model = PerceptronClassifier()
model.train(X_train, y_train)
print(model.score(X_train, y_train))
print(model.analyze(X_train, y_train))
```

## 📝 License

Mathy is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Mathy/blob/main/LICENSE.txt).

