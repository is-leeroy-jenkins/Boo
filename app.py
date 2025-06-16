'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                app.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="main.py" company="Terry D. Eppler">

         Boo is a df analysis tool integrating GenAI, Text Processing, and Machine-Learning
         algorithms for federal analysts.
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
    app.py
  </summary>
  ******************************************************************************************
'''
from src.boogr import Error, ErrorDialog
from dash import Dash, Input, Output, callback, dash_table, dcc, html
import mglearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
from openai import OpenAI
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.datasets import load_iris, make_classification, load_breast_cancer
from sklearn.manifold import Isomap, TSNE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis
from sklearn.linear_model import (LinearRegression, Ridge, RidgeClassifier, LogisticRegression,
                                  BayesianRidge, SGDRegressor, SGDClassifier, Perceptron,
                                  Lasso, ElasticNet, BayesianRidge, HuberRegressor)
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, \
	PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier,
                              ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              IsolationForest)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (
	classification_report, r2_score, mean_squared_error, accuracy_score, precision_recall_curve,
	average_precision_score, precision_score, recall_score, f1_score, PrecisionRecallDisplay, RocCurveDisplay,
	ConfusionMatrixDisplay
)

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

# Load the Excel file
file_path_balances = r'C:\Users\terry\source\repos\Boo\data\excel\Account Balances.xlsx'
df_balances = pd.read_excel( file_path_balances, sheet_name = 'Data' )
numeric_columns = [ 'CarryoverAuthority', 'CarryoverAdjustments', 'AnnualAppropriations',
                    'BorrowingAuthority', 'ContractAuthority', 'OffsettingReceipts',
                    'Obligations', 'Recoveries', 'UnobligatedBalance', 'Outlays', 'TotalResources' ]
numeric_subset = [ 'UnobligatedBalance', 'Obligations', 'Outlays' ]

# Filter target_values
all = [ 'AccountBalancesId', 'AgencyIdentifier', 'AgencyName', 'BeginningPeriodOfAvailability',
        'EndingPeriodOfAvailability', 'Availability', 'MainAccountCode', 'SubAccountCode',
        'TreasuryAccountSymbol', 'TreasuryAccountName', 'BudgetFunction', 'BudgetSubFunction',
        'FederalAccountSymbol', 'FederalAccountName', 'LastModified',
        'SubmissionPeriod' ] + numeric_columns
fields = [ 'AgencyName' ]
features = fields + numeric_subset

# Define datasets
df_fullset = df_balances[ all ].fillna( 0 )
df_subset = df_balances[ features ].fillna( 0 )
df_numeric = df_balances[ numeric_columns ].fillna( 0 )
agency_names = [ name.title( ) for name in df_subset[ 'AgencyName' ].unique( ) ]
df_agency = df_subset.groupby( by='AgencyName' ).sum( )
df_agency[ 'UnobligatedBalance'].replace( df_agency[ 'UnobligatedBalance']/1000000  )
df_agency[ 'Obligations'].replace( df_agency[ 'Obligations']/1000000 )
df_agency[ 'Outlays'].replace( df_agency[ 'Outlays']/1000000 )
df_agency = df_agency.round( 2 )



# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

app = Dash( )

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
app.layout = dash_table.DataTable( df_agency.to_dict('records'), [{"name": i, "id": i} for i in df_agency.columns] )

if __name__ == '__main__':
    app.run( debug=True )



