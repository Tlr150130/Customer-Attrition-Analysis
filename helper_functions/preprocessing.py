"""
Develop Preprocessing Functions: preprocessing.py
    Here are the helper functions for data preprocessing
    All classes are created in sklearn pipeline format and can be placed in one
"""
##### Libraries #######################################################
import pandas as pd
import numpy as np
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# extend pirectory to get helper functions
sys.path.insert(0, "./helper_functions")

from reporting import dtype_selector

# pandas config
pd.options.mode.chained_assignment = None # stops slice warning

##### Functions and Classes to be called ###########################################
class Preprocessing(BaseEstimator, TransformerMixin):
    """
    Prepare data for feature engineering step [Remove ID's, cap Age at 100]
    
    Parameters:
    ===========
    remove_id: Boolean 
    cap_age: Boolean
    
    Methods:
    ========
    fit: required to sklearn pipeline
    transform: remove ID's and cap age
    """
    
    def __init__(self, remove_id = True, cap_age = True, typecast = True):
        self.remove_id = remove_id
        self.cap_age = cap_age
        self.typecast = typecast
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):        
        # remove id
        if self.remove_id:
            X_copy = X.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
        
        # typecast
        if self.typecast:
            X_copy[["Age", "Tenure"]] = X[["Age", "Tenure"]].astype("float")
            
        # cap age at 100
        if self.cap_age:
            X_copy.loc[(X["Age"] > 100), "Age"] = 100.0
        
        return X_copy
    

class Imputer(BaseEstimator, TransformerMixin):
    """
    Impute missing data based on feature type: [categorical string, continuous
    numeric, or categorical integers]
    
    methods:
    ========
    fit: gets column names according to feature type and trains the imputing column
         transformer
    
    transform: imputes missing values in the features and transform data back into
               dataframe
    """
    def fit(self, X, y = None):
        ### get the column names
        self.cat_cols = X.select_dtypes(include = "O").columns # categorical strings
        self.num_cols = X.select_dtypes(include = "float").columns # continuous numerical
        self.int_cols = X.select_dtypes(include = "int64").columns # categorical ints
        
        ### fit imputer column transformer
        self.imputer = ColumnTransformer([
            # Categorical Features 
            ("cat", SimpleImputer(strategy = "most_frequent"), self.cat_cols),
            # Numerical Features
            ("num", SimpleImputer(strategy = "median"), self.num_cols),            
            # Numeric Binary Features
            ("num_bin", SimpleImputer(strategy = "most_frequent"), self.int_cols)
        ])
        
        self.imputer.fit(X)
        
        return self
    
    def transform(self, X, y = None):
        ### imputation
        imputed_features = self.imputer.transform(X)
        
        ### return to pandas dataframe
        # get column names
        cols = []
        for col in [list(self.cat_cols), list(self.num_cols), list(self.int_cols)]:
            cols.extend(col)
        
        # transform arrays inot dataframe
        imputed_features_df = pd.DataFrame(data = imputed_features, columns = cols)
        
        # typecast columns back to original data types (Object was default)
        imputed_features_df[self.num_cols] = imputed_features_df[self.num_cols].astype("float")
        imputed_features_df[self.int_cols] = imputed_features_df[self.int_cols].astype("int64")
        
        return imputed_features_df
    

class Feature_Transformation(BaseEstimator, TransformerMixin):
    """
    Transforms features based on datatype: [Categorical string features are 
    one-hot-encoded and Numeric features are standardized]
    
    Methods:
    ========
    fit: seperates features based on type and fits the column transformer
    
    transform: transforms the data input and converts the output array to 
               a dataframe
    """
    def fit(self, X, y = None):
        # get column names
        self.cat_cols = X.select_dtypes(include = "O").columns
        self.num_cols = X.select_dtypes(include = "float").columns
        
        # create column transformer to one-hot-encode and standardize the data
        self.one_hot_and_scale = ColumnTransformer([
                ("cat", OneHotEncoder(drop = "first"), self.cat_cols),
                ("num", StandardScaler(), self.num_cols)],
                remainder = "passthrough" # allows numeric binary columns through
        ) 
        
        # fit the column tran
        self.one_hot_and_scale.fit(X, y)
        
        return self
    
    def transform(self, X, y = None):        
        # one-hot and scale   
        transformed_features = self.one_hot_and_scale.transform(X)
        transformed_features_df = pd.DataFrame(
            data = transformed_features, 
            columns = self.one_hot_and_scale.get_feature_names_out()
        )
        
        return transformed_features_df
