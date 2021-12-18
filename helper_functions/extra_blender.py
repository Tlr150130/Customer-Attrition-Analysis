"""
Model Deployment Functions: model_eval.py
    Here are the helper functions for model evaluation
"""
##### Libraries #######################################################
import pandas as pd
import numpy as np
import sys
import os
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score

# extend pirectory to get helper functions
sys.path.insert(0, "./helper_functions")
sys.path.insert(0, "./model")

# pandas config
pd.options.mode.chained_assignment = None # stops slice warning

##### Functions and Classes to be called ###########################################
    
def get_models(): 
    """
    Uses a pre-trained model from 'Xgboost.pkl' to predict labels 
    based on inputed features.
    
    Parameters:
    ===========
    features: DataFrame
        processed features
        
    threshold: float
        cuttoff for labelprobabilities
        
    Returns:
    ========
    predictions: Array, shape(n_samples,)
        predicted salary
    """
    try:
        PROJECT_ROOT_DIR = "."
        FILE_NAME = "model_list.pkl"
        model_path = os.path.join(PROJECT_ROOT_DIR, "model", FILE_NAME)
        model = joblib.load(model_path)
    
    except:
        raise ValueError("There is no '{}'. Please fit and save model.").format(FILE_NAME)
    
    return model




