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
def preprocess_pipeline(features): 
    """
    Uses a pre-fitted pipeline from 'pipeline.pkl' to process raw data.
    
    Parameters:
    ===========
    features: DataFrame
        raw data
        
    Returns:
    ========
    processed_features: DataFrame
    """
    try:
        PROJECT_ROOT_DIR = "."
        pipeline_path = os.path.join(PROJECT_ROOT_DIR, "model", "pipeline.pkl")
        feature_transformation_pipeline = joblib.load(pipeline_path)
    
    except:
        raise ValueError("There is no 'pipeline.pkl'. Please fit and \
                         save pipeline.")
    
    processed_features = feature_transformation_pipeline.transform(features)
    return processed_features
    
def model_predict(features, threshold = 0.5): 
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
        model_path = os.path.join(PROJECT_ROOT_DIR, "model", "Xgboost.pkl")
        model = joblib.load(model_path)
    
    except:
        raise ValueError("There is no 'best_model.pkl'. Please fit and \
                         save model.")
    
    # get probabilities
    pred_probs = model.predict_proba(features)
    
    # label decision based on threshold
    predictions = [int(i > threshold) for i in pred_probs[:, 1]]
    
    return predictions

def save_predictions(data):
    """
    Saves predictions to a csv file
    
    Parameters:
    ===========
    data: DataFrame
        holds prediction data
    """
    # get date and time for accurate labeling
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d--%H-%M-%S")
    
    # create file path
    PROJECT_ROOT_DIR = "."
    
    RESULTS_PATH = os.path.join(PROJECT_ROOT_DIR, "results")
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    file_path = os.path.join(RESULTS_PATH,  
                        "Test_Predictions_{}.csv".format(date_string))
    
    # save file
    data.to_csv(file_path)
    print("Saved to {}".format(file_path))
    

def deployment_pipeline(data, threshold = 0.5):
    """
    Pipeline that takes raw data, processes it, and generates predictions
    using the previously found best model. The best model can be updated
    by replacing the best model file.
    
    Parameters: 
    =========== 
    data: DataFrame
        raw data
        
    Returns:
    ========
    results: DataFrame [jobId, predicted_salary]
    """
    # copy features to preserve data integrety
    features = data.copy()
    
    # get pretrained pipeline
    processed_features = preprocess_pipeline(features)
    
    # get predictions
    predictions = model_predict(processed_features, threshold)
    
    # save predictions as a csv file
    #save_predictions(pd.DataFrame({"Predictions": predictions})) 
    
    # return predictions
    return predictions

def test_data_metrics(labels, predictions):
    """
    Report the accuracy, precision, and recall of the model
    """
    print('\n{0:*^80}'.format(' Test Data Metrics '))
    print("Accuracy:\t{}".format(accuracy_score(labels, predictions)))
    print("Precision:\t{}".format(precision_score(labels, predictions)))
    print("Recall:\t\t{}".format(recall_score(labels, predictions)))