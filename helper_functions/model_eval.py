"""
Develop Model Evaluation Functions: model_eval.py
    Here are the helper functions for model evaluation
"""
##### Libraries #######################################################
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from statistics import mean

# extend pirectory to get helper functions
sys.path.insert(0, "./helper_functions")

from reporting import dtype_selector

# pandas config
pd.options.mode.chained_assignment = None # stops slice warning

##### Functions and Classes to be called ###########################################
def cv_mse_stats(name, model, X, y):
    """
    Calculates mean accuracy, percision, and recall
    
    Parameters:
    ===========
    name: String name of the trial
    model: Sklearn algorithm 
    X: Dataframe or Array of features
    y: Dataframe or Array of target values
    
    Returns:
    ========
    scoring: array[name, mean(accuracy), mean(percision), mean(recall)]
    """
    # container 
    scoring = [name]
    
    for scoring_metric in ["accuracy", "precision", "recall"]:
        measurement = cross_val_score(model,
                                      X, 
                                      y, 
                                      scoring = scoring_metric, 
                                      cv = 5,
                                      verbose = 0,
                                      n_jobs = 4)
        
        scoring.append(mean(measurement))

    return scoring


def stratified_cv(features, labels, cv = 5, threshold = 0.5, model = None):
    """
    Performs stratified cv testing
    
    Parameters:
    ===========
    features: DataFrame of features (Required)
    labels: Series of target labels (Required)
    cv: int
        the number of cross validations desired
    threshold: float
        the probability threshold for class prediction
    model: estimator
        Able to take a pretrained model, or use a standard logistic regression
    
    Returns:
    ========
    results: Dataframe that contains AUC metrics
    """
    # initialize the splitting
    sss = StratifiedKFold(n_splits = cv)
    
    # create containers
    accuracy = []
    precision = []
    recall = []
    
    # begin cv
    for train_index, test_index in sss.split(features, labels):
        # initialize logistic regressor
        if model == None:
            model_cl = LogisticRegression(max_iter = 400, random_state = 42)
            
        if model != None:
            model_cl = model
        
        # split data
        train_features = features.iloc[train_index, :]
        test_features = features.iloc[test_index, :]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]
        
        # train model
        model_cl.fit(train_features, train_labels)
        
        # predict
        pred_prob = model_cl.predict_proba(test_features)
        pred = [int(i > threshold) for i in pred_prob[:, 1]]
        
        # scoring
        accuracy.append(accuracy_score(test_labels, pred))
        precision.append(precision_score(test_labels, pred))
        recall.append(recall_score(test_labels, pred))
        
    results =  pd.DataFrame(data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })
    
    return results 


def threshold_testing(features, labels, model, model_name):
    """
    Tests the model across mulitple thresholds
    
    Parameters:
    ===========
    features: Dataframe or Array
    labels: Series, array, or list
    model: estimator
    model_name: String
    
    Returns:
    ========
    threshold_performance_df: Dataframe
        contains 
    """
    threshold_performance = []
    for threshold_value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # get AUC measures for theshold
        results = stratified_cv(features, 
                                labels, 
                                threshold = threshold_value,
                                model = model)

        # organize results
        mean_results = results.mean(axis = 0)
        full_metric = [model_name,
                       threshold_value, 
                       round(mean_results[0], 4), 
                       round(mean_results[1], 4), 
                       round(mean_results[2], 4)]

        # add to container        
        threshold_performance = np.concatenate(
            (threshold_performance, full_metric), 
            axis = 0
        )
        
    threshold_performance_df = pd.DataFrame(
        data = threshold_performance.reshape((8, 5)),
        columns = ["Model", "Threshold", "accuracy", "precision", "recall"]
    )
        
    return threshold_performance_df
def temp():
    """
    TEMP
    
    Parameters:
    ===========
    
    
    Returns:
    ========
    """
    return 0
