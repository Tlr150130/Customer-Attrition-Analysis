"""
Discover Visualization Functions: visualization.py
    Here are the helper functions for data visualization
"""
##### Libraries #######################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, "./helper_functions")

from reporting import dtype_selector
from file_management import save_figure

# pandas config
pd.options.mode.chained_assignment = None # stops slice warning

##### Functions for feature_visualizer ###################################
def cat_visualizer(data, target_col):
    """
    Creates 2 EDA plots: 
        [histogram of overall counts, histogram of class counts]
    
    Parameters:
    ===========
    data: training dataframe
    col: feature name that we want to plot
    target_col: name of dependent variable
    """
    # get columns names except target 
    cols = data.columns.drop(target_col)
    
    # loop through all categorical features
    for col in cols:
        plt.figure(figsize = (12, 6))
        sns.set(font_scale = 1.3)
        sns.set_style("darkgrid")
        plt.suptitle("Distribution of {}".format(col))   

        ### histogram
        plt.subplot(1, 2, 1)
        sns.countplot(y = col, data = data, color = "salmon")

        ### histogram
        plt.subplot(1, 2, 2)
        sns.countplot(y = col, hue = target_col, data = data)
        plt.tight_layout()

        # save plot
        save_figure("cat_eda_plot_{}".format(col))
        
        ### show plot and close to prevent memory overload
        plt.show()
        plt.close('all')
    
def num_visualizer(data, target_col):
    """
    Creates 2 EDA plots: [histogram, boxplot]
    
    Parameters:
    ===========
    data: filtered dataframe
    target_col: name of dependent variable
    """
    # get columns names except target
    cols = data.columns.drop(target_col)
    
    # loop through all numeric features
    for col in cols:
        plt.figure(figsize = (10, 4))
        sns.set_style("darkgrid")
        plt.suptitle("Distribution of {}".format(col))
        plt.tight_layout()

        ### box plot
        plt.subplot(1, 2, 1)    
        sns.boxplot(data = data, x = target_col, y = col)

        ### histogram
        plt.subplot(1, 2, 2)
        sns.histplot(data, x = col, bins = 50)
        plt.ylabel(" ")

        ### save plot
        save_figure("num_eda_plot_{}".format(col))
        
        ### show plot and close to prevent memory overload
        plt.show()
        plt.close('all')

def target_visualizer(target, target_col):
    """
    Creates histogram
    
    Parameters:
    ===========
    target: series of target values
    target_col: name of target variable
    """
    
    # figure configuration
    plt.figure(figsize = (6, 4))
    sns.set_style("darkgrid")
    plt.suptitle("Distribution of {}".format(target_col))
    plt.tight_layout()
    
    ### histogram
    counts = np.array(target.value_counts().sort_values(ascending = False))
    index = np.array(target.value_counts().sort_values(ascending = False).index)
    sns.barplot(x = index, y = counts, color = "salmon") 
    plt.ylabel(" ")

    # save plot
    save_figure("target_eda_plot_{}".format(target_col))

##### Functions to be called ###########################################
def feature_visualizer(data, feature_type, target_col):
    """
    Visualizes data based on type [categorical, numerical, target]
    
    Parameters:
    ===========
    data: DataFrame
    feature_type: String ["cat", "num", "target"]
        Denotes the dtype of feature and desired visualizations
        
    target_col: String
        Name of target feature 
    """
    # Guard Function
    if feature_type not in ["cat", "num", "target"]:
        raise ValueError("Please enter 'num', 'cat', or 'target' to specify data type")
        
    if feature_type == "cat":
        # filter data
        cat_data = dtype_selector(data, "cat")
        cat_data[target_col] = data[target_col]
        
        # Visualize
        cat_visualizer(cat_data, target_col)
        
    if feature_type == "num":
        # filter data
        num_data = dtype_selector(data, "num")
        num_data[target_col] = data[target_col]
        
        # Visualize
        num_visualizer(num_data, target_col)
        
    if feature_type == "target":
        # filter data
        target_data = data[target_col]
        
        # Visualize
        target_visualizer(target_data, target_col)


def positive_class_percentages(data, target_col):
    """
    Print percentage of positive class for feature combinations
    
    Paramters:
    ==========
    data: DataFrame
    """
    data_cat = dtype_selector(data, "cat")
    cols = list(data_cat.drop(target_col, axis = 1).columns)
    
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols) - 1):
            # set feature names
            col1 = cols[i]
            col2 = cols[j]
            print("")
            print(data_cat.groupby([col1, col2])[target_col].mean())


def heatmap(data):
    """
    Create a heap map of numerical features
    
    Parameters:
    ===========
    data: DataFrame
    """
    plt.figure(figsize = (12, 10))
    sns.heatmap(data = data.corr(), cmap = "rocket", annot = True)
    plt.show()

def cat_num_pivot_tables(data, target_col):
    """
    Print percentage of positive class for feature combinations
    
    Paramters:
    ==========
    data: DataFrame
    """
    data_cat = dtype_selector(data, "cat")
    data_num = dtype_selector(data, "num")
    cat_cols = list(data_cat.drop(target_col, axis = 1).columns)
    num_cols = data_num.columns
    
    for cat_col in cat_cols:
        for num_col in num_cols:
            # set feature names
            print('\n{0:*^80}'.format(' {} - {} '.format(cat_col, num_col)))
            print(data.groupby([cat_col, target_col])[num_col].mean())
            
def plot_model_results(model_results_df):
    """
    Plot Model Results: barplots of Accuracy, Precision, Recall
    
    Parameters:
    ===========
    model_results_df: DataFrame[columns: 'Name', 'Accuracy', 'Precision', 'Recall']
    """
    
    plt.figure(figsize = (15, 4))
    sns.set_style("darkgrid")
    plt.suptitle("Model Results")
    plt.tight_layout()

    ### Accuracy
    plt.subplot(1, 3, 1)    
    g = sns.barplot(data = model_results_df, x = "Model", y = "Accuracy", ci = None)
    plt.ylim(0.6, 0.8)

    for index, row in model_results_df.iterrows():
            g.text(row.name, 
                   row["Accuracy"] + 0.005, 
                   round(row["Accuracy"], 2), 
                   color='black', 
                   ha="center") 

    ### Precision
    plt.subplot(1, 3, 2)    
    g = sns.barplot(data = model_results_df, x = "Model", y = "Precision", ci = None)
    plt.ylim(0.35, 0.5)

    for index, row in model_results_df.iterrows():
            g.text(row.name, 
                   row["Precision"] + 0.005, 
                   round(row["Precision"], 2), 
                   color='black', 
                   ha="center") 

    ### Recall
    plt.subplot(1, 3, 3)    
    g = sns.barplot(data = model_results_df, x = "Model", y = "Recall", ci = None)
    plt.ylim(0.6, 0.8)
    
    # add labels
    for index, row in model_results_df.iterrows():
            g.text(row.name, 
                   row["Recall"] + 0.005, 
                   round(row["Recall"], 2), 
                   color='black', 
                   ha="center") 

    # save plot
    save_figure("Model_Results")
    
    # show plot and close all the plots
    plt.show()
    plt.close('all')
    
def plot_feature_importance(importance, names):
    """
    Create arrays from feature importance and feature names
    
    Parameters:
    ===========
    importance: list of floats
        relative importance of each features according to tree model
        
    names: list of strings
        feature names   
    """
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    feature_importance_df = pd.DataFrame(data = {'feature_names': feature_names, 
                                                 'feature_importance': feature_importance})

    #Sort the DataFrame in order decreasing feature importance
    feature_importance_df.sort_values(by = ['feature_importance'], ascending=False, inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    
    #Plot Searborn bar chart
    sns.barplot(x = feature_importance_df['feature_importance'], 
                y = feature_importance_df['feature_names'])
    
    #Add chart labels
    title = 'XGBOOST FEATURE IMPORTANCE'
    plt.title(title)
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    
    # save image
    save_figure(title)