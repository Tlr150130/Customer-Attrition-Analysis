"""
File Management Functions: file_management.py
    Here are the helper functions for managing data and image files
"""
##### Libraries #######################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os
import joblib

sys.path.insert(0, "./helper_functions")
sys.path.insert(0, "./data_storage")

from reporting import dtype_selector
##### Functions for feature_visualizer ###################################


##### Functions for outlier_detector ###################################

##### Functions to be called ###########################################
def data_import(dataset_type):
    """
    Import either train or test dataset
    
    Parameters:
    ===========
    dataset_type: String ["train", "test"]
        Denotes the desired dataset
        
    Returns:
    ========
    data: DataFrame
    """
    # Guard Function
    if dataset_type not in ["train", "test"]:
        raise ValueError("Please enter 'train' or 'test' to specify desired dataset")
     
    # import data
    file_path = "data_storage/{}_data.pkl".format(dataset_type)
    pickle_path = open(file_path, "rb")
    
    data = pickle.load(pickle_path)
    
    pickle_path.close()
    
    return data


def save_figure(fig_name, tight_layout=True, fig_extension="png"):
    """
    Saves image into the image folder
    
    Parameters:
    ===========
    fig_name: String
        filename for figure
    
    tight_layout: Boolean
        Condition for tight layout format for plotting
        
    fig_extension: String
        file extension for image
        
    resolution: int
        resolution of image
    """
    # create folder if folder is not created
    PROJECT_ROOT_DIR = "."
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    os.makedirs(IMAGES_PATH, exist_ok=True)

    # create path to save file
    path = os.path.join(IMAGES_PATH, fig_name + "." + fig_extension)
    print("Saving figure", fig_name)
    
    # add tight layout
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension)
    

def save_model(results, filename):
    """
    Saves results as a .pkl file
    
    Parameters:
    ===========
    results: estimator 
        
    filename: String
        Desired name of the file
    """
     
    # set folder path
    PROJECT_ROOT_DIR = "."
    folder_path = os.path.join(PROJECT_ROOT_DIR, "model")
    
    # create folder if folder does not exist
    os.makedirs(folder_path, exist_ok = True)
    
    # set filepath
    filepath = os.path.join(folder_path, filename + '.pkl')
    
    # save file         
    joblib.dump(results, filepath)
    print("Results Saved to {}".format(filepath))


def temp():
    """
    TEMP
    
    Parameters:
    ===========
    
    
    Returns:
    ========
    """
    return 0