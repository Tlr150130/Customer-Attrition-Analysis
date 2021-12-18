"""
Design Reporting Functions: reporting.py
    Here are the helper functions for data import and reporting characteristics concerning data
"""
##### Libraries #######################################################
import pandas as pd

##### Functions for logic_assessment###################################
def dtype_selector(data, var_type):
    """Returns dataframe with only columns of desired dtype"""
    """
    Parameters:
    ===========
    data: Dataframe
    var_type: String ["cat", "num"] that denotes desired column dtype
    
    Returns:
    ========
    selected_data: Dataframe with only specific column dtypes
    """
    # select columns of desired dtype
    if var_type not in ["cat", "num"]:
        raise ValueError("Please enter 'num' or 'cat' to specify data type")
    
    if var_type == "num":
        selected_data = data.select_dtypes(include = "number")
    if var_type == "cat":
        selected_data = data.select_dtypes(include = "object")
    
    return selected_data

##### Functions for clean_data_report function ###########################
def missing(data):
    """Prints missing values by columns"""
    print("Features           Missing Values")
    missing_values = data.isnull().sum(axis = 0).sort_values(ascending = False)
    print(missing_values)
    
def duplicates(data):
    """Prints Number of Duplicates"""
    if (data.duplicated().sum() == 0):
        print("There are no duplicates in the data.")
        
    else:
        print("There are duplicates in the dataset.")
        
def logic_assessment(data):
    """Assesses business logic for each feature"""
    # drop missing values if any
    data_no_na = data.dropna()
    
    # No negative values
    num_features = dtype_selector(data, "num")
    print("Negative Value Assessment")
    print("\nFeature            Negative Values")
    print((num_features < 0).sum(axis = 0))
    
    # Binary Feature Evaluation
    expected = ["HasChckng", "IsActiveMember", "Exited"] # change for future data
    observed = data.columns[data.isin([0,1]).all()] # filters for binary features
    
    print("\nExpected Binary Feature Evauation:")
    if set(expected) == set(observed):
        print("\nFeature\t\tIs Binary?")
        print("HaHasChckng:\tBinary")
        print("IsActiveMember:\tBinary")
        print("Exited:\t\tBinary")
    

##### Functions to be called ################################################
def dataset_glimpse(data):
    """Print data charateristics"""
    # dimensions
    print('{0:*^80}'.format(' Characteristics of Churn Dataset '))
    print("Dimensions of data: {} observations, {} features".format(data.shape[0], data.shape[1]))

    # data types
    print('\n{0:*^80}'.format(' Feature and data types '))
    print(data.dtypes)
    
    # head
    print('\n{0:*^80}'.format(' First 6 Observations '))
    display(data.head())

def clean_data_report(data):
    """Reports any data that is missing, duplicated, or defies business logic"""
    # missing values
    print('{0:*^80}'.format(' Missing Values '))
    missing(data)
    
    # duplicated
    print('\n{0:*^80}'.format(' Duplicates '))
    duplicates(data)
    
    # defy business logic
    print('\n{0:*^80}'.format(' Businesss Logic Defiance '))
    logic_assessment(data)


