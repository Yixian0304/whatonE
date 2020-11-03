"""
This module is for FIT3162 (Final Year Project of Monash University) and created for data preprocessing purposes.

It contains the following methods :
    a) preprocess_data :- reads in a DATA file or csv file and preprocesses the dataset
    b) clean_data :- impute missing values in an attribute/feature
    c) attributize_data :- concat attribute names to their values, to ensure they are differentiable in the rules.
    d) encode_data :- convert all attributes to binary attributes
    e) standardize_data :- scales numerical attributes to a range from -1 to 1 using standard deviation
    f) dicretize_data :- discretize the numerical attribute into categorical data
    g) information_gain :- find the information gain between attributes/features with the class label
    h) correlation :- find the correlation between attributes/features with the class label

Authors: Team What On Earth
        (Low Yi Xian, Cheryl Neoh Yi Ming & Vhera Kaey Vijayaraj)

Supervised by: Dr Ong Huey Fang
"""

# Inbuilt imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer, scale, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif


def preprocess_data(path_to_data, header=None, missing_value_symbol=None, numeric_columns=None, remove_columns=None,column_names = None):
    """"
    This function reads the dataset file and preprocesses the data based on the user's input. It can preprocesses the data
    by imputing missing values, converting numeric attributes to categorical attributes or concatenating attribute names
    to the values of the attribute.

    Args:
        path_to_data (str): Path to the dataset file
        header (bool, optional): If it is True, then it will read the dataset with the attribute name as the column name. Defaults to None.
        missing_value_symbol (str, optional): A character/string that is used to identifiy missing values in the dataset. Defaults to None.
        numeric_columns ([int], optional): A list of column numbers that indicates that they are numerical attributes. Defaults to None.
        remove_columns ([int], optional): A list of column numbers that the user would like to remove. Defaults to None.
        column_names ([str], optional): A list of strings that denotes the attributes/features name. Defaults to None.

    Returns:
        list: A python list array of the preprocessed data
    """
    # read the dataset
    data = pd.read_csv(path_to_data, header=header, dtype="str")

    # convert any missing values in the data into NULL, so that it recognizable by python
    # then, impute the missing values in the dataset
    if missing_value_symbol is not None:
        data.replace(to_replace=[missing_value_symbol],
                     value=np.nan, inplace=True)
        clean_data(data)

    # convert numerical attributes into categorical attributes, by standardizing the attributes first before
    # discretizing the attributes.
    if numeric_columns is not None:
        standardize_data(data, numeric_columns)
        discretize_data(data, numeric_columns)

    # concatenate attribute names to the values of the attribute
    attributize_data(data, column_names)

    # remove any columns that the user has provided
    # if remove_columns:
    #     data.drop(columns=remove_columns, axis=1, inplace = True)

    return data.values.tolist()


def clean_data(data):
    """
    This function deals with missing values in a dataset, the imputing works on two types of data: categorical and numerical.
    For the categorical missing value, it will impute using the most frequent value of the attribute. As for the numerical
    missing value, it will impute using the mean value of the attribute.

    Args:
        data (pd.DataFrame): The dataset which contains missing values.

    Returns:
        None, but it changes the dataset of the caller function.
    """
    # get categorical columns and columns with missing values
    categorical_columns = list(set(data.columns) - set(data._get_numeric_data().columns))
    missing_values = list(data.isnull().any())

    for column in range(len(missing_values)):
        if missing_values[column]:
            # Impute missing value for categorical attributes
            if column in categorical_columns:
                imputer = SimpleImputer(strategy='most_frequent')

            # Impute missing value for numerical attributes
            else:
                imputer = SimpleImputer(strategy='mean')

            imputer.fit(data)
            imputed_data = pd.DataFrame(imputer.transform(data))
            data[column] = imputed_data[column]

def attributize_data(data, column_names):
    """
    This function concatenates the column name with the values, so that when it is used for frequent itemset mining and generation of Classification
    Association Rules (CARs), it will be able to differentiate which column it is taken from. Since, each column has a different meaning. If the user
    provides the column names it will be used instead. If it isn't then by default it will just concat "Feature" and a sequence of numbers starting from 0.

    Args:
        data (pd.DataFrame): the dataset
        column_names: column names provided by the user

    Returns:
        None, but it changes the column name in the dataset of the caller function.
    """

    # renaming the columns by default
    if column_names == None:
        for column in data.columns:
            data[column] = "Feature " + str(column) + " : " + data[column]

    # rename the columns provided by the user
    else:
        i = 0
        for column in data.columns:
            data[column] = column_names[i] + " : " + data[column]
            i+=1


def encode_data(data, class_column):
    """
    It converts all the attributes in the dataset into binary attributes.

    Args:
        data (pd.DataFrame): the dataset
        class_column (int): The index of the class column

    Returns:
        list, list:  two list containing the discretized features and encoded target
    """
    # Encoding target
    target = data[class_column]
    lb_encoder = LabelEncoder()
    target = lb_encoder.fit_transform(target)

    # Discretizing features
    features = data.drop(columns=class_column, axis=1)
    encoder = OneHotEncoder(sparse=False, dtype=int)
    features = encoder.fit_transform(features)

    return features, target


def standardize_data(data, numeric_columns):
    """
    This function scales the numerical attributes into the range of -1 to 1 using the standard deviation.
    This is used before the discretization of the attributes method to reduce the number of categorical attributes produced.

    Args:
        data (pd.DataFrame): The data which contains the numerical attributes.
        numeric_columns ([int]): A list containing the indexes of the numerical attributes in the data.
    """
    numeric_data = data.iloc[:, numeric_columns]
    scaled_data = scale(numeric_data, with_std=True)

    for i in range(len(numeric_columns)):
        data.iloc[:, numeric_columns[i]] = scaled_data[:, i]


def discretize_data(data, numeric_columns):
    """
    This function takes the scaled numerical attributes and discretizes the attributes. The number of bins used by the
    discretization method depends on the largest value in all the numerical attributes, to ensure that the number of bins
    is generalised to each dataset.

    Args:
        data (pd.DataFrame): The data which contains the scaled numerical attributes.
        numeric_columns ([int]): A list containing the indexes of the numerical attributes in the data.
    """
    numeric_data = data.iloc[:, numeric_columns]

    # setting the number of bins
    if max(numeric_data) < 2:
        number_of_bins = 2
    else:
        number_of_bins = max(numeric_data)

    discretizer = KBinsDiscretizer(n_bins=number_of_bins, encode='ordinal', strategy='quantile')
    discretized_data = discretizer.fit_transform(numeric_data)
    discretized_data = discretized_data.astype(int).astype(str)  # convert it to string, as categorical

    for i in range(len(numeric_columns)):
        data.iloc[:, numeric_columns[i]] = discretized_data[:, i]


def information_gain(data, class_column):
    """
    This function calculates the information gain (IG) of each feature with the target in the dataset, it is used in preprocessing
    of the dataset, which is rule ranking and feature selection. It also removes redundant features in the dataset which has an information
    gain lower than 0.

    Args:
        data (pd.DataFrame): The dataset which has been preprocessed to contain only categorical attributes
        class_column (int): The index of the target in the dataset

    Returns:
        dict: A dictionary where the key is the index of the column and the value is the information gain.
    """
    target = data[class_column]
    features = data.drop(columns=class_column, axis=1)
    feature_columns = list(features.columns)

    # calculating information gain of the features with the target
    information_gain = mutual_info_classif(features.values.tolist(), target, discrete_features=True)

    # make a dictionary and obtain the columns of the features to be removed from the dataset
    info_gain = {}
    columns_removed = []
    for index in range(len(information_gain)):
        if information_gain[index] > 0:
            info_gain[feature_columns[index]] = information_gain[index]
        else:
            columns_removed.append(feature_columns[index])

    # remove the redundant features
    data.drop(columns=columns_removed, axis=1, inplace=True)
    return info_gain

def correlation(data, class_column):
    """
    This function calculates how correlated the attribute is to the class column. It is used together with information gain for rule
    ranking and feature selection

    Args:
        data (pd.DataFrame): The dataset which has been preprocessed to contain only categorical attributes
        class_column (int): The index of the target in the dataset

    Returns:
        list : a list containing the correlation value of each attribute to the class column
    """
    ordinal_encoder = OrdinalEncoder()
    data = ordinal_encoder.fit_transform(data)
    data = pd.DataFrame(data)
    corr = data.corr()[class_column]
    return corr.values.tolist()
