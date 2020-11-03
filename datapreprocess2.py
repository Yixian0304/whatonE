import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer, minmax_scale
from sklearn.feature_selection import mutual_info_classif

def preprocess_data(path_to_data, class_col_num,header=None, missing_value_symbol=None,numeric_columns = None):
    """ This function reads a file from the path provided, it can only read DATA Files (attributes seperated by commas (,) only)
    and csv files. It will also check for any missing values, and perform imputation. File must only contain categorical features.

    Args:
        path_to_data (str): the path to the file
        header (bool/int, optional): If the file contains a header, supply with the index of the header in the file. Defaults to None.
        missing_value_symbol (char, optional): If there is missing values, input the symbol used to indicate a missing value. Defaults to None.

    Returns:
        list: A python list array of the data
    """
    print(numeric_columns)
    print("hellloooo")
    data = pd.read_csv(path_to_data, header=header, dtype=str)
    class_col = data.iloc[:,class_col_num]
    data.drop(data.columns[class_col_num],axis=1,inplace=True)
    data[len(data.columns)+1] = class_col

    data.columns = [np.arange(0, data.shape[1])]

    # convert any missing values in the data into NULL, so that it recognizable by python
    # then, replace the missing values in the data by imputing
    if missing_value_symbol is not None:
        data.replace(to_replace=[missing_value_symbol],
                     value=np.nan, inplace=True)
        clean_data(data)

    # scalling using min-max normalization and
    # change numerical attribute columns into categorical attributes

    if numeric_columns == None:
        numeric_columns = list(data._get_numeric_data().columns)

    if (numeric_columns is not None) and (numeric_columns != []):
        #min_max(data, numeric_columns)
        discretize_data(data, numeric_columns)

    attributize_data(data)
    # print(data.values.tolist())
    return data.values.tolist()


def min_max(data, numeric_columns):
    """ This function scales the numerical attributes into the range of 0 to 1 using the Min-Max Normalisation method.
    This is used before the discretization of the attributes method to reduce the number of categorical attributes produced.

    Args:
        data (pdarray): The data which contains the numerical attributes.
        numeric_columns ([int]): A list containing the indexes of the numerical attributes in the data.
    """
    numeric_data = data.iloc[:, numeric_columns]
    scaled_data = minmax_scale(numeric_data)

    for i in range(len(numeric_columns)):
        data.iloc[:, numeric_columns[i]] = scaled_data[:, i]


def discretize_data(data, numeric_columns):
    """ This function takes the scaled numerical attributes and discretizes the attributes. The number of bins used by the
    discretization method depends on the largest value in all the numerical attributes, to ensure that the number of bins
    is generalised to each dataset.

    Args:
        data (pdarray): The data which contains the scaled numerical attributes.
        numeric_columns ([int]): A list containing the indexes of the numerical attributes in the data.
    """
    numeric_data = data.iloc[:, numeric_columns]
    discretizer = KBinsDiscretizer(n_bins=
        10, encode='ordinal', strategy='quantile')
    discretized_data = discretizer.fit_transform(numeric_data)
    discretized_data = discretized_data.astype(int).astype(str)  # convert it to string, as categorical

    for i in range(len(numeric_columns)):
        data.iloc[:, numeric_columns[i]] = discretized_data[:, i]

def clean_data(data):
    """ This function deals with missing values in a dataset, the imputing works on two types of
    data: categorical and numerical. For the categorical missing value, it will impute using the
    most frequent value. As for the numerical missing value, it will impute using the mean value.

    Args:
        data (pdarray): The dataset which could contain some missing values.

    Returns:
        pdarray: The dataset with no missing values.
    """
    # get categorical columns and columns with missing values
    categorical_columns = list(
        set(data.columns) - set(data._get_numeric_data().columns))
    missing_values = list(data.isnull().any())
    data = pd.DataFrame(data)
    cat_col =[]
    for item in categorical_columns:
        cat_col.append(item[0])

    for column in range(len(missing_values)):
        if missing_values[column]:
            if column in cat_col:
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                imputer = SimpleImputer(strategy='mean')
            imputer.fit(data)
            imputed_data = (imputer.transform(data))
            imputed_data = pd.DataFrame(imputed_data)
            data.iloc[:,column] = imputed_data.iloc[:,column]



def attributize_data(data):
    """ This function adds concatenates the column name with the values, so that when it is used for frequent itemset mining and generation of Classification
    Association Rules (CARs), it will be able to differentiate which column it is taken from. Since, each column has a different meaning.

    Args:
        data (pdarray): The dataset.
    """
    for column in data.columns:
        data[column] = "Feature " + str(column) + " : " + data[column]


def encode_data(data, class_column):
    """ This function discretizes the features and encodes the target column in the dataset. For the features, it converts all the categorical columns in dataset
    into seperate columns, where each value becomes 0 and 1. As for the target column, it will only be bina encode a categorical attribute ( for class labels )

    Args:
        data (pdarray): The data to be discretized and encoded and seperated into target and features
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
