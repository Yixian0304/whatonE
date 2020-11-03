  
import pandas as pd
import numpy as np
from modifiedfpgrowth import get_classification_association_rules
from sklearn.feature_selection import mutual_info_classif
from datapreprocess2 import preprocess_data
from sklearn.model_selection import train_test_split


def information_gain(data, class_column):
    target = data[class_column]
    features = data.drop(columns=class_column, axis=1) # dropping the class column
    feature_columns = list(features.columns) # reamining columns

    information_gain = mutual_info_classif(features.values.tolist(), target, discrete_features = True)
    average = sum(information_gain)/len(list(features.columns))

    info_gain = {}
    columns_removed = []
    for index in range(len(information_gain)):
        if information_gain[index] >= average:
            info_gain[ feature_columns[index] ] = information_gain[index]
        else:
            columns_removed.append(feature_columns[index])

    data.drop(columns=columns_removed, axis=1, inplace = True)
    #print(pd.DataFrame(data)) # dataset we can use to generate the rules
    print('info_gain',info_gain)
    return info_gain # infomation gain of each column (we need to rank according to this)

    # print(dict(zip(list(features.columns), information_gain)))
    # print(sum(information_gain)/len(list(features.columns)))
    # print(info_gain)
    # print(columns_removed)

def list_to_dict_IG(alist):
    """
    This function converts a list to a dictionary
    """
    dictionary = {}
    for i in range(len(alist)):
        #confidence = alist[i][0]
        #support = alist[i][1]
        info_gain = alist[i][0]
        antecedent = alist[i][1]
        consequent = alist[i][2]
        dictionary[antecedent] = (consequent,info_gain)
    return dictionary

def rankRule_IG(cars,info_gain,feature_info):
    #cars = {('Feature (3,) : 0-2', 'Feature (7,) : central'): (('Feature (9,) : no-recurrence-events',), 0.8), ('Feature (3,) : 0-2', 'Feature (4,) : no', 'Feature (7,) : central'): (('Feature (9,) : no-recurrence-events',), 0.8), ('Feature (3,) : 0-2', 'Feature (7,) : central', 'Feature (8,) : no'): (('Feature (9,) : no-recurrence-events',), 0.8)}
    ranked_list = []
    for rule in cars:
        consequent = cars.get(rule)[0]
        antecedant = rule
        key_list = []
        total = 0
        for feature in rule:
            for key in feature_info.keys():
               if feature in feature_info[key]:
                   key_list.append(key)
        for i in range(len(key_list)):
            for item in info_gain:
                if key_list[i] == item:
                    total += info_gain[item]
        average = total/len(key_list)
        ranked_list.append([average,antecedant,consequent])
    ranked_list.sort(reverse=True)
    print(ranked_list)
    ranked_dict = list_to_dict_IG((ranked_list))
    print(ranked_dict)

    return ranked_dict

def list_to_dict_IG_conf_sup(alist):
    """
    This function converts a list to a dictionary
    """
    dictionary = {}
    for i in range(len(alist)):
        confidence = alist[i][0]
        support = alist[i][1]
        info_gain = alist[i][2]
        antecedent = alist[i][3]
        consequent = alist[i][4]
        dictionary[antecedent] = (consequent,info_gain,confidence,support)
    return dictionary

def rankRule_IG_conf_sup(cars, info_gain, feature_info):
    # cars = {('Feature (3,) : 0-2', 'Feature (7,) : central'): (('Feature (9,) : no-recurrence-events',), 0.8), ('Feature (3,) : 0-2', 'Feature (4,) : no', 'Feature (7,) : central'): (('Feature (9,) : no-recurrence-events',), 0.8), ('Feature (3,) : 0-2', 'Feature (7,) : central', 'Feature (8,) : no'): (('Feature (9,) : no-recurrence-events',), 0.8)}
    ranked_list = []
    for rule in cars:
        consequent = cars.get(rule)[0]
        confidence = cars.get(rule)[1]
        support = cars.get(rule)[2]
        antecedant = rule
        key_list = []
        total = 0
        for feature in rule:
            for key in feature_info.keys():
                if feature in feature_info[key]:
                    key_list.append(key)
        for i in range(len(key_list)):
            for item in info_gain:
                if key_list[i] == item:
                    total += info_gain[item]
        average = total / len(key_list)
        ranked_list.append([average,confidence,support, antecedant, consequent])
    ranked_list.sort(reverse=True)
    print(ranked_list)
    ranked_dict = list_to_dict((ranked_list))
    print(ranked_dict)

    return ranked_dict



def test_IG(data,min_support,min_confidence):
    data = pd.DataFrame(data)
    feature_info = {column: list(data[column].unique())
                    for column in data.columns}
    #print(feature_info)
    data = np.array(data)
    class_column = len(pd.DataFrame(data).columns) - 1

    # gets information gain for each attribute
    info_gain = information_gain(pd.DataFrame(data), class_column)
    training_data, testing_data, _, _ = train_test_split(
        data, range(len(data)), test_size=0.3, random_state=1998)
    training_data = list(training_data)
    # obtain information of the feature from the data
    data = pd.DataFrame(data)
    feature_info = {column: list(data[column].unique())
                    for column in data.columns}

    # Generate cars using FP-growth ARM algorithm
    CARs = get_classification_association_rules(
        training_data, feature_info[class_column], min_support, min_confidence)

    rankedCARs = rankRule_IG(CARs,info_gain,feature_info)




if __name__ == "__main__":
    # Dataset imports
    breast_cancer = preprocess_data("./Datasets/breast-cancer.data",0, missing_value_symbol='?')  # total = 286, training = 200, testing = 86
    test_IG(breast_cancer,8,0.5)

    # print(pd.DataFrame(breast_cancer))
    # information_gain(pd.DataFrame(breast_cancer),9)

    # balance_scale = preprocess_data("./Datasets/balance-scale.data")
    # car = preprocess_data("./Datasets/car.data")   # class_column = 6
    # lymphography = preprocess_data("./Datasets/lymphography.data")
    # iris = preprocess_data("./Datasets/iris.data", numeric_columns = [0,1,2,3])
    # lymphography = pd.DataFrame(lymphography)
    # information_gain(lymphography, 0)