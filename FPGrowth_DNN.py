"""
This code is written for FIT 3162 (Final Year Project of Monash Univerisity)
Our proposed method called FP Growth NN is a novel Associative Classification method.
We use FP Growth to mine the rules for rule generation. We used a formula to rank the rules
which makes use of information gain and correlation. We then used a backpropagation
neural network for classification. We tested our method using many datasets from the
UCI Machine Learning Repository

Authors: Team What On Earth
        (Low Yi Xian, Cheryl Neoh Yi Ming & Vhera Kaey Vijayaraj)

        Supervised by: Dr Ong Huey Fang
"""

# Inbuilt Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Local Imports
from datapreprocess import preprocess_data, encode_data, information_gain
from modifiedfpgrowth import get_classification_association_rules, rankRule


def SEQ_DNN(X_train, y_train, X_test, y_test, X_valid, y_valid, number_of_classes, hidden_nodes):
    """
        This function builds the Sequential model (Neural Network) and compiles it
        to perform classification and obtain the accuracy. It has 3 layers: An Input Layer,
        A Hidden layer with 50 hidden nodes, and an Output Layer. The Input layer's number
        of nodes corresponds to the number of attributes we are using.

        Arguments:
            X_train: 70% of the dataset (excluding class column) after splitting to train the NN (excluding class)
            y_train: 70% of the class column after splitting to train the NN (only
            X_test:  15% of the dataset (excluding class column) to test the model built
            y_test:  15% of the class column as a result to test the model built
            X_valid:
            y_valid:
            number_of_classes: number of classses
            hidden_nodes: Number of hidden nodes to be used in the hidden layer

        Returns:
            loss: loss of the model
            accuracy: accuracy of the model after testing
            model
        """

    # building the Sequential model
    model = Sequential()

    # adding the layers: Input, Hidden and Output layer
    model.add(Dense(hidden_nodes, input_dim=X_train.shape[1], activation='sigmoid'))
    model.add(Dense(number_of_classes, activation='softmax'))

    # creating early stopping optimizer to avoid overfitting
    # we are monitoring the loss
    earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)

    # fitting the model with the data. Since we are doing multiclass classification,
    # we are using sparse categorical crossentropy as the loss function.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[earlystop_callback],
              verbose=0)

    # obtaining the accuracy and loss after testing the model with the test data
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy * 100, loss, model


def feature_selection(rankedCARs, feature_info, data, class_column, k):
    """
       This function selects features from the ranked CARs based on the value k (50, in our case).
       It extracts the values from the columns based
       """
    # extracting the top k rules
    N = len(rankedCARs)
    string = "Number of CARs generated : " + str(N) + "\n" + "\n"
    f = open("answer.txt", "a")
    f.write(string)
    f.close()
    top_k_rules = [antecedent for antecedent in list(rankedCARs)[:k]]

    # extracting the columns from these 50 rules. We only want the columns where the
    # attributes of the rules are present.
    features_to_extract = [class_column]
    for rule in top_k_rules:
        for feature in rule:
            for key in feature_info.keys():
                if feature in feature_info[key]:
                    features_to_extract.append(key)
    features_to_extract = list(set(features_to_extract))

    # filter dataset based on columns (features) that we extracted
    data_k = data.loc[:, data.columns.isin(features_to_extract)]
    return data_k, features_to_extract


def get_stats(number_of_classes, model, testing_data, testing_target, class_names):
    """
      This function gets the statistics for our neural network model after training.
      We will provide a classification report that will be output to the user and the area under
      the curve score as well. These measurements will determine the strength of our model.
      """
    predicted = model.predict_classes(testing_data)
    cnf_matrix = confusion_matrix(testing_target, predicted)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)

    print('TPR:', TPR)
    print('TNR:', TNR)
    print('FP:', FP, 'FPR:', FPR)
    print('FN:', FN, 'FNR:', FNR)

    # prints out the classification report to the output file
    string = "=============================================================" + "\n"
    string1 = "                    Classification Report                    " + "\n"
    string2 = "=============================================================" + "\n"
    f = open("answer.txt", "a")
    f.write(string)
    f.write(string1)
    f.write(string2)
    f.close()

    report = classification_report(testing_target, predicted)
    string = str(report) + "\n"
    f = open("answer.txt", "a")
    f.write(string)
    f.close()

    predicted_p = model.predict(testing_data)  # get the predicted probabilities after using test data

    # gets the area under the curve score. Some AUC's cant be obtained due to having the class
    # not present in the testing data. Therefore, the score would not be able to be calculates
    try:  # try to compute the score
        # in the case of binary classification (only 2 classes)
        if number_of_classes == 2:
            roc = roc_auc_score(testing_target, predicted_p[:, 1])
            string = "ROC AUC score : " + str(roc) + "\n"
            f = open("answer.txt", "a")
            f.write(string)
            f.close()
        else:
            # non binary classification (multi class classification)
            roc = roc_auc_score(testing_target, predicted_p, multi_class='ovr')
            string = "ROC AUC score : " + str(roc) + "\n"
            f = open("answer.txt", "a")
            f.write(string)
            f.close()
    except ValueError:  # if all classes are not present in the test data, we will print this
        print("Class is not present in testing target")
        string = "Class is not present in testing target" + "\n"
        f = open("answer.txt", "a")
        f.write(string)
        f.close()


def classification(data, class_column, min_support, min_confidence):
    """
    This function is the main function which runs our AC algorithm using FP Growth and Neural Network. We
    will first generate rules (CARs) using FP Growth. We will then rank the rules using information gain
    and corelation. After ranking them, we will take the top 50 rules and extract the features (columns)
    to use in the neural network. We will then use a Sequential Model which is a backpropagation neural
    network to get the accuracy of our classifier. We will then obtain the classificaation report and 
    area under the curve score to measure the performance of our classifier.

    Arguments:
        data: dataset we are using after preprocessing
        class_column: class column of the data
        min_support: minimum support of the data to be used for FP Growth
        min_confidence: minimum confidence of the data to be used for FP Growth
    """

    # getting the information gain of the dataset
    info_gain = information_gain(pd.DataFrame(data), class_column)
    temp = data.copy()
    temp = pd.DataFrame(temp)

    # getting the class column
    class_col = temp.iloc[:, class_column]

    # getting the class column values
    class_names = class_col.unique()

    # writing the string to the output file
    string = "===============================================================" + "\n"
    string1 = "            Top 50 Classification Associative Rules           " + "\n"
    string2 = "===============================================================" + "\n"
    f = open("answer.txt", "w")
    f.write(string)
    f.write(string1)
    f.write(string2)
    f.close()

    # Splitting the data into 70:15:15 ratio for training data : testing data : validation
    training_data, testing_data, train_rows, test_rows = train_test_split(data, range(len(data)), test_size=0.3,
                                                                          random_state=1998)

    # obtaining the information of the feature from the data
    data = pd.DataFrame(data)
    feature_info = {column: list(data[column].unique())
                    for column in data.columns}

    # Generate CARs using FP-growth ARM algorithm and ranking them using information gain and correlation
    CARs = get_classification_association_rules(training_data, feature_info[class_column], min_support, min_confidence)
    ranked_CARs = rankRule(training_data, CARs, info_gain, class_column, feature_info, use_conf_supp=True)

    printTop50Rules(ranked_CARs)  # to print all the top 50 rules

    # Feature selection using Top 50 CARs and setting up the training data to fit the NN classifier
    topkdata, columns_used = feature_selection(ranked_CARs, feature_info, data, class_column, 50)

    # prepares the data for the neural network by encoding them into binary
    topkdata, target = encode_data(topkdata, class_column)

    # preparing the training data
    training_data = topkdata[train_rows]
    training_target = target[train_rows]

    # preparing validation data and testing data
    testing_data, validation_data, testing_target, validation_target = train_test_split(topkdata[test_rows],
                                                                                        target[test_rows],
                                                                                        test_size=0.5,
                                                                                        random_state=1998)

    # Fitting the NN classifier
    number_of_classes = len(feature_info[class_column])
    accuracy_seq, loss, model = SEQ_DNN(training_data, training_target, testing_data, testing_target, validation_data,
                                        validation_target, number_of_classes, hidden_nodes=50)

    # to get classification report and area under the curve score
    get_stats(number_of_classes, model, testing_data, testing_target, class_names)

    # writes the columns used for the neural network in the output file
    string = "Columns used for Neural Network : " + str(columns_used) + "\n"
    f = open("answer.txt", "a")
    f.write(string)
    f.close()

    # write the accuracy of the neural network to the output file
    string = "Accuracy of Sequential DNN : " + str(accuracy_seq) + "\n"
    f = open("answer.txt", "a")
    f.write(string)
    f.close()


def printTop50Rules(rules):
    """
    This function prints out the top 50 rules after we have ranked the CARs
    Arguments:
        rules: CARs after ranking
    """

    counter = 1
    for rule in rules:

        # we will loop 50 times to print out the rules
        if counter < 51:

            # gets the antecedant, consequent, confidence, interestingness and support from the rule
            antecedant = rule
            consequent = rules.get(rule)[0]
            confidence = rules.get(rule)[1]
            support = rules.get(rule)[2]
            interestingness = rules.get(rule)[3]

            # appends it accordingly for printing
            string = "Rule " + str(counter) + ":" + str(antecedant) + " ---> " + str(consequent) + "\n"
            string2 = "Interestingness: " + str(interestingness) + " " + "Confidence: " + str(
                confidence) + " " + "Support: " + str(support) + '\n'
            space = "\n"

            # writes string to the text file
            f = open("answer.txt", "a")
            f.write(string)
            f.write(string2)
            f.write(space)
            f.close()
            counter += 1
        else:
            break


def runAC(classCol, supportValue, confidenceValue, nmli=None, columnName=None):
    #    data = preprocess_data("dataFile.data",int(classCol),numeric_columns=nmli,column_names= columnName)
    data = preprocess_data("dataFile.data", numeric_columns=nmli, column_names=columnName)
    classification(data, int(classCol), supportValue, confidenceValue)
