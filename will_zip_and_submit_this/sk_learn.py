
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



def sklearn_performance(X, Y, train_size=0.80):
    print("sklearn performance")


  
    # Y_one_column are the labels associated with each audio file
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        train_size=train_size,
                                                        random_state=0)

    def support_vector_classifier_performance():

        '''
        Support Vector Classifier (SVC)
        '''
        from sklearn.svm import SVC

        clf = SVC(kernel='linear', C=1.0, random_state=0)

        clf.fit(X_train, Y_train)

        Y_predict = clf.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_predict)
        print("SVC accuracy:", accuracy)


        report = classification_report(Y_test, Y_predict)
        print("Report: ", report)
    
    def gaussian_naive_bayes_performance():
        '''
        Gaussian Naive Bayes' Classifier
        '''
        from sklearn.naive_bayes import GaussianNB

        clf = GaussianNB()

        clf.fit(X_train, Y_train)

        Y_predict = clf.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_predict)
        print("Gaussian Naive Bayes accuracy:", accuracy)

    def random_forest_classifier_performance():
        '''
        Random Forest Classifier
        '''
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=100, random_state=0)

        clf.fit(X_train, Y_train)

        Y_predict = clf.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_predict)
        print("RandomForest accuracy:", accuracy)

    support_vector_classifier_performance()
    gaussian_naive_bayes_performance()
    random_forest_classifier_performance()





def main():


    root_path = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/"



    np.random.seed(0)   # seed

    # 1. Load training data
    df = pd.read_csv(root_path + "mfcc_13_labels.csv")


    # 2. One hot encode labels
    Y_labels = df.iloc[:,-1]   # labels "blues", "classical", etc.
    Y_encoded = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(Y_labels))


    # 3. Standardize X 
    X = df.iloc[:,:-1]   # coefficients 
    X = (X - X.mean(axis=0)) / X.std(axis=0)


    # 4. Add column of ones to X
    X.insert(0, "X0", 1)


    sklearn_performance(X,Y_labels)




    exit()


    # 4. Split X and Y

    # X is (900 x 13) and Y_encoded is (900 x 10) 
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y_encoded, train_size=0.8, random_state=0)

    # Transpose y_train and y_test to be (10 x 900) 
    y_train = y_train.T
    y_validate = y_validate.T  







if __name__ == "__main__":
    main()
