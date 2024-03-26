
def sklearn_performance(X, Y, test_size=.2):
    print("sklearn performance")
    Y_one_column = Y


    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Y_one_column are the labels associated with each audio file
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y_one_column, 
                                                        train_size=.2,
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
