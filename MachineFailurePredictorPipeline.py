import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import warnings


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv('ai4i2020.csv')

    class_count_0, class_count_1 = df['Machine failure'].value_counts()

    # Separate class
    class_0 = df[df['Machine failure'] == 0]
    class_1 = df[df['Machine failure'] == 1]
    # # print the shape of the class
    # print('class 0:', class_0.shape)
    # print('class 1:', class_1.shape)

    class_0_under = class_0.sample(class_count_1)
    new_data = pd.concat([class_0_under, class_1], axis=0)

    # # pre-processed dataset now new_data
    # print("total class of 1 and 0:")
    # print(new_data['Machine failure'].value_counts())

    # take the letters out of the product ID
    new_data['Product ID'] = new_data['Product ID'].str[1:]
    # hot encode the Type (H, M, L)
    dummies = pd.get_dummies(new_data['Type'])
    # drop Type column and attach dummies to new dataset, now called result
    new_data = new_data.loc[:, df.columns != 'Type']
    result = pd.concat([new_data, dummies], axis=1)
    result.to_csv('rawdata.csv', index=False)

    # y: Machine failure
    # X: everything else
    y = result['Machine failure']
    X = result.loc[:, result.columns != 'Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Artificial Neural Network
    mlp = MLPClassifier(max_iter=500)
    param_grid_mlp = {
        'hidden_layer_sizes': [(200,), (226,), (226, 226), (226, 226, 226)],
        'activation': ['identity', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.01, 0.05, 0.10],
        'learning_rate': ['constant', 'adaptive']
    }
    grid_mlp = GridSearchCV(mlp, param_grid_mlp, n_jobs=-1, cv=2)
    grid_mlp.fit(X_train, y_train)
    scores_mlp = cross_val_score(grid_mlp.best_estimator_, X_train, y_train, cv=5, scoring='f1_macro')
    f1_mlp = scores_mlp.mean()

    # Support Vector Machine
    svc = SVC()
    param_grid_svc = {
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=5, n_jobs=-1)
    grid_svc.fit(preprocessing.scale(X_train), y_train)
    scores_svc = cross_val_score(grid_svc.best_estimator_, X_train, y_train, cv=5, scoring='f1_macro')
    f1_svc = scores_svc.mean()

    # Naive Bayes - best was Bernoulli but
    gnb = GaussianNB()
    param_grid_gnb = {
        'var_smoothing': np.logspace(0, -9, num=100)
    }

    grid_gnb = GridSearchCV(estimator=gnb, param_grid=param_grid_gnb, cv=5, n_jobs=-1)
    grid_gnb.fit(X_train, y_train)
    scores_gnb = cross_val_score(grid_gnb.best_estimator_, X_train, y_train, cv=5, scoring='f1_macro')
    f1_gnb = scores_gnb.mean()
    # score: 0.913
    # print(f1_gnb)

    # mnb = MultinomialNB()
    # scores_mnb = cross_val_score(mnb, X_train, y_train, cv=5, scoring='f1_macro')
    # f1_mnb = scores_mnb.mean()
    # print(f1_mnb)

    bnb = BernoulliNB()
    param_grid_bnb = {
        'alpha': [0.5, 1.0],
        'fit_prior': [True, False]
    }

    grid_bnb = GridSearchCV(estimator=bnb, param_grid=param_grid_bnb, cv=5, n_jobs=-1)
    grid_bnb.fit(X_train, y_train)
    scores_bnb = cross_val_score(grid_bnb.best_estimator_, X_train, y_train, cv=5, scoring='f1_macro')
    f1_bnb = scores_bnb.mean()
    # score: 0.983
    # print(f1_bnb)

    # cnb = ComplementNB()
    # scores_cnb = cross_val_score(cnb, X_train, y_train, cv=5, scoring='f1_macro')
    # f1_cnb = scores_cnb.mean()
    # print(f1_cnb)

    ada = AdaBoostClassifier()
    param_grid_ada = {
        'n_estimators': [50, 70, 100, 150],
        'learning_rate': [1.0, 2.0, 5.0, 10.0]
    }

    grid_ada = GridSearchCV(estimator=ada, param_grid=param_grid_ada, cv=5, n_jobs=-1)
    grid_ada.fit(X_train, y_train)
    scores_ada = cross_val_score(grid_ada.best_estimator_, X_train, y_train, cv=5, scoring='f1_macro')
    f1_ada = scores_ada.mean()

    # bootstrap data for the random forest classifier
    bootstrap_data = result.sample(frac=1.0, replace=True)

    y_boot = bootstrap_data['Machine failure']
    X_boot = bootstrap_data.loc[:, bootstrap_data.columns != 'Machine failure']

    X_train_boot, X_test_boot, y_train_boot, y_test_boot = train_test_split(X_boot, y_boot, test_size=0.3,
                                                                            random_state=42)

    forest = RandomForestClassifier()
    param_grid_forest = {
        'n_estimators': [10, 20, 50, 100],
        'criterion': ['gini', 'entropy', 'log_loss'],
    }

    grid_forest = GridSearchCV(estimator=forest, param_grid=param_grid_forest, cv=5, n_jobs=-1)
    grid_forest.fit(X_train_boot, y_train_boot)
    scores_forest = cross_val_score(grid_forest.best_estimator_, X_train_boot, y_train_boot, cv=5, scoring='f1_macro')
    f1_forest = scores_forest.mean()

    # print table
    training_table = [
        ['ML Trained Model', 'Best Set of Parameters', 'F1-score on the 5-fold Cross Validation on Training Data'],
        ['Artificial Neural Networks (MLPClassifier)', grid_mlp.best_params_, f1_mlp],
        ['Support Vector Machine (SVC)', grid_svc.best_params_, f1_svc],
        ['Naive Bayes (BernoulliNB)', grid_bnb.best_params_, f1_bnb],
        ['AdaBoost', grid_ada.best_params_, f1_ada],
        ['Random Forest', grid_forest.best_params_, f1_forest]]

    # testing data
    f1_mlp_test = f1_score(y_test, grid_mlp.predict(X_test), average='macro')
    f1_svc_test = f1_score(y_test, grid_svc.predict(X_test), average='macro')
    f1_bnb_test = f1_score(y_test, grid_bnb.predict(X_test), average='macro')
    f1_ada_test = f1_score(y_test, grid_ada.predict(X_test), average='macro')
    f1_forest_test = f1_score(y_test, grid_forest.predict(X_test), average='macro')

    testing_table = [
        ['ML Trained Model', 'Best Set of Parameters', 'F1-score on the 5-fold Cross Validation on Training Data'],
        ['Artificial Neural Networks (MLPClassifier)', grid_mlp.best_params_, f1_mlp_test],
        ['Support Vector Machine (SVC)', grid_svc.best_params_, f1_svc_test],
        ['Naive Bayes (BernoulliNB)', grid_bnb.best_params_, f1_bnb_test],
        ['AdaBoost', grid_ada.best_params_, f1_ada_test],
        ['Random Forest', grid_forest.best_params_, f1_forest_test]]

    # finds index of best ml model
    f1_scores = []
    for i in range(1, 6, 1):
        f1_scores.append(testing_table[i][2])
    best_index = f1_scores.index(max(f1_scores)) + 1

    print("Training ML Models:")
    print(tabulate(training_table, headers='firstrow', tablefmt='fancy_grid'))
    print("Testing ML Models:")
    print(tabulate(testing_table, headers='firstrow', tablefmt='fancy_grid'))
    print("")
    print("Best ML Model: ", testing_table[best_index][0])


if __name__ == "__main__":
    main()
