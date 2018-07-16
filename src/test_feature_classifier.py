# Copyright 2018 Northwest University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def main():
    # input: -train_data: Expert selector training sets
    #        -train_label: Training label of expert
    #        -test_data：Expert selector testing sets, each row contains one kind of feature
    #        -model_label: the label of selected experts for testing sets
    # output: -predict_label: the predicted label, which person

    # x_train, y _train: the training set and the person label of the training sets
    x_train = np.load('train_data.npy')
    y_train = np.load('train_label.npy')

    # testing sets, each row contains one kind of feature
    Test = np.load('test_data.npy')
    model_label = np.load('model_label.npy')
    # feature_name: Four kinds of wireless signal features used
    feature_name = ["Stat", "Comp", "Spec", "Tran"]
    # model_name: Six classification techniques used
    model_name = ["NB", "RF", "SVM", "LinearSVM", "KNN", "Adaboost"]
    predict_label = []
    for i in range(len(Test.shape)):
        data_te = Test[i]
        label_model = model_label[i]
        classifier_num = len(model_name)
        div = label_model // classifier_num
        mod = label_model % classifier_num

        # x_test: the feature the testing sets should use
        x_test = data_te[div]

        # Normalization of training set and test set
        scaler = StandardScaler()
        X = scaler.fit_transform(x_train)
        Y = scaler.transform(x_test)

        # the classification techniques the testing sets should use
        if mod == 0:
            #   Naive Bayes Classifier
            clf = GaussianNB(priors=None)
        elif mod == 1:
            # Random Forests Classifier
            clf = RandomForestClassifier(max_depth=50, random_state=0)
        elif mod == 2:
            #SVM Classifier
            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(X, y_train)
            print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))
            bestc = grid.best_params_['C']
            bestgamma = grid.best_params_['gamma']
            clf = svm.SVC(C=bestc, kernel='rbf', gamma=bestgamma, decision_function_shape='ovr')
        elif mod == 3:
            # LinearSVM Classifier
            clf = svm.LinearSVC()
        elif mod == 4:
            # Neighbours Classifier
            n_neighbors = 3
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        else:
            # AdaBoost Classifier
            clf = AdaBoostClassifier()
        clf.fit(X, y_train.ravel())
        # predict the testing sets
        # y_hat: the predicted person label
        y_hat = clf.predict(Y)
        print('The predicted label：', y_hat)
        predict_label.append(y_hat)

if __name__ == '__main__':
    main()