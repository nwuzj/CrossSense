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
import h5py

def feature_model_choose(data, each_person_sample):
    # input :-data: recognition results of the expert selector training sets
    #        -each_person_sample: The number of samples per user
    # output: choose_sample: the correctly classified samples and its corresponding label of experts
    m=data.shape[0]
    person_num=int(m/each_person_sample)
    label = []
    for k in range(person_num):
        label1 = []
        for j in range(each_person_sample):
            label1.append(k+1)
        row_vec = np.array(label1)
        dataLabels = np.array([row_vec]).T
        dataLabels = np.array(dataLabels)
        if k == 0:
            label = dataLabels
        else:
            label = np.vstack((label, dataLabels))
    choose_model_sample = []
    choose_sample_row = []
    for k in range(m):
        if int(data[k]) == label[k,0]:
            choose_model_sample.append(1)
            choose_sample_row.append(k)
        else:
            choose_model_sample.append(0)
    row_vec1 = np.array(choose_sample_row)
    choose_sample = np.array([row_vec1]).T
    return np.array(choose_sample)

def main():
    # input : -data: recognition results of the expert selector training sets
    # output: -feature_model_sample: the correctly classified samples
    #         -feature_model_label: The label of experts
    # feature_name: Four kinds of wireless signal features used
    feature_name = ["Stat", "Comp","Spec","Tran"]
    # model_name: Six classification techniques used
    model_name = ["NB", "RF", "SVM", "LinearSVM","KNN","Adaboost"]
    # pathname: modified according to your path
    pathname = "./scene1/"
    # each_person_sample: The number of samples per user in training sets
    each_person_sample=10
    # feature_model_num: Record the number of experts
    feature_model_num=0
    feature_model_sample = []
    for i in range(len(feature_name)):
        for j in range(len(model_name)):
            f_feature = feature_name[i]
            m_model = model_name[j]
            # the accuracy is tested on expert selector training sets, it contains the predicted labels
            filename = pathname + "accuracy_" + m_model + "_" + f_feature + ".mat"
            print(filename)
            f = h5py.File(filename)
            predict = np.transpose(f['predict'])
            data = predict[:,0]
            data = np.array(data)
            choose_sample = feature_model_choose(data, each_person_sample)
            feature_model_sample.append(choose_sample)
            print(feature_model_num)
            feature_model_num = feature_model_num + 1
    feature_model_label = []
    for i in range(feature_model_num):
        feature_model_label.append(i+1)
        print(feature_model_sample[i].shape)
        print(feature_model_label[i])
    np.save('feature_model_sample.npy',feature_model_sample)
    np.save('feature_model_label.npy', feature_model_label)

if __name__ == '__main__':
    main()