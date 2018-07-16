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
from sklearn.metrics import euclidean_distances

from dtw import dtw
from data_load import data_load_all

def main():
    # input: -Train: Denoised signals of training sets
    #        -Test: Denoised signals of testing sets
    # output: -model_label: the label of experts that the testing sets should use

    # The Function: data_load_all() - load the training sets and testing sets
    Train, Test, person_num= data_load_all()
    # feature_model_sample: the correctly classified samples and its corresponding label of experts
    feature_model_sample = np.load('feature_model_sample.npy')
    # feature_model_label: The label of experts according to the percentages of correctly classified samples
    feature_model_label = np.load('feature_model_label.npy')
    row_vec1 = np.array(feature_model_label)
    feature_model_label = np.array([row_vec1]).T
    each_person_testsample = int(len(Test)/person_num)
    test_label = []
    for k in range(person_num):
        label1 = []
        for j in range(each_person_testsample):
            label1.append(k + 1)
        row_vec = np.array(label1)
        dataLabels = np.array([row_vec]).T
        dataLabels = np.array(dataLabels)
        if k == 0:
            test_label = dataLabels
        else:
            test_label = np.vstack((test_label, dataLabels))
    model_label = []
    for i in range(len(Test)):
        test_data = Test[i]
        test_data = np.array(test_data)
        top_3_all = []
        for p in range(test_data.shape[1]):
            print("p=" + str(p))
            test_subcarrier = test_data[:,p]
            dmins = []
            for j in range(len(feature_model_sample)):
                fm_person_choosenums = feature_model_sample[j]
                fm_person_choosenums = np.array(fm_person_choosenums)
                fm_person_choosenum = fm_person_choosenums[:,0]
                dist = []
                for k in range(len(fm_person_choosenum)):
                    train_data = Train[fm_person_choosenum[k]]
                    train_subcarrier = train_data[:,p]
                    Dist = dtw(train_subcarrier, test_subcarrier)
                    # Dist, D, DD = dtw(train_subcarrier, test_subcarrier)
                    print(Dist)
                    dist.append(Dist)
                dmin = min(dist)
                dmins.append(dmin)
            print(len(dmins))
            print(dmins)
            print("+++++++++++++++++++++")
            row_vec1 = np.array(dmins)
            dmins = np.array([row_vec1]).T
            dist_min = np.hstack((np.array(dmins), feature_model_label))
            dist_min = np.array(dist_min)
            B = dist_min[dist_min[:, 0].argsort()]
            Top_3 = B[0:3,:]
            if p == 0:
                top_3_all = Top_3
            else:
                top_3_all = np.vstack((top_3_all,Top_3))
        count = []
        fm_label = []
        for k in range(len(feature_model_sample)):
            count.append(np.sum( top_3_all[:,1] == k+1 ))
            fm_label.append(k+1)
        row_vec1 = np.array(count)
        count = np.array([row_vec1]).T
        row_vec2 = np.array(fm_label)
        fm_label = np.array([row_vec2]).T
        count = np.hstack((count, fm_label))
        colcount = np.lexsort(-count[:, ::-1].T)
        rcount = count[colcount]
        print(rcount)
        count_num=[]
        count_num.append(rcount[0, 1])
        for k in range(len(rcount)):
            if rcount[k,0] == rcount[k+1,0]:
                count_num.append(rcount[k+1,1])
            else:
                break
        row_vec1 = np.array(count_num)
        count_num = np.array([row_vec1]).T
        print(count_num)
        if len(count_num) > 1:
            right_num = []
            for kk in range(len(count_num)):
                right_num.append(len(feature_model_sample[count_num[kk,0]-1]))
            row_vec1 = np.array(right_num)
            right_num = np.array([row_vec1]).T
            fm_model_num = np.hstack((right_num,count_num))
            colcount1 = np.lexsort(-fm_model_num[:, ::-1].T)
            rcount = fm_model_num[colcount1]
        model_label.append(rcount[0, 1])
        print(model_label)

if __name__ == '__main__':
    main()