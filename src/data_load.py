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

import h5py

import numpy as np

# data_load_all(): load the training sets and testing sets
def data_load_all():
    f = h5py.File('gait_100.mat')
    # data: Denoised signals
    data = np.transpose(f['data_all_1'])
    nums = np.transpose(f['num_idxs'])
    # num_idxs: The training sets and testing sets are chosen at random
    num_idxs = np.transpose(f[nums[0, 0]])
    print(num_idxs.shape)
    # person_num: The number of participants
    person_num = 100
    # each_person_sample: The number of samples per user
    each_person_sample = 20
    Train = []
    Test = []
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for i in range(person_num):
        num_idx1 = num_idxs[i,:]
        num_idx2 = [p for p in num if p not in num_idx1]
        for j in range(int(each_person_sample/2)):
            Train.append(np.transpose(f[data[i, int(num_idx1[j])-1]]))
            Test.append(np.transpose(f[data[i, int(num_idx2[j])-1]]))
    return Train,Test,person_num