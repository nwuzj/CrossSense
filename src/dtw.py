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

import sys

import numpy


def dtw(t,r):
    n=len(t)
    m=len(r)
    d=[]
    for i in range(n):
        d1=[]
        for j in range(m):
            dist1 = numpy.square(t[i]-r[j])
            d1.append(dist1)
        if i==0:
            d = numpy.array(d1)
        else:
            d = numpy.vstack((d, numpy.array(d1)))
    temp_1 = sys.float_info.max
    D = [None]*n
    for i in range(n):
        D[i] = [1]*m
    D = numpy.array(D)
    D = D * temp_1
    D[0,0] = d[0,0]
    for i in range(n-1):
        for j in range(m):
            D1 = D[i,j]
            if j>0:
                D2 = D[i,j-1]
            else:
                D2 = temp_1
            if j>1:
                D3 = D[i,j-2]
            else:
                D3 = temp_1
            D[i+1,j] = d[i+1,j] + min([D1,D2,D3])
    dist = D[n-1,m-1]
    return dist