# -*- coding: utf-8 -*-

""" Convolutional Neural Network for XPS dataset classification task.
Authers: Xing Wang (xingwang1991@gmail.com)
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d, conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from numpy import random
import numpy as np

import matplotlib.pyplot as plt
from Generate_databases import spectrum

#---------------------------------------------
# generate training and eval data
nums = 10000
numx = 200
spectrums = []
amp=random.uniform(2000, 6000, nums)
cen=random.uniform(1020, 1180, nums)
wid=random.uniform(5, 50, nums)
npeaks = 4
peak = random.randint(1, (npeaks + 1), nums)
for i in range(nums-npeaks):
    xps = spectrum(numx)
    for j in range(peak[i]):
        xps.add_gaussian(amp[i + j], cen[i + j], wid[i + j], noise = 0.005)
    xps.add_backg([1, 5, 0.05])
    spectrums.append(xps)
#------------------------------------------------
trainx = []
trainy = []
testx = []
testy = []
for i in range(nums-npeaks-int(nums*0.1)):
  trainx.append(spectrums[i].y['gaussian'])
  lab = [0]*npeaks
  lab[spectrums[i].npeaks - 1] = 1
  trainy.append(lab)
for i in range(nums-npeaks-int(nums*0.001), nums-npeaks):
  testx.append(spectrums[i].y['gaussian'])
  lab = [0]*npeaks
  lab[spectrums[i].npeaks - 1] = 1
  testy.append(lab)
  spectrums[i].savefig('figs/{0}_{1}.jpg'.format(i, spectrums[i].npeaks))
  
#----------------------------------------------------
trainx = np.array(trainx)
trainy = np.array(trainy)
testx = np.array(testx)
testy = np.array(testy)
trainx = trainx.reshape([-1, numx, 1])

testx = testx.reshape([-1, numx, 1])
print(testx.shape)
# Building convolutional network
network = input_data(shape=[None, numx, 1], name='input')
network = conv_1d(network, 16, 3, activation='relu')
network = max_pool_1d(network, 2)
network = batch_normalization(network)
network = conv_1d(network, 32, 3, activation='relu')
network = max_pool_1d(network, 2)
network = conv_1d(network, 64, 3, activation='relu')
network = max_pool_1d(network, 2)
network = batch_normalization(network)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.8)
# network = fully_connected(network, 128, activation='relu')
# network = dropout(network, 0.5)
network = fully_connected(network, npeaks, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': trainx}, {'target': trainy}, n_epoch=20,
           validation_set=({'input': testx}, {'target': testy}),
snapshot_step=100, show_metric=True, run_id='convnet_xps')
# Save
# Manually save model
model.save("modeldata/model.tfl")

# Load a model
# model.load("model.tfl")
usex = testx
network = dropout(network, 1)
results = model.predict(usex)
print(results)