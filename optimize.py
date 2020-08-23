caffe_home = "/home/jinsian/caffe-master"

import numpy as np
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import caffe
plt.rcParams['figure.figsize'] = (15, 15)

o_accuracy = 0.56   #original accuracy
e_accuracy = 0.005  #acceptable accuracy loss
max_test = 9
num_layers = 3
delta = np.zeros((num_layers+1, max_test), dtype='float32')
ratio = np.zeros((num_layers+1, max_test), dtype='float32')

accuracy = np.fromfile("./data/fc6-accuracy.txt", dtype='float32')
for i in range(max_test):
    delta[0][i] = o_accuracy-accuracy[i+1]
accuracy = np.fromfile("./data/fc7-accuracy.txt", dtype='float32')
for i in range(max_test):
    delta[1][i] = o_accuracy-accuracy[i+1] 
accuracy = np.fromfile("./data/fc8-accuracy.txt", dtype='float32')
for i in range(max_test):       
    delta[2][i] = o_accuracy-accuracy[i+1] 

delta[0] = (-0.002,0,0.001,0.001,0.002,0.004,0.003,0.005,0.009)

read_line = open("./data/compression_ratios_fc6.txt").read().split()
for i in range(max_test):
    ratio[0][i] = float(read_line[i*3+2])
read_line = open("./data/compression_ratios_fc7.txt").read().split()
for i in range(max_test):
    ratio[1][i] = float(read_line[i*3+2])
read_line = open("./data/compression_ratios_fc8.txt").read().split()
for i in range(max_test):
    ratio[2][i] = float(read_line[i*3+2])

min_size = ratio[0][0] + ratio[1][0] + ratio[2][0]
eb_fc6 = 0
eb_fc7 = 0
eb_fc8 = 0

for i in range(max_test):
    for j in range(max_test):
        for k in range(max_test):
            if (delta[0][i] + delta[1][j] + delta[2][k] <= e_accuracy) and (ratio[0][i] + ratio[1][j] + ratio[2][k] < min_size):
                min_size = ratio[0][i] + ratio[1][j] + ratio[2][k]
                eb_fc6 = i
                eb_fc7 = j
                eb_fc8 = k
'''
t_size = np.zeros((num_layers+1, int(e_accuracy*10000*2)), dtype='float32')
for i in range(num_layers):
    for j in range(max_test):
        for k in range(int(e_accuracy*10000)):
            if (int(k-(delta[i,j]*10000)) >= 0):
                if (t_size[i+1, k] == 0):
                    t_size[i+1,k] = ratio[i,j] + t_size[i, int(k-(delta[i,j]*10000))]
                if (t_size[i+1,k] > ratio[i,j] + t_size[i, int(k-(delta[i,j]*10000))]):
                    t_size[i+1,k] = ratio[i,j] + t_size[i, int(k-(delta[i,j]*10000))]
'''
               
print(delta)
print(ratio) 


caffe.set_mode_cpu()
model_def = './deploy.prototxt'
model_weights = './caffenet_pruned.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)
line = "./data/fc6-" + str(eb_fc6+1) + "E-3.dat"
feat=np.fromfile(line, dtype='float32')
feat = np.reshape(feat, (4096, 9216))
net.params['fc6'][0].data[:]=feat
line = "./data/fc7-" + str(eb_fc7+1) + "E-3.dat"
feat=np.fromfile(line, dtype='float32')
feat = np.reshape(feat, (4096, 4096)) 
net.params['fc7'][0].data[:]=feat
line = "./data/fc8-" + str(eb_fc8+1) + "E-3.dat"
feat=np.fromfile(line, dtype='float32') 
feat = np.reshape(feat, (1000, 4096))
net.params['fc8'][0].data[:]=feat

line = "./decompressed_model/AlexNet-" + str(eb_fc6+1) + str(eb_fc7+1) + str(eb_fc8+1) + ".caffemodel"
net.save(line)

line_exc = caffe_home + "/build/tools/caffe test -model ./train_val.prototxt -weights " + line + " -iterations 4 &> ./decompressed_model/caffe_test_log.txt"
os.system(line_exc)
line = "cat ./decompressed_model/caffe_test_log.txt | grep \"] accuracy =\" > ./decompressed_model/accuracy.txt"
os.system(line_exc)
accuracy = float(open("./decompressed_model/accuracy.txt").read().split()[6])

print(min_size, eb_fc6+1, eb_fc7+1, eb_fc8+1)
print("final optimzation done, accuracy after reconstruct is:")
print(accuracy)


