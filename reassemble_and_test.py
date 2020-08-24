#1st arg should be layer number

caffe_home = "/home/jinsian/caffe-master"

import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import caffe
plt.rcParams['figure.figsize'] = (15, 15)

folder = os.path.exists("./decompressed_model")
if not folder:
    os.makedirs("./decompressed_model")

data_size = 0
layer_num = sys.argv[1]
if (layer_num == "6"):
    data_size = 4096*9216
    x = 4096
    y = 9216
if (layer_num == "7"):
    data_size = 4096*4096
    x = 4096
    y = 4096
if (layer_num == "8"):
    data_size = 4096*1000
    x = 1000
    y = 4096

caffe.set_mode_cpu()
model_def = './deploy.prototxt'
model_weights = './caffenet_pruned.caffemodel'
accuracy = np.zeros(11, dtype='float32')

for i in range(1, 10):
    net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)
    feat = np.zeros(data_size,dtype='float32')
    data_line = "./data/fc" + str(layer_num) + "-data-" + str(i) + "E-3.dat"
    index_line = "./data/fc" + str(layer_num) + "-index-o.dat"
    data = np.fromfile(data_line, dtype='float32')
    index = np.fromfile(index_line, dtype='uint8')
    k = 0
    q = 0
    for j in range(len(index)):
        if (index[j]==0) and (j <> 0):
            k = k + 255
        else:
            k = k + index[j]
            feat[k] = data[q]
            q = q + 1
    feat = np.reshape(feat, (x, y))
    line = "fc" + str(layer_num)
    net.params[line][0].data[:]=feat
    line = "./data/fc" + str(layer_num) + "-" + str(i) + "E-3.dat"
    feat.astype('float32').tofile(line)


    print("testing...")
    line = "./decompressed_model/AlexNet-" + str(i) + "00.caffemodel"
    net.save(line)
    line = caffe_home + "/build/tools/caffe test -model ./train_val.prototxt -weights ./decompressed_model/AlexNet-" + str(i) + "00.caffemodel -iterations 1000 -gpu 0 &> ./decompressed_model/caffe_test_log.txt"
    os.system(line)
    line = "cat ./decompressed_model/caffe_test_log.txt | grep \"] accuracy =\" > ./decompressed_model/accuracy.txt"
    os.system(line)  
#    log_line = np.fromfile(100, dtype='char')
    accuracy[i] = float(open("./decompressed_model/accuracy.txt").read().split()[6])
line = "./data/fc" + str(layer_num) + "-accuracy.txt"
accuracy.astype('float32').tofile(line)

