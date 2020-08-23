import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import caffe
plt.rcParams['figure.figsize'] = (15, 15)
import os

caffe.set_mode_cpu()
model_def = './deploy.prototxt'
model_weights = './caffenet_pruned.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)

print "Initialization done!"
print "Start converting weights into sparse representation ..."

folder = os.path.exists("./data")
if not folder:
    os.makedirs("./data")

feat = net.params['fc6'][0].data
feat = np.reshape(feat, 4096*9216)

k = len(feat)
data = np.zeros(k, dtype='float32')
index = np.zeros(k, dtype='uint8')

k = 0
kk = 0
bit = 0

for i in range(len(feat)):
    if (bit == 255) and (feat[i] == 0):
        index[k] = 0
        k = k+1
        bit = 0
    if (feat[i] <> 0):
        data[kk] = feat[i]
        index[k] = bit
        k = k+1
        kk = kk+1
        bit = 0
    bit = bit + 1

a = np.zeros(kk, dtype='float32')
b = np.zeros(k, dtype='uint8')
for i in range(kk):
    a[i] = data[i]
for i in range(k):
    b[i] = index[i]

a.astype('float32').tofile('./data/fc6-data-o.dat')
b.astype('uint8').tofile('./data/fc6-index-o.dat')

print "fc6 transferred"
print "start compressing and decompressing ..."

bash_line = "python ./bash_script.py " + str(kk) + " 6"
os.system(bash_line)
bash_line = "source ./SZ_compress_script/fc6_script.sh"
os.system(bash_line)

print"fc6 decompression finished"

feat = net.params['fc7'][0].data
feat = np.reshape(feat, 4096*4096)

k = len(feat)
data = np.zeros(k, dtype='float32')
index = np.zeros(k, dtype='uint8')

k = 0
kk = 0
bit = 0

for i in range(len(feat)):
    if (bit == 255) and (feat[i] == 0):
        index[k] = 0
        k = k+1
        bit = 0
    if (feat[i] <> 0):
        data[kk] = feat[i]
        index[k] = bit
        k = k+1
        kk = kk+1
        bit = 0
    bit = bit + 1

a = np.zeros(kk, dtype='float32')
b = np.zeros(k, dtype='uint8')
for i in range(kk):
    a[i] = data[i]
for i in range(k):
    b[i] = index[i]

a.astype('float32').tofile('./data/fc7-data-o.dat')
b.astype('uint8').tofile('./data/fc7-index-o.dat')

print "fc7 transferred"
print "start compressing and decompressing ..."
                                 
bash_line = "python ./bash_script.py " + str(kk) + " 7"
os.system(bash_line)             
bash_line = "source ./SZ_compress_script/fc7_script.sh"                                                                       
os.system(bash_line)
                                                                                                        
print"fc7 decompression finished"


feat = net.params['fc8'][0].data
feat = np.reshape(feat, 1000*4096)

k = len(feat)
data = np.zeros(k, dtype='float32')
index = np.zeros(k, dtype='uint8')

k = 0
kk = 0
bit = 0

for i in range(len(feat)):
    if (bit == 255) and (feat[i] == 0):
        index[k] = 0
        k = k+1
        bit = 0
    if (feat[i] <> 0):
        data[kk] = feat[i]
        index[k] = bit
        k = k+1
        kk = kk+1
        bit = 0
    bit = bit + 1

a = np.zeros(kk, dtype='float32')
b = np.zeros(k, dtype='uint8')
for i in range(kk):
    a[i] = data[i]
for i in range(k):
    b[i] = index[i]

a.astype('float32').tofile('./data/fc8-data-o.dat')
b.astype('uint8').tofile('./data/fc8-index-o.dat')

print "fc8 transferred"
print "start compressing and decompressing ..."
                                 
bash_line = "python ./bash_script.py " + str(kk) + " 8"
os.system(bash_line)             
bash_line = "source ./SZ_compress_script/fc8_script.sh"                                                                       
os.system(bash_line)
                                                                                                                              
print"fc8 decompression finished"


