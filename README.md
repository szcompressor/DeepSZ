# DeepSZ

## About DeepSZ

DeepSZ is an accuracy-loss expected neural network compression framework, which involves four key steps: network pruning, error bound assessment, optimization for error bound configuration, and compressed model generation, featuring a high compression ratio and low encoding time. The paper is available at: https://dl.acm.org/doi/10.1145/3307681.3326608.

This repo is an implementation of DeepSZ based on Caffe deep learning framework [1] and SZ lossy compressor [2]. Below is the instruction to run DeepSZ on AlexNet using [TACC Frontera system](https://www.tacc.utexas.edu/systems/frontera), which can be adapted to other DNN models (such as VGG-16) and HPC systems with some modifications to the code and scripts (mainly for the network architecture information). 

## Prerequisites
```
Anaconda 3 with Python 3.7
Caffe 1.0
NVIDIA CUDA 10.0
GCC 6.3 or 7.3
ImageNet validation dataset
```

## Install Caffe/PyCaffe (via Anaconda)
- Download and install Anaconda:
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

- Create conda new environment and install dependencies:
```
conda create -n deepsz_env
conda activate deepsz_env
conda install protobuf glog gflags hdf5 openblas boost snappy leveldb lmdb pkgconfig zlib opencv cudnn
```
Note that "conda install cudnn" will automatically install another cudatoolkit (different from your system default one), so you can use "conda install cudatoolkit=10.1" to make sure the two version are consistent; otherwise, runtime will report an error. For example, in TACC Frontera, you can use "module load cuda/10.1" and "conda install cudnn cudatoolkit=10.1". 

- Download DeepSZ:
```
git clone https://github.com/szcompressor/DeepSZ.git
```

- Download Caffe/PyCaffe:
```
git clone https://github.com/BVLC/caffe.git
cp DeepSZ/Makefile.config caffe/
cd caffe
```

- Modify dependencies’ paths in Makefile.config, including CUDA_DIR (line 30), CUDA_ARCH (line 39), PYTHON_LIBRARIES (line 78), PYTHON_INCLUDE (line 79-80), INCLUDE_DIRS (line 94), LIBRARY_DIRS (line 95), USE_NCCL (line 103).

- Note if you are using Python 3.7, please change "boost_python3" to "boost_python37" (line 57) of Makefile.config.

- Note that if your system is running with NVIDIA Tesla GPUs (e.g., V100) or higher generations of NVIDIA GPUs (e.g., RTX 5000), please comment out GPU arch lower than compute_52 and sm_52. CUDA_ARCH should look like below:
```
CUDA_ARCH := -gencode arch=compute_52,code=sm_52 \
                -gencode arch=compute_60,code=sm_60 \ 
                -gencode arch=compute_61,code=sm_61 \
                -gencode arch=compute_70,code=sm_70 \ 
                -gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_75,code=compute_75 
```

- Note that if you wan to enable NCCL (USE_NCCL := 1), please add your NCCL path to INCLUDE_DIRS and LIBRARY_DIRS in Makefile.config and LD_LIBRARY_PATH. 

- Compile and Install Caffe/PyCaffe:
```
make all -j 4
make pycaffe
```

- Install PyCaffe's dependency:
```
pip install scikit-image
```

## Test PyCaffe
- Please add your PyCaffe path into your PYTHONPATH, e.g.:
```
export PYTHONPATH=$PYTHONPATH:/home1/06128/dtao/caffe/python
```

- Then, please try “import caffe” in Python; if no error is reported, PyCaffe is working!!

## Download Validation Dataset and DNN Model
- Please download AlexNet model into DeepSZ root directory:
```
wget https://eecs.wsu.edu/~dtao/deepsz/caffenet_pruned.caffemodel
```

- Please download ImageNet validation data and put them to e.g. /work/06128/dtao/frontera/caffedata:
```
wget https://eecs.wsu.edu/~dtao/deepsz/imagenet_mean.binaryproto
wget https://eecs.wsu.edu/~dtao/deepsz/ilsvrc12_val_lmdb.tar.gz
tar -xzvf ilsvrc12_val_lmdb.tar.gz
````

## Run DeepSZ

- After installing PyCaffe, please go to DeepSZ’s root directory and modify the first lines of "launch.sh", "reassemble_and_test.py", and "optimize.py" to your Caffe (e.g., /home1/06128/dtao/caffe) and PyCaffe directories (e.g., /home1/06128/dtao/caffe/python).

- Then, please change line 40 and line 51 of “train_val.prototxt” to match your imagenet_mean.binaryproto file location (e.g., /work/06128/dtao/frontera/caffedata) and your ImageNet validation data file location (e.g., /work/06128/dtao/frontera/caffedata/ilsvrc12_val_lmdb).

- Finally, please use the below command to run DeepSZ to compress the network and test the accuracy with the decompressed model:
```
bash ./launch.sh
```

- Note that the script will automatically download and compile SZ lossy compression software. 

[1] Jia, Yangqing, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the 22nd ACM international conference on Multimedia, pp. 675-678. 2014.

[2] Tao, Dingwen, Sheng Di, Zizhong Chen, and Franck Cappello. "Significantly improving lossy compression for scientific data sets based on multidimensional prediction and error-controlled quantization." In 2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 1129-1139. IEEE, 2017. 
