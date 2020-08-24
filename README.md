# DeepSZ

This is a draft implementation of DeepSZ, paper available at:
https://arxiv.org/abs/1901.09124

Essential tools needed: Caffe, PyCaffe, ImageNet validation dataset and Python.
GPU version of Caffe is recommended

This instruction is for DeepSZ implementation on AlexNet. Can be adopt to on VGG-16 and LeNet with some modifications to the code.

########################################################

To start, please install Caffe following this instruction:
https://caffe.berkeleyvision.org/installation.html

Makefile.config needs to be modified to fit your system. Note that for machine running CUDA 10 with V100 GPUs, you also need to comment GPU arch lower than 52. Should looks something like:

CUDA_ARCH := -gencode arch=compute_52,code=sm_52 \
                -gencode arch=compute_60,code=sm_60 \ 
                -gencode arch=compute_61,code=sm_61 \
                -gencode arch=compute_70,code=sm_70 \ 
                -gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_75,code=compute_75 

After building Caffe, PyCaffe is also required. Installed by "make pycaffe"
Note you may need to change "boost_python3" to "boost_python37" if you are using python 3.7 in line 57 of Makefile.config

########################################################

After git clone DeepSZ, you will need to modify reassemble_and_test.py and optimize.py in their first line to address your caffe location.

Then please change train_val.prototxt line 40 and line 51 to match your imagenet_mean.binaryproto and your ImageNet validation file location.

Finally, you can launch DeepSZ with command "source ./launch.sh"
