
## description

This code optimizes the BPTT (back propagation through time) based SNN training on GPU. We also include the distributed training with multi-GPUs. 

## enviroment
pytorch 1.8.1

CUDA 10.2


## setup
```
cd ./snnlib
bash install.sh
```

## interface
The CUDA codes locate in ./snnlib/cuda

The forward and backward of each SNN layer are implemented in [snnlib_func.py](https://github.com/liangling76/snn_gpu_training_bptt/blob/main/snnlib/example/snnlib_func.py)

The pytorch interfaces are provided in [snnlib_op.py](https://github.com/liangling76/snn_gpu_training_bptt/blob/main/snnlib/example/snnlib_op.py)

Users can build any SNN models based on the functions provided in [snnlib_op.py](https://github.com/liangling76/snn_gpu_training_bptt/blob/main/snnlib/example/snnlib_op.py)

## examples single functions
We demonstrate the correctness of our code by comparing the results of our framework with the original pytorch imaplementaion for each single function. The test files locate in ./snnlib/examples/test*


## training on CIFAR
We provide an end-to-end example on [CIFAR](https://github.com/liangling76/snn_gpu_training_bptt/blob/main/snnlib/example/train_test_cifar.py)

