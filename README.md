# FPGA vs GPU for machine learning

## What is Machine learning in simple explaination

![](readme/machinelearning.png)

## Computational Power

|           |   CPU   |   GPU   |   FPGA    |
|-----------|---------|---------|-----------|
| Latency   | fastest |  fast   |   fast    |
| Throughput| too low |  high   |   high    |
| Power     | medium  |  high   |   low     |
| Access    | easy    |  medium |   hard    |


## How machine learning framwork interact with hardware?

![](readme/framework.png)

## GPU vs FPGA(DPU as interpreted by xilinx)

### for GPU:
  - matrix interpreted as Tensor(ndarray with specific shape and dtype, dtype would usually be f32)
  - Using CUDA + cuDNN (For Nvidia GPU) or OpenCL (supported only a few AMD or other brands' GPU) to map the model + parameter (weights) + input Tensor to GPU memory.
  - With Tensorflow framwork + CUDA (Nvidia GPU), system can ultilize multiple GPUs for single model prediction, training and pruning.
  - Tons of custom models or concepts developed with GPU, and GPU support all kinds of layers, neuron and algorithms.
  - Wide range of products to choose from, Nvidia GTX series, RTX series and Tesla, also Jet for mobile or embedded design
