## In order to run GPU test, do the following.

  - Make sure you have a Nvidia GPU and appropriate CUDA + cuDNN installed
* **CUDA >= 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))

* **cuDNN >= 7.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )

  * Install all dependencies via pip
  ```
  pip install -r requirement.txt
  ```
  
  * To switch to the Tensorflow backend, change your `~/.keras/keras.json` file to
  ```
  {"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow", "image_data_format": "channels_first"}
  ```
  
  * Get ImageNet 2012 and put them in `/images`. 
  
  * Then in 'FPGA_VS_GPU/Code' run:
  ```
  python main.py
  ```
  or
  ```
  python3 main.py
  ```
  depends on your python version
  
## On Aliyun ecs.gn6v-c8g1.2xlarge ECS, 8 vCPU(Intel Xeon(Skylake) Platinum 8163) + 1 * Nvidia Tesla V100 + 32GiB RAM, with Ubuntu 20.04, CUDA==11.0, cuDNN==7.6.5 + 8.0.1 rc2:
### GPU usage on run:
```
Sun Aug 30 20:48:42 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:07.0 Off |                    0 |
| N/A   38C    P0    60W / 300W |  15653MiB / 16160MiB |     27%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     25226      C   python3                         15651MiB |
+-----------------------------------------------------------------------------+
```
### run log:
```
2020-08-30 20:47:53.211532: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-30 20:47:54.701173: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2020-08-30 20:47:54.740872: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:54.741557: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:00:07.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
2020-08-30 20:47:54.741589: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-30 20:47:54.743440: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-08-30 20:47:54.745278: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2020-08-30 20:47:54.745640: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2020-08-30 20:47:54.747447: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2020-08-30 20:47:54.748294: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2020-08-30 20:47:54.752928: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2020-08-30 20:47:54.753051: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:54.753744: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:54.754349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2020-08-30 20:47:55.542947: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-08-30 20:47:55.550848: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2499980000 Hz
2020-08-30 20:47:55.551261: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x83343b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-30 20:47:55.551281: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-08-30 20:47:55.772556: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:55.773308: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x83c89a0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-08-30 20:47:55.773331: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
2020-08-30 20:47:55.773575: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:55.774226: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:00:07.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
2020-08-30 20:47:55.774265: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-30 20:47:55.774294: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-08-30 20:47:55.774305: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2020-08-30 20:47:55.774317: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2020-08-30 20:47:55.774328: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2020-08-30 20:47:55.774339: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2020-08-30 20:47:55.774351: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2020-08-30 20:47:55.774421: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:55.775078: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:55.775680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2020-08-30 20:47:55.775718: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-08-30 20:47:56.342348: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-08-30 20:47:56.342402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2020-08-30 20:47:56.342413: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2020-08-30 20:47:56.342672: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:56.343374: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-08-30 20:47:56.344010: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14764 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:07.0, compute capability: 7.0)
WARNING:tensorflow:From main.py:183: The name tf.keras.backend.get_session is deprecated. Please use tf.compat.v1.keras.backend.get_session instead.

WARNING:tensorflow:From /home/mingwen/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training_v1.py:2070: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
2020-08-30 20:48:01.413399: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-08-30 20:48:01.657705: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
Predicted Class:  282 , Class Name:  n02123159 tiger cat
/home/mingwen/Documents/GoogLeNet/images
```
### Result(Only prediction time is calculated, preprocess time is not included):
```
Total image read:  5000 Average Process Time:  6.3156261444091795 ms/image, Average image/s:  158.4
```

  
## To run benchmark on FPGA, do the following:

  - Get a Alveo U200/U250 for your test device or have AWS ec2.f1.2xlarge running with ml-suite AMI.
  
  - follow the instruction here:
  
  * https://github.com/Xilinx/ml-suite/blob/master/examples/tensorflow/README.md
