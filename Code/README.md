## In order to run GPU test, do the following.

  - Make sure you have a Nvidia GPU and appropriate CUDA + cuDNN installed
**CUDA >= 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))

**cuDNN >= 7.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )

  - Install all dependencies via pip
  ```
  pip install -r requirement.txt
  ```
  
  - To switch to the Tensorflow backend, change your `~/.keras/keras.json` file to
  ```
  {"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow", "image_data_format": "channels_first"}
  ```
