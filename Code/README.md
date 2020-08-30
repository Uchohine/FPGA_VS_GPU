## In order to run this test, do the following.

  - Make sure you have a Nvidia GPU and appropriate CUDA + cuDNN installed

  - Install all dependencies via pip
  pip install -r requirement.txt
  
  - To switch to the Tensorflow backend, change your ~/.keras/keras.json file to
  {"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow", "image_data_format": "channels_first"}
