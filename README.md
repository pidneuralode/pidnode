# PID Neural Ordinary Differential Equations

This is the official implementation of PID Neural Ordinary Differential Equations. 



## Main requirements
Before you run the code, the following packages are required:
- torch
- torchvision
- torchdiffeq
- tqdm
- imageio
- einops
These packages can be installed with the following command:
```
pip install torch torchvision torchdiffeq tqdm imageio einops
```
To simplify operations by reducing the number of core packages installed, we have pre-installed torchdiffeq.(The details information about torchdiffeq can be seen in https://github.com/rtqichen/torchdiffeq.)



The work flow of our proposed PIDNODE can be shown at the picture below.

![](/Users/wangpengkai/Desktop/PIDNODE-main/pidnode.PNG)