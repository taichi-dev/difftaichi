# 10 Differentiable Physical Simulators in the DiffTaichi paper
### Differentiable programming in Taichi allows you to optimize neural network controllers efficiently with brute-force gradient descent, instead of reinforcement learning.

(This Page is WIP)

[[Paper]](https://arxiv.org/abs/1910.00935) [[Video] (with instructions for reproducing every demo)](https://www.youtube.com/watch?v=Z1xvAZve9aE)

Questions to go to Yuanming Hu (yuanming __at__ mit.edu).

### Differentiable Elastic Object Simulator [`python3 diffmpm.py`]
Gardient descent iteration 0 and gradient descent iteration 80: 

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm00.gif"> <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm80.gif">

### Differentiable 3D Elastic Object Simulator [`python3 diffmpm3d.py`]
Gradient descent iteration 40: 

<img width="800px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm3d.gif">

### Differentiable Rigid Body Simulator [`python3 rigid_body.py [1/2] train`]
2048 time steps. Gardient descent iteration 0 and gradient descent iteration 20: 

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/rb_final1.gif"> <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/rb_final2.gif">

### Differentiable Mass-Spring Simulator [`python3 mass_spring.py [1/2/3] train`]
682 time steps.
Gardient descent iteration 0 and gradient descent iteration 20: 

<img width="266px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms1_final-cropped.gif">  <img width="266px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms2_final-cropped.gif">  <img width="266px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms3_final-cropped.gif"> 

### Differentiable Billiard Simulator [`python3 billiards.py`]
Gardient descent iteration 0 and gradient descent iteration 100: 

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/billiard0000.gif"> <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/billiard0100.gif">

### GIFs of other 5 simulators to be added... See the video for all simulators.
