<div align="center">
  <h3> python3 diffmpm.py </h3>
  <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm80.gif">
</div>        


# DiffTaichi: Differentiable Programming for Physical Simulation (ICLR 2020)

*Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, Frédo Durand*

[[Paper]](https://arxiv.org/abs/1910.00935) [[Video] (with instructions to reproduce every demo)](https://www.youtube.com/watch?v=Z1xvAZve9aE)

### Differentiable programming in Taichi allows you to optimize neural network controllers efficiently with brute-force gradient descent, instead of using reinforcement learning.

The *DiffTaichi* differentiable programming framework is now officially part of [Taichi](https://github.com/yuanming-hu/taichi). This repo only contains examples.

DiffTaichi significantly boosts the performance and productivity of differentiable physical simulators. For example, the differentiable elastic object simulator (ChainQueen) in DiffTaichi is 188x faster than an implementation in TensorFlow. The DiffTaichi version also runs as fast as the CUDA implementation, with the code being 4.2x shorter.

Most of the 10 differentiable simulators can be implemented **within 2-3 hours**.

Questions regarding the simulators/autodiff compiler go to Yuanming Hu (yuanming __at__ mit.edu) or [Issues](https://github.com/yuanming-hu/difftaichi/issues).

### Note: please make sure you are using Taichi >= v0.6.7

## How to run
Step 1: Install [`Taichi`](https://github.com/taichi-dev/taichi) with `pip`:

(Most examples do **not** need a GPU to run.)
```bash
python3 -m pip install taichi
```
Step 2: Run example scripts in the `examples` folder: (Please wait for all GIFs to load :-)


### Differentiable Elastic Object Simulator [`python3 diffmpm.py`]
Gradient descent iteration 0 and gradient descent iteration 80: 

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm00.gif"> <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm80.gif">

### Differentiable 3D Elastic Object Simulator [`python3 diffmpm3d.py`]
Gradient descent iteration 40: 

<img width="800px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm3d.gif">

### Differentiable 3D Fliud Simulator [`python3 liquid.py`]
Gradient descent iteration 450: 

<img width="800px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/liquid.gif">

### Differentiable Height Field Water Simulator [`python3 wave.py`]
Gradient descent iteration 180:

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/wave.gif">

### Differentiable (Adversarial) Water Renderer [`python3 water_renderer.py`]
Differentiable water simulation + differentiable water rendering + (differentiable) CNN

**Optimization goal:** find an initial water height field, so that after simulation and shading, VGG16 thinks the squirrel image is a goldfish. Input image: VGG16=fox squirrel (42.21%)

<img width="800px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/three-stage.jpg">

**Left:** center activation .  **Right:** An activation that fools VGG (VGG16=goldfish (99.91%))

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/water_wave_center.gif"><img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/water_wave_iter10.gif">


### Differentiable Rigid Body Simulator [`python3 rigid_body.py [1/2] train`]
2048 time steps. Gardient descent iteration 20: 

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/rb_final1.gif"> <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/rb_final2.gif">

### Differentiable Mass-Spring Simulator [`python3 mass_spring.py [1/2/3] train`]
682 time steps.
Gardient descent iteration 20: 

<img width="266px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms1_final-cropped.gif">  <img width="266px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms2_final-cropped.gif">  <img width="266px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms3_final-cropped.gif"> 

### Differentiable Billiard Simulator [`python3 billiards.py`]
Gardient descent iteration 0 and gradient descent iteration 100: 

<img width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/billiard0000.gif"> <img  width="400px" src="https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/billiard0100.gif">

See the video for the remaining two simulators.

## Bibtex

```
@article{hu2019difftaichi,
  title={DiffTaichi: Differentiable Programming for Physical Simulation},
  author={Hu, Yuanming and Anderson, Luke and Li, Tzu-Mao and Sun, Qi and Carr, Nathan and Ragan-Kelley, Jonathan and Durand, Fr{\'e}do},
  journal={arXiv preprint arXiv:1910.00935},
  year={2019}
}
```
