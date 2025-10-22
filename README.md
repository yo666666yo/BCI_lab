# BCI_lab
We are a research team from [the School of Computer Science and Engineering, Software Engineering, and Artificial Intelligence at Southeast University](https://cse.seu.edu.cn/), Nanjing, China. We mainly focus on EEG-BCI signal processing and decoding algorithms, and this repository contains most of our preparation works.

The folder `EEGNets` contains implementations of several classical network architectures [1, 2, 3, 4, 5] using [PyTorch](https://github.com/pytorch/pytorch), notice that `EEG_residual.py` is modified based on `EEG_deep.py` [3], and `MultiDecoderEEG.py` is a mixed decoding module based on `EEG_residual.py` and `EEG_TCNet.py`[4]. `train_P300.py` and others provide the training and validation framework, using `P300` and `BCI Competition IV 2a` datasets provided by [MOABB](https://github.com/NeuroTechX/moabb). The folder `results_imgs` contains train & test results and data visualizations, and `example_usage` is a example taht shows how to call a EEGNet in a python script.

Most of our works were conducted on [Google Colab](https://colab.research.google.com), where provides a built-in Jupyter Notebook environment. The file `EEG_TCNet.ipynb` shows a sample of actual outputs while training.

References:  
[1] EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces  
[2] EEG-based brain-computer interface enables real-time robotic hand control at individual finger level  
[3] Deep learning with convolutional neural networks for EEG decoding and visualization  
[4] EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain–Machine Interfaces  
[5] ViT-Based EEG Analysis Method for Auditory Attention Detection
