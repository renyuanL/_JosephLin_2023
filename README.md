# _JosephLin_2023

2023.03.25:
## Neural Network from scratch in Python

- https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

2023.04.08:
## Back-propagation in CNN
- https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

![](https://www.jefkine.com/assets/images/conv.png)

% <![CDATA[
\begin{align}
\frac{\partial E}{\partial x_{i',j'}^{l}} &= \sum_{m = 0}^{k_1 - 1} \sum_{n = 0}^{k_2 - 1} \delta^{l+1}_{i^{\prime} - m,j^{\prime} - n} w_{m,n}^{l+1} f'\left(x_{i',j'}^{l}\right) \\
& = \text{rot}_{180^\circ} \left\{ \sum_{m = 0}^{k_1 - 1} \sum_{n = 0}^{k_2 - 1} \delta^{l+1}_{i^{\prime} + m,j^{\prime} + n} w_{m,n}^{l+1} \right\} f'\left(x_{i',j'}^{l}\right) \tag{21} \\
&= \delta^{l+1}_{i',j'} \ast \text{rot}_{180^\circ} \left\{ w_{m,n}^{l+1} \right\} f'\left(x_{i',j'}^{l} \right) \tag{22}
\end{align} %]]>

## Introduction to PyTorch
- https://learn.microsoft.com/en-us/training/paths/pytorch-fundamentals/

## Introduction to Computer Vision with PyTorch
- https://learn.microsoft.com/en-us/training/modules/intro-computer-vision-pytorch/
