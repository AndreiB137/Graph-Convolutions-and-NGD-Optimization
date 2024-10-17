# Graph-Convolutions-and-NGD-Optimization

As a sidenote, the repository is still far from what I would want to be, so I will still be working on it.

This repository is meant to start with an introduction to graph convolutions, motivating some of the ideas from the modern perspective of symmetries and representations. Afterwards, an example of an optimization to the classic GCN architecture, where in addition to the usual step in changing the weights by the gradient of the loss with respect to them, there is an additional matrix preconditioner that combines those gradients. This last idea will follow the original paper ["Optimization of Graph Neural Networks with Natural Gradient Descent"](https://arxiv.org/pdf/2008.09624). 

## Table of Contents
* [CNN and GCN Introduction](#Introduction-to-GCN-by-a-parallel-with-symmetries-in-CNN's)
    * [CNN's as a consequence of equivariance](#Symmetries-in-CNN's)
    * [Graph convolutions](#Graph-Convolution-networks-via-spectral-graph-theory)
* [Natural gradient descent](#NGD-optimization)
    * [Mathematical aspects](#Mathematical-details)
    * [Implementation aspects](#Implementation)

## Introduction to GCN by a parallel with symmetries in CNN's

I will use a connection between how you could derive the convolution of a graph with its counterpart the convolution of an image or a 3D (channels included) image.

<p align="center">
<img src="Images_examples/Screenshot 2024-10-15 at 22.05.53.png" width="850"/>
</p>

### Symmetries in CNN's

In ["Symmetry CNN Notes"](https://github.com/AndreiB137/Graph-Convolutions-and-NGD-Optimization/blob/main/Symmetry%20CNN%20Notes.pdf) we can see how, by exploiting particular symmetries that our architecture should possess under data transformations, we are able to derive the exact form of the convolution. The symmetries are justified. In the example of classification, a picture of a tree rotated or transleted by some amount to the right should still be classified as a tree. For the neural network to not distinguish between the original tree image and its rotation, the activations in the next layers of the convolution operation should be affected by the same rotation. In other words, the feature maps are also rotated. In this manner, the neural network uses esentially the same activations from the original image in classifying. The idea can be generalized to general finite or infinite dimensional group representations and gauge groups (or coordinate changes). The later is appropriate if we consider the data distribution as living on a manifold and we are looking for invariant characteristics at the intersection of two neighborhoods with different gauge groups (or coordinate transformations). Since this is not my main point to show in this repository, you can read more about it in the following book by [Maurice Weiler, Patric Forré, Erik Verlinde Max Welling](https://maurice-weiler.gitlab.io/cnn_book/EquivariantAndCoordinateIndependentCNNs.pdf).


<p align = "center">
<img src="Images_examples/GCN_vs_CNN_overview.png" height = 150 width="150"/>
<img src="Images_examples/Screenshot 2024-10-15 at 23.02.42.png" height = 150 width="450"/>
<img src="Images_examples/gcn_web.png" height = 150 width="350"/>
</p>

### Graph Convolution networks via spectral graph theory

We have seen the example of an image convolution. How should we construct a convolution operator in the context of graph-structured data? What symmetries should our graph data have? The two questions are very connected; if we can manufacture an operator that we specially call a convolution (it might not be familiar to the integral convolution or finite convolution in the case before), then we can find its symmetries. From the other perspective, knowing the symmetries of our graph will give us clues to what operators to consider. Indeed, the operators should be compatible with the graph structure, so we can't think of all the operators that satisfy the symmetries, but a subset of them. To give a short summary of everything that is in ["Graph_Convolution"](https://github.com/AndreiB137/Graph-Convolutions-and-NGD-Optimization/blob/main/Graph_Convolutions.pdf) note, we are using our previous understanding of CNN's to make a similar construction to graphs. As it is, graphs are a different type of structure. Due to the interest in function defined on the finite set of vertices, we can switch our perspective from this vector space to the isomorphic corespondent of R^n vectors. Then, we can define an operator named Laplacian due to being the approximation in the finite separated space of the actual differential operator. Next, as the Fourier transform on the real line is defined as a "sum" of the eigenfunctions corresponding to the Laplacian operator, this indicates we should consider the Fourier transform on a function defined on the graph as a linear combination of the eigenfunctions (or just eigenvectors) of the graph Laplacian operator. With that we try to implement a convolution between a kernel and a function, such that the operator maps functions to functions. One example is to multiply the Fourier coefficients of f in the Fourier transform by the kernel acting on the eigenvalue, so scaling the coefficients. Afterwards, by approximating the kernel (which we can assume to be continuous) via a minimax polynomial approximation (more concretely Chebyshev polynomials), we get our final linear model for the neural network. A consequence of the final expression is the permutation equivariance, which was a desired "symmetry". Permuting nodes in the graph should permute the activation layers as well. If you want to see more, there are the ["GCN"](https://arxiv.org/pdf/1609.02907) paper and the ["spectral graph approximation"](https://arxiv.org/pdf/0912.3848). 

## NGD Optimization

After we digested those various ideas, we turn to the NGD optimization. The paper shows an empirical accuracy advantage compared to using merely Adam or SGD with the gradient weights unmodified. In the case of SGD, it can be seen a substantial improvement with a triple or even more test accuracy in the same number of epochs trained. In Adam, the gradient step is optimized using adaptive momentum (or information from the previous updates), which helps for convergence efficiency, while SGD uses only the last update. We can see how SGD with a change in gradients provided by NGD is able to equal the performance of Adam with no NGD (in the same number of epochs), suggesting the importance of using a preconditioner, especially in the context of graph convolutional networks.

#### Mathematical details

I have been naming preconditioner for a couple of times, but we are going to look at it now. The full description is in ["NGD-Notes"](https://github.com/AndreiB137/Graph-Convolutions-and-NGD-Optimization/blob/main/NGD-Notes.pdf), here I will mention just some of the crucial points. The preconditioner is describing a coupling or connection between the computed gradients. This preconditioner might be a consequence of describing the data space as a manifold with a metric on it. Hence, we naturally get the inverse of the metric by multiplying the gradient. This expression essentially comes from the interpretation of the gradient on the manifold as the only vector field of that function, giving in the inner product with an arbitrary vector field the second vector field acting on the function. Thus, if we consider Euclidean space, we recover the usual gradient. A special scenario is a manifold with a Fisher metric. We know about the KL-divergence that is not symmetric, so it can't be a metric, but its second derivative is symmetric, resulting in what is called the Fisher information metric. So, if we compute the inverse and multiply by the computed gradients (modifying the gradients resulting from the backward pass), we obtain the desired step update in the weights. Computing inverses is expensive, especially in the context of huge graphs, but this is exacerbated by having mini-batches. Then, it is also time-intensive, and this inconvenience can't be fixed. As it was presented in the notes, the expression of the Fisher matrix is also time-inefficient to compute; therefore, several [approximations](https://proceedings.mlr.press/v37/martens15.pdf) are used.

### Implementation

<p align = "center">
<img src="Training Results/AdamWithNGD_epsilons.png" width="350"/>
<img src="Training Results/SGDWithNGD_epsilons.png" width="350"/>
</p>

<p align = "center">
<img src="Training Results/Screenshot 2024-10-16 at 11.30.39.png" width = "200"/>
</p>

Turning to implementation. At the time of writing, I can't match the performance obtained by the paper (I also get contrary results) authors with my own code. Also, all my tests have been done on the small Cora dataset. I am still debugging and fixing things. For Adam, I can obtain most of the test accuracy statistics on Cora, but for SGD, I am getting worse performance compared with the non-optimized version. As can be seen as a difference between the first and second plots above, the performance of the Adam model reaches its best for low values of epsilon, while the SGD for high values. This is in contrast with the faster convergence with both optimizers for low epsilon values. In the third picture, you can see how the accuracy of the model on the whole dataset increases or decreases depending on epsilon.

## Acknowledgements

## Citation

If you find this repository useful, please cite the following:

```
@misc{Bodnar2024GCN-NGD,
  author = {Bodnar, Andrei},
  title = {Graph-Convolutions-and-NGD-Optimization},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AndreiB137/Graph-Convolutions-and-NGD-Optimization}},
}
```

## Licence

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
