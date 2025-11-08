# TODO for implementing [TexGS](https://arxiv.org/pdf/2411.18625) sampling appraoch

## Some notes first
TexGS does a bunch of other things in the background to elevate performance. We ignore all of this, and just look at implementing the TexGS texture sampling functionality. This is described in Section 3.2 of the paper.

## TODO
Essentially, each gaussian has a small rectangular plane that is rotated with the softmax of the smallest scale axis and centered on the gaussian. The sample 


1. Ray-Gaussian intersection (implement our own pytorch version) Eq. 5 and 6
    1. $n_i$ normal corresponds to the direction of the smallest scale axis per gaussian (softmax approximation)
    2. $o$ corresponds to the origin of the ray, $\mu$ is the gaussian center, 