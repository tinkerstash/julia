---
layout: post
title: "GPU SoftImpute"
description: "GPU SoftImpute"
author: jiawei
category: machine learning
finished: true
---

# Links

* [Our Github page](https://github.com/tinkerstash/gpuimpute) with GPU-accelerated version of SoftImpute.
* [Accelerated proximal gradient for SoftImpute](https://arxiv.org/abs/1703.05487) paper by Quanming Yao, James T. Kwok.
* [Matlab implementation](https://github.com/quanmingyao/AIS-impute) by the same folks.
* [Original SoftImpute paper](https://web.stanford.edu/~hastie/Papers/mazumder10a.pdf) by Mazumder, Hastie, Tibshirani.


# Introduction

We are interested in GPU-accelerating the SoftImpute algorithm from Hastie et al's paper. This is a SVD-based collaborative filtering algorithm.

We use randomized SVD as it seems to work better with GPUs. We use the accelerated proximal gradient method also mentioned in Quanming Yao et al's paper. See below for references.

We compare these GPU-accelerated versions against the CPU version. We see that there is at least a *10X* speedup. We also compare against a simple SGD implementation with a bit of tuning. All CPU implmentations, SoftImpute or SGD, use Eigen or BLAS. They are pretty efficient, to be fair.

For the data, currently we only have the MovieLens 20M dataset. We split the data into 5 parts and use one part for measuring the error. (There is no validation set but we hardly do any tuning anyway.)

The results seem to be that SGD is still faster, even when on single core.

Our CPU is a I7-6700K. Our GPU is a Titan-X with 12G RAM.

![plot](https://github.com/tinkerstash/gpuimpute/blob/master/results/plot1.png?raw=true)

* The slowest is CPU-NoAcc which is the CPU version with un-accelerated proximal gradient.
* The next slowest is CPU-Acc which is the CPU version with accelerated proximal gradient.
* The GPU versions are all significantly faster, seemingly >10X. (We do have a fast GPU unfortunately.)
* However, SGD still seems to be the fastest.

The most tricky part of the GPU code is probably the evaluation of many short inner products. It seems to be the bottleneck, and we have to write a custom kernel to do that efficiently.
