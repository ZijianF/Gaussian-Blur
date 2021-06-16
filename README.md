# Gaussian-Blur
**All files are in the main branch.**
## Introduction ##
A Gaussian blur is a 2-D convolution operator that is used to blur images in order to reduce detail and noise.

Technically, a Gaussian blur is represented as a matrix of weights, also called kernel or mask. The convolution is the process of adding each pixel of the image to its local neighbors, weighted by the matrix.

This program utilizes the GPU's advantage of having large number of cuda cores to do parallel computaion. Hence, one cuda core will process one pixel at a time. The performance is significantly improved comparing to the serial version.

## How to use ##
There are two files, the .c file is the code for the serial version of the program which only utilizes one cpu core to do serial computation. It shows basic implementation of Gaussian Blur.
The .c version can be compiled using genuine C compiler.
It takes three command line arguments to operate:
1. The first argument specifies the name of the input binary PGM file.
2. The second argument specifies the name of the output (post-processed) binary PGM file.
3. The third argument specifies the sigma value to use for the blur. The sigma value should be considered valid only it’s greater than 0 and if the order of the resulting kernel matrix is not bigger than the input image’s width or height.

The .cu file is the cuda version which utilizes a GPU for computation.
It needs to be compiled using nvcc, which is almost the same as gcc excpet that you need to prefix all of the compiler flags with **-Xcompiler**

Example: $ nvcc -Xcompiler -Wall -Xcompiler -Werror ... -o gaussian_blur_cuda gaussian_blur_cuda.cu -lm

The cuda version takes the same three command argument as the serial version takes.


