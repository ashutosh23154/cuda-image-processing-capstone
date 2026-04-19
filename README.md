# CUDA Image Processing Capstone

## Purpose
The goal of this project is to accelerate image processing using GPU hardware. Specifically, I developed a CUDA C++ pipeline to read a high-resolution image, transfer the pixel data to the GPU, apply a grayscale conversion filter using parallel threads, and save the resulting image.

## Algorithms and Kernels
The project uses a custom CUDA kernel (`grayscaleKernel`). The image is represented as a 1D array of RGB values. I mapped the 2D image coordinates (x, y) to a 2D grid of CUDA threads (16x16 blocks). Each thread calculates the exact memory offset for its assigned pixel, reads the Red, Green, and Blue channels, applies the standard luminosity formula (`0.299*R + 0.587*G + 0.114*B`), and writes the grayscale value to the output array. 

## Execution and Performance
The code takes command-line arguments for the input and output files. 
I tested the code using a high-resolution test image from the SIPI Image Database (converted to JPG format). The GPU execution was incredibly fast, processing the image in approximately **1.86 milliseconds**.

## How to Compile and Run
This project includes a `Makefile` for easy compilation using the `nvcc` compiler.
1. Compile the code: `make build`
2. Run the executable: `./image_processor <input_image.jpg> <output_image.png>`

## Lessons Learned
One of the biggest challenges was handling image loading and saving in C++ without relying on massive, heavy frameworks like OpenCV. I learned how to integrate lightweight, single-header libraries (`stb_image`) to handle the file I/O on the CPU so that I could focus entirely on writing the raw CUDA memory transfers and kernel execution on the GPU.
