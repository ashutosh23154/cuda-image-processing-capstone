#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int offset = (y * width + x) * channels;
        unsigned char r = input[offset];
        unsigned char g = input[offset + 1];
        unsigned char b = input[offset + 2];
        output[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./image_processor <input> <output>\n";
        return -1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);

    if (h_input == NULL) {
        std::cout << "Failed to load image\n";
        return -1;
    }

    size_t imgSize = width * height * channels;
    size_t graySize = width * height;

    unsigned char* h_output = (unsigned char*)malloc(graySize);
    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, graySize);

    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    cudaMemcpy(h_output, d_output, graySize, cudaMemcpyDeviceToHost);

    stbi_write_png(outputFile, width, height, 1, h_output, width);

    stbi_image_free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
