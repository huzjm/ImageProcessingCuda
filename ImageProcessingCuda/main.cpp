#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>

// CUDA kernel declaration
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels);

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Load image using OpenCV
    cv::Mat image = cv::imread("Images/input.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    size_t size = width * height * channels * sizeof(unsigned char);

    // Allocate memory on the GPU
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    checkCudaError(cudaMalloc((void**)&d_input, size), "Failed to allocate device memory for input");
    checkCudaError(cudaMalloc((void**)&d_output, size), "Failed to allocate device memory for output");

    // Copy image data to GPU
    checkCudaError(cudaMemcpy(d_input, image.data, size, cudaMemcpyHostToDevice), "Failed to copy image data to device");

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    gaussianBlurKernel << <gridDim, blockDim >> > (d_input, d_output, width, height, channels);
    checkCudaError(cudaGetLastError(), "Failed to launch kernel");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");
    auto stop = std::chrono::high_resolution_clock::now();

    // Copy processed data back to CPU
    cv::Mat result(height, width, image.type());
    checkCudaError(cudaMemcpy(result.data, d_output, size, cudaMemcpyDeviceToHost), "Failed to copy output data to host");

    // Measure and print execution time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CUDA execution time: " << duration.count() << " ms" << std::endl;

    // Save the result
    cv::imwrite("Images/output.jpg", result);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}