#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>

// Forward declaration of the CUDA kernel
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

    // Measure execution time for GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    gaussianBlurKernel << <gridDim, blockDim >> > (d_input, d_output, width, height, channels);
    checkCudaError(cudaGetLastError(), "Failed to launch kernel");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");
    auto stop_gpu = std::chrono::high_resolution_clock::now();

    // Copy processed data back to CPU
    cv::Mat result_gpu(height, width, image.type());
    checkCudaError(cudaMemcpy(result_gpu.data, d_output, size, cudaMemcpyDeviceToHost), "Failed to copy output data to host");

    // Measure and print GPU execution time
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(stop_gpu - start_gpu);
    std::cout << "CUDA execution time: " << duration_gpu.count() << " ms" << std::endl;

    // CPU-based Gaussian blur
    cv::Mat result_cpu;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::GaussianBlur(image, result_cpu, cv::Size(15, 15), 0);
    auto stop_cpu = std::chrono::high_resolution_clock::now();

    // Measure and print CPU execution time
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu - start_cpu);
    std::cout << "CPU execution time: " << duration_cpu.count() << " ms" << std::endl;

    // Save the results
    cv::imwrite("Images/output_gpu.jpg", result_gpu);
    cv::imwrite("Images/output_cpu.jpg", result_cpu);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}