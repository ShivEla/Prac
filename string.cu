#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORDS 1000
#define MAX_WORD_LENGTH 50

__global__ void wordCountKernel(char *str, int *histogram, int numWords) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < numWords) {
        atomicAdd(&histogram[str[tid]], 1);
        tid += stride;
    }
}

int main() {
    char input[] = "Hello world hello CUDA hello world world";
    char *dev_input;
    int *histogram, *dev_histogram;

    // Allocate memory on host
    int numWords = 0;
    char *words[MAX_WORDS];
    char *token = strtok(input, " ");
    while (token != NULL && numWords < MAX_WORDS) {
        words[numWords++] = token;
        token = strtok(NULL, " ");
    }

    // Allocate memory on device
    cudaMalloc(&dev_input, strlen(input) + 1);
    cudaMalloc(&dev_histogram, MAX_WORDS * sizeof(int));

    // Copy input string to device
    cudaMemcpy(dev_input, input, strlen(input) + 1, cudaMemcpyHostToDevice);

    // Initialize histogram to zeros
    cudaMemset(dev_histogram, 0, MAX_WORDS * sizeof(int));

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (numWords + blockSize - 1) / blockSize;
    wordCountKernel<<<numBlocks, blockSize>>>(dev_input, dev_histogram, numWords);

    // Copy histogram back to host
    histogram = (int *)malloc(MAX_WORDS * sizeof(int));
    cudaMemcpy(histogram, dev_histogram, MAX_WORDS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print word counts
    for (int i = 0; i < numWords; ++i) {
        printf("%s: %d\n", words[i], histogram[i]);
    }

    // Cleanup
    free(histogram);
    cudaFree(dev_input);
    cudaFree(dev_histogram);

    return 0;
}
