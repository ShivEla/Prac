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


#Q1.PRINTING THREAD IDS
#include <stdio.h>

_global_ void printThreadID()
{
    int tid_x = threadIdx.x; // Thread index in x dimension
    int tid_y = threadIdx.y; // Thread index in y dimension
    int tid = tid_x + tid_y * blockDim.x; // Global thread ID

    printf("Thread ID: %d, threadIdx.x: %d, threadIdx.y: %d\n", tid, tid_x, tid_y);
}

int main()
{
    // Define grid and block dimensions
    dim3 threadsPerBlock(2, 3); // 2 threads in x dimension, 3 threads in y dimension
    dim3 numBlocks(1, 1); // 1 block in x dimension, 1 block in y dimension

    // Launch kernel
    printThreadID<<<numBlocks, threadsPerBlock>>>();

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    return 0;
}
#-------------------------------------------------------------------
Q2.SUM OF ELEMENTS IN 2D ARRAY
#include <stdio.h>

#define N 4

_global_ void sum2DArray(int *array, int *result)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    atomicAdd(result, array[tid]);
}

int main()
{
    int h_array[N][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    int h_result = 0;

    int *d_array, *d_result;
    cudaMalloc((void **)&d_array, N * N * sizeof(int));
    cudaMalloc((void **)&d_result, sizeof(int));

    cudaMemcpy(d_array, h_array, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    sum2DArray<<<1, threadsPerBlock>>>(d_array, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum of elements in 2D array: %d\n", h_result);

    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}
#------------------------------------------------------------------
Q3.CALCULATE DISTANCE OF ALL THE POINTS IN A GRID TO A SPECIFIC POINT (X,Y) WITH SINGLE BLOCK AND MULTIPLE THREADS

#include <stdio.h>
#include <cuda.h>
#define N 8

// Memory Allocated in Device
_device_ float dgrid[N][N];

// Kernel Function
_global_ void findDistance(int x, int y)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    float n = ((i - x) * (i - x)) + ((j - y) * (j - y));
    dgrid[i][j] = sqrt(n);
}

// Main Function
void main()
{
    int i, j;

    // Memory Allocated in Host
    float hgrid[N][N];

    // 1D Grid
    // 2D Block
    dim3 dBlock(N, N);

    // ----
    printf("Enter the x coordinate of node : ");
    scanf_s("%d", &i);

    printf("Enter the y coordinate of node : ");
    scanf_s("%d", &j);

    // Calling the kernel function with 1 - Grid, 1 - 2D_Block, 16x16 - Threads
    findDistance<<<1, dBlock>>>(i, j);
    // ----

    // Copy the matrix from device to host to print to console
    cudaMemcpyFromSymbol(&hgrid, dgrid, sizeof(dgrid));

    printf("Values in hgrid!\n\n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            printf("\t%.0lf", hgrid[i][j]);
        printf("\n\n");
    }
}
#-------------------------------------------------------------------
Q4.CALCULATE DISTANCE OF ALL THE POINTS IN A GRID TO A SPECIFIC POINT (X,Y) WITH MULTIPLE BLOCK AND MULTIPLE THREADS

#include <stdio.h>
#include <cuda.h>
#define N 16
#define D 2

// Memory Allocated in Device
_device_ float dgrid[N * D][N * D];

// Kernel Function
_global_ void findDistance(int x, int y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float n = ((i - x) * (i - x)) + ((j - y) * (j - y));
    dgrid[i][j] = sqrt(n);
}

// Main Function
void main()
{
    int i, j;

    // Memory Allocated in Host
    float hgrid[N * D][N * D];

    // 2D Grid (4 * 4 Blocks)
    dim3 dGrid(D, D);

    // 2D Block (16 * 16)
    dim3 dBlock(N, N);

    printf("Enter the x coordinate of node : ");
    scanf_s("%d", &i);
    printf("Enter the y coordinate of node : ");
    scanf_s("%d", &j);

    // Calling the kernel function with 1 - 2D_Grid, 1 - 2D_Block, 16x16 - Threads
    findDistance<<<dGrid, dBlock>>>(i, j);

    // Copy the matrix from device to host to print to console
    cudaMemcpyFromSymbol(&hgrid, dgrid, sizeof(dgrid));

    printf("Values in hgrid!\n\n");
    for (i = 0; i < N * D; i++)
    {
        for (j = 0; j < N * D; j++)
            printf("\t%.0lf", hgrid[i][j]);
        printf("\n\n");
    }
}
#----------------------------------------------------------------
Q5.CHARACTER ARRAY COPYING
#include <stdio.h>

#define N 10

_global_ void copyCharArrays(char *src, char *dest)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    dest[tid] = src[tid];
}

int main()
{
    char h_src[N] = "HelloCUDA";
    char h_dest[N];

    char *d_src, *d_dest;
    cudaMalloc((void **)&d_src, N * sizeof(char));
    cudaMalloc((void **)&d_dest, N * sizeof(char));

    cudaMemcpy(d_src, h_src, N * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 4; // Threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // Calculate number of blocks needed

    copyCharArrays<<<numBlocks, blockSize>>>(d_src, d_dest);

    cudaMemcpy(h_dest, d_dest, N * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Copied String: %s\n", h_dest);

    cudaFree(d_src);
    cudaFree(d_dest);

    return 0;
}
#-------------------------------------------------------------------
#Q6.CHARACTER ARRAY MANIPULATION
#include <stdio.h>

#define N 10

_global_ void manipulateCharArray(char *array)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (array[tid] >= 'a' && array[tid] <= 'z')
        array[tid] -= 32; // Convert lowercase letter to uppercase
}

int main()
{
    char h_array[N] = "helloCUDA";

    char *d_array;
    cudaMalloc((void **)&d_array, N * sizeof(char));

    cudaMemcpy(d_array, h_array, N * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 4; // Threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // Calculate number of blocks needed

    manipulateCharArray<<<numBlocks, blockSize>>>(d_array);

    cudaMemcpy(h_array, d_array, N * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Manipulated String: %s\n", h_array);

    cudaFree(d_array);

    return 0;
}
#--------------------------------------------------------------------
Q7.WORD COUNTER
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LENGTH 100
#define MAX_WORDS 100

_device_ int myStrCmp(const char *str1, const char *str2)
{
    int i = 0;
    while (str1[i] != '\0' && str2[i] != '\0' && str1[i] == str2[i])
    {
        i++;
    }
    return (str1[i] - str2[i]);
}

_device_ void myStrCpy(char *dest, const char *src)
{
    int i = 0;
    while ((dest[i] = src[i]) != '\0')
    {
        i++;
    }
}

// Kernel function to count word frequency
_global_ void countWordFrequency(char *sentence, int *wordCounts, int numWords, char *uniqueWords, int *totalWords)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numWords && wordCounts[tid] == 0)
    {
        int count = 1; // Initialize count for current word
        for (int i = tid + 1; i < numWords; ++i)
        {
            if (myStrCmp(&sentence[tid * MAX_LENGTH], &sentence[i * MAX_LENGTH]) == 0)
            {
                count++; // Increment count if the word matches
                wordCounts[i] = 1; // Mark word as counted
            }
        }
        wordCounts[tid] = count;

        // Store unique words
        if (wordCounts[tid] == 1)
        {
            int index = atomicAdd(totalWords, 1);
            myStrCpy(&uniqueWords[index * MAX_LENGTH], &sentence[tid * MAX_LENGTH]);
        }
    }
}

int main()
{
    char h_sentence[MAX_WORDS][MAX_LENGTH] = {
        "hello", "world", "hello", "cuda", "world"
    };
    int numWords = 5;

    // Copy sentence to device memory
    char *d_sentence;
    cudaMalloc((void **)&d_sentence, numWords * MAX_LENGTH * sizeof(char));
    cudaMemcpy(d_sentence, h_sentence, numWords * MAX_LENGTH * sizeof(char), cudaMemcpyHostToDevice);

    // Allocate memory for word counts on device
    int *d_wordCounts;
    cudaMalloc((void **)&d_wordCounts, numWords * sizeof(int));

    // Allocate memory for unique words on device
    char *d_uniqueWords;
    cudaMalloc((void **)&d_uniqueWords, MAX_WORDS * MAX_LENGTH * sizeof(char));

    // Total number of words
    int *d_totalWords;
    cudaMalloc((void **)&d_totalWords, sizeof(int));
    cudaMemcpy(d_totalWords, &numWords, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to count word frequency and extract unique words
    int blockSize = 256;
    int numBlocks = (numWords + blockSize - 1) / blockSize;
    countWordFrequency<<<numBlocks, blockSize>>>(d_sentence, d_wordCounts, numWords, d_uniqueWords, d_totalWords);

    // Copy unique words back to host
    char h_uniqueWords[MAX_WORDS][MAX_LENGTH];
    int totalWords;
    cudaMemcpy(&totalWords, d_totalWords, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uniqueWords, d_uniqueWords, totalWords * MAX_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);

    // Print word counts for unique words
    printf("\nWord Frequencies:\n");
    for (int i = 0; i < totalWords; ++i)
    {
        int count = 0;
        for (int j = 0; j < numWords; ++j)
        {
            if (strcmp(h_uniqueWords[i], h_sentence[j]) == 0)
            {
                count++;
            }
        }
        if (count != 0) {
            printf("%s : %d\n", h_uniqueWords[i], count);
        }
    }

    // Free device memory
    cudaFree(d_sentence);
    cudaFree(d_wordCounts);
    cudaFree(d_uniqueWords);
    cudaFree(d_totalWords);

    return 0;
}
