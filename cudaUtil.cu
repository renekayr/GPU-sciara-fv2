#include "cudaUtil.cuh"
#include <stdio.h>

// for error-handling on operations that return cudaError_t
void checkReturnedError(cudaError_t error, int line)
{
  if (error != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, line);
    // cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

// for error-handling on operations that return cudaError_t
// overload to accept custom error messages, making it easier
// to pinpoint errors
void checkReturnedError(cudaError_t error, int line, char* errorMsg)
{
  if (error != cudaSuccess)
  {
    printf("An error occurred! Error message:\n%s\n", errorMsg);
    printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, line);
    // cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

// for error-handling on operations that do not return any error
void checkError(int line)
{
  cudaError_t error = cudaGetLastError();
  checkReturnedError(error, line);
}
