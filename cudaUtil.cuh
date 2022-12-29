#ifndef _CUDAUTIL_CUH_
#define _CUDAUTIL_CUH_

void checkReturnedError(cudaError_t error, int line);
void checkReturnedError(cudaError_t error, int line, char* errorMsg);
void checkError(int line);

#endif
