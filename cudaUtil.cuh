#ifndef _CUDAUTIL_CUH_
#define _CUDAUTIL_CUH_

void checkReturnedError(cudaError_t error, int line);
void checkError(int line);

#endif
