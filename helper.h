#ifndef CUDA_SUPPORT	// for CUDA only
#define CUDA_SUPPORT

#include <cassert>
#include <cuda.h>

#define CUDA_CALL(ans) cudaAssert((ans), __FILE__, __LINE__)
inline cudaError_t cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
#if defined(DEBUG) || defined(_DEBUG)
	if (code != cudaSuccess) {
		fprintf(stderr,"GPU CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
#endif
	return code;
}

#define CUDA_M_MALLOC(ptr, type, number)					cudaMalloc((void**)&(ptr), (number) * sizeof(type))
#define CUDA_M_MALLOC_MANAGED(ptr, type, number)			cudaMallocManaged((void**)&(ptr), (number) * sizeof(type))
#define CUDA_M_COPY_HOSTTODEVICE(from, to, type, number)	cudaMemcpy((type*)(to), (type*)(from), (number) * sizeof(type), cudaMemcpyHostToDevice)
#define CUDA_M_COPY_DEVICETOHOST(from, to, type, number)	cudaMemcpy((type*)(to), (type*)(from), (number) * sizeof(type), cudaMemcpyDeviceToHost)
#define CUDA_M_COPY_DEVICETODEVICE(from, to, type, number)	cudaMemcpy((type*)(to), (type*)(from), (number) * sizeof(type), cudaMemcpyDeviceToDevice)

#endif
