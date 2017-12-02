#include <iostream>
#include <cuda.h>

#include "helper.h"
#include "ManagedStruct.h"

// CUDA kernel to add elements of two arrays
__global__ void add(uint32_t n, float* x, float* y) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

int main(int argc, char** argv) {
	/* manually set device */
	int device_id = 0;
	if (argc > 1) {
		device_id = atoi(argv[1]);
		
	}
	cudaSetDevice(device_id);
	
	/* start coding from here */
	uint32_t N = 1 << 20;
	
	ManagedStruct ms[2] = {N, N};
	for (int i = 0; i < 2; ++i) {
		ms[i].randomize();
	}
	
	//std::cout << ms[0] << std::endl;
	//std::cout << ms[1] << std::endl;
	
	uint32_t blockSize = 256;
	uint32_t numBlocks = (N + blockSize - 1) / blockSize;
	
	add<<<numBlocks, blockSize>>>(N, ms[0]._data, ms[1]._data);
	
	cudaDeviceSynchronize();
	//std::cout << ms[1] << std::endl;
	


	cudaDeviceReset();
	return 0;
}


/**
 * https://devblogs.nvidia.com/parallelforall/unified-memory-cuda-beginners/
 * https://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
 * 
 * 
 * */
