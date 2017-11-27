#include <iostream>
#include <cuda.h>

int main(int argc, char** argv) {
	/* manually set device */
	int device_id = 0;
	if (argc > 1) {
		device_id = atoi(argv[1]);
		
	}
	cudaSetDevice(device_id);
	
	/* start coding from here */


	cudaDeviceReset();
	return 0;
}
