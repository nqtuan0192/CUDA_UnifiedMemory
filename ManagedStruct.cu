#include "ManagedStruct.h"

#include <memory>
#include <algorithm>

#include "helper.h"

/** PRNG */
std::random_device rd;
std::mt19937 e(rd());
std::uniform_real_distribution<> r(0, 1);
float f_rand() {
	return r(e);
}

/** Default constructor */
ManagedStruct::ManagedStruct(uint32_t size) : _size(size) {
	std::cout << "Default constructor" << std::endl;
	CUDA_CALL(CUDA_M_MALLOC_MANAGED(this->_data, float, this->_size));
}

/** Copy constructor */
ManagedStruct::ManagedStruct(const ManagedStruct& other) : _size(other._size) {
	std::cout << "Copy constructor" << std::endl;
	CUDA_CALL(CUDA_M_MALLOC_MANAGED(this->_data, float, this->_size));
	CUDA_CALL(CUDA_M_COPY_DEVICETODEVICE(this->_data, other._data, float, this->_size));
}

/** Move constructor */
ManagedStruct::ManagedStruct(ManagedStruct&& other) noexcept : _size(other._size) {
	std::cout << "Move constructor" << std::endl;
	this->_data = other._data;
	other._data = nullptr;
}

/** Destructor */
ManagedStruct::~ManagedStruct() noexcept {
	std::cout << "Destructor" << std::endl;
	if (this->_data != nullptr) {
		CUDA_CALL(cudaFree(this->_data));
		this->_data = nullptr;
	}
}

/** Copy assignment operator */
ManagedStruct& ManagedStruct::operator=(const ManagedStruct& other) {
	std::cout << "Copy assignment operator" << std::endl;
	this->_size = other._size;
	CUDA_CALL(CUDA_M_COPY_DEVICETODEVICE(this->_data, other._data, float, this->_size));
	return *this;
}

/** Move assignment operator */
ManagedStruct& ManagedStruct::operator=(ManagedStruct&& other) noexcept {
	std::cout << "Move assignment operator" << std::endl;
	std::swap(this->_size, other._size);
	std::swap(this->_data, other._data);
	return *this;
}

void ManagedStruct::randomize() {
	// CUDA 8 and below do not support lambda, use functor instead.
	// std::for_each(this->_data, this->_data + this->size, [](float &n){ n = f_rand(); });
	
	// running on Host
	for (uint32_t i = 0; i < this->_size; ++i) {
		this->_data[i] = f_rand();
	}
}

std::ostream& operator<<(std::ostream& os, ManagedStruct& ms) {
	// running on Host
	os << "Array (size = " << ms._size << ") : ";
	for (uint32_t i = 0; i < ms._size; ++i) {
		os << ms._data[i] << "\t";
	}
	return os;
}
