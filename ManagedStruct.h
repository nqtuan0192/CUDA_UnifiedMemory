#include <cstdint>
#include <iostream>

struct ManagedStruct {
	uint32_t _size;
	float* _data;
	
	/** Default constructor */
	ManagedStruct(uint32_t size = 10);
	
    /** Copy constructor */
    ManagedStruct(const ManagedStruct& other);
    
    /** Move constructor */
    ManagedStruct(ManagedStruct&& other) noexcept;
    
    /** Destructor */
    ~ManagedStruct() noexcept;
    
	/** Copy assignment operator */
    ManagedStruct& operator=(const ManagedStruct& other);
    
    /** Move assignment operator */
    ManagedStruct& operator=(ManagedStruct&& other) noexcept;
    
    void randomize();
    
    friend std::ostream& operator<<(std::ostream& os, ManagedStruct& ms);
};
