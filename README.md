# This is a fork of DLPrimitives that uses Vulkan instead of OpenCL.

It is not intended to be merged with the original DLPrimitives library, and will eventually be renamed or factored into another project.

## To-do
- Port convolution layer kernels to Vulkan GLSL or write new ones entirely
- Add compatibility with more data types for certain operators that are currently only compatible with float32 (random, GEMM, etc.)
- Implement better caching and/or make static per-device references to each kernel (profiling showed that using the current kernel cache has significant performance impact)
- Modify kernels to use specialization constants instead of preprocessor defines when possible (mostly done)
	- Reduce dependency on pointwise and pointwise_broadcast_reduce
- Implement work-per-thread model that CLBlast has, as some kernels try to invoke more workgroups than a device will allow
- Fix subgroup under-utilization by some kernels (many have workgroup size set to 1, 1, 1 by default)
- Probably some other stuff I am forgetting now

## Building

### Linux

You will need CMake, the GCC toolchain, and the Vulkan development headers.
Instructions for obtaining these will depend on your distribution (will elaborate more later)

Once this is done, simply run the following:
```
git clone https://github.com/softcookiepp/dlprimitives
cd dlprimitives
git submodule update --remote --recursive
mkdir build && cd build
cmake .. && make
```

### Windows

So far I have no idea. Any help on this is appreciated!
