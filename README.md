# This is a fork of DLPrimitives that uses Vulkan instead of OpenCL.

It is not intended to be merged with the original DLPrimitives library, and will eventually be renamed or factored into another project.

## To-do
- Port convolution layer kernels to Vulkan GLSL or write new ones entirely
- Add compatibility with more data types for certain operators (random, GEMM, etc.)
- Implement better caching
- Modify kernels to use specialization constants instead of preprocessor defines when possible
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
