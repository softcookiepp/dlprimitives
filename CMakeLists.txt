cmake_minimum_required(VERSION 3.5)
project(dlprimitives)


enable_testing()

if(NOT WIN32)
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O2 -Wall")
endif()

option(USE_CL_HPP "Use older CL/cl.hpp instead of CL/opencl.hpp" OFF)
option(USE_CL2_HPP "Use older CL/cl2.hpp instead of CL/opencl.hpp" OFF)
option(USE_INTERNAL_BLAS "Use Internal Blas" OFF)
option(USE_HDF "Compile with HDF support" OFF)
option(BUILD_CORE_ONLY "Build only dlprimitives core" OFF)
option(DLPRIM_STATIC_BUILD "Build static version only" OFF)
option(USE_VULKAN_API "Build with Vulkan API" ON)

if(BUILD_CORE_ONLY)
    SET(USE_CL_HPP OFF)
    SET(USE_HDF OFF)
endif()

if(NOT USE_VULKAN_API)
	message(FATAL_ERROR "OpenCL API is being removed, do not use it")
endif()
add_subdirectory("CLBlast")

configure_file(config.cmake.hpp dlprim/config.hpp)
set(DLPRIM_PUBLIC_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/dlprim/config.hpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include") 
include_directories("${CMAKE_CURRENT_BINARY_DIR}") 
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/my_blas") 

find_program(PYTHON NAMES python3 python python2)
if(NOT PYTHON)
	message(FATAL_ERROR "Can't build without Python interpreter")
endif(NOT PYTHON)

if(USE_HDF)
    find_package(HDF5)
endif()

if(USE_HDF AND HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
else()
    add_definitions(-DDISABLE_HDF5)
endif()

if(NOT LIBDIR)
	set(LIBDIR lib CACHE STRING "Library installation directory" FORCE)
endif()


SET(TEST_DEV "0:0" CACHE STRING "OpenCL platform:device, default 0:0")

if(NOT USE_INTERNAL_BLAS)
    find_path(CBLAS_PATH cblas.h)
    find_library(CBLAS_LIB openblas blas) 
endif()


file(GLOB CL_SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels-glsl/*.glsl)

if(NOT BUILD_CORE_ONLY)
    find_package(PythonLibs 3)
endif()

if(PYTHONLIBS_FOUND AND NOT BUILD_CORE_ONLY)
    string(REPLACE "." ";" PLIB_VER_LIST ${PYTHONLIBS_VERSION_STRING})
    list(GET PLIB_VER_LIST 0 PLIB_VER_MAJOR)
    list(GET PLIB_VER_LIST 1 PLIB_VER_MINOR)
    set(PLIB_VER "${PLIB_VER_MAJOR}${PLIB_VER_MINOR}")
    find_package(Boost COMPONENTS python${PLIB_VER} numpy${PLIB_VER} python3 numpy3 python numpy)
else()
    set(PLIB_VER "00")
endif()

if(NOT BUILD_CORE_ONLY)

    if(NOT CBLAS_LIB OR NOT CBLAS_PATH) 
        set(EXTRA_SRC src/my_blas/my_cblas.cpp)
        set(BLAS_LIB )
        set(EXTRA_DEFS -DUSE_INTERNAL_BLAS)
    else()
        set(EXTRA_SRC )
        set(BLAS_LIB ${CBLAS_LIB})
        include_directories("${CBLAS_PATH}")
        set(EXTRA_DEFS )
    endif()

    add_definitions(${EXTRA_DEFS})
endif()

message("=== Status ===")
message("  Python: ${PYTHON}")
if(BLAS_LIB)
    message("  BLAS: include ${CBLAS_PATH}")
    message("        lib ${CBLAS_LIB}")
else()
    message("  BLAS: None")
endif()

if(HDF5_FOUND)
    message("  HDF5: include ${HDF5_INCLUDE_DIR}")
    message("        lib  ${HDF5_LIBRARIES} hdf5_cpp")
else()
    message("  HDF5: None")
endif()

if(BUILD_PYDLPRIM)
    message("  Python dlprim: enabled")
    message("  Python version: ${PLIB_VER}")
    message("  Python: lib ${PYTHON_LIBRARIES}")
    message("          include ${PYTHON_INCLUDE_DIRS}")
    message("  Boost: include ${Boost_INCLUDE_DIRS}")
    message("     boost_numpy ${BNUMPY_LIB}")
    message("     boost_python ${BPYTHON_LIB}")
else()
    message("  Python dlprim: disabled")
endif()

add_custom_command(
			  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp 
			  DEPENDS
			  ${CL_SOURCES} ${CL_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels-glsl/build_source.py
			  COMMAND 
			  ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels-glsl/build_source.py 
				-o ${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp
				-I ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels-glsl
				${CL_SOURCES}
			  )

set(DLPRIM_CORE_SRC
        src/context.cpp
        src/tensor.cpp
        src/program_cache.cpp
        src/gemm.cpp
        src/random.cpp
        src/shape.cpp
        src/core/ip.cpp
        src/core/init.cpp
        src/core/conv.cpp
        src/core/bn.cpp
        src/core/slice.cpp
        src/core/pool.cpp
        src/core/pointwise.cpp
        src/core/loss.cpp
        src/core/activation.cpp
        src/core/util.cpp
        src/core/interpolate.cpp
        #src/naive.cpp
        src/im2col.cpp
        src/softmax.cpp
        src/tiered_cache.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/kernels.cpp)

set(DLPRIM_SRC 
        src/activation.cpp
        src/ops/inner_product.cpp
        src/ops/batch_normalization.cpp
        src/ops/conv2d.cpp
        src/ops/activation.cpp
        src/ops/softmax.cpp
        src/ops/elementwise.cpp
        src/ops/pooling.cpp
        src/ops/bwd_bias.cpp
        src/ops/axpby.cpp
        src/ops/scal.cpp
        src/ops/concat.cpp
        src/ops/nll_loss.cpp
        src/ops/mse_loss.cpp
        src/ops/initialization.cpp
        src/ops/reshape.cpp
        src/ops/pointwise.cpp
        src/ops/reduction.cpp
        src/ops/parameter.cpp
        src/ops/interpolation.cpp
        ${EXTRA_SRC}
        )

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIBDIR}")

if(WIN32 OR DLPRIM_STATIC_BUILD)
	add_library(dlprim_core STATIC ${DLPRIM_CORE_SRC})
    if(NOT WIN32)
        set_target_properties(dlprim_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
else()
	add_library(dlprim_core SHARED ${DLPRIM_CORE_SRC})
endif()

target_link_libraries(dlprim_core PUBLIC tart clblast)


target_compile_definitions(dlprim_core PUBLIC VULKAN_API)
target_include_directories(dlprim_core PUBLIC CLBlast/tart/include CLBlast/tart/xxhashct)

if(NOT BUILD_CORE_ONLY)
		
    if(WIN32)
        add_library(dlprim STATIC ${DLPRIM_SRC})
    else()
        add_library(dlprim SHARED ${DLPRIM_SRC})
    endif()
	
	target_link_libraries(dlprim dlprim_core ${BLAS_LIB})
    if(HDF5_FOUND)
        target_link_libraries(dlprim ${HDF5_LIBRARIES} hdf5_cpp) 
    endif()

    set_target_properties(dlprim PROPERTIES 
                PUBLIC_HEADER "${DLPRIM_PUBLIC_HEADERS}")

    add_executable(test_context tests/test_context.cpp)
    add_executable(test_broadcast_reduce tests/test_broadcast_reduce.cpp)
    add_executable(test_util tests/test_util.cpp)

    target_link_libraries(test_context dlprim_core)
    target_link_libraries(test_util dlprim_core)
    target_link_libraries(test_broadcast_reduce dlprim_core)

    #add_executable(test_from_template tests/test_from_template.cpp)
    #add_executable(test_net tests/test_net.cpp)
    
    # benchmark uses DLPrim's OpenCL-dependent timing utility, which does not play nicely with Vulkan whatsoever.
    # As such, it is being temporarily disabled until a replacement can be made.
    #add_executable(dlprim_benchmark tools/benchmark.cpp)
    #add_executable(image_predict examples/cpp/image_predict.cpp)
    # same reason as above
    #add_executable(mnist tests/mnist.cpp)
    #add_executable(train_mnist examples/cpp/train_mnist.cpp)
    #add_executable(dlprim_flops tools/flops.cpp)
    add_executable(test_random tests/test_random.cpp)
    add_executable(test_gemm tests/test_gemm.cpp)
    add_executable(sanity_check tests/sanity_check.cpp)

    #target_link_libraries(test_from_template dlprim)
    #target_link_libraries(mnist dlprim)
    #target_link_libraries(train_mnist dlprim)
    
    #target_link_libraries(dlprim_benchmark dlprim)
    #target_link_libraries(image_predict dlprim)
    #target_link_libraries(dlprim_flops dlprim)
    #target_link_libraries(test_net dlprim)
    target_link_libraries(test_random dlprim)
    target_link_libraries(test_gemm dlprim)
    target_link_libraries(sanity_check dlprim)

endif()

set(EXTRA_INSTALL )

if(NOT BUILD_CORE_ONLY)
    file(GLOB TEST_TEMPLATES  ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_case_*.json)

    foreach(TEMPLATE ${TEST_TEMPLATES}) 
        get_filename_component(TEST_NAME ${TEMPLATE} NAME_WE)
        add_test("test_${TEST_NAME}" test_from_template ${TEST_DEV} ${TEMPLATE})
    endforeach()
    add_test(test_net test_net ${TEST_DEV} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_net.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weights.json)
    add_test(test_net_nonopt test_net "-k" ${TEST_DEV} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_net.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weights.json)
    add_test(test_json test_json)
    add_test(test_gemm test_gemm ${TEST_DEV})
    add_test(test_random test_random ${TEST_DEV})
    add_test(test_context test_context ${TEST_DEV})
    add_test(test_util test_util ${TEST_DEV})
    add_test(test_broadcast_reduce test_broadcast_reduce ${TEST_DEV})
endif()

#################
# Installation
#################

if(NOT BUILD_CORE_ONLY)
    set(EXTRA_INSTALL dlprim ${EXTRA_INSTALL})
endif()


install(TARGETS 
    dlprim_core ${EXTRA_INSTALL}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION ${LIBDIR}
	ARCHIVE DESTINATION ${LIBDIR}
	PUBLIC_HEADER DESTINATION include/dlprim)

install(DIRECTORY include/dlprim DESTINATION include)

