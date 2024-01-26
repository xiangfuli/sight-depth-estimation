cmake_minimum_required(VERSION 3.17)
project(sight_depth_estimation LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
message(STATUS "cuda version: " ${CMAKE_CUDA_COMPILER})

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

# add_library(sight_depth_estimation_cuda STATIC
# )
# set_target_properties(sight_depth_estimation_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(depth_est src/examples/depth_est.cpp)
target_link_libraries(depth_est PRIVATE CUDA::cudart)

