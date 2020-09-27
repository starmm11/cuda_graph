cmake_minimum_required(VERSION 3.17)
project(cuda_workshop CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_FLAGS "-gencode=arch=compute_50,code=sm_50")
add_executable(cuda_workshop task12_thrust.cu)


set_target_properties(
        cuda_workshop
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)