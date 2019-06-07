set(CMAKE_CXX_STANDARD 14)
add_library(cuda_flooding SHARED cuda_flooding.cu)
set_target_properties(cuda_flooding PROPERTIES LANGUAGE CXX C CUDA)
target_include_directories(cuda_flooding PRIVATE ../../include/)