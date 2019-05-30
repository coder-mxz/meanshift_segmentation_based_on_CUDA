set(CMAKE_CXX_STANDARD 14)
add_library(cuda_ms_filter SHARED cuda_ms_filter.cu)
set_target_properties(cuda_ms_filter PROPERTIES LANGUAGE CXX C CUDA)
target_include_directories(cuda_ms_filter PRIVATE ../../include/cuda_ms_filter)