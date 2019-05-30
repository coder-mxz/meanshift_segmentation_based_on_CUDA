set(CMAKE_CXX_STANDARD 14)
add_library(cuda_push_relabel SHARED cuda_push_relabel.cu)
set_target_properties(cuda_push_relabel PROPERTIES LANGUAGE CXX C CUDA)
target_include_directories(cuda_push_relabel PRIVATE ../../include/cuda_push_relabel)