set(CMAKE_CXX_STANDARD 14)

add_library(cuda_union_find SHARED cuda_union_find.cu main.cpp)
set_target_properties(cuda_ms_filter PROPERTIES LANGUAGE CXX C CUDA)
target_link_libraries(cuda_union_find ${CUDART_LIBRARY})
target_include_directories(cuda_union_find PRIVATE ../../include/cuda_union_find)
target_include_directories(cuda_union_find PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
