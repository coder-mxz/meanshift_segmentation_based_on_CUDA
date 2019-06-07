set(CMAKE_CXX_STANDARD 14)

add_library(cuda_union_find SHARED cuda_union_find_impl.cu)
add_executable(cuda_union_find_test main.cpp)

################################################################
# Configuration for library
################################################################
set_target_properties(cuda_union_find PROPERTIES LANGUAGE CXX C CUDA)
target_link_libraries(cuda_union_find ${CUDA_LIBRARIES})
target_include_directories(cuda_union_find PRIVATE ../../include/cuda_union_find)
target_include_directories(cuda_union_find PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

################################################################
# Configuration for test binary
################################################################
set_target_properties(cuda_union_find_test PROPERTIES LANGUAGE CXX C CUDA)
target_link_libraries(cuda_union_find_test
        cuda_union_find
        m pthread X11
        ${CUDA_LIBRARIES})
target_include_directories(cuda_union_find_test PRIVATE ../../include/cuda_union_find)
target_include_directories(cuda_union_find_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})