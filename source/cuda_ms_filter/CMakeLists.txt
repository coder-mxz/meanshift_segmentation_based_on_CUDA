set(CMAKE_CXX_STANDARD 14)

add_library(cuda_ms_filter SHARED cuda_ms_filter_impl.cu)
add_executable(cuda_ms_filter_test main.cpp)

################################################################
# Configuration for library
################################################################
set_target_properties(cuda_ms_filter PROPERTIES LANGUAGE CXX C CUDA)
target_link_libraries(cuda_ms_filter ${CUDA_LIBRARIES})
target_include_directories(cuda_ms_filter PRIVATE ../../include/cuda_ms_filter)
target_include_directories(cuda_ms_filter PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

################################################################
# Configuration for test binary
################################################################
set_target_properties(cuda_ms_filter_test PROPERTIES LANGUAGE CXX C CUDA)
target_link_libraries(cuda_ms_filter_test
        cuda_ms_filter
        m pthread X11
        ${CUDA_LIBRARIES})
target_include_directories(cuda_ms_filter_test PRIVATE ../../include/cuda_ms_filter)
target_include_directories(cuda_ms_filter_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})