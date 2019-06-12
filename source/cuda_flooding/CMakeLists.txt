set(CMAKE_CXX_STANDARD 14)

if(CUDA_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -rdynamic -lineinfo -g -G")
endif()

#add_library(cuda_flooding SHARED cuda_flooding_impl.cu cuda_flooding.cu main.cpp)
add_library(cuda_flooding SHARED cuda_flooding_impl.cu)
add_executable(cuda_flooding_test main.cpp)

################################################################
# Configuration for library
################################################################
set_target_properties(cuda_flooding PROPERTIES LANGUAGE CXX C CUDA)
target_link_libraries(cuda_flooding ${CUDA_LIBRARIES})
target_include_directories(cuda_flooding PRIVATE ../../include)
target_include_directories(cuda_flooding PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

################################################################
# Configuration for test binary
################################################################
set_target_properties(cuda_flooding_test PROPERTIES LANGUAGE CXX C CUDA)
target_link_libraries(cuda_flooding_test
        cuda_flooding
        m pthread X11
        ${CUDA_LIBRARIES})
target_include_directories(cuda_flooding_test PRIVATE ../../include)
target_include_directories(cuda_flooding_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})