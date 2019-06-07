set(CMAKE_CXX_STANDARD 14)
add_library(union_find SHARED union_find.cpp)
target_include_directories(union_find PRIVATE ../../include/union_find)