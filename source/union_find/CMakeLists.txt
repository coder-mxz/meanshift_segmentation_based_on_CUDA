set(CMAKE_CXX_STANDARD 14)

################################################################
# Configuration for pth_union_find
################################################################
add_library(union_find SHARED union_find.cpp union_find.h)

################################################################
# Configuration for pth_union_find_test
################################################################
add_executable(pth_union_find_test main.cpp)
target_link_libraries(pth_union_find_test m pthread X11)