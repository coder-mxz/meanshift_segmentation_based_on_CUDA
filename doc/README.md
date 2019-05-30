所有模块都按如下方式包装:
```c++

namespace CuMeanShift {
    class XXX {
    };
}

```

在使用clion等ide时, 为了能够自动补全cuda的函数, 可以在你的cu文件中包含<cuda_fake_headers.h>
另外需要在文件夹下新建一个main.cpp文件(clion不能正确处理主文件只有cu的情况), 然后对CMakeLists.txt作如下***临时***修改:
```cmake
add_library(xxx SHARED main.cpp xxx.cu)
```
commit前记得改回来
然后你的cu文件中可以这么写:
```c++
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif
...
```
如果要使用Eigen库:
```c++
#include <Eigen/...>
```
如果要使用CImg:
```c++
#include <CImg.h>
```
如果要使用spdlog
```c++
#include <spdlog/...>
```