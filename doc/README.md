## 项目说明
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

## 接口
#### main.cpp: 

1.使用CImg读取图像(读取出来的图像为uint8_t, 1个/3个通道)

2.使用cudaMallocPitch分配2D内存:

> 第一块data_1用于存放原图像数据, 注意可能有多个通道 \
> 第二块data_2用于存放float数据, 通道数=3 \
> 第三块data_3和data_2一致 \
> 第四块label_1为大小和原图像一致, 单通道的int数组 \
> 第五块label_2和label_1一致 \
> 第六块label_count为1个int大小

假设图像大小为w*h, x(x=1,3)个通道, 则:
> data_1: 每行 w * x * sizeof(uint8_t), 共h行 \
> data_2, data_3: 每行 w * sizeof(float), 共h * x行 \
> label_1, label_2: 每行 w * sizeof(int), 共h行 

3.使用cudaMemcpy2D将图像数据拷贝到data_1

4.使用一个cuda核函数将原数据(uchar1/uchar3)转换为三通道浮点数据(float)

单通道灰度数据转换: Gray=100, 则转换为R=G=B=Gray=100

5.调用Meanshift滤波:
> 需要给出区域半径等 \
> 输入data_2, data_2用texture mem存取 \
> 输出到data_3

6.调用Flooding:
> 需要给出循环次数等 \
> 输入data_3, data_3用texture mem存取, 邻域访问用shared mem优化 \
> 输出到label_1

7.调用union_find:
> 如果是串行版本的, union_find首先将数据从label_1, data_3拷贝回主机, 调用union_find函数
> 会输出一个新的label数组(2D, 和原图像大小一致), 以及不同的label的总数(如:400种)

> 如果是并行版本的, 直接把label_1, data_3传入给核函数
> 结果会输出到label_2, 将结果拷贝回主机, label总数会输出到label_count, 拷贝回主机

8.调用着色函数:
> 根据不同的label总数生成RGB颜色序列, 然后创建一个新的CImg图像, 根据label数组填入对应的RGB值, 保存图像

#### cuda_ms_filter
```c++
texture<float, 2, cudaReadModeElementType> in_tex; //存储图像输入

template<int disrange=13, int max_iter=5>
__kernel__ void mean_shift(float *output,
                           int channels,
                           int pitch, 
                           int width,
                           int height,
                           float color_range,
                           float min_shift)
}
```
#### cuda_flooding
```c++
texture<float, 2, cudaReadModeElementType> in_tex; //存储图像输入
__shared__ neighbor_pixels[...]; //存储邻域像素值

template<int radius=4>
__kernel__ void flooding(int *output, 
                         int channels,
                         int pitch, 
                         int width,
                         int height,
                         int row_offset,
                         int col_offset,
                         float color_range)
```
#### cuda_union_find
```c++
texture<float, 2, cudaReadModeElementType> in_tex; //存储图像输入
__shared__ blk_labels[...];
__shared__ blk_pixels[...]; //存储邻域像素值

// 还没弄完
...

```
#### union_find
```c++
略
```