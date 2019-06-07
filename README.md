# CUDA-Meanshift-segmentation

## Environment

```
gcc-5.5.0, CUDA 9.0
```

## Bundled Libraries
```
cimg-2.6.4
eigen-3.3.7
spdlog-1.3.1
```

## Run

## Run test module
使用cmake执行:
```cmake
make all
```
然后在**项目根目录**下执行
```bash
scripts/test_union_find.sh <test_path> && scripts/test_meanshift.sh <test_path2>
```
`test_path`和`test_path2`文件夹包含待测试模块的输出bin文件, 模块读取data文件夹下各个模块的输入数据, 然后用相同文件名输出到:
```bash
test_path/1.bin
test_path/2.bin
...
```
#### meanshift_filter
将滤波过后的**三通道**`CImg<float>`型图片输出, count=img.size(), data=img.data():
```objectivec
bool outputBin(const char *path, size_t count, float *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char*)&count, sizeof(size_t));
    file.write((const char*)data, count * sizeof(float));
    file.close();
    return true;
}
```

#### flooding
暂时没有测试数据

#### union_find
将labels数组按如下方式输出, data=labels
```objectivec
bool outputBin(const char *path, size_t count, int *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char*)&count, sizeof(size_t));
    file.write((const char *) data, count * sizeof(int));
    file.close();
    return true;
}
```
