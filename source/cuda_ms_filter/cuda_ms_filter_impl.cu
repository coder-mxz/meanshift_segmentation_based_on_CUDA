//
// Created by iffi on 19-6-7.
//

#include "cuda_ms_filter.cu"
template class CuMeanShift::CudaUnionFind<16, 16, 3, true>;
template class CuMeanShift::CudaUnionFind<16, 16, 3, false>;
template class CuMeanShift::CudaUnionFind<16, 16, 1, true>;
template class CuMeanShift::CudaUnionFind<16, 16, 1, false>;
template class CuMeanShift::CudaUnionFind<32, 32, 3, true>;
template class CuMeanShift::CudaUnionFind<32, 32, 3, false>;
template class CuMeanShift::CudaUnionFind<32, 32, 1, true>;
template class CuMeanShift::CudaUnionFind<32, 32, 1, false>;