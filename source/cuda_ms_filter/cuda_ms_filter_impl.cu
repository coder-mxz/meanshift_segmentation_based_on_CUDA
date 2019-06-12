//
// Created by iffi on 19-6-7.
//

#include "cuda_ms_filter.cu"

template
class CuMeanShift::CudaMsFilter<16, 16, 3>;

template
class CuMeanShift::CudaMsFilter<16, 16, 1>;

template
class CuMeanShift::CudaMsFilter<32, 32, 3>;


template
class CuMeanShift::CudaMsFilter<32, 32, 1>;
