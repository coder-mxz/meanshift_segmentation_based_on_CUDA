#include "cuda_flooding.cu"
template class CuMeanShift::CudaFlooding<16, 16, 3, 4>;
template class CuMeanShift::CudaFlooding<16, 16, 1, 4>;
template class CuMeanShift::CudaFlooding<32, 32, 3, 4>;
template class CuMeanShift::CudaFlooding<32, 32, 1, 4>;

template class CuMeanShift::CudaFlooding<16, 16, 3, 8>;
template class CuMeanShift::CudaFlooding<16, 16, 1, 8>;
template class CuMeanShift::CudaFlooding<32, 32, 3, 8>;
template class CuMeanShift::CudaFlooding<32, 32, 1, 8>;

template class CuMeanShift::CudaFlooding<16, 16, 3, 16>;
template class CuMeanShift::CudaFlooding<16, 16, 1, 16>;
template class CuMeanShift::CudaFlooding<32, 32, 3, 16>;
template class CuMeanShift::CudaFlooding<32, 32, 1, 16>;

template class CuMeanShift::CudaFlooding<16, 16, 3, 32>;
template class CuMeanShift::CudaFlooding<16, 16, 1, 32>;
