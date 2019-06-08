#include "cuda_flooding.cu"
template class CuMeanShift::CudaFlooding<16, 16, 3, 4>;
// template class CuMeanShift::CudaFlooding<16, 16, 3, 4>;
template class CuMeanShift::CudaFlooding<16, 16, 1, 4>;
// template class CuMeanShift::CudaFlooding<16, 16, 1, 4>;
template class CuMeanShift::CudaFlooding<32, 32, 3, 4>;
// template class CuMeanShift::CudaFlooding<32, 32, 3, 4>;
template class CuMeanShift::CudaFlooding<32, 32, 1, 4>;
// template class CuMeanShift::CudaFlooding<32, 32, 1, 4>;