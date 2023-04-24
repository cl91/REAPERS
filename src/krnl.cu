/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    krnl.cu

Abstract:

    This header file contains the CUDA kernels of the vector operations.

Revision History:

    2023-03-02  File created

--*/

#include "reapers.h"

using namespace REAPERS;

template<typename FpType>
struct CudaComplex;

template<>
struct CudaComplex<float> {
    using Type = cuComplex;
    static constexpr auto mkcomplex = make_cuComplex;
    static constexpr auto cuCmul = cuCmulf;
};

template<>
struct CudaComplex<double> {
    using Type = cuDoubleComplex;
    static constexpr auto mkcomplex = make_cuDoubleComplex;
    static constexpr auto cuCmul = ::cuCmul;
};

template<typename FpType>
using CudaComplexType = typename CudaComplex<FpType>::Type;

__global__ static void krnl_ftod_vec(CudaComplexType<double> *a,
				     const CudaComplexType<float> *b)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x = static_cast<double>(b[id].x);
    a[id].y = static_cast<double>(b[id].y);
}

__global__ static void krnl_dtof_vec(CudaComplexType<float> *a,
				     const CudaComplexType<double> *b)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x = __double2float_rz(b[id].x);
    a[id].y = __double2float_rz(b[id].y);
}

// Compute a += b
template<typename FpType>
__global__ static void krnl_vec_add(CudaComplexType<FpType> *a,
				    const CudaComplexType<FpType> *b) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x += b[id].x;
    a[id].y += b[id].y;
}

// Compute a += b + c
template<typename FpType>
__global__ static void krnl_vec_add3(CudaComplexType<FpType> *a,
				     const CudaComplexType<FpType> *b,
				     const CudaComplexType<FpType> *c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x = b[id].x + c[id].x;
    a[id].y = b[id].y + c[id].y;
}

template<typename FpType>
__device__ static __inline CudaComplexType<FpType> cmul(complex<FpType> z1,
							CudaComplexType<FpType> z2)
{
    auto z = CudaComplex<FpType>::mkcomplex(z1.real(), z1.imag());
    return CudaComplex<FpType>::cuCmul(z, z2);
}

template<typename FpType>
__global__ static void krnl_apply_ops(CudaComplexType<FpType> *res,
				      SpinOp<FpType> *ops, int numops,
				      const CudaComplexType<FpType> *vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < numops; j++) {
	typename SpinOp<FpType>::StateType n = 0;
	auto coeff = ops[j].apply(i, n);
	auto val = cmul(coeff, vec[n]);
	res[i].x += val.x;
	res[i].y += val.y;
    }
}

// Copy v1 into v0, converting float to double.
void GPUImpl::copy_vec(VecSizeType<double> size, BufType<double> v0,
		       ConstBufType<float> v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_ftod_vec<<<gridsize, blocksize>>>((CudaComplexType<double> *)v0.dev_ptr,
					   (const CudaComplexType<float> *)v1.dev_ptr);
    CUDA_CALL(cudaDeviceSynchronize());
}

// Copy v1 into v0, converting double to float by rounding toward zero.
void GPUImpl::copy_vec(VecSizeType<float> size, BufType<float> v0,
		       ConstBufType<double> v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_dtof_vec<<<gridsize, blocksize>>>((CudaComplexType<float> *)v0.dev_ptr,
					   (const CudaComplexType<double> *)v1.dev_ptr);
    CUDA_CALL(cudaDeviceSynchronize());
}

// Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
template<typename FpType>
void GPUImpl::add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
		      ConstBufType<FpType> v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_vec_add<<<gridsize, blocksize>>>((CudaComplexType<FpType> *)v0.dev_ptr,
					  (const CudaComplexType<FpType> *)v1.dev_ptr);
    CUDA_CALL(cudaDeviceSynchronize());
}

// Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
template<typename FpType>
void GPUImpl::add_vec(VecSizeType<FpType> size, BufType<FpType> res,
		      ConstBufType<FpType> v0, ConstBufType<FpType> v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_vec_add3<<<gridsize, blocksize>>>((CudaComplexType<FpType> *)res.dev_ptr,
					   (const CudaComplexType<FpType> *)v0.dev_ptr,
					   (const CudaComplexType<FpType> *)v1.dev_ptr);
    CUDA_CALL(cudaDeviceSynchronize());
}

// Compute res = ops * vec. res and vec are assumed to be different.
template<typename FpType>
void GPUImpl::apply_ops(typename SpinOp<FpType>::IndexType len, BufType<FpType> res,
			const SumOps<FpType> &ops, ConstBufType<FpType> vec) {
    ops.upload(len);
    if (ops.dev_ops != 0) {
	int blocksize, gridsize;
	get_block_grid_size(1ULL << len, blocksize, gridsize);
	krnl_apply_ops<<<gridsize, blocksize>>>(
	    (CudaComplexType<FpType> *)res.dev_ptr,
	    ops.dev_ops, ops.ops.size(),
	    (const CudaComplexType<FpType> *)vec.dev_ptr);
	CUDA_CALL(cudaDeviceSynchronize());
    } else {
	assert(ops.sparse_mat != nullptr);
    }
}

// We need the explicit template instantiations below due to link error.
template void GPUImpl::apply_ops(typename SpinOp<float>::IndexType len,
				 BufType<float> res,
				 const SumOps<float> &ops,
				 ConstBufType<float> vec);
template void GPUImpl::apply_ops(typename SpinOp<double>::IndexType len,
				 BufType<double> res,
				 const SumOps<double> &ops,
				 ConstBufType<double> vec);