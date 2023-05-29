/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    krnl.cu

Abstract:

    This source file contains the CUDA kernels of the vector operations.

Revision History:

    2023-03-02  File created

--*/

#include "reapers.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace REAPERS;

__global__ static void krnl_ftod_vec(CudaComplexPtr<double> a,
				     CudaComplexConstPtr<float> b)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x = static_cast<double>(b[id].x);
    a[id].y = static_cast<double>(b[id].y);
}

__global__ static void krnl_dtof_vec(CudaComplexPtr<float> a,
				     CudaComplexConstPtr<double> b)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x = __double2float_rz(b[id].x);
    a[id].y = __double2float_rz(b[id].y);
}

// Compute a += b
template<typename ComplexType>
__global__ static void krnl_vec_add(ComplexType *a,
				    const ComplexType *b) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x += b[id].x;
    a[id].y += b[id].y;
}

// Compute a += b + c
template<typename ComplexType>
__global__ static void krnl_vec_add3(ComplexType *a,
				     const ComplexType *b,
				     const ComplexType *c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id].x = b[id].x + c[id].x;
    a[id].y = b[id].y + c[id].y;
}

template<RealScalar FpType>
__device__ static __inline CudaComplexType<FpType> cmul(complex<FpType> z1,
							CudaComplexType<FpType> z2)
{
    auto z = CudaComplex<FpType>::mkcomplex(z1.real(), z1.imag());
    return CudaComplex<FpType>::cuCmul(z, z2);
}

template<typename ComplexType>
__device__ static __inline ComplexType cmul(ComplexType z1, ComplexType z2);
template<>
__device__ __inline cuComplex cmul(cuComplex z1, cuComplex z2) {
    return CudaComplex<float>::cuCmul(z1, z2);
}
template<>
__device__ __inline cuDoubleComplex cmul(cuDoubleComplex z1,
					 cuDoubleComplex z2) {
    return CudaComplex<double>::cuCmul(z1, z2);
}

template<RealScalar FpType>
__global__ static void krnl_apply_ops(CudaComplexPtr<FpType> res,
				      SpinOp<FpType> *ops, int numops,
				      CudaComplexConstPtr<FpType> vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < numops; j++) {
	typename SpinOp<FpType>::StateType n = 0;
	auto coeff = ops[j].apply(i, n);
	auto val = cmul(coeff, vec[n]);
	res[i].x += val.x;
	res[i].y += val.y;
    }
}

template<typename ComplexType>
__global__ static void krnl_eye(ComplexType *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i*(blockDim.x*gridDim.x + 1)].x = 1.0;
}


template<typename ComplexType>
__global__ void krnl_copy(ComplexType *diag,
			  const ComplexType *mat)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    diag[i].x = mat[i*(blockDim.x*gridDim.x + 1)].x;
    diag[i].y = mat[i*(blockDim.x*gridDim.x + 1)].y;
}

template<RealScalar FpType>
__global__ void krnl_vec_exp(CudaComplexPtr<FpType> res,
			     const FpType *v, FpType re, FpType im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    FpType expre = exp(v[i] * re);
    FpType vim = v[i] * im;
    FpType cosim = cos(vim);
    FpType sinim = sin(vim);
    res[i].x = expre * cosim;
    res[i].y = expre * sinim;
}

template<typename ComplexType>
__global__ void krnl_scale_vecs(ComplexType *res,
				const ComplexType *v,
				const ComplexType *lambda,
				unsigned int shift)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = cmul(lambda[i>>shift], v[i]);
}

#define cuda_sync()				\
    CUDA_CALL(cudaPeekAtLastError());		\
    CUDA_CALL(cudaDeviceSynchronize());

// Copy v1 into v0, converting float to double.
void GPUImpl::copy_vec(VecSizeType<double> size, BufType<double> v0,
		       ConstBufType<float> v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_ftod_vec<<<gridsize, blocksize>>>((CudaComplexPtr<double>)v0.dev_ptr,
					   (CudaComplexConstPtr<float>)v1.dev_ptr);
    cuda_sync();
}

// Copy v1 into v0, converting double to float by rounding toward zero.
void GPUImpl::copy_vec(VecSizeType<float> size, BufType<float> v0,
		       ConstBufType<double> v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_dtof_vec<<<gridsize, blocksize>>>((CudaComplexPtr<float>)v0.dev_ptr,
					   (CudaComplexConstPtr<double>)v1.dev_ptr);
    cuda_sync();
}

// Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
template<RealScalar FpType>
void GPUImpl::add_vec(size_t size, complex<FpType> *v0,
		      const complex<FpType> *v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_vec_add<<<gridsize, blocksize>>>((CudaComplexPtr<FpType>)v0,
					  (CudaComplexConstPtr<FpType>)v1);
    cuda_sync();
}

// Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
template<RealScalar FpType>
void GPUImpl::add_vec(size_t size, complex<FpType> *res,
		      const complex<FpType> *v0, const complex<FpType> *v1) {
    int blocksize, gridsize;
    get_block_grid_size(size, blocksize, gridsize);
    krnl_vec_add3<<<gridsize, blocksize>>>((CudaComplexPtr<FpType>)res,
					   (CudaComplexConstPtr<FpType>)v0,
					   (CudaComplexConstPtr<FpType>)v1);
    cuda_sync();
}

// Compute res = ops * vec. res and vec are assumed to be different.
template<RealScalar FpType>
void GPUImpl::apply_ops(typename SpinOp<FpType>::IndexType len, BufType<FpType> res,
			const DevSumOps<FpType> &ops, ConstBufType<FpType> vec) {
    ops.upload(len);
    // dev_mat can be generated due to calls to get_eigensystem etc, so we
    // must check dev_ops to see if the Hamiltonian is sparse (ie. has fewer
    // terms than 1<<len).
    if (ops.dev_ops != 0) {
	int blocksize, gridsize;
	get_block_grid_size(1ULL << len, blocksize, gridsize);
	krnl_apply_ops<<<gridsize, blocksize>>>(
	    (CudaComplexPtr<FpType>)res.dev_ptr,
	    ops.dev_ops, ops.ops.size(),
	    (CudaComplexConstPtr<FpType>)vec.dev_ptr);
	cuda_sync();
    } else {
	mat_mul(1ULL << len, res, *ops.dev_mat, vec);
    }
}

// Set the given matrix to the identity matrix.
template<RealScalar FpType>
void GPUImpl::eye(typename DevSumOps<FpType>::MatrixType &res) {
    assert(res.rowdim == res.coldim);
    CUDA_CALL(cudaMemset(res.dev_ptr, 0, res.datasize()));
    int blocksize, gridsize;
    get_block_grid_size(res.rowdim, blocksize, gridsize);
    krnl_eye<<<gridsize, blocksize>>>((CudaComplexPtr<FpType>)res.dev_ptr);
    cuda_sync();
}

template<RealScalar FpType>
complex<FpType> GPUImpl::mat_tr(const typename DevSumOps<FpType>::MatrixType &mat) {
    assert(mat.coldim == mat.rowdim);
    typename DevSumOps<FpType>::MatrixType diag(mat.rowdim, 1);
    int blocksize, gridsize;
    get_block_grid_size(mat.rowdim, blocksize, gridsize);
    krnl_copy<<<gridsize, blocksize>>>((CudaComplexPtr<FpType>)diag.dev_ptr,
				       (CudaComplexConstPtr<FpType>)mat.dev_ptr);
    cuda_sync();
    thrust::device_ptr<complex<FpType>> d(diag.dev_ptr);
    return thrust::reduce(d, d + mat.rowdim, complex<FpType>{},
			  thrust::plus<complex<FpType>>());
}

template<RealScalar FpType>
void GPUImpl::exp_vec(typename DevSumOps<FpType>::MatrixType &res,
		      const FpType *v, complex<FpType> c) {
    assert(res.coldim == 1);
    int blocksize, gridsize;
    get_block_grid_size(res.rowdim, blocksize, gridsize);
    krnl_vec_exp<<<gridsize, blocksize>>>((CudaComplexPtr<FpType>)res.dev_ptr,
					  v, c.real(), c.imag());
    cuda_sync();
}

template<RealScalar FpType>
void GPUImpl::scale_vecs(typename DevSumOps<FpType>::MatrixType &res,
			 const typename DevSumOps<FpType>::MatrixType &v,
			 const typename DevSumOps<FpType>::MatrixType &lambda,
			 unsigned int len) {
    size_t dim{1ULL << len};
    assert(dim == res.rowdim);
    assert(dim == res.coldim);
    assert(dim == v.rowdim);
    assert(dim == v.coldim);
    assert(dim == lambda.rowdim);
    assert(lambda.coldim == 1);
    int blocksize, gridsize;
    get_block_grid_size(dim * dim, blocksize, gridsize);
    krnl_scale_vecs<<<gridsize, blocksize>>>(
	(CudaComplexPtr<FpType>)res.dev_ptr,
	(CudaComplexConstPtr<FpType>)v.dev_ptr,
	(CudaComplexConstPtr<FpType>)lambda.dev_ptr, len);
    cuda_sync();
}

// We need the explicit template instantiations below due to link error.
template void GPUImpl::add_vec(size_t size, complex<float> *v0,
			       const complex<float> *v1);
template void GPUImpl::add_vec(size_t size, complex<double> *v0,
			       const complex<double> *v1);
template void GPUImpl::add_vec(size_t size, complex<float> *res,
			       const complex<float> *v0, const complex<float> *v1);
template void GPUImpl::add_vec(size_t size, complex<double> *res,
			       const complex<double> *v0, const complex<double> *v1);
template void GPUImpl::apply_ops(typename SpinOp<float>::IndexType len,
				 BufType<float> res,
				 const DevSumOps<float> &ops,
				 ConstBufType<float> vec);
template void GPUImpl::apply_ops(typename SpinOp<double>::IndexType len,
				 BufType<double> res,
				 const DevSumOps<double> &ops,
				 ConstBufType<double> vec);
template void GPUImpl::eye<float>(typename DevSumOps<float>::MatrixType &res);
template void GPUImpl::eye<double>(typename DevSumOps<double>::MatrixType &res);
template complex<float> GPUImpl::mat_tr(const typename DevSumOps<float>::MatrixType &);
template complex<double> GPUImpl::mat_tr(const typename DevSumOps<double>::MatrixType &);
template void GPUImpl::exp_vec(typename DevSumOps<float>::MatrixType &res,
			       const float *v, complex<float> c);
template void GPUImpl::exp_vec(typename DevSumOps<double>::MatrixType &res,
			       const double *v, complex<double> c);
template void GPUImpl::scale_vecs<float>(typename DevSumOps<float>::MatrixType &,
					 const typename DevSumOps<float>::MatrixType &,
					 const typename DevSumOps<float>::MatrixType &,
					 unsigned int len);
template void GPUImpl::scale_vecs<double>(typename DevSumOps<double>::MatrixType &,
					  const typename DevSumOps<double>::MatrixType &,
					  const typename DevSumOps<double>::MatrixType &,
					  unsigned int len);