#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

#ifndef __HIPCC__
#error "Do not include this file unless compiling on AMDGPU."
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

// We only use Eigen3 for host matrix operations so disable CUDA and HIP support for Eigen3
#define EIGEN_NO_CUDA
#define EIGEN_NO_HIP

#define HIPBLAS_V2

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include <hipsolver/hipsolver.h>

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMalloc hipMalloc
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaSetDevice hipSetDevice
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cuComplex hipComplex
#define cuDoubleComplex hipDoubleComplex
#define cuCmul hipCmul
#define cuCmulf hipCmulf
#define make_cuComplex make_hipComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define CUDA_R_32F HIP_R_32F
#define CUDA_C_32F HIP_C_32F
#define CUDA_R_64F HIP_R_64F
#define CUDA_C_64F HIP_C_64F

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasCreate hipblasCreate
#define cublasCaxpy_64 hipblasCaxpy_64
#define cublasSscal_64 hipblasSscal_64
#define cublasCscal_64 hipblasCscal_64
#define cublasCdotc_64 hipblasCdotc_64
#define cublasScnrm2_64 hipblasScnrm2_64
#define cublasCgemv_64 hipblasCgemv_64
/* TODO: When hipBLAS supports 64-bit interfaces for level 2 APIs,
 * change the following function to hipblasCgemm_64. */
#define cublasCgemm_64 hipblasCgemm
#define cublasZaxpy_64 hipblasZaxpy_64
#define cublasDscal_64 hipblasDscal_64
#define cublasZscal_64 hipblasZscal_64
#define cublasZdotc_64 hipblasZdotc_64
#define cublasDznrm2_64 hipblasDznrm2_64
#define cublasZgemv_64 hipblasZgemv_64
/* TODO: When hipBLAS supports 64-bit interfaces for level 2 APIs,
 * change the following function to hipblasZgemm_64. */
#define cublasZgemm_64 hipblasZgemm
#define cublasFillMode_t hipblasFillMode_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER

#define curandGenerator_t hiprandGenerator_t
#define curandStatus_t hiprandStatus_t
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble
#define curandCreateGenerator hiprandCreateGenerator
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT

#define cusolverStatus_t hipsolverStatus_t
#define cusolverEigMode_t hipsolverEigMode_t
#define cusolverDnHandle_t hipsolverHandle_t
#define cusolverDnParams_t hipsolverDnParams_t
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnCreateParams hipsolverDnCreateParams
#define cusolverDnDestroyParams hipsolverDnDestroyParams
#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR

/* TODO: When hipSOLVER supports the DnX interface, change the following
 * accordingly. For now we use the old syevd interface (deprecated in CUDA 12). */
template<typename Ty>
inline cusolverStatus_t
cusolverDnXsyevd_bufferSize(cusolverDnHandle_t hnd, cusolverDnParams_t params,
			    cusolverEigMode_t jobz, cublasFillMode_t uplo,
			    size_t dim, int ty, Ty *mat, size_t rdim, int rty, float *vals,
			    int rrty, size_t *dworksz, size_t *lworksz)
{
    int size{};
    auto ret = hipsolverDnCheevd_bufferSize(hnd, jobz, uplo, dim, (hipComplex *)mat,
					    rdim, vals, &size);
    *dworksz = size;
    return ret;
}

template<typename Ty>
inline cusolverStatus_t
cusolverDnXsyevd_bufferSize(cusolverDnHandle_t hnd, cusolverDnParams_t params,
			    cusolverEigMode_t jobz, cublasFillMode_t uplo,
			    size_t dim, int ty, Ty *mat, size_t rdim, int rty, double *vals,
			    int rrty, size_t *dworksz, size_t *lworksz)
{
    int size{};
    auto ret = hipsolverDnZheevd_bufferSize(hnd, jobz, uplo, dim, (hipDoubleComplex *)mat,
					    rdim, vals, &size);
    *dworksz = size;
    return ret;
}

template<typename Ty>
inline cusolverStatus_t
cusolverDnXsyevd(cusolverDnHandle_t hnd, cusolverDnParams_t params,
		 cusolverEigMode_t jobz, cublasFillMode_t uplo,
		 size_t dim, int ty, Ty *mat, size_t rdim, int rty, float *vals,
                 int rrty, void *d_work, int dworksz, void *l_work, int lworksz,
                 int *d_info)
{
    return hipsolverDnCheevd(hnd, jobz, uplo, dim, (hipComplex *)mat,
			     rdim, vals, (hipComplex *)d_work, dworksz, d_info);
}

template<typename Ty>
inline cusolverStatus_t
cusolverDnXsyevd(cusolverDnHandle_t hnd, cusolverDnParams_t params,
		 cusolverEigMode_t jobz, cublasFillMode_t uplo,
		 size_t dim, int ty, Ty *mat, size_t rdim, int rty, double *vals,
                 int rrty, void *d_work, int dworksz, void *l_work, int lworksz,
                 int *d_info)
{
    return hipsolverDnZheevd(hnd, jobz, uplo, dim, (hipDoubleComplex *)mat,
			     rdim, vals, (hipDoubleComplex *)d_work, dworksz, d_info);
}
