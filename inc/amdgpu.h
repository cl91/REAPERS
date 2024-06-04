#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

#ifndef __HIPCC__
#error "Do not include this file unless compiling on AMDGPU."
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include <hipsolver/hipsolver.h>

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
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
#define cublasCaxpy_64 hipblasCaxpy_v2_64
#define cublasSscal_64 hipblasSscal_v2_64
#define cublasCscal_64 hipblasCscal_v2_64
#define cublasCdotc_64 hipblasCdotc_v2_64
#define cublasScnrm2_64 hipblasScnrm2_v2_64
#define cublasCgemv_64 hipblasCgemv_v2_64
#define cublasCgemm_64 hipblasCgemm_v2_64
#define cublasZaxpy_64 hipblasZaxpy_v2_64
#define cublasDscal_64 hipblasDscal_v2_64
#define cublasZscal_64 hipblasZscal_v2_64
#define cublasZdotc_64 hipblasZdotc_v2_64
#define cublasDznrm2_64 hipblasDznrm2_v2_64
#define cublasZgemv_64 hipblasZgemv_v2_64
#define cublasZgemm_64 hipblasZgemm_v2_64
#define cublasFillMode_t hipblasFillMode_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#define curandGenerator_t hiprandGenerator_t
#define curandStatus_t hiprandStatus_t
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS

#define cusolverDnHandle_t hipsolverDnHandle_t
#define cusolverStatus_t hipsolverStatus_t
#define cusolverDnParams_t hipsolverDnParams_t
#define cusolverEigMode_t hipsolverEigMode_t
