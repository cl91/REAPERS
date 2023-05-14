/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    reapers.h

Abstract:

    REAPERS: A REAsonably PERformant Spin model simulator

Revision History:

    2023-01-03  File created

--*/

#pragma once

#include <complex>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <exception>
#include <memory>
#include <random>
#include <tuple>
#include <set>
#include <vector>

#ifdef REAPERS_NOGPU
#define DEVHOST
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cuda/std/complex>
#define DEVHOST __device__ __host__
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#pragma nv_diag_suppress 20236
#endif

#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

#define _REAPERS_H_

namespace REAPERS {

#if defined(REAPERS_FP64) && defined(REAPERS_FP32)
#error "You can only define one of REAPERS_FP64 or REAPERS_FP32"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#elif defined(REAPERS_FP64)
using DefFpType = double;
#else
// If REAPERS_FP64 is not defined, we default to fp32
using DefFpType = float;
#endif

#ifdef REAPERS_NOGPU
template<typename FpType = DefFpType>
using complex = std::complex<FpType>;
#else
template<typename FpType = DefFpType>
using complex = cuda::std::complex<FpType>;
#endif	// REAPERS_NOGPU

template<typename FpType = DefFpType>
auto epsilon = std::numeric_limits<FpType>::epsilon;

#include "except.h"
#include "ops.h"
#include "cpuimpl.h"

#ifdef REAPERS_NOGPU
using DefImpl = CPUImpl;
#else
#include "gpuimpl.h"
#endif	// REAPERS_NOGPU

template<typename FpType = DefFpType>
using SumOps = DefImpl::SumOps<FpType>;

// Define a helper function to convert a SpinOp to its matrix form
template<typename FpType, typename T>
typename SumOps<FpType>::MatrixType get_matrix(const T &op, int len);

template<typename FpType>
inline typename SumOps<FpType>::MatrixType get_matrix(const SpinOp<FpType> &op,
						      int len) {
    SumOps<FpType> ops(op);
    return ops.get_matrix(len);
}

template<typename FpType>
inline typename SumOps<FpType>::MatrixType get_matrix(const SumOps<FpType> &ops,
						      int len) {
    return ops.get_matrix(len);
}

#include "state.h"

#ifdef REAPERS_USE_PARITY_BLOCKS
#include "blkops.h"
#else
#include "fermions.h"
#endif

namespace Model {
#include "syk.h"
}
}
