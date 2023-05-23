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
#include <concepts>

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

    template<typename T>
    concept RealScalar = std::floating_point<T>;

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
    template<RealScalar FpType = DefFpType>
    using complex = std::complex<FpType>;
    #else
    template<RealScalar FpType = DefFpType>
    using complex = cuda::std::complex<FpType>;
    #endif	// REAPERS_NOGPU

    template<RealScalar FpType = DefFpType>
    auto epsilon = std::numeric_limits<FpType>::epsilon;

    template<typename T>
    concept ComplexScalar = std::convertible_to<T, complex<double>>
	|| std::convertible_to<T, std::complex<double>>;

    template<typename T>
    concept ScalarType = RealScalar<T> || ComplexScalar<T>;

    #include "except.h"
    #include "ops.h"
    #include "cpuimpl.h"

    #ifdef REAPERS_NOGPU
    using DefImpl = CPUImpl;
    #else
    #include "gpuimpl.h"
    #endif	// REAPERS_NOGPU

    template<RealScalar FpType = DefFpType>
    using SumOps = DefImpl::SumOps<FpType>;

    template<RealScalar FpType = DefFpType>
    inline SumOps<FpType> operator+(const SpinOp<FpType> &op0,
				    const SpinOp<FpType> &op1) {
	return SumOps<FpType>(op0) + op1;
    }

    template<RealScalar FpType = DefFpType>
    inline SumOps<FpType> operator-(const SpinOp<FpType> &op0,
				    const SpinOp<FpType> &op1) {
	return SumOps<FpType>(op0) - op1;
    }

    #include "blkops.h"
    #include "state.h"

    namespace Model {
    #include "fermions.h"
    #include "syk.h"
    }
}
