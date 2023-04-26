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
#include <curand.h>
#include <cuda/std/complex>
#define DEVHOST __device__ __host__
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#pragma nv_diag_suppress 20236
#endif

#include <Eigen/Eigen>
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
template<typename T>
using complex = std::complex<T>;
#else
template<typename T>
using complex = cuda::std::complex<T>;
#endif

template<typename FpType>
auto epsilon = std::numeric_limits<FpType>::epsilon;

#include "except.h"
#include "ops.h"
#include "cpuimpl.h"

#ifdef REAPERS_NOGPU
using DefImpl = CPUImpl;
#else
#include "gpuimpl.h"
#endif	// REAPERS_NOGPU

template<typename FpType>
using SumOps = DefImpl::SumOps<FpType>;

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
