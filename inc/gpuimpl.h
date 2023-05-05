/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    gpuimpl.h

Abstract:

    This header file contains the CUDA implementation of vector operations.

Revision History:

    2023-01-05  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

#define CUDA_CALL(x)						\
    do {							\
	auto __REAPERS__res = (x);				\
	if (__REAPERS__res != cudaSuccess) {			\
	    throw RuntimeError(__FILE__, __LINE__, __func__,	\
			       "CUDA", __REAPERS__res);		\
	}							\
    } while(0)

#define CUBLAS_CALL(x)						\
    do {							\
	auto __REAPERS__res = (x);				\
	if (__REAPERS__res != CUBLAS_STATUS_SUCCESS) {		\
	    throw RuntimeError(__FILE__, __LINE__, __func__,	\
			       "CUBLAS", __REAPERS__res);	\
	}							\
    } while(0)

#define CURAND_CALL(x)						\
    do {							\
	auto __REAPERS__res = (x);				\
	if (__REAPERS__res != CURAND_STATUS_SUCCESS) {		\
	    throw RuntimeError(__FILE__, __LINE__, __func__,	\
			       "CURAND", __REAPERS__res);	\
	}							\
    } while(0)

// This class represents a sum of spin operators in device memory
template<typename FpType>
class DevSumOps : public HostSumOps<FpType> {
    mutable SpinOp<FpType> *dev_ops;
    mutable void *sparse_mat;
    friend class GPUImpl;

    // Upload the host summed operators to the device. If the number of operators
    // is less than 2^len, then simply copy the operators to the device side.
    // Otherwise, generate the CUSPARSE matrix representing the sum of operators.
    void upload(typename SpinOp<FpType>::IndexType len) const {
	if (dev_ops || sparse_mat) {
	    return;
	}
//	    if (ops.size() < (1ULL << len)) {
	if (true) {
	    auto size = this->ops.size() * sizeof(SpinOp<FpType>);
	    CUDA_CALL(cudaMalloc((void **)&dev_ops, size));
	    CUDA_CALL(cudaMemcpy(dev_ops, this->ops.data(),
				 size, cudaMemcpyHostToDevice));
	} else {
	    // TODO!
	    assert(false);
	}
    }

    // Free the device copy of the host summed operators. This should
    // be called after every update to the operator list, so next time
    // GPUImpl::apply_ops can upload a fresh copy of the latest SumOps.
    void release() {
	// We must call the parent class release() function.
	HostSumOps<FpType>::release();
	if (dev_ops) {
	    assert(!sparse_mat);
	    cudaFree(dev_ops);
	    dev_ops = nullptr;
	}
	if (sparse_mat) {
	    /* FREE sparse_mat */
	    assert(false);
	    sparse_mat = nullptr;
	}
    }

public:
    DevSumOps(const SpinOp<FpType> &op) : HostSumOps<FpType>(op), dev_ops(nullptr),
				       sparse_mat(nullptr) {}
    DevSumOps(const DevSumOps &ops) : HostSumOps<FpType>(ops), dev_ops(nullptr),
				sparse_mat(nullptr) {}
    template<typename FpType1>
    explicit DevSumOps(const DevSumOps<FpType1> &ops) : HostSumOps<FpType>(ops),
						  dev_ops(nullptr),
						  sparse_mat(nullptr) {}
    explicit DevSumOps(const HostSumOps<FpType> &ops) : HostSumOps<FpType>(ops),
						     dev_ops(nullptr),
						     sparse_mat(nullptr) {}
    DevSumOps(const std::initializer_list<SpinOp<FpType>> &l = {}) :
	HostSumOps<FpType>(l), dev_ops(nullptr), sparse_mat(nullptr) {}
    ~DevSumOps() { release(); }

    friend std::ostream &operator<<(std::ostream &os, const DevSumOps &s) {
	os << static_cast<const HostSumOps<FpType> &>(s) << " devptr " << s.dev_ops;
	return os;
    }
};

// Class that implements basic vector operations on the GPU using CUDA.
class GPUImpl {
    template<typename FpType>
    class DeviceVec {
	using ComplexScalar = complex<FpType>;
	ComplexScalar *dev_ptr;
	size_t dim;
	friend class GPUImpl;
	DeviceVec(const DeviceVec &) = delete;
	DeviceVec &operator=(const DeviceVec &) = delete;
	DeviceVec &operator=(DeviceVec &&rhs) = delete;
    public:
	DeviceVec(size_t dim) : dim(dim) {
	    CUDA_CALL(cudaMalloc((void **)&dev_ptr, dim * sizeof(ComplexScalar)));
	}

	DeviceVec(DeviceVec &&v) {
	    dev_ptr = v.dev_ptr;
	    v.dev_ptr = nullptr;
	}

	explicit DeviceVec(const CPUImpl::VecType<FpType> &v) : dim(v.size()) {
	    CUDA_CALL(cudaMalloc((void **)&dev_ptr, dim * sizeof(ComplexScalar)));
	    CUDA_CALL(cudaMemcpy(dev_ptr, v.get(), dim, cudaMemcpyHostToDevice));
	}

	explicit operator CPUImpl::VecType<FpType>() const {
	    CPUImpl::VecType<FpType> v(dim);
	    CUDA_CALL(cudaMemcpy(v.get(), dev_ptr, dim, cudaMemcpyDeviceToHost));
	    return v;
	}

	~DeviceVec() {
	    if (dev_ptr != nullptr) {
		cudaFree(dev_ptr);
	    }
	}

	DeviceVec &get() {
	    return *this;
	}

	const DeviceVec &get() const {
	    return *this;
	}

	friend void swap(DeviceVec &lhs, DeviceVec &rhs) {
	    std::swap(lhs.dev_ptr, rhs.dev_ptr);
	}
    };

    class GPUContext {
	friend class GPUImpl;
	cublasHandle_t hcublas;
	curandGenerator_t randgen;
	GPUContext() {
	    CUBLAS_CALL(cublasCreate(&hcublas));
	    CURAND_CALL(curandCreateGenerator(&randgen, CURAND_RNG_PSEUDO_DEFAULT));
	    std::random_device rd;
	    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randgen, rd()));
	}
    };

    inline static GPUContext ctx;

public:
    template<typename FpType>
    using VecType = DeviceVec<FpType>;

    template<typename FpType>
    using VecSizeType = typename SpinOp<FpType>::StateType;

    template<typename FpType>
    using BufType = DeviceVec<FpType> &;

    template<typename FpType>
    using ConstBufType = const DeviceVec<FpType> &;

    template<typename FpType>
    using ElemRefType = complex<FpType> &;

    template<typename FpType>
    using ElemConstRefType = const complex<FpType> &;

    template<typename FpType>
    using SumOps = DevSumOps<FpType>;

private:
    template<typename FpType>
    static curandStatus_t gen_norm(FpType *vec, size_t n);

    template<typename FpType>
    static cublasStatus_t cublas_axpy(VecSizeType<FpType> n, complex<FpType> s,
				      const complex<FpType> *x, complex<FpType> *y);

    template<typename FpType>
    static cublasStatus_t cublas_scal(VecSizeType<FpType> n, FpType s,
				      complex<FpType> *x);

    template<typename FpType>
    static cublasStatus_t cublas_scal(VecSizeType<FpType> n, complex<FpType> s,
				      complex<FpType> *x);

    template<typename FpType>
    static cublasStatus_t cublas_dotc(VecSizeType<FpType> n, complex<FpType> &res,
				      const complex<FpType> *x, const complex<FpType> *y);

    template<typename FpType>
    static cublasStatus_t cublas_nrm2(VecSizeType<FpType> n, FpType &res,
				      const complex<FpType> *x);

    static void get_block_grid_size(size_t size, int &blocksize, int &gridsize) {
	blocksize = (size > 1024ULL) ? 1024 : size;
	gridsize = size / blocksize;
    }

public:
    // Initialize the vector with zero, ie. set v0[i] to 0.0 for all i.
    template<typename FpType>
    static void zero_vec(VecSizeType<FpType> size, BufType<FpType> v) {
	CUDA_CALL(cudaMemset(v.dev_ptr, 0, size * sizeof(complex<FpType>)));
    }

    // Initialize the vector to a Haar random state.
    template<typename FpType>
    static void init_random(VecSizeType<FpType> size, BufType<FpType> v);

    // Copy v1 into v0.
    template<typename FpType>
    static void copy_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			 ConstBufType<FpType> v1) {
	CUDA_CALL(cudaMemcpy(v0.dev_ptr, v1.dev_ptr, size * sizeof(complex<FpType>),
			     cudaMemcpyDeviceToDevice));
    }

    // Copy v1 into v0, converting double to float by rounding toward zero.
    static void copy_vec(VecSizeType<float> size, BufType<float> v0,
			 ConstBufType<double> v1);

    // Copy v1 into v0, converting float to double.
    static void copy_vec(VecSizeType<double> size, BufType<double> v0,
			 ConstBufType<float> v1);

    // Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			ConstBufType<FpType> v1);

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			complex<FpType> s, ConstBufType<FpType> v1);

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0, FpType s,
			ConstBufType<FpType> v1);

    // Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> res,
			ConstBufType<FpType> v0, ConstBufType<FpType> v1);

    // Compute v *= s where s is a real scalar.
    template<typename FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v,
			  FpType s);

    // Compute v *= s where s is a complex scalar.
    template<typename FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v,
			  complex<FpType> s);

    // Compute the vector inner product of the complex vector v0 and v1. In other words,
    // compute v = <v0 | v1> = v0^\dagger \cdot v1
    template<typename FpType>
    static complex<FpType> vec_prod(VecSizeType<FpType> size, ConstBufType<FpType> v0,
				    ConstBufType<FpType> v1);

    // Compute the vector norm of the complex vector v. In other words, compute
    // n = sqrt(<v|v>) = sqrt(v^\dagger \cdot v)
    template<typename FpType>
    static FpType vec_norm(VecSizeType<FpType> size, ConstBufType<FpType> v);

    // Compute the matrix multiplication res = mat * vec, where mat is of
    // type HostSumOps::MatrixType. Here res and vec are assumed to be
    // different buffers.
    template<typename FpType>
    static void mat_mul(VecSizeType<FpType> dim, BufType<FpType> res,
			const typename HostSumOps<FpType>::MatrixType &mat,
			ConstBufType<FpType> vec) {
	// TODO
    }

    // Compute res = ops * vec. res and vec are assumed to be different.
    template<typename FpType>
    static void apply_ops(typename SpinOp<FpType>::IndexType len, BufType<FpType> res,
			  const DevSumOps<FpType> &ops, ConstBufType<FpType> vec);
};

template<>
inline curandStatus_t GPUImpl::gen_norm<float>(float *vec, size_t n) {
    return curandGenerateNormal(ctx.randgen, vec, n, 0.0, 1.0);
}
template<>
inline curandStatus_t GPUImpl::gen_norm<double>(double *vec, size_t n) {
    return curandGenerateNormalDouble(ctx.randgen, vec, n, 0.0, 1.0);
}

template<>
inline cublasStatus_t GPUImpl::cublas_axpy(VecSizeType<float> n, complex<float> s,
					   const complex<float> *x, complex<float> *y) {
    cuComplex c = make_cuComplex(s.real(), s.imag());
    return cublasCaxpy(ctx.hcublas, n, &c, (const cuComplex *)x, 1,
		       (cuComplex *)y, 1);
}
template<>
inline cublasStatus_t GPUImpl::cublas_axpy(VecSizeType<double> n, complex<double> s,
					   const complex<double> *x, complex<double> *y) {
    cuDoubleComplex c = make_cuDoubleComplex(s.real(), s.imag());
    return cublasZaxpy(ctx.hcublas, n, &c, (const cuDoubleComplex *)x, 1,
		       (cuDoubleComplex *)y, 1);
}

template<>
inline cublasStatus_t GPUImpl::cublas_scal(VecSizeType<float> n, float s,
					   complex<float> *x) {
    // n*2 might exceed 2^32 so we need the 64-bit interface for cublas
    int64_t n2 = static_cast<int64_t>(n) * 2;
    return cublasSscal_64(ctx.hcublas, n2, &s, (float *)x, 1);
}
template<>
inline cublasStatus_t GPUImpl::cublas_scal(VecSizeType<double> n, double s,
					   complex<double> *x) {
    // n*2 might exceed 2^32 so we need the 64-bit interface for cublas
    int64_t n2 = static_cast<int64_t>(n) * 2;
    return cublasDscal_64(ctx.hcublas, n2, &s, (double *)x, 1);
}
template<>
inline cublasStatus_t GPUImpl::cublas_scal(VecSizeType<float> n, complex<float> s,
					   complex<float> *x) {
    cuComplex c = make_cuComplex(s.real(), s.imag());
    return cublasCscal(ctx.hcublas, n, &c, (cuComplex *)x, 1);
}
template<>
inline cublasStatus_t GPUImpl::cublas_scal(VecSizeType<double> n, complex<double> s,
					   complex<double> *x) {
    cuDoubleComplex c = make_cuDoubleComplex(s.real(), s.imag());
    return cublasZscal(ctx.hcublas, n, &c, (cuDoubleComplex *)x, 1);
}

template<>
inline cublasStatus_t GPUImpl::cublas_dotc(VecSizeType<float> n, complex<float> &res,
					   const complex<float> *x, const complex<float> *y) {
    return cublasCdotc(ctx.hcublas, n, (const cuComplex *)x, 1,
		       (const cuComplex *)y, 1, (cuComplex *)&res);
}
template<>
inline cublasStatus_t GPUImpl::cublas_dotc(VecSizeType<double> n, complex<double> &res,
					   const complex<double> *x, const complex<double> *y) {
    return cublasZdotc(ctx.hcublas, n, (const cuDoubleComplex *)x, 1,
		       (const cuDoubleComplex *)y, 1, (cuDoubleComplex *)&res);
}

template<>
inline cublasStatus_t GPUImpl::cublas_nrm2(VecSizeType<float> n, float &res,
					   const complex<float> *x) {
    return cublasScnrm2(ctx.hcublas, n, (const cuComplex *)x, 1, &res);
}
template<>
inline cublasStatus_t GPUImpl::cublas_nrm2(VecSizeType<double> n, double &res,
					   const complex<double> *x) {
    return cublasDznrm2(ctx.hcublas, n, (const cuDoubleComplex *)x, 1, &res);
}

// Initialize the vector to a Haar random state.
template<typename FpType>
inline void GPUImpl::init_random(VecSizeType<FpType> size, BufType<FpType> v) {
    // Generate 2*size random numbers
    CURAND_CALL(gen_norm((FpType *)v.dev_ptr, 2*size));
    // Normalize the resulting vector
    FpType c = 1/vec_norm(size, v);
    scale_vec(size, v, c);
}

// Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
template<typename FpType>
inline void GPUImpl::add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			     complex<FpType> s, ConstBufType<FpType> v1) {
    CUBLAS_CALL(cublas_axpy(size, s, v1.dev_ptr, v0.dev_ptr));
}

// Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
template<typename FpType>
inline void GPUImpl::add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			     FpType s, ConstBufType<FpType> v1) {
    CUBLAS_CALL(cublas_axpy(size, complex<FpType>(s), v1.dev_ptr, v0.dev_ptr));
}

// Compute v *= s where s is a real scalar.
template<typename FpType>
inline void GPUImpl::scale_vec(VecSizeType<FpType> size, BufType<FpType> v, FpType s) {
    CUBLAS_CALL(cublas_scal(size, s, v.dev_ptr));
}

// Compute v *= s where s is a complex scalar.
template<typename FpType>
inline void GPUImpl::scale_vec(VecSizeType<FpType> size, BufType<FpType> v,
			       complex<FpType> s) {
    CUBLAS_CALL(cublas_scal(size, s, v.dev_ptr));
}

// Compute the vector inner product of the complex vector v0 and v1. In other words,
// compute v = <v0 | v1> = v0^\dagger \cdot v1
template<typename FpType>
inline complex<FpType> GPUImpl::vec_prod(VecSizeType<FpType> size,
					 ConstBufType<FpType> v0,
					 ConstBufType<FpType> v1) {
    complex<FpType> res = 0.0;
    CUBLAS_CALL(cublas_dotc(size, res, v0.dev_ptr, v1.dev_ptr));
    return res;
}

// Compute the vector norm of the complex vector v. In other words, compute
// n = sqrt(<v|v>) = sqrt(v^\dagger \cdot v)
template<typename FpType>
inline FpType GPUImpl::vec_norm(VecSizeType<FpType> size, ConstBufType<FpType> v) {
    FpType n = 0.0;
    CUBLAS_CALL(cublas_nrm2(size, n, v.dev_ptr));
    return n;
}

using DefImpl = GPUImpl;
