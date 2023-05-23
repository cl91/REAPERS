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

#define CUSOLVER_CALL(x)					\
    do {							\
	auto __REAPERS__res = (x);				\
	if (__REAPERS__res != CUSOLVER_STATUS_SUCCESS) {	\
	    throw RuntimeError(__FILE__, __LINE__, __func__,	\
			       "CUSOLVER", __REAPERS__res);	\
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

#define cuda_alloc(p, sz)			\
    CUDA_CALL(cudaMalloc((void **)&(p), (sz)));	\
    if ((p) == nullptr) {			\
	ThrowException(DevOutOfMem, (sz));	\
    }

// CUDA defines its own complex number type which is different from
// cuda::std::complex so we need to convert between them. Fortunately
// they both have the same in-memory layout so we can trivially
// convert their pointers.
template<RealScalar FpType>
struct CudaComplex;

template<>
struct CudaComplex<float> {
    using Type = cuComplex;
    static constexpr auto cuda_realtype = CUDA_R_32F;
    static constexpr auto cuda_datatype = CUDA_C_32F;
    static constexpr auto mkcomplex = make_cuComplex;
    static constexpr auto cuCmul = cuCmulf;
    static constexpr auto gen_norm = curandGenerateNormal;
    static constexpr auto cublas_axpy = cublasCaxpy_64;
    static constexpr auto cublas_scal = cublasSscal_64;
    static constexpr auto cublas_cscal = cublasCscal_64;
    static constexpr auto cublas_dotc = cublasCdotc_64;
    static constexpr auto cublas_nrm2 = cublasScnrm2_64;
    static constexpr auto cublas_gemv = cublasCgemv_64;
    static constexpr auto cublas_gemm = cublasCgemm_64;
};

template<>
struct CudaComplex<double> {
    using Type = cuDoubleComplex;
    static constexpr auto cuda_realtype = CUDA_R_64F;
    static constexpr auto cuda_datatype = CUDA_C_64F;
    static constexpr auto mkcomplex = make_cuDoubleComplex;
    static constexpr auto cuCmul = ::cuCmul;
    static constexpr auto gen_norm = curandGenerateNormalDouble;
    static constexpr auto cublas_axpy = cublasZaxpy_64;
    static constexpr auto cublas_scal = cublasDscal_64;
    static constexpr auto cublas_cscal = cublasZscal_64;
    static constexpr auto cublas_dotc = cublasZdotc_64;
    static constexpr auto cublas_nrm2 = cublasDznrm2_64;
    static constexpr auto cublas_gemv = cublasZgemv_64;
    static constexpr auto cublas_gemm = cublasZgemm_64;
};

template<RealScalar FpType>
using CudaComplexType = typename CudaComplex<FpType>::Type;
template<RealScalar FpType>
using CudaComplexPtr = CudaComplexType<FpType> *;
template<RealScalar FpType>
using CudaComplexConstPtr = const CudaComplexType<FpType> *;

// This class represents a sum of spin operators in device memory
template<RealScalar FpType>
class DevSumOps : public HostSumOps<FpType> {
    friend class GPUImpl;

public:
    // This class is exposed to the end user so we need to be especially careful.
    class MatrixType {
	friend class GPUImpl;
	friend class DevSumOps;
	// Host matrix is assumed to be in column major order
	using HostMatrix = HostSumOps<FpType>::MatrixType;
	// Matrix is of dimension rowdim by coldim
	size_t rowdim;
	size_t coldim;
	// Matrix is stored in column major order because this is what cuBLAS expects
	complex<FpType> *dev_ptr;

	void release() {
	    if (dev_ptr) {
		cudaFree(dev_ptr);
	    }
	    dev_ptr = nullptr;
	    rowdim = coldim = 0;
	}

	size_t datasize() const {
	    return rowdim * coldim * sizeof(complex<FpType>);
	}

	void resize(size_t newrowdim, size_t newcoldim) {
	    if ((rowdim != newrowdim) || (coldim != newcoldim)) {
		release();
		allocate(newrowdim, newcoldim);
	    }
	}

	// You must either start with an uninitialized object or have
	// called release() before calling this function.
	void allocate(size_t newrowdim, size_t newcoldim = 1) {
	    rowdim = newrowdim;
	    coldim = newcoldim;
	    if (datasize()) {
		cuda_alloc(dev_ptr, datasize());
	    } else {
		dev_ptr = nullptr;
	    }
	}

	// All public constructors and member functions take signed integers
	// for indices. This is to match Eigen3 behavior.
	MatrixType(size_t rowdim, size_t coldim = 1) {
	    allocate(rowdim, coldim);
	}

	FpType mat_norm() const;

    public:
	// Eigen3 defines rows() and cols() as returning ssize_t so we follow them.
	ssize_t rows() const { return (ssize_t)rowdim; }
	ssize_t cols() const { return (ssize_t)coldim; }
	ssize_t size() const { return rows() * cols(); }

	// Likewise for resize(). Eigen3 defines resize() to take signed integers.
	void resize(ssize_t newrows, ssize_t newcols) {
	    resize((size_t)newrows, (size_t)newcols);
	}

	MatrixType() : rowdim{}, coldim{}, dev_ptr{} {}

	MatrixType(ssize_t rowdim, ssize_t coldim = 1)
	    : MatrixType((size_t)rowdim, (size_t)coldim) {}

	MatrixType(const HostMatrix &m) : MatrixType(m.rows(), m.cols()) {
	    *this = m;
	}

	MatrixType(const MatrixType &m) : MatrixType(m.rowdim, m.coldim) {
	    *this = m;
	}

	~MatrixType() {
	    release();
	}

	static MatrixType Identity(ssize_t rowdim, ssize_t coldim);

	MatrixType &operator=(const HostMatrix &m) {
	    resize(m.rows(), m.cols());
	    if (datasize()) {
		CUDA_CALL(cudaMemcpy(dev_ptr, m.data(),
				     datasize(), cudaMemcpyHostToDevice));
	    }
	    return *this;
	}

	MatrixType &operator=(const MatrixType &rhs) {
	    resize(rhs.rowdim, rhs.coldim);
	    if (datasize()) {
		CUDA_CALL(cudaMemcpy(dev_ptr, rhs.dev_ptr,
				     datasize(), cudaMemcpyDeviceToDevice));
	    }
	    return *this;
	}

	complex<FpType> operator()(ssize_t rowidx, ssize_t colidx) const {
	    complex<FpType> res;
	    auto r = (size_t)rowidx;
	    auto c = (size_t)colidx;
	    if (!(r < rowdim && c < coldim)) {
		ThrowException(InvalidArgument, "matrix indices",
			       "must be smaller than matrix dimension");
	    }
	    CUDA_CALL(cudaMemcpy(&res, dev_ptr + c * rowdim + r,
				 sizeof(complex<FpType>), cudaMemcpyDeviceToHost));
	    return res;
	}

	bool operator==(const MatrixType &other) const {
	    // Treat the special case of a zero-dimensional matrix
	    if (rowdim == 0 || coldim == 0) {
		return other.rowdim == 0 || other.coldim == 0;
	    }
	    // If the dimensions don't match, the matrices aren't equal
	    if (rowdim != other.rowdim || coldim != other.coldim) {
		return false;
	    }
	    // Otherwise, compute the norm of the difference
	    auto diff = *this - other;
	    return diff.mat_norm() == 0;
	}

	MatrixType &operator+=(const MatrixType &other);
	friend MatrixType operator+(const MatrixType &m0, const MatrixType &m1) {
	    MatrixType res(m0);
	    res += m1;
	    return res;
	}
	MatrixType &operator-=(const MatrixType &other);
	friend MatrixType operator-(const MatrixType &m0, const MatrixType &m1) {
	    MatrixType res(m0);
	    res -= m1;
	    return res;
	}
	MatrixType operator*(const MatrixType &other) const;
	MatrixType &operator*=(complex<FpType> c);
	friend MatrixType operator*(complex<FpType> c, const MatrixType &m) {
	    MatrixType res(m);
	    res *= c;
	    return res;
	}
	friend MatrixType operator*(const MatrixType &m, complex<FpType> c) {
	    return c * m;
	}
	complex<FpType> trace() const;
    };

    // For the eigenvalues we expose the host vector to the user.
    using EigenVals = HostSumOps<FpType>::EigenVals;
    using EigenVecs = MatrixType;

private:
    mutable SpinOp<FpType> *dev_ops;
    mutable std::unique_ptr<MatrixType> dev_mat;
    mutable EigenVals host_eigenvals;
    mutable FpType *dev_eigenvals;
    mutable std::unique_ptr<EigenVecs> eigenvecs;
    mutable void *sparse_mat;

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
	    cuda_alloc(dev_ops, size);
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
	    cudaFree(dev_ops);
	    dev_ops = nullptr;
	}
	dev_mat.reset(nullptr);
	host_eigenvals.resize(0);
	if (dev_eigenvals) {
	    cudaFree(dev_eigenvals);
	    dev_eigenvals = nullptr;
	}
	if (sparse_mat) {
	    /* FREE sparse_mat */
	    assert(false);
	    sparse_mat = nullptr;
	}
    }

public:
    DevSumOps(const SpinOp<FpType> &op)
	: HostSumOps<FpType>(op), dev_ops{}, dev_eigenvals{}, sparse_mat{} {}
    DevSumOps(const DevSumOps &ops)
	: HostSumOps<FpType>(ops), dev_ops{}, dev_eigenvals{}, sparse_mat{} {}
    template<RealScalar FpType1>
    explicit DevSumOps(const DevSumOps<FpType1> &ops)
	: HostSumOps<FpType>(ops), dev_ops{}, dev_eigenvals{}, sparse_mat{} {}
    explicit DevSumOps(const HostSumOps<FpType> &ops)
	: HostSumOps<FpType>(ops), dev_ops{}, dev_eigenvals{}, sparse_mat{} {}
    DevSumOps(const std::initializer_list<SpinOp<FpType>> &l = {})
	: HostSumOps<FpType>(l), dev_ops{}, dev_eigenvals{}, sparse_mat{} {}
    using HostSumOps<FpType>::operator=;
    DevSumOps &operator=(const DevSumOps &rhs) {
	HostSumOps<FpType>::operator=(rhs);
	return *this;
    }
    ~DevSumOps() { release(); }

    // Returns the GPU (dense) matrix corresponding to the sum of operators.
    const MatrixType &get_matrix(int spin_chain_len) const {
	if (!dev_mat || dev_mat->rows() != (1LL << spin_chain_len)) {
	    auto &&hostmat = HostSumOps<FpType>::get_matrix(spin_chain_len);
	    dev_mat = std::make_unique<MatrixType>(hostmat);
	}
	return *dev_mat;
    }

    // Same as HostSumOps::get_eigensystem(), except we compute the eigensystem
    // using cuSOLVER.
    using EigenSystem = std::tuple<const EigenVals &, const EigenVecs &>;
    EigenSystem get_eigensystem(int len) const;

    // Same as HostSumOps::matexp(), except computations are done on the GPU.
    MatrixType matexp(complex<FpType> c, int len) const;

    friend std::ostream &operator<<(std::ostream &os, const DevSumOps &s) {
	os << static_cast<const HostSumOps<FpType> &>(s) << " devptr " << s.dev_ops;
	return os;
    }
};

// Class that implements basic vector operations on the GPU using CUDA.
class GPUImpl {
    template<RealScalar FpType>
    friend class DevSumOps;
    template<RealScalar FpType>
    friend class DevSumOps<FpType>::MatrixType;

    // This class represents a vector in device memory. Note the end user
    // is not supposed to access this object directly (they should access
    // it via the State object) so we explicitly disable the copy ctor and
    // (both copy and move) assignment operators.
    template<RealScalar FpType>
    class DeviceVec {
	complex<FpType> *dev_ptr;
	size_t dim;
	friend class GPUImpl;
	DeviceVec(const DeviceVec &) = delete;
	DeviceVec &operator=(const DeviceVec &) = delete;
	DeviceVec &operator=(DeviceVec &&rhs) = delete;
	class ElemRefType {
	    DeviceVec &vec;
	    size_t idx;
	public:
	    ElemRefType(DeviceVec &vec, size_t idx) : vec(vec), idx(idx) {}
	    ElemRefType &operator=(complex<FpType> s) {
		CUDA_CALL(cudaMemcpy(vec.dev_ptr+idx, &s, sizeof(complex<FpType>),
				     cudaMemcpyHostToDevice));
		return *this;
	    }
	};
    public:
	DeviceVec(size_t dim) : dim(dim) {
	    cuda_alloc(dev_ptr, dim * sizeof(complex<FpType>));
	}

	DeviceVec(DeviceVec &&v) : dim(v.dim) {
	    dev_ptr = v.dev_ptr;
	    v.dev_ptr = nullptr;
	}

	explicit DeviceVec(const CPUImpl::VecType<FpType> &v) : dim(v.dimension()) {
	    cuda_alloc(dev_ptr, dim * sizeof(complex<FpType>));
	    CUDA_CALL(cudaMemcpy(dev_ptr, v.get(), dim * sizeof(complex<FpType>),
				 cudaMemcpyHostToDevice));
	}

	explicit operator CPUImpl::VecType<FpType>() const {
	    CPUImpl::VecType<FpType> v(dim);
	    CUDA_CALL(cudaMemcpy(v.get(), dev_ptr, dim * sizeof(complex<FpType>),
				 cudaMemcpyDeviceToHost));
	    return v;
	}

	~DeviceVec() {
	    if (dev_ptr != nullptr) {
		cudaFree(dev_ptr);
	    }
	}

	DeviceVec &get() { return *this; }
	const DeviceVec &get() const { return *this; }

	friend void swap(DeviceVec &lhs, DeviceVec &rhs) {
	    std::swap(lhs.dev_ptr, rhs.dev_ptr);
	}

	complex<FpType> operator[](size_t idx) const {
	    complex<FpType> c;
	    CUDA_CALL(cudaMemcpy(&c, dev_ptr+idx, sizeof(complex<FpType>),
				 cudaMemcpyDeviceToHost));
	    return c;
	}

	ElemRefType operator[](size_t idx) {
	    return ElemRefType(*this, idx);
	}
    };

    class GPUContext {
	friend class GPUImpl;
	cublasHandle_t hcublas;
	cusolverDnHandle_t hcusolver;
	curandGenerator_t randgen;
	GPUContext() {
	    CUBLAS_CALL(cublasCreate(&hcublas));
	    CUSOLVER_CALL(cusolverDnCreate(&hcusolver));
	    CURAND_CALL(curandCreateGenerator(&randgen, CURAND_RNG_PSEUDO_DEFAULT));
	    std::random_device rd;
	    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randgen, rd()));
	}
    };

    inline static GPUContext ctx;

public:
    template<RealScalar FpType>
    using VecType = DeviceVec<FpType>;

    template<RealScalar FpType>
    using VecSizeType = typename SpinOp<FpType>::StateType;

    template<RealScalar FpType>
    using BufType = DeviceVec<FpType> &;

    template<RealScalar FpType>
    using ConstBufType = const DeviceVec<FpType> &;

    template<RealScalar FpType>
    using ElemRefType = typename DeviceVec<FpType>::ElemRefType;

    template<RealScalar FpType>
    using SumOps = DevSumOps<FpType>;

private:
    template<RealScalar FpType>
    static curandStatus_t gen_norm(FpType *vec, size_t n) {
	return CudaComplex<FpType>::gen_norm(ctx.randgen, vec, n, 0.0, 1.0);
    }

    template<RealScalar FpType>
    static cublasStatus_t cublas_axpy(VecSizeType<FpType> n, complex<FpType> s,
				      const complex<FpType> *x, complex<FpType> *y) {
	return CudaComplex<FpType>::cublas_axpy(ctx.hcublas, n,
						(CudaComplexConstPtr<FpType>)&s,
						(CudaComplexConstPtr<FpType>)x, 1,
						(CudaComplexPtr<FpType>)y, 1);
    }

    template<RealScalar FpType>
    static cublasStatus_t cublas_scal(VecSizeType<FpType> n, FpType s,
				      complex<FpType> *x) {
	// n*2 might exceed 2^32 so we need to be careful with integer casting.
	int64_t n2 = static_cast<int64_t>(n) * 2;
	return CudaComplex<FpType>::cublas_scal(ctx.hcublas, n2, &s, (FpType *)x, 1);
    }

    template<RealScalar FpType>
    static cublasStatus_t cublas_scal(VecSizeType<FpType> n, complex<FpType> s,
				      complex<FpType> *x) {
	return CudaComplex<FpType>::cublas_cscal(ctx.hcublas, n,
						 (CudaComplexConstPtr<FpType>)&s,
						 (CudaComplexPtr<FpType>)x, 1);
    }

    template<RealScalar FpType>
    static cublasStatus_t cublas_dotc(VecSizeType<FpType> n,
				      complex<FpType> &res,
				      const complex<FpType> *x,
				      const complex<FpType> *y) {
	return CudaComplex<FpType>::cublas_dotc(ctx.hcublas, n,
						(CudaComplexConstPtr<FpType>)x, 1,
						(CudaComplexConstPtr<FpType>)y, 1,
						(CudaComplexPtr<FpType>)&res);
    }

    template<RealScalar FpType>
    static cublasStatus_t cublas_nrm2(VecSizeType<FpType> n, FpType &res,
				      const complex<FpType> *x) {
	return CudaComplex<FpType>::cublas_nrm2(ctx.hcublas, n,
						(CudaComplexConstPtr<FpType>)x, 1, &res);
    }

    template<RealScalar FpType>
    static cublasStatus_t cublas_gemv(size_t rowdim, size_t coldim,
				      complex<FpType> *res,
				      const complex<FpType> alpha,
				      const complex<FpType> *A,
				      const complex<FpType> *x,
				      const complex<FpType> beta,
				      bool adjoint = false) {
	return CudaComplex<FpType>::cublas_gemv(ctx.hcublas,
						adjoint ? CUBLAS_OP_C : CUBLAS_OP_N,
						(int64_t)rowdim, (int64_t)coldim,
						(CudaComplexConstPtr<FpType>)&alpha,
						(CudaComplexConstPtr<FpType>)A,
						(int64_t)rowdim,
						(CudaComplexConstPtr<FpType>)x, 1,
						(CudaComplexConstPtr<FpType>)&beta,
						(CudaComplexPtr<FpType>)res, 1);
    }

    template<RealScalar FpType>
    static cublasStatus_t cublas_gemm(size_t rowdim, size_t coldimA, size_t coldimB,
				      complex<FpType> *res,
				      const complex<FpType> alpha,
				      const complex<FpType> *A,
				      const complex<FpType> *B,
				      const complex<FpType> beta,
				      bool adjointA = false, bool adjointB = false) {
	return CudaComplex<FpType>::cublas_gemm(ctx.hcublas,
						adjointA ? CUBLAS_OP_C : CUBLAS_OP_N,
						adjointB ? CUBLAS_OP_C : CUBLAS_OP_N,
						(int64_t)rowdim, (int64_t)coldimB,
						(int64_t)coldimA,
						(CudaComplexConstPtr<FpType>)&alpha,
						(CudaComplexConstPtr<FpType>)A,
						(int64_t)rowdim,
						(CudaComplexConstPtr<FpType>)B,
						(int64_t)coldimA,
						(CudaComplexConstPtr<FpType>)&beta,
						(CudaComplexPtr<FpType>)res,
						(int64_t)rowdim);
    }

    template<RealScalar FpType>
    static cusolverStatus_t cusolver_syevd_get_bufsize(cusolverDnParams_t params,
						       cusolverEigMode_t jobz,
						       cublasFillMode_t uplo,
						       size_t dim,
						       complex<FpType> *mat,
						       FpType *vals,
						       size_t &dworksz,
						       size_t &lworksz) {
	auto ty = CudaComplex<FpType>::cuda_datatype;
	auto rty = CudaComplex<FpType>::cuda_realtype;
	return cusolverDnXsyevd_bufferSize(ctx.hcusolver, params, jobz, uplo,
					   dim, ty, mat, dim, rty, vals, ty,
					   &dworksz, &lworksz);
    }

    template<RealScalar FpType>
    static cusolverStatus_t cusolver_syevd(cusolverDnParams_t params,
					   cusolverEigMode_t jobz,
					   cublasFillMode_t uplo,
					   size_t dim,
					   complex<FpType> *mat,
					   FpType *vals,
					   void *d_work,
					   int dworksz,
					   void *l_work,
					   int lworksz,
					   int *d_info) {
	auto ty = CudaComplex<FpType>::cuda_datatype;
	auto rty = CudaComplex<FpType>::cuda_realtype;
	return cusolverDnXsyevd(ctx.hcusolver, params, jobz, uplo,
				dim, ty, mat, dim, rty, vals, ty,
				d_work, dworksz, l_work, lworksz, d_info);
    }

    // Compute the block size and grid size for CUDA kernels.
    static void get_block_grid_size(size_t size, int &blocksize, int &gridsize) {
	blocksize = (size > 1024ULL) ? 1024 : size;
	gridsize = size / blocksize;
    }

    // Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(size_t size, complex<FpType> *v0,
			const complex<FpType> *v1);

    // Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
    template<RealScalar FpType>
    static void add_vec(size_t size, complex<FpType> *res,
			const complex<FpType> *v0, const complex<FpType> *v1);

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(size_t size, complex<FpType> *v0,
			complex<FpType> s, const complex<FpType> *v1) {
	CUBLAS_CALL(cublas_axpy(size, s, v1, v0));
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(size_t size, complex<FpType> *v0,
			FpType s, const complex<FpType> *v1) {
	CUBLAS_CALL(cublas_axpy(size, {s,0}, v1, v0));
    }

    // Compute v *= s where s is a real scalar.
    template<RealScalar FpType>
    static void scale_vec(size_t size, complex<FpType> *v, FpType s) {
	CUBLAS_CALL(cublas_scal(size, s, v));
    }

    // Compute v *= s where s is a complex scalar.
    template<RealScalar FpType>
    static void scale_vec(size_t size, complex<FpType> *v, complex<FpType> s) {
	CUBLAS_CALL(cublas_scal(size, s, v));
    }

    // Compute the vector norm of the complex vector v. In other words, compute
    // n = sqrt(<v|v>) = sqrt(v^\dagger \cdot v)
    template<RealScalar FpType>
    static FpType vec_norm(size_t size, complex<FpType> *v) {
	FpType n = 0.0;
	CUBLAS_CALL(cublas_nrm2(size, n, v));
	return n;
    }

    // Compute res[i] = exp(c*v[i]) where res is a complex dim-by-1 matrix
    // and v is a real dim dimensional vector.
    template<RealScalar FpType>
    static void exp_vec(typename DevSumOps<FpType>::MatrixType &res,
			const FpType *v, complex<FpType> c);

    // Compute res[i] = lambda[i] * v[i] where v is a list of vectors stored
    // sequentially (and likewise for res). Since MatrixType stores a matrix
    // using column-major order, this function effectively computes
    // M_res = M_v * diag(lambda). Here M_v is the matrix whose in-memory
    // representation matches that of v (ie. v[i] is the i-th column vector
    // of M_v), and likewise for M_res.
    template<RealScalar FpType>
    static void scale_vecs(typename DevSumOps<FpType>::MatrixType &res,
			   const typename DevSumOps<FpType>::MatrixType &v,
			   const typename DevSumOps<FpType>::MatrixType &lambda,
			   unsigned int len);

    // The functions below are exposed to the State object. The private member
    // functions above can only be accessed by DevSumOps and DevSumOps::MatrixType.
public:
    // Initialize the vector with zero, ie. set v0[i] to 0.0 for all i.
    template<RealScalar FpType>
    static void zero_vec(VecSizeType<FpType> size, BufType<FpType> v) {
	CUDA_CALL(cudaMemset(v.dev_ptr, 0, size * sizeof(complex<FpType>)));
    }

    // Initialize the vector to a Haar random state.
    template<RealScalar FpType>
    static void init_random(VecSizeType<FpType> size, BufType<FpType> v);

    // Copy v1 into v0.
    template<RealScalar FpType>
    static void copy_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			 ConstBufType<FpType> v1) {
	CUDA_CALL(cudaMemcpy(v0.dev_ptr, v1.dev_ptr,
			     size * sizeof(complex<FpType>),
			     cudaMemcpyDeviceToDevice));
    }

    // Copy v1 into v0, converting double to float by rounding toward zero.
    static void copy_vec(VecSizeType<float> size, BufType<float> v0,
			 ConstBufType<double> v1);

    // Copy v1 into v0, converting float to double.
    static void copy_vec(VecSizeType<double> size, BufType<double> v0,
			 ConstBufType<float> v1);

    // Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			ConstBufType<FpType> v1) {
	add_vec(size, v0.dev_ptr, v1.dev_ptr);
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			complex<FpType> s, ConstBufType<FpType> v1) {
	add_vec(size, v0.dev_ptr, s, v1.dev_ptr);
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0, FpType s,
			ConstBufType<FpType> v1) {
	add_vec(size, v0.dev_ptr, s, v1.dev_ptr);
    }

    // Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> res,
			ConstBufType<FpType> v0, ConstBufType<FpType> v1) {
	add_vec(size, res.dev_ptr, v0.dev_ptr, v1.dev_ptr);
    }

    // Compute v *= s where s is a real scalar.
    template<RealScalar FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v, FpType s) {
	scale_vec(size, v.dev_ptr, s);
    }

    // Compute v *= s where s is a complex scalar.
    template<RealScalar FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v,
			  complex<FpType> s) {
	scale_vec(size, v.dev_ptr, s);
    }

    // Compute the vector inner product of the complex vector v0 and v1.
    // In other words, compute v = <v0 | v1> = v0^\dagger \cdot v1
    template<RealScalar FpType>
    static complex<FpType> vec_prod(size_t size, ConstBufType<FpType> v0,
				    ConstBufType<FpType> v1) {
	complex<FpType> res = 0.0;
	CUBLAS_CALL(cublas_dotc(size, res, v0.dev_ptr, v1.dev_ptr));
	return res;
    }

    // Compute the vector norm of the complex vector v. In other words, compute
    // n = sqrt(<v|v>) = sqrt(v^\dagger \cdot v)
    template<RealScalar FpType>
    static FpType vec_norm(size_t size, ConstBufType<FpType> v) {
	return vec_norm(size, v.dev_ptr);
    }

    // Set the given matrix to the identity matrix.
    template<RealScalar FpType>
    static void eye(typename DevSumOps<FpType>::MatrixType &res);

    // Compute the matrix multiplication res = mat * vec, where mat is of
    // type HostSumOps::MatrixType. Here res and vec are assumed to be
    // different buffers.
    template<RealScalar FpType>
    static void mat_mul(VecSizeType<FpType> dim, BufType<FpType> res,
			const typename DevSumOps<FpType>::MatrixType &mat,
			ConstBufType<FpType> vec, bool adjoint = false) {
	assert(dim == mat.rowdim);
	assert(dim == mat.coldim);
	CUBLAS_CALL(cublas_gemv(dim, dim, res.dev_ptr, {1,0}, mat.dev_ptr,
				vec.dev_ptr, {}, adjoint));
    }

    // Compute the matrix multiplication res = mat0 * mat1, where mat0 and
    // mat1 are of type HostSumOps::MatrixType. Here the res pointer is
    // assumed to be different from mat0 and mat1.
    template<RealScalar FpType>
    static void mat_mul(typename DevSumOps<FpType>::MatrixType &res,
			const typename DevSumOps<FpType>::MatrixType &mat0,
			const typename DevSumOps<FpType>::MatrixType &mat1,
			bool adjoint0 = false, bool adjoint1 = false) {
	assert(res.rowdim == mat0.rowdim);
	assert(mat0.coldim == mat1.rowdim);
	assert(mat1.coldim == res.coldim);
	CUBLAS_CALL(cublas_gemm(res.rowdim, mat0.coldim, mat1.coldim,
				res.dev_ptr, {1,0}, mat0.dev_ptr, mat1.dev_ptr,
				{}, adjoint0, adjoint1));
    }

    // Compute the trace of the given matrix.
    template<RealScalar FpType>
    static complex<FpType> mat_tr(const typename DevSumOps<FpType>::MatrixType &mat);

    // Compute res = ops * vec. res and vec are assumed to be different.
    template<RealScalar FpType>
    static void apply_ops(typename SpinOp<FpType>::IndexType len, BufType<FpType> res,
			  const DevSumOps<FpType> &ops, ConstBufType<FpType> vec);
};

// Initialize the vector to a Haar random state.
template<RealScalar FpType>
inline void GPUImpl::init_random(VecSizeType<FpType> size, BufType<FpType> v) {
    // Generate 2*size random numbers
    CURAND_CALL(gen_norm((FpType *)v.dev_ptr, 2*size));
    // Normalize the resulting vector
    FpType c = 1/vec_norm(size, v);
    scale_vec(size, v, c);
}

template<RealScalar FpType>
inline DevSumOps<FpType>::MatrixType DevSumOps<FpType>::MatrixType::Identity(
    ssize_t rowdim, ssize_t coldim) {
    MatrixType id(rowdim, coldim);
    GPUImpl::eye<FpType>(id);
    return id;
}

template<RealScalar FpType>
inline DevSumOps<FpType>::MatrixType &DevSumOps<FpType>::MatrixType::operator+=(
    const DevSumOps<FpType>::MatrixType &other) {
    if (rowdim != other.rowdim || coldim != other.coldim) {
	ThrowException(InvalidArgument, "matrix dimensions", "must match");
    }
    GPUImpl::add_vec<FpType>(rowdim * coldim, dev_ptr, other.dev_ptr);
    return *this;
}

template<RealScalar FpType>
inline DevSumOps<FpType>::MatrixType &DevSumOps<FpType>::MatrixType::operator-=(
    const DevSumOps<FpType>::MatrixType &other) {
    if (rowdim != other.rowdim || coldim != other.coldim) {
	ThrowException(InvalidArgument, "matrix dimensions", "must match");
    }
    GPUImpl::add_vec<FpType>(rowdim * coldim, dev_ptr, {-1,0}, other.dev_ptr);
    return *this;
}

template<RealScalar FpType>
inline DevSumOps<FpType>::MatrixType DevSumOps<FpType>::MatrixType::operator*(
    const DevSumOps<FpType>::MatrixType &other) const {
    if (coldim != other.rowdim) {
	std::stringstream ss;
	ss << "must match the row dimension of the second matrix ("
	   << coldim << " != " << other.rowdim << ")";
	ThrowException(InvalidArgument, "column dimension of first matrix",
		       ss.str().c_str());
    }
    MatrixType res(rowdim, other.coldim);
    GPUImpl::mat_mul<FpType>(res, *this, other);
    return res;
}

template<RealScalar FpType>
inline DevSumOps<FpType>::MatrixType &DevSumOps<FpType>::MatrixType::operator*=(
    complex<FpType> c) {
    GPUImpl::scale_vec<FpType>(rowdim * coldim, dev_ptr, c);
    return *this;
}

template<RealScalar FpType>
inline complex<FpType> DevSumOps<FpType>::MatrixType::trace() const {
    if (coldim != rowdim) {
	ThrowException(InvalidArgument, "column dimension of matrix",
		       "must equal the row dimension");
    }
    return GPUImpl::mat_tr<FpType>(*this);
}

template<RealScalar FpType>
inline FpType DevSumOps<FpType>::MatrixType::mat_norm() const {
    return GPUImpl::vec_norm(rowdim * coldim, dev_ptr);
}

template<RealScalar FpType>
inline DevSumOps<FpType>::EigenSystem DevSumOps<FpType>::get_eigensystem(int len) const {
    size_t dim = 1ULL << len;
    if (!dev_eigenvals || eigenvecs->rowdim != dim) {
	if (dev_eigenvals) {
	    cudaFree(dev_eigenvals);
	}
	cuda_alloc(dev_eigenvals, dim * sizeof(FpType));
	assert(dev_eigenvals);
	eigenvecs = std::make_unique<MatrixType>(get_matrix(len));

	// Set up cuSOLVER parameters. Note that params is a pointer type.
	cusolverDnParams_t params;
	CUSOLVER_CALL(cusolverDnCreateParams(&params));
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	// Compute buffer sizes and prepare workspaces.
	size_t dworksz{}, lworksz{};	  // Workspace sizes
	CUSOLVER_CALL(GPUImpl::cusolver_syevd_get_bufsize(params, jobz, uplo,
							  dim, eigenvecs->dev_ptr,
							  dev_eigenvals,
							  dworksz, lworksz));
	void *d_work{};   // Workspace on the device
	if (dworksz) {
	    cuda_alloc(d_work, dworksz);
	}
	std::unique_ptr<char[]> l_work; // Workspace on the host
	if (lworksz) {
	    l_work = std::make_unique<char[]>(lworksz);
	}
	int *d_info;	    // Result information status on the device
	cuda_alloc(d_info, sizeof(int));

	// Compute the eigenvalues and eigenvectors for a Hermitian matrix.
	// At the end of the call the eigenvecs matrix is overwritten with
	// the actual column eigenvectors.
	CUSOLVER_CALL(GPUImpl::cusolver_syevd(params, jobz, uplo, dim,
					      eigenvecs->dev_ptr, dev_eigenvals,
					      d_work, dworksz, l_work.get(),
					      lworksz, d_info));
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy the status information from device to host and check if is an error
	int info{};
	CUDA_CALL(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
	if (info != 0) {
	    ThrowException(RuntimeError, "cusolverDnXsyevd", info);
	}

	// Copy the eigenvalues from device to host, and free temporary spaces.
	host_eigenvals.resize(dim);
	CUDA_CALL(cudaMemcpy(host_eigenvals.data(), dev_eigenvals,
			     sizeof(FpType) * dim, cudaMemcpyDeviceToHost));
	cudaFree(d_info);
	cudaFree(d_work);
    }
    assert(dev_eigenvals);
    assert(host_eigenvals.rows() == (ssize_t)dim);
    assert(host_eigenvals.cols() == 1);
    assert(eigenvecs->rowdim == dim);
    assert(eigenvecs->coldim == dim);
    return {host_eigenvals, *eigenvecs};
}

template<RealScalar FpType>
inline DevSumOps<FpType>::MatrixType DevSumOps<FpType>::matexp(complex<FpType> c,
							       int len) const {
    get_eigensystem(len);
    assert(dev_eigenvals);
    assert(host_eigenvals.rows() == (1LL << len));
    assert(host_eigenvals.cols() == 1);
    assert(eigenvecs->rows() == (1LL << len));
    assert(eigenvecs->cols() == (1LL << len));
    auto dim = eigenvecs->rowdim;
    MatrixType vexp(dim, 1);
    GPUImpl::exp_vec(vexp, dev_eigenvals, c);
    MatrixType tmp(dim, dim);
    GPUImpl::scale_vecs<FpType>(tmp, *eigenvecs, vexp, len);
    MatrixType res(dim, dim);
    GPUImpl::mat_mul<FpType>(res, tmp, *eigenvecs, false, true);
    return res;
}

using DefImpl = GPUImpl;
