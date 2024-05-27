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
	    throw CudaError(__FILE__, __LINE__, __func__,	\
			    __REAPERS__res);			\
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

#ifdef REAPERS_NO_MULTIGPU
#define REAPERS_MAX_GPU_COUNT (1)
#elif !defined(REAPERS_MAX_GPU_COUNT)
#define REAPERS_MAX_GPU_COUNT (8)
#endif

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

class GPUImpl;

// This class represents a sum of spin operators in device memory
template<RealScalar FpType>
class DevSumOps : public HostSumOps<FpType> {
    friend class GPUImpl;
    // This is needed so that the _matexp_helper static member function from
    // the parent class can access the derived class matexp_cache member.
    friend class HostSumOps<FpType>;

    using Impl = GPUImpl;

public:
    using RealScalarType = FpType;
    using ComplexScalarType = complex<FpType>;

    // This class is exposed to the end user so we need to be especially careful.
    class MatrixType {
	friend class GPUImpl;
	friend class DevSumOps;
	// Host matrix is assumed to be in column major order
	using HostMatrix = typename HostSumOps<FpType>::MatrixType;
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

	MatrixType(MatrixType &&m) : rowdim{m.rowdim}, coldim{m.coldim},
				     dev_ptr{m.dev_ptr} {
	    m.rowdim = m.coldim = 0;
	    m.dev_ptr = nullptr;
	}

	~MatrixType() { release(); }

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
	    if (this == &rhs) { return *this; }
	    resize(rhs.rowdim, rhs.coldim);
	    if (datasize()) {
		CUDA_CALL(cudaMemcpy(dev_ptr, rhs.dev_ptr,
				     datasize(), cudaMemcpyDeviceToDevice));
	    }
	    return *this;
	}

	MatrixType &operator=(MatrixType &&rhs) {
	    if (this == &rhs) { return *this; }
	    std::swap(rowdim, rhs.rowdim);
	    std::swap(coldim, rhs.coldim);
	    std::swap(dev_ptr, rhs.dev_ptr);
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
    using EigenVals = typename HostSumOps<FpType>::EigenVals;
    using EigenVecs = MatrixType;

private:
    mutable std::vector<SpinOp<FpType> *> dev_ops;
    mutable std::unique_ptr<MatrixType> dev_mat;
    mutable EigenVals host_eigenvals;
    mutable FpType *dev_eigenvals;
    mutable std::unique_ptr<EigenVecs> eigenvecs;
    // Note this hides the matexp_cache member of the parent class.
    mutable MatexpCache<FpType,MatrixType,GPUImpl> matexp_cache;

    // Upload the host summed operators to the device. If the number of operators
    // is less than 2^len, then simply copy the operators to the device side.
    // Otherwise, generate the (dense) matrix representing the sum of operators.
    void upload(typename SpinOp<FpType>::IndexType len) const;

    // Free the device copy of the host summed operators. This should
    // be called after every update to the operator list, so next time
    // GPUImpl::apply_ops can upload a fresh copy of the latest SumOps.
    void release() const;

    void mini_release() const {
	HostSumOps<FpType>::mini_release();
	matexp_cache.clear();
    }

    void swap_internals(DevSumOps &&rhs) {
	using std::swap;
	swap(dev_ops, rhs.dev_ops);
	swap(dev_mat, rhs.dev_mat);
	swap(host_eigenvals, rhs.host_eigenvals);
	swap(dev_eigenvals, rhs.dev_eigenvals);
	swap(eigenvecs, rhs.eigenvecs);
	matexp_cache = std::move(rhs.matexp_cache);
    }

public:
    DevSumOps(const SpinOp<FpType> &op)
	: HostSumOps<FpType>(op), dev_ops{}, dev_eigenvals{} {}
    DevSumOps(const HostSumOps<FpType> &ops)
	: HostSumOps<FpType>(ops), dev_ops{}, dev_eigenvals{} {}
    DevSumOps(const DevSumOps &ops)
	: HostSumOps<FpType>(ops), dev_ops{}, dev_eigenvals{} {}
    DevSumOps(DevSumOps &&ops) : HostSumOps<FpType>(std::move(ops)),
				 dev_ops{}, dev_eigenvals{} {
	swap_internals(std::move(ops));
    }
    template<RealScalar FpType1>
    explicit DevSumOps(const DevSumOps<FpType1> &ops)
	: HostSumOps<FpType>(ops), dev_ops{}, dev_eigenvals{} {}
    DevSumOps(const std::initializer_list<SpinOp<FpType>> &l = {})
	: HostSumOps<FpType>(l), dev_ops{}, dev_eigenvals{} {}
    using HostSumOps<FpType>::operator=;
    DevSumOps &operator=(const DevSumOps &rhs) {
	HostSumOps<FpType>::operator=(rhs);
	return *this;
    }
    DevSumOps &operator=(DevSumOps &&rhs) {
	if (this == &rhs) { return *this; }
	*this = static_cast<HostSumOps<FpType> &&>(rhs);
	swap_internals(std::move(rhs));
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
private:
    MatrixType _matexp(complex<FpType> c, int len) const;
public:
    const MatrixType &matexp(complex<FpType> c, int len) const {
	return REAPERS::HostSumOps<FpType>::_matexp_helper(*this, c, len);
    }

    template<RealScalar Fp>
    friend std::ostream &operator<<(std::ostream &os, const DevSumOps<Fp> &s);
};

// Class that implements basic vector operations on the GPU using CUDA.
class GPUImpl {
    template<RealScalar FpType>
    friend class DevSumOps;

    // This class represents a vector in device memory. Note the end user
    // is not supposed to access this object directly (they should access
    // it via the State object) so we explicitly disable the copy ctor and
    // (both copy and move) assignment operators.
    template<RealScalar FpType>
    class DeviceVec {
	std::vector<complex<FpType> *> dev_ptrs;
	size_t dev_dim;		// Dimension per device
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
		int dev = idx / vec.dev_dim;
		size_t dev_idx = idx & (vec.dev_dim - 1);
		CUDA_CALL(cudaMemcpy(vec.dev_ptrs[dev] + dev_idx, &s, sizeof(complex<FpType>),
				     cudaMemcpyHostToDevice));
		return *this;
	    }
	};
    public:
	DeviceVec(size_t dim) : dev_ptrs(GPUImpl::get_device_count()),
				dev_dim(dim / GPUImpl::get_device_count()) {
	    for (int i = 0; i < GPUImpl::get_device_count(); i++) {
		set_device(i);
		cuda_alloc(dev_ptrs[i], dev_dim * sizeof(complex<FpType>));
	    }
	}

	DeviceVec(DeviceVec &&v) : dev_dim(v.dev_dim) {
	    swap(dev_ptrs, v.dev_ptrs);
	}

	explicit DeviceVec(const CPUImpl::VecType<FpType> &v) :
	    dev_ptrs(GPUImpl::get_device_count()),
	    dev_dim(v.dimension() / GPUImpl::get_device_count()) {
	    for (int i = 0; i < GPUImpl::get_device_count(); i++) {
		set_device(i);
		cuda_alloc(dev_ptrs[i], dev_dim * sizeof(complex<FpType>));
	    }
	    for (int i = 0; i < GPUImpl::get_device_count(); i++) {
		CUDA_CALL(cudaMemcpyAsync(dev_ptrs[i], v.get() + dev_dim * i,
					  dev_dim * sizeof(complex<FpType>),
					  cudaMemcpyHostToDevice));
	    }
	    sync_devices();
	}

	explicit operator CPUImpl::VecType<FpType>() const {
	    CPUImpl::VecType<FpType> v(dev_dim * GPUImpl::get_device_count());
	    for (int i = 0; i < GPUImpl::get_device_count(); i++) {
		CUDA_CALL(cudaMemcpyAsync(v.get() + dev_dim * i,
					  dev_ptrs[i], dev_dim * sizeof(complex<FpType>),
					  cudaMemcpyDeviceToHost));
	    }
	    sync_devices();
	    return v;
	}

	~DeviceVec() {
	    if (dev_ptrs.size()) {
		assert((int)dev_ptrs.size() == GPUImpl::get_device_count());
		for (int i = 0; i < GPUImpl::get_device_count(); i++) {
		    if (dev_ptrs[i] != nullptr) {
			cudaFree(dev_ptrs[i]);
		    }
		}
	    }
	}

	DeviceVec &get() { return *this; }
	const DeviceVec &get() const { return *this; }

	friend void swap(DeviceVec &lhs, DeviceVec &rhs) {
	    std::swap(lhs.dev_ptrs, rhs.dev_ptrs);
	}

	complex<FpType> operator[](size_t idx) const {
	    complex<FpType> c;
	    int dev = idx / dev_dim;
	    size_t dev_idx = idx & (dev_dim - 1);
	    CUDA_CALL(cudaMemcpy(&c, dev_ptrs[dev] + dev_idx, sizeof(complex<FpType>),
				 cudaMemcpyDeviceToHost));
	    return c;
	}

	ElemRefType operator[](size_t idx) {
	    return ElemRefType(*this, idx);
	}
    };

    class GPUContext {
	friend class GPUImpl;
	int num_dev;
	std::vector<cublasHandle_t> vhcublas;
	std::vector<curandGenerator_t> vrandgen;
	cusolverDnHandle_t hcusolver;
	size_t vram_total_size;
	size_t vram_free_size;
	GPUContext() {
#ifdef REAPERS_NO_MULTIGPU
	    num_dev = 1;
#else
	    CUDA_CALL(cudaGetDeviceCount(&num_dev));
	    // According to nvidia docs this function never returns num_dev == 0,
	    // but to be safe let's make sure it's not zero.
	    if (!num_dev) {
		ThrowException(NoGpuDevice);
	    }
	    // If the number of device is not a power of two, reduce it to the
	    // largest power of two that does not exceed the original number.
	    if (num_dev & (num_dev - 1)) {
		num_dev |= num_dev >> 1;
		num_dev |= num_dev >> 2;
		num_dev |= num_dev >> 4;
		num_dev |= num_dev >> 8;
		num_dev |= num_dev >> 16;
		num_dev -= num_dev >> 1;
	    }
	    // Check if all devices can access each other through P2P link.
	    for (int i = 0; i < num_dev; i++) {
		CUDA_CALL(cudaSetDevice(i));
		for (int j = 0; j < num_dev; j++) {
		    if (i == j) {
			continue;
		    }
		    int ok = 0;
		    CUDA_CALL(cudaDeviceCanAccessPeer(&ok, i, j));
		    if (!ok) {
			ThrowException(NoP2PAccess, i, j);
		    }
		    CUDA_CALL(cudaDeviceEnablePeerAccess(j, 0));
		}
	    }
#endif
	    std::random_device rd;
	    vhcublas.resize(num_dev);
	    vrandgen.resize(num_dev);
	    for (int i = 0; i < num_dev; i++) {
		CUDA_CALL(cudaSetDevice(i));
		cublasHandle_t hcublas;
		CUBLAS_CALL(cublasCreate(&hcublas));
		vhcublas[i] = hcublas;
		curandGenerator_t randgen;
		CURAND_CALL(curandCreateGenerator(&randgen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randgen, rd()));
		vrandgen[i] = randgen;
	    }
	    set_device(0);
	    CUDA_CALL(cudaMemGetInfo(&vram_free_size, &vram_total_size));
	    CUSOLVER_CALL(cusolverDnCreate(&hcusolver));
	}
    };

    inline static GPUContext ctx;
    // By default we will use 50% of available VRAM for the matexp cache.
    inline static size_t max_cache_size = ctx.vram_free_size / 2;
    // Current cache size
    inline static size_t cache_size = 0;

public:
    // You must call cache.evict() if this function returns false.
    static bool cache_reserve(size_t sz) {
	if ((cache_size + sz) > max_cache_size) {
	    return false;
	} else {
	    cache_size += sz;
	    return true;
	}
    }

    static void cache_release(size_t sz) {
	assert(sz <= cache_size);
	if (sz <= cache_size) {
	    cache_size -= sz;
	}
    }

    static void set_max_cache_size(size_t sz) {
	max_cache_size = sz;
    }

    static void set_max_cache_size(float fraction) {
	max_cache_size = (size_t)(ctx.vram_free_size * fraction);
    }

    static size_t get_max_cache_size() {
	return max_cache_size;
    }

    static size_t get_current_cache_size() {
	return cache_size;
    }

    static int get_device_count() {
	return ctx.num_dev;
    }

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
    static void set_device(int i) {
#ifndef REAPERS_NO_MULTIGPU
	CUDA_CALL(cudaSetDevice(i));
#endif
    }

    static void sync_devices() {
	for (int i = 0; i < get_device_count(); i++) {
	    sync_device(i);
	}
    }

    static void sync_device(int i) {
	CUDA_CALL(cudaSetDevice(i));
	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaDeviceSynchronize());
    }

    template<RealScalar FpType>
    static curandStatus_t gen_norm(int dev, FpType *vec, size_t n) {
	set_device(dev);
	return CudaComplex<FpType>::gen_norm(ctx.vrandgen[dev], vec, n, 0.0, 1.0);
    }

    // Compute y += s * x. x and y are the dev-th tile (ie. part of the state
    // vector on device number dev) of two different state vectors. n is the dimension
    // of the Hilbert space divided by the number of available GPU devices.
    template<RealScalar FpType>
    static cublasStatus_t cublas_axpy(int dev, VecSizeType<FpType> n, complex<FpType> s,
				      const complex<FpType> *x, complex<FpType> *y) {
	set_device(dev);
	return CudaComplex<FpType>::cublas_axpy(ctx.vhcublas[dev], n,
						(CudaComplexConstPtr<FpType>)&s,
						(CudaComplexConstPtr<FpType>)x, 1,
						(CudaComplexPtr<FpType>)y, 1);
    }

    // Compute x *= s. x is the dev-th tile of a state vector. n is the dimension
    // of the Hilbert space divided by the number of available GPU devices.
    template<RealScalar FpType>
    static cublasStatus_t cublas_scal(int dev, VecSizeType<FpType> n, FpType s,
				      complex<FpType> *x) {
	// n*2 might exceed 2^32 so we need to be careful with integer casting.
	int64_t n2 = static_cast<int64_t>(n) * 2;
	set_device(dev);
	return CudaComplex<FpType>::cublas_scal(ctx.vhcublas[dev], n2, &s, (FpType *)x, 1);
    }

    // Compute x *= s. x is the dev-th tile of a state vector. n is the dimension
    // of the Hilbert space divided by the number of available GPU devices.
    template<RealScalar FpType>
    static cublasStatus_t cublas_scal(int dev, VecSizeType<FpType> n, complex<FpType> s,
				      complex<FpType> *x) {
	set_device(dev);
	return CudaComplex<FpType>::cublas_cscal(ctx.vhcublas[dev], n,
						 (CudaComplexConstPtr<FpType>)&s,
						 (CudaComplexPtr<FpType>)x, 1);
    }

    // Compute <x|y> = x^dagger \cdot y. x and y are the dev-th tile (ie. part of the state
    // vector on device number dev) of two state vectors (x and y can be the same). n is the
    // dimension of the Hilbert space divided by the number of available GPU devices.
    template<RealScalar FpType>
    static cublasStatus_t cublas_dotc(int dev, VecSizeType<FpType> n,
				      complex<FpType> &res,
				      const complex<FpType> *x,
				      const complex<FpType> *y) {
	set_device(dev);
	return CudaComplex<FpType>::cublas_dotc(ctx.vhcublas[dev], n,
						(CudaComplexConstPtr<FpType>)x, 1,
						(CudaComplexConstPtr<FpType>)y, 1,
						(CudaComplexPtr<FpType>)&res);
    }

    // Compute \sqrt(<x|y>) = \sqrt(x^dagger \cdot y). x and y must be on the 0-th device.
    template<RealScalar FpType>
    static cublasStatus_t cublas_nrm2(VecSizeType<FpType> n, FpType &res,
				      const complex<FpType> *x) {
	set_device(0);
	return CudaComplex<FpType>::cublas_nrm2(ctx.vhcublas[0], n,
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
	set_device(0);
	return CudaComplex<FpType>::cublas_gemv(ctx.vhcublas[0],
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
	set_device(0);
	return CudaComplex<FpType>::cublas_gemm(ctx.vhcublas[0],
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
	set_device(0);
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
	set_device(0);
	return cusolverDnXsyevd(ctx.hcusolver, params, jobz, uplo,
				dim, ty, mat, dim, rty, vals, ty,
				d_work, dworksz, l_work, lworksz, d_info);
    }

    // Compute the block size and grid size for CUDA kernels.
    static void get_block_grid_size(size_t size, int &blocksize, int &gridsize) {
	blocksize = (size > 1024ULL) ? 1024 : size;
	gridsize = size / blocksize;
    }

    // Compute v0 += v1. v0 and v1 are assumed to point to different buffers
    // on the same device (with device id dev).
    template<RealScalar FpType>
    static void add_vec(int dev, size_t size, complex<FpType> *v0,
			const complex<FpType> *v1);

    // Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
    // All vectors must be on the same device (with device id dev).
    template<RealScalar FpType>
    static void add_vec(int dev, size_t size, complex<FpType> *res,
			const complex<FpType> *v0, const complex<FpType> *v1);

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers
    // on the same device (with device id dev).
    template<RealScalar FpType>
    static void add_vec(int dev, size_t size, complex<FpType> *v0,
			complex<FpType> s, const complex<FpType> *v1) {
	CUBLAS_CALL(cublas_axpy(dev, size, s, v1, v0));
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers
    // on the same device (with device id dev).
    template<RealScalar FpType>
    static void add_vec(int dev, size_t size, complex<FpType> *v0,
			FpType s, const complex<FpType> *v1) {
	CUBLAS_CALL(cublas_axpy(dev, size, {s,0}, v1, v0));
    }

    // Compute v *= s where s is a real scalar. v must be on the device id dev.
    template<RealScalar FpType>
    static void scale_vec(int dev, size_t size, complex<FpType> *v, FpType s) {
	CUBLAS_CALL(cublas_scal(dev, size, s, v));
    }

    // Compute v *= s where s is a complex scalar. v must be on the device id dev.
    template<RealScalar FpType>
    static void scale_vec(int dev, size_t size, complex<FpType> *v, complex<FpType> s) {
	CUBLAS_CALL(cublas_scal(dev, size, s, v));
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
	assert(size == v.dev_dim * get_device_count());
	for (int i = 0; i < get_device_count(); i++) {
	    set_device(i);
	    CUDA_CALL(cudaMemsetAsync(v.dev_ptrs[i], 0, v.dev_dim * sizeof(complex<FpType>)));
	}
	sync_devices();
    }

    // Initialize the vector to a Haar random state.
    template<RealScalar FpType>
    static void init_random(VecSizeType<FpType> size, BufType<FpType> v);

    // Copy v1 into v0.
    template<RealScalar FpType>
    static void copy_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			 ConstBufType<FpType> v1) {
	assert(size == v0.dev_dim * get_device_count());
	assert(v0.dev_dim == v1.dev_dim);
	for (int i = 0; i < get_device_count(); i++) {
	    set_device(i);
	    CUDA_CALL(cudaMemcpyAsync(v0.dev_ptrs[i], v1.dev_ptrs[i],
				      v0.dev_dim * sizeof(complex<FpType>),
				      cudaMemcpyDeviceToDevice));
	}
	sync_devices();
    }

    // Copy v1 into v0, converting double to float by rounding toward zero.
    static void copy_vec(VecSizeType<float> size, BufType<float> v0,
			 ConstBufType<double> v1);

    // Copy v1 into v0, converting float to double.
    static void copy_vec(VecSizeType<double> size, BufType<double> v0,
			 ConstBufType<float> v1);

    // Copy the n-th column vector of the given column major matrix into
    // the buffer res.
    template<RealScalar FpType>
    static void copy_vec(VecSizeType<FpType> size, BufType<FpType> res,
			 const typename DevSumOps<FpType>::MatrixType &mat,
			 ssize_t rowidx) {
	assert(size == res.dev_dim * get_device_count());
	for (int i = 0; i < get_device_count(); i++) {
	    set_device(i);
	    CUDA_CALL(cudaMemcpyAsync(res.dev_ptrs[i],
				      mat.dev_ptr + rowidx * size + i * res.dev_dim,
				      sizeof(complex<FpType>) * res.dev_dim,
				      cudaMemcpyDeviceToDevice));
	}
	sync_devices();
    }

    // Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			ConstBufType<FpType> v1) {
	assert(size == v0.dev_dim * get_device_count());
	assert(v0.dev_dim == v1.dev_dim);
	for (int i = 0; i < get_device_count(); i++) {
	    add_vec(i, v0.dev_dim, v0.dev_ptrs[i], v1.dev_ptrs[i]);
	}
	sync_devices();
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			complex<FpType> s, ConstBufType<FpType> v1) {
	assert(size == v0.dev_dim * get_device_count());
	assert(v0.dev_dim == v1.dev_dim);
	for (int i = 0; i < get_device_count(); i++) {
	    add_vec(i, v0.dev_dim, v0.dev_ptrs[i], s, v1.dev_ptrs[i]);
	}
	sync_devices();
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0, FpType s,
			ConstBufType<FpType> v1) {
	assert(size == v0.dev_dim * get_device_count());
	assert(v0.dev_dim == v1.dev_dim);
	for (int i = 0; i < get_device_count(); i++) {
	    add_vec(i, v0.dev_dim, v0.dev_ptrs[i], s, v1.dev_ptrs[i]);
	}
	sync_devices();
    }

    // Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> res,
			ConstBufType<FpType> v0, ConstBufType<FpType> v1) {
	assert(size == v0.dev_dim * get_device_count());
	assert(v0.dev_dim == v1.dev_dim);
	assert(res.dev_dim == v1.dev_dim);
	for (int i = 0; i < get_device_count(); i++) {
	    add_vec(i, v0.dev_dim, res.dev_ptrs[i], v0.dev_ptrs[i], v1.dev_ptrs[i]);
	}
	sync_devices();
    }

    // Compute v *= s where s is a real scalar.
    template<RealScalar FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v, FpType s) {
	assert(size == v.dev_dim * get_device_count());
	for (int i = 0; i < get_device_count(); i++) {
	    scale_vec(i, v.dev_dim, v.dev_ptrs[i], s);
	}
	sync_devices();
    }

    // Compute v *= s where s is a complex scalar.
    template<RealScalar FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v,
			  complex<FpType> s) {
	assert(size == v.dev_dim * get_device_count());
	for (int i = 0; i < get_device_count(); i++) {
	    scale_vec(i, v.dev_dim, v.dev_ptrs[i], s);
	}
	sync_devices();
    }

    // Compute the vector inner product of the complex vector v0 and v1.
    // In other words, compute v = <v0 | v1> = v0^\dagger \cdot v1
    template<RealScalar FpType>
    static complex<FpType> vec_prod(size_t size, ConstBufType<FpType> v0,
				    ConstBufType<FpType> v1) {
	assert(size == v0.dev_dim * get_device_count());
	assert(v0.dev_dim == v1.dev_dim);
	std::vector<complex<FpType>> vres(get_device_count());
	for (int i = 0; i < get_device_count(); i++) {
	    CUBLAS_CALL(cublas_dotc(i, v0.dev_dim, vres[i], v0.dev_ptrs[i], v1.dev_ptrs[i]));
	}
	sync_devices();
	complex<FpType> res{0.0};
	for (int i = 0; i < get_device_count(); i++) {
	    res += vres[i];
	}
	return res;
    }

    // Compute the vector norm of the complex vector v. In other words, compute
    // n = sqrt(<v|v>) = sqrt(v^\dagger \cdot v)
    template<RealScalar FpType>
    static FpType vec_norm(size_t size, ConstBufType<FpType> v) {
	return std::sqrt(vec_prod(size, v, v).real());
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
	assert(get_device_count() == 1);
	CUBLAS_CALL(cublas_gemv(dim, dim, res.dev_ptrs[0], {1,0}, mat.dev_ptr,
				vec.dev_ptrs[0], {}, adjoint));
	sync_device(0);
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
	sync_device(0);
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
    assert(size == v.dev_dim * get_device_count());
    for (int i = 0; i < get_device_count(); i++) {
	CURAND_CALL(gen_norm(i, (FpType *)v.dev_ptrs[i], v.dev_dim * 2));
    }
    sync_devices();
    // Normalize the resulting vector
    FpType c = 1/vec_norm(size, v);
    scale_vec(size, v, c);
}

template <RealScalar FpType>
inline void DevSumOps<FpType>::upload(typename SpinOp<FpType>::IndexType len) const {
    if (dev_ops.size() || dev_mat) {
	return;
    }
    if (this->ops.size() < (1ULL << len)) {
	dev_ops.resize(GPUImpl::get_device_count());
	auto size = this->ops.size() * sizeof(SpinOp<FpType>);
	for (int i = 0; i < GPUImpl::get_device_count(); i++) {
	    GPUImpl::set_device(i);
	    cuda_alloc(dev_ops[i], size);
	}
	for (int i = 0; i < GPUImpl::get_device_count(); i++) {
	    CUDA_CALL(cudaMemcpyAsync(dev_ops[i], this->ops.data(),
				      size, cudaMemcpyHostToDevice));
	}
	GPUImpl::sync_devices();
    } else {
	get_matrix(len);
    }
}

template <RealScalar FpType>
inline void DevSumOps<FpType>::release() const {
    // We must call the parent class release() function.
    HostSumOps<FpType>::release();
    mini_release();
    if (dev_ops.size()) {
	for (int i = 0; i < GPUImpl::get_device_count(); i++) {
	    cudaFree(dev_ops[i]);
	    dev_ops[i] = nullptr;
	}
    }
    dev_mat.reset(nullptr);
    host_eigenvals.resize(0);
    if (dev_eigenvals) {
	cudaFree(dev_eigenvals);
	dev_eigenvals = nullptr;
    }
}

template<RealScalar FpType>
inline typename DevSumOps<FpType>::MatrixType DevSumOps<FpType>::MatrixType::Identity(
    ssize_t rowdim, ssize_t coldim) {
    MatrixType id(rowdim, coldim);
    GPUImpl::eye<FpType>(id);
    GPUImpl::sync_device(0);
    return id;
}

template<RealScalar FpType>
inline typename DevSumOps<FpType>::MatrixType &DevSumOps<FpType>::MatrixType::operator+=(
    const DevSumOps<FpType>::MatrixType &other) {
    if (rowdim != other.rowdim || coldim != other.coldim) {
	ThrowException(InvalidArgument, "matrix dimensions", "must match");
    }
    GPUImpl::add_vec<FpType>(0, rowdim * coldim, dev_ptr, other.dev_ptr);
    GPUImpl::sync_device(0);
    return *this;
}

template<RealScalar FpType>
inline typename DevSumOps<FpType>::MatrixType &DevSumOps<FpType>::MatrixType::operator-=(
    const DevSumOps<FpType>::MatrixType &other) {
    if (rowdim != other.rowdim || coldim != other.coldim) {
	ThrowException(InvalidArgument, "matrix dimensions", "must match");
    }
    GPUImpl::add_vec<FpType>(0, rowdim * coldim, dev_ptr, {-1,0}, other.dev_ptr);
    GPUImpl::sync_device(0);
    return *this;
}

template<RealScalar FpType>
inline typename DevSumOps<FpType>::MatrixType DevSumOps<FpType>::MatrixType::operator*(
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
    GPUImpl::sync_device(0);
    return res;
}

template<RealScalar FpType>
inline typename DevSumOps<FpType>::MatrixType &DevSumOps<FpType>::MatrixType::operator*=(
    complex<FpType> c) {
    GPUImpl::scale_vec<FpType>(0, rowdim * coldim, dev_ptr, c);
    GPUImpl::sync_device(0);
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
    FpType n = 0.0;
    CUBLAS_CALL(GPUImpl::cublas_nrm2(rowdim * coldim, n, dev_ptr));
    GPUImpl::sync_device(0);
    return n;
}

template<RealScalar FpType>
inline typename DevSumOps<FpType>::EigenSystem DevSumOps<FpType>::get_eigensystem(int len) const {
    size_t dim = 1ULL << len;
    if (!dev_eigenvals || eigenvecs->rowdim != dim) {
	if (dev_eigenvals) {
	    cudaFree(dev_eigenvals);
	}
	GPUImpl::set_device(0);
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
	GPUImpl::sync_device(0);

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
	CUSOLVER_CALL(cusolverDnDestroyParams(params));
    }
    assert(dev_eigenvals);
    assert(host_eigenvals.rows() == (ssize_t)dim);
    assert(host_eigenvals.cols() == 1);
    assert(eigenvecs->rowdim == dim);
    assert(eigenvecs->coldim == dim);
    return {host_eigenvals, *eigenvecs};
}

template<RealScalar FpType>
inline typename DevSumOps<FpType>::MatrixType DevSumOps<FpType>::_matexp(complex<FpType> c,
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


template<RealScalar FpType>
inline std::ostream &operator<<(std::ostream &os, const DevSumOps<FpType> &s) {
    os << static_cast<const HostSumOps<FpType> &>(s) << " devptrs";
    for (int i = 0; i < GPUImpl::get_device_count(); i++) {
	os << " " << s.dev_ops[i];
    }
    return os;
}

using DefImpl = GPUImpl;
