/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    cpuimpl.h

Abstract:

    This header file contains the OpenMP implementation of vector operations.

Revision History:

    2023-04-21  File created as part of the code restructuring

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

// This class represents a sum of spin operators in host memory
//
//   O = \sum_i O_i
//
// where each O_i is a spin operator.
template<RealScalar FpType = DefFpType>
class HostSumOps {
    template<RealScalar FpType1>
    friend class HostSumOps;

public:
    using MatrixType = Eigen::Matrix<std::complex<FpType>,
				     Eigen::Dynamic, Eigen::Dynamic>;
    using EigenVals = Eigen::Matrix<FpType, Eigen::Dynamic, 1>;
    using EigenVecs = MatrixType;

protected:
    std::vector<SpinOp<FpType>> ops;
    mutable MatrixType mat;
    mutable EigenVals eigenvals;
    mutable EigenVecs eigenvecs;

    // Release the mutable internal state used for caching the results of the
    // get_matrix() and related functions. Note this should NOT be marked as
    // const (despite the fact that it does not modify any non-mutable member),
    // because we should not call release() in a const member function.
    // Additionally, any derived class MUST call the parent class release()
    // function if it has any mutable internal state (and therefore has to
    // override this function).
    virtual void release() {
	// This will delete the underlying arrays for the matrices.
	mat.resize(0, 0);
	eigenvals.resize(0, 1);
	eigenvecs.resize(0, 0);
    }

public:
    // We can implicitly convert a spin operator into a HostSumOps with one term
    HostSumOps(const SpinOp<FpType> &op) : ops{op} {}

    // Conversion between sum operators of different floating point precision
    // must be explicit.
    template<RealScalar FpType1>
    explicit HostSumOps(const HostSumOps<FpType1> &op) {
	for (auto o : op) {
	    if (o != 0.0) {
		ops.emplace_back(o);
	    }
	}
    }

    // We can initialize a HostSumOps with a list of SpinOp objects
    HostSumOps(const std::initializer_list<SpinOp<FpType>> &l = {}) {
	for (auto &&o : l) {
	    if (o != 0.0) {
		*this += o;
	    }
	}
    }

    // It's good idea to mark the base class destructor as virtual (strictly
    // speaking not necessary, as long as the user never deletes through the
    // base pointer, but it's a good habit to do so).
    virtual ~HostSumOps() {}

    // Assignment operator. We must release the internal mutable state.
    HostSumOps &operator=(const HostSumOps &rhs) {
	// Guard against self assignment
	if (this == &rhs) {
	    return *this;
	}
	release();
	ops = rhs.ops;
	return *this;
    }

    // Assignment operator. We must release the internal mutable state.
    // Note here we must convert the floating point type.
    template<RealScalar FpType1>
    HostSumOps &operator=(const HostSumOps<FpType1> &rhs) {
	release();
	ops.clear();
	for (auto op : rhs) {
	    if (op != 0.0) {
		ops.emplace_back(SpinOp<FpType>(op));
	    }
	}
	return *this;
    }

    // Assignment operator. We must release the internal mutable state.
    HostSumOps &operator=(const SpinOp<FpType> &rhs) {
	release();
	ops.clear();
	if (rhs != 0.0) {
	    ops.emplace_back(rhs);
	}
	return *this;
    }

    bool operator==(const HostSumOps &rhs) const {
	HostSumOps res{*this - rhs};
	return res.ops.empty();
    }

    template<ScalarType S>
    bool operator==(S s) const {
	if (s == S{}) {
	    return ops.empty();
	} else {
	    return (ops.size() == 1) && (ops[0] == s);
	}
    }

    // We don't allow the user to modify the operators using iterators.
    auto begin() const { return std::cbegin(ops); }
    auto end() const { return std::cend(ops); }

    // Add another spin operator to the summed operators. If we already have the
    // same spin operator (except with potentially a different coefficient) in the
    // list of operators, just add up the coefficients. Otherwise, append the new
    // operator to the list of operators.
    HostSumOps &operator+=(const SpinOp<FpType> &op) {
	release();
	if (op == 0) {
	    return *this;
	}
	// If we already have a spin operator with the same base (ie. the bits
	// member of SpinOp) in our sum, just add up the coefficients.
	for (auto it = ops.begin(); it != ops.end(); it++) {
	    if (it->bits == op.bits) {
		it->coeff += op.coeff;
		// If the coefficient is now zero, we remove the operator.
		if (it->coeff == 0.0) {
		    ops.erase(it);
		}
		return *this;
	    }
	}
	// Otherwise, add the operator to the end of the list.
	ops.push_back(op);
	return *this;
    }

    HostSumOps &operator-=(SpinOp<FpType> op) {
	release();
	op *= -1;
	return *this += op;
    }

    // Add another HostSumOps to this one
    HostSumOps &operator+=(const HostSumOps &rhs) {
	release();
	for (auto &&o : rhs) {
	    *this += o;
	}
	return *this;
    }

    HostSumOps &operator-=(HostSumOps rhs) {
	release();
	rhs *= -1;
	return *this += rhs;
    }

    // Add the supplied operator and this operator together, without changing this
    // operator, and return the result.
    HostSumOps operator+(const SpinOp<FpType> &op) const {
	HostSumOps res(*this);
	res += op;
	return res;
    }

    HostSumOps operator+(const HostSumOps &op) const {
	HostSumOps res(*this);
	res += op;
	return res;
    }

    HostSumOps operator-(const SpinOp<FpType> &op) const {
	HostSumOps res(*this);
	res -= op;
	return res;
    }

    HostSumOps operator-(const HostSumOps &op) const {
	HostSumOps res(*this);
	res -= op;
	return res;
    }

    // Multiply the specified operator from the right with this operator
    HostSumOps &operator*=(const HostSumOps &rhs) {
	release();
	*this = operator*(rhs);
	return *this;
    }

    // Scalar multiplication
    HostSumOps &operator*=(const complex<FpType> &rhs) {
	release();
	*this = operator*(rhs);
	return *this;
    }

    // Scalar "division", defined as scalar multiplication by 1/s
    HostSumOps &operator/=(const complex<FpType> &s) {
	release();
	*this = operator*(complex<FpType>{1.0,0.0}/s);
	return *this;
    }

    // Multiply the specified operator from the right with this operator (without
    // changing this operator), and return the result.
    HostSumOps operator*(const HostSumOps &rhs) const {
	HostSumOps res;
	for (auto &&op0 : *this) {
	    for (auto &&op1 : rhs) {
		res += op0 * op1;
	    }
	}
	return res;
    }

    // Multiply the specified complex scalar with this operator (without
    // changing this operator), and return the result.
    HostSumOps operator*(complex<FpType> rhs) const {
	HostSumOps res;
	for (auto &&op0 : *this) {
	    res += rhs * op0;
	}
	return res;
    }

    // Scalar multiplication can be done from both directions.
    friend HostSumOps operator*(complex<FpType> s, const HostSumOps &ops) {
	return ops * s;
    }

    // Returns true if all operators in the sum are Hermitian.
    bool is_hermitian() const {
	for (auto &op : *this) {
	    if (!op.is_hermitian()) {
		return false;
	    }
	}
	return true;
    }

    // Returns the (dense) matrix corresponding to the sum of operators.
    const MatrixType &get_matrix(int spin_chain_len) const {
	size_t dim = 1ULL << spin_chain_len;
	if ((size_t)mat.rows() == dim) {
	    assert((size_t)mat.cols() == dim);
	    return mat;
	}
	mat = MatrixType::Zero(dim, dim);
	for (auto &op : ops) {
	    auto spmat{op.get_sparse_matrix(spin_chain_len)};
	    const auto &bitmap{std::get<0>(spmat)};
	    const auto &cols{std::get<1>(spmat)};
	    #pragma omp parallel for
	    for (size_t i = 0; i < dim; i++) {
		auto v = bitmap[i] ? -op.coeff : op.coeff;
		mat(i, cols[i]) += std::complex{v.real(), v.imag()};
	    }
	}
	return mat;
    }

    using EigenSystem = std::tuple<const EigenVals &, const EigenVecs &>;

    // Returns the eigen system (eigen values and eigen vectors) of the sum
    // of operators. EigenVals is a column vector of dimension n where n=1<<len.
    // EigenVecs is a nxn matrix. Here len is the length of the spin chain.
    // These satisfy EigenVecs * diag(EigenVals) * EigenVecs^dagger = M,
    // where M is the matrix corresponding to the operator sum.
    // Note the operator sum must be Hermitian. We assert if it is not.
    EigenSystem get_eigensystem(int len) const {
	assert(is_hermitian());
	if (eigenvals.rows() == (1LL << len)) {
	    assert(eigenvals.cols() == 1);
	    assert(eigenvecs.rows() == (1LL << len));
	    assert(eigenvecs.cols() == (1LL << len));
	    return {eigenvals, eigenvecs};
	}
	Eigen::SelfAdjointEigenSolver<MatrixType> solver(get_matrix(len));
	if (solver.info() != Eigen::Success) {
	    ThrowException(RuntimeError, "Eigen", solver.info());
	}
	eigenvals = solver.eigenvalues();
	eigenvecs = solver.eigenvectors();
	return {eigenvals, eigenvecs};
    }

    // Compute the matrix exponential using exact diagonalization. Note you
    // should NOT call this function if you want to compute state evolution
    // (especially if you want to use Krylov). Call State::evolve() instead.
    // Here len is the length of the spin chain.
    MatrixType matexp(complex<FpType> c, int len) const {
	EigenSystem eigensys = get_eigensystem(len);
	const EigenVals &eigenvals = std::get<0>(eigensys);
	assert(eigenvals.rows() == (1LL << len));
	assert(eigenvals.cols() == 1);
	const EigenVecs &eigenvecs = std::get<1>(eigensys);
	assert(eigenvecs.rows() == (1LL << len));
	assert(eigenvecs.cols() == (1LL << len));
	using ComplexVector = Eigen::Matrix<std::complex<FpType>,
	    Eigen::Dynamic, 1>;
	ComplexVector vexp = eigenvals.unaryExpr([=](FpType v){
	    FpType re = v * c.real();
	    FpType im = v * c.imag();
	    return std::exp(re) * std::complex{std::cos(im), std::sin(im)};
	});
	MatrixType vexpevd = vexp.asDiagonal() * eigenvecs.adjoint();
	return eigenvecs * vexpevd;
    }

    friend std::ostream &operator<<(std::ostream &os, const HostSumOps &s) {
	bool first = true;
	if (s.ops.empty()) {
	    os << "0";
	}
	for (auto &&op : s) {
	    std::stringstream ss;
	    ss << op;
	    std::string str(ss.str());
	    if (str == "0") {
		// Skip this operator if it is zero
		continue;
	    }
	    if (!first) {
		if (str[0] == '-') {
		    // Insert additional spaces around the minus sign if the operator
		    // coefficient is a negative real number.
		    str.insert(str.begin()+1, ' ');
		    str.insert(str.begin(), ' ');
		} else {
		    // Otherwise, push an additional plus sign
		    str.insert(0, " + ");
		}
	    }
	    os << str;
	    first = false;
	}
	return os;
    }
};

// Class that implements basic vector operations on the CPU. This is for testing
// purpose only and is not suitable for realistic calculations.
// Note: we cannot make this a template because this class has static members
// that can only be initialized once. In terms of design patterns this is a
// so-called 'singleton' class.
class CPUImpl {
    template<RealScalar FpType>
    class HostVec {
	size_t dim;
	std::unique_ptr<complex<FpType>[]> vec;
	// We don't allow copy assignment or copy construction of vectors.
	// Move construction and move assignment are allowed.
	HostVec(const HostVec &) = delete;
	HostVec &operator=(const HostVec &) = delete;
    public:
	HostVec(size_t dim) : dim(dim), vec(std::make_unique<complex<FpType>[]>(dim)) {}
	HostVec(HostVec &&) = default;
	HostVec &operator=(HostVec &&) = default;
	complex<FpType> *get() { return vec.get(); }
	complex<FpType> *get() const { return vec.get(); }
	friend void swap(HostVec &lhs, HostVec &rhs) {
	    HostVec v0(std::move(lhs));
	    lhs = std::move(rhs);
	    rhs = std::move(v0);
	}
	size_t dimension() const { return dim; }
    };

    class Context {
	friend class CPUImpl;
	std::random_device rd;
	std::mt19937 randgen;
	Context() : randgen(rd()) {}
    };

    inline static Context ctx;

public:
    template<RealScalar FpType>
    using VecType = HostVec<FpType>;

    template<RealScalar FpType>
    using VecSizeType = typename SpinOp<FpType>::StateType;

    template<RealScalar FpType>
    using BufType = complex<FpType> *;

    template<RealScalar FpType>
    using ConstBufType = const complex<FpType> *;

    template<RealScalar FpType>
    using ElemRefType = complex<FpType> &;

    template<RealScalar FpType>
    using SumOps = HostSumOps<FpType>;

    // Initialize the vector to a Haar random state.
    template<RealScalar FpType>
    static void init_random(VecSizeType<FpType> size, BufType<FpType> v0) {
	std::normal_distribution<FpType> norm(0.0, 1.0);
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] = complex<FpType>{norm(ctx.randgen), norm(ctx.randgen)};
	}
	FpType s = 1.0/vec_norm(size, v0);
	scale_vec(size, v0, s);
    }

    // Initialize the vector with zero, ie. set v0[i] to 0.0 for all i.
    template<RealScalar FpType>
    static void zero_vec(VecSizeType<FpType> size, BufType<FpType> v0) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] = 0.0;
	}
    }

    // Copy v1 into v0, possibly with a different floating point precision.
    template<RealScalar FpType0, RealScalar FpType1>
    static void copy_vec(VecSizeType<FpType0> size, BufType<FpType0> v0,
			 ConstBufType<FpType1> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType0> i = 0; i < size; i++) {
	    v0[i] = v1[i];
	}
    }

    // Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] += v1[i];
	}
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			FpType s, ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] += s * v1[i];
	}
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			complex<FpType> s, ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] += s * v1[i];
	}
    }

    // Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
    template<RealScalar FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> res,
			ConstBufType<FpType> v0, ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    res[i] = v0[i] + v1[i];
	}
    }

    // Compute v0 *= s where s is a real scalar.
    template<RealScalar FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v0, FpType s) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] *= s;
	}
    }

    // Compute v0 *= s where s is a complex scalar.
    template<RealScalar FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			  complex<FpType> s) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] *= s;
	}
    }

    // Compute the vector inner product of the complex vector v0 and v1. In other words,
    // compute v = <v0 | v1> = v0^\dagger \cdot v1
    template<RealScalar FpType>
    static complex<FpType> vec_prod(VecSizeType<FpType> size,
				    ConstBufType<FpType> v0,
				    ConstBufType<FpType> v1) {
	FpType real = 0.0;
	FpType imag = 0.0;
	#pragma omp parallel for reduction (+:real,imag)
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    complex<FpType> res = conj(v0[i]) * v1[i];
	    real += res.real();
	    imag += res.imag();
	}
	return complex<FpType>{real, imag};
    }

    // Compute the vector norm of the complex vector v0. In other words, compute
    // v = sqrt(<v0 | v0>) = sqrt(v0^\dagger \cdot v0)
    template<RealScalar FpType>
    static FpType vec_norm(VecSizeType<FpType> size, ConstBufType<FpType> v0) {
	return std::sqrt(vec_prod(size, v0, v0).real());
    }

    // Compute the matrix multiplication res = mat * vec, where mat is of
    // type HostSumOps::MatrixType. Here res and vec are assumed to be
    // different buffers.
    template<RealScalar FpType>
    static void mat_mul(VecSizeType<FpType> dim, BufType<FpType> res,
			const typename HostSumOps<FpType>::MatrixType &mat,
			ConstBufType<FpType> vec) {
	using VectorType = Eigen::Matrix<complex<FpType>, Eigen::Dynamic, 1>;
	Eigen::Map<VectorType> mres(res,dim,1);
	Eigen::Map<const VectorType> mvec(vec,dim,1);
	mres = mat * mvec;
    }

    // Compute res += ops * vec. res and vec are assumed to be different.
    template<RealScalar FpType>
    static void apply_ops(typename SpinOp<FpType>::IndexType len, BufType<FpType> res,
			  const HostSumOps<FpType> &ops, ConstBufType<FpType> vec) {
	assert(len <= 64);
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < (VecSizeType<FpType>(1) << len); i++) {
	    for (const auto &op : ops) {
		VecSizeType<FpType> n = 0;
		auto coeff = op.apply(i, n);
		res[i] += coeff * vec[n];
	    }
	}
    }
};
