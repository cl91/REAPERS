/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    state.h

Abstract:

    This header file contains the State class definition for the REAPERS library.

Revision History:

    2023-01-05  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

// Forward declaration. This is needed by algo.h
template<RealScalar FpType, typename Impl>
class State;

namespace EvolutionAlgorithm {
#include "algo.h"
}

#ifdef REAPERS_USE_EXACT_DIAGONALIZATION
template<typename Impl>
using DefEvolutionAlgorithm = EvolutionAlgorithm::ExactDiagonalization<Impl>;
#else
template<typename Impl>
using DefEvolutionAlgorithm = EvolutionAlgorithm::Krylov<Impl>;
#endif

// This class represents a vector in the single-parity Hilbert space of the spin model.
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
class State {
    template<RealScalar FpTy1, typename Impl1>
    friend class State;

    using IndexType = typename SpinOp<FpType>::StateType;

    // Spin chain length. In other words, the dimension of the Hilbert space is 1<<len;
    typename SpinOp<FpType>::IndexType len;

    // Current buffer index. The current state vector is bufs[curbuf]. All other vectors
    // in bufs are free to use as scratch space during state evolution.
    int curbuf;

    // During state evolution we will need multiple internal buffers. To reduce the
    // number of allocation and deallocation calls, we delay the deallocation until
    // user explicitly calls gc().
    std::vector<typename Impl::template VecType<FpType>> bufs;

    void check_state_index(IndexType n) const {
	if (n >= dim()) {
	    ThrowException(StateIndexTooLarge, n, dim()-1);
	}
    }

public:
    // Default constructor. We set len to zero to indicate that the state is
    // zero. This is so parity block operations can skip the zero parity blocks.
    State() : len(0), curbuf(0) {}

    // Construct an uninitialized state vector
    State(typename SpinOp<FpType>::IndexType len) : len(len), curbuf(0) {
	if (len == 0) {
	    ThrowException(InvalidArgument, "len", "cannot be zero");
	}
	if (len > NUM_SPIN_SITES) {
	    ThrowException(InvalidArgument, "len", "cannot be larger than NUM_SPIN_SITES");
	}
	bufs.emplace_back(dim());
    }

    // Copy constructor. Note that copy-ctor cannot be a template.
    State(const State &st) : len(st.len), curbuf(0) {
	if (!len) return;
	bufs.emplace_back(dim());
	Impl::copy_vec(dim(), buf(), st.buf());
    }

    // Move constructor.
    State(State &&st) : State() {
	using std::swap;
	swap(len, st.len);
	swap(bufs, st.bufs);
	swap(curbuf, st.curbuf);
    }

    // Destructor. The dtor for bufs is called automatically by the compiler.
    ~State() { len = 0; curbuf = 0; }

    // Conversion constructor between states with different floating point
    // precisions. In this case the conversions must be explicit.
    template<RealScalar FpType1>
    explicit State(const State<FpType1, Impl> &st) : len(st.len), curbuf(0) {
	if (!len) return;
	bufs.emplace_back(dim());
	Impl::copy_vec(dim(), buf(), st.buf());
    }

    // Conversion constructor between states with different vector operation
    // implementations (ie. from host vector to device vector, or vice verse).
    // In this case the conversions must be explicit.
    template<RealScalar FpType1, typename Impl1>
    explicit State(const State<FpType1, Impl1> &st) : len(st.len), curbuf(0) {
	if (!len) return;
	bufs.emplace_back(static_cast<typename Impl::template VecType<FpType>>(st.bufs[st.curbuf % st.num_bufs()]));
    }

    // Make the state a Haar random state
    void random_state() {
	if (!len) return;
	Impl::init_random(dim(), buf());
    }

    // Make the state a zero state
    void zero_state() {
	if (!len) return;
	Impl::zero_vec(dim(), buf());
    }

    int num_bufs() const {
	assert(bufs.size() != 0);
	return bufs.size();
    }

    // Length of the spin chain
    auto spin_chain_length() const { return len; }

    // Dimension of the Hilbert space.
    IndexType dim() const { return 1 << len; }

    // Be careful when calling this function as this returns the raw pointer to the
    // current state vector. You should never save this pointer.
    typename Impl::template BufType<FpType> buf(int i = 0) {
	assert(len != 0);
	assert(curbuf >= 0);
	assert(curbuf < num_bufs());
	assert(num_bufs() >= (i+1));
	return bufs[(curbuf+i) % num_bufs()].get();
    }

    typename Impl::template ConstBufType<FpType> buf(int i = 0) const {
	assert(len != 0);
	assert(curbuf >= 0);
	assert(curbuf < num_bufs());
	assert(num_bufs() >= (i+1));
	return bufs[(curbuf+i) % num_bufs()].get();
    }

    void inc_curbuf(int i = 1) {
	assert(len != 0);
	assert(num_bufs() != 0);
	if (!len) return;
	curbuf = (curbuf + i) % num_bufs();
    }

    // Allocate additional internal buffers such that total number of buffers is
    // at least n.
    void enlarge(int n) {
	assert(len != 0);
	if (!len) return;
	int nbufs = num_bufs();
	if (nbufs < n) {
	    for (int i = 0; i < (n - nbufs); i++) {
		bufs.emplace_back(dim());
	    }
	}
    }

    void swap_bufs(int i, int j) {
	assert(len != 0);
	if (!len) return;
	auto n = num_bufs();
	i += curbuf;
	j += curbuf;
	i %= n;
	j %= n;
	if (i != j) {
	    swap(bufs[i], bufs[j]);
	}
    }

    // Compute the ground state of the given Hamiltonian, using the Lanczos
    // algorithm. When finished, the state is set the ground state and the
    // energy is returned. You can optionally specify the Krylov dimension
    // and the tolerance used in the algorithm.
    FpType ground_state(const typename Impl::template SumOps<FpType> &ham,
			int krydim = 5, FpType eps = epsilon<FpType>()) {
	assert(len != 0);
	if (!len) return {};
	assert(krydim > 1);
	// We need krydim internal buffers for the krydim Krylov vectors, ie.
	// v[0] = buf(0), v[1] = buf(1), ..., v[krydim-1] = buf(krydim-1).
	// When one Lanczos iteration finishes, we compute the ground state and
	// increment the internal buffer pointer to point to the ground state.
	enlarge(krydim+1);
	std::vector<FpType> alpha(krydim); // Lanczos diagonal entries
	std::vector<FpType> beta(krydim-1); // Lanczos off-diagonal entries
	// Initialize the state to a Haar random state
	random_state();

    iter:
	// Do the Lanczos procedure to generate the Krylov subspace
	// Although we only need to generate (krydim-1) Krylov vectors,
	// we do an additional iteration to compute alpha[i].
	for (int i = 0; i < krydim; i++) {
	    // Each iteration will compute the next Krylov vector v[i+1]
	    // and the current Krylov coefficients, alpha[i] and beta[i].
	    // We start from a zero vector v[i+1]
	    Impl::zero_vec(dim(), buf(i+1));
	    // v[i+1] = H * v[i]
	    Impl::apply_ops(len, buf(i+1), ham, buf(i));
	    // alpha[i] = <v[i]|v[i+1]> = <v[i]|H|v[i]>.
	    // For Hermitian H this is always real.
	    alpha[i] = Impl::vec_prod(dim(), buf(i), buf(i+1)).real();
	    // For the last iteration we only need alpha[i]
	    if (i == (krydim - 1)) {
		break;
	    }
	    // v[i+1] -= alpha[i] * v[i]
	    Impl::add_vec(dim(), buf(i+1), -alpha[i], buf(i));
	    if (i != 0) {
		// v[i+1] -= beta[i-1] * v[i-1]
		Impl::add_vec(dim(), buf(i+1), -beta[i-1], buf(i-1));
	    }
	    // beta[i] = sqrt(<v[i+1]|v[i+1]>)
	    beta[i] = Impl::vec_norm(dim(), buf(i+1));
	    // v[i+1] /= beta[i]
	    Impl::scale_vec(dim(), buf(i+1), 1/beta[i]);
	}

	// Compute the ground state by diagonalizing in Krylov subspace
	using SmallMat = Eigen::Matrix<FpType, Eigen::Dynamic, Eigen::Dynamic>;
	using SmallVec = Eigen::Matrix<FpType, Eigen::Dynamic, 1>;
	Eigen::SelfAdjointEigenSolver<SmallMat> solver;
	SmallVec a = Eigen::Map<SmallVec>(alpha.data(), krydim, 1);
	SmallVec b = Eigen::Map<SmallVec>(beta.data(), krydim-1, 1);
	solver.computeFromTridiagonal(a, b, Eigen::ComputeEigenvectors);
	int idx;
	auto energy = solver.eigenvalues().minCoeff(&idx);
	auto vec = solver.eigenvectors().col(idx);

	// Compute the ground state.
	Impl::zero_vec(dim(), buf(krydim));
	for (int i = 0; i < krydim; i++) {
	    Impl::add_vec(dim(), buf(krydim), vec[i], buf(i));
	}
	// Increment the internal buffer pointer to point to the ground state.
	inc_curbuf(krydim);
	normalize();

	// If tolerance is not reached, repeat the procedure but this time
	// start from the ground state that we just computed.
	if (beta[0] > eps) {
	    goto iter;
	}
	return energy;
    }

    auto operator[](IndexType i) const { assert(len); return buf()[i]; }
    auto operator[](IndexType i) { assert(len); return buf()[i]; }

    // We must repeat the non-template operator= definition since the template
    // version below does not cover the non-template version (this is similar
    // to the copy-ctor case).
    State &operator=(const State &st) {
	// Guard against self-assignment
	if (this == &st) {
	    return *this;
	}
	assert(st.len);
	// If dimensions match, the only thing we need to do is copying.
	// Otherwise, if the dimensions don't match, we release all the
	// buffers in this State and make a copy of the other State.
	if (dim() != st.dim()) {
	    bufs.clear();
	    len = st.len;
	    if (len) { bufs.emplace_back(dim()); }
	}
	if (len) { Impl::copy_vec(dim(), buf(), st.buf()); }
	return *this;
    }

    // Move assignment operator
    State &operator=(State &&st) {
	// Just in case anyone is stupid enough to write st = std::move(st)
	// (possibly due to aliasing: State &st1 = st0; st0 = std::move(st1)),
	// we guard against self assignment in the move assignment op as well.
	if (this == &st) { return *this; }
	using std::swap;
	swap(len, st.len);
	swap(curbuf, st.curbuf);
	swap(bufs, st.bufs);
	return *this;
    }

    template<RealScalar FpType1>
    State &operator=(const State<FpType1,Impl> &st) {
	assert(st.len);
	if (dim() != st.dim()) {
	    bufs.clear();
	    len = st.len;
	    if (len) { bufs.emplace_back(dim()); }
	}
	if (len) { Impl::copy_vec(dim(), buf(), st.buf()); }
	return *this;
    }

    State &operator+=(const State &rhs) {
	if (len != rhs.len) {
	    ThrowException(InvalidArgument, "dimensions of states must match");
	}
	assert(len);
	if (!len) return {};
	Impl::add_vec(dim(), buf(), rhs.buf());
	return *this;
    }

    State operator+(const State &rhs) const {
	if (len != rhs.len) {
	    ThrowException(InvalidArgument, "dimensions of states must match");
	}
	assert(len);
	if (!len) return {};
	State res(rhs.len);
	Impl::add_vec(dim(), res.buf(), buf(), rhs.buf());
	return res;
    }

    State &operator-=(const State &rhs) {
	if (len != rhs.len) {
	    ThrowException(InvalidArgument, "dimensions of states must match");
	}
	assert(len);
	if (len) {
	    Impl::add_vec(dim(), buf(), FpType(-1.0), rhs.buf());
	}
	return *this;
    }

    State operator-(const State &rhs) const {
	assert(len);
	if (!len) return {};
	State res{*this};
	res -= rhs;
	return res;
    }

    State &operator*=(const typename Impl::template SumOps<FpType> &ops) {
	assert(len);
	if (!len) return *this;
	enlarge(2);
	assert(num_bufs() >= 2);
	typename Impl::template BufType<FpType> v = buf();
	inc_curbuf();
	typename Impl::template BufType<FpType> res = buf();
	Impl::zero_vec(dim(), res);
	Impl::apply_ops(len, res, ops, v);
	return *this;
    }

    State &operator*=(complex<FpType> s) {
	assert(len);
	if (len) {
	    Impl::scale_vec(dim(), buf(), s);
	}
	return *this;
    }

    FpType norm() const {
	assert(len);
	if (!len) return {};
	return Impl::vec_norm(dim(), buf());
    }

    void normalize() {
	assert(len);
	if (!len) return;
	*this *= 1.0 / norm();
    }

    bool operator==(const State &st) const {
	if (len != st.len) {
	    return false;
	}
	if (!len) {
	    return true;
	}
	State diff{*this - st};
	return diff.norm() == 0.0;
    }

    friend State operator*(const typename Impl::template SumOps<FpType> &ops,
			   const State &s) {
	assert(s.len);
	if (!s.len) { return {}; }
	State res(s.len);
	res.zero_state();
	Impl::apply_ops(s.len, res.buf(), ops, s.buf());
	return res;
    }

    friend State operator*(const typename Impl::template SumOps<FpType>::MatrixType &mat,
			   const State &st) {
	assert(st.len);
	if (!st.len) { return {}; }
	if (mat.cols() != mat.rows()) {
	    std::stringstream ss;
	    ss << "(" << mat.cols() << ") must match its row dimension ("
	       << mat.rows() << ")";
	    ThrowException(InvalidArgument, "column dimension of matrix",
			   ss.str().c_str());
	}
	if (mat.cols() != (1LL << st.len)) {
	    std::stringstream ss;
	    ss << "(" << mat.cols() << ") must match the dimension of the state ("
	       << (1LL << st.len) << ")";
	    ThrowException(InvalidArgument, "column dimension of matrix",
			   ss.str().c_str());
	}
	State res(st.len);
	Impl::template mat_mul<FpType>(1ULL << st.len, res.buf(), mat, st.buf());
	return res;
    }

    friend complex<FpType> operator*(const State &s0, const State &s1) {
	if (s0.dim() != s1.dim()) {
	    ThrowException(InvalidArgument, "Dimensions of states", "must match");
	}
	assert(s0.len);
	if (!s0.len) { return {}; }
	return Impl::vec_prod(s0.dim(), s0.buf(), s1.buf());
    }

    friend std::ostream &operator<<(std::ostream &os, const State &s) {
	os << "State(dim=" << s.dim() << "){";
	if (!s.len) { os << "}"; return os; }
	for (IndexType i = 0; i < s.dim(); i++) {
	    if (i) {
		os << ", ";
	    }
	    auto c = s[i];
	    if (c == 1.0_i) {
		os << "i";
	    } else if (c == -1.0_i) {
		os << "-i";
	    } else if (c.real() * c.imag() != 0) {
		os << c.real() << (c.imag() > 0 ? " + " : " - ")
		   << (c.imag() > 0 ? c.imag() : -c.imag()) << "i";
	    } else if (c.imag() == 0) {
		os << c.real();
	    } else {
		os << c.imag() << "i";
	    }
	}
	os << "}";
	return os;
    }

    // Evolve the state using the following
    //   |psi> = exp(-(it + beta)H) |psi>
    template<template<typename>typename Algo = DefEvolutionAlgorithm,
	     RealScalar FpType1, RealScalar FpType2, typename...Args>
    void evolve(const typename Impl::template SumOps<FpType> &ham,
		FpType1 t, FpType2 beta = 0.0, Args&&...args) {
	assert(len);
	if ((t == 0.0) && (beta == 0.0)) { return; }
	if (!len) return;
	Algo<Impl>::evolve(*this, ham, static_cast<FpType>(t),
			   static_cast<FpType>(beta), std::forward<Args>(args)...);
    }

    // Release all the internal buffers allocated.
    void gc() {
	assert(len);
	if (!len) return;
	assert(curbuf >= 0);
	assert(curbuf < num_bufs());
	if (curbuf != 0) {
	    swap_bufs(0, -curbuf);
	}
	// We cannot call resize() because our Impl::VecType is not
	// copy construtable.
	for (int i = 0; i < (num_bufs()-1); i++) {
	    bufs.pop_back();
	}
	curbuf = 0;
    }
};

// Represents a state vector in parity block form (ie. column vector of left
// and right block).
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
struct BlockState : BlockVec<State<FpType,Impl>> {
    // Note here the spin chain length is the spin chain length of each
    // parity block, ie. the full Hilbert space dimension is 1<<(len+1).
    BlockState(typename SpinOp<FpType>::IndexType len)
	: BlockVec<State<FpType,Impl>>{len, len} {}

    void random_state() {
	this->L.random_state();
	this->R.random_state();
	normalize();
    }

    void zero_state() {
	this->L.zero_state();
	this->R.zero_state();
	// Note here we don't set nullL and nullR to true because both L
	// and R, despite being zero states, are not null objects (ie. they
	// are not equal to {}, a default constructed state).
	this->nullL = false;
	this->nullR = false;
    }

    // Length of the spin chain.
    typename SpinOp<FpType>::IndexType spin_chain_length() const {
	assert(this->L.spin_chain_length() == this->R.spin_chain_length());
	return this->L.spin_chain_length();
    }

    // Dimension of the Hilbert space. Note L.dim() might be 2^31 so we need
    // to cast to size_t.
    size_t dim() const { return (size_t)(this->L.dim()) * 2; }

    // Construct the ground state of the given Hamiltonian. Each parity block
    // is computed separately and we normalize the result.
    template<typename HamOpTy>
    requires std::derived_from<HamOpTy, typename Impl::template SumOps<FpType>>
    FpType ground_state(const BlockDiag<HamOpTy> &ham,
			int krydim = 3, FpType eps = 100*epsilon<FpType>()) {
	auto g0 = this->L.ground_state(ham.LL, krydim, eps);
	auto g1 = this->R.ground_state(ham.RR, krydim, eps);
	if (g0 <= g1) {
	    this->R = {};
	    this->nullR = true;
	    return g0;
	} else {
	    this->L = {};
	    this->nullL = true;
	    return g1;
	}
    }

    // The compiler doesn't seem to generate this automatically so
    // we need to explicitly call the base assignment operator.
    template<BlockVecType B>
    BlockState &operator=(B &&st) {
	BlockVec<State<FpType,Impl>>::operator=(std::forward<B>(st));
	return *this;
    }

    // We cannot use automatic type deduction here because the operator[]
    // of the State class might return a non-default construtable object.
    complex<FpType> operator[](size_t i) const {
	if (this->nullL && this->nullR) {
	    return {};
	} else if (this->nullL) {
	    // nullR is false here
	    auto dim = this->R.dim();
	    if (i < dim) {
		return {};
	    } else {
		return this->R[i-this->L.dim()];
	    }
	} else if (this->nullR) {
	    // nullL is false here
	    auto dim = this->L.dim();
	    if (i < dim) {
		return this->L[i];
	    } else {
		return {};
	    }
	}
	return i < this->L.dim() ? this->L[i] : this->R[i-this->L.dim()];
    }

    // For the non-const access operator we will have to throw if any of
    // the parity blocks is undefined.
    auto operator[](size_t i) {
	assert(!this->nullL && !this->nullR);
	if (this->nullL || this->nullR) {
	    ThrowException(InvalidArgument, "Both parity blocks must be defined.");
	}
	return i < this->L.dim() ? this->L[i] : this->R[i-this->L.dim()];
    }

    FpType norm() const {
	using Ty = std::remove_reference_t<decltype(this->L.norm())>;
	Ty nl = this->nullL ? Ty{} : this->L.norm();
	Ty nr = this->nullR ? Ty{} : this->R.norm();
	return std::sqrt(nl*nl + nr*nr);
    }

    void normalize() {
	if (this->nullL && this->nullR) { return; }
	*this *= 1.0 / norm();
    }

    // Evolve the state using the following formula
    //   |psi> = exp(-(it + beta)H) |psi>
    template<template<typename>typename Algo = DefEvolutionAlgorithm,
	     RealScalar FpType1, RealScalar FpType2, typename HamOpTy,
	     typename...Args>
    requires std::derived_from<HamOpTy, typename Impl::template SumOps<FpType>>
    void evolve(const BlockDiag<HamOpTy> &ham,
		FpType1 t, FpType2 beta = 0.0, Args&&...args) {
	if ((t == 0.0) && (beta == 0.0)) {
	    return;
	}
	if (!this->nullL) {
	    Algo<Impl>::evolve(this->L, ham.LL, static_cast<FpType>(t),
			       static_cast<FpType>(beta), args...);
	}
	if (!this->nullR) {
	    Algo<Impl>::evolve(this->R, ham.RR, static_cast<FpType>(t),
			       static_cast<FpType>(beta), args...);
	}
    }

    // Release all the internal buffers allocated.
    void gc() {
	if (!this->nullL) { this->L.gc(); }
	if (!this->nullR) { this->R.gc(); }
    }
};
