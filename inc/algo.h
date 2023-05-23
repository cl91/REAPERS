/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    algo.h

Abstract:

    This header file contains implementations of the state evolution algorithm
    for the REAPERS library.

Revision History:

    2023-03-08  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

// Compute the state evolution by expanding the matrix exponential to n-th power
template<typename Impl>
class MatrixPower {
    template<RealScalar FpType>
    using SumOps = typename Impl::template SumOps<FpType>;
public:
    // Compute s = exp(-(beta+it)H) * s by expanding to the n-th power.
    template<RealScalar FpType>
    static void evolve(State<FpType, Impl> &s, const SumOps<FpType> &ops,
		       FpType t, FpType beta = 0.0, int n = 1) {
	auto dim = s.dim();
	auto len = s.spin_chain_length();
	auto norm = s.norm();
	s *= 1/norm;
	// For CPUImpl, swap() will resolve to std::swap<std::unique_ptr>.
	using std::swap;
	// We need two internal buffers for H^k v
	s.enlarge(3);
	typename Impl::template BufType<FpType> orig = s.buf();
	typename Impl::template BufType<FpType> v1 = s.buf(1); // H^i v
	typename Impl::template BufType<FpType> v2 = s.buf(2); // H^{i+1} v
	complex<FpType> c = 1.0;
	Impl::copy_vec(dim, v1, orig);
	for (int i = 0; i < n; i++) {
	    Impl::zero_vec(dim, v2);
	    Impl::apply_ops(len, v2, ops, v1);
	    c *= complex<FpType>{-beta/(i+1), -t/(i+1)};
	    Impl::add_vec(dim, orig, c, v2);
	    swap(v1, v2);
	}
	s *= norm;
    }
};

// Compute the state evolution using exact diagonalization.
template<typename Impl>
class ExactDiagonalization {
    template<RealScalar FpType>
    using SumOps = typename Impl::template SumOps<FpType>;
public:
    template<RealScalar FpType>
    static void evolve(State<FpType, Impl> &s, const SumOps<FpType> &ops,
		       FpType t, FpType beta = 0.0) {
	s.enlarge(2);
	auto mexp = ops.matexp({-beta,-t}, s.spin_chain_length());
	Impl::mat_mul(s.dim(), s.buf(1), mexp, s.buf());
	s.inc_curbuf(1);
    }
};

// Compute the state evolution using the Krylov subspace method.
template<typename Impl>
class Krylov {
    template<RealScalar FpType>
    using SumOps = typename Impl::template SumOps<FpType>;
    // Low dimensional complex matrices that are stored and computed on the host
    template<RealScalar FpType>
    using SmallMat = Eigen::Matrix<std::complex<FpType>, Eigen::Dynamic, Eigen::Dynamic>;

    template<RealScalar FpType>
    using SmallVec = Eigen::Matrix<std::complex<FpType>, Eigen::Dynamic, 1>;

    // Orthogonalize buf(j+1) against buf(0) through buf(j) using the iterative
    // (classical) Gram-Schmidt algorithm. Returns false if orthogonalization has failed.
    // The orthogonalization coefficients will written to the H matrix, unless j is
    // equal to (m-1). The norm of the final vector is written to beta.
    template<RealScalar FpType>
    static bool orthogonalize(State<FpType, Impl> &s, SmallMat<FpType> &H, int j, int m,
			      FpType &beta, FpType eta = 1/sqrt(2), int max_iters = 3) {
	auto dim = s.dim();
	SmallVec<FpType> r(j+1), h(j+1);
	r.setZero();
	h.setZero();
	FpType old_norm{1.0}, norm{eta/2};
	int niter{0};
	// Only attempt another iteration if all of the following is true:
	//   1) We haven't tried enough times
	//   2) The norm of the resulting vector from the last iteration is not zero.
	//   3) The selective criterion eta * old_norm <= norm is not satisfied.
	while ((niter < max_iters) && norm && (norm < eta * old_norm)) {
	    niter++;
	    for (int i = 0; i <= j; i++) {
		auto a = Impl::vec_prod(dim, s.buf(i), s.buf(j+1));
		r(i) = std::complex{a.real(), a.imag()};
		Impl::add_vec(dim, s.buf(j+1), -a, s.buf(i));
	    }
	    h += r;
	    old_norm = norm;
	    norm = Impl::vec_norm(dim, s.buf(j+1));
	}
	beta = norm;
	for (int i = 0; i <= j; i++) {
	    H(i,j) = std::complex{h(i).real(), h(i).imag()};
	}
	// If the norm is not zero and the selective criterion is satisfied, the
	// orthogonalization procedure successfully produced a linearly independent vector
	return norm && (norm >= eta * old_norm);
    }

    // Computes the Arnoldi factorization associated with a matrix.
    //
    // This is a whole-sale copy of the BVMatArnoldi function in slepc.
    //
    // Input Parameters:
    //   A - the matrix
    //   H - the upper Hessenberg matrix
    //   k - number of locked columns
    //   m - dimension of the Arnoldi basis, will be modified if breakdown occurs
    //
    // Output Parameters:
    //   H - the upper Hessenberg matrix
    //   beta - norm of last vector before normalization
    //   m - dimension of the Arnoldi basis
    // Return Value:
    //   boolean value where false indicates that breakdown has occurred
    //
    // Notes:
    //   Computes an m-step Arnoldi factorization for matrix A. The first k columns
    //   are assumed to be locked and therefore they are not modified. On exit, the
    //   following relation is satisfied
    //
    //      A * V - V * H = beta * v_m * e_m
    //
    //   where the columns of V are the m Arnoldi vectors, [v_0, v_1, ..., v_{m-1}],
    //   all of which are orthonormal. v_m is the m-th vector generated from the
    //   procedure. H is a m*m upper Hessenberg matrix. e_m is the m-th row vector
    //   of the canonical basis (starting from e_0). On exit, beta contains the norm
    //   of v_m before normalization.
    //
    //   A false return value indicates that orthogonalization failed (norm too small).
    //   In that case, on exit m contains the index of the column that failed.
    //
    //   To create an Arnoldi factorization from scratch, set k = 0 and make sure the
    //   current buffer index points to the normalized initial vector.
    template<RealScalar FpType>
    static bool arnoldi(State<FpType,Impl> &s, const SumOps<FpType> &A,
			SmallMat<FpType> &H, int k, int &m, FpType &beta) {
	assert(k >= 0);
	assert(m >= (k+1));
	auto dim = s.dim();
	auto len = s.spin_chain_length();
	// Populate the internal buffers with Krylov vectors, starting from
	// the initial vector v[k] = buf(k). This will generate the Krylov
	// vectors from v[k+1] to v[m-1], as well as the final vector v[m],
	// unless the process terminates prematurely due to v[j+1] having
	// zero norm (in which case m is modified to indicate which vector
	// has a zero norm).
	for (int j = k; j < m; j++) {
	    // v[j+1] = A * v[j]
	    Impl::zero_vec(dim, s.buf(j+1));
	    Impl::apply_ops(len, s.buf(j+1), A, s.buf(j));
	    // Orthogonalize v[j+1] against v[0]..v[j].
	    bool ok = orthogonalize(s, H, j, m, beta);
	    if (!ok) {
		m = j+1;
		return false;
	    }
	    if (j != (m-1)) {
		H(j+1,j) = beta;
	    }
	    Impl::scale_vec(dim, s.buf(j+1), 1/beta);
	}
	return true;
    }

public:
    // Compute s = exp(-(beta+it)H) * s using the restarted Krylov subspace method.
    //
    // See "A Restarted Krylov Subspace Method for the Evaluation of Matrix Functions",
    //   Michael Eiermann, and Oliver G. Ernst.
    //   SIAM J. Numer. Anal. Vol. 44, No. 6, pp. 2481-2504
    //
    // This is the exact same algorithm implemented by slepc that dynamite calls.
    template<RealScalar FpType>
    static void evolve(State<FpType,Impl> &s, const SumOps<FpType> &ops, FpType t_real,
		       FpType t_imag = 0.0, int krydim = 5,
		       FpType tol = epsilon<FpType>(), int max_iters = 0) {
	assert(krydim >= 2);
	auto dim = s.dim();
	// We need krydim+1 internal buffers for krydim+1 Krylov vectors.
	// buf(0) through buf(krydim) are the krydim+1 Krylov vectors,
	// buf(krydim+1) is the final answer.
	s.enlarge(krydim+2);
	// Zero the final answer vector
	Impl::zero_vec(dim, s.buf(krydim+1));
	// Save the vector norm of the initial state vector v
	auto beta0 = Impl::vec_norm(dim, s.buf());
	// Set the zeroth Krylov vector to normalized v
	Impl::scale_vec(dim, s.buf(), 1/beta0);
	// Full upper Hessenberg matrix constructed from the iterations below. At the
	// end of each iterations its dimension is (niters*krydim) by (niters*krydim),
	// unless the Arnoldi procedure breaks down prematurely.
	SmallMat<FpType> Hfull(krydim, krydim);
	// Number of iterations
	int niters = 0;
	// Beta from the previous loop
	FpType oldbeta = 0.0;
	// Loop until algorithm has converged
	while (true) {
	    niters++;
	    // Compute the Arnoldi factorization
	    int m = krydim;
	    FpType beta;
	    SmallMat<FpType> H(m, m);
	    H.setZero();
	    bool brokedown = !arnoldi(s, ops, H, 0, m, beta);

	    // If we are the first iteration, simply set Hfull to H
	    if (niters == 1) {
		Hfull = H;
	    } else {
		// Append the block matrix H to Hfull in the following manner
		// (Algorithm 2 of the Eiermann paper)
		//
		// Hfull = [         OldHfull  0 ]
		//         [ 0 ... 0 oldbeta   H ] Row (niters-1)*krydim
		//                             ^
		//                    Column (niters-1)*krydim
		//
		// OldHfull is (niters-1)*krydim by (niters-1)*krydim so the
		// oldbeta element is in row (niters-1)*krydim and column
		// (niters-1)*krydim-1
		Hfull.conservativeResize((niters-1)*krydim+m, (niters-1)*krydim+m);
		for (int i = 0; i < (niters-1)*krydim; i++) {
		    for (int j = 0; j < m; j++) {
			Hfull(i, (niters-1)*krydim+j) = 0;
			Hfull((niters-1)*krydim+j, i) = 0;
		    }
		}
		Hfull((niters-1)*krydim, (niters-1)*krydim-1) = oldbeta;
		for (int i = 0; i < m; i++) {
		    for (int j = 0; j < m; j++) {
			Hfull((niters-1)*krydim+i, (niters-1)*krydim+j) = H(i,j);
		    }
		}
	    }

	    // Evaluate Hexp = exp(-(t_imag+it_real)Hfull).
	    SmallMat<FpType> tHfull = std::complex{-t_imag,-t_real} * Hfull;
	    SmallMat<FpType> Hexp = tHfull.exp();

	    // The update will be built from the last m elements of
	    // the zeroth column of Hexp
	    SmallVec<FpType> Hexp0(m);
	    for (int i = 0; i < m; i++) {
		Hexp0(i) = Hexp.col(0)((niters-1)*krydim+i);
	    }

	    // Update the final answer vector, buf(krydim+1) += beta0*V*Hexp0
	    for (int i = 0; i < m; i++) {
		std::complex<FpType> c = beta0 * Hexp0(i);
		Impl::add_vec(dim, s.buf(krydim+1), {c.real(),c.imag()}, s.buf(i));
	    }

	    // Compute the relative norm of the update ||u||/beta0.
	    // This is simply the Euclidean 2-norm of the last m elements
	    // of the Hexp.col(0) vector (ie. Hexp0).
	    FpType nrm = abs(Hexp0.norm());

	    // Check convergence. If a maximum number of iteration is specified,
	    // and we have reached it, then exit. If the Arnoldi procedure
	    // brokedown (this indicates an invariant subspace, ie. convergence),
	    // then exit. Otherwise, exit when the tolerance is achieved.
	    if ((max_iters && (niters >= max_iters)) || brokedown || nrm < tol) {
		s.inc_curbuf(krydim+1);
		break;
	    }

	    // If not converged, restart with vector buf(krydim)
	    s.swap_bufs(0, krydim);
	    oldbeta = beta;
	}
    }
};
