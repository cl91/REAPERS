/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    fermions.h

Abstract:

    This header file contains definitions for data structures pertaining
    to fermionic field operators.

Revision History:

    2023-03-01  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

// This class represents the n-th Majorana fermion field operator b_n. Some common
// alternative notations for this is $\gamma_n$, $\phi_n$ or $\chi_n$. In what
// follows we will use b_n and gamma_n interchangeably. If standard_gamma is set to
// true in the constructor, these operators satisfy {b_i,b_j} = delta_ij. If on the
// otherhand, standard_gamma is set to false in the constructor, then these satisfy
// {b_i,b_j} = 2delta_ij.
//
// We generate the i-th Fermio field operators $b_i$ using the following recursive
// formula (here i/2 stands for floor(i/2))
//
//   N = 2:  b_0 <- \sigma_x(0),  b_1 <- \sigma_y(0)
//   N <- N+2:
//     b_i <- \sigma_x(N/2) . b_i, for 0 <= i < N.
//        Here b_i on the RHS is the b_i from the previous iteration.
//     b_N <- \sigma_x(N/2) . \sigma_z(N/2-1)
//     b_N+1 <- \sigma_y(N/2)
//  Finally, if standard_gamma is true, divide b_i by sqrt(2).
//
// Here \sigma_{x,y,z}(i) stands for the Pauli matrices on spin site i. Here spin
// site index starts from zero. Here dot (.) represents the tensor product, which
// is the same as a matrix product if one extends the single spin site operators
// to operators on the full Hilbert space.
//
// We can work out the first four cases (N = 2, 4, 6, 8). To simplify notations
// we use sx to stand for \sigma_x, etc. We also omit the spin index when it is
// apparent from the context (so sx.sy.sz represents sx(2).sy(1).sz(0)). We also
// use I(i) to stand for the identity matrix at site i, and omit the pre-factor
// 1/sqrt(2) for the standard gamma normalization.
//
//   N = 2, b_0 = sx(0),       b_1 = sy(0)
//   N = 4, b_0 = sx(1).sx(0), b_1 = sx(1).sy(0),
//          b_2 = sx(1).sz(0), b_3 = sy(1).I(0)
//   N = 6, b_0 = sx.sx.sx,    b_1 = sx.sx.sy,
//          b_2 = sx.sx.sz,    b_3 = sx.sy.I,
//          b_4 = sx.sz.I,     b_5 = sy.I.I
//   N = 8, b_0 = sx.sx.sx.sx, b_1 = sx.sx.sx.sy,
//          b_2 = sx.sx.sx.sz, b_3 = sx.sx.sy.I,
//          b_4 = sx.sx.sz.I,  b_5 = sx.sy.I.I,
//          b_6 = sx.sz.I.I,   b_7 = sy.I.I.I
//
// In other words, in non-recursive form the formula is
//
//   b_n = {sy or sz}((n-1)/2) . \prod_{(n+1)/2 <= i <= N/2-1} sx(i)
//     (divide by sqrt(2) if standard_gamma is true)
//
// Here {sy or sz} is sy when n is odd and sz when n is even. and is simply not
// present when n=0.
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
class FermionNonBlockOp : public Impl::template SumOps<FpType> {
    using IndexType = u8;
    IndexType N, n;

    static void check_field_index(IndexType N, IndexType n) {
	if (N < 2) {
	    DbgThrow(InvalidArgument, "N", "must be at least two");
	}
	if (N & 1) {
	    DbgThrow(InvalidArgument, "N", "must be even");
	}
	if (n > N) {
	    DbgThrow(FieldIndexTooLarge, n, N);
	}
    }

public:
    // Product of two fermion operators is another spin operator (a SumOps
    // with one term). We can probably have the compiler deduce this using
    // some decltype magic, but let's not do that.
    using ProductType = typename Impl::template SumOps<FpType>;

    // Construct the n-th Fermion field operator using the formula discussed above
    FermionNonBlockOp(IndexType N, IndexType n, bool standard_gamma = true)
	: Impl::template SumOps<FpType>(SpinOp<FpType>::identity()), N(N), n(n) {
	check_field_index(N, n);
	for (int i = (n+1)/2; i < N/2; i++) {
	    *this *= SpinOp<FpType>::sigma_x(i);
	}
	if (n != 0) {
	    *this *= (n & 1) ? SpinOp<FpType>::sigma_y((n-1)/2)
		: SpinOp<FpType>::sigma_z((n-1)/2);
	}
	// If standard_gamma is true, we normalize the Fermionic field
	// operators to {gamma_i, gamma_j} = delta_ij. Otherwise, the
	// fermion fields have normalization {gamma_i,gamma_j}=2delta_ij
	// and they preserve the norm of a state vector when acting on it.
	if (standard_gamma) {
	    *this /= sqrt(2.0);
	}
    }

    using Impl::template SumOps<FpType>::operator+;
    using Impl::template SumOps<FpType>::operator-;
    using Impl::template SumOps<FpType>::operator*;
    using Impl::template SumOps<FpType>::operator+=;
    using Impl::template SumOps<FpType>::operator-=;
    using Impl::template SumOps<FpType>::operator*=;
    using Impl::template SumOps<FpType>::operator/=;
    using Impl::template SumOps<FpType>::operator==;

    int spin_chain_length() const { return N/2; }

    using MatrixType = typename Impl::template SumOps<FpType>::MatrixType;
    operator MatrixType() const {
	return this->get_matrix(spin_chain_length());
    }

    // Impl::SumOps doesn't define operator* with MatrixType (because
    // it doesn't store the spin chain length) so we must define it here
    MatrixType operator*(const MatrixType &psi) const {
	return static_cast<MatrixType>(*this) * psi;
    }
};

// This class represents the n-th Majorana fermion field operator b_n in
// terms of parity blocks.
//
// Observe that since
//
// sx = ( 0  1 )    and   sy = ( 0  -i )
//      ( 1  0 )               ( i   0 )
//
// all b_i can be written in the block off-diagonal form
//
// b_i = (   0     bLR_i )
//       ( bRL_i     0   )
//
// where for 0 <= i <= N-3, the bLR_i and bRL_i matrices are identical to each
// other and are exactly the b_i matrix from the previous (N-2) iteration. For
// i == N-2, bLR and bRL are also identical to each other and are simply the
// single site operator sz(N/2-2) (divided by sqrt2 if standard_gamma is true),
// and for i == N-1, bLR is simply -i times the identity matrix (divided by
// sqrt2 if needed), and bRL is simply i times the identity matrix (div. sqrt2).
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
class FermionBlockOp : public BlockAntiDiag<typename Impl::template SumOps<FpType>> {
    using IndexType = u8;
    IndexType N, n;

    static void check_field_index(IndexType N, IndexType n) {
	if (N < 2) {
	    ThrowException(InvalidArgument, "N", "must be at least two");
	}
	if (N & 1) {
	    ThrowException(InvalidArgument, "N", "must be even");
	}
	if (n > N) {
	    ThrowException(FieldIndexTooLarge, n, N);
	}
    }

public:
    // Product of two fermion operators is block diagonal.
    using ProductType = BlockDiag<typename Impl::template SumOps<FpType>>;

    // Construct the n-th Fermion field operator using the formula discussed above
    FermionBlockOp(IndexType N, IndexType n, bool standard_gamma = true) :
	BlockAntiDiag<typename Impl::template SumOps<FpType>>(
	    SpinOp<FpType>::identity(), SpinOp<FpType>::identity()), N(N), n(n) {
	check_field_index(N, n);
	if (n == (N-1)) {
	    this->LR *= complex<FpType>{0, -1};
	    this->RL *= complex<FpType>{0, 1};
	} else {
	    for (int i = (n+1)/2; i < N/2-1; i++) {
		this->LR *= SpinOp<FpType>::sigma_x(i);
	    }
	    if (n != 0) {
		this->LR *= (n & 1) ? SpinOp<FpType>::sigma_y((n-1)/2)
		    : SpinOp<FpType>::sigma_z((n-1)/2);
	    }
	    this->RL = this->LR;
	}
	if (standard_gamma) {
	    *this /= sqrt(2.0);
	}
    }

    int spin_chain_length() const { return N/2-1; }

    using MatrixType = typename Impl::template SumOps<FpType>::MatrixType;
    operator BlockAntiDiag<MatrixType>() const {
	return this->get_matrix(spin_chain_length());
    }

    // Impl::SumOps doesn't define operator* with MatrixType (because
    // it doesn't store the spin chain length) so we must define it here
    template<template<typename> typename B>
    requires BlockForm<B<MatrixType>>
    auto operator*(const B<MatrixType> &psi) const {
	return static_cast<BlockAntiDiag<MatrixType>>(*this) * psi;
    }
};

#ifdef REAPERS_USE_PARITY_BLOCKS
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
using FermionOp = FermionBlockOp<FpType, Impl>;
#else
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
using FermionOp = FermionNonBlockOp<FpType, Impl>;
#endif
