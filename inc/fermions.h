/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    fermions.h

Abstract:

    This header file contains definitions for data structures pertaining
    to fermionic field operators without splitting them into parity blocks.

Revision History:

    2023-03-01  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

#ifdef REAPERS_USE_PARITY_BLOCKS
#error "You must not define REAPERS_USE_PARITY_BLOCKS before including this file."
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
template<typename FpType>
class FermionOp : public SpinOp<FpType> {
    using IndexType = u8;

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
    // Construct the n-th Fermion field operator using the formula discussed above
    FermionOp(IndexType N, IndexType n, bool standard_gamma = true)
	: SpinOp<FpType>(SpinOp<FpType>::identity()) {
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
};

template<typename FpType>
using HamOp = SumOps<FpType>;
