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

template<typename T>
class SubspaceView;

template<typename T>
concept SubspaceViewType = internal::BareTypeSpecializes<T, SubspaceView>;

// A SubspaceView of a SumOps is simply a SumOps with the spin chain length
// fixed. In other words, it's as if SumOps has been projected into the sub-
// space defined by the spin sites from 0 to len-1. We define this template
// class to improve the ergonomics of get_matrix and friends. In the future
// we might extend this class to encompass general subspaces of a contiguous
// range of spin sites (ie. those with non-zero starting spin site).
template<typename SumOpsTy>
class SubspaceView : public SumOpsTy {
    template<typename SumOpsTy1>
    friend class SubspaceView;

    // Spin chain length of the subspace
    int len;

    void ensure_equal_len(const SubspaceView &rhs) const {
	if (this->empty() || rhs.empty()) {
	    // If one of the view is empty, we don't check for spin chain length.
	    return;
	}
	if (len != rhs.len) {
	    ThrowException(InvalidArgument,
			   "spin chain length", "must be equal");
	}
    }

public:
    SubspaceView() : len{} {}
    template<typename SumOpsTy1>
    SubspaceView(int len, SumOpsTy1 &&ops)
	: SumOpsTy(std::forward<SumOpsTy1>(ops)), len(len) {}
    SubspaceView(const SubspaceView &) = default;
    SubspaceView(SubspaceView &&) = default;
    SubspaceView &operator=(const SubspaceView &) = default;
    SubspaceView &operator=(SubspaceView &&) = default;

    template<typename SumOpsTy1>
    SubspaceView(const SubspaceView<SumOpsTy1> &view)
	: SumOpsTy(view.ops), len(view.len) {}

    template<SubspaceViewType V>
    SubspaceView &operator=(V &&v) {
	SumOpsTy::operator=(std::forward<V>(v));
	len = v.len;
	return *this;
    }

    static auto identity(int len) {
	return SubspaceView<SumOpsTy>(len, SpinOp<typename SumOpsTy::RealScalarType>::identity());
    }

    using SumOpsTy::operator+=;
    using SumOpsTy::operator-=;
    using SumOpsTy::operator*=;
    using SumOpsTy::operator/=;
    using SumOpsTy::operator==;

    SubspaceView operator+(const SumOpsTy &rhs) const {
	return SubspaceView(len, SumOpsTy::operator+(rhs));
    }

    template<typename OpTy>
    friend SubspaceView operator+(OpTy &&lhs, const SubspaceView &rhs) {
	return SubspaceView{rhs.len,
	    std::forward<OpTy>(lhs) + static_cast<const SumOpsTy &>(rhs)};
    }

    SubspaceView operator+(const SubspaceView &rhs) const {
	ensure_equal_len(rhs);
	if (len) {
	    return *this + static_cast<const SumOpsTy &>(rhs);
	} else {
	    // If this view is empty, we return the other view;
	    assert(this->empty());
	    return rhs;
	}
    }

    SubspaceView &operator+=(const SubspaceView &rhs) {
	ensure_equal_len(rhs);
	if (len) {
	    SumOpsTy::operator+=(rhs);
	} else {
	    *this = rhs;
	}
	return *this;
    }

    SubspaceView operator-(const SumOpsTy &rhs) const {
	return SubspaceView(len, SumOpsTy::operator-(rhs));
    }

    template<typename OpTy>
    friend SubspaceView operator-(OpTy &&lhs, const SubspaceView &rhs) {
	return SubspaceView{rhs.len,
	    std::forward<OpTy>(lhs) - static_cast<const SumOpsTy &>(rhs)};
    }

    SubspaceView operator-(const SubspaceView &rhs) const {
	ensure_equal_len(rhs);
	if (len) {
	    return *this - static_cast<const SumOpsTy &>(rhs);
	} else {
	    return static_cast<const SumOpsTy &>(*this) - rhs;
	}
    }

    SubspaceView &operator-=(const SubspaceView &rhs) {
	ensure_equal_len(rhs);
	if (!len) {
	    len = rhs.len;
	}
	SumOpsTy::operator-=(rhs);
	return *this;
    }

    SubspaceView operator*(const SumOpsTy &rhs) const {
	return SubspaceView(len, SumOpsTy::operator*(rhs));
    }

    template<typename OpTy>
    friend SubspaceView operator*(OpTy &&lhs, const SubspaceView &rhs) {
	return SubspaceView{rhs.len,
	    std::forward<OpTy>(lhs) * static_cast<const SumOpsTy &>(rhs)};
    }

    SubspaceView operator*(const SubspaceView &rhs) const {
	ensure_equal_len(rhs);
	return static_cast<const SumOpsTy &>(*this) * rhs;
    }

    SubspaceView &operator*=(const SubspaceView &rhs) {
	ensure_equal_len(rhs);
	SumOpsTy::operator*=(rhs);
	return *this;
    }

    // Compute the tensor product of two subspace views.
    // rhs has higher spin indices, ie. we return (rhs \otimes *this).
    SubspaceView tensor(const SubspaceView &rhs) const {
	return SubspaceView{len+rhs.len, *this * (rhs << len)};
    }

    bool operator==(const SubspaceView &rhs) const {
	if (len != rhs.len) { return false; }
	return SumOpsTy::operator==(rhs);
    }

    int spin_chain_length() const { return len; }

    using MatrixType = typename SumOpsTy::MatrixType;

    const MatrixType &get_matrix() const {
	return SumOpsTy::get_matrix(spin_chain_length());
    }

    operator const MatrixType &() const {
	return get_matrix();
    }

    // SumOps doesn't define operator* with MatrixType (because it
    // doesn't store the spin chain length) so we must define it here
    MatrixType operator*(const MatrixType &psi) const {
	return get_matrix() * psi;
    }

    const MatrixType &matexp(typename SumOpsTy::RealScalarType c) const {
	return SumOpsTy::matexp(c, spin_chain_length());
    }

    const MatrixType &matexp(typename SumOpsTy::ComplexScalarType c) const {
	return SumOpsTy::matexp(c, spin_chain_length());
    }
};

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
class FermionOpNonBlockForm : public SubspaceView<typename Impl::template SumOps<FpType>> {
    using BaseType = SubspaceView<typename Impl::template SumOps<FpType>>;
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
    using ProductType = BaseType;

    // Construct the n-th Fermion field operator using the formula discussed above
    FermionOpNonBlockForm(IndexType N, IndexType n, bool standard_gamma = true)
	: BaseType(N/2, SpinOp<FpType>::identity()), N(N), n(n) {
	check_field_index(N, n);
	static_assert(std::is_same_v<ProductType,
		      std::remove_cvref_t<decltype(*this * *this)>>,
		      "Incorrect product type of fermion operators");
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

    using BaseType::operator+;
    using BaseType::operator-;
    using BaseType::operator*;
    using BaseType::operator+=;
    using BaseType::operator-=;
    using BaseType::operator*=;
    using BaseType::operator/=;
    using BaseType::operator==;
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
class FermionOpBlockForm
    : public BlockAntiDiag<SubspaceView<typename Impl::template SumOps<FpType>>> {
    using BaseType = SubspaceView<typename Impl::template SumOps<FpType>>;
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
    using ProductType = BlockDiag<BaseType>;

    // Construct the n-th Fermion field operator using the formula discussed above
    FermionOpBlockForm(IndexType N, IndexType n, bool standard_gamma = true) :
	BlockAntiDiag<BaseType>({N/2-1, SpinOp<FpType>::identity()},
				{N/2-1, SpinOp<FpType>::identity()}), N(N), n(n) {
	check_field_index(N, n);
	static_assert(std::is_same_v<ProductType,
		      std::remove_cvref_t<decltype(*this * *this)>>,
		      "Incorrect product type of fermion operators");
	if (n == (N-1)) {
	    this->LR *= complex<FpType>{0, -1};
	    this->RL *= complex<FpType>{0, 1};
	} else {
	    this->LR = FermionOpNonBlockForm(N-2, n, false);
	    this->RL = this->LR;
	}
	if (standard_gamma) {
	    *this /= sqrt(2.0);
	}
    }
};

#ifdef REAPERS_USE_PARITY_BLOCKS
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
using FermionOp = FermionOpBlockForm<FpType, Impl>;
#else
template<RealScalar FpType = DefFpType, typename Impl = DefImpl>
using FermionOp = FermionOpNonBlockForm<FpType, Impl>;
#endif
