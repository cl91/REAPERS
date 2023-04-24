/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    blkops.h

Abstract:

    This header file contains definitions for data structures pertaining
    to fermionic field operators in terms of parity blocks.

Revision History:

    2023-03-01  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

#ifndef REAPERS_USE_PARITY_BLOCKS
#error "You must define REAPERS_USE_PARITY_BLOCKS before including this file."
#endif

// This class represents a block diagonal matrix
//    B = ( bLL  0  )
//        ( 0   bRR )
template<typename T>
struct BlockDiag {
    T LL, RR;

    BlockDiag(T LL = {}, T RR = {}) : LL(LL), RR(RR) {}

    template<typename T1>
    BlockDiag(const BlockDiag<T1> &b) : LL(b.LL), RR(b.RR) {}

    BlockDiag &operator+=(const BlockDiag &rhs) {
	LL += rhs.LL;
	RR += rhs.RR;
	return *this;
    }
};

// This class represents a block anti-diagonal matrix
//    B = ( 0   bLR )
//        ( bRL  0  )
template<typename T>
struct BlockAntiDiag {
    T LR, RL;

    BlockAntiDiag(T LR = {}, T RL = {}) : LR(LR), RL(RL) {}

    template<typename T1>
    BlockAntiDiag(const BlockAntiDiag<T1> &b) : LR(b.LR), RL(b.RL) {}

    BlockAntiDiag &operator+=(const BlockAntiDiag &rhs) {
	LR += rhs.LR;
	RL += rhs.RL;
	return *this;
    }
};

template<typename T>
inline BlockDiag<T> operator+(const BlockDiag<T> &op0,
			      const BlockDiag<T> &op1) {
    return BlockDiag(op0.LL + op1.LL, op0.RR + op1.RR);
}

template<typename T>
inline BlockAntiDiag<T> operator+(const BlockAntiDiag<T> &op0,
				  const BlockAntiDiag<T> &op1) {
    return BlockAntiDiag(op0.LR + op1.LR, op0.RL + op1.RL);
}

template<typename T, typename FpType>
inline BlockDiag<T> operator*(FpType op0, const BlockDiag<T> &op1) {
    return BlockDiag(op0 * op1.LL, op0 * op1.RR);
}

template<typename T, typename FpType>
inline BlockDiag<T> operator*(complex<FpType> op0, const BlockDiag<T> &op1) {
    return BlockDiag(op0 * op1.LL, op0 * op1.RR);
}

template<typename T, typename FpType>
inline BlockDiag<T> operator*(const BlockDiag<T> &op0, complex<FpType> op1) {
    return BlockDiag(op0.LL * op1, op0.RR * op1);
}

template<typename T, typename FpType>
inline BlockAntiDiag<T> operator*(FpType op0, const BlockAntiDiag<T> &op1) {
    return BlockAntiDiag(op0 * op1.LR, op0 * op1.RL);
}

template<typename T, typename FpType>
inline BlockAntiDiag<T> operator*(complex<FpType> op0, const BlockAntiDiag<T> &op1) {
    return BlockAntiDiag(op0 * op1.LR, op0 * op1.RL);
}

template<typename T, typename FpType>
inline BlockAntiDiag<T> operator*(const BlockAntiDiag<T> &op0, complex<FpType> op1) {
    return BlockDiag(op0.LR * op1, op0.RL * op1);
}

template<typename T>
inline BlockDiag<T> operator*(const BlockDiag<T> &op0,
			      const BlockDiag<T> &op1) {
    return BlockDiag(op0.LL * op1.LL, op0.RR * op1.RR);
}

template<typename T>
inline BlockAntiDiag<T> operator*(const BlockDiag<T> &op0,
				  const BlockAntiDiag<T> &op1) {
    return BlockAntiDiag(op0.LL * op1.LR, op0.RR * op1.RL);
}

template<typename T>
inline BlockAntiDiag<T> operator*(const BlockAntiDiag<T> &op0,
				  const BlockDiag<T> &op1) {
    return BlockAntiDiag(op0.LR * op1.RR, op0.RL * op1.LL);
}

template<typename T>
inline BlockDiag<T> operator*(const BlockAntiDiag<T> &op0,
			      const BlockAntiDiag<T> &op1) {
    return BlockDiag(op0.LR * op1.RL, op0.RL * op1.LR);
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const BlockDiag<T> &op) {
    os << "BlockDiag<" << op.LL << ", " << op.RR << ">";
    return os;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const BlockAntiDiag<T> &op) {
    os << "BlockAntiDiag<" << op.LR << ", " << op.RL << ">";
    return os;
}

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
// b_i = ( 0       bLR_i )
//       ( bRL_i   0     )
//
// where for 0 <= i <= N-3, the bLR_i and bRL_i matrices are identical to each
// other and are exactly the b_i matrix from the previous (N-2) iteration. For
// i == N-2, bLR and bRL are also identical to each other and are simply the
// single site operator sz(N/2-2), and for i == N-1, bLR is simply -i times the
// identity matrix, and bRL is simply i times the identity matrix.
template<typename FpType>
class FermionOp : public BlockAntiDiag<SpinOp<FpType>> {
    using IndexType = u8;
    using ComplexScalar = typename SpinOp<FpType>::ComplexScalar;

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
    // Construct the n-th Fermion field operator using the formula discussed above
    FermionOp(IndexType N, IndexType n) :
	BlockAntiDiag<SpinOp<FpType>>(SpinOp<FpType>::identity(),
				      SpinOp<FpType>::identity()) {
	check_field_index(N, n);
	if (n == (N-1)) {
	    this->LR *= ComplexScalar{0, -1};
	    this->RL *= ComplexScalar{0, 1};
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
    }
};

// A Hamiltonian is just a sum of spin operators, and it's block diagonal.
template<typename FpType>
using HamOp = BlockDiag<SumOps<FpType>>;
