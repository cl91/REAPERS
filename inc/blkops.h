/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    blkops.h

Abstract:

    This header file contains definitions for data structures pertaining
    to parity blocks.

Revision History:

    2023-03-01  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

template<typename T>
concept NonReferenceType = std::same_as<T, std::remove_reference_t<T>>;

namespace internal {
    template<typename T>
    concept HasEval = requires(T t) { t.eval(); };

    template<HasEval T>
    auto eval(T &&t) -> decltype(t.eval());

    template<typename T>
    auto eval(T &&t) -> T;
}

#define _REAPERS_evaltype(x)					\
    std::remove_reference_t<decltype(internal::eval(x))>

// This class represents a block diagonal matrix
//    B = ( bLL  0  )
//        ( 0   bRR )
template<NonReferenceType T>
struct BlockDiag {
    T LL, RR;
    // nullLL indicates that LL is a zero block, ie. LL == 0.
    // This allows us to skip the additions and multiplications below
    // if a block is zero. Likewise for nullRR, and the classes below..
    bool nullLL, nullRR;

    BlockDiag() : LL{}, RR{}, nullLL(true), nullRR(true) {}
    BlockDiag(T LL) : LL(LL), RR{}, nullLL(false), nullRR(true) {}
    BlockDiag(T LL, T RR) : LL(LL), RR(RR), nullLL(false), nullRR(false) {}

    template<typename T1>
    BlockDiag(const BlockDiag<T1> &b)
	: LL(b.LL), RR(b.RR), nullLL(b.nullLL), nullRR(b.nullRR) {}

    BlockDiag &operator=(const BlockDiag &b) {
	if (this == &b) { return *this; }
	if (b.nullLL) { LL = T{}; } else { LL = b.LL; }
	if (b.nullRR) { RR = T{}; } else { RR = b.RR; }
	nullLL = b.nullLL;
	nullRR = b.nullRR;
	return *this;
    }

    template<typename T1>
    BlockDiag &operator=(const BlockDiag<T1> &b) {
	if (b.nullLL) { LL = T{}; } else { LL = b.LL; }
	if (b.nullRR) { RR = T{}; } else { RR = b.RR; }
	nullLL = b.nullLL;
	nullRR = b.nullRR;
	return *this;
    }

    BlockDiag &operator+=(const BlockDiag &rhs) {
	if (!rhs.nullLL) {
	    LL += rhs.LL;
	    nullLL = false;
	}
	if (!rhs.nullRR) {
	    RR += rhs.RR;
	    nullRR = false;
	}
	return *this;
    }

    template<ScalarType S>
    BlockDiag &operator*=(S c) {
	if (!nullLL) {
	    LL *= c;
	}
	if (!nullRR) {
	    RR *= c;
	}
	if (c == S{}) {
	    nullLL = nullRR = true;
	}
	return *this;
    }

    // This member function would not be defined if T does not have a
    // T::get_matrix(int) member function (this is an example of SFINAE).
    // Likewise for matexp and BlockAntiDiag::get_matrix etc.
    auto get_matrix(int len) const {
	using Ty = _REAPERS_evaltype(LL.get_matrix(len));
	BlockDiag<Ty> res(
	    nullLL ? Ty{} : LL.get_matrix(len),
	    nullRR ? Ty{} : RR.get_matrix(len));
	res.nullLL = nullLL;
	res.nullRR = nullRR;
	return res;
    }

    template<RealScalar Fp>
    auto matexp(complex<Fp> c, int len) const {
	using Ty = _REAPERS_evaltype(LL.matexp(c,len));
	BlockDiag<Ty> res(
	    nullLL ? Ty{} : LL.matexp(c,len),
	    nullRR ? Ty{} : RR.matexp(c,len));
	res.nullLL = nullLL;
	res.nullRR = nullRR;
	return res;
    }

    auto trace() const {
	using Ty = _REAPERS_evaltype(LL.trace());
	return (nullLL ? Ty{} : LL.trace()) + (nullRR ? Ty{} : RR.trace());
    }
};

// This class represents a block anti-diagonal matrix
//    B = ( 0   bLR )
//        ( bRL  0  )
template<NonReferenceType T>
struct BlockAntiDiag {
    T LR, RL;
    bool nullLR, nullRL;

    BlockAntiDiag() : LR{}, RL{}, nullLR(true), nullRL(true) {}
    BlockAntiDiag(T LR) : LR(LR), RL{}, nullLR(false), nullRL(true) {}
    BlockAntiDiag(T LR, T RL) : LR(LR), RL(RL), nullLR(false), nullRL(false) {}

    template<typename T1>
    BlockAntiDiag(const BlockAntiDiag<T1> &b)
	: LR(b.LR), RL(b.RL), nullLR{b.nullLR}, nullRL{b.nullRL} {}

    BlockAntiDiag &operator=(const BlockAntiDiag &b) {
	if (this == &b) { return *this; }
	if (b.nullLR) { LR = T{}; } else { LR = b.LR; }
	if (b.nullRL) { RL = T{}; } else { RL = b.RL; }
	nullLR = b.nullLR;
	nullRL = b.nullRL;
	return *this;
    }

    template<typename T1>
    BlockAntiDiag &operator=(const BlockAntiDiag<T1> &b) {
	if (b.nullLR) { LR = T{}; } else { LR = b.LR; }
	if (b.nullRL) { RL = T{}; } else { RL = b.RL; }
	nullLR = b.nullLR;
	nullRL = b.nullRL;
	return *this;
    }

    BlockAntiDiag &operator+=(const BlockAntiDiag &rhs) {
	if (!rhs.nullLR) {
	    LR += rhs.LR;
	    nullLR = false;
	}
	if (!rhs.nullRL) {
	    RL += rhs.RL;
	    nullRL = false;
	}
	return *this;
    }

    template<ScalarType S>
    BlockAntiDiag &operator*=(S c) {
	if (!nullLR) {
	    LR *= c;
	}
	if (!nullRL) {
	    RL *= c;
	}
	if (c == S{}) {
	    nullLR = nullRL = true;
	}
	return *this;
    }

    auto get_matrix(int len) const {
	using Ty = _REAPERS_evaltype(LR.get_matrix(len));
	BlockAntiDiag<Ty> res(
	    nullLR ? Ty{} : LR.get_matrix(len),
	    nullRL ? Ty{} : RL.get_matrix(len));
	res.nullLR = nullLR;
	res.nullRL = nullRL;
	return res;
    }

    double trace() const { return 0.0; }
};

// This class represents a general block matrix
//    B = (LL LR)
//        (RL RR)
template<NonReferenceType T>
struct BlockOp : BlockDiag<T>, BlockAntiDiag<T> {
    BlockOp(T LL = {}, T LR = {}, T RL = {}, T RR = {})
	: BlockDiag<T>(LL,RR), BlockAntiDiag<T>(LR,RL) {}

    template<typename T1>
    explicit BlockOp(const BlockDiag<T1> &b) : BlockDiag<T>(b) {}
    template<typename T1>
    explicit BlockOp(const BlockAntiDiag<T1> &b) : BlockAntiDiag<T>(b) {}
    template<typename T1>
    BlockOp(const BlockOp<T1> &b) : BlockDiag<T>(b), BlockAntiDiag<T>(b) {}

    template<typename T1>
    BlockOp &operator=(const BlockDiag<T1> &b) {
	BlockDiag<T>::operator=(b);
	BlockAntiDiag<T>::operator=({});
	return *this;
    }

    template<typename T1>
    BlockOp &operator=(const BlockAntiDiag<T1> &b) {
	BlockDiag<T>::operator=({});
	BlockAntiDiag<T>::operator=(b);
	return *this;
    }

    template<typename T1>
    BlockOp &operator=(const BlockOp<T1> &b) {
	BlockDiag<T>::operator=(b);
	BlockAntiDiag<T>::operator=(b);
	return *this;
    }

    template<typename T1>
    BlockOp &operator+=(const BlockDiag<T1> &rhs) {
	BlockDiag<T>::operator+=(rhs);
	BlockAntiDiag<T>::operator+=({});
	return *this;
    }

    template<typename T1>
    BlockOp &operator+=(const BlockAntiDiag<T1> &rhs) {
	BlockDiag<T>::operator+=({});
	BlockAntiDiag<T>::operator+=(rhs);
	return *this;
    }

    template<typename T1>
    BlockOp &operator+=(const BlockOp<T1> &rhs) {
	BlockDiag<T>::operator+=(rhs);
	BlockAntiDiag<T>::operator+=(rhs);
	return *this;
    }

    template<ScalarType S>
    BlockOp &operator*=(S c) {
	BlockDiag<T>::operator*=(c);
	BlockAntiDiag<T>::operator*=(c);
	return *this;
    }

    auto get_matrix(int len) const {
	using Ty = _REAPERS_evaltype(this->LL.get_matrix(len));
	BlockOp<Ty> res(
	    this->nullLL ? Ty{} : this->LL.get_matrix(len),
	    this->nullLR ? Ty{} : this->LR.get_matrix(len),
	    this->nullRL ? Ty{} : this->RL.get_matrix(len),
	    this->nullRR ? Ty{} : this->RR.get_matrix(len));
	res.nullLL = this->nullLL;
	res.nullLR = this->nullLR;
	res.nullRL = this->nullRL;
	res.nullRR = this->nullRR;
	return res;
    }

    auto trace() const { return BlockDiag<T>::trace(); }
};

// This class represents a column vector in block form
//    V = (L)
//        (R)
template<NonReferenceType T>
struct BlockVec {
    T L, R;

    BlockVec(T L = {}, T R = {}) : L(L), R(R) {}

    template<typename T1>
    BlockVec(const BlockVec<T1> &b) : L(b.L), R(b.R) {}

    template<typename T1>
    BlockVec &operator=(const BlockVec<T1> &b) {
	L = b.L;
	R = b.R;
	return *this;
    }

    template<typename T1>
    BlockVec &operator+=(const BlockVec<T1> &rhs) {
	L += rhs.L;
	R += rhs.R;
	return *this;
    }

    template<ScalarType S>
    BlockVec &operator*=(S c) {
	L *= c;
	R *= c;
	return *this;
    }
};

// Some concept magic that checks if class type Specialization (the implicit
// concept argument) is indeed a specialization of TemplateClass type (e.g.
// satisfied for TemplateClass=BlockVec and Specialization=BlockVec<A>).
// Also accepts classes deriving from specialized TemplateClass.
template<class Specialization, template<typename> class TemplateClass,
         typename ...PartialSpecialization>
concept Specializes = requires (Specialization s) {
    []<typename ...TemplateArgs>(
        TemplateClass<PartialSpecialization..., TemplateArgs...>&){}(s);
};

// Define a concept which checks whether the class B is a specialization of
// one of the four block matrix forms. This is needed when we define the
// operators below.
template<typename B>
concept BlockForm = Specializes<B, BlockDiag> || Specializes<B, BlockAntiDiag>
    || Specializes<B, BlockOp> || Specializes<B, BlockVec>;

template<typename T0, typename T1>
inline auto operator+(const BlockDiag<T0> &op0,
		      const BlockDiag<T1> &op1) {
    return BlockDiag(
	(op0.nullLL||op1.nullLL) ? (op0.nullLL?op1.LL:op0.LL) : (op0.LL + op1.LL),
	(op0.nullRR||op1.nullRR) ? (op0.nullRR?op1.RR:op0.RR) : (op0.RR + op1.RR));
}

template<typename T0, typename T1>
inline auto operator+(const BlockAntiDiag<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    return BlockAntiDiag(
	(op0.nullLR||op1.nullLR) ? (op0.nullLR?op1.LR:op0.LR) : (op0.LR + op1.LR),
	(op0.nullRL||op1.nullRL) ? (op0.nullRL?op1.RL:op0.RL) : (op0.RL + op1.RL));
}

// Scalar multiplication is commutative and can be done from both sides.
// Note we must restrict B to be one of the four template classes above.
// Otherwise this produces an ambiguous overloaded operator for std::complex.
template<ScalarType S, BlockForm B>
inline auto operator*(S op0, const B &op1) {
    B res(op1);
    res *= op0;
    return res;
}

template<ScalarType S, BlockForm B>
inline auto operator*(const B &op0, S op1) {
    return op1 * op0;
}

template<ScalarType S, BlockForm B>
inline auto operator/=(B &op, S c) {
    return op *= S{1} / c;
}

template<typename T0, typename T1>
inline auto operator*(const BlockDiag<T0> &op0,
		      const BlockDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL*op1.LL);
    BlockDiag<Ty> res(
	(op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL,
	(op0.nullRR || op1.nullRR) ? Ty{} : op0.RR*op1.RR);
    res.nullLL = op0.nullLL || op1.nullLL;
    res.nullRR = op0.nullRR || op1.nullRR;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockDiag<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL*op1.LR);
    BlockAntiDiag<Ty> res(
	(op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR,
	(op0.nullRR || op1.nullRL) ? Ty{} : op0.RR*op1.RL);
    res.nullLR = op0.nullLL || op1.nullLR;
    res.nullRL = op0.nullRR || op1.nullRL;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockAntiDiag<T0> &op0,
		      const BlockDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR*op1.RR);
    BlockAntiDiag<Ty> res(
	(op0.nullLR || op1.nullRR) ? Ty{} : op0.LR*op1.RR,
	(op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL);
    res.nullLR = op0.nullLR || op1.nullRR;
    res.nullRL = op0.nullRL || op1.nullLL;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockAntiDiag<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR*op1.RL);
    BlockDiag<Ty> res(
	(op0.nullLR || op1.nullRL) ? Ty{} : op0.LR*op1.RL,
	(op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR);
    res.nullLL = op0.nullLR || op1.nullRL;
    res.nullRR = op0.nullRL || op1.nullLR;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockOp<T0> &op0,
		      const BlockDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL*op1.LL);
    BlockOp<Ty> res(
	(op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL,
	(op0.nullLR || op1.nullRR) ? Ty{} : op0.LR*op1.RR,
	(op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL,
	(op0.nullRR || op1.nullRR) ? Ty{} : op0.RR*op1.RR);
    res.nullLL = op0.nullLL || op1.nullLL;
    res.nullLR = op0.nullLR || op1.nullRR;
    res.nullRL = op0.nullRL || op1.nullLL;
    res.nullRR = op0.nullRR || op1.nullRR;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockOp<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR*op1.RL);
    BlockOp<Ty> res(
	(op0.nullLR || op1.nullRL) ? Ty{} : op0.LR*op1.RL,
	(op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR,
	(op0.nullRR || op1.nullRL) ? Ty{} : op0.RR*op1.RL,
	(op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR);
    res.nullLL = op0.nullLR || op1.nullRL;
    res.nullLR = op0.nullLL || op1.nullLR;
    res.nullRL = op0.nullRR || op1.nullRL;
    res.nullRR = op0.nullRL || op1.nullLR;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockDiag<T0> &op0,
		      const BlockOp<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL*op1.LL);
    BlockOp<Ty> res(
	(op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL,
	(op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR,
	(op0.nullRR || op1.nullRL) ? Ty{} : op0.RR*op1.RL,
	(op0.nullRR || op1.nullRR) ? Ty{} : op0.RR*op1.RR);
    res.nullLL = op0.nullLL || op1.nullLL;
    res.nullLR = op0.nullLL || op1.nullLR;
    res.nullRL = op0.nullRR || op1.nullRL;
    res.nullRR = op0.nullRR || op1.nullRR;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockAntiDiag<T0> &op0,
		      const BlockOp<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR*op1.RL);
    BlockOp<Ty> res(
	(op0.nullLR || op1.nullRL) ? Ty{} : op0.LR*op1.RL,
	(op0.nullLR || op1.nullRR) ? Ty{} : op0.LR*op1.RR,
	(op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL,
	(op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR);
    res.nullLL = op0.nullLR || op1.nullRL;
    res.nullLR = op0.nullLR || op1.nullRR;
    res.nullRL = op0.nullRL || op1.nullLL;
    res.nullRR = op0.nullRL || op1.nullLR;
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockOp<T0> &op0,
		      const BlockOp<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL*op1.LL);
    Ty LL = (op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL;
    if (!op0.nullLR && !op1.nullRL) {
	LL += op0.LR*op1.RL;
    }
    Ty LR = (op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR;
    if (!op0.nullLR && !op1.nullRR) {
	LR += op0.LR*op1.RR;
    }
    Ty RL = (op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL;
    if (!op0.nullRR && !op1.nullRL) {
	LR += op0.RR*op1.RL;
    }
    Ty RR = (op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR;
    if (!op0.nullRR && !op1.nullRR) {
	LR += op0.RR*op1.RR;
    }
    BlockOp res(LL, LR, RL, RR);
    res.nullLL = (op0.nullLL || op1.nullLL) && (op0.nullLR || op1.nullRL);
    res.nullLR = (op0.nullLL || op1.nullLR) && (op0.nullLR || op1.nullRR);
    res.nullRL = (op0.nullRL || op1.nullLL) && (op0.nullRR || op1.nullRL);
    res.nullRR = (op0.nullRL || op1.nullLR) && (op0.nullRR || op1.nullRR);
    return res;
}

template<typename T0, typename T1>
inline auto operator*(const BlockDiag<T0> &op, const BlockVec<T1> &b) {
    return BlockVec(op.LL * b.L, op.RR * b.R);
}

template<typename T0, typename T1>
inline auto operator*(const BlockAntiDiag<T0> &op, const BlockVec<T1> &b) {
    return BlockVec(op.LR * b.R, op.RL * b.L);
}

template<typename T0, typename T1>
inline auto operator*(const BlockOp<T0> &op, const BlockVec<T1> &b) {
    // Our State<> is not default construtible so we will need at least
    // one non-zero block for this operation to work. In the future we
    // might want to fix this as this is too restrictive.
    if (op.nullLL) {
	assert(!op.nullLR);
    }
    if (op.nullRR) {
	assert(!op.nullRL);
    }
    auto L = op.nullLL ? op.LR*b.R : op.LL*b.L;
    if (!op.nullLL && !op.nullLR) {
	// In the case where nullLL is true we already added op.LR*b.R
	// so we shouldn't do it in that case.
	L += op.LR*b.R;
    }
    auto R = op.nullRL ? op.RR*b.R : op.RL*b.L;
    if (!op.nullRL && !op.nullRR) {
	R += op.RR*b.R;
    }
    return BlockVec(L, R);
}

template<typename T0, typename T1>
inline auto operator*(const BlockVec<T0> &a, const BlockVec<T1> &b) {
    return a.L * b.L + a.R * b.R;
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

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const BlockOp<T> &op) {
    os << "BlockOp<" << op.LL << ", " << op.LR << ", "
       << op.RL << ", " << op.RR << ">";
    return os;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const BlockVec<T> &b) {
    os << "BlockVec<" << b.L << ", " << b.R << ">";
    return os;
}
