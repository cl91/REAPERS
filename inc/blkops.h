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

namespace internal {
    template<typename T>
    concept HasEval = requires(T t) { t.eval(); };

    template<HasEval T>
    auto eval(T &&t) -> decltype(t.eval());

    template<typename T>
    auto eval(T &&t) -> T;

    template<typename T>
    concept HasSize = requires(T t) {
	{ t.size() } -> std::convertible_to<bool>;
    };

    template<typename T>
    concept BareType = std::same_as<T, std::remove_cvref_t<T>>;

    // Some concept magic that checks if class type S (the implicit concept
    // argument) is indeed a specialization of T type (e.g. for T=BlockVec
    // and S=BlockVec<Args>, the concept is satisfied). Also accepts classes
    // deriving from specializations of T. Basically we check whether the
    // lambda function defined below accepts an object of type S.
    template<typename S, template<typename> typename T, typename ...Args>
    concept Specializes = requires (S s) {
	[]<typename ...TmplArgs>(T<Args..., TmplArgs...> &){}(s);
    };

    template<typename S, template<typename> typename T, typename ...Args>
    concept BareTypeSpecializes = Specializes<std::remove_cvref_t<S>, T, Args...>;

    // We need T to have a T::size() method with which we can check whether
    // the specified object is empty.
    template<typename T>
    concept BlockType = BareType<T> && HasSize<T>;
}

#define _REAPERS_evaltype(x)					\
    std::remove_cvref_t<decltype(internal::eval(x))>

// Forwards the cv-qualifier and reference type from the class variable t
// to its member a. In other words, if t is an rvalue reference we cast
// t.a to its rvalue reference, and if t is a const lvalue ref we cast
// t.a to its const lvalue ref, likewise for non-const lvalue ref. Note
// here T must be a universal (forwarding) reference (ie. template<T> T&&).
#define _REAPERS_forward(T, t, a)			\
    static_cast<decltype((std::forward<T>(t).a))>(t.a)

template<internal::BlockType T>
struct BlockDiag;

template<typename B>
concept BlockDiagType = internal::BareTypeSpecializes<B, BlockDiag>;

// This class represents a block diagonal matrix
//    B = ( bLL  0  )
//        ( 0   bRR )
// We define this as a struct so everything is exposed. In general
// you should have no need to modify the members directly and can
// simply use the overloaded arithmetic operators that act on both
// blocks. However, if you do modify LL or RR you need to make sure
// that nullLL and nullRR are modified accordingly. This applies to
// the other block matrix forms as well.
template<internal::BlockType T>
struct BlockDiag {
    // nullLL indicates that LL is a zero block, ie. LL.size() == false.
    // This allows us to skip the additions and multiplications below
    // if a block is zero. Likewise for the other Block classes below.
    bool nullLL, nullRR;
    T LL, RR;

    BlockDiag() : nullLL(true), nullRR(true), LL{}, RR{} {}
    BlockDiag(T B) : LL(B), RR(B) {
	nullLL = !B.size();
	nullRR = !B.size();
    }
    BlockDiag(T LL, T RR) : LL(LL), RR(RR) {
	nullLL = !LL.size();
	nullRR = !RR.size();
    }

    BlockDiag(const BlockDiag &) = default;
    BlockDiag(BlockDiag &&) = default;
    bool operator==(const BlockDiag &) const = default;

    template<BlockDiagType T1>
    explicit BlockDiag(T1 &&b) : LL(_REAPERS_forward(T1,b,LL)),
				 RR(_REAPERS_forward(T1,b,RR)),
				 nullLL(b.nullLL), nullRR(b.nullRR) {}

    template<BlockDiagType T1>
    BlockDiag &operator=(T1 &&b) {
	if (b.nullLL) { LL = T{}; } else {
	    LL = _REAPERS_forward(T1,b,LL);
	}
	if (b.nullRR) { RR = T{}; } else {
	    RR = _REAPERS_forward(T1,b,RR);
	}
	nullLL = b.nullLL;
	nullRR = b.nullRR;
	return *this;
    }

    template<BlockDiagType T1>
    BlockDiag &operator+=(T1 &&rhs) {
	if (!rhs.nullLL) {
	    if (!nullLL) {
		LL += rhs.LL;
	    } else {
		LL = _REAPERS_forward(T1,rhs,LL);
	    }
	    nullLL = false;
	}
	if (!rhs.nullRR) {
	    if (!nullRR) {
		RR += rhs.RR;
	    } else {
		RR = _REAPERS_forward(T1,rhs,RR);
	    }
	    nullRR = false;
	}
	return *this;
    }

    template<ScalarType S>
    BlockDiag &operator*=(S c) {
	if (c == S{}) { nullLL = nullRR = true; LL = T{}; RR = T{}; return *this; }
	if (!nullLL) { LL *= c; }
	if (!nullRR) { RR *= c;	}
	return *this;
    }

    // This member function would not be defined if T does not have a
    // T::get_matrix(int) member function (this is an example of SFINAE).
    // Likewise for matexp and BlockAntiDiag::get_matrix etc.
    template<typename...Args>
    auto get_matrix(Args&&...args) const {
	using Ty = _REAPERS_evaltype(LL.get_matrix(std::forward<Args>(args)...));
	BlockDiag<Ty> res(nullLL ? Ty{} : LL.get_matrix(std::forward<Args>(args)...),
			  nullRR ? Ty{} : RR.get_matrix(std::forward<Args>(args)...));
	res.nullLL = nullLL;
	res.nullRR = nullRR;
	return res;
    }

    template<typename...Args>
    auto matexp(Args&&...args) const {
	using Ty = _REAPERS_evaltype(LL.matexp(std::forward<Args>(args)...));
	BlockDiag<Ty> res(nullLL ? Ty{} : LL.matexp(std::forward<Args>(args)...),
			  nullRR ? Ty{} : RR.matexp(std::forward<Args>(args)...));
	res.nullLL = nullLL;
	res.nullRR = nullRR;
	return res;
    }

    template<typename...Args>
    auto trace(Args&&...args) const {
	using Ty = _REAPERS_evaltype(LL.trace(std::forward<Args>(args)...));
	return (nullLL ? Ty{} : LL.trace(std::forward<Args>(args)...))
	    + (nullRR ? Ty{} : RR.trace(std::forward<Args>(args)...));
    }

    template<typename...Args>
    static BlockDiag<T> identity(Args&&...args) {
	return T::identity(std::forward<Args>(args)...);
    }
};

template<internal::BlockType T>
struct BlockAntiDiag;

template<typename B>
concept BlockAntiDiagType = internal::BareTypeSpecializes<B, BlockAntiDiag>;

// This class represents a block anti-diagonal matrix
//    B = ( 0   bLR )
//        ( bRL  0  )
template<internal::BlockType T>
struct BlockAntiDiag {
    bool nullLR, nullRL;
    T LR, RL;

    BlockAntiDiag() : nullLR(true), nullRL(true), LR{}, RL{} {}
    BlockAntiDiag(T LR, T RL) : LR(LR), RL(RL) {
	nullLR = !LR.size();
	nullRL = !RL.size();
    }

    BlockAntiDiag(const BlockAntiDiag &) = default;
    BlockAntiDiag(BlockAntiDiag &&) = default;
    bool operator==(const BlockAntiDiag &) const = default;

    template<BlockAntiDiagType T1>
    explicit BlockAntiDiag(T1 &&b) : nullLR{b.nullLR}, nullRL{b.nullRL},
				     LR(_REAPERS_forward(T1,b,LR)),
				     RL(_REAPERS_forward(T1,b,RL)) {}

    template<BlockAntiDiagType T1>
    BlockAntiDiag &operator=(T1 &&b) {
	if (b.nullLR) { LR = T{}; } else {
	    LR = _REAPERS_forward(T1,b,LR);
	}
	if (b.nullRL) { RL = T{}; } else {
	    RL = _REAPERS_forward(T1,b,RL);
	}
	nullLR = b.nullLR;
	nullRL = b.nullRL;
	return *this;
    }

    template<BlockAntiDiagType T1>
    BlockAntiDiag &operator+=(T1 &&rhs) {
	if (!rhs.nullLR) {
	    if (!nullLR) {
		LR += rhs.LR;
	    } else {
		LR = _REAPERS_forward(T1,rhs,LR);
	    }
	    nullLR = false;
	}
	if (!rhs.nullRL) {
	    if (!nullRL) {
		RL += rhs.RL;
	    } else {
		RL = _REAPERS_forward(T1,rhs,RL);
	    }
	    nullRL = false;
	}
	return *this;
    }

    template<ScalarType S>
    BlockAntiDiag &operator*=(S c) {
	if (c == S{}) { nullLR = nullRL = true; LR = T{}; RL = T{}; return *this; }
	if (!nullLR) { LR *= c; }
	if (!nullRL) { RL *= c; }
	return *this;
    }

    template<typename...Args>
    auto get_matrix(Args&&...args) const {
	using Ty = _REAPERS_evaltype(LR.get_matrix(std::forward<Args>(args)...));
	BlockAntiDiag<Ty> res(nullLR ? Ty{} : LR.get_matrix(std::forward<Args>(args)...),
			      nullRL ? Ty{} : RL.get_matrix(std::forward<Args>(args)...));
	res.nullLR = nullLR;
	res.nullRL = nullRL;
	return res;
    }

    double trace() const { return 0.0; }
};

template<internal::BlockType T>
struct BlockOp;

template<typename B>
concept BlockOpType = internal::BareTypeSpecializes<B, BlockOp>;

// This class represents a general block matrix
//    B = (LL LR)
//        (RL RR)
template<internal::BlockType T>
struct BlockOp : BlockDiag<T>, BlockAntiDiag<T> {
    BlockOp(T LL = {}, T LR = {}, T RL = {}, T RR = {})
	: BlockDiag<T>(LL,RR), BlockAntiDiag<T>(LR,RL) {
	this->nullLL = !LL.size();
	this->nullRR = !RR.size();
	this->nullLR = !LR.size();
	this->nullRL = !RL.size();
    }

    BlockOp(const BlockOp &) = default;
    BlockOp(BlockOp &&) = default;
    bool operator==(const BlockOp &) const = default;

    template<BlockDiagType T1>
    explicit BlockOp(T1 &&b) : BlockDiag<T>(std::forward<T1>(b)) {}
    template<BlockAntiDiagType T1>
    explicit BlockOp(T1 &&b) : BlockAntiDiag<T>(std::forward<T1>(b)) {}
    template<BlockOpType T1>
    explicit BlockOp(T1 &&b) : BlockDiag<T>(std::forward<T1>(b)),
			       BlockAntiDiag<T>(std::forward<T1>(b)) {}

    template<BlockDiagType T1> requires (!BlockOpType<T1>)
    BlockOp &operator=(T1 &&b) {
	BlockDiag<T>::operator=(std::forward<T1>(b));
	BlockAntiDiag<T>::operator=({});
	return *this;
    }

    template<BlockAntiDiagType T1> requires (!BlockOpType<T1>)
    BlockOp &operator=(T1 &&b) {
	BlockDiag<T>::operator=({});
	BlockAntiDiag<T>::operator=(std::forward<T1>(b));
	return *this;
    }

    template<BlockOpType T1>
    BlockOp &operator=(T1 &&b) {
	BlockDiag<T>::operator=(std::forward<T1>(b));
	BlockAntiDiag<T>::operator=(std::forward<T1>(b));
	return *this;
    }

    template<BlockDiagType T1> requires (!BlockOpType<T1>)
    BlockOp &operator+=(T1 &&rhs) {
	BlockDiag<T>::operator+=(std::forward<T1>(rhs));
	BlockAntiDiag<T>::operator+=({});
	return *this;
    }

    template<BlockAntiDiagType T1> requires (!BlockOpType<T1>)
    BlockOp &operator+=(T1 &&rhs) {
	BlockDiag<T>::operator+=({});
	BlockAntiDiag<T>::operator+=(std::forward<T1>(rhs));
	return *this;
    }

    template<BlockOpType T1>
    BlockOp &operator+=(T1 &&rhs) {
	BlockDiag<T>::operator+=(std::forward<T1>(rhs));
	BlockAntiDiag<T>::operator+=(std::forward<T1>(rhs));
	return *this;
    }

    template<ScalarType S>
    BlockOp &operator*=(S c) {
	BlockDiag<T>::operator*=(c);
	BlockAntiDiag<T>::operator*=(c);
	return *this;
    }

    template<typename...Args>
    auto get_matrix(Args&&...args) const {
	using Ty = _REAPERS_evaltype(this->LL.get_matrix(std::forward<Args>(args)...));
	BlockOp<Ty> res(this->nullLL ? Ty{} : this->LL.get_matrix(std::forward<Args>(args)...),
			this->nullLR ? Ty{} : this->LR.get_matrix(std::forward<Args>(args)...),
			this->nullRL ? Ty{} : this->RL.get_matrix(std::forward<Args>(args)...),
			this->nullRR ? Ty{} : this->RR.get_matrix(std::forward<Args>(args)...));
	res.nullLL = this->nullLL;
	res.nullLR = this->nullLR;
	res.nullRL = this->nullRL;
	res.nullRR = this->nullRR;
	return res;
    }

    template<typename...Args>
    auto trace(Args&&...args) const {
	return BlockDiag<T>::trace(std::forward<Args>(args)...);
    }
};

// Define a concept which checks whether the class B is a specialization of
// one of the three block matrix forms.
template<typename B>
concept BlockMatrix = internal::Specializes<B, BlockDiag>
    || internal::Specializes<B, BlockAntiDiag>
    || internal::Specializes<B, BlockOp>;

template<internal::BlockType T>
struct BlockVec;

template<typename B>
concept BlockVecType = internal::BareTypeSpecializes<B, BlockVec>;

// This class represents a column vector in block form
//    V = (L)
//        (R)
template<internal::BlockType T>
struct BlockVec {
    bool nullL, nullR;
    T L, R;

    BlockVec(T L = {}, T R = {}) : nullL{!L.size()}, nullR{!R.size()}, L(L), R(R) {}

    BlockVec(const BlockVec &) = default;
    BlockVec(BlockVec &&) = default;
    bool operator==(const BlockVec &) const = default;

    template<BlockVecType T1>
    BlockVec(T1 &&b) : nullL(b.nullL), nullR(b.nullR),
		       L(_REAPERS_forward(T1,b,L)), R(_REAPERS_forward(T1,b,R)) {}

    template<BlockVecType T1>
    BlockVec &operator=(T1 &&b) {
	if (b.nullL) { L = T{}; } else {
	    L = _REAPERS_forward(T1,b,L);
	}
	if (b.nullR) { R = T{}; } else {
	    R = _REAPERS_forward(T1,b,R);
	}
	nullL = b.nullL;
	nullR = b.nullR;
	return *this;
    }

    template<BlockVecType T1>
    BlockVec &operator+=(T1 &&rhs) {
	if (!rhs.nullL) {
	    if (!nullL) {
		L += rhs.L;
	    } else {
		L = _REAPERS_forward(T1,rhs,L);
	    }
	    nullL = false;
	}
	if (!rhs.nullR) {
	    if (!nullR) {
		R += rhs.R;
	    } else {
		R = _REAPERS_forward(T1,rhs,R);
	    }
	    nullR = false;
	}
	return *this;
    }

    template<ScalarType S>
    BlockVec &operator*=(S c) {
	if (c == S{}) { nullL = nullR = true; L = T{}; R = T{}; }
	if (!nullL) { L *= c; }
	if (!nullR) { R *= c; }
	return *this;
    }

    template<BlockMatrix B>
    BlockVec &operator*=(B op) {
	*this = op * *this;
	return *this;
    }
};

// Define a concept which checks whether the class B is a specialization of
// one of the four block forms. This is needed when we define the
// operators below.
template<typename B>
concept BlockForm = BlockMatrix<B> || internal::Specializes<B, BlockVec>;

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator+(const BlockDiag<T0> &op0,
		      const BlockDiag<T1> &op1) {
    return BlockDiag(
	(op0.nullLL||op1.nullLL) ? (op0.nullLL?op1.LL:op0.LL) : (op0.LL + op1.LL),
	(op0.nullRR||op1.nullRR) ? (op0.nullRR?op1.RR:op0.RR) : (op0.RR + op1.RR));
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator-(const BlockDiag<T0> &op0,
		      const BlockDiag<T1> &op1) {
    BlockDiag<T1> op{op1};
    op *= -1;
    op += op0;
    return op;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator+(const BlockAntiDiag<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    return BlockAntiDiag(
	(op0.nullLR||op1.nullLR) ? (op0.nullLR?op1.LR:op0.LR) : (op0.LR + op1.LR),
	(op0.nullRL||op1.nullRL) ? (op0.nullRL?op1.RL:op0.RL) : (op0.RL + op1.RL));
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator-(const BlockAntiDiag<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    BlockAntiDiag<T1> op{op1};
    op *= -1;
    op += op0;
    return op;
}

template<internal::BlockType T>
inline auto operator+(const BlockOp<T> &op0,
		      const BlockOp<T> &op1) {
    BlockOp<T> op{op0};
    op += op1;
    return op;
}

template<internal::BlockType T>
inline auto operator-(const BlockOp<T> &op0,
		      const BlockOp<T> &op1) {
    BlockOp<T> op{op1};
    op *= -1;
    op += op0;
    return op;
}

template<internal::BlockType T>
inline auto operator+(const BlockVec<T> &op0,
		      const BlockVec<T> &op1) {
    BlockVec<T> op{op0};
    op += op1;
    return op;
}

template<internal::BlockType T>
inline auto operator-(const BlockVec<T> &op0,
		      const BlockVec<T> &op1) {
    BlockVec<T> op{op1};
    op *= -1;
    op += op0;
    return op;
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

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockDiag<T0> &op0,
		      const BlockDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL * op1.LL);
    BlockDiag<Ty> res((op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL,
		      (op0.nullRR || op1.nullRR) ? Ty{} : op0.RR*op1.RR);
    res.nullLL = op0.nullLL || op1.nullLL;
    res.nullRR = op0.nullRR || op1.nullRR;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockDiag<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL * op1.LR);
    BlockAntiDiag<Ty> res((op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR,
			  (op0.nullRR || op1.nullRL) ? Ty{} : op0.RR*op1.RL);
    res.nullLR = op0.nullLL || op1.nullLR;
    res.nullRL = op0.nullRR || op1.nullRL;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockAntiDiag<T0> &op0,
		      const BlockDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR * op1.RR);
    BlockAntiDiag<Ty> res((op0.nullLR || op1.nullRR) ? Ty{} : op0.LR*op1.RR,
			  (op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL);
    res.nullLR = op0.nullLR || op1.nullRR;
    res.nullRL = op0.nullRL || op1.nullLL;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockAntiDiag<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR * op1.RL);
    BlockDiag<Ty> res((op0.nullLR || op1.nullRL) ? Ty{} : op0.LR*op1.RL,
		      (op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR);
    res.nullLL = op0.nullLR || op1.nullRL;
    res.nullRR = op0.nullRL || op1.nullLR;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockOp<T0> &op0,
		      const BlockDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL * op1.LL);
    BlockOp<Ty> res((op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL,
		    (op0.nullLR || op1.nullRR) ? Ty{} : op0.LR*op1.RR,
		    (op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL,
		    (op0.nullRR || op1.nullRR) ? Ty{} : op0.RR*op1.RR);
    res.nullLL = op0.nullLL || op1.nullLL;
    res.nullLR = op0.nullLR || op1.nullRR;
    res.nullRL = op0.nullRL || op1.nullLL;
    res.nullRR = op0.nullRR || op1.nullRR;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockOp<T0> &op0,
		      const BlockAntiDiag<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR * op1.RL);
    BlockOp<Ty> res((op0.nullLR || op1.nullRL) ? Ty{} : op0.LR*op1.RL,
		    (op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR,
		    (op0.nullRR || op1.nullRL) ? Ty{} : op0.RR*op1.RL,
		    (op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR);
    res.nullLL = op0.nullLR || op1.nullRL;
    res.nullLR = op0.nullLL || op1.nullLR;
    res.nullRL = op0.nullRR || op1.nullRL;
    res.nullRR = op0.nullRL || op1.nullLR;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockDiag<T0> &op0,
		      const BlockOp<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL * op1.LL);
    BlockOp<Ty> res((op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL,
		    (op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR,
		    (op0.nullRR || op1.nullRL) ? Ty{} : op0.RR*op1.RL,
		    (op0.nullRR || op1.nullRR) ? Ty{} : op0.RR*op1.RR);
    res.nullLL = op0.nullLL || op1.nullLL;
    res.nullLR = op0.nullLL || op1.nullLR;
    res.nullRL = op0.nullRR || op1.nullRL;
    res.nullRR = op0.nullRR || op1.nullRR;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockAntiDiag<T0> &op0,
		      const BlockOp<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LR * op1.RL);
    BlockOp<Ty> res((op0.nullLR || op1.nullRL) ? Ty{} : op0.LR*op1.RL,
		    (op0.nullLR || op1.nullRR) ? Ty{} : op0.LR*op1.RR,
		    (op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL,
		    (op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR);
    res.nullLL = op0.nullLR || op1.nullRL;
    res.nullLR = op0.nullLR || op1.nullRR;
    res.nullRL = op0.nullRL || op1.nullLL;
    res.nullRR = op0.nullRL || op1.nullLR;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockOp<T0> &op0,
		      const BlockOp<T1> &op1) {
    using Ty = _REAPERS_evaltype(op0.LL * op1.LL);
    Ty LL = (op0.nullLL || op1.nullLL) ? Ty{} : op0.LL*op1.LL;
    if (!op0.nullLR && !op1.nullRL) {
	if (op0.nullLL || op1.nullLL) {
	    LL = op0.LR*op1.RL;
	} else {
	    LL += op0.LR*op1.RL;
	}
    }
    Ty LR = (op0.nullLL || op1.nullLR) ? Ty{} : op0.LL*op1.LR;
    if (!op0.nullLR && !op1.nullRR) {
	if (op0.nullLL || op1.nullLR) {
	    LR = op0.LR*op1.RR;
	} else {
	    LR += op0.LR*op1.RR;
	}
    }
    Ty RL = (op0.nullRL || op1.nullLL) ? Ty{} : op0.RL*op1.LL;
    if (!op0.nullRR && !op1.nullRL) {
	if (op0.nullRL || op1.nullLL) {
	    RL = op0.RR*op1.RL;
	} else {
	    RL += op0.RR*op1.RL;
	}
    }
    Ty RR = (op0.nullRL || op1.nullLR) ? Ty{} : op0.RL*op1.LR;
    if (!op0.nullRR && !op1.nullRR) {
	if (op0.nullRL || op1.nullLR) {
	    RR = op0.RR*op1.RR;
	} else {
	    RR += op0.RR*op1.RR;
	}
    }
    BlockOp res(LL, LR, RL, RR);
    res.nullLL = (op0.nullLL || op1.nullLL) && (op0.nullLR || op1.nullRL);
    res.nullLR = (op0.nullLL || op1.nullLR) && (op0.nullLR || op1.nullRR);
    res.nullRL = (op0.nullRL || op1.nullLL) && (op0.nullRR || op1.nullRL);
    res.nullRR = (op0.nullRL || op1.nullLR) && (op0.nullRR || op1.nullRR);
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockDiag<T0> &op, const BlockVec<T1> &b) {
    using Ty = _REAPERS_evaltype(op.LL * b.L);
    Ty L = (op.nullLL || b.nullL) ? Ty{} : op.LL * b.L;
    Ty R = (op.nullRR || b.nullR) ? Ty{} : op.RR * b.R;
    BlockVec res(L, R);
    res.nullL = op.nullLL || b.nullL;
    res.nullR = op.nullRR || b.nullR;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockAntiDiag<T0> &op, const BlockVec<T1> &b) {
    using Ty = _REAPERS_evaltype(op.LR * b.R);
    Ty L = (op.nullLR || b.nullR) ? Ty{} : op.LR * b.R;
    Ty R = (op.nullRL || b.nullL) ? Ty{} : op.RL * b.L;
    BlockVec res(L, R);
    res.nullL = op.nullLR || b.nullR;
    res.nullR = op.nullRL || b.nullL;
    return res;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockOp<T0> &op, const BlockVec<T1> &b) {
    return static_cast<const BlockDiag<T0> &>(op) * b
	+ static_cast<const BlockAntiDiag<T0> &>(op) * b;
}

template<internal::BlockType T0, internal::BlockType T1>
inline auto operator*(const BlockVec<T0> &a, const BlockVec<T1> &b) {
    using Ty = _REAPERS_evaltype(a.L * b.L);
    Ty resL = (a.nullL || b.nullL) ? Ty{} : a.L * b.L;
    Ty resR = (a.nullR || b.nullR) ? Ty{} : a.R * b.R;
    return resL + resR;
}

template<internal::BlockType T>
inline std::ostream &operator<<(std::ostream &os, const BlockDiag<T> &op) {
    os << "BlockDiag<" << op.LL << ", " << op.RR << ">";
    return os;
}

template<internal::BlockType T>
inline std::ostream &operator<<(std::ostream &os, const BlockAntiDiag<T> &op) {
    os << "BlockAntiDiag<" << op.LR << ", " << op.RL << ">";
    return os;
}

template<internal::BlockType T>
inline std::ostream &operator<<(std::ostream &os, const BlockOp<T> &op) {
    os << "BlockOp<" << op.LL << ", " << op.LR << ", "
       << op.RL << ", " << op.RR << ">";
    return os;
}

template<internal::BlockType T>
inline std::ostream &operator<<(std::ostream &os, const BlockVec<T> &b) {
    os << "BlockVec<" << b.L << ", " << b.R << ">";
    return os;
}
