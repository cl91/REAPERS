/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    ops.h

Abstract:

    This header file contains definitions for data structures pertaining
    to the spin operators and the fermionic field operators.

Revision History:

    2023-01-03  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#endif

// We can only have at most 32 spin sites (in other words, for the single SYK model
// N is at most 64). If you want 64 spin sites, you need to change SpinOp::ReprType
// and State::IndexType. In practice you won't be able to go above N=66 unless you
// have several (many) very powerful graphics cards.
#ifdef REAPERS_SPIN_SITES
#if REAPERS_SPIN_SITES > 32
#error "You can only specify at most 32 spin sites"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif
#define NUM_SPIN_SITES		(REAPERS_SPIN_SITES)
#else
#define NUM_SPIN_SITES		(32)
#endif
#define MAX_SPIN_INDEX		(NUM_SPIN_SITES-1)
#define NUM_FERMION_FIELDS	(2*NUM_SPIN_SITES)
#define MAX_FIELD_INDEX		(NUM_FERMION_FIELDS-1)

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr complex<DefFpType> operator""_i(unsigned long long d)
{
    return complex<DefFpType>{0.0, static_cast<DefFpType>(d)};
}

constexpr complex<DefFpType> operator""_i(long double d)
{
    return complex<DefFpType>{0.0, static_cast<DefFpType>(d)};
}

constexpr complex<float> operator""_if(unsigned long long d)
{
    return complex<float>{0.0, static_cast<float>(d)};
}

constexpr complex<float> operator""_if(long double d)
{
    return complex<float>{0.0, static_cast<float>(d)};
}

constexpr complex<double> operator""_id(unsigned long long d)
{
    return complex<double>{0.0, static_cast<double>(d)};
}

constexpr complex<double> operator""_id(long double d)
{
    return complex<double>{0.0, static_cast<double>(d)};
}

template<typename FpType>
constexpr inline bool operator==(const complex<FpType> &lhs, long double rhs) {
    return lhs == static_cast<FpType>(rhs);
}

template<typename FpType>
constexpr inline bool operator==(long double lhs, const complex<FpType> &rhs) {
    return static_cast<FpType>(lhs) == rhs;
}

constexpr inline bool operator==(const complex<float> &lhs,
				 const complex<double> &rhs) {
    return static_cast<complex<double>>(lhs) == rhs;
}

constexpr inline bool operator==(const complex<double> &lhs,
				 const complex<float> &rhs) {
    return lhs == static_cast<complex<double>>(rhs);
}

template<typename FpType>
constexpr inline bool operator!=(const complex<FpType> &lhs, long double rhs) {
    return lhs != static_cast<FpType>(rhs);
}

template<typename FpType>
constexpr inline bool operator!=(long double lhs, const complex<FpType> &rhs) {
    return static_cast<FpType>(lhs) != rhs;
}

constexpr inline bool operator!=(const complex<float> &lhs,
				 const complex<double> &rhs) {
    return static_cast<complex<double>>(lhs) != rhs;
}

constexpr inline bool operator!=(const complex<double> &lhs,
				 const complex<float> &rhs) {
    return lhs != static_cast<complex<double>>(rhs);
}

// This class represents a spin operator.
template<typename FpType = DefFpType>
class SpinOp {
public:
    // Integer type for the spin site index, eg. for spin state |011001>, there are
    // 6 spin sites, and spin site index goes between 0 and 5.
    using IndexType = u8;

    // Integer type for a spin state, ie. the bit-string 011001 in the state |011001>.
    using StateType = u32;

    // Integer type for the internal representation of a spin operator. We use two
    // bits for each spin site, so this is always twice as wide as StateType.
    using ReprType = u64;

    using ComplexScalar = complex<FpType>;

private:
    static constexpr u8 SPIN_SHIFT = 2;
    static constexpr ReprType SPIN_MASK = 0b11;

    // sx = ( 0 1 )   sy = ( 0 -1 )   sz = ( 1  0 )
    // 01   ( 1 0 )   11   ( 1  0 )   10   ( 0 -1 )
    enum class Sigma : u8 {
	id = 0b00,
	x = 0b01,
	y = 0b11,
	z = 0b10
    };

    // Compute the "carry-over" minus sign that results from multiplying two sigma
    // matrices. There are only four cases here:
    //
    //   sz.sx = -sy,  sy.sx = -sz,  sy.sy = -id,  sz.sy = -sx
    //
    // When we multiply two spin operators we must take care of these minus signs.
    constexpr static bool prod_has_minus(Sigma op0, Sigma op1) {
	switch ((u8(op0) << SPIN_SHIFT) | u8(op1)) {
	case 0b1001:	// sz . sx
	case 0b1011:	// sz . sy
	case 0b1101:	// sy . sx
	case 0b1111:	// sy . sy
	    return true;
	default:
	    return false;
	}
    }

    ComplexScalar coeff;
    ReprType bits;

    template<typename FpType1>
    friend class SpinOp;

    // This is needed so HostSumOps and DevSumOps can access the coeff and bits
    // members in operator+=().
    template<typename FpType1>
    friend class HostSumOps;
#ifndef REAPERS_NOGPU
    template<typename FpType1>
    friend class DevSumOps;
#endif

    // This is marked as private as the end user should use the static members
    // functions below to construct spin operators.
    SpinOp(ComplexScalar coeff, ReprType bits) : coeff(coeff), bits(bits) {}

    friend std::ostream &operator<<(std::ostream &os, const Sigma &s) {
	switch (s) {
	case Sigma::id:
	    os << "id";
	    break;
	case Sigma::x:
	    os << "sx";
	    break;
	case Sigma::y:
	    os << "sy";
	    break;
	case Sigma::z:
	    os << "sz";
	    break;
	}
	return os;
    }

    static void check_spin_index(IndexType n) {
	if (n > MAX_SPIN_INDEX) {
	    DbgThrow(SpinIndexTooLarge, n, MAX_SPIN_INDEX);
	}
    }

    int count_sy() const {
	int sy_count = 0;
	for (auto [s, _] : *this) {
	    if (s == Sigma::y) {
		sy_count++;
	    }
	}
	return sy_count & 3;
    }

    // (Const) iterator class for SpinOp. We do not allow the user to modify the
    // spin operator using an iterator so all iterators are const iterators.
    class Iterator {
	int idx;
	const SpinOp *op;

	Iterator(const SpinOp *op, int n = -1) : idx(n), op(op) {
	    if (idx >= 0) {
		// If the current spin site is an identity operator,
		// find the next non-identity sigma matrix.
		while (idx <= MAX_SPIN_INDEX && (*op)[idx] == Sigma::id) {
		    idx++;
		}
		// If we have advanced past the last non-identity operator,
		// mark the iterator as finished.
		if (idx > MAX_SPIN_INDEX) {
		    idx = -1;
		}
	    }
	}

	friend class SpinOp;

    public:
	std::tuple<Sigma, IndexType> operator*() const {
	    return { (*op)[idx], IndexType(idx) };
	}

	Iterator &operator++() {
	    if (idx >= 0) {
		// Find the next non-identity sigma matrix.
		do {
		    idx++;
		} while (idx <= MAX_SPIN_INDEX && (*op)[idx] == Sigma::id);
		// If we have advanced past the last non-identity operator,
		// mark the iterator as finished.
		if (idx > MAX_SPIN_INDEX) {
		    idx = -1;
		}
	    }
	    return *this;
	}

	bool operator==(const Iterator &other) const {
	    return op == other.op && idx == other.idx;
	}

	bool operator!=(const Iterator &other) const {
	    return !operator==(other);
	}
    };

public:
    template<typename FpType1>
    explicit SpinOp(const SpinOp<FpType1> &op) : coeff(op.coeff), bits(op.bits) {}

    static SpinOp zero() {
	return SpinOp(0, 0);
    }

    static SpinOp identity() {
	return SpinOp(1, 0);
    }

    static SpinOp sigma_x(IndexType n) {
	check_spin_index(n);
	return SpinOp(1, ReprType(Sigma::x) << (n*2));
    }

    static SpinOp sigma_y(IndexType n) {
	check_spin_index(n);
	return SpinOp(1_i, ReprType(Sigma::y) << (n*2));
    }

    static SpinOp sigma_z(IndexType n) {
	check_spin_index(n);
	return SpinOp(1, ReprType(Sigma::z) << (n*2));
    }

    SpinOp &operator*=(ComplexScalar s) {
	coeff *= s;
	return *this;
    }

    SpinOp &operator/=(ComplexScalar s) {
	coeff *= ComplexScalar{1.0,0.0}/s;
	return *this;
    }

    // In the case of operator *= the RHS always right-multiplies the LHS.
    SpinOp &operator*=(const SpinOp &rhs) {
	*this = *this * rhs;
	return *this;
    }

    friend SpinOp operator*(ComplexScalar s, SpinOp op) {
	return SpinOp(op.coeff * s, op.bits);
    }

    SpinOp operator*(ComplexScalar s) {
	return SpinOp(coeff * s, bits);
    }

    SpinOp operator*(const SpinOp &op1) const {
	bool minus = false;
	for (auto [s, idx] : *this) {
	    if (prod_has_minus(s, op1[idx])) {
		minus = !minus;
	    }
	}
	return SpinOp((minus ? -coeff : coeff) * op1.coeff, bits ^ op1.bits);
    }

    // Returns an iterator at the zeroth spin site
    Iterator begin() const { return Iterator(this, 0); }

    // Returns an iterator past the end of spin chain
    Iterator end() const { return Iterator(this); }

    Sigma operator[](IndexType n) const {
	check_spin_index(n);
	return Sigma { u8((bits >> (SPIN_SHIFT*n)) & SPIN_MASK) };
    }

    // Returns the coefficient of the spin operator. In other words, for spin operator
    // S = c sigma_x(0).sigma_y(1).sigma_x(2)...sigma_x(n) we return c. Note here we
    // use the standard definition of the sigma_y matrix, which includes the i, ie.
    //
    //   sigma_y = ( 0  -i )
    //             ( i   0 )
    //
    // as opposed to the sy matrix defined above.
    ComplexScalar coefficient() const {
	int sy_count = count_sy();
	if (sy_count == 0) {
	    return coeff;
	} else if (sy_count == 1) {
	    return complex<FpType>{0,-1} * coeff;
	} else if (sy_count == 2) {
	    return -coeff;
	} else {
	    return complex<FpType>{0,1} * coeff;
	}
    }

    // Returns true if the operator is hermitian.
    bool is_hermitian() const {
	return coefficient().imag() == 0;
    }

    // Returns true if the spin operator is a scalar operator
    DEVHOST bool is_scalar() const {
	return !bits;
    }

    // Compute the trace of the operator. We note that the trace is always zero
    // for non-identity operators, so this is in fact trivial to compute.
    DEVHOST ComplexScalar trace(int spin_chain_length) const {
	if (is_scalar()) {
	    return coeff * FpType(1ULL << spin_chain_length);
	} else {
	    return {};
	}
    }

    // Compute the result of applying this spin operator to a state vector. More
    // specifically, given the following vector
    //
    //   |psi> = \sum_{n = 000, 001, 010, 011, ...} psi_n |n>
    //
    // and a spin operator, say S = c*sy(2).sx(1).sz(0) (here sy is the sigma matrix
    // defined above, without i), the result of applying S to |psi> is the following
    // state vector
    //
    //   |psi'> = \sum_{n = 000, 001, ...} psi'_n |n> = S|psi>
    //
    // where psi'_000 = -c*psi_111, psi'_001 = c*psi_111, psi'_010 = -c*psi_100,
    // psi'_011 = c*psi_101, psi'_100 = c*psi_010, psi'_101 = -c*psi_011,
    // psi'_110 = c*psi_000, psi'_111 = -c*psi_001. These can be written in the form
    //
    //   psi'_n = (+/- c) * psi_res
    //
    // Given n, this function returns res and (+/- c).
    //
    // Note although in our convention, sx.sz = sy, for the spin state index n we
    // apply sx first and then sy. In other words, it's a contra-variant functor.
    // This is why we define sx to be 0b01 and sz to be 0b10.
    DEVHOST ComplexScalar apply(StateType n, StateType &res) const {
	bool minus = false;
	ReprType b = bits;
	for (int i = 0; i < NUM_SPIN_SITES; i++) {
	    if (b & 1) {
		n ^= 1 << i;
	    }
	    b >>= 1;
	    if ((b & 1) && ((n>>i) & 1)) {
		minus = !minus;
	    }
	    b >>= 1;
	    // This isn't strictly speaking necessary. We need to benchmark this.
	    if (!b) {
		break;
	    }
	}
	res = n;
	return minus ? -coeff : coeff;
    }

    class BitMap {
	using Word = u64;
	static constexpr auto WordBits = sizeof(Word) * 8;
	std::vector<Word> bits;
	size_t bit_count;
	friend class SpinOp;
	// Duplicate the given bitmap and put the copy right after it, optionally
	// flipping the bits. For instance, given the following bitmap
	//     1011
	//     ^  ^---- Least significant bit
	//     |
	//     Most significant bit
	// if both neg0 and neg1 are false, the duplicated bitmap will be
	// [MSB]1011`1011[LSB]. If neg0 is true and neg1 is false, the
	// duplicated bitmap will be [MSB]1011`0100[LSB], etc. Here 1<<n is
	// the number of bits in the bitmap. For n=0 we simply initialize the
	// bitmap to {neg0,neg1}. Once the function returns the number of bits
	// will be 1<<(n+1).
	void duplicate(bool neg0, bool neg1) {
	    assert(bit_count < (bits.size() * WordBits));
	    if (bit_count == 0) {
		bits[0] = (neg1 << 1) | neg0;
		bit_count = 2;
		return;
	    }
	    if (bit_count < WordBits) {
		auto word = bits[0];
		if (neg0) {
		    bits[0] ^= (1ULL << bit_count) - 1;
		}
		if (neg1) {
		    word ^= (1ULL << bit_count) - 1;
		}
		word <<= bit_count;
		bits[0] |= word;
	    } else {
		size_t word_count = bit_count / WordBits;
		#pragma omp parallel for
		for (size_t i = 0; i < word_count; i++) {
		    bits[i + word_count] = neg1 ? ~bits[i] : bits[i];
		    if (neg0) {
			bits[i] = ~bits[i];
		    }
		}
	    }
	    bit_count <<= 1;
	}
	// Only SpinOp can construct BitMap.
	constexpr BitMap() : bits(1), bit_count{2} {}
	constexpr BitMap(size_t capacity)
	    : bits((capacity < WordBits) ? 1 : (capacity/WordBits)), bit_count{0} {}
    public:
	// Other classes can access the n-th bit of the bitmap (but not modify).
	bool operator[](size_t n) const {
	    size_t word_idx = n / (sizeof(Word) * 8);
	    size_t rem = n % (sizeof(Word) * 8);
	    if (word_idx >= bits.size()) {
		ThrowException(InvalidArgument,
			       "n", "cannot exceed bitmap size");
	    }
	    return bits[word_idx] & (1ULL << rem);
	}
	// Allow other classes to display the bitmap.
	friend std::ostream &operator<<(std::ostream &os, const BitMap &bm) {
	    for (size_t i = 0; i < bm.bit_count; i++) {
		if (bm[i]) {
		    os << "-";
		} else {
		    os << "+";
		}
		if ((i+1) != bm.bit_count) {
		    os << " ";
		}
	    }
	    return os;
	}
    };

    using ColList = std::vector<StateType>;

private:
    // Duplicate the given column coordinate list. Each column coordinate
    // in the original list is bit-wise prepended with col0 left shifted
    // by n bits. The copied list is prepended with col1 left shifted by n
    // bits and is placed immediately after the original list. The number
    // of coordinates is given by 1<<n. If n == 0, cols will be initialized
    // to {col0, col1}.
    void duplicate_cols(ColList &cols, int n, StateType col0, StateType col1) const {
	if (n == 0) {
	    cols[0] = col0;
	    cols[1] = col1;
	    return;
	}
	size_t num_coords = 1ULL << n;
	assert(num_coords <= cols.size());
	#pragma omp parallel for
	for (size_t i = 0; i < num_coords; i++) {
	    cols[i + num_coords] = cols[i] | (col1 << n);
	    cols[i] |= (col0 << n);
	}
    }

public:
    // Returns the sparse matrix corresponding to the sigma opeartor part
    // (ie. without the coefficient) of the spin operator, in the form of
    // a bitmap and the column coordinate list. Since the non-zero element
    // of id,sx,sy,sz can only be one or minus one, we use a bitmap indicate
    // whether the non-zero element has a minus sign (ie. true == has minus
    // sign). Since the row coordinate list is always {0,1,2,...,n-1} where
    // n is the Hilbert space dimension, we do not need to compute the row
    // coordinate list. Here the input parameter len is the spin chain length.
    std::tuple<BitMap, ColList> get_sparse_matrix(int len) const {
	assert(len >= 0);
	// If the length is zero, simply return the identity matrix
	if (len == 0) {
	    return {{}, {0,1}};
	}
	// Allocate the objects for the COO matrix. There are exactly 1<<len
	// non-zero elements in the matrix so we know how much space we need.
	BitMap bm(1ULL << len);
	ColList cols(1ULL << len);
	for (int i = 0; i < len; i++) {
	    switch ((*this)[i]) {
	    case Sigma::id:
		bm.duplicate(false, false);
		duplicate_cols(cols, i, 0, 1);
		break;
	    case Sigma::x:
		bm.duplicate(false, false);
		duplicate_cols(cols, i, 1, 0);
		break;
	    case Sigma::y:
		bm.duplicate(true, false);
		duplicate_cols(cols, i, 1, 0);
		break;
	    case Sigma::z:
		bm.duplicate(false, true);
		duplicate_cols(cols, i, 0, 1);
		break;
	    }
	}
	assert(bm.bit_count == (1ULL << len));
	assert(cols.size() == (1ULL << len));
	return {bm, cols};
    }

    friend std::ostream &operator<<(std::ostream &os, const SpinOp &op) {
	if (op.coeff == 0.0) {
	    os << "0";
	    return os;
	}
	ComplexScalar coeff = op.coefficient();
	if (coeff != 1.0 && coeff != -1.0) {
	    if (coeff == 1.0_i) {
		os << "i";
	    } else if (coeff == -1.0_i) {
		os << "-i";
	    } else if (coeff.real() * coeff.imag() != 0) {
		os << "(" << coeff.real() << "+" << coeff.imag() << "i" << ")";
	    } else if (coeff.imag() == 0) {
		os << coeff.real();
	    } else {
		os << coeff.imag() << "i";
	    }
	    os << "*";
	} else if (coeff == -1.0) {
	    os << "-";
	}
	if (op.is_scalar()) {
	    os << "I";
	    return os;
	}
	bool dot = false;
	for (auto [s, id] : op) {
	    if (dot) {
		os << ".";
	    }
	    os << s << "(" << u32(id) << ")";
	    dot = true;
	}
	return os;
    }
};
