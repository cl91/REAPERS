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
template<typename FpType>
class HostSumOps {
    template<typename FpType1>
    friend class HostSumOps;

protected:
    std::vector<SpinOp<FpType>> ops;

public:
    using ComplexScalar = typename SpinOp<FpType>::ComplexScalar;

    // We can implicitly convert a spin operator into a HostSumOps with one element
    HostSumOps(const SpinOp<FpType> &op) : ops{op} {}

    // Conversion between sum operators of different floating point precision
    // must be explicit.
    template<typename FpType1>
    explicit HostSumOps(const HostSumOps<FpType1> &op) {
	for (auto o : op) {
	    ops.emplace_back(o);
	}
    }

    // We can initialize a HostSumOps with a list of SpinOp objects
    HostSumOps(const std::initializer_list<SpinOp<FpType>> &l = {}) {
	for (auto &&o : l) {
	    *this += o;
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
	for (auto &o : ops) {
	    if (o.bits == op.bits) {
		o.coeff += op.coeff;
		return *this;
	    }
	}
	ops.push_back(op);
	return *this;
    }

    // Add another HostSumOps to this one
    HostSumOps &operator+=(const HostSumOps &rhs) {
	for (auto &&o : rhs) {
	    *this += o;
	}
	return *this;
    }

    // Add the supplied operator and this operator together, without changing this
    // operator, and return the result.
    HostSumOps operator+(const SpinOp<FpType> &op) const {
	HostSumOps res(*this);
	res += op;
	return res;
    }

    // Multiply the specified operator from the right with this operator
    HostSumOps &operator*=(const HostSumOps &rhs) {
	*this = operator*(rhs);
	return *this;
    }

    // Multiply the specified operator from the right with this operator
    HostSumOps &operator*=(const ComplexScalar &rhs) {
	*this = operator*(rhs);
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
    HostSumOps operator*(ComplexScalar rhs) const {
	HostSumOps res;
	for (auto &&op0 : *this) {
	    res += rhs * op0;
	}
	return res;
    }

    // Scalar multiplication can be done from both directions.
    friend HostSumOps operator*(ComplexScalar s, const HostSumOps &ops) {
	return ops * s;
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
// Note: we cannot make this a template because in GPUImpl the class has static
// members that can only be initialized once (and we must follow GPUImpl because
// the State template takes Impl as a template parameter). In terms of design
// patterns this is a so-called 'singleton' class.
class CPUImpl {
    template<typename FpType>
    class HostVec {
	using ComplexScalar = complex<FpType>;
	size_t dim;
	std::unique_ptr<ComplexScalar[]> vec;
	// We don't allow copy assignment or copy construction of vectors.
	// Move construction and move assignment are allowed.
	HostVec(const HostVec &) = delete;
	HostVec &operator=(const HostVec &) = delete;
    public:
	HostVec(size_t dim) : dim(dim), vec(std::make_unique<ComplexScalar[]>(dim)) {}
	HostVec(HostVec &&) = default;
	HostVec &operator=(HostVec &&) = default;
	ComplexScalar *get() { return vec.get(); }
	ComplexScalar *get() const { return vec.get(); }
	friend void swap(HostVec &lhs, HostVec &rhs) {
	    HostVec v0(std::move(lhs));
	    lhs = std::move(rhs);
	    rhs = std::move(v0);
	}
	size_t size() const { return dim; }
    };

public:
    template<typename FpType>
    using VecType = HostVec<FpType>;

    template<typename FpType>
    using VecSizeType = typename SpinOp<FpType>::StateType;

    template<typename FpType>
    using ComplexScalar = complex<FpType>;

    template<typename FpType>
    using BufType = ComplexScalar<FpType> *;

    template<typename FpType>
    using ConstBufType = const ComplexScalar<FpType> *;

    template<typename FpType>
    using ElemRefType = ComplexScalar<FpType> &;

    template<typename FpType>
    using ElemConstRefType = const ComplexScalar<FpType> &;

    template<typename FpType>
    using SumOps = HostSumOps<FpType>;

    // Initialize the vector to a Haar random state.
    template<typename FpType>
    static void init_random(VecSizeType<FpType> size, BufType<FpType> v0) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<FpType> norm(0.0, 1.0);
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] = ComplexScalar<FpType>{norm(gen), norm(gen)};
	}
	FpType s = 1.0/vec_norm(size, v0);
	scale_vec(size, v0, s);
    }

    // Initialize the vector with zero, ie. set v0[i] to 0.0 for all i.
    template<typename FpType>
    static void zero_vec(VecSizeType<FpType> size, BufType<FpType> v0) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] = 0.0;
	}
    }

    // Copy v1 into v0, possibly with a different floating point precision.
    template<typename FpType0, typename FpType1>
    static void copy_vec(VecSizeType<FpType0> size, BufType<FpType0> v0,
			 ConstBufType<FpType1> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType0> i = 0; i < size; i++) {
	    v0[i] = v1[i];
	}
    }

    // Compute v0 += v1. v0 and v1 are assumed to point to different buffers.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] += v1[i];
	}
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			FpType s, ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] += s * v1[i];
	}
    }

    // Compute v0 += s * v1. v0 and v1 are assumed to point to different buffers.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			ComplexScalar<FpType> s, ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] += s * v1[i];
	}
    }

    // Compute res = v0 + v1. res is assumed to be different from both v0 and v1.
    template<typename FpType>
    static void add_vec(VecSizeType<FpType> size, BufType<FpType> res,
			ConstBufType<FpType> v0, ConstBufType<FpType> v1) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    res[i] = v0[i] + v1[i];
	}
    }

    // Compute v0 *= s where s is a real scalar.
    template<typename FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v0, FpType s) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] *= s;
	}
    }

    // Compute v0 *= s where s is a complex scalar.
    template<typename FpType>
    static void scale_vec(VecSizeType<FpType> size, BufType<FpType> v0,
			  ComplexScalar<FpType> s) {
	#pragma omp parallel for
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    v0[i] *= s;
	}
    }

    // Compute the vector inner product of the complex vector v0 and v1. In other words,
    // compute v = <v0 | v1> = v0^\dagger \cdot v1
    template<typename FpType>
    static ComplexScalar<FpType> vec_prod(VecSizeType<FpType> size,
					  ConstBufType<FpType> v0,
					  ConstBufType<FpType> v1) {
	FpType real = 0.0;
	FpType imag = 0.0;
	#pragma omp parallel for reduction (+:real,imag)
	for (VecSizeType<FpType> i = 0; i < size; i++) {
	    ComplexScalar<FpType> res = conj(v0[i]) * v1[i];
	    real += res.real();
	    imag += res.imag();
	}
	return ComplexScalar<FpType>{real, imag};
    }

    // Compute the vector norm of the complex vector v0. In other words, compute
    // v = sqrt(<v0 | v0>) = sqrt(v0^\dagger \cdot v0)
    template<typename FpType>
    static FpType vec_norm(VecSizeType<FpType> size, ConstBufType<FpType> v0) {
	return std::sqrt(vec_prod(size, v0, v0).real());
    }

    // Compute res += ops * vec. res and vec are assumed to be different.
    template<typename FpType>
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
