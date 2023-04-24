/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    syk.h

Abstract:

    This header file contains implementations of the SYK model Hamiltonian
    for the REAPERS library.

Revision History:

    2023-03-09  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

template<typename FpType>
class SYK {
    int N;
    FpType sp;
    std::vector<FermionOp<FpType>> gamma;
    FpType scale;
    FpType p;

public:
    SYK(int N, FpType sp) : N(N), sp(sp) {
	if (N < 4) {
	    ThrowException(InvalidArgument, "N", "must be at least four");
	} else if (N & 1) {
	    ThrowException(InvalidArgument, "N", "must be even");
	}
	scale = std::sqrt(3.0/(8.0*N*N*N));
	p = sp*24 / ((N-1)*(N-2)*(N-3));
	if (sp != 0.0) {
	    scale /= sqrt(p);
	}
	for (int i = 0; i < N; i++) {
	    gamma.emplace_back(FermionOp<FpType>(N, i));
	}
    }

    SYK(int N) : SYK(N, 0.0) {}

    int num_fermions() const { return N; }

    FpType sparsity() const { return sp; }

    bool is_sparse() const { return sp != 0.0; }

    const FermionOp<FpType> &fermion_ops(int n) const {
	if (!(n >= 0) && (n < N)) {
	    ThrowException(InvalidArgument, "n", "must be in [0,N)");
	}
	return gamma[n];
    }

    const std::vector<FermionOp<FpType>> &fermion_ops() const {
	return gamma;
    }

    // Generates the Hamiltonian for the standard single SYK model.
    // If sparsity is zero, it generates dense SYK. Otherwise it's sparse.
    template<typename RandGen>
    HamOp<FpType> gen_ham(RandGen &gen) const {
	HamOp<FpType> ham;
	std::normal_distribution<FpType> norm(0.0, 1.0);
	std::bernoulli_distribution bern(p);
	for (int i = 0; i < N; i++) {
	    for (int j = 0; j < i; j++) {
		for (int k = 0; k < j; k++) {
		    for (int l = 0; l < k; l++) {
			if (sp != 0.0 && !bern(gen)) {
			    continue;
			}
			ham += scale * norm(gen) * FermionOp<FpType>(N, i)
			    * FermionOp<FpType>(N, j) * FermionOp<FpType>(N, k)
			    * FermionOp<FpType>(N, l);
		    }
		}
	    }
	}
	return ham;
    }
};
