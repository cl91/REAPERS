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

    bool is_multiset_empty(const std::vector<int> &v) const {
	for (auto i : v) {
	    if (i != 0) {
		return false;
	    }
	}
	return true;
    }

    int multiset_elem_count(const std::vector<int> &v) const {
	int count = 0;
	for (auto i : v) {
	    count += i;
	}
	return count;
    }

    template<typename RandGen>
    std::tuple<bool,std::tuple<u8,u8,u8,u8>> get_edge(const std::vector<int> &v,
						      RandGen &rg) const {
	std::set<u8> edge;
	std::uniform_int_distribution<int> unif_int(0,N-1);
	while (edge.size() != 4) {
	    int vert = unif_int(rg);
	    if (v[vert] > 0 && !edge.contains(vert)) {
		edge.insert(vert);
	    }
	    // Check whether we have enough vertices available. If not, return false.
	    std::set<u8> avail_vert;
	    for (u8 av = 0; av < N; av++) {
		if (v[av] && !edge.contains(av)) {
		    avail_vert.insert(av);
		}
	    }
	    if (avail_vert.size() + edge.size() < 4) {
		return {false, {}};
	    }
	}
	std::vector<u8> res(4);
	int idx = 0;
	for (auto vert : edge) {
	    res[idx++] = vert;
	}
	return {true, {res[0], res[1], res[2], res[3]}};
    }

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

    // Generates the Hamiltonian for the regularized sparse SYK model.
    // Regularization here means we ensure that there are no disconnected
    // fermion blocks that don't interact with other blocks. See [1,2].
    // The algorithm we implement is modified from the algorithm described
    // in Appx. A of [1]. We generate an "almost" kq-regular hypergraph
    // where the vast majority of vertices have an exact degree of kq and
    // the remaining few have, say kq+1 or kq-1. The total number of edges
    // is kN. It's probably not the smartest algorithm but it seems to work.
    //
    // [1] Sparse Sachdev-Ye-Kitaev model, quantum chaos, and gravity duals
    //     PHYSICAL REVIEW D 103, 106002 (2021)
    // [2] A Sparse Model of Quantum Holography, arxiv.org/abs/2008.02303
    template<typename RandGen>
    HamOp<FpType> gen_ham_regularized(RandGen &gen) const {
	// If you want a regularized sparse SYK, you can only specify
	// a sparsity k = n/4 where n in an integer.
	if ((float)((int)(4*sp)) != 4*sp) {
	    ThrowException(InvalidArgument, "4k", "must be an integer");
	}
	// Every vertex in the hypergraph must have the exact same
	// degree which for 4-SYK is 4k.
	int deg = (int)(4*sp);
	// Array of available vertices. This is a so-called multi-set
	// where a vertex can appear multiple times (v[i] stands for
	// how many i-th vertices there are in the multi-set).
	std::vector<int> v(N);
	// Edges of the final constructed 4k-regular hypergraph.
	std::set<std::tuple<u8,u8,u8,u8>> edges;
    try_again:
	// Each vertex has initially 4k available degrees.
	for (int i = 0; i < N; i++) {
	    v[i] = deg;
	}
	// Start with an empty set of edges.
	edges.clear();
	// If every iteration generates an edge we will only iterate kN times.
	// The extra 10% more iteractions below is to account for the case
	// where the edges are rejected.
	for (int i = 0; i < (int)(sp*N*1.1); i++) {
	    // Select four different vertices from the available set of vertices.
	    bool avail;
	    std::tuple<u8,u8,u8,u8> edge;
	    std::tie(avail,edge) = get_edge(v, gen);
	    // If no more edges are available, go back and start fresh
	    if (!avail) {
		goto try_again;
	    }
	    int v0, v1, v2, v3;
	    std::tie(v0,v1,v2,v3) = edge;
	    // If the existing edges already contain the new edge, try
	    // generating a different edge.
	    if (edges.contains(edge)) {
		continue;
	    }
	    // Otherwise, decrease the availability of the four vertices
	    v[v0]--;
	    v[v1]--;
	    v[v2]--;
	    v[v3]--;
	    // And add the new edge to the set of edges.
	    edges.insert(edge);
	}
	// If we don't have a 4k-regular hypergraph, we are at least very close
	// to it. In this case just add the remaining ones without trying to
	// maintain regularity.
	if (!is_multiset_empty(v)) {
	    int count = multiset_elem_count(v) / 4;
	    // We don't care about regularity now, so just set v[i] to 1.
	    for (int i = 0; i < N; i++) {
		v[i] = 1;
	    }
	    for (int i = 0; i < count; i++) {
		bool avail;
		std::tuple<u8,u8,u8,u8> edge;
		std::tie(avail,edge) = get_edge(v, gen);
		if (edges.contains(edge)) {
		    i--;
		    continue;
		}
		edges.insert(edge);
	    }
	}
	// We have a 4k-regular hypergraph, so construct the sparse SYK.
	std::normal_distribution<FpType> norm(0.0, 1.0);
	HamOp<FpType> ham;
	for (auto edge : edges) {
	    u8 i, j, k, l;
	    std::tie(i,j,k,l) = edge;
	    ham += scale * norm(gen) * FermionOp<FpType>(N, i)
		* FermionOp<FpType>(N, j) * FermionOp<FpType>(N, k)
		* FermionOp<FpType>(N, l);
	}
	return ham;
    }

    // Generates the Hamiltonian for the standard single SYK model.
    // If sparsity is zero, it generates dense SYK. Otherwise it's sparse.
    template<typename RandGen>
    HamOp<FpType> gen_ham(RandGen &gen, bool regularize = false) const {
	if (regularize && sp != 0.0) {
	    return gen_ham_regularized(gen);
	}
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
