/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    lindblad.cc

Abstract:

    This module file contains the implementation of the Monte Carlo algorithm
    for computing the entanglement entropy of the SYK model. It is placed here
    due to publication requirement of having a single repo for the code of our
    paper [1] and is not designed to be readable by any stretch of the word.

    [1] Entanglement Transition and Replica Wormhole in the Dissipative
        Sachdev-Ye-Kitaev Model, arXiv:2306.12571 [quant-ph]

Revision History:

    2023-03-08  File created

--*/

#include <ctime>
#include <iomanip>

#ifdef MAX_NUM_FERMIONS
#define REAPERS_SPIN_SITES	(MAX_NUM_FERMIONS/2 - 1)
#endif

#define REAPERS_USE_PARITY_BLOCKS
#include <reapers.h>
#include "common.h"

using namespace REAPERS;
using namespace REAPERS::Model;
using namespace REAPERS::EvolutionAlgorithm;
template<RealScalar FpType>
using MatrixType = typename SumOps<FpType>::MatrixType;
template<RealScalar FpType>
using HamOp = typename SYK<FpType>::HamOp;

// Evaluator class for the entanglement entropy of the Lindblad SYK.
class Eval : protected ArgParser {
    bool gndst;
    bool use_krylov;
    int m;
    float dt;
    float mu;
    float t0;

    void parse_hook() {
	optdesc.add_options()
	    ("ground-state", progopts::bool_switch(&gndst)->default_value(false),
	     "Start with the ground state rather than random Haar-invariant state.")
	    ("num-trajectories,m", progopts::value<int>(&m)->required(),
	     "Specify the number of quantum trajectories (per disorder "
	     "realization) to simulate.")
	    ("dt", progopts::value<float>(&dt)->required(), "Specify dt.")
	    ("mu", progopts::value<float>(&mu)->required(), "Specify mu.")
	    ("t0", progopts::value<float>(&t0)->default_value(0.0),
	     "Specify the starting time point.")
	    ("use-krylov",
	     progopts::bool_switch(&use_krylov)->default_value(false),
	     "Use the Krylov algorithm to compute state evolution instead of exact"
	     " diagonalization. This is very slow and should only be used for"
	     " testing purpose.");
    }

    bool optcheck_hook() {
	if (t0 > tmax) {
	    std::cout << "Starting time cannot be greater than maximum time."
		      << std::endl;
	    return false;
	}
	if (trace && gndst) {
	    std::cout << "You cannot specify both --trace and --ground-state."
		      << std::endl;
	    return false;
	}
	if (trace && use_krylov) {
	    std::cout << "You cannot specify both --trace and --use-krylov."
		      << std::endl;
	    return false;
	}
	return true;
    }

    template<RealScalar FpType>
    void evolve(BlockState<FpType> &psi, const HamOp<FpType> &ham, float t) {
	if (use_krylov) {
	    psi.template evolve<Krylov>(ham, t, 0.0, krylov_dim);
	} else {
	    psi.template evolve<ExactDiagonalization>(ham, t, 0.0);
	}
    }

    template<RealScalar FpType>
    void evolve(BlockOp<MatrixType<FpType>> &psi,
		const HamOp<FpType> &ham, float t) {
	auto expmat = ham.matexp(complex<FpType>{0,-t});
	psi = expmat * psi;
    }

    // Generate a quantum trajectory (strictly speaking, half of a quantum
    // trajectory) for time point T. Once the function returns, traj will
    // be a list of integers [-1, -2, 0, 3, 0, ..], where each positive
    // integer n represents A^n where A = exp(+/-iHdt) and each non-positive
    // integer n represents the jump operator psi_{-n}.
    template<RealScalar FpType, typename RandGen>
    std::vector<int> gen_traj(const SYK<FpType> &syk, const HamOp<FpType> &ham,
			      float T, RandGen &rg) {
	std::vector<int> traj;
	std::uniform_real_distribution<FpType> unif_real(0,1);
	std::uniform_int_distribution<int> unif_int(0,N-1);
	// Subdivide T into segments of dt
	int nseg = (int)(T/dt);
	FpType p = 1 - std::exp(-N*mu*dt/2);
	int prev = 0;
	bool extra = false;
	for (int i = 0; i < nseg; i++) {
	    FpType eps = unif_real(rg);
	    if (p < eps) {
		if (prev <= 0) {
		    prev = 1;
		} else {
		    prev++;
		}
		extra = true;
	    } else {
		traj.push_back(prev);
		traj.push_back(-unif_int(rg));
		prev = 0;
		extra = false;
	    }
	}
	if (extra) {
	    traj.push_back(prev);
	}
	return traj;
    }

    template<RealScalar FpType, typename St>
    void evolve_traj(const SYK<FpType> &syk, St &psi, const HamOp<FpType> &ham,
		     const std::vector<int> &traj, bool forward) {
	for (auto n : traj) {
	    if (n > 0) {
		FpType t = forward ? (n*dt):(-n*dt);
		evolve<FpType>(psi, ham, t);
	    } else {
		psi = syk.fermion_ops(-n) * psi;
	    }
	}
    }

    // Compute the purity using the trace method
    template<RealScalar FpType>
    double compute_purity(const SYK<FpType> &syk, const HamOp<FpType> &ham,
			  const std::vector<int> &fwd_traj,
			  const std::vector<int> &bac_traj) {
	// Hilbert space dimension of each parity block
	ssize_t dim = 1LL << (N/2-1);
	// Initial matrix is the identity matrix
	BlockOp<MatrixType<FpType>> mat{BlockDiag<MatrixType<FpType>>{
		MatrixType<FpType>::Identity(dim,dim),
		MatrixType<FpType>::Identity(dim,dim)}};
	// Compute the forward evolution
	evolve_traj(syk, mat, ham, fwd_traj, true);
	// Starting from the result of the forward evolution,
	// compute the backward evolution
	evolve_traj(syk, mat, ham, bac_traj, false);
	// Compute the trace and divide by the number of matrix elements
	double tr = abs(mat.trace());
	return tr * tr / (1ULL << N);
    }

    // Compute the purity by compute the overlap between initial and final states
    template<RealScalar FpType>
    double compute_purity(const SYK<FpType> &syk, const HamOp<FpType> &ham,
			  const std::vector<int> &fwd_traj,
			  const std::vector<int> &bac_traj,
			  const BlockState<FpType> &init) {
	BlockState<FpType> psi(init);
	// Compute the forward evolution
	evolve_traj(syk, psi, ham, fwd_traj, true);
	// Starting from the result of the forward evolution,
	// compute the backward evolution
	evolve_traj(syk, psi, ham, bac_traj, false);
	// Compute the overlap
	return norm(psi * init);
    }

    double get_entropy(double purity) {
	return -std::log(purity)/N;
    }

    // Simulate one disorder realization, using the trace definition.
    template<RealScalar FpType, typename RandGen>
    void runone(const SYK<FpType> &syk, RandGen &rg, std::ofstream &outf,
		Logger &logger, const std::vector<FpType> &tarray,
		std::vector<double> &purity_avg) {
	auto start_time = time(nullptr);
	// Spin chain length
	int len = N/2-1;
	auto ham = syk.gen_ham(rg);
	std::unique_ptr<BlockState<FpType>> init;
	if (!trace) {
	    init = std::make_unique<BlockState<FpType>>(len);
	    if (gndst) {
		FpType gs_energy;
		if (use_krylov) {
		    gs_energy = init->template ground_state<Krylov>(ham, krylov_dim);
		} else {
		    gs_energy = init->template ground_state<ExactDiagonalization>(ham);
		}
		logger << "Ground state energy " << gs_energy
		       << " . Time elapsed: " << time(nullptr) - start_time
		       << "s." << endl;
		init->gc();
	    } else {
		init->random_state();
	    }
	}
	for (size_t u = 0; u < tarray.size(); u++) {
	    auto T = tarray[u];
	    logger << "Computing time point " << T << endl;
	    double purity = 0.0;
	    // Compute m quantum trajectories
	    for (int i = 0; i < m; i++) {
		// Generate the forward quantum trajectory
		auto fwd_traj = gen_traj(syk, ham, T, rg);
		// Generate the backward quantum trajectory
		auto bac_traj = gen_traj(syk, ham, T, rg);
		double overlap = 0.0;
		if (trace) {
		    overlap = compute_purity(syk, ham, fwd_traj, bac_traj);
		} else {
		    overlap = compute_purity(syk, ham, fwd_traj, bac_traj, *init);
		}
		if (!(i % 128)) {
		    logger << "Traj " << i << " done. Time elapsed: "
			   << time(nullptr) - start_time << "s." << endl;
		}
		outf << T << " " << overlap << std::endl;
		purity += overlap;
	    }
	    // Compute the entanglement entropy
	    purity_avg[u] = purity /= m;
	    logger << "T = " << T << ", entropy = " << get_entropy(purity)
		   << ". Time elapsed: " << time(nullptr) - start_time
		   << "s." << endl;
	}
    }

    template <RealScalar FpType>
    void runall() {
	std::stringstream ss;
	if (trace) {
	    ss << "Trace";
	} else if (gndst) {
	    ss << "Gndst";
	} else {
	    ss << "Rndst";
	}
	ss << "N" << N << "M" << M << "dt" << dt << "mu" << mu << "m" << m
	   << "k" << sparsity << "t0" << t0 << "tmax" << tmax
	   << "nsteps" << nsteps << "fp" << sizeof(FpType) * 8;
	if (!trace) {
	    if (use_krylov) {
		ss << "krydim" << krylov_dim;
	    } else {
		ss << "ed";
	    }
	}
	constexpr auto max_precision{std::numeric_limits<FpType>::digits10 + 1};
	std::ofstream outf;
	outf.rdbuf()->pubsetbuf(0, 0);
	outf.open(ss.str());
	outf << std::setprecision(max_precision);
	std::ofstream entf;
	entf.rdbuf()->pubsetbuf(0, 0);
	entf.open(std::string("Ent") + ss.str());
	entf << std::setprecision(max_precision);
	std::ofstream avgf;
	avgf.rdbuf()->pubsetbuf(0, 0);
	avgf.open(std::string("Avg") + ss.str());
	avgf << std::setprecision(max_precision);
	std::ofstream logf;
	logf.rdbuf()->pubsetbuf(0, 0);
	logf.open(std::string("Log") + ss.str());
	Logger logger(verbose, logf);
	logger << "Running Lindblad calculation using build "
	       << GITHASH << ".\nParameters are: N " << N
	       << " M " << M << " k " << sparsity
	       << " tmax " << tmax << " nsteps " << nsteps
	       << " m " << m << " dt " << dt << " mu " << mu
	       << (trace ? " trace" : "")
	       << (gndst ? " ground state" : "")
	       << (!(trace || gndst) ? " random state" : "");
	if (use_krylov) {
	    logger << " krylov dim " << krylov_dim;
	}
	logger << endl;

	std::random_device rd;
	std::mt19937 rg(rd());
	// Here we set standard_gamma to false, so the gamma_i operators
	// will be normalized to {gamma_i,gamma_j}=2delta_ij, which saves
	// us some time because it preserves the norm of state vectors.
	SYK<FpType> syk(N, sparsity, 1.0, false);
	std::vector<FpType> tarray;
	if (t0 != 0.0) {
	    tarray.push_back(t0);
	}
	if (t0 != tmax) {
	    for (int i = 1; i <= nsteps; i++) {
		tarray.push_back(t0 + (tmax-t0)*i/nsteps);
	    }
	}
	std::vector<double> entropy_sum(tarray.size());
	std::vector<double> purity_sum(tarray.size());
	for (int i = 0; i < M; i++) {
	    std::vector<double> purity(tarray.size());
	    logger << "Computing disorder " << i << endl;
	    runone(syk, rg, outf, logger, tarray, purity);
	    for (size_t j = 0; j < tarray.size(); j++) {
		entf << tarray[j] << " " << get_entropy(purity[j]) << std::endl;
		entropy_sum[j] += get_entropy(purity[j]);
		purity_sum[j] += purity[j];
	    }
	}
	for (size_t i = 0; i < tarray.size(); i++) {
	    entropy_sum[i] /= M;
	    purity_sum[i] /= M;
	    avgf << tarray[i] << " " << entropy_sum[i] << " "
		 << get_entropy(purity_sum[i]) << std::endl;
	}
    }

public:
    // Simulate all disorder realizations and compute the ensemble average
    int run(int argc, const char *argv[]) {
	if (!parse(argc, argv)) {
	    return 1;
	}

	if (fp32) {
	    runall<float>();
	}

	if (fp64) {
	    runall<double>();
	}
	return 0;
    }
};



int main(int argc, const char *argv[])
{
    try {
	std::cout << "Human. This is LINDBLAD"
#ifndef REAPERS_NOGPU
		  << "-GPU"
#endif
#ifdef __INTEL_LLVM_COMPILER
		  << "-AVX"
#endif
		  << "-" GITHASH ", powered by the REAPERS library.\n" << std::endl;
	Eval eval;
	return eval.run(argc, argv);
    } catch (const std::exception &e) {
	std::cerr << "Program terminated abnormally due to the following error:"
		  << std::endl << e.what() << std::endl;
	return 1;
    } catch (...) {
	std::cerr << "Program terminated abnormally due an unknown exception."
		  << std::endl;
	return 1;
    }
    return 0;
}
