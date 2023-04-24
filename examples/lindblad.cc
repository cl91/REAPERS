/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    lindblad.cc

Abstract:

    This module file contains the implementation of the Monte Carlo algorithm
    for computing the entanglement entropy of the SYK model.

Revision History:

    2023-03-08  File created

--*/

#include <ctime>
#include <filesystem>
#include "argparser.h"

#ifdef MAX_NUM_FERMIONS
#define REAPERS_SPIN_SITES	(MAX_NUM_FERMIONS / 2)
#endif

#define REAPERS_USE_MATRIX_POWER
#include "reapers.h"

using namespace REAPERS;
using namespace REAPERS::Model;

// Evaluator class for the entanglement entropy of the Lindblad SYK.
class Eval : protected ArgParser {
    bool gndst;
    bool use_krylov;
    int matpow;
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
	    ("matrix-power", progopts::value<int>(&matpow)->default_value(1),
	     "Specify the order of matrix power expansion when computing exp(-iHt)."
	     " This option is ignored if --use-krylov is specified.")
	    ("use-krylov", progopts::bool_switch(&use_krylov)->default_value(false),
	     "Use the Krylov algorithm for state evolution, instead of expanding the"
	     " matrix exponential using Taylor series. This is disabled by default."
	     " When dt is large you should enable it.");
    }

    bool optcheck_hook() {
	// If user has specified a data directory, switch to it.
	std::filesystem::create_directory(data_dir);
	std::filesystem::current_path(data_dir);
	return true;
    }

    template<typename FpType, typename FpType1>
    void evolve(State<FpType> &psi, const HamOp<FpType> &ham, FpType1 t) {
	if (use_krylov) {
	    psi.template evolve<EvolutionAlgorithm::Krylov>(ham, t, 0, krylov_dim);
	} else {
	    psi.evolve(ham, t, 0, matpow);
	}
    }

    template<typename FpType, typename RandGen>
    void step(const SYK<FpType> &syk, const HamOp<FpType> &ham, State<FpType> &psi,
	      RandGen &rg, float mu, float dt, bool forward) {
	std::uniform_real_distribution<FpType> unif_real(0,1);
	std::uniform_int_distribution<int> unif_int(0,N-1);
	FpType p = 1 - std::exp(-N*mu*dt/2);
	FpType eps = unif_real(rg);
	if (p < eps) {
	    evolve(psi, ham, forward ? dt : -dt);
	} else {
            psi *= syk.fermion_ops(unif_int(rg));
	}
	psi.normalize();
    }

    // Simulate one disorder realization
    template<typename FpType, typename RandGen>
    void runone(const SYK<FpType> &syk, RandGen &rg, std::ofstream &outf,
		const std::vector<FpType> &tarray, std::vector<FpType> &entropy) {
	auto N = syk.num_fermions();
	auto ham = syk.gen_ham(rg);
	State<FpType> init(N/2);
	if (gndst) {
	    auto gs_energy = init.ground_state(ham);
	    if (verbose) {
		std::cout << "Ground state energy " << gs_energy << std::endl;
	    }
	    init.gc();
	} else {
	    init.random_state();
	}
	State<FpType> psi(N/2);
	for (size_t u = 0; u < tarray.size(); u++) {
	    auto T = tarray[u];
	    if (verbose) {
		std::cout << "Computing time point " << T << std::endl;
	    }
	    // Subdivide T into segments of dt and compute the purity
	    int nseg = (int)(T/dt);
	    FpType purity = 0.0;
	    // Compute m quantum trajectories
	    for (int i = 0; i < m; i++) {
		psi = init;
		// Compute the forward evolution
		for (int j = 0; j < nseg; j++) {
		    step(syk, ham, psi, rg, mu, dt, true);
		}
		// Starting from the result of the forward evolution,
		// compute the backward evolution
		for (int j = 0; j < nseg; j++) {
		    step(syk, ham, psi, rg, mu, dt, false);
		}
		// Find the overlap between the final and the initial state
		auto overlap = norm(init * psi);
		purity += overlap;
		if (verbose && !(i % 128)) {
		    std::cout << "Traj " << i << " done." << std::endl;
		}
		outf << T << " " << overlap << std::endl;
	    }
	    // Compute the entanglement entropy
	    purity /= m;
	    entropy[u] = -std::log(purity)*2/N;
	    if (verbose) {
		std::cout << "T = " << T << ", entropy = " << entropy[u] << std::endl;
	    }
	}
    }

    template <typename FpType>
    void runall() {
	std::stringstream ss;
	if (gndst) {
	    ss << "Gndst";
	} else {
	    ss << "Rndst";
	}
	ss << "N" << N << "M" << M << "dt" << dt << "mu" << mu << "m" << m
	   << "k" << sparsity << "t0" << t0 << "tmax" << tmax
	   << "nsteps" << nsteps << "fp" << sizeof(FpType) * 8;
	if (use_krylov) {
	    ss << "krydim" << krylov_dim;
	} else {
	    ss << "matpow" << matpow;
	}
	std::ofstream outf;
	outf.rdbuf()->pubsetbuf(0, 0);
	outf.open(ss.str());
	std::ofstream entf;
	entf.rdbuf()->pubsetbuf(0, 0);
	entf.open(std::string("Ent") + ss.str());
	std::ofstream avgf;
	avgf.rdbuf()->pubsetbuf(0, 0);
	avgf.open(std::string("Avg") + ss.str());

	std::random_device rd;
	std::mt19937 rg(rd());
	SYK<FpType> syk(N, sparsity);
	std::vector<FpType> tarray;
	if (t0 != 0.0) {
	    tarray.push_back(t0);
	}
	for (int i = 1; i <= nsteps; i++) {
	    tarray.push_back(t0 + (tmax-t0)*i/nsteps);
	}
	std::vector<FpType> entropy_sum(tarray.size());
	for (int i = 0; i < M; i++) {
	    std::vector<FpType> entropy(tarray.size());
	    runone(syk, rg, outf, tarray, entropy);
	    for (size_t j = 0; j < tarray.size(); j++) {
		entf << tarray[j] << " " << entropy[j] << std::endl;
		entropy_sum[j] += entropy[j];
	    }
	}
	for (size_t i = 0; i < tarray.size(); i++) {
	    entropy_sum[i] /= M;
	    avgf << tarray[i] << " " << entropy_sum[i] << std::endl;
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
