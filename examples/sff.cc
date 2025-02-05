/*++

Copyright (c) 2025  Dr. Chang Liu, PhD.

Module Name:

    sff.cc

Abstract:

    This program computes the spectral form factor of the SYK_4 model.

Revision History:

    2025-02-04  File created

--*/

#include <ctime>
#include <memory>
#include "pcg_random.h"

#ifdef MAX_NUM_FERMIONS
#define REAPERS_SPIN_SITES	(MAX_NUM_FERMIONS / 2)
#endif

#include <reapers.h>
#include "common.h"

using namespace REAPERS;
using namespace REAPERS::Model;
using namespace REAPERS::EvolutionAlgorithm;

template<RealScalar FpType>
using MatrixType = typename SumOps<FpType>::MatrixType;

template<RealScalar FpType>
using HamOp = typename SYK<FpType>::HamOp;

// Argument parser for the SYK simulation program.
struct SykArgParser : ArgParser {
    bool dump_ham;
    bool use_mt19937;
    bool regularize;
    bool seed_from_time;
    bool exact_diag;
    bool swap;
    float j_coupling;
    float kry_tol;

private:
    void parse_hook() {
	optdesc.add_options()
	    ("dump-ham", progopts::bool_switch(&dump_ham)->default_value(false),
	     "Dump the Hamiltonians constructed onto disk.")
	    ("use-mt19937", progopts::bool_switch(&use_mt19937)->default_value(false),
	     "Use the MT19937 from the C++ standard library as the pRNG."
	     " By default, PCG64 is used.")
	    ("regularize", progopts::bool_switch(&regularize)->default_value(false),
	     "Generate regularized sparse SYK.")
	    ("J", progopts::value<float>(&j_coupling)->default_value(1.0),
	     "Specifies the coupling strength J. The default is 1.")
	    ("tol", progopts::value<float>(&kry_tol)->default_value(0.0),
	     "Specifies the tolerance of the Krylov algorithm. The default is"
	     " machine precision.")
	    ("seed-from-time",
	     progopts::bool_switch(&seed_from_time)->default_value(false),
	     "Seed the pRNG using system time. By default, we use std::random_device"
	     " from the C++ stanfard library. Enable this if your C++ implementation's"
	     " std::random_device is broken (eg. deterministic).")
	    ("exact-diag", progopts::bool_switch(&exact_diag)->default_value(false),
	     "Use exact diagonalization rather than the Krylov method to compute"
	     " the state evolution exp(-iHt).")
	    ("swap", progopts::bool_switch(&swap)->default_value(false),
	     "Offload the initial state from VRAM to host memory.");
    }

    bool optcheck_hook() {
	if (exact_diag && trace) {
	    std::cerr << "You cannot specify both --exact-diag and --trace."
		      << std::endl;
	    return false;
	}

	return true;
    }
};

// Base evaluator class for n-point functions which computes a single disorder
// realization.
template<RealScalar FpType>
class BaseEval {
    virtual void evolve(const HamOp<FpType> &ham, State<FpType> &s, FpType t) = 0;
    virtual complex<FpType> evolve_trace(const HamOp<FpType> &ham, FpType t) = 0;

    FpType compute_ground_state_energy(const HamOp<FpType> &ham) {
	State<FpType> s1(args.N/2);
	return s1.template ground_state<Krylov>(ham, args.krylov_dim, (FpType)0.1);
    }

protected:
    const SykArgParser &args;

    ~BaseEval() {}

    void evolve_step(const SumOps<FpType> &ham, State<FpType> &s,
		     FpType t, FpType beta) {
	if (args.exact_diag) {
	    s.template evolve<ExactDiagonalization>(ham, t, beta);
	} else if (args.kry_tol != 0.0) {
	    s.evolve(ham, t, beta, args.krylov_dim, (FpType)args.kry_tol);
	} else {
	    s.evolve(ham, t, beta, args.krylov_dim);
	}
    }

public:
    std::string name;

    BaseEval(const char *name, const SykArgParser &args) : args(args), name(name) {}

    // Evaluate the n-point function from 0 to t_max. In the case where trace
    // is false (the default), we compute the inner product of s0 and s, where
    // s0 is a random state, and s is the result of evolve() applied to s0.
    // Note that in this case, before calling this function you must set s0
    // to a random state. If trace is true, we call evolve_trace(), which
    // the child classes will override to compute the final n-point function by
    // tracing over the n-point operator. In both cases results will be written
    // to vector v (after normalizing them by dividing by v[0]) as well as file
    // outf, and accumulated into sum.
    void eval(const SYK<FpType> &syk, HamOp<FpType> &ham,
	      std::unique_ptr<State<FpType>> &s0,
	      std::vector<complex<FpType>> &v, std::ofstream &outf,
	      std::vector<complex<FpType>> &sum,
	      std::vector<FpType> &abssum, Logger &logger) {
	auto start_time = time(nullptr);
	auto current_time = start_time;
	logger << "Running fp" << sizeof(FpType)*8 << " calculation." << endl;
	FpType dt = args.tmax / args.nsteps;
	if (args.trace) {
	    assert(!s0);
	    for (int i = 0; i <= args.nsteps; i++) {
		v[i] = evolve_trace(ham, i*dt) / complex<FpType>(1ULL << (args.N / 2));
		auto tm = time(nullptr);
		logger << "Time step " << i*dt << " done. Time for this step: "
		       << tm - current_time << "s"
		       << ". Total runtime so far for this disorder: "
		       << tm - start_time << "s." << endl;
		current_time = tm;
	    }
	} else {
	    assert(s0);
	    auto gnd = compute_ground_state_energy(ham);
	    ham += (-gnd) * SpinOp<FpType>::identity();
	    s0->gc();
	    std::unique_ptr<State<FpType,CPUImpl>> hostst;
	    if (args.swap) {
		hostst = std::make_unique<State<FpType,CPUImpl>>(*s0);
	    }
	    auto tm = time(nullptr);
	    logger << "Pre-evolve done. Time spent: "
		   << tm - current_time << "s." << endl;
	    current_time = tm;
	    for (int i = 0; i <= args.nsteps; i++) {
		State<FpType> s(*s0);
		if (args.swap) {
		    s0.reset();
		}
		evolve(ham, s, i*dt);
		if (args.swap) {
		    s.gc();
		    s0 = std::make_unique<State<FpType>>(*hostst);
		}
		v[i] = (*s0) * s;
		auto tm = time(nullptr);
		logger << "Time step " << i*dt << " done. Time for this step: "
		       << tm - current_time << "s"
		       << ". Total runtime so far for this disorder: "
		       << tm - start_time << "s." << endl;
		current_time = tm;
	    }
	}
	for (int i = 0; i <= args.nsteps; i++) {
	    using std::norm;
	    outf << dt*i << " " << v[i].real() << " " << v[i].imag() << std::endl;
	    sum[i] += v[i];
	    abssum[i] += norm(v[i]);
	}
    }
};

// This is the evaluator class for the SFF.
template<RealScalar FpType>
class SFF : public BaseEval<FpType> {
    void evolve(const HamOp<FpType> &ham, State<FpType> &s, FpType t) {
        this->evolve_step(ham, s, -t, 0);
    }

    complex<FpType> evolve_trace(const HamOp<FpType> &ham, FpType t) {
	return ham.matexp({0,t}).trace();
    }

public:
    SFF(const SykArgParser &args) : BaseEval<FpType>("SFF", args) {}
};

// Helper class which execute the computational tasks represented by
// the evaluator classes. We inherit from SykArgParser so we don't
// have to write args.??? in front of the command line parameters.
class Runner : protected SykArgParser {
    // Compute all disorder realization for the given n-point function
    template<template<typename> typename Eval, typename RandGen>
    void runjob(RandGen &rg) {
	std::stringstream ss;
	ss << "N" << N << "M" << M;
	if (sparsity != 0.0) {
	    ss << "k" << sparsity;
	} else {
	    ss << "dense";
	}
	ss << (regularize ? "reg" : "")
	   << "tmax" << tmax << "nsteps" << nsteps;
	if (!exact_diag && !trace) {
	    ss << "krydim" << krylov_dim;
	    if (kry_tol != 0.0) {
		ss << "krytol" << kry_tol;
	    }
	}
	if (j_coupling != 1.0) {
	    ss << "J" << j_coupling;
	}
	if (exact_diag) {
	    ss << "ed";
	}
	if (trace) {
	    ss << "trace";
	}
	Eval<float> eval32(*this);
	Eval<double> eval64(*this);
	auto jobname = eval32.name;
	auto outfname = jobname + "Disorders" + ss.str();
	auto avgfname = jobname + ss.str();
	std::ofstream outf32, outf64;
	std::ofstream avgf32, avgf64;
	if (fp32) {
	    outf32.rdbuf()->pubsetbuf(0, 0);
	    outf32.open(outfname + "fp32");
	    avgf32.rdbuf()->pubsetbuf(0, 0);
	    avgf32.open(avgfname + "fp32");
	}
	if (fp64) {
	    outf64.rdbuf()->pubsetbuf(0, 0);
	    outf64.open(outfname + "fp64");
	    avgf64.rdbuf()->pubsetbuf(0, 0);
	    avgf64.open(avgfname + "fp64");
	}
	std::ofstream hamf;
	if (dump_ham) {
	    hamf.rdbuf()->pubsetbuf(0, 0);
	    hamf.open(std::string("Ham") + outfname);
	}
	std::vector<complex<float>> v32(nsteps+1);
	std::vector<complex<double>> v64(nsteps+1);
	std::vector<complex<float>> sum32(nsteps+1);
	std::vector<complex<double>> sum64(nsteps+1);
	std::vector<float> abssum32(nsteps+1);
	std::vector<double> abssum64(nsteps+1);
	std::ofstream logf;
	logf.rdbuf()->pubsetbuf(0, 0);
	logf.open(std::string("Log") + jobname + ss.str());
	Logger logger(verbose, logf);
	logger << "Running " << jobname << " calculation using build "
	       << GITHASH << ".\nParameters are: N " << N
	       << " M " << M << " k " << sparsity
	       << (regularize ? " regularized" : "")
	       << " tmax " << tmax << " nsteps " << nsteps
	       << " krydim " << krylov_dim
	       << (use_mt19937 ? " MT19937" : " PCG64")
	       << (exact_diag ? " exact diag" : "")
	       << (trace ? " trace" : "")
	       << " J" << j_coupling;
	if (kry_tol != 0.0) {
	    logger << " krylov tolerance " << kry_tol;
	}
	logger << endl;
	std::unique_ptr<State<float>> init32;
	std::unique_ptr<State<double>> init64;
	if (fp32 && !trace) {
	    init32 = std::make_unique<State<float>>(N/2);
	}
	if (fp64 && !trace) {
	    init64 = std::make_unique<State<double>>(N/2);
	}
	SYK<float> syk32(N, sparsity, j_coupling);
	SYK<double> syk64(N, sparsity, j_coupling);
	HamOp<float> ham32;
	HamOp<double> ham64;
	double dt = tmax / nsteps;
	for (int u = 0; u < M; u++) {
	    logger << "Computing disorder " << u << endl;
	    if (fp32) {
		if (!trace) {
		    init32->random_state();
		}
		ham32 = syk32.gen_ham(rg, regularize);
	    }
	    if (fp64) {
		if (!trace) {
		    init64->random_state();
		}
		ham64 = syk64.gen_ham(rg, regularize);
	    }
	    if (dump_ham) {
		if (fp32) {
		    hamf << "H = " << ham32 << std::endl;
		}
		if (fp64) {
		    hamf << "H = " << ham64 << std::endl;
		}
	    }
	    // Compute one single disorder realization.
	    if (fp32) {
		eval32.eval(syk32, ham32, init32, v32, outf32, sum32, abssum32, logger);
	    }
	    if (fp64) {
		eval64.eval(syk64, ham64, init64, v64, outf64, sum64, abssum64, logger);
	    }
	}
	for (int i = 0; i <= nsteps; i++) {
	    using std::norm;
	    if (fp32) {
		sum32[i] /= M;
		abssum32[i] /= M;
		float sff32 = abssum32[i] - norm(sum32[i]);
		avgf32 << dt*i << " " << sff32 << std::endl;
	    }
	    if (fp64) {
		sum64[i] /= M;
		abssum64[i] /= M;
		double sff64 = abssum64[i] - norm(sum64[i]);
		avgf64 << dt*i << " " << sff64 << std::endl;
	    }
	}
    }

public:
    int run(int argc, const char *argv[]) {
	if (!parse(argc, argv)) {
	    return 1;
	}
	unsigned int seed = 0;
	if (seed_from_time) {
	    seed = time(nullptr);
	} else {
	    std::random_device rd;
	    seed = rd();
	}
	if (use_mt19937) {
	    // The user requested mt19937 as the pRNG. Use it.
	    std::mt19937 rg(seed);
	    runjob<SFF>(rg);
	} else {
	    // Use PCG64 as the random number engine
	    pcg64 rg(seed);
	    runjob<SFF>(rg);
	}
	return 0;
    }
};

int main(int argc, const char *argv[])
{
    try {
	std::cout << "Human. This is SYK"
#ifndef REAPERS_NOGPU
		  << "-GPU"
#endif
#ifdef __INTEL_LLVM_COMPILER
		  << "-AVX"
#endif
		  << "-" GITHASH ", powered by the REAPERS library.\n" << std::endl;
	Runner runner;
	runner.run(argc, argv);
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
