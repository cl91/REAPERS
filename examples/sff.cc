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

// Argument parser for the SYK simulation program.
struct SykArgParser : ArgParser {
    bool dump_ham;
    bool want_sff;
    bool want_greenfcn;
    bool want_otoc;
    bool regularize;
    bool seed_from_time;
    bool exact_diag;
    bool swap;
    float beta;
    float j_coupling;
    float kry_tol;

private:
    void parse_hook() {
	optdesc.add_options()
	    ("sff", progopts::bool_switch(&want_sff)->default_value(false),
	     "Compute the spectral form factor.")
	    ("green", progopts::bool_switch(&want_greenfcn)->default_value(false),
	     "Compute the Green's function.")
	    ("otoc", progopts::bool_switch(&want_otoc)->default_value(false),
	     "Compute the OTOC. If neither --green or --otoc is specified, compute both.")
	    ("dump-ham", progopts::bool_switch(&dump_ham)->default_value(false),
	     "Dump the Hamiltonians constructed onto disk.")
	    ("beta,b", progopts::value<float>(&beta)->required(),
	     "Specifies the inverse temperature of the simulation.")
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
	// If user didn't specify anything, compute all of them
	if (!want_sff && !want_greenfcn && !want_otoc) {
	    want_sff = want_greenfcn = want_otoc = true;
	}

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
    virtual void pre_evolve(const HamOp<FpType> &ham, BlockState<FpType> &s,
			    FpType beta) {}
    virtual void evolve(const HamOp<FpType> &ham, BlockState<FpType> &s, FpType t,
			FpType beta, const std::vector<FermionOp<FpType>> &ops) = 0;
    virtual complex<FpType> evolve_trace(const HamOp<FpType> &ham, FpType t, FpType beta,
					 const std::vector<FermionOp<FpType>> &ops) = 0;

    FpType compute_ground_state_energy(const HamOp<FpType> &ham) {
	BlockState<FpType> s1(args.N/2);
	return s1.template ground_state<Krylov>(ham, args.krylov_dim, (FpType)0.1);
    }

protected:
    const SykArgParser &args;
    bool normalize = false;

    ~BaseEval() {}

    void evolve_step(const HamOp<FpType> &ham, BlockState<FpType> &s,
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
    bool sff = false;

    BaseEval(const char *name, const SykArgParser &args) : args(args), name(name) {}

    // Evaluate the n-point function from 0 to t_max. In the case where trace
    // is false (the default), we compute the inner product of s0 and s, where
    // s0 is the result of pre_evolve() applied to a random state, and s is the
    // result of evolve() applied to s0. Note that in this case, before calling
    // this function you must set s0 to a random state, and the state s0 is
    // modified during the call so if you need the original initial state, you
    // need to make a copy of it before calling this function. This is to minimize
    // memory allocation overhead. If trace is set to true, this function
    // computes exp(-beta H/4) as a matrix and then calls evolve_trace(), which
    // the child classes will override to compute the final n-point function by
    // tracing over the n-point operator. In both cases results will be written
    // to vector v (after normalizing them by dividing by v[0]) as well as file
    // outf, and accumulated into sum.
    void eval(const SYK<FpType> &syk, HamOp<FpType> &ham,
	      std::unique_ptr<BlockState<FpType>> &s0,
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
		v[i] = evolve_trace(ham, i*dt, args.beta, syk.fermion_ops());
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
	    ham.LL += (-gnd) * SpinOp<FpType>::identity();
	    ham.RR += (-gnd) * SpinOp<FpType>::identity();
	    pre_evolve(ham, *s0, args.beta);
	    s0->gc();
	    std::unique_ptr<BlockState<FpType,CPUImpl>> hostst;
	    if (args.swap) {
		hostst = std::make_unique<BlockState<FpType,CPUImpl>>(*s0);
	    }
	    auto tm = time(nullptr);
	    logger << "Pre-evolve done. Time spent: "
		   << tm - current_time << "s." << endl;
	    current_time = tm;
	    for (int i = 0; i <= args.nsteps; i++) {
		BlockState<FpType> s(*s0);
		if (args.swap) {
		    s0.reset();
		}
		evolve(ham, s, i*dt, args.beta, syk.fermion_ops());
		if (args.swap) {
		    s.gc();
		    s0 = std::make_unique<BlockState<FpType>>(*hostst);
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
	auto v0 = v[0];
	for (int i = 0; i <= args.nsteps; i++) {
	    if (normalize) {
		v[i] /= v0;
	    }
	    outf << dt*i << " " << v[i].real() << " " << v[i].imag() << std::endl;
	    sum[i] += v[i];
	    using std::norm;
	    abssum[i] += norm(v[i]);
	}
    }
};

// This is the evaluator class for the SFF.
template<RealScalar FpType>
class SFF : public BaseEval<FpType> {
    void evolve(const HamOp<FpType> &ham, BlockState<FpType> &s, FpType t, FpType beta,
		const std::vector<FermionOp<FpType>> &ops) {
        this->evolve_step(ham, s, -t, beta);
    }

    complex<FpType> evolve_trace(const HamOp<FpType> &ham, FpType t, FpType beta,
				 const std::vector<FermionOp<FpType>> &ops) {
	auto N = this->args.N;
	return ham.matexp(complex{-beta,t}).trace() / FpType(1ULL << (N/2));
    }

public:
    SFF(const SykArgParser &args) : BaseEval<FpType>("SFF", args) {
	this->normalize = false;
	this->sff = true;
    }
};

// This is the evaluator class for the Green's function.
template<RealScalar FpType>
class Green : public BaseEval<FpType> {
    void evolve(const HamOp<FpType> &ham, BlockState<FpType> &s, FpType t, FpType beta,
		const std::vector<FermionOp<FpType>> &ops) {
	auto N = this->args.N;
	auto op = ops[N-1];
        this->evolve_step(ham, s, 0, beta/2);
	s *= op;
	this->evolve_step(ham, s, t, 0);
	s *= op;
	this->evolve_step(ham, s, -t, beta/2);
    }

    complex<FpType> evolve_trace(const HamOp<FpType> &ham, FpType t, FpType beta,
				 const std::vector<FermionOp<FpType>> &ops) {
	auto m0 = ham.matexp(complex<FpType>{0,-t});
	auto m1 = ham.matexp(complex<FpType>{-beta,t});
	auto N = this->args.N;
	auto op = ops[N-1].get_matrix();
	return (m1 * op * m0 * op).trace();
    }

public:
    Green(const SykArgParser &args) : BaseEval<FpType>("Green", args) {}
};

// This is the evaluator class for the OTOC.
template<RealScalar FpType>
class OTOC : public BaseEval<FpType> {
    void evolve(const HamOp<FpType> &ham, BlockState<FpType> &s, FpType t, FpType beta,
		const std::vector<FermionOp<FpType>> &ops) {
	auto N = this->args.N;
	auto op0 = ops[N-2];
	auto op1 = ops[N-1];
	this->evolve_step(ham, s, 0, beta/8);
	s *= op0;
	this->evolve_step(ham, s, t, beta/4);
	s *= op1;
	this->evolve_step(ham, s, -t, beta/4);
	s *= op0;
	this->evolve_step(ham, s, t, beta/4);
	s *= op1;
	this->evolve_step(ham, s, -t, beta/8);
    }

    complex<FpType> evolve_trace(const HamOp<FpType> &ham, FpType t, FpType beta,
				 const std::vector<FermionOp<FpType>> &ops) {
	auto N = this->args.N;
	auto op0 = ops[N-2].get_matrix();
	auto op1 = ops[N-1].get_matrix();
	auto m0 = ham.matexp(complex<FpType>{0,-t});
	auto m1 = ham.matexp(complex<FpType>{0,t});
	auto m2 = ham.matexp(complex<FpType>{-beta,t});
	return (m2 * op1 * m0 * op0 * m1 * op1 * m0 * op0).trace();
    }

public:
    OTOC(const SykArgParser &args) : BaseEval<FpType>("OTOC", args) {}
};

// Helper class which execute the computational tasks represented by
// the evaluator classes. We inherit from SykArgParser so we don't
// have to write args.??? in front of the command line parameters.
class Runner : protected SykArgParser {
    // Compute all disorder realization for the given n-point function
    template<template<typename> typename Eval, typename FpType>
    void runjob(auto rg) {
	std::stringstream ss;
	ss << "N" << N << "M" << M << "beta" << beta;
	if (sparsity != 0.0) {
	    ss << "k" << sparsity;
	} else {
	    ss << "dense";
	}
	ss << (regularize ? "reg" : "") << "tmax" << tmax << "nsteps" << nsteps;
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
	ss << "fp" << sizeof(FpType) * 8;

	Eval<FpType> eval(*this);
	auto jobname = eval.name;
	auto outfname = jobname + "Disorders" + ss.str();
	auto avgfname = jobname + ss.str();
	std::ofstream outf, avgf, hamf;
	outf.rdbuf()->pubsetbuf(0, 0);
	outf.open(outfname);
	avgf.rdbuf()->pubsetbuf(0, 0);
	avgf.open(avgfname);
	if (dump_ham) {
	    hamf.rdbuf()->pubsetbuf(0, 0);
	    hamf.open(std::string("Ham") + outfname);
	}
	std::vector<complex<FpType>> v(nsteps+1);
	std::vector<complex<FpType>> sum(nsteps+1);
	std::vector<FpType> abssum(nsteps+1);
	std::ofstream logf;
	logf.rdbuf()->pubsetbuf(0, 0);
	logf.open(std::string("Log") + jobname + ss.str());
	Logger logger(verbose, logf);
	logger << "Running " << jobname << " calculation using build "
	       << GITHASH << ".\nParameters are: N " << N
	       << " M " << M << " beta " << beta << " k " << sparsity
	       << (regularize ? " regularized" : "")
	       << " tmax " << tmax << " nsteps " << nsteps
	       << " krydim " << krylov_dim
	       << " PCG64"
	       << (exact_diag ? " exact diag" : "")
	       << (trace ? " trace" : "")
	       << " J" << j_coupling;
	if (kry_tol != 0.0) {
	    logger << " krylov tolerance " << kry_tol;
	}
	logger << endl;
	std::unique_ptr<BlockState<FpType>> init;
	if (!trace) {
	    init = std::make_unique<BlockState<FpType>>(N/2);
	}
	SYK<FpType> syk(N, sparsity, j_coupling);
	HamOp<FpType> ham;
	double dt = tmax / nsteps;
	for (int u = 0; u < M; u++) {
	    logger << "Computing disorder " << u << endl;
	    if (!trace) {
		init->random_state();
	    }
	    ham = syk.gen_ham(rg, regularize);
	    if (dump_ham) {
		hamf << "H = " << ham << std::endl;
	    }
	    // Compute one single disorder realization.
	    eval.eval(syk, ham, init, v, outf, sum, abssum, logger);
	}
	for (int i = 0; i <= nsteps; i++) {
	    sum[i] /= M;
	    abssum[i] /= M;
	    if (eval.sff) {
		using std::norm;
		FpType sff = abssum[i] - norm(sum[i]);
		avgf << dt*i << " " << sff << std::endl;
	    } else {
		avgf << dt*i << " " << sum[i].real() << " "
		     << sum[i].imag() << std::endl;
	    }
	}
    }

    template<typename FpType>
    void run_jobs(auto rg) {
	if (want_sff) {
	    runjob<SFF, FpType>(rg);
	}
	if (want_otoc) {
	    runjob<OTOC, FpType>(rg);
	}
	if (want_greenfcn) {
	    runjob<Green, FpType>(rg);
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
	// Use PCG64 as the random number engine
	pcg64 rg(seed);
	if (fp32) {
	    run_jobs<float>(rg);
	}
	if (fp64) {
	    run_jobs<double>(rg);
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
