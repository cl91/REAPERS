/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    syk.cc

Abstract:

    This program computes the retarded Green's function and the OTOC of the SYK model.

Revision History:

    2023-01-03  File created

--*/

#include <ctime>
#include <memory>
#include <filesystem>
#include "argparser.h"
#include "pcg_random.h"

#ifdef MAX_NUM_FERMIONS
#define REAPERS_SPIN_SITES	(MAX_NUM_FERMIONS / 2 - 1)
#endif

#define REAPERS_USE_PARITY_BLOCKS
#include <reapers.h>

using namespace REAPERS;
using namespace REAPERS::Model;

// Argument parser for the SYK simulation program.
struct SykArgParser : ArgParser {
    bool dump_ham;
    bool want_greenfcn;
    bool want_otoc;
    bool truncate_fp64;
    bool use_mt19937;
    bool regularize;
    bool non_standard_gamma;
    float beta;
    float j_coupling;
    int otoc_idx;
    float kry_tol;

private:
    void parse_hook() {
	optdesc.add_options()
	    ("green", progopts::bool_switch(&want_greenfcn)->default_value(false),
	     "Compute the retarded Green's function.")
	    ("otoc", progopts::bool_switch(&want_otoc)->default_value(false),
	     "Compute the OTOC. If neither --green or --otoc is specified, compute both.")
	    ("dump-ham", progopts::bool_switch(&dump_ham)->default_value(false),
	     "Dump the Hamiltonians constructed onto disk.")
	    ("beta,b", progopts::value<float>(&beta)->required(),
	     "Specifies the inverse temperature of the simulation.")
	    ("truncate-fp64", progopts::bool_switch(&truncate_fp64)->default_value(false),
	     "When both --fp32 and --fp64 are specified, truncate the fp64"
	     " couplings and random states to fp32, rather than promoting"
	     " fp32 to fp64. By default, fp32 couplings and random states"
	     " are promoted to fp64.")
	    ("use-mt19937", progopts::bool_switch(&use_mt19937)->default_value(false),
	     "Use the MT19937 from the C++ standard library as the pRNG."
	     " By default, PCG64 is used.")
	    ("regularize", progopts::bool_switch(&regularize)->default_value(false),
	     "Generate regularized sparse SYK.")
	    ("non-standard-gamma", progopts::bool_switch(&non_standard_gamma)->default_value(false),
	     "Use the non-standard normalization of gamma matrices {gamma_i, gamma_j}"
	     " = 2 delta_ij. This is for testing purpose only and should give the exact"
	     " same simulation results.")
	    ("J", progopts::value<float>(&j_coupling)->default_value(1.0),
	     "Specifies the coupling strength J. The default is 1.")
	    ("otoc-index", progopts::value<int>(&otoc_idx)->default_value(-1),
	     "Specifies the second fermion index in the OTOC. The default is N-2.")
	    ("tol", progopts::value<float>(&kry_tol)->default_value(0.0),
	     "Specifies the tolerance of the Krylov algorithm. The default is"
	     " machine precision.");
    }

    bool optcheck_hook() {
	// If user didn't specify Green fcn or OTOC, compute both
	if (!want_greenfcn && !want_otoc) {
	    want_greenfcn = want_otoc = true;
	}

	if (truncate_fp64 && !(fp32 && fp64)) {
	    std::cerr << "You cannot specify --truncate-fp64 unless you specified"
		" both --fp32 and --fp64." << std::endl;
	    return false;
	}

	// If user has specified a data directory, switch to it.
	std::filesystem::create_directory(data_dir);
	std::filesystem::current_path(data_dir);

	return true;
    }
};

// Helper class for logging
class Logger {
    bool verbose;
    std::ofstream &logf;

public:
    Logger(bool verbose, std::ofstream &logf) : verbose(verbose), logf(logf) {}

    // Overload the << operator so we can write logs using *this << obj
    template<typename T>
    Logger &operator<<(T &&c) {
	logf << c;
	if (verbose) {
	    std::cout << c;
	}
	return *this;
    }

    // We can also just say *this << "\n" (because we have disabled IO buffering
    // for log files), but we want to be fancy.
    friend Logger &endl(Logger &ev);

    // This is the magic that makes *this << endl work.
    Logger &operator<<(Logger &(*f)(Logger &)) {
	return f(*this);
    }
};

inline Logger &endl(Logger &ev) {
    ev.logf << std::endl;
    if (ev.verbose) {
	std::cout << std::endl;
    }
    return ev;
}

// Base evaluator class for n-point functions which computes a single disorder
// realization.
template<typename FpType>
class BaseEval {
    virtual void pre_evolve(const SumOps<FpType> &ham, State<FpType> &s,
			    FpType beta) = 0;
    virtual void evolve(const SumOps<FpType> &hLL, const SumOps<FpType> &hRR,
			State<FpType> &s, FpType t, FpType beta,
			const std::vector<FermionOp<FpType>> &ops) = 0;

protected:
    const SykArgParser &args;

    ~BaseEval() {}

    void evolve_step(const SumOps<FpType> &ham, State<FpType> &s,
		     FpType t, FpType beta) {
	if (args.kry_tol != 0.0) {
	    s.evolve(ham, t, beta, args.krylov_dim, (FpType)args.kry_tol);
	} else {
	    s.evolve(ham, t, beta, args.krylov_dim);
	}
    }

public:
    std::string name;

    BaseEval(const char *name, const SykArgParser &args) : args(args), name(name) {}

    // Evaluate the n-point function from 0 to t_max. The quantity computed
    // is the inner product of s0 and s, where s0 is the result of pre_evolve()
    // applied to a random state, and s is the result of evolve() applied to
    // s0. The output is normalized by dividing <s0|s0>. Note that before
    // calling this function you must set s0 to a random state, and the state
    // s0 is modified during the call so if you need the original initial
    // state, you need to make a copy of it before calling this function.
    // This is to minimize memory allocation overhead. Results will be written
    // to vector v as well as file outf, and accumulated into sum.
    void eval(const SYK<FpType> &syk, const HamOp<FpType> &ham, State<FpType> &s0,
	      State<FpType> &s, std::vector<complex<FpType>> &v,
	      std::ofstream &outf, std::vector<complex<FpType>> &sum,
	      Logger &logger) {
	auto start_time = time(nullptr);
	auto current_time = start_time;
	logger << "Running fp" << sizeof(FpType)*8 << " calculation." << endl;
	FpType dt = args.tmax / args.nsteps;
	pre_evolve(ham.LL, s0, args.beta);
	s0.gc();
	auto tm = time(nullptr);
	logger << "Pre-evolve done. Time spent: " << tm - current_time << "s." << endl;
	current_time = tm;
	for (int i = 0; i <= args.nsteps; i++) {
	    s = s0;
	    evolve(ham.LL, ham.RR, s, i*dt, args.beta, syk.fermion_ops());
	    v[i] = s0 * s;
	    auto tm = time(nullptr);
	    logger << "Time step " << i*dt << " done. Time for this step: "
		   << tm - current_time << "s"
		   << ". Total runtime so far for this disorder: "
		   << tm - start_time << "s." << endl;
	    current_time = tm;
	}
	auto v0 = v[0];
	for (int i = 0; i <= args.nsteps; i++) {
	    v[i] /= v0;
	    outf << dt*i << " " << v[i].real() << " " << v[i].imag() << std::endl;
	    sum[i] += v[i];
	}
    }
};

// This is the evaluator class for the retarded Green's function.
template<typename FpType>
class Green : public BaseEval<FpType> {
    void pre_evolve(const SumOps<FpType> &ham, State<FpType> &s, FpType beta) {
        this->evolve_step(ham, s, 0, beta/2);
    }

    void evolve(const SumOps<FpType> &hLL, const SumOps<FpType> &hRR,
		State<FpType> &s, FpType t, FpType beta,
		const std::vector<FermionOp<FpType>> &ops) {
        this->evolve_step(hRR, s, t, 0);
	this->evolve_step(hLL, s, -t, 0);
    }

public:
    Green(const SykArgParser &args) : BaseEval<FpType>("Green", args) {}
};

// This is the evaluator class for the OTOC.
template<typename FpType>
class OTOC : public BaseEval<FpType> {
    void pre_evolve(const SumOps<FpType> &ham, State<FpType> &s, FpType beta) {
        this->evolve_step(ham, s, 0, beta/8);
    }

    void evolve(const SumOps<FpType> &hLL, const SumOps<FpType> &hRR,
		State<FpType> &s, FpType t, FpType beta,
		const std::vector<FermionOp<FpType>> &ops) {
	auto N = this->args.N;
	auto otoc_idx = this->args.otoc_idx;
	auto op = ops[N-2].LR;
	if ((otoc_idx >= 0) && (otoc_idx <= (N-2))) {
	    op = ops[otoc_idx].LR;
	}
	s *= op;
	this->evolve_step(hRR, s, t, beta/4);
	this->evolve_step(hLL, s, -t, beta/4);
	s *= op;
	this->evolve_step(hRR, s, t, beta/4);
	this->evolve_step(hLL, s, -t, 0);
    }

public:
    OTOC(const SykArgParser &args) : BaseEval<FpType>("OTOC", args) {}
};

// Helper class which execute the computational tasks represented by
// the evaluator classes. We inherit from SykArgParser so we don't
// have to write args.??? in front of the command line parameters.
class Runner : protected SykArgParser {
    // Compute all disorder realization for the given n-point function
    template<template<typename> typename Eval, typename RandGen>
    void runjob(RandGen &rg) {
	std::stringstream ss;
	ss << "N" << N << "M" << M << "beta" << beta << "k" << sparsity
	   << (regularize ? "reg" : "") << (non_standard_gamma ? "nsgamma" : "")
	   << "tmax" << tmax << "nsteps" << nsteps << "krydim" << krylov_dim;
	if (j_coupling != 1.0) {
	    ss << "J" << j_coupling;
	}
	if (otoc_idx >= 0) {
	    ss << "otocidx" << otoc_idx;
	}
	if (kry_tol != 0.0) {
	    ss << "tol" << kry_tol;
	}
	Eval<float> eval32(*this);
	Eval<double> eval64(*this);
	auto jobname = eval32.name;
	auto outfname = jobname + ss.str();
	auto avgfname = jobname + "Avg" + ss.str();
	std::ofstream outf32, outf64, outfdiff;
	std::ofstream avgf32, avgf64, avgfdiff;
	if (fp32 && fp64) {
	    if (truncate_fp64) {
		outfname += "crstrunc";
		avgfname += "crstrunc";
	    } else {
		outfname += "crsprom";
		avgfname += "crsprom";
	    }
	}
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
	if (fp32 && fp64) {
	    outfdiff.rdbuf()->pubsetbuf(0, 0);
	    outfdiff.open(outfname + "diff");
	    avgfdiff.rdbuf()->pubsetbuf(0, 0);
	    avgfdiff.open(avgfname + "diff");
	}
	std::ofstream hamf;
	if (dump_ham) {
	    hamf.rdbuf()->pubsetbuf(0, 0);
	    hamf.open(std::string("Ham") + outfname);
	}
	std::vector<complex<float>> v32(nsteps+1);
	std::vector<complex<double>> v64(nsteps+1);
	std::vector<double> vdiff(nsteps+1);
	std::vector<complex<float>> sum32(nsteps+1);
	std::vector<complex<double>> sum64(nsteps+1);
	std::vector<double> sumdiff(nsteps+1);
	std::ofstream logf;
	logf.rdbuf()->pubsetbuf(0, 0);
	logf.open(std::string("Log") + jobname + ss.str());
	Logger logger(verbose, logf);
	logger << "Running " << jobname << " calculation using build "
	       << GITHASH << ".\nParameters are: N " << N
	       << " M " << M << " beta " << beta << " k " << sparsity
	       << (regularize ? " regularized" : "")
	       << " tmax " << tmax << " nsteps " << nsteps
	       << " krydim " << krylov_dim << " "
	       << (use_mt19937 ? "MT19937" : "PCG64") << endl;
	std::unique_ptr<State<float>> init32, s32;
	std::unique_ptr<State<double>> init64, s64;
	if (fp32) {
	    init32 = std::make_unique<State<float>>(N/2-1);
	    s32 = std::make_unique<State<float>>(N/2-1);
	}
	if (fp64) {
	    init64 = std::make_unique<State<double>>(N/2-1);
	    s64 = std::make_unique<State<double>>(N/2-1);
	}
	SYK<float> syk32(N, sparsity, j_coupling, non_standard_gamma);
	SYK<double> syk64(N, sparsity, j_coupling, non_standard_gamma);
	HamOp<float> ham32;
	HamOp<double> ham64;
	double dt = tmax / nsteps;
	for (int u = 0; u < M; u++) {
	    logger << "Computing disorder " << u << endl;
	    if (!fp32 || truncate_fp64) {
		init64->random_state();
		ham64 = syk64.gen_ham(rg, regularize);
	    } else {
		init32->random_state();
		ham32 = syk32.gen_ham(rg, regularize);
	    }
	    if (dump_ham) {
		if (!fp32 || truncate_fp64) {
		    hamf << "H_LL = " << ham64.LL << std::endl;
		    hamf << "H_RR = " << ham64.RR << std::endl;
		} else {
		    hamf << "H_LL = " << ham32.LL << std::endl;
		    hamf << "H_RR = " << ham32.RR << std::endl;
		}
	    }
	    // Compute one single disorder realization.
	    if (fp32) {
		// If fp64 is also requested, we should make sure we evolve the
		// states using the same Hamiltonian and same initial state as
		// fp64. There are two options here. We can either promote
		// the fp32 Hamiltonian and initial states to fp64, or
		// truncating fp64 into fp32. If user specified to truncate,
		// do the truncation now.
		if (fp64 && truncate_fp64) {
		    *init32 = *init64;
		    ham32 = ham64;
		}
		eval32.eval(syk32, ham32, *init32, *s32, v32,
			    outf32, sum32, logger);
	    }
	    if (fp64) {
		// If user specfied both --fp32 and --fp64 but did not specify
		// --truncate-fp64, promote the fp32 Hamiltonian and initial states
		// to fp64 instead.
		if (fp32 && !truncate_fp64) {
		    *init64 = *init32;
		    ham64 = ham32;
		}
		eval64.eval(syk64, ham64, *init64, *s64, v64,
			    outf64, sum64, logger);
	    }
	    if (fp32 && fp64) {
		// If user requested both fp32 and fp64, also compute the difference
		for (int i = 0; i <= nsteps; i++) {
		    vdiff[i] = abs(v64[i] - complex<double>(v32[i]));
		    outfdiff << dt*i << " " << vdiff[i] << std::endl;
		    sumdiff[i] += vdiff[i];
		}
	    }
	}
	for (int i = 0; i <= nsteps; i++) {
	    if (fp32) {
		sum32[i] /= M;
		avgf32 << dt*i << " " << sum32[i].real() << " "
		       << sum32[i].imag() << std::endl;
	    }
	    if (fp64) {
		sum64[i] /= M;
		avgf64 << dt*i << " " << sum64[i].real() << " "
		       << sum64[i].imag() << std::endl;
	    }
	    if (fp32 && fp64) {
		sumdiff[i] /= M;
		avgfdiff << dt*i << " " << sumdiff[i] << std::endl;
	    }
	}
    }

public:
    int run(int argc, const char *argv[]) {
	if (!parse(argc, argv)) {
	    return 1;
	}
	if (use_mt19937) {
	    // The user request for mt19937 as the pRNG. Use it.
	    std::random_device rd;
	    std::mt19937 rg(rd());
	    if (want_otoc) {
		runjob<OTOC>(rg);
	    }
	    if (want_greenfcn) {
		runjob<Green>(rg);
	    }
	} else {
	    // Seed with a real random value, if available
	    pcg_extras::seed_seq_from<std::random_device> seed_source;
	    // Make a random number engine
	    pcg64 rg(seed_source);
	    if (want_otoc) {
		runjob<OTOC>(rg);
	    }
	    if (want_greenfcn) {
		runjob<Green>(rg);
	    }
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
}
