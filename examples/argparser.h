/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    argparser.h

Abstract:

    This module file contains the command line parsing code common to the
    example programs in this directory.

Revision History:

    2023-04-11  File created

--*/

#ifdef MAX_NUM_FERMIONS
#if MAX_NUM_FERMIONS > 64
#error "MAX_NUM_FERMIONS cannot be larger than 64"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif
#if MAX_NUM_FERMIONS & 1
#error "MAX_NUM_FERMIONS must be an even integer"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif
#endif

#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <filesystem>

namespace progopts = boost::program_options;

class ArgParser {
    virtual void parse_hook() {}
    virtual bool optcheck_hook() { return true; }

protected:
    ~ArgParser() {}

public:
    progopts::options_description optdesc;
    progopts::variables_map varmap;
    bool verbose;
    bool fp32;
    bool fp64;
    std::string data_dir;
    int N;
    int M;
    int nsteps;
    float sparsity;
    float tmax;
    int krylov_dim;
    bool trace;
    float cache_size_vram;
    float cache_size_gb;

    ArgParser() : optdesc("Allowed options") {}

    bool parse(int argc, const char *argv[]) {
	// Declare the default supported options.
	optdesc.add_options()
	    ("help,h", "Produce this help message.")
	    ("verbose,v", progopts::bool_switch(&verbose)->default_value(false),
	     "Enable loggings of diagnostics.")
	    ("fp32", progopts::bool_switch(&fp32)->default_value(false),
	     "Use single precision floating point.")
	    ("fp64", progopts::bool_switch(&fp64)->default_value(false),
	     "Use double precision floating point.")
	    ("data-dir", progopts::value<std::string>(&data_dir)->default_value("."),
	     "Specify the data output directory.")
	    ("num-fermions,N", progopts::value<int>(&N)->required(),
	     "Specify the number of fermions to simulate.")
	    ("num-realizations,M", progopts::value<int>(&M)->required(),
	     "Specify the number of disorder realizations to simulate.")
	    ("tmax,t", progopts::value<float>(&tmax)->required(),
	     "Specifies the maximum time. Time will be divided into n steps.")
	    ("nsteps,n", progopts::value<int>(&nsteps)->required(),
	     "Specifies the number of grid points in time.")
	    ("sparsity,k", progopts::value<float>(&sparsity)->default_value(0.0),
	     "If non-zero, simulate a sparse SYK with kN connections "
	     "rather than a fully connected dense SYK.")
	    ("krydim", progopts::value<int>(&krylov_dim)->default_value(5),
	     "Specifies the dimension of the Krylov subspace. Default value is 5.")
	    ("trace", progopts::bool_switch(&trace)->default_value(false),
	     "When computing the n-point function or purity, use the full trace"
	     " definition rather than the inner product of initial and final state.")
#ifndef REAPERS_NOGPU
	    ("cache-size-vram",
	     progopts::value<float>(&cache_size_vram)->default_value(0.0),
	     "Set the matexp cache size to be the specified value times the"
	     " amount of available VRAM at program startup. Default is 0.5 (50%).")
#endif
	    ("cache-size-gb",
	     progopts::value<float>(&cache_size_gb)->default_value(0.0),
	     "Set the matexp cache size to be the specified GB. For CPU backend"
	     " the default is 16GB. For GPU backend the default is 50% of"
	     " available VRAM at program startup.");

	// Let our children add their own supported options.
	parse_hook();

	progopts::store(progopts::parse_command_line(argc, argv, optdesc), varmap);

	if (varmap.count("help")) {
	    std::cout << optdesc << std::endl;
	    return false;
	}

	try {
	    progopts::notify(varmap);
	} catch (const boost::program_options::required_option &e) {
	    std::cerr << "Error: " << e.what() << std::endl;
	    std::cerr << optdesc << std::endl;
	    return false;
	}

#ifdef MAX_NUM_FERMIONS
	if (N > MAX_NUM_FERMIONS) {
	    std::cerr << "N cannot be larger than " << MAX_NUM_FERMIONS << std::endl;
	    return false;
	}
#endif

	if (!fp32 && !fp64) {
	    std::cerr << "You must specify --fp32 or --fp64" << std::endl;
	    return false;
	}

	// If user has specified a data directory, switch to it.
	std::filesystem::create_directory(data_dir);
	std::filesystem::current_path(data_dir);

	// If user has specified the matexp cache size, set the cache size.
	if (cache_size_gb != 0.0) {
	    if (cache_size_gb < 0.0) {
		std::cerr << "Cache size in GB must be positive."
			  << std::endl;
		return false;
	    }
	    REAPERS::DefImpl::set_max_cache_size((size_t)(cache_size_gb * 1024) << 20);
	}
#ifndef REAPERS_NOGPU
	if (cache_size_vram != 0.0) {
	    if (cache_size_gb != 0.0) {
		std::cerr << "You cannot specify both --cache-size-gb and "
		    "--cache-size-vram" << std::endl;
		return false;
	    }
	    if (cache_size_vram < 0.0) {
		std::cerr << "Cache size must be positive." << std::endl;
		return false;
	    }
	    REAPERS::GPUImpl::set_max_cache_size(cache_size_vram);
	}
#endif

	// Let our children add their own option checking.
	return optcheck_hook();
    }
};
