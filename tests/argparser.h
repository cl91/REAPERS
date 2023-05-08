/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    argparser.h

Abstract:

    This module file contains the command line parsing code common to the
    test programs in this directory.

Revision History:

    2023-05-08  File created

--*/

#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

namespace progopts = boost::program_options;

class BaseArgParser {
    virtual void parse_hook() {}
    virtual bool optcheck_hook() { return true; }

protected:
    ~BaseArgParser() {}

public:
    progopts::options_description optdesc;
    progopts::variables_map varmap;

    BaseArgParser() : optdesc("Allowed options") {}

    bool parse(int argc, const char *argv[]) {
	// Declare the default supported options.
	optdesc.add_options()("help,h", "Produce this help message.");

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

	// Let our children add their own option checking.
	return optcheck_hook();
    }
};
