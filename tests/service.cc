// Perform the computations specified in the standard input (as a JSON
// object) and return the result as JSON.

#define REAPERS_FP64
#include <reapers.h>
#include <ctime>
#include <iostream>
#include <sstream>
#include <vector>
#include "json.h"

using namespace REAPERS;
using namespace REAPERS::Model;
using namespace REAPERS::EvolutionAlgorithm;
using json = nlohmann::json;
using EigenVals = SumOps<>::EigenVals;
using MatrixType = SumOps<>::MatrixType;
using EigenVecs = SumOps<>::EigenVecs;

json to_json(const State<> &s) {
    json j;
    auto len = s.spin_chain_length();
    size_t dim = 1ULL << len;
    std::vector<std::tuple<double,double>> vec(dim);
    for (size_t i = 0; i < dim; i++) {
	auto c = s[i];
	vec[i] = {c.real(), c.imag()};
    }
    j["len"] = s.spin_chain_length();
    j["vec"] = vec;
    return j;
}

State<> state_from_json(const json &j) {
    int len;
    j.at("len").get_to(len);
    State st(len);
    size_t i = 0;
    for (auto v : j.at("vec")) {
	double re = v.at("real").get<double>();
	double im = v.at("imag").get<double>();
	st[i++] = complex<>{re, im};
    }
    return st;
}

json to_json(const SpinOp<> &op) {
    json j;
    j["bits"] = op.get_bit_string();
    j["coeff"] = op.coefficient().real();
    return j;
}

json to_json(const SumOps<> &ops) {
    json j = json::array();
    for (auto op : ops) {
	j.emplace_back(to_json(op));
    }
    return j;
}

// Serialize matrix in row-major order
json to_json(const MatrixType &mat) {
    std::vector<std::tuple<double,double>> v(mat.size());
    for (int row = 0; row < mat.rows(); row++) {
	for (int col = 0; col < mat.cols(); col++) {
	    complex<> c = mat(row, col);
	    v[row*mat.cols()+col] = {c.real(), c.imag()};
	}
    }
    return json(v);
}

struct GenHamRequest {
    int N;
    double sparsity;
    bool regularize;
    GenHamRequest(const json &j) {
	N = j.at("N").get<int>();
	sparsity = j.at("sparsity").get<int>();
	regularize = j.at("regularize").get<bool>();
    }
};

struct GenHamResponse {
    int N;
    HamOp<> ham;
    std::string hamstr;
    GenHamResponse(const GenHamRequest &req) : N(req.N) {
	SYK syk(N, req.sparsity);
	std::random_device rd;
	std::mt19937 rg(rd());
	ham = syk.gen_ham(rg, req.regularize);
	std::stringstream ss;
	ss << ham;
	hamstr = ss.str();
    }
    json to_json() const {
	json js;
	js["N"] = N;
	js["ham"] = ::to_json(ham);
	js["hamstr"] = hamstr;
	js["mat"] = ::to_json(ham.get_matrix(N/2));
	return js;
    }
};

struct GetEigenSysResponse : GenHamResponse {
    EigenVals evals;
    EigenVecs evecs;
    GetEigenSysResponse(const GenHamRequest &req) : GenHamResponse(req) {
	std::tie(evals,evecs) = ham.get_eigensystem(req.N/2);
    }
    json to_json() const {
	json js = GenHamResponse::to_json();
	js["eigenvals"] = std::vector<double>(evals.data(),
					      evals.data() + evals.size());
	js["eigenvecs"] = ::to_json(evecs);
	return js;
    }
};

struct EvolveStateRequest : GenHamRequest {
    double t;
    double beta;
    bool exdiag;
    EvolveStateRequest(const json &js) : GenHamRequest(js) {
	t = js.at("t").get<double>();
	beta = js.at("beta").get<double>();
	exdiag = js.at("exdiag").get<bool>();
    }
};

struct EvolveStateResponse : GenHamResponse {
    State<> inist;
    State<> finst;
    EvolveStateResponse(const EvolveStateRequest &req)
	: GenHamResponse(req), inist(req.N/2), finst(req.N/2) {
	inist.random_state();
	finst = inist;
	if (req.exdiag) {
	    finst.evolve<ExactDiagonalization>(ham, req.t, req.beta);
	} else {
	    finst.evolve(ham, req.t, req.beta);
	}
    }
    json to_json() const {
	json js = GenHamResponse::to_json();
	js["init-state"] = ::to_json(inist);
	js["final-state"] = ::to_json(finst);
	return js;
    }
};

void run() {
    json jsn;
    std::cin >> jsn;
    auto req = jsn.at("request").get<std::string>();
    json params = jsn.at("params");
    json reply;
    if (req == "gen-ham") {
	reply = GenHamResponse(GenHamRequest(params)).to_json();
    } else if (req == "get-eigensys") {
	// get-eigensys shares the same request params as gen-ham
	reply = GetEigenSysResponse(GenHamRequest(params)).to_json();
    } else if (req == "evolve-state") {
	reply = EvolveStateResponse(EvolveStateRequest(params)).to_json();
    } else {
	throw std::runtime_error(std::string("Unknown request ") + req);
    }
    std::cout << reply << std::endl;
}

int main()
{
    try {
	run();
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
