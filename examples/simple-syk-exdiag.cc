// Demonstrate how easy it is to write a program in REAPERS that computes
// the OTOC of the SYK model using exact diagonalization. This program is
// optimized for the number of lines of code. For the one optimized for
// feature and speed, see syk.cc.

#include <iostream>
#include <random>
#define REAPERS_FP64
#include <reapers.h>

using namespace REAPERS;
using namespace REAPERS::Model;
using Matrix = SumOps<>::MatrixType;

// Compute the OTOC using the following definition:
//     F(t) = tr (rq*A(t)*rq*B*rq*A(t)*rq*B)
// where rq = e^{-beta H/4} and A(t) = e^{iHt} A e^{-iHt}
// Here A = gamma_{N-1} and B = gamma_{N-2}
void otoc(int N, double beta, double tmax, int nsteps)
{
    std::cerr << "Computing OTOC using trace method, N = " << N
	      << " beta = " << beta << std::endl;
    // Set up random number generator
    std::random_device rd;
    std::mt19937 rg(rd());
    // Define dense SYK model with N fermions
    SYK syk(N);
    // Generate one disorder realization and obtain its Hamiltonian.
    auto ham = syk.gen_ham(rg);
    // Define the operators we want for the OTOC.
    Matrix A = syk.fermion_ops(N-1);
    Matrix B = syk.fermion_ops(N-2);
    // Compute exp(-beta H/4). Here N/2 is the length of the spin chain.
    Matrix rq = ham.matexp(-beta/4, N/2);
    // We compute t = 0..tmax with nsteps
    std::vector<complex<double>> v(nsteps+1);
    for (int i = 0; i <= nsteps; i++) {
	auto t = tmax * i / nsteps;
	Matrix Ax = A * ham.matexp({0,-t}, N/2);
	Matrix At = ham.matexp({0,t}, N/2) * Ax;
	Matrix twopt = rq * At * rq * B;
	v[i] = (twopt * twopt).trace();
    }
    // Normalize the result by dividing with v[0]
    auto v0 = v[0];
    for (int i = 0; i <= nsteps; i++) {
	v[i] /= v0;
	std::cout << tmax*i/nsteps << " " << v[i].real()
		  << " " << v[i].imag() << std::endl;
    }
}

int main(int argc, char **argv)
{
    if (argc != 5) {
	std::cout << "usage: ./simple-syk-exdiag N beta tmax nsteps" << std::endl;
	return 1;
    }
    int N = atoi(argv[1]);
    double beta = atof(argv[2]);
    double tmax = atof(argv[3]);
    int nsteps = atoi(argv[4]);
    otoc(N, beta, tmax, nsteps);
}
