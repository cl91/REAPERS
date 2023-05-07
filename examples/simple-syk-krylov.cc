// Demonstrate how easy it is to write a program in REAPERS that computes
// the OTOC of the SYK model using the Krylov algorithm. This program is
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
//   F(t) = <psi| sqrt(rq)*A(t)*rq*B*rq*A(t)*rq*B*sqrt(rq) |psi>
// where rq = e^{-beta H/4} and A(t) = e^{iHt} A e^{-iHt}
// Here A = gamma_{N-1} and B = gamma_{N-2}
void otoc_krylov(int N, double beta, double tmax, int nsteps)
{
    std::cerr << "Computing OTOC using Krylov method, N = " << N
	      << " beta = " << beta << std::endl;
    // Set up random number generator
    std::random_device rd;
    std::mt19937 rg(rd());
    // Define dense SYK model with N fermions
    SYK syk(N);
    // Generate one disorder realization and obtain its Hamiltonian.
    auto ham = syk.gen_ham(rg);
    // Define the operators we want for the OTOC.
    auto opA = syk.fermion_ops(N-1);
    auto opB = syk.fermion_ops(N-2);
    // Define the states we need. Here N/2 is the spin chain length.
    State init(N/2), psi(N/2);
    // Generate a random Haar-state for the initial state.
    init.random_state();
    // Evolve the initial state using the Krylov algorithm.
    init.evolve(ham, 0.0, beta/8);
    // We compute t = 0..tmax with nsteps
    std::vector<complex<double>> v(nsteps+1);
    for (int i = 0; i <= nsteps; i++) {
	auto t = tmax * i / nsteps;
	// Save the (evolved) initial state because we need it to
	// compute the inner product later
	psi = init;
	// Act on the state psi with B (ie. compute psi = B * psi).
	psi *= opB;
	// Evolve the state psi further
	psi.evolve(ham, t, beta/4);
	psi *= opA;
	psi.evolve(ham, -t, beta/4);
	psi *= opB;
	psi.evolve(ham, t, beta/4);
	psi *= opA;
	psi.evolve(ham, -t, 0.0);
	// Compute the inner product between psi and init
	v[i] = init * psi;
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
	std::cout << "usage: ./simple-syk-krylov N beta tmax nsteps" << std::endl;
	return 1;
    }
    int N = atoi(argv[1]);
    double beta = atof(argv[2]);
    double tmax = atof(argv[3]);
    int nsteps = atoi(argv[4]);
    otoc_krylov(N, beta, tmax, nsteps);
}
