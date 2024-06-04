# REAPERS: a REAsonably PERformant Simulator for qubit systems

REAPERS is a C++ library for simulating qubit systems (ie. quantum spin-1/2 chains) on CPUs
and nVidia GPUs. It provides an intuitive, ergonomic interface similar to that of
[dynamite](https://github.com/GregDMeyer/dynamite), which allows the user to construct
spin-1/2 chain Hamiltonians and simulate their time evolutions using the Krylov subspace
method. We provide optimized CUDA kernels for computing the actions of spin operators on
quantum states, on top of which we have implemented the restarted Krylov subspace method that
computes the state time evolution, ie. action of exp(-iHt) on a quantum state. The Krylov
algorithm implemented here is the exact same restarted Krylov algorithm provided by SLEPc.
However, instead of relying on PETSc/SLEPc as does dynamite, we implement (almost) everything
on our own --- the only external dependency (apart from the C++ standard library and the
CUDA libraries) is eigen3 which is used for host (ie. CPU) matrix operations. This provides
a smoother user experience by not having to configure and build PETSc/SLEPc, and allows
us to achieve a higher performance by eliminating the various overheads due to the ancient
and slow PETSc library.

Using REAPERS we [2] have successfully calculated the out-of-time-ordered correlators (OTOC)
of the sparse SYK model with N=64 fermions on a single A100 graphics card for single precision
floating point and N=62 fermions for double precision floating point, in a reasonable amount of
time (typically less than one hour per disorder realization per data point). 
The previous state of the art was a similar calculation of N=50 dense SYK on a single V100 [4],
which we can also reproduce. For the list of publications using REAPERS, see the Publications
section below.

## Why this stupid name?
I'm a massive fan of Mass Effect (well only the first one though. ME2 and ME3 are trash. Don't
even get me started on Andromeda).

## Features

1. Optimized CUDA kernels for computing the action of spin operators on quantum states.
   Spin operators by default are represented in a matrix-free manner, without constructing
   the matrix of the spin operator itself, so the maximum spin chain length is only
   constrained by the dimension of the Hilbert space (rather than the square of the Hilbert
   space dimension).
2. Optimized Krylov subspace method implementations that minimize (video) memory
   allocation overhead. C++11 move semantics is implemented to avoid unnecessary copies.
3. Can switch between CPU and GPU backends at runtime --- in fact you can run both at
   the same time, maximizing the utility of computational resources.
4. Supports single-node multi-GPU computing. This requires GPU peer-to-peer access being
   available.
5. Supports AMDGPU with ROCm. This requires at least ROCm 6.2.0.
6. Supports both single precision floating point (FP32) and double precision (FP64), and
   can also be switched at runtime. This allows you to run simulations on consumer-grade
   nVidia cards, as they usually have reduced FP64 performance but decent FP32 performance.
7. Easy to use. No need to build or install as we are a header-only library for the host
   (ie. CPU) code. There is one source file for the CUDA kernels which you simply link
   with after you've built the host code (you can also do it in one single step, by using
   `nvcc` to build both the host code and CUDA kernels).
8. No external dependency apart from eigen3 (which is a header-only library as well and
   requires no configuring or building). Simply install a C++ compiler (and CUDA if you
   want to use GPU) and call `g++` or `nvcc`. You don't have to deal with the mess that
   is C++ package management at all.
9. In addition to Krylov, you can also use what we call "exact diagonalization" (this
   term may mean different things to different people, but we stick with the following
   sense) to compute exp(-iHt), by finding the eigensystem of H and exponentiating the
   eigenvalues. Since we need to store the explicit matrix of the Hamiltonian in this
   case, the spin chain length is constrained by the square of the dimension of the
   Hilbert space.

## Quick Start

Make sure you have a recent C++ compiler installed. We need the concepts feature in C++20
so for GCC you will need at least G++ 10, and for CLANG you will need at least CLANG 10.
Make sure you have eigen3 installed, and optionally the boost library if you want to run
the `syk.cc` and `lindblad.cc` sample programs shipped with the repository. If you want
to use GPU, make sure you have the latest CUDA (as of this writing, CUDA 12) installed.
We need the new APIs in CUDA 12 so versions prior to CUDA 12 will not work. There is no
need to build or install this library as it consists of a header-only library for the host
code (under the `inc` folder) and a single CUDA `krnl.cu` file with the CUDA kernels (under
the `src` folder). If you are not planning to use GPUs, then you don't need to worry about
the `.cu` file at all. If you are going to use GPUs, simply link your host code with
`krnl.cu` using `nvcc` (see `examples/build.sh` for how to do this).

Have a look at the sample programs under the `examples` folder, as well as the `build.sh`
script for the compiler directives. To build all the sample programs, issue
```
./build.sh [opt|intel|nvgpu|nvgpu-icpx|amdgpu] [multi]
```
under the `examples` folder. The default build target is debug build which has extra checks
for program correctness. It is recommended that you test the debug builds first when writing
new programs using REAPERS. If you specify `opt`, then optimized executables will be generated.
If you specify `intel`, the build script will generate Intel AVX-enabled executables using
the Intel LLVM-based C++ compiler `icpx`, which must be available in the `PATH`. This is in
general the fastest CPU-only builds (assuming you are using Intel CPUs). If you specify
`nvgpu`, the build script will use the CUDA `nvcc` to build both the CUDA kernels as well as
the host code (ie. CPU-side of the programs). If you specify `nvgpu-icpx`, the host code will
instead be compiled using the Intel LLVM C++ compiler and this is generally speaking a lot
faster than stock GCC (which `nvcc` calls by default). Optimizations are enabled for both
the host code and the CUDA kernels. Note if you installed eigen3 or cuda in a different
place, you need to modify the `EIGEN_INC` and `CUDA_INC` paths in `build.sh`.

For AMD ROCm, you can specify `amdgpu` which will invoke `hipcc` to build the programs. For
both nVidia and AMD platforms, you can specify an additional paramater `multi` which will
enable multi-GPU support. Multi-GPU support has a small overhead on single GPU systems, so
it is disabled by default.

Once you have built the sample programs, try running them. The SYK program will compute the
Green's functions and OTOCs of dense or sparse SYK models. Use `./syk --help` to see the list
of options accepted by the program. Likewise, the Lindblad program will compute the
entanglement entropies for the dense or sparse SYK models using the Monte Carlo algorithm
described in [3]. The program options are also available with `./lindblad --help`.

If you have successfully run the sample programs (try plotting the results of the OTOC or
Green's functions to see if they match the well-known analytic results), you can then start
reading the API manual, which contains step-by-step tutorials on how to write quantum
simulation programs using REAPERS. The sample program directory contains two simpler SYK
examples, `simple-syk-exdiag` and `simple-syk-krylov`, that computes the SYK OTOC using the
exact diagonalization algorithm and the Krylov subspace method.

## Citing Us

First of all, thank you for your interest. To cite us in a paper, use the following format
`[INSERT-CPC-PROG-LIB-PAPER]`.

## Publications

- [1]. `[INSERT-CPC-PROG-LIB-PAPER]`
- [2]. Sparsity independent Lyapunov exponent in the Sachdev-Ye-Kitaev model, García-García et al,
       [arXiv:2311.00639](https://arxiv.org/abs/2311.00639)
- [3]. Entanglement Transition and Replica Wormhole in the Dissipative
       Sachdev-Ye-Kitaev Model, Wang et al, [arXiv:2306.12571](https://arxiv.org/abs/2306.12571)

## Related Work

- [4]. Many-Body Chaos in the Sachdev-Ye-Kitaev Model, Kobrin et al,
       Phys. Rev. Lett. 126, 030602, [arXiv:2002.05725](https://arxiv.org/abs/2002.05725)

## Manual (including Tutorial, API Manual, and Debugging Tips)
See `docs/manual.md`.
