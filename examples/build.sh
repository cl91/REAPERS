EIGEN_INC=/usr/include/eigen3
CUDA_INC=/opt/cuda/include

OPT=0
NVGPU=0
AMDGPU=0
MULTI_GPU=0
INTEL=0
buildty="debug"

if [[ $1 == "opt" ]]; then
    OPT=1
    buildty="opt"
fi

if [[ $1 == "nvgpu" ]]; then
    NVGPU=1
    buildty="nvgpu"
fi

if [[ $1 == "amdgpu" ]]; then
    AMDGPU=1
    buildty="amdgpu"
fi

if [[ $1 == "intel" ]]; then
    INTEL=1
    buildty="intel"
fi

# Use the Intel LLVM-based C++ compiler as the host compiler.
# This may be faster than stock GCC.
if [[ $1 == "nvgpu-icpx" ]]; then
    NVGPU_ICPX=1
    buildty="nvgpu-icpx"
fi

if [[ $2 == "multi" ]]; then
    MULTI_GPU=1
    buildty+="-multi"
fi

# Switch current directory to where this script is located.
cd "$(dirname "$0")"
GITHASH=$(git show -s --format="%h (%ci)" HEAD)
COMMONOPTS="-std=c++20 -Wall -Wno-uninitialized -I${EIGEN_INC} -I../inc"
LINKLIBS="-lboost_program_options"
DBGOPTS="-DREAPERS_DEBUG -g"
NDBGOPTS="-DNDEBUG -O3 -mtune=native"
if (( $AMDGPU )); then
    COMMONOPTS+=" -Wno-unused-but-set-variable -Wno-unused-result"
else
    COMMONOPTS+=" -Wno-maybe-uninitialized"
fi
if (( $OPT )) || (( $INTEL )) || (( $NVGPU )) || (( $AMDGPU )) || (( $NVGPU_ICPX )); then
    COMMONOPTS+=" $NDBGOPTS"
else
    COMMONOPTS+=" $DBGOPTS"
fi
if (( $MULTI_GPU )); then
    COMMONOPTS+=" -DREAPERS_MULTIGPU"
fi
# You can also specify -DMAX_NUM_FERMIONS=<N> to hard-code the maximum number of fermions.
# This might make things a little faster, but probably won't matter much.
NVCC="nvcc -forward-unknown-to-host-compiler $COMMONOPTS -Wno-unknown-pragmas -diag-suppress 68 -lcublas -lcusolver -lcurand ../src/krnl.cu -march=native -fopenmp $LINKLIBS -arch sm_80"
HIPCC="hipcc $COMMONOPTS -D__HIPCC__ -lhipblas -lhipsolver -lhiprand ../src/krnl.cu -march=native -fopenmp $LINKLIBS"
CXX="c++ -DREAPERS_NOGPU $COMMONOPTS -march=native -fopenmp $LINKLIBS"
ICXX="icpx -DREAPERS_NOGPU $COMMONOPTS -xhost -fiopenmp -Wno-tautological-constant-compare -Wno-unused-but-set-variable $LINKLIBS"
ICXX_BUILD="icpx $COMMONOPTS -fPIE -xhost -fiopenmp -Wno-tautological-constant-compare -Wno-unused-but-set-variable -Wno-unknown-pragmas -I/opt/cuda/include"
if (( $NVGPU_ICPX )); then
    ICXX_LIB_PATH="$(dirname $(which icpx))/../compiler/lib/intel64_lin"
    INTEL_LIBS="irc iomp5 imf svml"
    NVCC_ICXX_OPTS=""
    for lib in $INTEL_LIBS; do
	NVCC_ICXX_OPTS+=" $ICXX_LIB_PATH/lib${lib}.a"
    done
fi

PROJS="simple-syk-exdiag simple-syk-krylov syk lindblad"

for i in $PROJS; do
    if (( $NVGPU )); then
	$NVCC -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-$buildty
    elif (( $AMDGPU )); then
	$HIPCC -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-$buildty
    elif (( $NVGPU_ICPX )); then
	$ICXX_BUILD -DGITHASH="\"$GITHASH\"" $i.cc  -c -o /tmp/$i-icpx.o &&
	$NVCC $NVCC_ICXX_OPTS /tmp/$i-icpx.o  -o $i-$buildty
    elif (( $INTEL )); then
	$ICXX -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-$buildty
    else
	$CXX -DGITHASH="\"$GITHASH\"" $i.cc -o $i-$buildty
    fi
done
