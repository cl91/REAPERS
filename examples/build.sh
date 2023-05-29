GPU=0
INTEL=0

if [[ $1 == "gpu" ]]; then
    GPU=1
fi

if [[ $1 == "intel" ]]; then
    INTEL=1
fi

# Use the Intel LLVM-based C++ compiler as the host compiler.
# This is the fastest mode.
if [[ $1 == "gpu-icpx" ]]; then
    GPU_ICPX=1
fi

GITHASH=$(git show -s --format="%h (%ci)" HEAD)
COMMONOPTS="-std=c++20 -Wall -Wno-maybe-uninitialized -Wno-uninitialized -I/usr/include/eigen3 -I../inc"
LINKLIBS="-lboost_program_options"
DBGOPTS="-DREAPERS_DEBUG -g"
NDBGOPTS="-DNDEBUG -O3 -ffast-math -mtune=native"
# You can also specify -DMAX_NUM_FERMIONS=<N> to hard-code the maximum number of fermions.
# This might make things a little faster, but probably won't matter much.
NVCC="nvcc -forward-unknown-to-host-compiler $COMMONOPTS $NDBGOPTS -Wno-unknown-pragmas -lcublas -lcusolver -lcurand ../src/krnl.cu -march=native -fopenmp $LINKLIBS"
CXX="c++ -DREAPERS_NOGPU $COMMONOPTS $DBGOPTS -march=native -fopenmp $LINKLIBS"
ICXX="icpx -DREAPERS_NOGPU $COMMONOPTS $NDBGOPTS -xhost -fiopenmp -Wno-tautological-constant-compare -Wno-unused-but-set-variable $LINKLIBS"
ICXX_BUILD="icpx $COMMONOPTS $NDBGOPTS -fPIE -xhost -fiopenmp -Wno-tautological-constant-compare -Wno-unused-but-set-variable -Wno-unknown-pragmas -I/opt/cuda/include"
ICXX_LIB_PATH="$(dirname $(which icpx))/../compiler/lib/intel64_lin"
INTEL_LIBS="irc iomp5 imf svml"
NVCC_ICXX_OPTS=""
for lib in $INTEL_LIBS; do
    NVCC_ICXX_OPTS+=" $ICXX_LIB_PATH/lib${lib}.a"
done
PROJS="simple-syk-exdiag simple-syk-krylov syk lindblad"

for i in $PROJS; do
    if (( $GPU )); then
	$NVCC -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-gpu
    elif (( $GPU_ICPX )); then
	$ICXX_BUILD -DGITHASH="\"$GITHASH\"" $i.cc  -c -o /tmp/$i-icpx.o
	$NVCC $NVCC_ICXX_OPTS /tmp/$i-icpx.o  -o $i-gpu-icpx
    elif (( $INTEL )); then
	$ICXX -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-intel
    else
	$CXX -DGITHASH="\"$GITHASH\"" $i.cc -o $i-debug
    fi
done
