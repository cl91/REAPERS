GPU=0
INTEL=0

if [[ $1 == "gpu" ]]; then
    GPU=1
fi

if [[ $1 == "intel" ]]; then
    INTEL=1
fi

GITHASH=$(git show -s --format="%h (%ci)" HEAD)
COMMONOPTS="-std=c++20 -Wall -Wno-maybe-uninitialized -Wno-uninitialized -I/usr/include/eigen3 -I../inc -lboost_program_options"
DBGOPTS="-DREAPERS_DEBUG -g"
NDBGOPTS="-DNDEBUG -O3 -ffast-math -mtune=native"
# You can also specify -DMAX_NUM_FERMIONS=<N> to hard-code the maximum number of fermions.
# This might make things a little faster, but probably won't matter much.
NVCC="nvcc -forward-unknown-to-host-compiler $COMMONOPTS $NDBGOPTS -march=native -fopenmp -Wno-unknown-pragmas -lcublas -lcurand ../src/krnl.cu"
CXX="c++ -DREAPERS_NOGPU $COMMONOPTS $DBGOPTS -march=native -fopenmp"
ICXX="icpx -DREAPERS_NOGPU $COMMONOPTS $NDBGOPTS -xhost -fiopenmp -Wno-tautological-constant-compare -Wno-unused-but-set-variable"
PROJS="simple-syk-exdiag simple-syk-krylov syk lindblad"

for i in $PROJS; do
    if (( $GPU )); then
	$NVCC -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-gpu
    elif (( $INTEL )); then
	$ICXX -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-intel
    else
	$CXX -DGITHASH="\"$GITHASH\"" $i.cc -o $i-debug
    fi
done
