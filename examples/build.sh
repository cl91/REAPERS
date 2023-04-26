GPU=0

if [[ $1 == "gpu" || $2 == "gpu" ]]; then
    GPU=1
fi

GITHASH=$(git show -s --format="%h (%ci)" HEAD)
#DBGOPTS="-DREAPERS_DEBUG -g"
DBGOPTS=""
# You can also specify -DMAX_NUM_FERMIONS=<N> to hard-code the maximum number of fermions.
# This might make things a little faster, but probably won't matter much.
OPTS="$DBGOPTS -Wall -Wno-maybe-uninitialized -Wno-uninitialized -O3 -ffast-math -fopenmp -march=native -mtune=native -I/usr/include/eigen3 -I../inc -lboost_program_options"
NVCC="nvcc -std=c++20 -forward-unknown-to-host-compiler $OPTS -Wno-unknown-pragmas -lcublas -lcurand ../src/krnl.cu"
CXX="c++ -std=c++20 -DREAPERS_NOGPU $OPTS"
PROJS="syk lindblad"

for i in $PROJS; do
    if (( $GPU )); then
	$NVCC -DGITHASH="\"$GITHASH\"" $i.cc  -o $i-gpu
    else
	$CXX -DGITHASH="\"$GITHASH\"" $i.cc -o $i
    fi
done
