GPU=0

if [[ $1 == "gpu" ]]; then
    GPU=1
fi

OPTS="-std=c++20 -Wall -Wno-maybe-uninitialized -Wno-uninitialized -I/usr/include/eigen3 -I../inc -DREAPERS_DEBUG -fopenmp"
NVCC="nvcc -forward-unknown-to-host-compiler $OPTS -Wno-unknown-pragmas -lcublas -lcurand ../src/krnl.cu"
CXX="c++ -DREAPERS_NOGPU $OPTS"
TESTS="ops"

for i in $TESTS; do
    if [[ $GPU == 1 ]]; then
	echo "Building tests $i for GPU..."
	$NVCC $i.cc  -o $i-gpu || exit 1
	echo "Running tests $i for GPU..."
	./$i-gpu
    else
	echo "Building tests $i for CPU..."
	$CXX $i.cc -o $i || exit 1
	echo "Running tests $i for CPU..."
	./$i
    fi
done
