GPU=1

if [[ $1 == "no-gpu" ]]; then
    GPU=0
fi

OPTS="-std=c++20 -Wall -Wno-maybe-uninitialized -Wno-uninitialized -I/usr/include/eigen3 -I../inc -DREAPERS_DEBUG -fopenmp"
NVCC="nvcc -forward-unknown-to-host-compiler $OPTS -Wno-unknown-pragmas -lcublas -lcurand ../src/krnl.cu"
CXX="c++ -DREAPERS_NOGPU $OPTS"
TESTS="ops"

for i in $TESTS; do
    echo "Building tests $i for CPU..."
    $CXX $i.cc -o $i || exit 1
    echo "Running tests $i for CPU..."
    ./$i
    if [[ $GPU == 1 ]]; then
	echo "Building tests $i for GPU..."
	$NVCC $i.cc  -o $i-gpu || exit 1
	echo "Running tests $i for GPU..."
	./$i-gpu
    fi
done
