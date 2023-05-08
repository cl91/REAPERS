#!/bin/bash

OPTS=

if [[ $1 == "gpu" ]]; then
    OPTS="--gpu"
fi

if [[ $1 == "intel" ]]; then
    OPTS="--intel"
fi

betas="0.005 0.05 0.5 1.0 5.0 10.0 20.0 30.0"

for N in $(seq 12 2 22); do
    for t in $(seq 1.0 1.0 20.0); do
	for beta in $betas; do
	    for M in $(seq 20); do
		./test-suite.py $OPTS -n --run-funclet evolve-state $N $t $beta
	    done
	done
    done
done
