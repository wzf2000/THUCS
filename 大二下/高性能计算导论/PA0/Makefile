.PHONY: clean

all: openmp_pow mpi_pow

openmp_pow: openmp_pow.cpp
	g++ $^ -O3 -std=c++11 -fopenmp -o $@

mpi_pow: mpi_pow.cpp
	mpicxx $^ -O3 -std=c++11 -o $@

clean:
	rm mpi_pow openmp_pow
