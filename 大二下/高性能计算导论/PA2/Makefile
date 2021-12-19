CXXFLAGS ?= -std=c++17 -fopenmp -O2 -g -Wall -Wextra -Werror -pedantic

headers := $(wildcard *.h)
common_src := bfs_common.cpp graph.cpp

.PHONY: clean check_mpi

all: check_mpi bfs_omp bfs_omp_mpi

check_mpi:
ifeq ($(shell which mpicxx),)
	$(error No mpicxx found, please load OpenMPI first!)
endif

bfs_omp: main_omp.cpp bfs_omp.cpp $(common_src)
	g++ $(CXXFLAGS) -o $@ $^
bfs_omp_mpi: main_omp_mpi.cpp bfs_omp_mpi.cpp $(common_src)
	mpicxx $(CXXFLAGS) -o $@ $^

clean:
	rm -rf bfs_omp bfs_omp_mpi *~ *.*~
