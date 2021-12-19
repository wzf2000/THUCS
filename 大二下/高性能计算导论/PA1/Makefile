CFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Werror -pedantic -DDEBUG

.PHONY: clean check_mpi

all: check_mpi generate odd_even_sort

check_mpi:
ifeq ($(shell which mpicxx),)
	$(error No mpicxx found, please load OpenMPI first!)
endif

generate: generate.cpp
	g++ $(CFLAGS) $^ -o $@ 

odd_even_sort: main.cpp worker.cpp odd_even_sort.cpp
	mpicxx -g $(CFLAGS) $^ -o $@

clean:
	rm -f generate odd_even_sort

