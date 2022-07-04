CC := g++
CFLAGS := -O0 -D_GNU_SOURCE
SRC = $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp,%,$(SRC))

all: $(TARGETS)

%: %.cpp lib.h
	$(CC) $< -o $@ $(CFLAGS)
	# ./$@ > log_$@.txt

clean:
	rm $(TARGETS) log_*.txt