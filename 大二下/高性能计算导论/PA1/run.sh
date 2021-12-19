#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
srun -N 2 -n 56 --cpu-bind sockets $*

