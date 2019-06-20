P=viterbiMPI
R= mpiv
INC=../common/inc
NVCCFLAGS=-I$(INC)
CCFLAGS=-g -Wall
OMPFLAG=-fopenmp
CC=gcc
MPICC=mpiCC
NVCC=nvcc

all: $(P)

$(P): $(P).cpp
	$(MPICC) $(P).cpp -o $(R)

clean:
	rm $(P)
