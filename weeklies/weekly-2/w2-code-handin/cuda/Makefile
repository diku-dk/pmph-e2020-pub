COMPILER?=nvcc
OPT_FLAGS?=-O3 -arch=compute_35
LDFLAGS?=-lOpenCL

TEST_PBB=test-pbb
TEST_SPMV=test-spmv

.PHONY: clean all run

default: compile run

compile: $(TEST_PBB) $(TEST_SPMV)

$(TEST_PBB): testPBB.cu hostSkel.cu.h pbbKernels.cu.h constants.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_PBB) testPBB.cu

$(TEST_SPMV): spMV-Mul-kernels.cu.h spMV-Mul-main.cu
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_SPMV) spMV-Mul-main.cu

all: $(TEST_PBB) $(TEST_SPMV)


run: $(TEST_PBB) $(TEST_SPMV)
	./$(TEST_PBB) 50003565 1024
	./$(TEST_SPMV) 11033 2076 1024

run-pbb: $(TEST_PBB)
	./$(TEST_PBB) 50003565 128

run-spmv: $(TEST_SPMV)
	./$(TEST_SPMV) 11033 2076 256

clean:
	rm -f $(TEST_PBB) $(TEST_SPMV)




