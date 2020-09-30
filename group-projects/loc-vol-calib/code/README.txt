To compile and run:
    $ make clean; make; make run_small
                                _medium
                                _large

Folder `OrigImpl' contains the original implementation:
    -- `ProjectMain.cpp'   contains the main function
    -- `ProjCoreOrig.cpp'  contains the core functions 
                                (to parallelize)
    -- `ProjHelperFun.cpp' contains the functions that compute
                                the input parameters, and 
                                (can be parallelize as well)

Folder `include' contains
    -- `ParserC.h'     implements a simple parser
    -- `ParseInput.h'  reads the input/output data
                        and provides validation.
    -- `OpenmpUtil.h'  some OpenMP-related helpers.        
    -- `Constants.h'   currently only sets up REAL
                        to either double or float
                        based on the compile-time
                        parameter WITH_FLOATS.

    -- `CudaUtilProj.cu.h' provides stubs for calling
                        transposition and inclusive 
                        (segmented) scan.
    -- `TestCudaUtil.cu'  a simple tester for 
                        transposition and scan.

Folder `ParTridagCuda` contains code that demonstrates how TRIDAG can be parallelized by intra-block scans, i.e., it assumes that NUM_X, NUM_Y are a power of 2 less than or equal to 1024.
