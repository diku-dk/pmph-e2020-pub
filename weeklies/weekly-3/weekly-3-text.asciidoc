= Third Weekly Assignment for the PMPH Course

This is the text of the third weekly assignment for the DIKU course
"Programming Massively Parallel Hardware", 2020-2021.

Hand in your solution in the form of a short report in text or PDF
format, along with the missing code.   We hand-in incomplete code in 
the archive `wa3-code.zip`.   You are supposed to fill in the missing
code such that all the tests are valid and to report performance 
results. Please send back the same files under the same structure that
was handed in---implement the missing parts directly in the provided files.
There are comments in the source files that are supposed to guide you
(together with the text of this assignment).

Unziping the handed in archive `wa3-code.zip` will create the `w3-code-handin`
folder, which contain:

* A `Makefile` that by default compiles and runs all programs, but the
    built-in validation will fail because some of the implementation is
    missing
* Files `transpose-main.cu`, `transpose-host.cu.h` and `transpose-kernels.cu.h`
    contain the full implementation of matrix transposition and of the original
    program of task `3`. You need to fill in files `transpose-main.cu` and
    `transpose-kernels.cu.h` the CPU orchestrating call and the GPU kernel that
    implements the transformed program in which spatial locality is optimized,
    i.e., it features only coalesced accesses to global memory.

* Files `mmm-main.cu` and `mmm-kernels.cu.h` contain the naive and the 
    block-tiled implementation of (dense) matrix-matrix multiplication.
    To solve task 4, please fill in files `mmm-main.cu` and `mmm-kernels.cu.h` 
    the CPU orchestration code and the GPU kernels that perform a combination
    of register and block tiling.

Write a neat and short report containing the solutions to the first two theoretical
questions, and also the missing code and short explanations for tasks 3 and 4.
Also provide comments regarding the performance behaviur of your programs:

* why do the programs that you have to implement for Task 3 have better performance
  than the original (provided) GPU program? Why does the optimized program---in
  which transpositions are inlined---also outperformed the program in which 
  transpositions are manifested/explicit?

* For task 4, what is the speedup over block-tiled matrix multiplication and
    how do you justify the speedup?

* Try to run your code on both gpu04 and gpu02, since you might be surprised
  to see that the impact of your optimizations is sensitive to the hardware.
    
== Task 1: Pen and Paper Exercise Aimed at Applying Dependency-Analysis Transformations (2 pts)

Consider the C-like pseudocode below:

----
float A[2*M];

for (int i = 0; i < N; i++) {
    A[0] = N;

    for (int k = 1; k < 2*M; k++) {
        A[k] = sqrt(A[k-1] * i * k);
    }

    for (int j = 0; j < M; j++) {
        B[i+1, j+1] = B[i, j] * A[2*j  ];
        C[i,   j+1] = C[i, j] * A[2*j+1];
    }
}
----

Your task is to apply privatization, array expansion, loop distribution 
and loop interchange in order to parallelize as many loops as possible.
Answer the following in your report:

* Explain why in the (original) code above *neither* the outer loop (of index `i`)
    *nor* the inner loops (of indices `k` and `j`) *are parallel*;
* Explain why is it safe to privatize array `A`;
* Once privatized, explain why is it safe to distribute the outermost loop across the 
    `A[0] = N;` statement and across the other two inner loops.
    Perform (safely!) the loop distribution, while remembering to perform
    array expansion for `A`.
* Now you can use direction vectors to determine which loops in the
    resulted three loop nests are parallel. Please explain and 
    annotate each loop with the comment `// parallel` or `// sequential`.
* Can loop interchange be applied so that each loop nest contains one
    parallel loop?  Please explain why (or why not) and show the
    code exhibiting maximum parallelism --- please annotate each loop
    with the comment `// parallel` or `// sequential`.
 

== Task 2: Pen and Paper Exercise Aimed at Recognizing Parallel Operators (2 pts)

Assume that both A and B are matrices with N rows and 64 columns. Consider the pseudocode below:

----
float A[N,64];
float B[N,64];
float accum, tmpA;
for (int i = 0; i < N; i++) { // outer loop
    accum = 0;
    for (int j = 0; j < 64; j++) { // inner loop
        tmpA = A[i, j];
        accum = sqrt(accum) + tmpA*tmpA; // (**)
        B[i,j] = accum;
    }
}
----

Reason about the loop-level parallelism of the code above and answer the following in your report:

* Why is the outer loop *not* parallel? 
* What technique can be used to make it parallel and why is it safe to apply it? 
  Re-write the code such that the outer loop is parallel, 
        i.e., the outer loop does not carry any dependencies.
* Explain why the inner loop is *not* parallel.
* Assume the line marked with `(**)` is re-written as `accum = accum + tmpA*tmpA`.
  Now it is possible to rewrite both the inner and the outer loop as a nested 
    composition of parallel operators! Please write in your report the 
    semantically-equivalent Futhark program.

== Task 3: Optimizing Spatial Locality by Transposition in CUDA (3 pts)

Please read from the lecture notes:

* Section 5.7 "Loop Strip-mining, Block and Register Tiling",
* Section 6.1 "Optimizing the Spatial Locality of Read/Write Accesses on GPUs by Transposition", and
* Section 6.2 "Transposition: Block Tiling Optimizes Spatial Locality"

Your *task* (*3pts*) is to write a CUDA implementation for the program of Task 2,
which uses (manifested) transpositions to achieve coalesced accesses to
global memory---see Section 6.1 in lecture notes. This refers to implementing
kernel `transfProg` in file `transpose-kernels.cu.h` and filling in the missing 
CPU orchestration in file `transpose-main.cu`. 

An *optional (bonus) task*---i.e., for the braves, but not required---is to implement
the two lines of function `glb2shmem` that is used by kernel `optimProg` to copy in 
coalesced way from global to shared memory (without manifesting/performing the transpositions). 
Both are located in file `transpose-kernels.cu.h`. 

Briefly comment in your report on:

* the code implementing your solution, i.e., present the code and comment on
    its correctness and on how it optimizes spatial locality (i.e., coalesced
    access to global memory)

* specify whether your implementation validates,
    specify the GB/sec achieved by your implementations, and compare it with
    `memcpy` bandwidth and with the original GPU program.

* If you do the bonus task, what is the speedup of `optimProg` in comparison
    with  `transfProg` and how do you explain it? 

The original program is similar to the one shown in pseudocode in Task 2,
and it is already implemented in the handed-in code.


== Task 4: Implement Block and Register Tiling for Matrix-Matrix Multiplication in CUDA  (3 pts)

The text for this task is available in lecture notes:

* Section 6.4 "Exercise: Block and Register Tiling for Matrix-Matrix Multiplication"

* Files `mmm-main.cu` and `mmm-kernels.cu.h` contain the naive and the  
   block-tiled implementation of (dense) matrix-matrix multiplication.

To solve task 4:

* please fill in files `mmm-main.cu` and `mmm-kernels.cu.h` the CPU orchestration 
  code and the GPU kernels that perform a combination of register and block tiling, 
  as described in Section 6.4 of the lecture notes,
* submit the filled-in files, and
* write in your report whether your implementation validates,
    the GFlops of your implementation, and the speedup obtained
    in comparison with the naive and block-tiled versions of matrix-matrix
    multiplication. 

