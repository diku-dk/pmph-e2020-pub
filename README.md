<!--
# pmph-e2020-pub
--PMPH course 2020 public repo
--> 

# Programming Massively Parallel Hardware (PMPH), Block 1 2020

## Course Structure

PMPH is structured to have four hours of virtual lectures and four
hours of mixed (physical + virtual) labs per week; potentially we
will have no lectures in the last few weeks of the course, so you
can concentrate on project work (to be announced).

### Lectures (zoom links will be posted on Absalon):

* Tuesday  10:15 - 12:00
* Thursday 10:15 - 12:00

### Labs: 

* Thursday 13:00 - 17:00 (or later if students ask for it)

We have not been yet assigned classes, but the plan is that half of
the enrolled students (say according to alphabetical ordering) may
physically attend the lab from 13:00 - 15:00 and the other half
attends virtually on zoom, and then we switched roles for the
session between 15:00 - 17:00.  This way, each student is guaranteed
to physically attend two hours of lab per week, if she/he wishes
to attend (of course).

If you like the idea of attending the labs physically, then I
suggest you come to DIKU for the other lab session as well (the 
one in which you are only guaranteed virtual presence by default),
because I will allow such students to join the lab until the corona
class capacity is reached (using a first-come first-serve policy).
This is because typically some students absentee the physical lab
sessions, so there might be space for the others. If we are out of
space you can find a quit place to attend by zoom via edurom.

### Evaluation

Throughout the course, you will hand in four weekly assignments,
which will count for 40\% of the final grade. In the last month
of the course, you will work on a group project (up to three
students per group), and will submit the report and accompanying
code. The group project will be presented orally at the exam together
with the answers to some individual questions, and this will count
for 60\% of your final grade.

The "weekly-assignments" (W-assignments) are tentatively planned to be
published in each Thursday of the first four weeks. They have one week
editing time. If a serious attempt was made but the solution is not
satisfactory (or simply if you want to improve your assignment, hence grade),
an updated solution should be resubmitted one week after the time when
the assignment was graded.  Extensions may be possible, but you will need
to agree with your TA.

For the group project no re-submission is possible; the deadline is the
Friday just before the exam week.

The oral examination will be hold in the exam week (Wednesday, Thursday and Friday if necessary). The final evaluation will take up to 20 minutes per student, but probably the whole group will be examined at a time (unless you wish otherwise).

**Weekly and group assignment handin is still on Absalon.**

### Teacher and Teaching Assistants (TAs)

The main teacher is **[Cosmin Oancea](mailto:cosmin.oancea@diku.dk)**.

One TA is **[Anders Holst](mailto:anersholst@gmail.com)**. 

There will be another TA, to be announced.

The plan is that the teacher will conduct the lectures and the lab.
The TAs will be mainly in charge of grading the weekly assignments,
patrolling the Absalon discussion forum and perhaps helping with the
virtual lab.

### Course Tracks and Resources

All lectures and lab sessions will be delivered in English.  The
assignments and projects will be posted in English, and while you can
chose to hand in solutions in either English or Danish, English is
preferred. All course material except for the hardware book is distributed via this GitHub page. (Assignment handin is still on Absalon.)

* **The hardware track** of the course covers (lecture) topics related to processor, memory and interconnect design, including cache coherency, which are selected from the book [Parallel Computer Organization and Design, by Michel Dubois, Murali Annavaram and Per Stenstrom,  ISBN 978-521-88675-8. Cambridge University Press, 2012](https://www.cambridge.org/dk/academic/subjects/engineering/computer-engineering/parallel-computer-organization-and-design?format=HB&isbn=9780521886758). The book is available at the local bookstore (biocenter). It is not mandatory to buy it---Cosmin thinks that it is possible to understand the material from the lecture slides, which are detailed enough---but also note that lecture notes are not provided for the hardware track, because of copyright issues.

* **The software track** covers (lecture) topics related to parallel-programming models and recipes to recognize and optimize parallelism and locality of reference.  It demonstrates that compiler optimizations are essential to fully utilizing hardware, and that some optimizations can be implemented both in hardware and software, but with different pro and cons.   [The lecture notes are available here](material/lecture-notes-pmph.pdf), and additional (facultative) reading material (papers) will be linked with individual lectures; see Course Schedule Section below.

* **The lab track** teaches GPGPU hardware specifics and programming in Futhark, CUDA, and OpenMP. The intent is that the lab track applies in practice some of the parallel programming principles and optimizations techniques discussed in the software tracks. It is also intended to provide help for the weekly assignment, project, etc.

## Course Schedule

This course schedule is tentative and will be updated as we go along.

The lab sessions are aimed at providing help for the weeklies and
group project.  Do not assume you can solve them without attending
the lab sessions.

| Date | Time | Topic | Material |
| --- | --- | --- | --- |
| 18/11 | 13:00-15:00 | *Cancelled* |
| 18/11 | 15:00-17:00 | [Intro, deterministic parallelism, data parallelism, Futhark](slides/L1-determ-prog.pdf) | [Parallel Programming in Futhark](https://futhark-book.readthedocs.io/en/latest/), sections 1-4 | |
| 20/11 | 10:00-12:00 | Lab ([**Assignment 1 handout**](weekly-1/)) | [Futhark exercises](bootstrap-exercises.md) |
| 20/11 | 13:00-15:00 | [Cost models, advanced Futhark](slides/L2-advanced-futhark-cost-models.pdf) | [Guy Blelloch: Programming Parallel Algorithms](material/blelloch-programming-parallel-algorithms.pdf), [Prefix Sums and Their Applications](material/prefix-sums-and-their-applications.pdf), [A Provable Time and Space Efficient Implementation of NESL](material/a-provable-time-and-space-efficient-implementation-of-nesl.pdf) |
| 25/11 | 13:00-15:00 | Lab | |
| 25/11 | 15:00-17:00 | [Regular flattening: moderate and incremental](slides/L3-regular-flattening.pdf) | [Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates](https://futhark-lang.org/publications/pldi17.pdf), [Incremental Flattening for Nested Data Parallelism](https://futhark-lang.org/publications/ppopp19.pdf) (particularly the latter) |
| 27/11 | 10:00-12:00 | [Full/irregular flattening](slides/L4-irreg-flattening.pdf) | [Transforming High-Level Data-Parallel Programs into Vector Operations](material/flattening/NeslFlatTechPaper.pdf), [Harnessing the Multicores: Nested Data Parallelism in Haskell](material/flattening/harnessing-multicores.pdf) (not easy to read)|
| 27/11 | 13:00-15:00 | Lab ([**Assignment 2 handout**](weekly-2/)) | |
| 2/12 | 13:00-15:00 | [Task parallelism (parallel Haskell)](slides/L5-parallel-haskell.pdf/) ([code](slides/L5-parallel-haskell-code/)) | [Parallel and Concurrent Programming in Haskell](https://www.oreilly.com/library/view/parallel-and-concurrent/9781449335939/), chapter 4. [Seq no more: Better Strategies for Parallel Haskell](material/seq-no-more.pdf) |
| 2/12 | 15:00-17:00 | Lab | |
| 4/12 | 10:00-12:00 | [Halide](slides/L7-Halide.pdf) | [Halide: A Language and Compiler for Optimizing Parallelism, Locality and Recomputation in Image Processing Pipelines](material/halide-pldi13.pdf) | |
| 4/12 | 13:00-15:00 | Lab ([**Assignment 3 handout**](weekly-3)) | |
| 9/12 | 13:00-15:00 | [Polyhedral Analysis](slides/L9-polyhedral.pdf) | [PMPH Dependence Analysis](material/poly/L5-LoopParI.pdf); [Sven Verdoolaege: Presburger Formulas and Polyhedral Compilation (tutorial)](material/poly/polycomp-tutorial.pdf); [Sven Verdoolaege: Presburger Sets and Relations: from High-Level Modelling to Low-Level Implementation (slides)](material/poly/poly-in-detail.pdf)
| 9/12 | 15:00-17:00 | Lab | [Code Examples Using the ISLPY library](material/poly/poly-code-egs/) |
| 11/12 | 10:00-12:00 | [Vector programming with ISPC](slides/L8-ispc.pdf) | [ispc: A SPMD Compiler for High-Performance CPU Programming](material/ispc_inpar_2012.pdf) |
| 11/12 | 13:00-15:00 | Lab ([**Assignment 4 handout**](weekly-4)) | |
| 16/12 | 13:00-15:00 | [Optimizing Locality of Reference](slides/L10-LocOfRef.pdf) | [Related Literature](material/Opt-Loc-Ref-Lit/) |
| 16/12 | 15:00-17:00 | Lab (with project proposals) | |
| 18/12 | 10:00-12:00 | Lab | |
| 18/12 | 13:00-15:00 | Lab | |

## Weekly assignments

The weekly assignments are **mandatory**, must be solved
**individually**, and make up 40% of your final grade.  Submission is
on Absalon.

You will receive feedback a week after the handin deadline (at the
latest).  You then have another week to prepare a resubmission.  That
is, **the resubmission deadline is two weeks after the original handin
deadline**.

### Weekly 1 (due November 28)

* [Assignment text]()
* [Code handout]()
* [Hopefully helpful notes on how to do these kinds of assignments]()

### Weekly 2 (due December 5)

to be announced

### Weekly 3 (due December 12)

to be announced

### Weekly 4 (due December 19)

to be announced

## Group project

to be announced

## GPU + MultiCore Machines

All students will be provided individual accounts on a multi-core and GPGPU machine that supports multi-core programming via C++/OpenMP and CUDA programming.
Login to GPU & 16 multicore machines will become operational after 3rd of September:

You log in by first SSHing to the bastion server
`ssh-diku-apl.science.ku.dk` using your KU license plate (`abc123`) as
the user name, and then SSHing on to one of the GPU machines.

```bash
$ ssh -l <ku_id> ssh-diku-apl.science.ku.dk
$ ssh gpu04-diku-apl
````
(or gpu02-diku-apl or gpu03-diku-apl).
 
Despite their names, they each have 16 cores with 2-way hyperthreading CPUs and plenty of RAM as well. The GPUs are:
  * `gpu02-diku-apl`, `gpu03-diku-apl` have dual GTX780 Ti GPUs.

  * `gpu04-diku-apl` has a GTX 2080 Ti GPU (by far the fastest).

For CUDA to work, you may need to add the following to your `$HOME/.bash_profile` or `$HOME/.bashrc` file (on one of the gpu02/4-diku-apl machines):

```bash
CUDA_DIR=/usr/local/cuda
export PATH=$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_DIR/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDA_DIR/include:$C_INCLUDE_PATH
```

## Other resources

### Futhark and CUDA

* We will use a basic subset of Futhark during the course. Futhark related documentation can be found at [Futhark's webpage](https://futhark-lang.org), in particular a [tutorial](https://futhark-book.readthedocs.io/en/latest/) and [user guide](https://futhark.readthedocs.io/en/stable/)

* [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) you may want to browse through this guide to see what offers. No need to read all of it closely.


### Other Related Books

* Some of the compiler transformations taught in the software track can be found
in this book [Optimizing Compilers for Modern Architectures. Randy Allen and Ken Kennedy, Morgan Kaufmann, 2001](https://www.elsevier.com/books/optimizing-compilers-for-modern-architectures/allen/978-0-08-051324-9), but you are not expected to buy it or read for the purpose of PMPH.

* Similarly, some course topics are further developed in this book [High-Performance Computing Paradigm and Infrastructure](https://www.wiley.com/en-dk/High+Performance+Computing%3A+Paradigm+and+Infrastructure-p-9780471732709), e.g., Chapters 3, 8 and 11, but again, you are not expected to buy it or read for the purpose of PMPH.

