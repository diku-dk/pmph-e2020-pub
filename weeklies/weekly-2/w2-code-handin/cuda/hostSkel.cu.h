#ifndef SCAN_HOST
#define SCAN_HOST

#include "constants.cu.h"
#include "pbbKernels.cu.h"

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

uint32_t closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

void log2UB(uint32_t n, uint32_t* ub, uint32_t* lg) {
    uint32_t r = 0;
    uint32_t m = 1;
    if( n <= 0 ) { printf("Error: log2(0) undefined. Exiting!!!"); exit(1); }
    while(m<n) {
        r = r + 1;
        m = m * 2;
    }
    *ub = m;
    *lg = r;
}

/**
 * `N` is the input-array length
 * `B` is the CUDA block size
 * This function attempts to virtualize the computation so
 *   that it spawns at most 1024 CUDA blocks; otherwise an
 *   error is thrown. It should not throw an error for any
 *   B >= 64.
 * The return is the number of blocks, and `CHUNK * (*num_chunks)`
 *   is the number of elements to be processed sequentially by
 *   each thread so that the number of blocks is <= 1024. 
 */
template<int CHUNK>
uint32_t getNumBlocks(const uint32_t N, const uint32_t B, uint32_t* num_chunks) {
    const uint32_t max_inp_thds = (N + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    *num_chunks = max(1, N / min_elms_all_thds);

    const uint32_t seq_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_thds = (N + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + B - 1) / B;

    if(num_blocks > MAX_BLOCK) {
        printf("Broken Assumption: number of blocks %d exceeds maximal block size: %d. Exiting!"
              , num_blocks, MAX_BLOCK);
        exit(1);
    }

    return num_blocks;
}


/**
 * Host Wrapper orchestraiting the execution of a naive reduce
 * that uses neither shared memory nor efficient sequentialization.
 * d_in  is the input array;
 * the first element of d_out is the result (of the reduce) 
 */
template<class OP>
void mapredNaive( const uint32_t     B         // desired CUDA block size ( <= 1024, multiple of 32)
                , const uint32_t     N         // length of the input array
                , typename OP::RedElTp* d_out  // output array of length: N
                , typename OP::InpElTp* d_in   // input  array of length: N
) {
    uint32_t Q, K;
    log2UB(N, &Q, &K);

    {
        // first invocation
        uint32_t T = Q/2;
        uint32_t num_blocks = (T + B - 1) / B;
        redNaiveKernel1<OP><<< num_blocks, B >>>(d_out, d_in, N, T);
    }
    gpuAssert( cudaPeekAtLastError() );
    uint32_t offs_inp = 0;
    uint32_t offs_out = Q/2;

    for(int k=2; k<=K; k++) {
        uint32_t tmp;
        uint32_t T = Q >> k;
        uint32_t num_blocks = (T + B - 1) / B;
        redNaiveKernel2<OP><<< num_blocks, B >>>(d_out, offs_inp, offs_out, 2*T, T);
        tmp = offs_inp;
        offs_inp = offs_out;
        offs_out = tmp;
    }
}


/**
 * Host Wrapper orchestraiting the execution of an efficient map-reduce:
 * `B` is the CUDA block size
 * `N` is the length of the input array `d_in`
 * `d_out` is the result array, whose maximal length is 1024.
 * The result of the reduce is published in the first element of `d_out`.
 */
template<class OP>
void mapReduce( const uint32_t     B        // desired CUDA block size ( <= 1024, multiple of 32)
              , const uint32_t     N        // length of the input array
              , typename OP::RedElTp* d_out // device array of max length: MAX_BLOCK
              , typename OP::InpElTp* d_in  // device array of length: N
) {
    const uint32_t CHUNK = ELEMS_PER_THREAD*4/sizeof(typename OP::InpElTp);
    uint32_t num_blocks;

    // First stage of a reduction splits the input across blocks, 
    //   then each CUDA block is responsible to (partially) reduce
    //   its elements independent of the others. 
    // 1. it efficiently sequentializes the parallelism in excess of
    //    the maximal number of hardware threads `MAX_HWDTH`
    // 2. it reads/write from/to global memory in coalesced fashion---i.e.,
    //    consecutive threads read/write consecutive array elements---otherwise
    //    the penalty is severe (5-10x slowdowns). 
    // 3. performs the reduction in (fast) shared memory
    if(OP::commutative) {
        // if the operator is commutative, then things are relatively 
        // simple: each thread iterates with a stride `T` over the input
        // and sequentially reduces its elements; this traversal naturally
        // results in "coalesced" access to global memory. Then the threads
        // in a CUDA block cooperatively reduce the `B` elements resulted
        // from the previous step.
        const uint32_t T0 = min(MAX_HWDTH, N);
        num_blocks = min(MAX_BLOCK, (T0 + B - 1) / B);
        const uint32_t T = num_blocks * B;
        const size_t shmem_size = B * sizeof(typename OP::RedElTp);
        redCommuKernel<OP><<< num_blocks, B, shmem_size >>>(d_out, d_in, N, T);
    } else {
        // if the operator is not commutative, things are more tricky because
        // the thread cannot traverse with a stride `T` as before (that traversal
        // assumes commutativity). It follows that each thread has to (sequentially)
        // process consecutive elements, but if done naively this would result
        // in uncoalesced access to global memory. To solve this we need the
        // threads in a block to cooperatively read from global memory in coalesced
        // fashion and put the result in shared memory, and then read their elements
        // from there.
        uint32_t       num_seq_chunks;
        num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
        const size_t shmem_size = B * max( sizeof(typename OP::InpElTp) * CHUNK
                                         , sizeof(typename OP::RedElTp) );
        redAssocKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, N, num_seq_chunks);
    }

    { // The first stage of reduction was thought to split the computation
      // into a number of blocks <= 1024, the maximal CUDA block size.
      // As such, we can spawn one CUDA block to reduce across the results
      // of the previous stage. The reduced result is at index `0` in `d_out`.
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        redAssoc1Block<OP><<< 1, block_size, shmem_size>>>(d_out, num_blocks);
    }
}


/**
 * Host Wrapper orchestraiting the execution of scan:
 * d_in  is the input array
 * d_out is the result array (result of scan)
 * t_tmp is a temporary array (used to scan in-place across the per-block results)
 * Implementation consist of three phases:
 *   1. elements are partitioned across CUDA blocks such that the number of
 *      spawned CUDA blocks is <= 1024. Each CUDA block reduces its elements
 *      and publishes the per-block partial result in `d_tmp`. This is 
 *      implemented in the `redAssocKernel` kernel.
 *   2. The per-block reduced results are scanned within one CUDA block;
 *      this is implemented in `scan1Block` kernel.
 *   3. Then with the same partitioning as in step 1, the whole scan is
 *      performed again at the level of each CUDA block, and then the
 *      prefix of the previous block---available in `d_tmp`---is also
 *      accumulated to each-element result. This is implemented in
 *      `scan3rdKernel` kernel and concludes the whole scan.
 */
template<class OP>                     // element-type and associative operator properties
void scanInc( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
            , const size_t       N     // length of the input array
            , typename OP::RedElTp* d_out // device array of length: N
            , typename OP::InpElTp* d_in  // device array of length: N
            , typename OP::RedElTp* d_tmp // device array of max length: MAX_BLOCK
) {
    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
    const size_t   shmem_size = B * max_tp_size * CHUNK;

    //
    redAssocKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks);

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
    }

    scan3rdKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks);
}

/**
 * Host Wrapper orchestraiting the execution of segmented-inclusive scan.
 * The implementation is similar to the one of the `scanInc`, except that
 * is adapted for segmented scan, i.e., the flag arrays and the lifted 
 * operator that works on flag-value pairs.
 *
 * `B` is the CUDA block size
 * `N` is the input-array length
 * `d_inp`   is the input array
 * `d_flags` is the flag array
 * `d_out` is the result array (result of scan)
 * `d_tmp_vals`  is a temporary array (used to scan in-place across the per-block results)
 * `d_tmp_flags` is the temporary array of flags as above.
 */
template<class OP>                        // element-type and associative operator properties
void sgmScanInc( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
               , const size_t       N     // length of the input array
               , typename OP::RedElTp* d_out        // device array of length: N
               , char*                 d_flags      // device array of length: N
               , typename OP::InpElTp* d_inp        // device array of length: N
               , typename OP::RedElTp* d_tmp_vals   // device array max length: MAX_BLOCK
               , char*                 d_tmp_flags  // device array of length: N
) {
    const uint32_t tot_red_sz = sizeof(typename OP::RedElTp) + sizeof(char);
    const uint32_t tot_inp_sz = sizeof(typename OP::InpElTp) + sizeof(char);
    const uint32_t red_sz     = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (tot_inp_sz > red_sz) ? tot_inp_sz : red_sz;
    
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
    const size_t   shmem_size = B * max( max_tp_size * CHUNK, tot_red_sz );

    redSgmScanKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>
        (d_tmp_flags, d_tmp_vals, d_flags, d_inp, N, num_seq_chunks);

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * ( sizeof(typename OP::RedElTp) + sizeof(char) );
        sgmScan1Block<OP><<< 1, block_size, shmem_size >>>( d_tmp_vals, d_tmp_flags, num_blocks );
    }

    sgmScan3rdKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>
        (d_out, d_inp, d_flags, d_tmp_vals, d_tmp_flags, N, num_seq_chunks);
}

#endif //SCAN_HOST
