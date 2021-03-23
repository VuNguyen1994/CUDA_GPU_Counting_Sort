#define HISTOGRAM_SIZE 256 /* Histogram has 256 bins */

/* Write GPU code to perform the step(s) involved in counting sort. 
 Add additional kernels and device functions as needed. */

__global__ void find_prefix_kernel(int *input_data, int *prefix_array, int num_elements, int range)
{
    __shared__ unsigned int s[HISTOGRAM_SIZE];
    __shared__ unsigned int s_temp[HISTOGRAM_SIZE];

    /* Initialize shared memory */ 
    if(threadIdx.x <= range){
        s[threadIdx.x] = 0;
        s_temp[threadIdx.x] = 0;
    }

    __syncthreads();

    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (offset < num_elements) {
        atomicAdd(&s[input_data[offset]], 1);
        offset += stride;
    }

    __syncthreads();

    /* Step 2: Calculate starting indices in output array for storing sorted elements. 
     * Use inclusive scan of the bin elements. */
    int off = 1;
    int pingpong_flag = 1;
    int tid = threadIdx.x;
    while(off < num_elements){
        if (pingpong_flag){
            if (tid >= off)
                s_temp[tid] = s[tid] + s[tid - off];
            else
                s_temp[tid] = s[tid];
        }
        else{
            if (tid >= off)
                s[tid] = s_temp[tid] + s_temp[tid - off];
            else
                s[tid] = s_temp[tid];
        }
        __syncthreads();
        pingpong_flag = !pingpong_flag;
        off = 2*off;
    }

    /* Accumulate prefix array in shared memory into global memory, and send to CPU */
    if (threadIdx.x <= range) 
        atomicAdd(&prefix_array[threadIdx.x], s[threadIdx.x]);

    return;
}

__global__ void counting_sort_kernel(int *prefix_array, int *sorted_array, int num_elements, int range)
{
    __shared__ unsigned int prefix_shared[HISTOGRAM_SIZE];
    /* Get prefix array from CPU, copy to shared mem and arrange the sorted array */
    int tid = threadIdx.x;
    if (tid <= range)
        prefix_shared[tid] = prefix_array[tid];
    
    __syncthreads();

    int start_idx = 0;
    int j = 0;
    if (tid == 0)
        start_idx = 0;
    else
        start_idx = prefix_shared[tid-1];

    int end_idx = prefix_shared[tid];

    for (j = start_idx; j < end_idx; j++) 
            sorted_array[j] = tid;
    return;
}