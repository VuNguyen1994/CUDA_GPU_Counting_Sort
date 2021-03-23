/* Host-side code to perform counting sort 
 * 
 * Author: Naga Kandasamy
 * Date modified: March 2, 2021
 * 
 * Student name(s): Dinh Nguyen, Tri Pham, Manh Cuong Phi
 * Date modified: 03/14/2021
 * 
 * Compile as follows: make clean && make
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "counting_sort_kernel.cu"

/* Do not change the range value */
#define MIN_VALUE 0 
#define MAX_VALUE 255
#define THREAD_BLOCK_SIZE 256 
#define NUM_BLOCKS 4 
#define HISTOGRAM_SIZE 256 /* Histogram has 256 bins */

/* Uncomment to spit out debug info */
// #define DEBUG

extern "C" int counting_sort_gold(int *, int *, int, int);
int rand_int(int, int);
void print_array(int *, int);
void print_min_and_max_in_array(int *, int);
void compute_on_device(int *, int *, int, int);
int check_if_sorted(int *, int);
int compare_results(int *, int *, int);
void check_for_error(const char *);

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s num-elements\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_elements = atoi(argv[1]);
    int range = MAX_VALUE - MIN_VALUE;
    int *input_array, *sorted_array_reference, *sorted_array_d;

    /* Populate input array with random integers between [0, RANGE] */
    printf("Generating input array with %d elements in the range 0 to %d\n", num_elements, range);
    input_array = (int *)malloc(num_elements * sizeof(int));
    if (input_array == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++)
        input_array[i] = rand_int (MIN_VALUE, MAX_VALUE);

#ifdef DEBUG
    print_array(input_array, num_elements);
    print_min_and_max_in_array(input_array, num_elements);
#endif

    struct timeval start, stop;

    /* Sort elements in input array using reference implementation. 
     * The result is placed in sorted_array_reference. */
    printf("\nSorting array on CPU\n");
    int status;
    sorted_array_reference = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_reference == NULL) {
        perror("malloc"); 
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_reference, 0, num_elements);
    gettimeofday(&start, NULL);
    status = counting_sort_gold(input_array, sorted_array_reference, num_elements, range);
    gettimeofday(&stop, NULL);
    if (status == -1) {
        exit(EXIT_FAILURE);
    }

    status = check_if_sorted(sorted_array_reference, num_elements);
    if (status == -1) {
        printf("Error sorting the input array using the reference code\n");
        exit(EXIT_FAILURE);
    }

    printf("Counting sort was successful on the CPU\n");
    fprintf(stderr, "CPU Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000));

#ifdef DEBUG
    print_array(sorted_array_reference, num_elements);
#endif

    /* FIXME: Write function to sort elements in the array in parallel fashion. 
     * The result should be placed in sorted_array_mt. */
    printf("\nSorting array on GPU\n");
    sorted_array_d = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_d == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_d, 0, num_elements);
    compute_on_device(input_array, sorted_array_d, num_elements, range);

#ifdef DEBUG
    print_array(sorted_array_d, num_elements);
#endif
    /* Check the two results for correctness */
    printf("\nComparing CPU and GPU results\n");
    status = compare_results(sorted_array_reference, sorted_array_d, num_elements);
    if (status == 0)
        printf("Test passed\n");
    else
        printf("Test failed\n");

    exit(EXIT_SUCCESS);
}


/* FIXME: Write the GPU implementation of counting sort */
void compute_on_device(int *input_array, int *sorted_array, int num_elements, int range)
{
    struct timeval start, stop;

    int *input_array_on_device = NULL;
	int *sorted_array_on_device = NULL;
    int *prefix_array = (int *)malloc(HISTOGRAM_SIZE * sizeof(int));
    int *prefix_array_on_device = NULL;

    /* Set up the execution grid on GPU */
	dim3 thread_block(THREAD_BLOCK_SIZE, 1);
	dim3 grid(NUM_BLOCKS,1);

    
    /* Allocate space on GPU for input data */
	cudaMalloc((void**)&input_array_on_device, num_elements * sizeof(int));
	cudaMemcpy(input_array_on_device, input_array, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    /* Allocate space on GPU  initialize contents to zero */
	cudaMalloc((void**)&sorted_array_on_device, num_elements * sizeof(int));
	cudaMemset(sorted_array_on_device, 0, num_elements * sizeof(int));

	cudaMalloc((void**)&prefix_array_on_device, HISTOGRAM_SIZE * sizeof(int));
	cudaMemset(prefix_array_on_device, 0, HISTOGRAM_SIZE * sizeof(int));

    gettimeofday(&start, NULL);
    // Launch kernel to find prefix array
    find_prefix_kernel<<<grid, thread_block>>>(input_array_on_device, prefix_array_on_device, num_elements, range);
    cudaDeviceSynchronize();
    // Launch kernel to form sorted array using the prefix array as input
    counting_sort_kernel<<<grid,thread_block>>>(prefix_array_on_device, sorted_array_on_device, num_elements, range);
    cudaDeviceSynchronize();

    gettimeofday(&stop, NULL);
	fprintf(stderr, "GPU Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +(stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Copy result back from GPU */ 
	cudaMemcpy(sorted_array, sorted_array_on_device, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    check_for_error("KERNEL FAILURE");

    /* Free memory */
	cudaFree(input_array_on_device);
	cudaFree(sorted_array_on_device);
    cudaFree(prefix_array_on_device);
    free(prefix_array);

    return;
}

/* Check for errors during kernel execution */
void check_for_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 

/* Check if array is sorted */
int check_if_sorted(int *array, int num_elements)
{
    int status = 0;
    int i;
    for (i = 1; i < num_elements; i++) {
        if (array[i - 1] > array[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Check if the arrays elements are identical */ 
int compare_results(int *array_1, int *array_2, int num_elements)
{
    int status = 0;
    int i;
    for (i = 0; i < num_elements; i++) {
        if (array_1[i] != array_2[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Return random integer between [min, max] */ 
int rand_int(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
    return (int)floorf(min + (max - min) * r);
}

/* Print given array */
void print_array(int *this_array, int num_elements)
{
    printf("Array: ");
    int i;
    for (i = 0; i < num_elements; i++)
        printf("%d ", this_array[i]);
    
    printf("\n");
    return;
}

/* Return min and max values in given array */
void print_min_and_max_in_array(int *this_array, int num_elements)
{
    int i;

    int current_min = INT_MAX;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] < current_min)
            current_min = this_array[i];

    int current_max = INT_MIN;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] > current_max)
            current_max = this_array[i];

    printf("Minimum value in the array = %d\n", current_min);
    printf("Maximum value in the array = %d\n", current_max);
    return;
}


