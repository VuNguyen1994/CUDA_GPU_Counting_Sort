# CUDA_GPU_Counting_Sort

Counting sort is an efficient non-comparison based algorithm to sort an array of integers in the
range 0 to r for some integer r. Given an input array comprising n elements, counting sort runs in
O(n) time, and trades the time complexity for space complexity when compared to other sorting
algorithms.

The algorithm is to create a historgram with each bin is for each of the integers that appears in the input array.
So if the number in the array is in range from 0 to 255, the histogram will have 256 bins.

Using the histogram, it will run an inclusize prefix scan to accumulate the value in the bins so we can get the start and end
index of the corresponding bins in the output array.

After that, locate the numbers from the histogram back to the main array. The index of the histogram bin is the number we need to put back in 
the main array, the value in the bin is the start and end index of the output array where we need to assign the corresponding number. 

Using CUDA, we can implement the hitogram independently and using mutex lock to update the shared memory histogram.
After that, we can use inclusize scan algorithms to update the prefix array without the race condition.
Then, each thread will be used to update each bin in the prefix array.

More details is located at description.pdf file and report.pdf file.


