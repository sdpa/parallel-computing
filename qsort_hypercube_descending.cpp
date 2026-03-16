// -----------------------------------------------------------------------------
// Hypercube Quicksort to sort a list of integers in DESCENDING order
// distributed across processors - MPI-based implementation
//
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <mpi.h>

#define MAX_LIST_SIZE_PER_PROC	268435456

#ifndef VERBOSE
#define VERBOSE 0			// Use VERBOSE to control output 
#endif

int compare_int_desc(const void *, const void *);

class HyperCube_Class {
    public:
	void Initialize(int, int, int);
	void HyperCube_QuickSort();
	void print_list();
	void check_list();

	int dimension;		// Hypercube dimension

	int num_procs; 		// Number of MPI processes
	int my_id;		// Rank/id of this process

	int list_size;		// Local list size
	int list_size_initial;	// Size of initial local list (before sorting)

    private:
	int * initialize_list(int);
	int neighbor_along_dim_k(int );
	int * merged_list(int *, int, int *, int); 
	int split_list_index (int *, int, int); 
	void print_local_list();

	int *list;		// Local list
};

//
// Initialize HyperCube
//
void HyperCube_Class::Initialize(int dim, int size, int type) {

    dimension = dim; 		// Hypercube dimension

    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);	// num_procs = number of MPI processes
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);	// my_id = rank of this process

    list_size_initial = size;
    list_size = list_size_initial; 

    // Call initialize_list after initializing 
    // num_procs, my_id, and list_size
    list = initialize_list(type);	// Initialize local list
}

// Computes the rank of neighbor process along dimension k (k > 0) of 
// the hypercube. Rank is computed by flipping the kth bit of rank/id of 
// this process
//
int HyperCube_Class::neighbor_along_dim_k(int k) {
    int mask = 1 << (k-1); 
    return (my_id ^ mask); 
}

// Merge two sorted lists (in DESCENDING order) and return the merged list
// Input:
//   list1, list1_size	- first list and its size (descending order)
//   list2, list2_size	- second list and its size (descending order)
// Output:
//   list		- merged list in descending order (size not returned!)
//
int * HyperCube_Class::merged_list(int * list1, int list1_size, int * list2, int list2_size) {
    int * list = new int[list1_size+list2_size];
    int idx1 = 0; 
    int idx2 = 0; 
    int idx = 0; 
    while ((idx1 < list1_size) && (idx2 < list2_size)) {
	if (list1[idx1] >= list2[idx2]) {  // Changed from <= to >= for descending order
	    list[idx] = list1[idx1]; 
	    idx++; idx1++;
	} else {
	    list[idx] = list2[idx2]; 
	    idx++; idx2++;
	}
    }
    while (idx1 < list1_size) {
	list[idx] = list1[idx1]; 
	idx++; idx1++;
    }
    while (idx2 < list2_size) {
	list[idx] = list2[idx2]; 
	idx++; idx2++;
    }
    return list;
}

// Search for smallest element in a sorted (DESCENDING) list which is less than pivot
// Uses binary search since list is sorted in descending order.
// Input:
//   list, list_size	- list and its size (sorted in descending order)
//   pivot		- value to search for
// Output:
//   last 	- index of the smallest element that is less than the pivot
//              list[0 ... last-1] >= pivot
//              list[last ... list_size-1] < pivot
//
int HyperCube_Class::split_list_index (int *list, int list_size, int pivot) {
    int first, last, mid;  
    first = 0; last = list_size; mid = (first+last)/2;
    while (first < last) {
	if (list[mid] >= pivot) {  // Changed from <= to >= for descending order
	    first = mid+1; mid = (first+last)/2;
	} else {
	    last = mid; mid = (first+last)/2;
	}
    }
    return last;
}

// Print local list
//
void HyperCube_Class::print_local_list() {
    int j;
    for (j = 0; j < list_size; j++) {
	if ((j % 8) == 0) printf("[Proc: %0d]", my_id);
	printf(" %8d", list[j]); 
	if ((j % 8) == 7) printf("\n"); 
    }
    printf("\n"); 
    return;
}

// Print list: processes print local lists in order of their ranks (from 0 to p-1)
//
void HyperCube_Class::print_list() {
    int tag = 0;
    int dummy = 0;
    MPI_Status status; 
    if (my_id-1 >= 0) {
	MPI_Recv(&dummy, 1, MPI_INT, my_id-1, tag, MPI_COMM_WORLD, &status);
    }
    print_local_list(); 
    if (my_id+1 < num_procs) {
	MPI_Send(&dummy, 1, MPI_INT, my_id+1, tag, MPI_COMM_WORLD);
    }
}

// Allocate and initialize local list
// Input: 
//   type 	- initialization type (elements in increasing  order, 
//  		  decreasing order, random, other types can be added)
// Output: 
//   list	- integer array of size list_size containing elements of
//   		  list
//
int * HyperCube_Class::initialize_list(int type) {
    int j;
    int * list = new int[list_size]; 
    switch (type) {
	case -1:	// Elements are in descending order
	    for (j = 0; j < list_size; j++) {
		list[j] = (num_procs-my_id)*list_size-j;
	    }
	    break;
	case -2:	// Elements are in ascending order
	    for (j = 0; j < list_size; j++) {
		list[j] = my_id*list_size+j+1;
	    }
	    break;
	default: 
	    srand48(type + my_id); 
	    list[0] = lrand48() % 100;
	    for (j = 1; j < list_size; j++) {
		list[j] = lrand48() % 100;
	    }
	    break;
    }
    return list;
}

// Check if list is sorted in DESCENDING order. 
// Each process verifies that its local list is sorted in descending order.
// The process also checks that its list has values smaller than or equal to
// the smallest value on the process before it (i.e., on process (my_id-1).
// For descending order: P0 has largest values, P(n-1) has smallest values
// Prints result of error check if VERBOSE > 1 
//
void HyperCube_Class::check_list() {
    int tag = 0;
    int min_nbr = 2147483647;	// Initialize to max int value
    int error, local_error;
    int j, my_min;
    MPI_Status status; 
    // Receive smallest list value from process with rank (my_id-1)
    if (my_id-1 >= 0) {
	MPI_Recv(&min_nbr, 1, MPI_INT, my_id-1, tag, MPI_COMM_WORLD, &status);
	// Good practice to check status!
    }
    // Check that the local list is sorted in descending order and that 
    // elements are smaller than or equal to the smallest on process with rank (my_id-1)
    // (error is set to 1 if a pair of elements is not sorted correctly)
    local_error = 0;
    if (list_size > 0) {
	if (list[0] > min_nbr) local_error = 1;  // First element should be <= min from previous proc
	for (j = 1; j < list_size; j++) {
	    if (list[j] > list[j-1]) local_error = 1;  // Should be descending
	}
	my_min = list[list_size-1];  // Last element is smallest in descending order
    } else {
	my_min = min_nbr;
    }
    if (VERBOSE > 1) {
	printf("[Proc: %0d] check_list: local_error = %d\n", my_id, local_error);
    }
    // Send smallest list value to process with rank (my_id+1)
    if (my_id+1 < num_procs) {
	MPI_Send(&my_min, 1, MPI_INT, my_id+1, tag, MPI_COMM_WORLD);
	// Good practice to check status!
    }
    // Collect errors from all processes
    // error = 0;
    MPI_Reduce(&local_error, &error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (my_id == 0) {
	if (error == 0) {
	    printf("[Proc: %0d] Congratulations. The list has been sorted correctly in DESCENDING order.\n", my_id);
	} else {
	    printf("[Proc: %0d] Error encountered. The list has not been sorted correctly.\n", my_id);
	}
    }
}
//
// HyperCube Quicksort - DESCENDING ORDER
// For descending order: lower-ranked processes keep larger elements
//
void HyperCube_Class::HyperCube_QuickSort() {
    int k; 			// Sub hypercube dimension
    int nbr_k; 			// Neighbor of this process along dim-k

    int local_median;		// Median of local list
    int pivot;			// Value used to split local list 
    int idx;			// index where local list is split
    int list_size_geq;		// Number of elements >= pivot
    int list_size_lt;		// Number of elements < pivot
    int * nbr_list; 		// Sublist received from nbr process
    int nbr_list_size;		// nbr_list size
    int * new_list; 		// List obtained by merging nbr list with
    // MPI variables
    int tag = 0;
    MPI_Status status; 

    int error, mask, i;		// Miscellaneous work variables

    // Hypercube group and communicator variables to facilitate pivot computation
    // An MPI Group and an MPI communicator are created for hypercube of 
    // dimension k; this hypercube includes all processes with ranks 
    // that differ from this process in the lowest k bits only
    //
    int sub_hypercube_size; 		// Number of processors in dim-k hypercube
    int * sub_hypercube_processors;	// List of processors in dim-k hypercube
    MPI_Group sub_hypercube_group; 	// Group of processes in dim-k hypercube
    MPI_Comm sub_hypercube_comm;	// Communicator for dim-k hypercube
    MPI_Group hypercube_group;		// Group of all processes

    // Sort local list in DESCENDING order
    qsort(list, list_size, sizeof(int), compare_int_desc);

    // Initialize processor group for hypercube
    MPI_Comm_group(MPI_COMM_WORLD, &hypercube_group);

    // Hypercube Quicksort
    for (k = dimension; k > 0; k--) {

	// Find processes that make up the sub-hypercube of dimension k and 
	// include this process; the sub-hypercube includes all processes with 
	// ranks that differ from this process in the lowest k bits only 
	sub_hypercube_size = (int) pow(2,k); mask = (~0) << k;
	sub_hypercube_processors = new int[sub_hypercube_size]; 
	sub_hypercube_processors[0] = my_id & mask; 
	for (i = 1; i < sub_hypercube_size; i++) {
	    sub_hypercube_processors[i] = sub_hypercube_processors[i-1]+1;
	}

	// Construct processor group for sub-hypercube
	MPI_Group_incl(hypercube_group, sub_hypercube_size, sub_hypercube_processors, &sub_hypercube_group);

	// Construct processor communicator to simplify computation of pivot
	// via MPI_Allreduce within the sub-hypercube 
	MPI_Comm_create(MPI_COMM_WORLD, sub_hypercube_group, &sub_hypercube_comm);

	// Find median of sorted local list
	local_median = list[list_size/2];

	// Compute pivot for hypercube of dimension k (pivot = mean of medians)
	MPI_Allreduce(&local_median, &pivot, 1, MPI_INT, MPI_SUM, sub_hypercube_comm);

	pivot = pivot/sub_hypercube_size;

	// Search for smallest element in list which is less than pivot
	// Upon return (for DESCENDING order):
	//   list[0 ... idx-1] >= pivot (larger elements)
	//   list[idx ... list_size-1] < pivot (smaller elements)
	idx = split_list_index(list, list_size, pivot);

	list_size_geq = idx;
	list_size_lt = list_size - idx;

	// Communicate with neighbor along dimension k
	nbr_k = neighbor_along_dim_k(k); 

	// For DESCENDING order: lower-ranked processes keep larger elements (>= pivot)
	if (nbr_k > my_id) {
	    // I have lower rank, so I keep larger elements (>= pivot) and send smaller elements (< pivot)
	    
	    // Send number of elements less than pivot
	    MPI_Send(&list_size_lt, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    // Receive number of elements greater than or equal to pivot
	    MPI_Recv(&nbr_list_size, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    // Allocate storage for neighbor's list
	    nbr_list = new int[nbr_list_size];

	    // Send list[idx ... list_size-1] (smaller elements) to neighbor
	    MPI_Send(&list[idx], list_size_lt, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    // Receive neighbor's list of elements that are greater than or equal to pivot
	    MPI_Recv(nbr_list, nbr_list_size, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    // Merge local list of elements >= pivot with neighbor's list
	    new_list = merged_list(list, idx, nbr_list, nbr_list_size); 

	    // Replace local list with new_list, update size
	    delete [] list; 
	    delete [] nbr_list; 
	    list = new_list; 
	    list_size = list_size_geq+nbr_list_size;

	} else {
	    // I have higher rank, so I keep smaller elements (< pivot) and send larger elements (>= pivot)
	    
	    // Receive number of elements less than pivot
	    MPI_Recv(&nbr_list_size, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    // Send number of elements greater than or equal to pivot
	    MPI_Send(&list_size_geq, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    // Allocate storage for neighbor's list
	    nbr_list = new int[nbr_list_size]; 

	    // Receive neighbor's list of elements that are less than the pivot
	    MPI_Recv(nbr_list, nbr_list_size, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    // Send list[0 ... idx-1] (larger elements) to neighbor
	    MPI_Send(list, list_size_geq, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    // Merge local list of elements < pivot with neighbor's list
	    new_list = merged_list(&list[idx], list_size_lt, nbr_list, nbr_list_size); 

	    // Replace local list with new_list, update size
	    delete [] list; 
	    delete [] nbr_list; 
	    list = new_list; 
	    list_size = list_size_lt+nbr_list_size;
	}
	// Deallocate processor group, processor communicator, 
	// sub_hypercube_processors array; these variables will be 
	// reused in the next iteration of this for loop for a hypercube of 
	// dimension (k-1)
	MPI_Group_free(&sub_hypercube_group);
	MPI_Comm_free(&sub_hypercube_comm);
	delete [] sub_hypercube_processors;
    }
}

// Comparison routine for qsort (stdlib.h) which is used to sort local list in DESCENDING order
// Used as follows to sort elements of list[0 ... list_size-1]: 
//
// 	qsort(list, list_size, sizeof(int), compare_int_desc)
//
int compare_int_desc(const void *a0, const void *b0) {
    int a = *(int *)a0;
    int b = *(int *)b0;
    if (a > b) {  // Reversed comparison for descending order
	return -1; 
    } else if (a < b) {
	return 1;
    } else {
	return 0;
    }
}

//------------------------------------------------------------------------------
// Main program
// 
int main(int argc, char *argv[])
{
    // User inputs
    int size; 				// List size on each process
    int type;				// type for list initialization
    int dim;				// Hypercube dimension 

    // MPI 
    int my_id; 				// My MPI rank
    int num_procs;			// Hypercube size 
    double start, total_time;		// Timing variables

    // MPI
    MPI_Init(&argc,&argv);		// Initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);	// num_procs = number of processes
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);	// my_id = rank of this process

    //  Check inputs
    if (argc != 3)  {
	if (my_id == 0) 
	    printf("Usage: mpirun -np <number_of_processes> <executable_name> <list_size_per_process> <type>\n");
	exit(0);
    }

    size = atoi(argv[argc-2]);
    if ((size <= 0) || (size > MAX_LIST_SIZE_PER_PROC)) {
	if (my_id == 0)
	    printf("List size outside range [%d ... %d]. Aborting ...\n", 1, MAX_LIST_SIZE_PER_PROC);
	exit(0);
    }

    type = atoi(argv[argc-1]);

    // Compute hypercube dimension: 2^dim = num_procs
    dim = (int) log2((double) num_procs); 
    if (num_procs != (int) pow(2,dim)) {
	if (my_id == 0) 
	    printf("Number of processors must be power of 2. Aborting ...\n"); 
	exit(0); 
    }

    // Hypercube Initializations +++++++++++++++++++++++++++++++++++++++++++

    HyperCube_Class HyperCube; 			// Create Hypercube node
    HyperCube.Initialize(dim, size, type);	// Initialize Hypercube node

    if (VERBOSE > 2) {
	HyperCube.print_list();
    }

    // Start Hypercube Quicksort (DESCENDING) ..................................
    start = MPI_Wtime(); 
    HyperCube.HyperCube_QuickSort(); 
    total_time = MPI_Wtime()-start;
    // End Hypercube Quicksort ..............................................

    if (my_id == 0) {
	printf("[Proc: %0d] number of processes = %d, ", HyperCube.my_id, HyperCube.num_procs);
	printf("initial local list size = %d, ", HyperCube.list_size_initial);
	printf("hypercube quicksort (DESCENDING) time = %f\n", total_time); 
    }

    // Check if list has been sorted correctly
    HyperCube.check_list();

    if (VERBOSE > 2) {
	HyperCube.print_list();
    }

    MPI_Finalize();				// Finalize MPI
}

