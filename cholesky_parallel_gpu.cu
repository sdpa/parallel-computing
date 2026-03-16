// Parallel Compute triangular factors of an SPD matrix using GPU
// - given an SPD matrix A, compute upper triangular matrix R such that 
//   A = R'*R, where R' is the transpose of R
//
// SPD = Symmetric Positive Definite, doesn't require pivoting during factorization
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>

#define MAX_MATRIX_SIZE 4096
#define TOL 1.0e-8

#define ERR_MALLOC 1
#define ERR_MEMCPY 2
#define ERR_KERNEL 3

#define DEBUG 1

// Define Matrix
typedef struct {
    int  n;                 // order of matrix (number of rows and columns)
    double **elements;      // Allows access to array values as a matrix
    double *array;          // Linear array, stores matrix row-by-row
} Matrix;

// Device routines ..............................................................

__global__ void device_create_matrix_on_device(Matrix); 
__global__ void device_cholesky_factorization(Matrix);
__global__ void parallel_cholesky_row_normalize(Matrix A, int k);
__global__ void parallel_cholesky_update(Matrix A, int k);
__global__ void parallel_cholesky_zero_lower(Matrix A, int k);

// Initializes A.elements of matrix A to point to the start of each row in A.array 
__global__ void device_create_matrix_on_device(Matrix A) {
    for (int i = 0; i < A.n; i++) A.elements[i] = &(A.array[i*A.n]);
}

// Original single-thread Cholesky factorization on device
__global__ void device_cholesky_factorization(Matrix A) {
    double sqrt_pivot;
    for (int k = 0; k < A.n; k++) {
        sqrt_pivot = sqrt(A.elements[k][k]);
        for (int j = k; j < A.n; j++) {
            A.elements[k][j] = A.elements[k][j]/sqrt_pivot;
        }
        for (int i = k+1; i < A.n; i++) {
            for (int j = k+1; j < A.n; j++) {
                A.elements[i][j] -= A.elements[k][j] * A.elements[k][i];
            }
        }
        for (int j = k+1; j < A.n; j++) {
            A.elements[j][k] = 0.0;
        }
    }
}

// Parallel kernel to normalize row k
__global__ void parallel_cholesky_row_normalize(Matrix A, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute sqrt of pivot in first thread only
    __shared__ double sqrt_pivot;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sqrt_pivot = sqrt(A.elements[k][k]);
    }
    __syncthreads();
    
    // Normalize row k from column k to n
    if (j >= k && j < A.n) {
        A.elements[k][j] = A.elements[k][j] / sqrt_pivot;
    }
}

// Parallel kernel to update submatrix after elimination of column k
__global__ void parallel_cholesky_update(Matrix A, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update elements in the submatrix below and to the right of (k,k)
    if (i >= k+1 && i < A.n && j >= k+1 && j < A.n) {
        A.elements[i][j] -= A.elements[k][j] * A.elements[k][i];
    }
}

// Parallel kernel to zero out lower triangular part of column k
__global__ void parallel_cholesky_zero_lower(Matrix A, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= k+1 && j < A.n) {
        A.elements[j][k] = 0.0;
    }
}

// Host routines ..............................................................

Matrix cholesky_factorization(Matrix&);
Matrix product_with_transpose(Matrix& R);
int compare_matrix(Matrix&, Matrix&);
Matrix clone_matrix(Matrix& A); 
void initialize_spd_matrix(Matrix&, double);
Matrix create_matrix(int, int); 
void free_matrix_memory(Matrix&);
void print_matrix(Matrix&); 
void check_error(cudaError_t, int); 
void print_device_properties(); 

// Cholesky factorization on the host (provided for reference only)
Matrix cholesky_factorization(Matrix& A) {
    double sqrt_pivot;
    Matrix R = clone_matrix(A);
    for (int k = 0; k < R.n; k++) {
        sqrt_pivot = sqrt(R.elements[k][k]);
        for (int j = k; j < R.n; j++) {
            R.elements[k][j] = R.elements[k][j]/sqrt_pivot;
        }
        for (int i = k+1; i < R.n; i++) {
            for (int j = k+1; j < R.n; j++) {
                R.elements[i][j] -= R.elements[k][j] * R.elements[k][i];
            }
        }
        for (int j = k+1; j < R.n; j++) {
            R.elements[j][k] = 0.0;
        }
    }
    return R;
}

// Product with transpose - return C = R' * R
Matrix product_with_transpose(Matrix& R) {
    Matrix C = create_matrix(R.n,R.n); 
    for (int i = 0; i < R.n; i++) {
        for (int j = 0; j < R.n; j++) {
            C.elements[i][j] = 0.0;
            for (int k = 0; k < R.n; k++)
                C.elements[i][j] += R.elements[k][i]*R.elements[k][j];
        }
    }
    return C;
}

// Compare if matrix is identical to another
int compare_matrix(Matrix& A, Matrix& B) {
    int error = 0;
    if (A.n != B.n) return 1;
    for (int i = 0; i < A.n; i++) {
        for (int j = 0; j < A.n; j++) {
            if (fabs(A.elements[i][j] - B.elements[i][j]) > TOL) error = 1;
        }
    }
    return error;
}   

// Clone matrix 
Matrix clone_matrix(Matrix& A) {
    Matrix C = create_matrix(A.n, A.n);
    for (int i = 0; i < C.n; i++) {
        for (int j = 0; j < C.n; j++) {
            C.elements[i][j] = A.elements[i][j];
        }
    }
    return C;
}

// Initialize an SPD matrix (for testing factorization routine)
void initialize_spd_matrix(Matrix& A, double delta) {
    double value;
    for (int i = 0; i < A.n; i++) {
        A.elements[i][i] = delta;
    }
    for (int i = 0; i < A.n; i++) {
        for (int j = i+1; j < A.n; j++) {
            value =(double)(rand())/(double)(RAND_MAX);
            A.elements[i][j] = value;
            A.elements[j][i] = value;
            A.elements[i][i] += fabs(A.elements[i][j]);
            A.elements[j][j] += fabs(A.elements[i][j]);
        }
    }
}

// Create new matrix
Matrix create_matrix(int num_rows, int num_cols) {
    Matrix A;
    A.n = num_rows;
    A.elements = new double *[A.n];
    A.array = new double[A.n*A.n];
    for (int i = 0; i < A.n; i++) A.elements[i] = &(A.array[i*A.n]);
    return A;
}

// Delete matrix arrays 
void free_matrix_memory(Matrix& A) {
    delete[] A.elements;
    delete[] A.array;
}

// Print matrix
void print_matrix(Matrix& A) {
    printf("\n... Printing matrix ... \n");
    for (int i = 0; i < A.n; i++) {
        for (int j = 0; j < A.n; j++) {
            printf(" %8.4f", A.elements[i][j]);
        }
        printf("\n");
    }
}

// Generic error
void check_error(cudaError_t err, int type) {
    if (err != cudaSuccess) {
        switch(type) {
            case ERR_MALLOC: 
                fprintf(stderr, "Failed cudaMalloc (error code %s)!\n", cudaGetErrorString(err));
                break;
            case ERR_MEMCPY: 
                fprintf(stderr, "Failed cudaMemcpy (error code %s)!\n", cudaGetErrorString(err));
                break;
            case ERR_KERNEL: 
                fprintf(stderr, "Failed kernel launch (error code %s)!\n", cudaGetErrorString(err));
                break;
        }
        exit(0);
    }
}

// Print device properties
void print_device_properties() {
    int i, deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    printf("------------------------------------------------------------\n");
    printf("Number of GPU devices found = %d\n", deviceCount);
    for ( i = 0; i < deviceCount; ++i ) {
        cudaGetDeviceProperties(&deviceProp, i);
        printf("[Device: %1d] Compute Capability %d.%d.\n", i, deviceProp.major, deviceProp.minor);
        printf(" ... multiprocessor count  = %d\n", deviceProp.multiProcessorCount);
        printf(" ... max threads per multiprocessor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf(" ... max threads per block = %d\n", deviceProp.maxThreadsPerBlock);
        printf(" ... max block dimension   = %d, %d, %d (along x, y, z)\n",
                deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf(" ... max grid size         = %d, %d, %d (along x, y, z)\n",
                deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf(" ... warp size             = %d\n", deviceProp.warpSize);
        printf(" ... clock rate            = %d MHz\n", deviceProp.clockRate/1000);
    }
    printf("------------------------------------------------------------\n");
}

// Main Program ................................................................

int main(int argc, char *argv[]) {

    cudaError_t err = cudaSuccess;

    // Timing variables
    cudaEvent_t start, stop;
    float time_serial, time_parallel;

    // Print device properties
    print_device_properties();

    // Get device information and set device to use
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(0);
    } else {
        printf("Warning: No GPU device found ... results may be incorrect\n");
    }

    // Timing initializations
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Read input, validate
    if (argc != 2) {
        printf("Need one integer as input \n");
        printf("Use: <executable_name> <matrix_size>\n");
        exit(0);
    }
    int matrix_size = atoi(argv[argc-1]);
    if (matrix_size > MAX_MATRIX_SIZE) {
        printf("Maximum matrix size allowed: %d.\n", MAX_MATRIX_SIZE);
        exit(0);
    };

    // Initialize matrix A
    Matrix A = create_matrix(matrix_size, matrix_size); 
    initialize_spd_matrix(A, 1.0);

    // Create a copy of A on the device
    Matrix dA;
    dA.n = A.n;
    
    // Allocate linear arrays on device
    size_t size_elements = dA.n*sizeof(double *);
    size_t size_array = dA.n*dA.n*sizeof(double);
    err = cudaMalloc(&dA.elements, size_elements); check_error(err, ERR_MALLOC); 
    err = cudaMalloc(&dA.array, size_array); check_error(err, ERR_MALLOC);

    // Copy matrix elements to device
    err = cudaMemcpy(dA.array, A.array, size_array, cudaMemcpyHostToDevice); check_error(err, ERR_MEMCPY);

    // Initialize row pointer array dA.elements on device
    device_create_matrix_on_device<<<1,1>>>(dA); 
    err = cudaGetLastError(); check_error(err, ERR_KERNEL);

    // ========== SERIAL VERSION (for comparison) ==========
    printf("\n========== Running Serial Version ==========\n");
    cudaEventRecord(start, 0);
    
    device_cholesky_factorization<<<1,1>>>(dA);
    err = cudaGetLastError(); check_error(err, ERR_KERNEL);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_serial, start, stop);

    // Copy result matrix from device to host
    Matrix R_serial = create_matrix(dA.n, dA.n);
    err = cudaMemcpy(R_serial.array, dA.array, size_array, cudaMemcpyDeviceToHost); 
    check_error(err, ERR_MEMCPY);

    // Verify serial result
    Matrix RtR_serial = product_with_transpose(R_serial);
    int error_serial = compare_matrix(A, RtR_serial);
    
    if (error_serial != 0) {
        printf("+++  Serial: Houston, we have a problem!\n");
    } else {
        printf("+++  Serial: Matrix successfully factored\n"); 
        printf("     Matrix size: %d, GPU execution time: %8.4f ms\n", A.n, time_serial); 
    }

    // ========== PARALLEL VERSION ==========
    printf("\n========== Running Parallel Version ==========\n");
    
    // Re-copy original matrix to device for parallel version
    err = cudaMemcpy(dA.array, A.array, size_array, cudaMemcpyHostToDevice); 
    check_error(err, ERR_MEMCPY);
    
    // Define block and grid dimensions
    int BLOCK_SIZE = 256;  // Can be tuned (128, 256, 512, etc.)
    dim3 blockDim1D(BLOCK_SIZE);
    dim3 gridDim1D((matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    int BLOCK_SIZE_2D = 16;  // Can be tuned (8, 16, 32)
    dim3 blockDim2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridDim2D((matrix_size + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
                   (matrix_size + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);

    cudaEventRecord(start, 0);
    
    // Parallel Cholesky factorization
    for (int k = 0; k < matrix_size; k++) {
        // Step 1: Normalize row k
        parallel_cholesky_row_normalize<<<gridDim1D, blockDim1D>>>(dA, k);
        cudaDeviceSynchronize();
        
        // Step 2: Update submatrix
        parallel_cholesky_update<<<gridDim2D, blockDim2D>>>(dA, k);
        cudaDeviceSynchronize();
        
        // Step 3: Zero lower triangular part
        parallel_cholesky_zero_lower<<<gridDim1D, blockDim1D>>>(dA, k);
        cudaDeviceSynchronize();
    }
    
    err = cudaGetLastError(); check_error(err, ERR_KERNEL);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_parallel, start, stop);

    // Copy result matrix from device to host
    Matrix R_parallel = create_matrix(dA.n, dA.n);
    err = cudaMemcpy(R_parallel.array, dA.array, size_array, cudaMemcpyDeviceToHost); 
    check_error(err, ERR_MEMCPY);

    // Verify parallel result
    Matrix RtR_parallel = product_with_transpose(R_parallel);
    int error_parallel = compare_matrix(A, RtR_parallel);

    if (error_parallel != 0) {
        printf("+++  Parallel: Houston, we have a problem!\n");
    } else {
        printf("+++  Parallel: Matrix successfully factored\n"); 
        printf("     Matrix size: %d, GPU execution time: %8.4f ms\n", A.n, time_parallel); 
        printf("     Speedup: %.2fx\n", time_serial / time_parallel);
        printf("     Block sizes used: 1D=%d, 2D=%dx%d\n", BLOCK_SIZE, BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    }

    // Free allocated arrays
    free_matrix_memory(A); 
    free_matrix_memory(R_serial); 
    free_matrix_memory(RtR_serial);
    free_matrix_memory(R_parallel); 
    free_matrix_memory(RtR_parallel);
    
    cudaFree(dA.elements);
    cudaFree(dA.array);
    
    return 0;
}