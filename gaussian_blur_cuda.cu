#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <sys/mman.h>

#define DIV_ROUND_UP(n, d)  (((n) + (d) - 1) / (d))

#define cuda_check(ret) _cuda_check((ret), __FILE__, __LINE__)
inline void _cuda_check(cudaError_t ret, const char *file, int line)
{
	if (ret != cudaSuccess) {
		fprintf(stderr, "CudaErr: %s (%s:%d)\n", cudaGetErrorString(ret), file, line);
		exit(1);
	}
}

void read_image(char **Input, size_t *width, size_t *height, size_t *max, unsigned char **pixels)
{
    FILE *input_fd = fopen(*Input, "r+");
    if (!input_fd)
    {
        fprintf(stderr, "file does not exist");
        exit(1);
    }
    fseek(input_fd, 3, SEEK_CUR); //skip P5\n
    //printf("current line pos: %lu\n", ftell(input_fd));
    char c;
    if (fscanf(input_fd, " %lu %lu %lu%c", width, height, max, &c) != 4)
    {
        fprintf(stderr, "Info reading error\n");
        exit(1);
    }
    //printf("current line pos: %lu\n", ftell(input_fd));
    // fprintf(stdout, "width %lu, height %lu, max %lu, c%d\n", *width, *height, *max, c == '\n');

    //read the image into memory
    *pixels = (unsigned char*)calloc((*width) * (*height), sizeof(unsigned char));
    if (fread(*pixels, sizeof(unsigned char), (*width) * (*height), input_fd) != (*width) * (*height))
    {
        fprintf(stderr, "Image reading error\n");
        exit(1);
    }

    fclose(input_fd);
}

// Write pgm format
void write_image(unsigned char *g_map, size_t width, size_t height, size_t max, char *Output)
{
    char *info = (char*)calloc(100, sizeof(char));
    sprintf(info, "P5\n%lu %lu\n%lu\n", width, height, max);
    FILE *output_fd = fopen(Output, "w");
    fwrite(info, sizeof(char), strlen(info), output_fd);
    fwrite(g_map, sizeof(unsigned char), width * height, output_fd);
    fprintf(output_fd, "\n");
    fclose(output_fd);
}

float *gaussian_blur_matrix(size_t order, size_t sigma)
{

    float *matrix = (float*)calloc(order * order, sizeof(float));
    size_t x_0, y_0;
    x_0 = order / 2;
    y_0 = x_0;
    for (size_t y = 0; y < order; y++)
    {
        for (size_t x = 0; x < order; x++)
        {
            float x_dis = (float)x_0 - (float)x;
            float y_dis = (float)y_0 - (float)y;
            //printf("x_dis %f 7_dis %f\n", x_dis, y_dis);
            matrix[y * order + x] = expf((-1) * (x_dis * x_dis + y_dis * y_dis) / (2 * sigma * sigma));
        }
    }
    return matrix;
}

//__shared__ width, height, order

__device__ float blur_kernel_old(unsigned char* pixels, long width, long height, long x, long y, long order, float* mat){
    long start_x = x - order / 2;
    long start_y = y - order / 2;
    long curr_x, curr_y;
    float val = 0;
    for (long j = 0; j < order; j++)
    {
        curr_y = start_y + j;
        for (long i = 0; i < order; i++)
        {
            curr_x = start_x + i;

            if ((curr_x < 0 || curr_x >= width) && (curr_y < 0 || curr_y >= height))
            {
                if (curr_x < 0)
                    curr_x = 0;
                else if (curr_x >= width)
                    curr_x = width - 1;
                if (curr_y < 0)
                    curr_y = 0;
                else if (curr_y >= height)
                    curr_y = height - 1;
            }
            else if (curr_x < 0 || curr_x >= width)
            {
                if (curr_x < 0)
                    curr_x = 0;
                else if (curr_x >= width)
                    curr_x = width - 1;
            }
            else if (curr_y < 0 || curr_y >= height)
            {
                if (curr_y < 0)
                    curr_y = 0;
                else if (curr_y >= height)
                    curr_y = height - 1;
            }
            val += pixels[curr_y * width + curr_x] * mat[j * order + i];
        }
    }
    return val;
}

__device__ float blur_kernel(unsigned char* pixels, long width, long height, long x, long y, long order, float* mat, float k){
    long start_x = x - order / 2;
    long start_y = y - order / 2;
    long curr_x, curr_y;
    float val = 0;
    for (long j = 0; j < order; j++)
    {
        curr_y = start_y + j;
        if (curr_y < 0)
            curr_y = 0;

        else if (curr_y >= height)
            curr_y = height - 1;

        long pixel_row = curr_y * width;
        long mat_row = j * order;

        for (long i = 0; i < order; i++)
        {
            curr_x = start_x + i;

            if (curr_x < 0)
                curr_x = 0;
                
            else if (curr_x >= width)
                curr_x = width - 1;

            val += pixels[pixel_row + curr_x] * mat[mat_row + i];
        }
    }
    val /= k;
    return val;
}

__global__ void gaussian_blur_kernel_old(unsigned char *pixels, float *mat, unsigned char *output, long width, long height, float max, long order, float k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("row %d, col %d\n" row, col);
    // Discard out of bound coordinates
    if(row >= height || col >= width)
        return;

    //switched row and col
    float val = blur_kernel(pixels, width, height, col, row, order, mat, k);
    output[row * width + col] = (unsigned char)(val);
}

__global__ void gaussian_blur_kernel(unsigned char *pixels, float *mat, unsigned char *output, long width, long height, float max, long order, float k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Discard out of bound coordinates
    if(row >= height || col >= width)
        return;

    long start_y = row - order / 2;
    long start_x = col - order / 2;
    long curr_x, curr_y;
    float val = 0;
    for (long j = 0; j < order; j++)
    {
        curr_y = start_y + j;
        if (curr_y < 0)
            curr_y = 0;

        else if (curr_y >= height)
            curr_y = height - 1;

        long pixel_row = curr_y * width;
        long mat_row = j * order;

        for (long i = 0; i < order; i++)
        {
            curr_x = start_x + i;

            if (curr_x < 0)
                curr_x = 0;

            else if (curr_x >= width)
                curr_x = width - 1;

            val += pixels[pixel_row + curr_x] * mat[mat_row + i];
        }
    }
    val /= k;
    output[row * width + col] = (unsigned char)(val);
}



unsigned char *gaussian_blur_apply_cuda(unsigned char *pixels, long width, long height, float sigma, float max)
{
    // long order = (sigma * 6 % 2) == 0 ? sigma * 6 + 1 : sigma * 6;

    unsigned char *out_pixels_device, *out_pixels, *pixels_device;
    float *mat_device;

    //new
    long order = (long)(ceilf(sigma * 6));
    order = order % 2 == 0 ? order + 1 : order;
    // printf("order: %long\n", order);

    // Allocate needed matrices locally
    float *mat = gaussian_blur_matrix(order, sigma); //__shared__
    out_pixels = (unsigned char*)calloc(width * height, sizeof(unsigned char));

    //new
    float k = (2 * M_PI * sigma * sigma);
    
    int image_size = width * height * sizeof(unsigned char);
    
    //  allocate memory on device for needed matrices
    cuda_check(cudaMalloc(&out_pixels_device, image_size));
    cuda_check(cudaMalloc(&pixels_device, image_size));
    cuda_check(cudaMalloc(&mat_device, order * order * sizeof(float)));

    // Copy data onto device
    cuda_check(cudaMemcpy(pixels_device, pixels, image_size, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(mat_device, mat, order * order * sizeof(float), cudaMemcpyHostToDevice));

    //Invoke kernel function
    dim3 block_dim(32, 32);
    dim3 grid_dim(DIV_ROUND_UP(width, block_dim.x), DIV_ROUND_UP(height, block_dim.y));
    
    //  Catch errors
    gaussian_blur_kernel_old<<<grid_dim, block_dim>>>(pixels_device, mat_device, out_pixels_device, width, height, max, order, k);
    cuda_check(cudaPeekAtLastError());      /* Catch configuration errors */
	cuda_check(cudaDeviceSynchronize());    /* Catch execution errors */

    //  Copy output from device back to host
    cuda_check(cudaMemcpy(out_pixels, out_pixels_device, image_size, cudaMemcpyDeviceToHost));

    // Free memory on device
    cuda_check(cudaFree(out_pixels_device));
	cuda_check(cudaFree(mat_device));
	cuda_check(cudaFree(pixels_device));

    // unsigned char *out_pixels = calloc(width * height, sizeof(unsigned char));

    free(mat);
    return out_pixels;
}


int main(int argc, char *argv[])
{
    char *Input = (char*)calloc(100, sizeof(char));
    unsigned char *pixels = NULL;
    unsigned char *out_pixels = NULL;
    char *Output = (char*)calloc(100, sizeof(char));
    float sigma;

    size_t width, height, max;
    if (argc != 4)
    {
        fprintf(stderr,
                "Usage: ./mandelbrot_serial order xcenter ycenter zoom cutoff\n");
        exit(1);
    }
    sscanf(argv[1], " %s", Input);
    sscanf(argv[2], " %s ", Output);
    sscanf(argv[3], " %f ", &sigma);
    //fprintf(stdout, "%s %s %lu\n", Input, Output, sigma);
    //read in original image
    read_image(&Input, &width, &height, &max, &pixels);

    //gaussian_blur the image
    out_pixels = gaussian_blur_apply_cuda(pixels, (long)width, (long)height, sigma, (float)max);
    //write out the final image
    write_image(out_pixels, width, height, max, Output); //writes the unprocessed image

    free(Input);
    free(Output);
    free(pixels);
    return 0;
}