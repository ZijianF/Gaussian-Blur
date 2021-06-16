#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <sys/mman.h>

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
    fprintf(stdout, "width %lu, height %lu, max %lu, c%d\n", *width, *height, *max, c == '\n');

    //read the image into memory
    *pixels = calloc((*width) * (*height), sizeof(unsigned char));
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
    char *info = calloc(100, sizeof(char));
    sprintf(info, "P5\n%lu %lu\n%lu\n", width, height, max);
    char *output_fn = calloc(strlen(Output), sizeof(char));
    memcpy(output_fn, Output, strlen(Output));
    fprintf(stdout, "%s\n", output_fn);
    FILE *output_fd = fopen(Output, "w");
    fwrite(info, sizeof(char), strlen(info), output_fd);
    fwrite(g_map, sizeof(unsigned char), width * height, output_fd);
    fprintf(output_fd, "\n");
    fclose(output_fd);
}

//the matrix without being divided by  (2 * M_PI * sigma * sigma)
float *gaussian_blur_matrix(size_t order, float sigma)
{

    float *matrix = calloc(order * order, sizeof(float));
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

unsigned char *gaussian_blur_apply(unsigned char *pixels, long width, long height, float sigma)
{
    long order = (long)(ceilf(sigma * 6));
    order = order % 2 == 0 ? order + 1 : order;
    float *mat = gaussian_blur_matrix(order, sigma); //__shared__
    unsigned char *out_pixels = calloc(width * height, sizeof(unsigned char));
    float k = (2 * M_PI * sigma * sigma);
    //loop through each pixel and apply the matrix to each peripheral pixels
    //replace the not existed pixels with the closest extant pixel
    for (long y = 0; y < height; y++)
    {
        long start_y = y - order / 2;
        for (long x = 0; x < width; x++)
        {
            long start_x = x - order / 2;
            long curr_x, curr_y;
            float val = 0;
            for (long j = 0; j < order; j++)
            {
                for (long i = 0; i < order; i++)
                {

                    curr_y = start_y + j;
                    if (curr_y < 0)
                        curr_y = 0;
                    else if (curr_y >= height)
                        curr_y = height - 1;
                    long pixel_row = curr_y * width;
                    long mat_row = j * order;
                    curr_x = start_x + i;

                    if (curr_x < 0)
                        curr_x = 0;
                    else if (curr_x >= width)
                        curr_x = width - 1;

                    val += pixels[pixel_row + curr_x] * mat[mat_row + i];
                }
            }
            val /= k;
            out_pixels[y * width + x] = (unsigned char)(val);
        }
    }
    free(mat);
    return out_pixels;
}

int main(int argc, char *argv[])
{
    char *Input = calloc(100, sizeof(char));
    unsigned char *pixels = NULL;
    unsigned char *out_pixels = NULL;
    char *Output = calloc(100, sizeof(char));
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
    out_pixels = gaussian_blur_apply(pixels, (long)width, (long)height, sigma);
    //write out the final image
    write_image(out_pixels, width, height, max, Output); //writes the unprocessed image

    free(Input);
    free(Output);
    free(pixels);
    return 0;
}