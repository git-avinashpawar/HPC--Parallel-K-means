#include <float.h>
#include <cmath>
#include <assert.h>
#include <string>
#include <cstring>
#include <omp.h>
#include <cstdlib>
#include <stdio.h>
#include <chrono>
#include <limits>

#define MAX_ITER 100
#define THRESHOLD 0.00001

double delta_global = THRESHOLD + 1; // Defined global delta
int no_pt_g;                         // Define Global variables
int no_threads_g;
int K_g;
int no_iter_g;
int *data_pt_g;
float *iter_cen_g;
int *data_pt_clust_g;
int **iter_clust_count_g;

void kmeans_threaded(int *tid)
{
    int *id = (int *)tid;

    // Distributing Data to each Thread
    int data_ln_thread = no_pt_g / no_threads_g;
    int start = (*id) * data_ln_thread;
    int end = start + data_ln_thread;

    if (end + data_ln_thread > no_pt_g)
    {
        end = no_pt_g;
        data_ln_thread = no_pt_g - start;
    }

    // std::printf("Thread ID:%d, data start:%d, data end:%d\n", *id, start, end);

    double current_dist;

    int *point_to_cluster_id = (int *)malloc(data_ln_thread * sizeof(int)); // Cluster id associated with each point
    float *clust_pt_sum = (float *)malloc(K_g * 3 * sizeof(float));         // Cluster location or centroid (x,y,z) coordinates for K clusters in a iteration
    int *points_inside_cluster_count = (int *)malloc(K_g * sizeof(int));    // No. of points in a cluster for a iteration

    int iter_counter = 0;
    while ((delta_global > THRESHOLD) && (iter_counter < MAX_ITER))
    {
        // Initializing to 0.0
        for (int i = 0; i < K_g * 3; i++)
            clust_pt_sum[i] = 0.0;
        for (int i = 0; i < K_g; i++)
            points_inside_cluster_count[i] = 0;

        for (int i = start; i < end; i++)
        {
            // Assign these points to their nearest cluster
            double min_dist = DBL_MAX;
            for (int j = 0; j < K_g; j++)
            { // compute distance from each centroids
                current_dist = pow((double)(iter_cen_g[(iter_counter * K_g + j) * 3] - (float)data_pt_g[i * 3]), 2.0) +
                               pow((double)(iter_cen_g[(iter_counter * K_g + j) * 3 + 1] - (float)data_pt_g[i * 3 + 1]), 2.0) +
                               pow((double)(iter_cen_g[(iter_counter * K_g + j) * 3 + 2] - (float)data_pt_g[i * 3 + 2]), 2.0);
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    point_to_cluster_id[i - start] = j; // assign point to cluster which has lowest distance
                }
            }

            // Update count number of points inside cluster
            points_inside_cluster_count[point_to_cluster_id[i - start]] += 1;

            // Update local sum of cluster data points
            clust_pt_sum[point_to_cluster_id[i - start] * 3] += (float)data_pt_g[i * 3];
            clust_pt_sum[point_to_cluster_id[i - start] * 3 + 1] += (float)data_pt_g[i * 3 + 1];
            clust_pt_sum[point_to_cluster_id[i - start] * 3 + 2] += (float)data_pt_g[i * 3 + 2];
        }

        // Update globaal counters after each thread arrival
        // formula:: (prev_iter_centroid_global * prev_iter_cluster_count + new_thread_clust_pt_sum) / (new_thread_cluster_count + prev_iter_cluster_count)
#pragma omp critical
        {
            for (int i = 0; i < K_g; i++)
            {
                if (points_inside_cluster_count[i] == 0)
                {
                    printf("Unlikely situation occured!\n");
                    continue;
                }
                iter_cen_g[((iter_counter + 1) * K_g + i) * 3] = (iter_cen_g[((iter_counter + 1) * K_g + i) * 3] * iter_clust_count_g[iter_counter][i] + clust_pt_sum[i * 3]) / (float)(iter_clust_count_g[iter_counter][i] + points_inside_cluster_count[i]);
                iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 1] = (iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 1] * iter_clust_count_g[iter_counter][i] + clust_pt_sum[i * 3 + 1]) / (float)(iter_clust_count_g[iter_counter][i] + points_inside_cluster_count[i]);
                iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 2] = (iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 2] * iter_clust_count_g[iter_counter][i] + clust_pt_sum[i * 3 + 2]) / (float)(iter_clust_count_g[iter_counter][i] + points_inside_cluster_count[i]);

                iter_clust_count_g[iter_counter][i] += points_inside_cluster_count[i];
            }
        }
        // Delta is the sum of squared distance between centroid of previous and current iterations.
#pragma omp barrier // Wait for all threads and execute for first thread only

        if (*id == 0)
        {
            double temp_delta = 0.0;
            for (int i = 0; i < K_g; i++)
            {
                temp_delta += (iter_cen_g[((iter_counter + 1) * K_g + i) * 3] - iter_cen_g[((iter_counter)*K_g + i) * 3]) *
                                  (iter_cen_g[((iter_counter + 1) * K_g + i) * 3] - iter_cen_g[((iter_counter)*K_g + i) * 3]) +
                              (iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 1] - iter_cen_g[((iter_counter)*K_g + i) * 3 + 1]) *
                                  (iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 1] - iter_cen_g[((iter_counter)*K_g + i) * 3 + 1]) +
                              (iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 2] - iter_cen_g[((iter_counter)*K_g + i) * 3 + 2]) *
                                  (iter_cen_g[((iter_counter + 1) * K_g + i) * 3 + 2] - iter_cen_g[((iter_counter)*K_g + i) * 3 + 2]);
            }
            delta_global = temp_delta; // Update delta_global with new delta
            no_iter_g++;
        }

#pragma omp barrier // Wait for all thread and update the iter_counter by +1
        iter_counter++;
    }

    // Assign points to final calculated centroids
    for (int i = start; i < end; i++)
    {
        // Assign points to clusters
        data_pt_clust_g[i * 4] = data_pt_g[i * 3];
        data_pt_clust_g[i * 4 + 1] = data_pt_g[i * 3 + 1];
        data_pt_clust_g[i * 4 + 2] = data_pt_g[i * 3 + 2];
        data_pt_clust_g[i * 4 + 3] = point_to_cluster_id[i - start];
        assert(point_to_cluster_id[i - start] >= 0 && point_to_cluster_id[i - start] < K_g);
    }
}

void kmeans_parallel(int num_threads, int N, int K, int *data_pt, int **data_pt_cluster_id, float **iter_centroids, int *no_itr)
{
    // Initialize global variables
    no_pt_g = N;
    no_threads_g = num_threads;
    no_iter_g = 0;
    K_g = K;
    data_pt_g = data_pt;

    // Allocating space N data points (4 per N)
    *data_pt_cluster_id = (int *)malloc(N * 4 * sizeof(int));
    data_pt_clust_g = *data_pt_cluster_id;

    // Allocating space of 3*K units for each iteration as we are dealing with 3D data
    iter_cen_g = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

    // Assigning first K points to be initial centroids (Can also assign Randomly)
    for (int i = 0; i < K; i++)
    {
        iter_cen_g[i * 3] = data_pt[i * 3];
        iter_cen_g[i * 3 + 1] = data_pt[i * 3 + 1];
        iter_cen_g[i * 3 + 2] = data_pt[i * 3 + 2];
    }

    // Print initial centroids
    std::printf("Initial Centroids are:\n");
    for (int i = 0; i < K; i++)
        std::printf("initial centroid #%d: %f,%f,%f\n", i + 1, iter_cen_g[i * 3], iter_cen_g[i * 3 + 1], iter_cen_g[i * 3 + 2]);

    iter_clust_count_g = (int **)malloc(MAX_ITER * sizeof(int *));
    for (int i = 0; i < MAX_ITER; i++)
        iter_clust_count_g[i] = (int *)calloc(K, sizeof(int));

    // setting omp threads
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        // std::printf("Thread with id: %d created!\n", ID);
        kmeans_threaded(&ID);
    }

    // save number of itrations
    *no_itr = no_iter_g;

    // Record number of iterations and store iter_cen_g data into iter_centroids
    int iter_centroids_size = (*no_itr + 1) * K * 3;
    std::printf("Number of total iterations :%d\n", *no_itr);
    *iter_centroids = (float *)calloc(iter_centroids_size, sizeof(float));
    for (int i = 0; i < iter_centroids_size; i++)
        (*iter_centroids)[i] = iter_cen_g[i];

    // Print final calculated centroids
    std::printf("Final calculated Centroids are:\n");
    for (int i = 0; i < K; i++)
        std::printf("centroid #%d: %f,%f,%f\n", i + 1, (*iter_centroids)[((*no_itr) * K + i) * 3], (*iter_centroids)[((*no_itr) * K + i) * 3 + 1], (*iter_centroids)[((*no_itr) * K + i) * 3 + 2]);
}

// Function to read data from file
void data_in(const char *fname, int *N, int **data_pt)
{
    FILE *fin = std::fopen(fname, "r");
    std::fscanf(fin, "%d", N);
    *data_pt = (int *)malloc(sizeof(int) * ((*N) * 3));

    for (int i = 0; i < (*N) * 3; i++)
        std::fscanf(fin, "%d", (*data_pt + i));

    std::fclose(fin);
}

// Function save cluster data as csv
void clusters_to_csv(const char *cluster_f, int N, int *clust_pt)
{
    FILE *writefile = std::fopen(cluster_f, "w");
    std::fprintf(writefile, "x, y, z, c\n");
    for (int i = 0; i < N; i++)
        std::fprintf(writefile, "%d, %d, %d, %d\n", *(clust_pt + (i * 4)), *(clust_pt + (i * 4) + 1), *(clust_pt + (i * 4) + 2), *(clust_pt + (i * 4) + 3));

    std::fclose(writefile);
}

// Function save centroid data as csv
void centroids_to_csv(const char *centroid_f, int K, int no_itr, float *iter_centroids)
{
    FILE *writefile = std::fopen(centroid_f, "w");

    for (int i = 0; i < no_itr + 1; i++)
    {
        for (int j = 0; j < K; j++)
            std::fprintf(writefile, "%f %f %f\t ", *(iter_centroids + (i * K + j) * 3), *(iter_centroids + (i * K + j) * 3 + 1), *(iter_centroids + (i * K + j) * 3 + 2));
        std::fprintf(writefile, "\n");
    }

    std::fclose(writefile);
}

int main(int argc, char **argv)
{
    // Initialising Variables
    int N;                 // Number of data points
    int K = 5;             // Number of clusters to be formed
    int num_threads;       // Number of threads to be used
    int *data_pt;          // Data points
    int no_itr;            // Number of iterations performed by algorithm
    float *iter_centroids; // centroids of each iteration
    int *clust_pt;         // clustered data points

    char *fname;
    int x;
    if (argc == 4)
    {
        fname = argv[1];
        x = atoi(argv[2]);
        num_threads = atoi(argv[3]);
    }
    else
    {
        printf("Please enter filename, datapoints, no of threads\n");
        return 0;
    }

    // Read Dataset
    data_in(fname, &N, &data_pt);

    // Run algorithm
    auto start_time = std::chrono::high_resolution_clock::now();
    kmeans_parallel(num_threads, N, K, data_pt, &clust_pt, &iter_centroids, &no_itr);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Creating filenames for different threads and different dataset
    char num_threads_char[10];
    snprintf(num_threads_char, 10, "%d", num_threads);

    char datapoint_f[10];
    snprintf(datapoint_f, 10, "%d", x);

    char cluster_f[100] = "cluster_t_";
    strcat(cluster_f, num_threads_char);
    strcat(cluster_f, "_");
    strcat(cluster_f, datapoint_f);
    strcat(cluster_f, ".csv");

    char centroid_f[100] = "centroid_t_";
    strcat(centroid_f, num_threads_char);
    strcat(centroid_f, "_");
    strcat(centroid_f, datapoint_f);
    strcat(centroid_f, ".csv");

    // saved as : x, y, z, clust_id
    clusters_to_csv(cluster_f, N, clust_pt);

    // saved as : x1, y1, z1, x2, y2, z2 ...
    centroids_to_csv(centroid_f, K, no_itr, iter_centroids);

    // Store computation time
    auto time = end_time - start_time;
    double computation_time = time.count();
    printf("Time Taken: %lf \n", computation_time);

    char time_file_omp[100] = "compute_time_t_";
    strcat(time_file_omp, num_threads_char);
    strcat(time_file_omp, ".csv");

    // Store compute time
    FILE *writefile = fopen(time_file_omp, "a");
    fprintf(writefile, "Data: %s, Time:  %f, Threads : %s \n", datapoint_f, computation_time, num_threads_char);
    fclose(writefile);

    return 0;
}