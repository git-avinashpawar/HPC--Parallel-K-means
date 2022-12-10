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
int K_g;
int no_iter_g;
int *data_pt_g;
float *iter_cen_g;
int *data_pt_clust_g;

void kmeans(int N, int K, int *data_pt, int **data_pt_clust_id, float **iter_centroids, int *num_iter)
{
    no_pt_g = N; // Initialize global variables
    K_g = K;
    no_iter_g = *num_iter;
    data_pt_g = data_pt;

    // Allocating space N data points (4 per N)
    *data_pt_clust_id = (int *)malloc(N * 4 * sizeof(int));
    data_pt_clust_g = *data_pt_clust_id;

    // Allocating space of 3*K units for each iteration as we are dealing with 3D data
    iter_cen_g = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

    // Assigning first K points to be initial centroids (Can also assign Randomly)
    for (int i = 0; i < K; i++)
    {
        iter_cen_g[i * 3] = data_pt[i * 3];
        iter_cen_g[i * 3 + 1] = data_pt[i * 3 + 1];
        iter_cen_g[i * 3 + 2] = data_pt[i * 3 + 2];
    }

    // Printing initial centroid
    std::printf("Initial Centroids are:\n");
    for (int i = 0; i < K; i++)
        std::printf("initial centroid #%d: %f,%f,%f\n", i + 1, iter_cen_g[i * 3], iter_cen_g[i * 3 + 1], iter_cen_g[i * 3 + 2]);

    // Runing k-means sequential function
    std::printf("Starting Sequential Implementation of K-Means\n");
    double min_dist, current_dist;
    int *pt_to_clust_id = (int *)malloc(no_pt_g * sizeof(int));     // Cluster id associated with each point
    float *clust_pt_sum = (float *)malloc(K_g * 3 * sizeof(float)); // Cluster location or centroid (x,y,z) coordinates for K clusters in a iteration
    int *pts_in_clust_count = (int *)malloc(K_g * sizeof(int));     // No. of points in a cluster for a iteration

    int iter = 0;
    while ((delta_global > THRESHOLD) && (iter < MAX_ITER))
    { // Initializing to 0.0
        for (int i = 0; i < K_g * 3; i++)
            clust_pt_sum[i] = 0.0;
        for (int i = 0; i < K_g; i++)
            pts_in_clust_count[i] = 0;

        for (int i = 0; i < no_pt_g; i++)
        {
            // Assign these points to their nearest cluster
            min_dist = DBL_MAX;
            for (int j = 0; j < K_g; j++)
            {
                // compute distance from each centroids
                current_dist = pow((double)(iter_cen_g[(iter * K_g + j) * 3] - (float)data_pt_g[i * 3]), 2.0) +
                               pow((double)(iter_cen_g[(iter * K_g + j) * 3 + 1] - (float)data_pt_g[i * 3 + 1]), 2.0) +
                               pow((double)(iter_cen_g[(iter * K_g + j) * 3 + 2] - (float)data_pt_g[i * 3 + 2]), 2.0);
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    pt_to_clust_id[i] = j; // assign point to cluster which has lowest distance
                }
            }

            // Update count number of points inside cluster
            pts_in_clust_count[pt_to_clust_id[i]] += 1;

            // Update local sum of cluster data points
            clust_pt_sum[pt_to_clust_id[i] * 3] += (float)data_pt_g[i * 3];
            clust_pt_sum[pt_to_clust_id[i] * 3 + 1] += (float)data_pt_g[i * 3 + 1];
            clust_pt_sum[pt_to_clust_id[i] * 3 + 2] += (float)data_pt_g[i * 3 + 2];
        }

        // Compute centroid from clust_pt_sum and store inside iter_cen_g in a iteration
        for (int i = 0; i < K_g; i++)
        {
            assert(pts_in_clust_count[i] != 0);
            iter_cen_g[((iter + 1) * K_g + i) * 3] = clust_pt_sum[i * 3] / pts_in_clust_count[i];
            iter_cen_g[((iter + 1) * K_g + i) * 3 + 1] = clust_pt_sum[i * 3 + 1] / pts_in_clust_count[i];
            iter_cen_g[((iter + 1) * K_g + i) * 3 + 2] = clust_pt_sum[i * 3 + 2] / pts_in_clust_count[i];
        }

        // Delta is the sum of squared distance between centroid of previous and current iterations.
        double t_delta = 0.0;
        for (int i = 0; i < K_g; i++)
        {
            t_delta += (iter_cen_g[((iter + 1) * K_g + i) * 3] - iter_cen_g[((iter)*K_g + i) * 3]) *
                           (iter_cen_g[((iter + 1) * K_g + i) * 3] - iter_cen_g[((iter)*K_g + i) * 3]) +
                       (iter_cen_g[((iter + 1) * K_g + i) * 3 + 1] - iter_cen_g[((iter)*K_g + i) * 3 + 1]) *
                           (iter_cen_g[((iter + 1) * K_g + i) * 3 + 1] - iter_cen_g[((iter)*K_g + i) * 3 + 1]) +
                       (iter_cen_g[((iter + 1) * K_g + i) * 3 + 2] - iter_cen_g[((iter)*K_g + i) * 3 + 2]) *
                           (iter_cen_g[((iter + 1) * K_g + i) * 3 + 2] - iter_cen_g[((iter)*K_g + i) * 3 + 2]);
        }
        delta_global = t_delta; // Update delta_global with new delta

        iter++;
    }
    no_iter_g = iter; // Store the number of iterations in global variable

    // Assign points to final calculated centroids
    for (int i = 0; i < no_pt_g; i++)
    {
        // Assign points to clusters
        data_pt_clust_g[i * 4] = data_pt_g[i * 3];
        data_pt_clust_g[i * 4 + 1] = data_pt_g[i * 3 + 1];
        data_pt_clust_g[i * 4 + 2] = data_pt_g[i * 3 + 2];
        data_pt_clust_g[i * 4 + 3] = pt_to_clust_id[i];
        assert(pt_to_clust_id[i] >= 0 && pt_to_clust_id[i] < K_g);
    }

    // Record number of iterations and store iter_cen_g data into iter_centroids
    *num_iter = no_iter_g;
    int centroids_size = (*num_iter + 1) * K * 3;
    std::printf("Number of total iterations :%d\n", no_iter_g);
    *iter_centroids = (float *)calloc(centroids_size, sizeof(float));

    for (int i = 0; i < centroids_size; i++)
        (*iter_centroids)[i] = iter_cen_g[i];

    // Print final calculated centroids
    std::printf("Final calculated Centroids are:\n");
    for (int i = 0; i < K; i++)
        std::printf("centroid #%d: %f,%f,%f\n", i + 1, (*iter_centroids)[((*num_iter) * K + i) * 3], (*iter_centroids)[((*num_iter) * K + i) * 3 + 1], (*iter_centroids)[((*num_iter) * K + i) * 3 + 2]);
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
    int N;                 // Number of datapoints
    int K = 5;             // Number of clusters
    int *data_pt;          // Data points
    int no_itr;            // no of iterations performed by algorithm
    int *clust_pt;         // clustered data points
    float *iter_centroids; // centroids of each iteration

    char *fname;
    int x;
    if (argc == 3)
    {
        fname = argv[1];
        x = atoi(argv[2]);
    }
    else
    {
        std::printf("Please enter filename, datapoints\n");
        return 0;
    }

    // Read Dataset
    data_in(fname, &N, &data_pt);

    // Run algorithm
    auto start_time = std::chrono::high_resolution_clock::now();
    kmeans(N, K, data_pt, &clust_pt, &iter_centroids, &no_itr);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Creating filenames for different dataset
    char datapoint_f[10];
    std::snprintf(datapoint_f, 10, "%d", x);

    char cluster_f[100] = "cluster_";
    std::strcat(cluster_f, datapoint_f);
    std::strcat(cluster_f, ".csv");

    char centroid_f[100] = "centroid_";
    std::strcat(centroid_f, datapoint_f);
    std::strcat(centroid_f, ".csv");

    // saved as : x, y, z, clust_id
    clusters_to_csv(cluster_f, N, clust_pt);

    // saved as : x1, y1, z1, x2, y2, z2 ...
    centroids_to_csv(centroid_f, K, no_itr, iter_centroids);

    // Store computation time
    auto time = end_time - start_time;
    double computation_time = time.count();
    std::printf("Time Taken: %lf \n", computation_time);

    char time_res[100] = "compute_time.csv";

    FILE *writefile = std::fopen(time_res, "a");
    std::fprintf(writefile, "Data: %s, Time:  %f\n", datapoint_f, computation_time);
    std::fclose(writefile);

    return 0;
}