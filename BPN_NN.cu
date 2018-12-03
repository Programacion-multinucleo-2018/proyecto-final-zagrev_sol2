/*
 * File:   BPN_NN.cu
 * Author: Cynthia Castillo
 * Student ID: A01374530
 *
 */

#include <iostream>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>

#include "common.h"
#include <cuda_runtime.h>

__device__ double sigmoidalGradiente(double z) {
  double g_z = 1.0 / (1.0 + std::exp(-z));

  return g_z * (1 - g_z);
}

__global__ void matrixMult(double *A, double *B, double *C, const int C_rows, const int C_cols)
{
    unsigned int ix_cols = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy_rows = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy_rows * C_rows + ix_cols;

    if (ix_cols < C_rows && iy_rows < C_cols)
    {
    	C[idx] = 0;
    	for (int shared_dim = 0; shared_dim < C_rows; shared_dim++)
    		//dot product
    		C[idx] += A[shared_dim * C_cols + ix_cols] * B[iy_rows * C_cols + shared_dim];  

    	C[idx] = sigmoidalGradiente(C[idx]);
    }
}

int main(int argc, char *argv[])
{
	//Parameters
	double sigma = 0.12; //weights range value
	int training_size = 1; // Number of examples
	int training_attr_size = 400; // number of features (400 values (20 x 20 pixels))
	int hidden_layer_size = 25;
	int output_layer_size = 10; //10 digits

	std::string filePath;
	
	if(argc < 2){
		filePath = "nueve.txt";
		training_size = 1;
	}
  	else{
  	  	filePath = argv[1];
  	  	training_size = std::stoi(argv[2]);
  	}

  	std::cout << "\nFile used: " << filePath <<  "\nNumber of examples: " << training_size << std::endl;


  	int lineCounter = 0;
  	// FILE READER
    int counter = 0;
	std::ifstream file(filePath);
	std::string line;
	char split_char = ' ';

	std::vector<std::string> values_s;
	std::vector<std::string> labels_s;

	while (std::getline(file, line))
	{
		counter = 0;

		std::istringstream split(line);
		for (std::string each; std::getline(split, each, split_char); counter <= training_attr_size ? values_s.push_back(each) : labels_s.push_back(each))
			counter++;

		lineCounter++;
	    //std::cout << "Counter: " << lineCounter << std::endl;

	    if (lineCounter == training_size)
	      break;
	}

	// TRANSFORM DATA TO DOUBLE
	std::vector<double> values(values_s.size());
	std::vector<double> labels(labels_s.size());

	std::transform(values_s.begin(), values_s.end(), values.begin(), [](const std::string& val)
	{
	    return std::stod(val);
	});

	std::transform(labels_s.begin(), labels_s.end(), labels.begin(), [](const std::string& val)
	{
	    return std::stoi(val);
	});

	// GENERATE WEIGHTS
	std::random_device rand_dev;
  	std::default_random_engine generator(rand_dev());
	std::uniform_real_distribution<double> distribution(-sigma, sigma);

	std::vector<double> w1;
	std::vector<double> w2;

	//Initialize W1
	for(int i = 0; i < hidden_layer_size * training_attr_size; i++)
		w1.push_back(distribution(generator));

	//Initialize W2
	for(int i = 0; i < output_layer_size * hidden_layer_size; i++)
		w2.push_back(distribution(generator));

	std::cout << "Size: " << values_s.size()/400 << std::endl;
	std::cout << "Values: " << values.size()/400 << std::endl;

	std::cout << "Size: " << output_layer_size * hidden_layer_size << std::endl;
	std::cout << "W2: " << w2.size() << std::endl;

	std::cout << "Size: " << hidden_layer_size * training_attr_size << std::endl;
	std::cout << "W1: " << w1.size() << std::endl;

	//Filling bias
	//std::vector<double> b1(training_attr_size);
	//std::vector<double> b2(hidden_layer_size);
	//std::fill (b1.begin(), b1.end(), 1);
	//std::fill (b2.begin(), b2.end(), 1);

	//std::vector<double> z2(output_layer_size * training_size);

	// Z2 will be the Results matrix
	double *h_values, *h_labels, *h_w1, *h_w2, *h_z2;
  	h_values = (double *)malloc(values.size() * sizeof(double));
	h_labels = (double *)malloc(labels.size() * sizeof(double));
	h_w1 = (double *)malloc(w1.size() * sizeof(double));
	h_w2 = (double *)malloc(w2.size() * sizeof(double));
	h_z2 = (double *)calloc(output_layer_size * training_size, sizeof(double));
	//h_b1 = (double *)malloc(b1.size());
	//h_b2 = (double *)malloc(b2.size());

	h_values = values.data();
	h_labels = labels.data();
	h_w1 = w1.data();
	h_w2 = w2.data();
	//h_z1 = z1.data();
	//h_z2 = z2.data();
	//h_b1 = b1.data();
	//h_b2 = b2.data();

	double *d_values, *d_labels, *d_w1, *d_w2, *d_z1, *d_z2, *d_b1, *d_b2;
	SAFE_CALL(cudaMalloc((void **)&d_values, values.size() * sizeof(double)), "Error allocating d_values");
	SAFE_CALL(cudaMalloc((void **)&d_labels, labels.size() * sizeof(double)), "Error allocating d_labels");
	SAFE_CALL(cudaMalloc((void **)&d_w1, w1.size() * sizeof(double)), "Error allocating d_w1");
	SAFE_CALL(cudaMalloc((void **)&d_w2, w2.size() * sizeof(double)), "Error allocating d_w2");
	SAFE_CALL(cudaMalloc((void **)&d_z1, hidden_layer_size * training_size * sizeof(double)), "Error allocating d_z1");
	SAFE_CALL(cudaMalloc((void **)&d_z2, output_layer_size * training_size * sizeof(double)), "Error allocating d_z2");
	SAFE_CALL(cudaMalloc((void **)&d_b1, training_attr_size * sizeof(double)), "Error allocating d_b1");
	SAFE_CALL(cudaMalloc((void **)&d_b2, hidden_layer_size * sizeof(double)), "Error allocating d_b2");

	// transfer data from host to device
	SAFE_CALL(cudaMemcpy(d_values, h_values, values.size() * sizeof(double), cudaMemcpyHostToDevice), "Error copying d_values");
	SAFE_CALL(cudaMemcpy(d_labels, h_labels, labels.size() * sizeof(double), cudaMemcpyHostToDevice), "Error copying d_labels");
	SAFE_CALL(cudaMemcpy(d_w1, h_w1, w1.size() * sizeof(double), cudaMemcpyHostToDevice), "Error copying d_w1");
	SAFE_CALL(cudaMemcpy(d_w2, h_w2, w2.size() * sizeof(double), cudaMemcpyHostToDevice), "Error copying d_w2");
	SAFE_CALL(cudaMemcpy(d_z2, h_z2, output_layer_size * training_size * sizeof(double), cudaMemcpyHostToDevice), "Error copying d_z2");

    
	// INVOKE KERNEL
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((training_attr_size + block.x - 1) / block.x, (training_size + block.y - 1) / block.y);

 	matrixMult<<<grid, block>>>(d_w1, d_values, d_z1, training_size, hidden_layer_size);
 	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
 	printf("multMatrix <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
 	// SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

 	double *h_z1;
	h_z1 = (double *)malloc(hidden_layer_size * training_size * sizeof(double));
	SAFE_CALL(cudaMemcpy(h_z1, d_z1, hidden_layer_size * training_size * sizeof(double), cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed");
	/*
	for (int i = 0; i <  hidden_layer_size * training_size; i++){
		if (i%25 == 0)
			std::cout << std::endl;
	std::cout << h_z1[i] << " ";
	}
	std::cout << std::endl << std::endl;
	*/

 	//matrixMult<<<grid, block>>>(d_w2, d_z1, d_z2, output_layer_size, training_size, hidden_layer_size);

 	//SAFE_CALL(cudaMemcpy(h_z2, d_z2, z2.size(), cudaMemcpyDeviceToHost), "CUDA Memcpy Device to host Failed");

	return 0;	
}











   	/*
  std::cout << " ** W1 ** " << std::endl;
  counter = 0;
  for (auto i = w1.begin(); i != w1.end(); ++i){
    std::cout << *i << ' ';
  counter++;
  if (counter == 24)
    std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;
	
	counter = 0;
	std::cout << " ** Values ** " << endl;
	for (auto i = values.begin(); i != values.end(); ++i){
    std::cout << *i << ' ';
    counter++;
	if (counter == 399)
		std::cout << endl << endl;
	}
	std::cout << endl << endl;
	*/

	/*
    double *h_z1;
    h_z1 = (double *)malloc(hidden_layer_size * training_size);
    SAFE_CALL(cudaMemcpy(h_z1, d_z1, training_size * hidden_layer_size, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

    for (int i = 0; i <  hidden_layer_size * training_size; i++){
    	if (i%25 == 0)
    		std::cout << std::endl;
		std::cout << h_z1[i] << " ";
    }
    std::cout << std::endl << std::endl;
    */