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

__device__ float sigmoidalGradiente(float z) {
  float g_z = 1.0 / (1.0 + std::exp(-z));

  return g_z * (1 - g_z);
}

__global__ void matrixMult(float *A, float *B, float *C, const int A_rows, const int B_cols)
{
    unsigned int ix_rows = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy_cols = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy_cols * A_rows + ix_rows;

    if (ix_rows < A_rows && iy_cols < B_cols)
    {
    	C[idx] = 0;
    	for (int shared_dim = 0; shared_dim < A_rows; shared_dim++)
    		//dot product
    		C[idx] += A[shared_dim * B_cols + ix_rows] * B[iy_cols * B_cols + shared_dim];  

    	C[idx] = sigmoidalGradiente(C[idx]);
    }
}

__global__ void transpose(float *A, float *C, const int rows, const int cols)
{
    unsigned int ix_rows = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy_cols = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = ix_rows * rows + iy_cols;
    unsigned int idy = iy_cols * cols + ix_rows;

    if (ix_rows < cols && iy_cols < rows)
    	C[idy] = A[idx];
}

__global__ void elemWise(float *A, float *B, float *C, const int opt, const int rows, const int cols)
{
    unsigned int ix_rows = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy_cols = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy_cols * rows + ix_rows;

    if (ix_rows < rows && iy_cols < cols)
    {
    	if (opt == '-')
    		C[idx] = A[idx] - B[idx];

    	else if (opt == '*')
    		C[idx] = A[idx] * B[idx];
    }
}

__global__ void costFunc(float *A, float *Y, float *cost, const int rows, const int cols)
{
    unsigned int ix_rows = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy_cols = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy_cols * rows + ix_rows;

    if (ix_rows < rows && iy_cols < cols)
    	cost[0] += (-Y[idx] * log(A[idx])) - ((1 - Y[idx])*log(1 - A[idx]));
}

__global__ void predict(float *X, float *W1, float *W2, float *y, const int rows, const int cols)
{

}

int main(int argc, char *argv[])
{
	//Parameters
	
	float sigma = 0.12; //weights range value
	int training_size = 1; // Number of examples
	int training_attr_size = 400; // number of features (400 values (20 x 20 pixels))
	int hidden_layer_size = 25;
	int output_layer_size = 10; //10 digits

	std::string filePath;
	
	if(argc < 3){
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

	// TRANSFORM DATA TO float
	std::vector<float> values(values_s.size());
	std::vector<int> labels(labels_s.size());

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
	std::uniform_real_distribution<float> distribution(-sigma, sigma);

	std::vector<float> w1;
	std::vector<float> w2;

	//Initialize W1
	for(int i = 0; i < training_attr_size * hidden_layer_size; i++)
		w1.push_back(distribution(generator));

	//Initialize W2
	for(int i = 0; i < hidden_layer_size * output_layer_size; i++)
		w2.push_back(distribution(generator));

	//std::cout << "Size: " << values_s.size()/400 << std::endl;
	//std::cout << "Values: " << values.size()/400 << std::endl;

	//std::cout << "Size: " << hidden_layer_size * output_layer_size << std::endl;
	//std::cout << "W2: " << w2.size() << std::endl;

	//std::cout << "Size: " << training_attr_size * hidden_layer_size << std::endl;
	//std::cout << "W1: " << w1.size() << std::endl;

	//Filling bias
	//std::vector<float> b1(training_attr_size);
	//std::vector<float> b2(hidden_layer_size);
	//std::fill (b1.begin(), b1.end(), 1);
	//std::fill (b2.begin(), b2.end(), 1);

	//std::vector<float> z2(training_size * output_layer_size);

	// Z2 will be the Results matrix
	float *h_values, *h_labels, *h_w1, *h_w2, *h_z2, *h_outputs, *h_J;
  	h_values = (float *)malloc(values.size() * sizeof(float));
	h_labels = (float *)malloc(labels.size() * sizeof(float));
	h_w1 = (float *)malloc(w1.size() * sizeof(float));
	h_w2 = (float *)malloc(w2.size() * sizeof(float));
	h_z2 = (float *)calloc(training_size * output_layer_size, sizeof(float));
	h_outputs = (float *)calloc(training_size * output_layer_size, sizeof(float));
	h_J = (float *)calloc(1, sizeof(float));
	for (int i = 0; i < training_size; i++)
		h_outputs[(labels[i]-1) * training_size + i] = 1;

	
	//h_b1 = (float *)malloc(b1.size());
	//h_b2 = (float *)malloc(b2.size());

	h_values = values.data();
	//h_labels = labels.data();
	h_w1 = w1.data();
	h_w2 = w2.data();
	//h_z1 = z1.data();
	//h_z2 = z2.data();
	//h_b1 = b1.data();
	//h_b2 = b2.data();

	float *d_values, *d_labels, *d_w1, *d_w2, *d_z1, *d_z2, *d_b1, *d_b2, *d_outputs, *d_delta3, *d_delta2, *d_transW2, *d_J;
	SAFE_CALL(cudaMalloc((void **)&d_values, values.size() * sizeof(float)), "Error allocating d_values");
	SAFE_CALL(cudaMalloc((void **)&d_labels, labels.size() * sizeof(float)), "Error allocating d_labels");
	SAFE_CALL(cudaMalloc((void **)&d_w1, w1.size() * sizeof(float)), "Error allocating d_w1");
	SAFE_CALL(cudaMalloc((void **)&d_w2, w2.size() * sizeof(float)), "Error allocating d_w2");
	SAFE_CALL(cudaMalloc((void **)&d_z1, training_size * hidden_layer_size * sizeof(float)), "Error allocating d_z1");
	SAFE_CALL(cudaMalloc((void **)&d_z2, training_size * output_layer_size * sizeof(float)), "Error allocating d_z2");
	SAFE_CALL(cudaMalloc((void **)&d_b1, training_attr_size * sizeof(float)), "Error allocating d_b1");
	SAFE_CALL(cudaMalloc((void **)&d_b2, hidden_layer_size * sizeof(float)), "Error allocating d_b2");
	SAFE_CALL(cudaMalloc((void **)&d_outputs, training_size * output_layer_size * sizeof(float)), "Error allocating d_outputs");
	SAFE_CALL(cudaMalloc((void **)&d_delta3, training_size * output_layer_size * sizeof(float)), "Error allocating d_delta3");
	SAFE_CALL(cudaMalloc((void **)&d_delta2, training_size * hidden_layer_size * sizeof(float)), "Error allocating d_delta2");
	SAFE_CALL(cudaMalloc((void **)&d_transW2, output_layer_size * hidden_layer_size * sizeof(float)), "Error allocating d_transW2");
	SAFE_CALL(cudaMalloc((void **)&d_J, 1 * sizeof(float)), "Error allocating d_J");

	// transfer data from host to device
	SAFE_CALL(cudaMemcpy(d_values, h_values, values.size() * sizeof(float), cudaMemcpyHostToDevice), "Error copying d_values");
	SAFE_CALL(cudaMemcpy(d_labels, h_labels, labels.size() * sizeof(float), cudaMemcpyHostToDevice), "Error copying d_labels");
	SAFE_CALL(cudaMemcpy(d_w1, h_w1, w1.size() * sizeof(float), cudaMemcpyHostToDevice), "Error copying d_w1");
	SAFE_CALL(cudaMemcpy(d_w2, h_w2, w2.size() * sizeof(float), cudaMemcpyHostToDevice), "Error copying d_w2");
	SAFE_CALL(cudaMemcpy(d_z2, h_z2, training_size * output_layer_size * sizeof(float), cudaMemcpyHostToDevice), "Error copying d_z2");
	SAFE_CALL(cudaMemcpy(d_outputs, h_outputs, training_size * output_layer_size * sizeof(float), cudaMemcpyHostToDevice), "Error copying d_outputs");
	cudaMemset(d_delta3, 0, training_size * output_layer_size*sizeof(float));
	cudaMemset(d_delta2, 0, training_size * hidden_layer_size*sizeof(float));
	cudaMemset(d_transW2, 0, output_layer_size * hidden_layer_size*sizeof(float));
	cudaMemset(d_J, 0, 1*sizeof(float));
    
	// INVOKE KERNEL
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((training_attr_size + block.x - 1) / block.x, (training_size + block.y - 1) / block.y);


	std::cout << "\nForward Propagation\n";

 	matrixMult<<<grid, block>>>(d_values, d_w1, d_z1, training_size, hidden_layer_size);
 	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel 1");
    SAFE_CALL(cudaGetLastError(), "Error with last error");
 	matrixMult<<<grid, block>>>(d_z1, d_w2, d_z2, training_size, output_layer_size);
 	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel 2");
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    //Calculating Cost
    std::cout << "\nCalculating Cost\n";
    costFunc<<<grid, block>>>(d_z2, d_outputs, d_J, training_size, output_layer_size);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel Cost");
    SAFE_CALL(cudaGetLastError(), "Error with last error");
    //SAFE_CALL(cudaMemcpy(h_J, d_J, 1 * sizeof(float), cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed");
    //std::cout << "Costo: " << h_J[0] << std::endl;

    //BACKPROPAGATION
    std::cout << "\nBackPropagation\n";
 	elemWise<<<grid, block>>>(d_z2, d_outputs, d_delta3, '-', training_size, output_layer_size);
 	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel 3");
    SAFE_CALL(cudaGetLastError(), "Error with last error");
    transpose<<<grid, block>>>(d_w2, d_transW2, hidden_layer_size, output_layer_size);
 	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel 4");
    SAFE_CALL(cudaGetLastError(), "Error with last error");
    matrixMult<<<grid, block>>>(d_delta3, d_transW2, d_delta2, training_size, hidden_layer_size);
 	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel 5");
    SAFE_CALL(cudaGetLastError(), "Error with last error");
    elemWise<<<grid, block>>>(d_delta2, d_z1, d_delta2, '*', training_size, hidden_layer_size);
 	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel 6");
    SAFE_CALL(cudaGetLastError(), "Error with last error");


    //Updating Weights


 	//float *h_w3;
	//h_w3 = (float *)malloc(output_layer_size * hidden_layer_size * sizeof(float));
	//SAFE_CALL(cudaMemcpy(h_w3, d_transW2, output_layer_size * hidden_layer_size * sizeof(float), cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed");
    
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
  if (counter%hidden_layer_size == 0)
    std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;
	
	counter = 0;
	std::cout << " ** Values ** " << std::endl;
	for (auto i = values.begin(); i != values.end(); ++i){
    std::cout << *i << ' ';
    counter++;
	if (counter%training_attr_size == 0)
		std::cout << std::endl << std::endl;
	}
	std::cout << std::endl << std::endl;
	

    for (int i = 0; i <  training_size * hidden_layer_size; i++){
    	if (i%hidden_layer_size == 0)
    		std::cout << std::endl;
		std::cout << h_z1[i] << " ";
    }
    std::cout << std::endl << std::endl;
    */