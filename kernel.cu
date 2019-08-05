
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>
#include <cufft.h>
#include <iostream>
#include <iomanip>
#include<vector>
#include <fstream>
#include <windows.h>
#include <omp.h>

using namespace std;

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void comparator(string songDir, string sampleDir);
double* cudaSizeCalc(cufftComplex* array, int size);
double cudaCosSim(double* song, double* sample, int size);
vector<string> get_file_names_in_folder(string folder);


__global__ void complexSizeKernel(cufftComplex *cmplx, double *dblArray, int size) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		dblArray[i] = sqrt(pow(cmplx[i].x, 2) + pow(cmplx[i].y, 2));
}

__global__ void normalization(double *dblArray, double *newdblArray, double* minMax, int size) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		newdblArray[i] = (dblArray[i] - minMax[0]) / (minMax[1] - minMax[0]);
}

__global__ void maxByReduction(double *array, double *max, int size) {
	extern __shared__ double maxData[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		maxData[tid] = array[i];
	else
		maxData[tid] = 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s) {
			if (maxData[tid] < maxData[tid + s])
				maxData[tid] = maxData[tid + s];
		}
		__syncthreads();
	}
	// write max for this block to global mem
	if (tid == 0) {
		max[blockIdx.x] = maxData[0];
	}
}

__global__ void minByReduction(double *array, double *min, int size) {
	extern __shared__ double minData[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		minData[tid] = array[i];
	else
		minData[tid] = 10000;


	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s) {
			if (minData[tid] > minData[tid + s])
				minData[tid] = minData[tid + s];
		}
		__syncthreads();
	}
	// write min for this block to global mem
	if (tid == 0) {
		min[blockIdx.x] = minData[0];
	}
}

__global__ void multiplyReduction(double *song, double *sample, double *result, int size) {
	extern __shared__ double multiply[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
		multiply[tid] = song[i] * sample[i];
	else
		multiply[tid] = 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s) {
			multiply[tid] += multiply[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) {
		result[blockIdx.x] = multiply[0];
	}
}

__global__ void vectorSize(double *song, double *result, int arraySize) {
	extern __shared__ double size[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < arraySize)
		size[tid] = pow(song[i], 2);
	else
		size[tid] = 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s) {
			size[tid] += size[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) {
		result[blockIdx.x] = size[0];
	}
}

double cosineSimilarity(double* a, double* b, int sampleSize) {
	double cosmul = 0.0;
	double asize = 0.0;
	double bsize = 0.0;
	for (int i = 0; i < sampleSize; i++) {
		cosmul += a[i] * b[i];
		asize += pow(a[i], 2);
		bsize += pow(b[i], 2);
	}
	cosmul = abs(cosmul);
	asize = sqrt(asize);
	bsize = sqrt(bsize);

	return cosmul / (asize * bsize);
}

double calculateSize(cufftComplex in) {
	return sqrt(pow(in.x, 2) + pow(in.y, 2));
}


double* normalize(cufftComplex* array, int size) {
	double min = 100000.0;
	double max = -100000.0;
	double* newArray = (double *)malloc(sizeof(double)*size);
	int i = 0;
	for (i = 0; i < size; i++) {
		newArray[i] = calculateSize(array[i]);
		if (min > newArray[i]) {
			min = newArray[i];
		}
		if (max < newArray[i]) {
			max = newArray[i];
		}
	}

	for (i = 0; i < size; i++) {
		newArray[i] = (newArray[i] - min) / (max - min);
	}

	return newArray;
}

cufftComplex* FourierCalc(cufftComplex* cmplxArray, int size) {
	cufftHandle plan;
	cufftComplex *data;

	cudaErrchk(cudaMalloc((void**)&data, sizeof(cufftComplex)*size));

	cudaErrchk(cudaMemcpy(data, cmplxArray, sizeof(cufftComplex)*size, cudaMemcpyHostToDevice));

	if (cufftPlan1d(&plan, size, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		exit(1);
	}

	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		exit(1);
	}


	cudaErrchk(cudaDeviceSynchronize());

	cudaErrchk(cudaMemcpy(cmplxArray, data, sizeof(cufftComplex)*size, cudaMemcpyDeviceToHost));

	cufftDestroy(plan);
	cudaFree(data);
	return cmplxArray;
}


cufftComplex* readData(const char* path, int &size) {

	ifstream inFile;
	float fdata;

	inFile.open(path);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1);
	}

	size = 0;
	while (inFile >> fdata) {
		size++;
	}
	inFile.close();
	inFile.open(path);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1);
	}

	cufftComplex *audioData = (cufftComplex *)malloc(sizeof(cufftComplex)*size);

	int i = 0;
	while (inFile >> fdata && i < size) {

		audioData[i].x = fdata;
		audioData[i].y = 0;
		i++;
	}

	inFile.close();
	return audioData;
}

int main(int argc, char* argv[]) {

	string Songdirectory = argv[1];//"C:\\Users\\ahmad\\Desktop\\cudame\\final\\songs\\";
	string sampledirectory = argv[2];//"C:\\Users\\ahmad\\Desktop\\cudame\\final\\songs\\6080.txt";
	comparator(Songdirectory, sampledirectory);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void comparator(string songDir, string sampleDir)
{
	vector<string> sampleNames = get_file_names_in_folder(sampleDir);
	for (int z = 0; z < sampleNames.size(); z++)
	{
		int songSize = 0;
		int sampleSize = 0;
		string similarSong;
		double mainMax = 0;
		cufftComplex * sampleData = readData((sampleDir + sampleNames.at(z)).c_str(), sampleSize);

		cufftComplex * sampleFourier = FourierCalc(sampleData, sampleSize);

		double* sampleSizeData = cudaSizeCalc(sampleFourier, sampleSize);

		int index = 100000;// (int)sampleSize / 4000;

		cufftComplex *songSection = (cufftComplex *)malloc(sizeof(cufftComplex)*sampleSize);

		vector<string> textNames = get_file_names_in_folder(songDir);

		for (int i = 0; i < textNames.size(); i++)
		{
			//printf("%s: \n", textNames.at(i));
			cufftComplex * song = readData((songDir + textNames.at(i)).c_str(), songSize);
			double max = 0;

			int x = 0;

			while (x < songSize) {
				if (songSize  <  sampleSize + x)
					break;


#pragma omp parallel for 
				for (int j = x; j < x + sampleSize; j++) {
					songSection[j - x] = song[j];
				}

				songSection = FourierCalc(songSection, sampleSize);
				double* songSizeSection = cudaSizeCalc(songSection, sampleSize);
				double val = cudaCosSim(songSizeSection, sampleSizeData, sampleSize);
				if (max < val) {
					max = val;
				}
				x += index;
			}
			// printf("song name: %s , sample name : %s %f\n", textNames.at(i), sampleNames.at(z), max);
			if (mainMax < max) {
				mainMax = max;
				similarSong = textNames.at(i);
			}
		}
		if (mainMax > 0.5)
			printf("similar song name: %s , sample name : %s %f\n", similarSong, sampleNames.at(z), mainMax);
		else
			printf("for  %s no song found\n", sampleNames.at(z));
	}
}

vector<string> get_file_names_in_folder(string folder) {
	vector<string> names;
	string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

double* cudaSizeCalc(cufftComplex* cmplxarray, int size) {
	int threadNum = 128;
	int blockNum = ceil(size / threadNum);

	cufftComplex *dev_CmplxArray;
	double *dev_dblArray;

	double* host_newDbl = (double *)malloc(sizeof(double)*  size);


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrchk(cudaSetDevice(0));

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaErrchk(cudaMalloc((void**)&dev_dblArray, size * sizeof(double)));

	cudaErrchk(cudaMalloc((void**)&dev_CmplxArray, size * sizeof(cufftComplex)));

	// Copy input vectors from host memory to GPU buffers.
	cudaErrchk(cudaMemcpy(dev_CmplxArray, cmplxarray, size * sizeof(cufftComplex), cudaMemcpyHostToDevice));

	//block numbers , thread bnumbers 

	complexSizeKernel << <blockNum, threadNum >> > (dev_CmplxArray, dev_dblArray, size);

	// Check for any errors launching the kernel
	cudaErrchk(cudaGetLastError());

	cudaErrchk(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	cudaErrchk(cudaMemcpy(host_newDbl, dev_dblArray, size * sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(dev_CmplxArray);
	cudaFree(dev_dblArray);

	return host_newDbl;
}

double cudaCosSim(double* song, double* sample, int size) {

	int threadNum = 128;
	int blockNum = ceil(size / threadNum);

	const int streamsNum = 2;
	cudaStream_t streams[streamsNum];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);

	double *dev_song;
	double *dev_sample;
	double *dev_result;
	double *dev_sng;
	double *dev_smp;

	double multiplication = 0.0;
	double size1 = 0.0;
	double size2 = 0.0;
	double similarity;

	double* host_result = (double *)malloc(sizeof(double)*  blockNum);
	double* host_sng = (double *)malloc(sizeof(double)*  blockNum);
	double* host_smp = (double *)malloc(sizeof(double)*  blockNum);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrchk(cudaSetDevice(0));
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaErrchk(cudaMalloc((void**)&dev_song, size * sizeof(double)));

	cudaErrchk(cudaMalloc((void**)&dev_sample, size * sizeof(double)));

	cudaErrchk(cudaMalloc((void**)&dev_result, blockNum * sizeof(double)));

	cudaErrchk(cudaMalloc((void**)&dev_sng, blockNum * sizeof(double)));

	cudaErrchk(cudaMalloc((void**)&dev_smp, blockNum * sizeof(double)));

	// Copy input vectors from host memory to GPU buffers.
	cudaErrchk(cudaMemcpy(dev_sample, sample, size * sizeof(double), cudaMemcpyHostToDevice));

	// Copy input vectors from host memory to GPU buffers.
	cudaErrchk(cudaMemcpy(dev_song, song, size * sizeof(double), cudaMemcpyHostToDevice));

	//block numbers , thread bnumbers 
	size_t shm_size = threadNum * sizeof(double);
	multiplyReduction << <blockNum, threadNum, shm_size >> > (dev_song, dev_sample, dev_result, size);

	// Check for any errors launching the kernel
	cudaErrchk(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrchk(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	cudaErrchk(cudaMemcpy(host_result, dev_result, blockNum * sizeof(double), cudaMemcpyDeviceToHost));

	cudaErrchk(cudaGetLastError());

	vectorSize << <blockNum, threadNum, shm_size, streams[0] >> > (dev_sample, dev_smp, size);

	// Check for any errors launching the kernel
	cudaErrchk(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrchk(cudaDeviceSynchronize());

	cudaErrchk(cudaMemcpy(host_smp, dev_smp, blockNum * sizeof(double), cudaMemcpyDeviceToHost));

	cudaErrchk(cudaGetLastError());

	vectorSize << <blockNum, threadNum, shm_size, streams[1] >> > (dev_song, dev_sng, size);

	// Check for any errors launching the kernel
	cudaErrchk(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrchk(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	cudaErrchk(cudaMemcpy(host_sng, dev_sng, blockNum * sizeof(double), cudaMemcpyDeviceToHost));

	// Check for any errors launching the kernel
	cudaErrchk(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaErrchk(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.

#pragma omp parallel for reduction( +: multiplication, size1 , size2 )
	for (int i = 0; i < blockNum; i++) {
		multiplication += host_result[i];
		size1 += host_sng[i];
		size2 += host_smp[i];
	}

	multiplication = abs(multiplication);
	size1 = sqrt(size1);
	size2 = sqrt(size2);

	similarity = multiplication / (size1 * size2);

	cudaFree(dev_song);
	cudaFree(dev_sample);
	cudaFree(dev_result);
	cudaFree(dev_sng);
	cudaFree(dev_smp);

	return similarity;
}
