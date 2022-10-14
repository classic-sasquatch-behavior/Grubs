#pragma once
#include "external_libs.h"



template<typename Type>
struct Device_Ptr {
    Type* data;
    const int dim[4];
    const int num_dims;
    const int size;
    const int bytesize;

    const int& x = dim[0];
    const int& y = dim[1];
    const int& z = dim[2];
    const int& w = dim[3];

    Device_Ptr(Matrix<Type> *input):data(input->device_data), num_dims(input->num_dims), size(input->size), bytesize(input->bytesize), dim(input->dim){}

    __device__ __inline__ Type& operator()(int x, int y = 0, int z = 0, int w = 0){
        return data[(((x * dim[1] + y) * dim[2] + z) * dim[3] + w];
    }

};

template<typename Type>
struct Matrix {

    Type* host_data;
    Type* device_data;
    int dim[4];
    int num_dims;
    int size;
    int bytesize;

    int& x = dim[0];
    int& y = dim[1];
    int& z = dim[2];
    int& w = dim[3];

    Matrix(){}

    Matrix(const std::vector<int> _dims, Type _constant = 0){
        num_dims = _dims;
        int _size = 1;
        for(int i = 0; i< num_dims; i++){
            dim[i] = _dims[i];
        }
        for(int i = num_dims; i < 4; i++){
            dim[i] = 1;
        }
        size = dim[0] * dim[1] * dim[2] * dim[3];
        bytesize = size * sizeof(Type);
        allocate();
        fill(_constant);
    }

    ~Matrix(){
        delete host_data;
        cudaFree(device_data);
    }

    Matrix(const Matrix<Type>& input):num_dims(input.num_dims), dim(input.dim), size(input.size), bytesize(input.bytesize){
        allocate();
        load(input.host_data);
    }

    void operator=(const Matrix<Type>& input):num_dims(input.num_dims), dim(input.dim), size(input.size), bytesize(input.bytesize){
        delete host_data;
        cudaFree(device_data);

        allocate();
        load(input.host_data);
    }

    operator Device_Ptr<Type>(){
        return Device_Ptr<Type>(this);
    }

    inline Type& operator()(int x, int y = 0, int z = 0, int w = 0){
        return host_data[(((x * dim[1] + y) * dim[2] + z) * dim[3] + w];
    }

    void allocate(){
        host_data = new Type[size];
        cudaMalloc((void**)&device_data, bytesize);
    }

    void load(Type* input){
        for(int i = 0; i < size; i++){
            host_data[i] = input[i];
        }
        upload();
    }

    void fill(int constant){
        cudaMemset(device_data, constant, bytesize);
        download();
    }

    void upload(){
        cudaMemcpy(device_data, host_data, bytesize, cudaMemcpyHostToDevice);
    }

    void download(){
        cudaMemcpy(host_data, device_data, bytesize, cudaMemcpyDeviceToHost);
    }

};



