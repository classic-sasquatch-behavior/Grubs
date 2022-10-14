#pragma once

#include"external_libs.h"


namespace Launch{

    static cudaError_t Error;

    static dim3 num_blocks;
    static dim3 threads_per_block;

    static void kernel_1d(int size){
        Launch::threads_per_block = {1024,1,1};

        uint grid_size = (size - (size % 1024))/1024;

        num_blocks = {grid_size + 1, 1, 1};

    }

    static void kernel_2d(int x_size, int y_size){
        threads_per_block = {32,32,1};

        uint block_x = (x_size - (x_size % 32))/32;
        uint block_y = (y_size - (y_size % 32))/32;

        num_blocks = {block_x + 1, block_y + 1, 1};
    }

    static void kernel_3d(int x_size, int y_size, int z_size){
        threads_per_block = {16,16,4};

        uint block_x = (x_size - (x_size % 16))/16;
        uint block_y = (y_size - (y_size % 16))/16;
        uint block_z = (z_size - (z_size % 4))/4;
        
        num_blocks = {block_x + 1, block_y + 1, block_z + 1};

    }


}

#define LAUNCH Launch::num_blocks, Launch::threads_per_block