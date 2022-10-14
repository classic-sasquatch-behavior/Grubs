#pragma once

#include<external_libs.h>
#include<Matrix/matrix.h>



namespace Cell {
    enum attribute {
        color_r = 0,
        color_g = 1,
        color_b = 2,
        attractor = 3,
        weight = 4,
        value = 5,
        move_maj = 6,
        move_min = 7,
    };

    const int num_attributes = 8;

}
 using namespace Cell;

__global__ void draw_environment(Device_Ptr<int> environment, Device_Ptr<uchar> output) {
    DIMS_2D(maj, min);
    BOUNDS_2D(environment.dim[0], environment.dim[1]);

    int environment_value = environment(maj, min);

    int polarity[3] = {0,0,0};

    if (environment_value > 0) { //positive pressure becomes red tinted
        polarity[0] += 20;
    }

    else if (environment_value < 0) { //negative pressure becomes blue tinted
        polarity[2] += 20;
    }

    int magnitude = fabsf(environment_value);

    for (int channel = 0; channel < 3; channel++) {
        output.data[(((channel * environment.dim[0]) + maj) * environment.dim[1]) + min] = fminf(logf(10 * magnitude), 90) + polarity[channel];
    }
}

//pretty goofy way to do this to be honest. But let's see how fast or slow it runs.
__global__ void draw_cells(Device_Ptr<int> cells, Device_Ptr<uchar> output) {
    DIMS_2D(maj, min);
    BOUNDS_2D(cells.dim[0], cells.dim[1]);

    if(cells(SELF, attribute::color_r) == 0 && cells(SELF, attribute::color_g) == 0 && cells(SELF, attribute::color_b) == 0) { return; }

    for (int channel = 0; channel < 3; channel++) {
        int channel_value = cells(maj, min, channel);

        output(maj, min, channel) = channel_value;
        //output.device_data[(((channel * cells.first_dim()) + maj ) * cells.second_dim()) + min] = channel_value;
    }
}

#define Random(_seed_, _min_, _max_) (((_seed_) % ((_max_) - (_min_)))+ (_min_))

__global__ void spawn(curandState* random_states, Device_Ptr<int> result) {
    DIMS_2D(maj, min);
    BOUNDS_2D(result.dim[0], result.dim[1]);

    int id = LINEAR_CAST(maj, min, result.dim[1]);

    curandState local_state = random_states[id];

    int random_0 = curand(&local_state);
    int random_1 = curand(&local_state);
    int random_2 = curand(&local_state);
    int random_3 = curand(&local_state);
    int random_4 = curand(&local_state); 
    int random_5 = curand(&local_state);

    result(SELF, attribute::color_r) = Random(random_0, 70, 255);
    result(SELF, attribute::color_g) = Random(random_1, 70, 255);
    result(SELF, attribute::color_b) = Random(random_2, 70, 255);
    result(SELF, attribute::value) = 50;
    result(SELF, attribute::attractor) = Random(random_3, -100000, 100000);
    result(SELF, attribute::weight) = Random(random_4, -10000, 10000);
    //result(SELF, attribute::freq_a) = Random(random_5, -100000, 100000);
    result(SELF, attribute::move_maj) = maj;
    result(SELF, attribute::move_min) = min;


}

//~15-16ms
__global__ void change_environment(Device_Ptr<int> environment, Device_Ptr<int> cells) {
    DIMS_2D(maj, min);
    BOUNDS_2D(environment.dim[0], environment.dim[1]);

    int value = cells(SELF, attribute::value);
    int weight = cells(SELF, attribute::weight);

    //int attraction = (cells(SELF, attribute::value) * cells(SELF, attribute::attractor))/100;
    FOR_MXN_INCLUSIVE(n_maj, n_min, 9, 9,
        atomicAdd(&environment(n_maj, n_min), (value * weight) / 100);
    );
}

__global__ void dampen_environment(const float damping_factor, Device_Ptr<int> environment) {
    DIMS_2D(maj, min);
    BOUNDS_2D(environment.dim[0], environment.dim[1]);

    float value = environment(maj, min);
    environment(maj, min) = -truncf(value * damping_factor);


}

__global__ void radiate_environment(Device_Ptr<int> environment) {
    DIMS_2D(maj, min);
    BOUNDS_2D(environment.dim[0], environment.dim[1]);

    int value = environment(SELF);
    FOR_MXN_EXCLUSIVE(n_maj, n_min, 3, 3, 
        
        atomicSub(&environment(n_maj, n_min), (value/100));
    
    )
}

__global__ void set_targets(Device_Ptr<int> environment, Device_Ptr<int> cells, Device_Ptr<int> targets) {
    DIMS_2D(maj, min);
    BOUNDS_2D(environment.dim[0], environment.dim[1]);


    int attractor = cells(SELF, attribute::attractor);
    int weight = cells(SELF, attribute::weight);
    if(attractor == 0 && weight == 0){return;}

    int largest_value = environment(maj, min) * attractor;
    int target_maj = maj;
    int target_min = min;

    FOR_NEIGHBOR(n_maj, n_min,
        int neighbor_value = environment(n_maj, n_min) * attractor;
        if ( neighbor_value > largest_value) {
            largest_value = neighbor_value;
            target_maj = n_maj;
            target_min = n_min;
        }
    );

    atomicAdd(&targets(target_maj, target_min), 1);
    cells(SELF, attribute::move_maj) = target_maj;
    cells(SELF, attribute::move_min) = target_min;
}

__global__ void conflict(Device_Ptr<int>cells, Device_Ptr<int>targets, Device_Ptr<int>future_cells, const int threshold = 3) {

    DIMS_2D(maj, min);
    BOUNDS_2D(cells.dim[0], cells.dim[1]);

    if (targets(maj, min) < 2) { return; }

    int participants[9][Cell::num_attributes - 2] = { 0 };
    int num_participants = 0;

    int highest_value = -1;
    int winning_cell = -1;
    int total_value = 0;

    FOR_3X3_INCLUSIVE(n_maj, n_min,

        if ((cells(n_maj, n_min, attribute::move_maj) != maj) || (cells(n_maj, n_min, attribute::move_min) != min)) { continue; }

        for (int attribute = 0; attribute < Cell::num_attributes - 2; attribute++) {
            participants[num_participants][attribute] = cells(n_maj, n_min, attribute);
        }
        total_value += participants[num_participants][attribute::value];

        if (participants[num_participants][attribute::value] > highest_value) {
            highest_value = participants[num_participants][attribute::value];
            winning_cell = num_participants;
        }

        num_participants++;
    );

    future_cells(SELF, attribute::value) = total_value;
    for (int i = 0; i < Cell::num_attributes - 3; i++) {
        future_cells(SELF, i) = participants[winning_cell][i];
    }
}

__global__ void move(Device_Ptr<int> environment, Device_Ptr<int> cells, Device_Ptr<int> targets, Device_Ptr<int> future_cells) {
    DIMS_2D(maj, min);
    BOUNDS_2D(environment.dim[0], environment.dim[1]);

    int attractor = cells(SELF, attribute::attractor);
    int weight = cells(SELF, attribute::weight);
    int target_maj = cells(SELF, attribute::move_maj);
    int target_min = cells(SELF, attribute::move_min);

    if(targets(target_maj, target_min) != 1){return;}
    if (attractor == 0 && weight == 0) { return; }

    //#pragma unroll
    for (int i = 0; i < Cell::num_attributes; i++) {
        future_cells(target_maj, target_min, i) = cells(SELF, i);
    }

}

__global__ void hatch(const int threshold, curandState* random_states, Device_Ptr<int> cells) {
    DIMS_2D(maj, min);
    BOUNDS_2D(cells.dim[0], cells.dim[1]);
    if (cells(SELF, attribute::value) < threshold){return;}

    int random[3];
    int id = LINEAR_CAST(maj, min, cells.dim[0]);
    curandState local_state = random_states[id];

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        random[i] = curand(&local_state);
    }

    FOR_NEIGHBOR(n_maj, n_min,
        if(cells(n_maj, n_min, attribute::attractor) == 0 && cells(n_maj, n_min, attribute::weight) == 0){
            cells(SELF, attribute::value) -= 10;

            //add attribute that controls how much hatching costs and at what value it occurs( I guess thats two attributes)
            cells(n_maj, n_min, attribute::color_r) = fmaxf( 70, fminf(cells(SELF, attribute::color_r), 255));
            cells(n_maj, n_min, attribute::color_g) = fmaxf(70, fminf(cells(SELF, attribute::color_g), 255));
            cells(n_maj, n_min, attribute::color_b) = fmaxf(70, fminf(cells(SELF, attribute::color_b), 255));
            cells(n_maj, n_min, attribute::attractor) = fmaxf(-100000, fminf(cells(SELF, attribute::attractor) + Random(random[0], -1, 1), 100000 ));
            cells(n_maj, n_min, attribute::weight) = fmaxf(-10000, fminf(cells(SELF, attribute::weight) + Random(random[1], -1, 1), 10000));
            //cells(n_maj, n_min, attribute::freq_a) = fmaxf(-100000, fminf(cells(SELF, attribute::freq_a) + Random(random[2], -1, 1), 100000) );
            cells(n_maj, n_min, attribute::value) = 10;
            return;
        }
    );

    //cells(SELF, attribute::value) /= 2;
}


namespace Substrate {

    namespace Species {

        namespace computer_fish {

            namespace Parameter {
                static bool running = false;
                const int environment_width = 768;
                const int environment_height = 768;
                const int environment_area = environment_width * environment_height; 
            }
            
            namespace Seed{
                static Matrix<int> cells(int value = 0) {

                    Matrix<int> result({Parameter::environment_width, Parameter::environment_height, Cell::num_attributes}, 0);

                    curandState* states = Random::Initialize::curand_xor(Parameter::environment_area, value);
    
                    Launch::kernel_2d(result.dim[0], result.dim[1]);
                    spawn<<<LAUNCH>>>(states, result); //try using the nvidia debugger
                    SYNC_KERNEL(spawn); 
    
                    cudaFree(states); //bad way of doing this, because it's not clear that one would have to call cudafree on curand_xor. should at least put it in on::Random::Delete
                    //TODO: create a struct to control curand more closely
    
                    return result;
    
                }
            };

            namespace Draw {
                static Matrix<uchar> frame(Matrix<int>& cells, Matrix<int>& environment) {

                    Matrix<uchar> output({cells.dim[0], cells.dim[1], 3}, 0);
    
                    Launch::kernel_2d(cells.dim[0], cells.dim[1]);
                    //draw_environment<<<LAUNCH>>>(environment, output);
                    //SYNC_KERNEL(draw_environment);
    
                    draw_cells <<<LAUNCH>>> (cells, output);
                    SYNC_KERNEL(draw_cells);
    
                    return output;
    
                }
            };

            namespace Step {
                static void polar(Matrix<int>& future_cells, Matrix<int>& environment, Matrix<int>& cells, Matrix<int>& targets, curandState* random) {
                    Launch::kernel_2d(environment.dim[0], environment.dim[1]);
                    
                    const int thresh = 20;
                    hatch << <LAUNCH >> > (thresh, random, cells);
                    SYNC_KERNEL(hatch);           
                    change_environment<<<LAUNCH>>> (environment, cells); 
                    SYNC_KERNEL(change_environment);
    
                    radiate_environment << <LAUNCH >> > (environment);
                    SYNC_KERNEL(radiate_environment);
    
                    const float damping_factor = 0.1;
                    dampen_environment<<<LAUNCH>>>(damping_factor, environment);
                    SYNC_KERNEL(dampen_environment);
    
                    set_targets<<<LAUNCH>>>(environment, cells, targets);
                    SYNC_KERNEL(set_targets);
    
                    const int threshold = 500;
                    conflict<<<LAUNCH>>>(cells, targets, future_cells, threshold);
                    SYNC_KERNEL(conflict);
    
                    move<<<LAUNCH>>> (environment, cells, targets, future_cells); 
                    SYNC_KERNEL(move);
    
                    cells = future_cells;
                }
            }
 

            static void run(Matrix<int> seed = Seed::cells(rand())) {

                curandState* random = Random::Initialize::curand_xor(Parameter::environment_area, rand());
                Parameter::running = true;

                Matrix<int> environment({Parameter::environment_width, Parameter::environment_height},0);
                Matrix<int> cells = seed; 
                Matrix<int> future_cells({Parameter::environment_width, Parameter::environment_height, 8}, 0);
                Matrix<int> targets({Parameter::environment_width, Parameter::environment_height}, 0);

                Matrix<uchar> frame({Parameter::environment_width, Parameter::environment_height, 3}, 0);

                //af::Window window(Parameter::environment_width, Parameter::environment_height);
                Window::open(Parameter::environment_width, Parameter::environment_height, "Substrate");

                int start_time = now_ms();
                int FPS = 60;
                do {
                    int current_time = now_ms();
                    int wait_time = (1000 / FPS) - (current_time - start_time);

                    Step::polar(future_cells, environment, cells, targets, random); 
                    //environment.fill_device_memory(0);
                    future_cells.fill_device_memory(0);
                    targets.fill_device_memory(0);
                    frame = Draw::frame(cells, environment); 

                    Window::render(frame);
                    //window.image(frame); 
                    
                    //std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
                    start_time = now_ms();
                    //std::cout << "FPS: " << 1000 / wait_time << std::endl;
                } while (Parameter::running);

                Window::close();
            }
        }
    }
}