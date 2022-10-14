#pragma once

//local macros
#include"macros.h"

//stdlib
#include<iostream>
#include<string>
#include<vector>

#include<fstream>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

//namespace fs = std::filesystem;

//opengl



#include<GL/glew.h>

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include<GL/gl.h>
#include<GLFW/glfw3.h>


//cuda
#include"cuda_includes.h"

//curand
#include<curand.h>
#include<curand_kernel.h>

//skeleton key
// #include<skeleton_key.h>


#include"../Display/Window.h"
#include"../Random/Random.h"

typedef unsigned int uint;
typedef unsigned char uchar;



