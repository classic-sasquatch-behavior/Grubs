#pragma once

#include<chrono>


inline int now_ms() { return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count(); }




#define DIMS_2D(_first_, _second_)\
const int _first_ = ( blockIdx.x * blockDim.x ) + threadIdx.x;\
const int _second_ = ( blockIdx.y * blockDim.y ) + threadIdx.y;\
const int& _FIRST = _first_;\
const int& _SECOND = _second_;


#define BOUNDS_2D(_first_, _second_)\
if((_FIRST >= _first_)||(_SECOND >= _second_)){return;}

#define SELF\
_FIRST, _SECOND