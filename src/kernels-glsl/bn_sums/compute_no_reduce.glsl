#version 450

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#define REQUIRES_REDUCE 0

#ifndef BACKWARD
#define BACKWARD 0
#endif

#include "compute_template.glsl"

void main()
{
	compute_impl();
}
