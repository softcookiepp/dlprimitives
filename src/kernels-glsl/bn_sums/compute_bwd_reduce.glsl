#version 450

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#define REQUIRES_REDUCE 1

#ifndef BACKWARD
#define BACKWARD 1
#endif

#include "compute_template.glsl"

void main()
{
	compute_impl();
}
