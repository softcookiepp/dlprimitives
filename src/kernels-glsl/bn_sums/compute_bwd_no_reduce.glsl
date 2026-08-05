#version 450

#define REQUIRES_REDUCE 0

#ifndef BACKWARD
#define BACKWARD 1
#endif

#include "compute_template.glsl"

void main()
{
	compute_impl();
}
