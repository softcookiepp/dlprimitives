#version 450

#define REQUIRES_REDUCE 1

#ifndef BACKWARD
#define BACKWARD 0
#endif

#include "compute_template.glsl"

void main()
{
	compute_impl();
}
