#version 450

#ifndef BACKWARD
#define BACKWARD 0
#endif

#include "reduce_template.glsl"

void main()
{
	reduce_impl();
}
