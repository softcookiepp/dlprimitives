#version 450
#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#ifndef BACKWARD
#define BACKWARD 0
#endif

#include "reduce_template.glsl"

void main()
{
	reduce_impl();
}
