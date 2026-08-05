#version 450

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#ifndef REQUIRES_REDUCE
	#if SECOND_REDUCE_SIZE > 1
		#define REQUIRES_REDUCE 1
	#else
		#define REQUIRES_REDUCE 0
	#endif
#endif

#ifndef BACKWARD
#define BACKWARD 0
#endif

#include "compute_template.glsl"

void main()
{
	compute_impl();
}
