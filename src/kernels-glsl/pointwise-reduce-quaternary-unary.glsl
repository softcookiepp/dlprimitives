#version 450
#define X_ARITY 4
#define Y_ARITY 1
#include "pointwise-common/pointwise-reduce-template.glsl"

void main()
{
	pointwise_reduce_naive_impl();
}

