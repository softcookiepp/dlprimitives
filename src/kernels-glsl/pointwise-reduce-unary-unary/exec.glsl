#version 450
#define X_ARITY 1
#define Y_ARITY 1
#include "../pointwise-strided-reduce/pointwise-reduce-template.glsl"

void main()
{
	pointwise_reduce_naive_impl();
}

