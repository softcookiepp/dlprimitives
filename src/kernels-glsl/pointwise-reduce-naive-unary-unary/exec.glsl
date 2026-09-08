#version 450
#define X_ARITY 1
#define Y_ARITY 1
#include "../pointwise-reduce-naive/pointwise-reduce-naive-template.glsl"

void main()
{
	pointwise_reduce_naive_impl();
}

