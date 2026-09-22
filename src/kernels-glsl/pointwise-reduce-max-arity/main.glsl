#version 450
#include "../pointwise-common/pointwise-reduce-template2.glsl"

void main()
{
	pointwise_reduce_naive_impl();
}

