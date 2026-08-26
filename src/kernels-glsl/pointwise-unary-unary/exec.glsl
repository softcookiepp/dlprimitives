#version 450
#define X_ARITY 1
#define Y_ARITY 1
#include "../pointwise-strided-common/pointwise-strided-template.glsl"

void main()
{
	pointwise_strided_impl();
}
