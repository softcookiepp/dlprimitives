#version 450
#define X_ARITY 1
#define Y_ARITY 1
#include "pointwise-common/pointwise-reduce-template.glsl"

/* This block is meant to be read by the compile script. Please do not remove it!
 * @dtype TYPEOF_X0 all
 * @dtype TYPEOF_Y0 all
 */

void main()
{
	pointwise_reduce_naive_impl();
}

