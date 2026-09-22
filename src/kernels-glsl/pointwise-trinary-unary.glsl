#version 450
#define X_ARITY 3
#define Y_ARITY 1
#include "pointwise-strided-common/pointwise-strided-template.glsl"

/* This block is meant to be read by the compile script. Please do not remove it!
 * @dtype TYPEOF_X0 all
 * @dtype TYPEOF_X1 all
 * @dtype TYPEOF_X2 all
 * @dtype TYPEOF_Y0 all
 */

void main()
{
	pointwise_strided_impl();
}
