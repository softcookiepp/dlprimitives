#version 450
#define SMALL_REDUCTION 1
#include "exec_template.glsl"
void main()
{
	exec_impl();
}
