#version 450
#define SOFTMAX_EPILOGUE_TYPE SOFTMAX_FORWARD_EPILOGUE
#include "softmax_forward_template.glsl"

void main()
{
	do_softmax_forward();
}
