#version 450

// define this before including so that it doesn't do the forward one
#ifndef SOFTMAX_EPILOGUE_TYPE
	#define SOFTMAX_EPILOGUE_TYPE SOFTMAX_BACKWARD_EPILOGUE
#endif

#include "softmax_backward_template.glsl"

void main()
{
	do_softmax_backward();
}
