#version 450
#ifndef SOFTMAX_EPILOGUE_TYPE
	#define SOFTMAX_EPILOGUE_TYPE SOFTMAX_FORWARD_EPILOGUE
#endif
#include "softmax_forward_template.glsl"

void main()
{
	do_softmax_forward();
}
