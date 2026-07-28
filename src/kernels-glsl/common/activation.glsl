#ifndef ACTIVATION_IDENTITY
	#define ACTIVATION_IDENTITY 0
	#define ACTIVATION_RELU     1
	#define ACTIVATION_TANH     2
	#define ACTIVATION_SIGMOID  3
	#define ACTIVATION_RELU6    4
#endif

// eventually all activations will be based on specialization constants in order to reduce source compilation overhead
#define PREPARE_ACTIVATION(cid) layout(constant_id = cid) const uint ACTIVATION = ACTIVATION_IDENTITY; \
dtype ACTIVATION_F(dtype x) \
{ \
	if (ACTIVATION == ACTIVATION_IDENTITY) return x; \
	else if (ACTIVATION == ACTIVATION_RELU) return max((x),dtype(0)); \
	else if (ACTIVATION == ACTIVATION_TANH) return tanh(x); \
	else if (ACTIVATION == ACTIVATION_SIGMOID) return (dtype(1) / (dtype(1) + exp(-(x)))); \
	else if (ACTIVATION == ACTIVATION_RELU6) return (min(max((x),dtype(0)), dtype(6))); \
	return dtype(0.0); \
} \
dtype ACTIVATION_FINV(dtype y, dtype dy) \
{ \
	if (ACTIVATION == ACTIVATION_IDENTITY) return dy; \
	else if (ACTIVATION == ACTIVATION_RELU) return ((y>0)?dy:0); \
	else if (ACTIVATION == ACTIVATION_TANH) return ((1-(y)*(y))*(dy)); \
	else if (ACTIVATION == ACTIVATION_SIGMOID) return ((y)*(1-(y))*(dy)); \
	else if (ACTIVATION == ACTIVATION_RELU6) return ((0<y && y<6)?dy:0); \
	return dtype(0.0); \
} \

