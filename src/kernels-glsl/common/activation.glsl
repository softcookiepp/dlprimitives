
#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU     1
#define ACTIVATION_TANH     2
#define ACTIVATION_SIGMOID  3
#define ACTIVATION_RELU6    4

#if 0
	#define PREPARE_ACTIVATION(cid)

	#ifndef ACTIVATION
		#define ACTIVATION ACTIVATION_IDENTITY
	#endif

	#if ACTIVATION == ACTIVATION_IDENTITY
	#   define ACTIVATION_F(x) (x)
	#   define ACTIVATION_FINV(y,dy) (dy)
	#   define ACTIVATION_NAME identity
	#elif ACTIVATION == ACTIVATION_RELU
	#   define ACTIVATION_F(x) (max((x),dtype(0)))
	#   define ACTIVATION_FINV(y,dy)  ((y>0)?dy:0)
	#   define ACTIVATION_NAME relu
	#elif ACTIVATION == ACTIVATION_TANH
	#   define ACTIVATION_F(x) (tanh((x)))
	#   define ACTIVATION_FINV(y,dy) ((1-(y)*(y))*(dy))
	#   define ACTIVATION_NAME tanh 
	#elif ACTIVATION == ACTIVATION_SIGMOID
	#   define ACTIVATION_F(x) (dtype(1) / (dtype(1) + exp(-(x))))
	#   define ACTIVATION_FINV(y,dy) ((y)*(1-(y))*(dy))
	#   define ACTIVATION_NAME sigmoid
	#elif ACTIVATION == ACTIVATION_RELU6
	#   define ACTIVATION_F(x) (min(max((x),dtype(0)), dtype(6)))
	#   define ACTIVATION_FINV(y,dy)  ((0<y && y<6)?dy:0)
	#   define ACTIVATION_NAME relu6
	#else
	#   error "Unknown activation"
	#endif

#else
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
	
	
#endif
