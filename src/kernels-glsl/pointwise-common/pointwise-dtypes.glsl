// This is seriously the easiest way to do variable input dtypes that still allow dtype-based flow control.
// Not even joking.

#ifndef TYPEOF_X0
	#define TYPEOF_X0 DTYPE_F32
#endif
#ifndef TYPEOF_X1
	#define TYPEOF_X1 DTYPE_F32
#endif
#ifndef TYPEOF_X2
	#define TYPEOF_X2 DTYPE_F32
#endif
#ifndef TYPEOF_X3
	#define TYPEOF_X3 DTYPE_F32
#endif

#if TYPEOF_X0 == DTYPE_F16
	#define typeof_x0 float16_t
#elif TYPEOF_X0 == DTYPE_F32
	#define typeof_x0 float
#elif TYPEOF_X0 == DTYPE_F64
	#define typeof_x0 double
#elif TYPEOF_X0 == DTYPE_I8
	#define typeof_x0 int8_t
#elif TYPEOF_X0 == DTYPE_I16
	#define typeof_x0 int16_t
#elif TYPEOF_X0 == DTYPE_I32
	#define typeof_x0 int
#elif TYPEOF_X0 == DTYPE_I64
	#define typeof_x0 int64_t
#elif TYPEOF_X0 == DTYPE_U8
	#define typeof_x0 uint8_t
#elif TYPEOF_X0 == DTYPE_U16
	#define typeof_x0 uint16_t
#elif TYPEOF_X0 == DTYPE_U32
	#define typeof_x0 uint
#elif TYPEOF_X0 == DTYPE_U64
	#define typeof_x0 uint64_t	
#else
	#error "dtype not implemented for pointwise"
#endif

#if TYPEOF_X1 == DTYPE_F16
	#define typeof_x1 float16_t
#elif TYPEOF_X1 == DTYPE_F32
	#define typeof_x1 float
#elif TYPEOF_X1 == DTYPE_F64
	#define typeof_x1 double
#elif TYPEOF_X1 == DTYPE_I8
	#define typeof_x1 int8_t
#elif TYPEOF_X1 == DTYPE_I16
	#define typeof_x1 int16_t
#elif TYPEOF_X1 == DTYPE_I32
	#define typeof_x1 int
#elif TYPEOF_X1 == DTYPE_I64
	#define typeof_x1 int64_t
#elif TYPEOF_X1 == DTYPE_U8
	#define typeof_x1 uint8_t
#elif TYPEOF_X1 == DTYPE_U16
	#define typeof_x1 uint16_t
#elif TYPEOF_X1 == DTYPE_U32
	#define typeof_x1 uint
#elif TYPEOF_X1 == DTYPE_U64
	#define typeof_x1 uint64_t
#else
	#error "dtype not implemented for pointwise"
#endif

#if TYPEOF_X2 == DTYPE_F16
	#define typeof_x2 float16_t
#elif TYPEOF_X2 == DTYPE_F32
	#define typeof_x2 float
#elif TYPEOF_X2 == DTYPE_F64
	#define typeof_x2 double
#elif TYPEOF_X2 == DTYPE_I8
	#define typeof_x2 int8_t
#elif TYPEOF_X2 == DTYPE_I16
	#define typeof_x2 int16_t
#elif TYPEOF_X2 == DTYPE_I32
	#define typeof_x2 int
#elif TYPEOF_X2 == DTYPE_I64
	#define typeof_x2 int64_t
#elif TYPEOF_X2 == DTYPE_U8
	#define typeof_x2 uint8_t
#elif TYPEOF_X2 == DTYPE_U16
	#define typeof_x2 uint16_t
#elif TYPEOF_X2 == DTYPE_U32
	#define typeof_x2 uint
#elif TYPEOF_X2 == DTYPE_U64
	#define typeof_x2 uint64_t
#else
	#error "dtype not implemented for pointwise"
#endif

#if TYPEOF_X3 == DTYPE_F16
	#define typeof_x3 float16_t
#elif TYPEOF_X3 == DTYPE_F32
	#define typeof_x3 float
#elif TYPEOF_X3 == DTYPE_F64
	#define typeof_x3 double
#elif TYPEOF_X3 == DTYPE_I8
	#define typeof_x3 int8_t
#elif TYPEOF_X3 == DTYPE_I16
	#define typeof_x3 int16_t
#elif TYPEOF_X3 == DTYPE_I32
	#define typeof_x3 int
#elif TYPEOF_X3 == DTYPE_I64
	#define typeof_x3 int64_t
#elif TYPEOF_X3 == DTYPE_U8
	#define typeof_x3 uint8_t
#elif TYPEOF_X3 == DTYPE_U16
	#define typeof_x3 uint16_t
#elif TYPEOF_X3 == DTYPE_U32
	#define typeof_x3 uint
#elif TYPEOF_X3 == DTYPE_U64
	#define typeof_x3 uint64_t
#else
	#error "dtype not implemented for pointwise"
#endif


#ifndef TYPEOF_Y0
	#define TYPEOF_Y0 DTYPE_F32
#endif
#ifndef TYPEOF_Y1
	#define TYPEOF_Y1 DTYPE_F32
#endif

#if TYPEOF_Y0 == DTYPE_F16
	#define typeof_y0 float16_t
#elif TYPEOF_Y0 == DTYPE_F32
	#define typeof_y0 float
#elif TYPEOF_Y0 == DTYPE_F64
	#define typeof_y0 double
#elif TYPEOF_Y0 == DTYPE_I8
	#define typeof_y0 int8_t
#elif TYPEOF_Y0 == DTYPE_I16
	#define typeof_y0 int16_t
#elif TYPEOF_Y0 == DTYPE_I32
	#define typeof_y0 int
#elif TYPEOF_Y0 == DTYPE_I64
	#define typeof_y0 int64_t
#elif TYPEOF_Y0 == DTYPE_U8
	#define typeof_y0 uint8_t
#elif TYPEOF_Y0 == DTYPE_U16
	#define typeof_y0 uint16_t
#elif TYPEOF_Y0 == DTYPE_U32
	#define typeof_y0 uint
#elif TYPEOF_Y0 == DTYPE_U64
	#define typeof_y0 uint64_t
#else
	#error "dtype not implemented for pointwise"
#endif


#if TYPEOF_Y1 == DTYPE_F16
	#define typeof_y1 float16_t
#elif TYPEOF_Y1 == DTYPE_F32
	#define typeof_y1 float
#elif TYPEOF_Y1 == DTYPE_F64
	#define typeof_y1 double
#elif TYPEOF_Y1 == DTYPE_I8
	#define typeof_y1 int8_t
#elif TYPEOF_Y1 == DTYPE_I16
	#define typeof_y1 int16_t
#elif TYPEOF_Y1 == DTYPE_I32
	#define typeof_y1 int
#elif TYPEOF_Y1 == DTYPE_I64
	#define typeof_y1 int64_t
#elif TYPEOF_Y1 == DTYPE_U8
	#define typeof_y1 uint8_t
#elif TYPEOF_Y1 == DTYPE_U16
	#define typeof_y1 uint16_t
#elif TYPEOF_Y1 == DTYPE_U32
	#define typeof_y1 uint
#elif TYPEOF_Y1 == DTYPE_U64
	#define typeof_y1 uint64_t
#else
	#error "dtype not implemented for pointwise"
#endif
