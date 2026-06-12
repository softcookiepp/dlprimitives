
#ifndef dtype
	#define dtype float
#endif

#if ATOMIC_FLOAT32
#	ifndef atomic_dtype
		#define atomic_dtype dtype
	#endif
	#define atomic_addf(idx, ptr, v) atomicAdd(ptr[idx], v)
#else


	#ifndef atomic_dtype
		#define atomic_dtype uint
	#endif

	// Ok, this is going to be tricky. We will need to make the default type of the float in question into an integer.
	// I will likely just come back to this later hehe
	#define atomic_addf(idx, ptr, v_init) \
	{ \
		dtype v = v_init; \
		dtype oldv = atomic_to_dtype(ptr[idx]); \
		while(true) \
		{ \
			dtype newv = oldv + v; \
			atomic_dtype prev = atomicCompSwap(ptr[idx], dtype_to_atomic(oldv), dtype_to_atomic(newv)); \
			if(prev == dtype_to_atomic(oldv)) \
				break; \
			oldv = atomic_to_dtype(prev); \
		} \
	}
#endif
