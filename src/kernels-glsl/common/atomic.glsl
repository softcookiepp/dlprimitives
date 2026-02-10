
#if ATOMIC_FLOAT32
#define atomic_addf(idx, ptr, v) atomicAdd(ptr[idx], v)
#else
// Ok, this is going to be tricky. We will need to make the default type of the float in question into an integer.
// I will likely just come back to this later hehe
#define atomic_addf(idx, ptr, v_init) \
{ \
	float v = v_init; \
	float oldv = atomic_to_dtype(ptr[idx]); \
    while(true) \
    { \
        float newv = oldv + v; \
		uint prev = atomicCompSwap(ptr[idx], dtype_to_atomic(oldv), dtype_to_atomic(newv)); \
        if(prev == dtype_to_atomic(oldv)) \
            break; \
        oldv = atomic_to_dtype(prev); \
    } \
}

#endif
