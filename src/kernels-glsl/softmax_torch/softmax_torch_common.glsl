dtype add_fn(dtype a, dtype b) { return a + b; }

#define spatialBlockReduceX(out_val_, r, shared_, shared_offset_, val_) \
{ \
	uint shared_offset = (shared_offset_) + get_local_id(1) * get_local_size(0); \
	barrier(); \
	shared_[get_local_id(0) + shared_offset] = val_; \
	uint offset = get_local_size(0) / 2; \
	while (offset > 0) \
	{ \
		barrier(); \
		if (get_local_id(0) < offset) \
			shared_[get_local_id(0) + shared_offset] = r(shared_[get_local_id(0) + shared_offset], shared_[get_local_id(0) + offset + shared_offset]); \
		offset /= 2; \
	} \
	barrier(); \
	out_val_ = shared_[shared_offset]; \
}\

#define SOFTMAX_FORWARD_EPILOGUE 1
#define SOFTMAX_BACKWARD_EPILOGUE 2
// TODO: log softmax

#ifndef SOFTMAX_EPILOGUE_TYPE
	#define SOFTMAX_EPILOGUE_TYPE SOFTMAX_FORWARD_EPILOGUE
#endif

#if SOFTMAX_EPILOGUE_TYPE == SOFTMAX_FORWARD_EPILOGUE
	struct Epilogue
	{
		dtype max_input;
		dtype sum;
	};
	
	#define do_epilogue(epilogue_, outp_, inp_) outp_ = exp((inp_) - (epilogue_).max_input)/(epilogue_).sum
#elif SOFTMAX_EPILOGUE_TYPE == SOFTMAX_BACKWARD_EPILOGUE

struct Epilogue
{
	dtype sum;
};

#define do_epilogue(epilogue_, gradOutput, output_) dtype((gradOutput) - exp(dtype(output_)) * (epilogue_).sum)

#endif
