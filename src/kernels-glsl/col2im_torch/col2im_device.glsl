
#ifndef accT
	#define accT dtype
#endif

void col2im_device(
	uint index,
	#if USE_BDA
		const dtype* data_col,
	#endif
	uint data_col_offset,
	uint height,
	uint width,
	uint kernel_h,
	uint kernel_w,
	uint pad_height,
	uint pad_width,
	uint stride_height,
	uint stride_width,
	uint dilation_height,
	uint dilation_width,
	uint height_col,
	uint width_col,
	#if USE_BDA
		dtype* data_im,
	#endif
	uint data_im_offset)
{
	accT val = accT(0);
	const uint w_im = index % width + pad_width;
	const uint h_im = (index / width) % height + pad_height;
	const uint c_im = index / (width * height);
	uint kernel_extent_w = (kernel_w - 1) * dilation_width + 1;
	uint kernel_extent_h = (kernel_h - 1) * dilation_height + 1;
	// compute the start and end of the output
	const uint w_col_start = (w_im < kernel_extent_w)
			? 0
			: (w_im - kernel_extent_w) / stride_width + 1;
	const uint w_col_end = min(w_im / stride_width + 1, width_col);
	const uint h_col_start = (h_im < kernel_extent_h)
			? 0
			: (h_im - kernel_extent_h) / stride_height + 1;
	const uint h_col_end = min(h_im / stride_height + 1, height_col);

	// TODO: use LCM of stride and dilation to avoid unnecessary loops
	for (uint h_col = h_col_start; h_col < h_col_end; h_col += 1) {
		for (uint w_col = w_col_start; w_col < w_col_end; w_col += 1) {
			uint h_k = (h_im - h_col * stride_height);
			uint w_k = (w_im - w_col * stride_width);
			if (h_k % dilation_height == 0 && w_k % dilation_width == 0) {
				h_k /= dilation_height;
				w_k /= dilation_width;
				uint data_col_index =
						(((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col +
							h_col) *
								width_col +
						w_col;
				val += accT(data_col[data_col_index + data_col_offset]);
			}
		}
	}
	data_im[index + data_im_offset] = dtype(val);
}
