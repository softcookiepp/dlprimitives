#define ROUTINE_FILL 0

dtype unary_unary_function(dtype x0)
{
	dtype y0;
	if (POINTWISE_ROUTINE == ROUTINE_FILL)
	{
		y0 = dtype(w[0]);
	}
	return y0;
}
