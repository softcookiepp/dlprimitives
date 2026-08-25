#define ROUTINE_IDENTITY 0
#define ROUTINE_FILL 1

typeof_y0 unary_unary_function(typeof_x0 x0)
{
	typeof_y0 y0;
	if (POINTWISE_ROUTINE == ROUTINE_IDENTITY)
	{
		y0 = typeof_y0(x0);
	}
	else if (POINTWISE_ROUTINE == ROUTINE_FILL)
	{
		y0 = typeof_y0(w[0]);
	}
	return y0;
}
