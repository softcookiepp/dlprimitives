#ifndef DIMS_MAX
	#define DIMS_MAX 8
#endif

struct Shape
{
    uint s[DIMS_MAX];
};

Shape getPosFromTriIndex(Shape shape, uint dims)
{
	Shape pos;
	if (dims == 1)
	{
		uint i0 = get_global_id(0);
		pos.s[0] = i0;
	}
	else if (dims == 2)
	{
		uint i1 = get_global_id(0);
		uint i0 = get_global_id(1);
		pos.s[0] = i0;
		pos.s[1] = i1;
	}
	else if (dims == 3)
	{
		uint i2 = get_global_id(0);
		uint i1 = get_global_id(1);
		uint i0 = get_global_id(2);
		pos.s[0] = i0;
		pos.s[1] = i1;
		pos.s[2] = i2;
	}
	else if (dims == 4)
	{
		uint ic = get_global_id(0);
		uint i1 = get_global_id(1);
		uint i0 = get_global_id(2);
		uint i2 = ic / shape.s[3];
		uint i3 = ic % shape.s[3];
		pos.s[0] = i0;
		pos.s[1] = i1;
		pos.s[2] = i2;
		pos.s[3] = i3;
	}
	else if (dims == 5)
	{
		uint i34 = get_global_id(0);
		uint i12 = get_global_id(1);
		uint i0 = get_global_id(2);
		uint i1  = i12 / shape.s[2];
		uint i2  = i12 % shape.s[2];
		uint i3  = i34 / shape.s[4];
		uint i4  = i34 % shape.s[4];
		pos.s[0] = i0;
		pos.s[1] = i1;
		pos.s[2] = i2;
		pos.s[3] = i3;
		pos.s[4] = i4;
	}
	else if (dims == 6)
	{
		uint i45 = get_global_id(0);
		uint i23 = get_global_id(1);
		uint i01 = get_global_id(2);
		uint i0  = i01 / shape.s[1];
		uint i1  = i01 % shape.s[1];
		uint i2  = i23 / shape.s[3];
		uint i3  = i23 % shape.s[3];
		uint i4  = i45 / shape.s[5];
		uint i5  = i45 % shape.s[5];
		pos.s[0] = i0;
		pos.s[1] = i1;
		pos.s[2] = i2;
		pos.s[3] = i3;
		pos.s[4] = i4;
		pos.s[5] = i5;
	}
	else if (dims == 7)
	{
		uint i456 = get_global_id(0);
		uint i23  = get_global_id(1);
		uint i01  = get_global_id(2);
		uint i0  = i01 / shape.s[1];
		uint i1  = i01 % shape.s[1];
		uint i2  = i23 / shape.s[3];
		uint i3  = i23 % shape.s[3];
		uint i4  = i456 / (shape.s[5]*shape.s[6]);
		uint i56 = i456 % (shape.s[5]*shape.s[6]);
		uint i5  = i56 / shape.s[6];
		uint i6  = i56 % shape.s[6];
		pos.s[0] = i0;
		pos.s[1] = i1;
		pos.s[2] = i2;
		pos.s[3] = i3;
		pos.s[4] = i4;
		pos.s[5] = i5;
		pos.s[6] = i6;
	}
	else if (dims == 8)
	{
		uint i567 = get_global_id(0);
		uint i234 = get_global_id(1);
		uint i01  = get_global_id(2);
		uint i0  = i01 / shape.s[1];
		uint i1  = i01 % shape.s[1];

		uint i2  = i234 / (shape.s[3]*shape.s[4]);
		uint i34 = i234 % (shape.s[3]*shape.s[4]);
		uint i3  = i34 / shape.s[4];
		uint i4  = i34 % shape.s[4];

		uint i5  = i567 / (shape.s[6]*shape.s[7]);
		uint i67 = i567 % (shape.s[6]*shape.s[7]);
		uint i6  = i67 / shape.s[7];
		uint i7  = i67 % shape.s[7];
		
		pos.s[0] = i0;
		pos.s[1] = i1;
		pos.s[2] = i2;
		pos.s[3] = i3;
		pos.s[4] = i4;
		pos.s[5] = i5;
		pos.s[6] = i6;
		pos.s[7] = i7;
	}
	
	return pos;
}

bool posValid(Shape shape, Shape pos, uint dims)
{
	bool valid = true;
	for (uint i = 0; i < dims; i += 1)
	{
		if (pos.s[i] >= shape.s[i]) valid = false;
	}
	return valid;
}

Shape getPos(uint gid, Shape shape, uint dims)
{
	Shape pos;
	uint coef = 1;
	// use int to avoid overflow
	for (int i = int(dims) - 1; i >= 0; i -= 1)
	{
		uint dLen = shape.s[i];
		uint mod = (gid/coef) % dLen;
		pos.s[i] = mod;
		coef *= dLen;
	}
	return pos;
}

uint getStridedIndexFromPos(Shape pos, Shape stride, uint dims)
{
	uint idx = 0;
	for (uint i = 0; i < dims; i += 1)
	{
		idx += pos.s[i]*stride.s[i];
	}
	return idx;
}

uint getStridedIndex(uint gid, Shape shape, Shape stride, uint dims)
{
	Shape pos = getPos(gid, shape, dims);
	return getStridedIndexFromPos(pos, stride, dims);
}
