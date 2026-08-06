#ifndef DIMS_MAX
	#define DIMS_MAX 8
#endif

struct Shape
{
    uint s[DIMS_MAX];
};


Shape get_pos_broadcast(Shape limits)
{
    Shape r;
	if (DIMS <= 1)
	{
		r.s[0] = get_global_id(0);
	}
	else if ( DIMS == 2)
	{
		r.s[0] = get_global_id(1);      
		r.s[1] = get_global_id(0);      
	}
	else if ( DIMS == 3    )
	{
		r.s[0] = get_global_id(2);      
		r.s[1] = get_global_id(1);      
		r.s[2] = get_global_id(0);      
	}
	else if ( DIMS == 4)    
	{
		r.s[0] = get_global_id(2);      
		r.s[1] = get_global_id(1);      
		r.s[2] = get_global_id(0) / limits.s[3];      
		r.s[3] = get_global_id(0) % limits.s[3];      
	}
	else if ( DIMS == 5)    
	{
		r.s[0] = get_global_id(2);      
		r.s[1] = get_global_id(1) / limits.s[2];      
		r.s[2] = get_global_id(1) % limits.s[2];      
		r.s[3] = get_global_id(0) / limits.s[4];      
		r.s[4] = get_global_id(0) % limits.s[4];      
	}
	else if ( DIMS == 6   ) 
	{
		r.s[0] = get_global_id(2) / limits.s[1];      
		r.s[1] = get_global_id(2) % limits.s[1];      
		r.s[2] = get_global_id(1) / limits.s[3];      
		r.s[3] = get_global_id(1) % limits.s[3];      
		r.s[4] = get_global_id(0) / limits.s[5];      
		r.s[5] = get_global_id(0) % limits.s[5];      
	}
	else if ( DIMS == 7)
	{
		r.s[0] = get_global_id(2) / limits.s[1];      
		r.s[1] = get_global_id(2) % limits.s[1];      
		r.s[2] = get_global_id(1) / limits.s[3];      
		r.s[3] = get_global_id(1) % limits.s[3];      
		r.s[4] = get_global_id(0) / (limits.s[5]*limits.s[6]);      
		uint s56 = get_global_id(0) % (limits.s[5]*limits.s[6]);
		r.s[5] = s56 / limits.s[6];      
		r.s[6] = s56 % limits.s[6];      
	}
	else if ( DIMS == 8)
	{
		r.s[0] = get_global_id(2) / limits.s[1];      
		r.s[1] = get_global_id(2) % limits.s[1];      

		r.s[2] = get_global_id(1) / (limits.s[3]*limits.s[4]);      
		uint s34 = get_global_id(1) % (limits.s[3]*limits.s[4]);
		r.s[3] = s34 / limits.s[4];      
		r.s[4] = s34 % limits.s[4];      

		r.s[5] = get_global_id(0) / (limits.s[6]*limits.s[7]);      
		uint s67 = get_global_id(0) % (limits.s[6]*limits.s[7]);
		r.s[6] = s67 / limits.s[7];      
		r.s[7] = s67 % limits.s[7];      
	}
    return r;
}

