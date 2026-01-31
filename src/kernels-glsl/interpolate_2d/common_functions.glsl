#if 0
int get_src_pos(int pos,float scale,int limit,float offset)
{
    int src_pos = (pos + offset) * scale;
    return min(src_pos,limit-1);
}
#endif
#if 0
int get_tgt_pos(int pos,float scale,int limit,float offset)
{
    int tgt_pos = ceil(pos * scale - offset);
    return min(tgt_pos,limit);
}
#endif
float calc_lin_pos(uint p,float scale, bool align_corners)
{
    if(align_corners)
        return float(p)*scale;
    return max(scale * (float(p)+0.5f) - 0.5f,0.0f);
}
