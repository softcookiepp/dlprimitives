uint get_src_pos(uint pos, float scale, uint limit, float offset)
{
    uint src_pos = uint( (float(pos) + offset) * scale);
    return min(src_pos, limit-1);
}

uint get_tgt_pos(uint pos, float scale, uint limit, float offset)
{
    uint tgt_pos = uint(ceil(float(pos) * scale - offset));
    return min(tgt_pos,limit);
}

float calc_lin_pos(uint p,float scale, bool align_corners)
{
    if(align_corners)
        return float(p)*scale;
    return max(scale * (float(p)+0.5f) - 0.5f, 0.0f);
}
