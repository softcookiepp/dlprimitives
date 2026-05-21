
#if CONVGEMM == 3
    void add_img_value(__global float *ptr,int matrix_row,int matrix_col,float dV)
    #else
    float get_img_value(__global float const *ptr,int matrix_row,int matrix_col)
    #endif
    {
#if KERN_W == 1 && KERN_H == 1 && STRIDE_H == 1 && STRIDE_W == 1 && DILATE_H == 1 && DILATE_W == 1 \
        && PAD_H == 0 && PAD_W == 0 && GROUPS == 1 && IMG_COLS == SRC_COLS && IMG_ROWS == SRC_ROWS 
        int channel = matrix_col;
        int b  = matrix_row / (IMG_COLS * IMG_ROWS);
        int rc = matrix_row % (IMG_COLS * IMG_ROWS);

        int address = (b * CHANNELS_IN + channel) * (SRC_ROWS * SRC_COLS) + rc;
        #if CONVGEMM != 3
            return ptr[address];
        #else
            #if REDUCE_K > 1
                atomic_addf(ptr+address,dV);
            #else
                ptr[address] += dV;
            #endif
        #endif
#else
        int channel = matrix_col / (KERN_H * KERN_W);
        int k_index = matrix_col % (KERN_H * KERN_W);
        
        int dy = k_index / KERN_W;
        int dx = k_index % KERN_W;
        
        int b  = matrix_row / (IMG_COLS * IMG_ROWS);
        int rc = matrix_row % (IMG_COLS * IMG_ROWS);

        int r  = rc / IMG_COLS;
        int c  = rc % IMG_COLS;

        int y_pos = -PAD_H + r * STRIDE_H;
        int x_pos = -PAD_W + c * STRIDE_W;

        int y = y_pos + dy * DILATE_H;
        int x = x_pos + dx * DILATE_W;

        int address = ((b *  (CHANNELS_IN*GROUPS) + channel) * SRC_ROWS + y) * SRC_COLS + x;

        #if CONVGEMM != 3
            #if PAD_W > 0 || PAD_H > 0
                if(x >= 0 && y >= 0 && x < SRC_COLS && y < SRC_ROWS) {
                    return ptr[address];
                }
                return 0;
            #else
                return ptr[address];
            #endif  
        #else
            #if PAD_W > 0 || PAD_H > 0
                if(x >= 0 && y >= 0 && x < SRC_COLS && y < SRC_ROWS) 
                    atomic_addf(ptr+address,dV);
            #else
                atomic_addf(ptr+address,dV);
            #endif
        #endif
#endif  
    }
