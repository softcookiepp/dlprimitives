#version 450
#include "../common/defs.glsl"
#include "../pointwise-common/pointwise-dtypes.glsl" // easiest way to have multiple dtypes
#define typeof_a typeof_x0
#define typeof_b typeof_x1
#define typeof_c typeof_y0

// adapted from https://github.com/Peterc3-dev/torch-vulkan
layout(local_size_x_id = 0, local_size_y_id = 0) in;
layout(constant_id = 0) const uint TILE_SIZE = 16;
layout(constant_id = 1) const bool USE_BETA = true; // if beta is 0, don't add c

layout(binding = 0, std430) readonly buffer BufA { typeof_a a[]; };
layout(binding = 1, std430) readonly buffer BufB { typeof_b b[]; };
layout(binding = 2, std430) buffer BufC { typeof_c c[]; };

layout(push_constant) uniform PushConstants
{
    uint M;
    uint K;
    uint N;
    float alpha;
    float beta;
} params;

shared typeof_a tileA[TILE_SIZE][TILE_SIZE];
shared typeof_b tileB[TILE_SIZE][TILE_SIZE];

void main()
{
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint localRow = gl_LocalInvocationID.y;
    uint localCol = gl_LocalInvocationID.x;

    dtype sum = 0.0;
    uint numTiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++)
    {
        // Load tile of A into shared memory
        uint aCol = t * TILE_SIZE + localCol;
        if (row < params.M && aCol < params.K)
        {
            tileA[localRow][localCol] = a[row * params.K + aCol];
        }
        else
        {
            tileA[localRow][localCol] = 0.0;
        }

        // Load tile of B into shared memory
        uint bRow = t * TILE_SIZE + localRow;
        if (bRow < params.K && col < params.N) {
            tileB[localRow][localCol] = b[bRow * params.N + col];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        // Sync to make sure tiles are loaded
        barrier();

        // Accumulate dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += dtype(tileA[localRow][k]) * dtype(tileB[k][localCol]);
        }

        // Sync before loading next tile
        barrier();
    }

    if (row < params.M && col < params.N)
    {
		if (USE_BETA)
			c[row * params.N + col] = typeof_c(params.beta)*c[row * params.N + col] + typeof_c(params.alpha)*typeof_c(sum);
		else
			c[row * params.N + col] = typeof_c(sum)*typeof_c(params.alpha);
    }
}
