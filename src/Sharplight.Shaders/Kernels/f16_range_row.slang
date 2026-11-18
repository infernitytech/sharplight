#pragma once
// One f32 row of K values (macro K) -> a row of a row-ranged f16 copy (common.slang, f16_range_scale), by one 256-thread
// workgroup: the row's largest magnitude first, then the conversion (8 values per thread and step). Rows of up to 8192 values
// stay in registers between the two steps (one read); longer ones are read twice. Returns the row's scale word (0 when the row
// is stored unscaled). Shared by to_f16_ranged (data.slang) and the ranged moe_gather_x16 (moe.slang).

groupshared float gRangeMax[8];

uint f16_range_row(float* xr, uint64_t yr, uint tid)
{
    #define RANGE_HELD (K <= 8192 ? (K + 2047) / 2048 : 1)
    float4 ha[RANGE_HELD], hb[RANGE_HELD];
    float m = 0;
    [ForceUnroll]
    for (uint i = 0; i < RANGE_HELD; i++)
    {
        uint c = tid * 8 + i * 2048;
        ha[i] = 0;
        hb[i] = 0;
        if (c < K)
        {
            ha[i] = ((float4*)(xr + c))[0];
            hb[i] = ((float4*)(xr + c))[1];
        }
        float4 a = abs(ha[i]), b = abs(hb[i]);
        m = max(m, max(max(max(a.x, a.y), max(a.z, a.w)), max(max(b.x, b.y), max(b.z, b.w))));
    }
    for (uint c = tid * 8 + RANGE_HELD * 2048; c < K; c += 2048)
    {
        float4 a = abs(((float4*)(xr + c))[0]), b = abs(((float4*)(xr + c))[1]);
        m = max(m, max(max(max(a.x, a.y), max(a.z, a.w)), max(max(b.x, b.y), max(b.z, b.w))));
    }
    m = WaveActiveMax(m);
    if (tid % 32 == 0) gRangeMax[tid / 32] = m;
    GroupMemoryBarrierWithGroupSync();
    m = max(max(max(gRangeMax[0], gRangeMax[1]), max(gRangeMax[2], gRangeMax[3])), max(max(gRangeMax[4], gRangeMax[5]), max(gRangeMax[6], gRangeMax[7])));
    float s = f16_range_scale(m);
    [ForceUnroll]
    for (uint i = 0; i < RANGE_HELD; i++)
    {
        uint c = tid * 8 + i * 2048;
        if (c < K)
        {
            float4 a = ha[i] * s, b = hb[i] * s;
            *(uint4*)(yr + uint64_t(c) * 2) = uint4(f32tof16(a.x) | (f32tof16(a.y) << 16), f32tof16(a.z) | (f32tof16(a.w) << 16),
                                                    f32tof16(b.x) | (f32tof16(b.y) << 16), f32tof16(b.z) | (f32tof16(b.w) << 16));
        }
    }
    for (uint c = tid * 8 + RANGE_HELD * 2048; c < K; c += 2048)
    {
        float4 a = ((float4*)(xr + c))[0] * s, b = ((float4*)(xr + c))[1] * s;
        *(uint4*)(yr + uint64_t(c) * 2) = uint4(f32tof16(a.x) | (f32tof16(a.y) << 16), f32tof16(a.z) | (f32tof16(a.w) << 16),
                                                f32tof16(b.x) | (f32tof16(b.y) << 16), f32tof16(b.z) | (f32tof16(b.w) << 16));
    }
    #undef RANGE_HELD
    return s == 1.0 ? 0u : asuint(1.0 / s);
}
