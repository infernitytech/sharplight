// The recurrence of one LSTM direction (Lstm, see StandardOps.Lstm): one workgroup walks the time steps; every thread owns hidden units
// j = tid, tid + 256, ... and computes their four gate pre-activations from the shared hidden state of the previous step.
// gates: [T, 4H] f32 (input projection and bias, gate order input, forget, cell, output); w: [4H, H] f32; y: [T, H] f32.
// Macros: H, REVERSE, PER (units per thread, ceil(H / 256)).
#include "common.slang"

#ifdef KERNEL_LSTM
struct LstmParams
{
    float* gates;
    float* w;
    float* y;
    uint T;
};

groupshared float hs[H];
groupshared float cs[H];

float sigmoid_(float v)
{
    return 1.0 / (1.0 + exp(-v));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lstm(uint3 gtid : SV_GroupThreadID, uniform LstmParams p)
{
    uint tid = gtid.x;
    for (uint j = tid; j < H; j += 256)
    {
        hs[j] = 0.0;
        cs[j] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 0; s < p.T; s++)
    {
        uint t = REVERSE ? p.T - 1 - s : s;
        float next[PER];
        for (uint q = 0; q < PER; q++)
        {
            uint j = tid + q * 256;
            next[q] = 0.0;
            if (j < H)
            {
                float z0 = p.gates[uint64_t(t) * 4 * H + j];
                float z1 = p.gates[uint64_t(t) * 4 * H + H + j];
                float z2 = p.gates[uint64_t(t) * 4 * H + 2 * H + j];
                float z3 = p.gates[uint64_t(t) * 4 * H + 3 * H + j];
                float* w0 = p.w + uint64_t(j) * H;
                float* w1 = p.w + uint64_t(H + j) * H;
                float* w2 = p.w + uint64_t(2 * H + j) * H;
                float* w3 = p.w + uint64_t(3 * H + j) * H;
                for (uint k = 0; k < H; k++)
                {
                    float h = hs[k];
                    z0 += w0[k] * h;
                    z1 += w1[k] * h;
                    z2 += w2[k] * h;
                    z3 += w3[k] * h;
                }
                float c = sigmoid_(z1) * cs[j] + sigmoid_(z0) * tanh(z2);
                cs[j] = c;
                next[q] = sigmoid_(z3) * tanh(c);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint q = 0; q < PER; q++)
        {
            uint j = tid + q * 256;
            if (j < H)
            {
                hs[j] = next[q];
                p.y[uint64_t(t) * H + j] = next[q];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
#endif
