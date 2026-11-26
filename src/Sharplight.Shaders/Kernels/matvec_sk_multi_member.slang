// One member of matvec_sk_multi (matmul.slang), included once per member inside its workgroup-uniform branch with
// MVM_SGN (subgroups per team: the member's own MV_SK_SG), MVM_RN (rows per team: its MV_SK_ROWS), MVM_CHAIN (1: the explicit fma
// chain of dot8, its MV_FMA_CHAIN), MVM_NM (its output rows), MVM_W (its weight rows), MVM_Y (its output) and MVM_WGI (the
// workgroup's index within the member) defined. A workgroup of 8 subgroups runs 8 / MVM_SGN teams; each team computes MVM_RN rows
// exactly as one workgroup of the member's own matvec_sk launch: thread t of the team adds the groups t, t + 32 * MVM_SGN, ... in
// order, then the subgroup sums and the team's partials are added in the same order, so every output equals that launch's.
{
    const uint teams = 8 / MVM_SGN;
    const uint team = sg / MVM_SGN, ts = sg % MVM_SGN, tt = tid % (SG * MVM_SGN);
    const uint row0 = (MVM_WGI * teams + team) * MVM_RN;
    float acc[MVM_RN];
    [ForceUnroll] for (int r = 0; r < MVM_RN; r++) acc[r] = 0;
    [unroll(MV_UNROLL)]
    for (uint g = tt; g < K / 8; g += SG * MVM_SGN)
    {
        float w[MVM_RN][8];
        [ForceUnroll]
        for (int r = 0; r < MVM_RN; r++)
        {
            // Rows past the member's last (a partial team or workgroup) read its last row; their sums are not stored.
            load8(MVM_W, ROW_BYTES, min(row0 + r, uint(MVM_NM) - 1), g, w[r]);
        }
        float4* xp = (float4*)(p.x + g * 8);
        float4 a = xp[0], b = xp[1];
        [ForceUnroll]
        for (int r = 0; r < MVM_RN; r++)
        {
#if MVM_CHAIN
            acc[r] += dot8_chain(w[r], a, b);
#else
            acc[r] += dot8_plain(w[r], a, b);
#endif
        }
    }
    [ForceUnroll]
    for (int r = 0; r < MVM_RN; r++)
    {
        float s = WaveActiveSum(acc[r]);
        if (lane == 0) skPartM[(team * MVM_RN + r) * MVM_SGN + ts] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < teams * MVM_RN)
    {
        uint t2 = tid / MVM_RN, r = tid % MVM_RN, row = (MVM_WGI * teams + t2) * MVM_RN + r;
        if (row < MVM_NM)
        {
            float s = 0;
            [ForceUnroll] for (int i = 0; i < MVM_SGN; i++) s += skPartM[(t2 * MVM_RN + r) * MVM_SGN + i];
            MVM_Y[row] = s;
        }
    }
}
#undef MVM_SGN
#undef MVM_RN
#undef MVM_CHAIN
#undef MVM_NM
#undef MVM_W
#undef MVM_Y
#undef MVM_WGI
