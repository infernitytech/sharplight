// Chunked (WY / UT-transform) form of the gated delta rule for prefill, on cooperative matrices (16x16x16 f16 x f16 + f32).
// Same recurrence as delta_rule (recurrent.slang); per head, with state S [DV, DK] and q, k L2-normalized:
//   S_t = exp(g_t) S_{t-1} + beta_t (v_t - exp(g_t) S_{t-1} k_t) k_t^T,   o_t = SCALE * S_t q_t.
// Tokens are cut into chunks of 64 rows that never straddle a sequence run. Within a chunk, with G_t = sum_{s <= t} g_s
// (inclusive, chunk-local) and S0 the state at the chunk start:
//   L  = strictly lower (beta_t exp(G_t - G_s) k_t . k_s),   X = (I + L)^-1,   Tb = X diag(beta)
//   P  = lower incl. diagonal (SCALE exp(G_t - G_s) q_t . k_s)
//   R  = V - diag(exp(G)) K S0^T,   U = Tb R
//   O  = SCALE diag(exp(G)) Q S0^T + P U
//   S^T <- exp(G_last) S0^T + K^T diag(exp(G_last - G)) U
// All exponents are <= 0. Three launches:
//   delta_chunk_qk     normalized q, k rows as f16 planes (zeros for the 64 rows past T, which chunk tiles may read)
//   delta_chunk_prep   per (value head, chunk), all chunks in parallel: G, Tb and P as f16 planes (the triangular inverse in
//                      f32 by forward substitution, one column per thread in registers, no workgroup barriers)
//   delta_chunk_state  per (value head, BV state columns, run), sequential over the run's chunks: the products with the state,
//                      which stays in f32 accumulators for the whole run.
// PARTS = 2: every matrix operand is split into f16 hi + lo (about 22 mantissa bits; hi*hi + hi*lo + lo*hi), close to f32.
//            lo is stored scaled by LO_SCALE: unscaled it would mostly be an f16 subnormal (|x| < 0.1 already gives
//            |lo| < 2^-14), which devices may flush to zero; the cross products accumulate separately and are scaled back.
// PARTS = 1: one f16 product (reduced precision).
// Scratch (indexed by global token row, so chunks of different runs never collide; SROWS = row bound + 64):
//   qk  half [PARTS][2 (q, k)][SROWS][HK][DK]
//   tp  half [PARTS][2 (Tb, P)][SROWS][HV][64]
//   gc  float [SROWS][HV]                  (chunk-local cumulative log decay G)
// Macros: HK HV DK DV QS0 QS1 KS0 KS1 VS0 VS1 SCALE NORM_Q EPS_Q NORM_K EPS_K MULTI_SEQ PARTS BV SROWS CHUNK_MIN
// STATE_THREADS (128 or 256: subgroups of delta_chunk_state), STATE_FMT (state16.slang: a 16-bit state is widened through the
// f32 staging tile at the run start and rounded once at its end; within a run it stays f32, unlike the token-by-token kernel,
// which rounds it after every row).
// Requires 32-wide subgroups, DK % 16 == 0, DV % BV == 0, BV % 16 == 0.
#include "common.slang"
#include "state16.slang"
using namespace linalg;

#define CH 64
#define NT 128
#ifndef PARTS
#define PARTS 1
#endif
#ifndef NORM_Q
#define NORM_Q 0
#endif
#ifndef NORM_K
#define NORM_K 0
#endif
#ifndef CHUNK_MIN
#define CHUNK_MIN 0
#endif

typealias MA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
typealias MCH = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

// Element offsets of the scratch planes.
uint64_t qk_off(uint part, uint which, uint row, uint hk) { return ((uint64_t(part * 2 + which) * SROWS + row) * HK + hk) * DK; }
uint64_t tp_off(uint part, uint which, uint row, uint h) { return ((uint64_t(part * 2 + which) * SROWS + row) * HV + h) * CH; }

#define LO_SCALE 2048.0

// The f16 parts of x: hi = f16(x), lo = f16((x - hi) * LO_SCALE).
// hi is x rounded to f16 precision with integer operations: drivers may fold an f32 -> f16 -> f32 round trip into x,
// which would leave lo = 0. Below the f16 normal range everything goes into the scaled lo part.
void split2(float x, out half hi, out half lo)
{
#if PARTS == 1
    hi = half(x);
    lo = half(0.0);
    return;
#endif
    uint u = asuint(x);
    float h = asfloat((u + 0xFFFu + ((u >> 13) & 1u)) & 0xFFFFE000u);
    h = abs(h) < 6.103515625e-5 ? 0.0 : h;
    hi = half(h);
    lo = half((x - h) * LO_SCALE);
}

struct ChunkParams
{
    float* q;
    float* k;
    float* v;
    float* g;
    float* beta;
    STATE_T* state;
    float* o;
    half* qk;
    half* tp;
    float* gc;
#if MULTI_SEQ
    int* runs;   // [U, 4] (slot, first row, row count, first position); grid z = run; state [S, HV, DV, DK]
#endif
    uint T;
};

// (slot, first row, row count, fresh) of this workgroup's run.
int4 chunk_run(ChunkParams p, uint z)
{
#if MULTI_SEQ
    return int4(p.runs[z * 4], p.runs[z * 4 + 1], p.runs[z * 4 + 2], p.runs[z * 4 + 3] == 0 ? 1 : 0);
#else
    return int4(0, 0, int(p.T), 0);
#endif
}

#ifdef KERNEL_DELTA_CHUNK_QK
// ---------------------------------------------------------------------------------------------
// One subgroup per (row, key head): q and k rows L2-normalized when NORM_Q / NORM_K, written as f16 planes.
// Rows [T, T + 64) are written as zeros: chunk tiles read up to 63 rows past a run's end.
#define QK_E ((DK + 31) / 32)
[shader("compute")]
[numthreads(NT, 1, 1)]
void delta_chunk_qk(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ChunkParams p)
{
    uint lane = ltid.x % 32;
    uint row = gid.x * (NT / 32) + ltid.x / 32;
    uint hk = gid.y;
    if (row >= p.T + CH || row >= SROWS) return;
    bool live = row < p.T;
    float qv[QK_E];
    float kv[QK_E];
    float qs = 0, ks = 0;
    [ForceUnroll]
    for (uint e = 0; e < QK_E; e++)
    {
        uint i = lane + e * 32;
        qv[e] = live && i < DK ? p.q[uint64_t(row) * QS0 + uint64_t(hk) * QS1 + i] : 0.0;
        kv[e] = live && i < DK ? p.k[uint64_t(row) * KS0 + uint64_t(hk) * KS1 + i] : 0.0;
        qs += qv[e] * qv[e];
        ks += kv[e] * kv[e];
    }
#if NORM_Q
    float qn = live ? rsqrt(WaveActiveSum(qs) + EPS_Q) : 0.0;
#else
    float qn = 1.0;
#endif
#if NORM_K
    float kn = live ? rsqrt(WaveActiveSum(ks) + EPS_K) : 0.0;
#else
    float kn = 1.0;
#endif
    [ForceUnroll]
    for (uint e = 0; e < QK_E; e++)
    {
        uint i = lane + e * 32;
        if (i >= DK) continue;
        half qh, ql, kh, kl;
        split2(qv[e] * qn, qh, ql);
        split2(kv[e] * kn, kh, kl);
        p.qk[qk_off(0, 0, row, hk) + i] = qh;
        p.qk[qk_off(0, 1, row, hk) + i] = kh;
#if PARTS > 1
        p.qk[qk_off(1, 0, row, hk) + i] = ql;
        p.qk[qk_off(1, 1, row, hk) + i] = kl;
#endif
    }
}
#endif

// acc += A * B over the parts: hi*hi into acc, the (scaled) cross products hi*lo + lo*hi into cor;
// MMA_DONE folds cor back into acc.
#if PARTS > 1
#define MMA_PARTS(acc, cor, aHi, aLo, bHi, bLo) { acc = coopMatMulAdd<float, false>(aHi, bHi, acc); cor = coopMatMulAdd<float, false>(aLo, bHi, coopMatMulAdd<float, false>(aHi, bLo, cor)); }
#define MMA_DONE(acc, cor) acc = acc + cor * MC(1.0 / LO_SCALE)
#else
#define MMA_PARTS(acc, cor, aHi, aLo, bHi, bLo) acc = coopMatMulAdd<float, false>(aHi, bHi, acc)
#define MMA_DONE(acc, cor)
#endif

// The chunk kernels use 8 subgroups. Global operand fragments of a product are loaded for its whole K loop before the
// products: the loads then overlap instead of each paying the memory latency in turn.
#define NTC 256
#define NSG (NTC / 32)

// Fragment arrays: hi part and (two parts) lo part.
#define DECL_FRAGS(T, name, n) T name##H[n]; T name##L[n];
#if PARTS > 1
#define LOAD_FRAG(T, name, i, layout, ptr, offHi, offLo, stride) { name##H[i] = T.Load<layout>(ptr, uint(offHi), uint(stride)); name##L[i] = T.Load<layout>(ptr, uint(offLo), uint(stride)); }
#else
#define LOAD_FRAG(T, name, i, layout, ptr, offHi, offLo, stride) { name##H[i] = T.Load<layout>(ptr, uint(offHi), uint(stride)); }
#endif

#ifdef KERNEL_DELTA_CHUNK_PREP
// ---------------------------------------------------------------------------------------------
// Workgroup (value head h, chunk c of run z): G, Tb = (I + L)^-1 diag(beta) and P of the chunk. Subgroup sg computes the
// 16x16 tiles (sg / 2, 2 (sg % 2) + {0, 1}) of K K^T and Q K^T (tiles above the diagonal are skipped).
#define PAS (CH + 4)
#define KB (DK / 16)
groupshared float pA[CH * PAS];     // K K^T -> L, later Q K^T
groupshared float pX[CH * PAS];     // (I + L)^-1
groupshared float pG[CH];
groupshared float pB[CH];

[shader("compute")]
[numthreads(NTC, 1, 1)]
void delta_chunk_prep(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ChunkParams p)
{
    uint h = gid.x, hk = h % HK, c = gid.y;
    uint tid = ltid.x, sg = tid / 32;
    int4 run = chunk_run(p, gid.z);
    uint n = uint(run.z);
    if (c * CH >= n || n < CHUNK_MIN) return;
    uint row0 = uint(run.y) + c * CH;
    uint nv = min(uint(CH), n - c * CH);
    const uint64_t qkStride = uint64_t(HK) * DK;

    // Chunk-local inclusive cumulative log decay (two 32-row subgroup scans) and beta.
    if (tid < CH)
    {
        float gv = tid < nv ? p.g[uint64_t(row0 + tid) * HV + h] : 0.0;
        pG[tid] = WavePrefixSum(gv) + gv;
        pB[tid] = tid < nv ? p.beta[uint64_t(row0 + tid) * HV + h] : 0.0;
    }

    // K K^T and Q K^T tiles of this subgroup; Q K^T stays in registers until pA is free again.
    uint rb = sg / 2;
    MC qkT[2];
    {
        DECL_FRAGS(MA, ak, KB)
        DECL_FRAGS(MA, aq, KB)
        [ForceUnroll]
        for (uint kk = 0; kk < KB; kk++)
        {
            LOAD_FRAG(MA, ak, kk, CoopMatMatrixLayout.RowMajor, p.qk, qk_off(0, 1, row0 + rb * 16, hk) + kk * 16, qk_off(1, 1, row0 + rb * 16, hk) + kk * 16, qkStride);
            LOAD_FRAG(MA, aq, kk, CoopMatMatrixLayout.RowMajor, p.qk, qk_off(0, 0, row0 + rb * 16, hk) + kk * 16, qk_off(1, 0, row0 + rb * 16, hk) + kk * 16, qkStride);
        }
        [ForceUnroll]
        for (uint j = 0; j < 2; j++)
        {
            uint cb = 2 * (sg % 2) + j;
            MC acc = MC(0.0), cor = MC(0.0), acq = MC(0.0), coq = MC(0.0);
            if (cb <= rb)
            {
                DECL_FRAGS(MB, b, KB)
                [ForceUnroll]
                for (uint kk = 0; kk < KB; kk++)
                {
                    LOAD_FRAG(MB, b, kk, CoopMatMatrixLayout.ColumnMajor, p.qk, qk_off(0, 1, row0 + cb * 16, hk) + kk * 16, qk_off(1, 1, row0 + cb * 16, hk) + kk * 16, qkStride);
                }
                [ForceUnroll]
                for (uint kk = 0; kk < KB; kk++)
                {
                    MMA_PARTS(acc, cor, akH[kk], akL[kk], bH[kk], bL[kk]);
                    MMA_PARTS(acq, coq, aqH[kk], aqL[kk], bH[kk], bL[kk]);
                }
                MMA_DONE(acc, cor);
                MMA_DONE(acq, coq);
                acc.Store<CoopMatMatrixLayout.RowMajor>(pA, rb * 16 * PAS + cb * 16, PAS);
            }
            qkT[j] = acq;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid >= 32 && tid < CH) pG[tid] += pG[31];
    GroupMemoryBarrierWithGroupSync();
    if (tid < nv) p.gc[uint64_t(row0 + tid) * HV + h] = pG[tid];

    // L (in place).
    for (uint e = tid; e < CH * CH; e += NTC)
    {
        uint t = e / CH, s = e % CH;
        pA[t * PAS + s] = t < nv && s < t ? pB[t] * exp(pG[t] - pG[s]) * pA[t * PAS + s] : 0.0;
    }
    GroupMemoryBarrierWithGroupSync();
    // X = (I + L)^-1 by forward substitution without workgroup barriers: thread j solves column j in registers
    // (fully unrolled, so x stays in registers; L is read as a broadcast). Two partial sums halve the dependency chain.
    if (tid < CH)
    {
        float x[CH];
        [ForceUnroll]
        for (uint i = 0; i < CH; i++)
        {
            float a0 = i == tid ? 1.0 : 0.0, a1 = 0.0;
            [ForceUnroll]
            for (uint s = 0; s + 1 < i; s += 2)
            {
                a0 -= pA[i * PAS + s] * x[s];
                a1 -= pA[i * PAS + s + 1] * x[s + 1];
            }
            if ((i & 1) != 0) a0 -= pA[i * PAS + i - 1] * x[i - 1];
            x[i] = a0 + a1;
            pX[i * PAS + tid] = x[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Tb = X diag(beta) for the live rows.
    for (uint e = tid; e < nv * CH; e += NTC)
    {
        uint t = e / CH, s = e % CH;
        half hi, lo;
        split2(pX[t * PAS + s] * pB[s], hi, lo);
        p.tp[tp_off(0, 0, row0 + t, h) + s] = hi;
#if PARTS > 1
        p.tp[tp_off(1, 0, row0 + t, h) + s] = lo;
#endif
    }

    [ForceUnroll]
    for (uint j = 0; j < 2; j++)
    {
        uint cb = 2 * (sg % 2) + j;
        if (cb <= rb) qkT[j].Store<CoopMatMatrixLayout.RowMajor>(pA, rb * 16 * PAS + cb * 16, PAS);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint e = tid; e < nv * CH; e += NTC)
    {
        uint t = e / CH, s = e % CH;
        float pv = s <= t ? SCALE * exp(pG[t] - pG[s]) * pA[t * PAS + s] : 0.0;
        half hi, lo;
        split2(pv, hi, lo);
        p.tp[tp_off(0, 1, row0 + t, h) + s] = hi;
#if PARTS > 1
        p.tp[tp_off(1, 1, row0 + t, h) + s] = lo;
#endif
    }
}
#endif

#ifdef KERNEL_DELTA_CHUNK_STATE
// The sequential pass runs STATE_THREADS (4 or 8 subgroups); 4 subgroups keep the registers low enough for two resident
// workgroups per multiprocessor (chosen by the lowering).
#ifndef STATE_THREADS
#define STATE_THREADS 256
#endif
#undef NTC
#define NTC STATE_THREADS
// ---------------------------------------------------------------------------------------------
// Workgroup (value head h, state columns [j0, j0 + BV), run z): walks the run's chunks; S^T [DK, BV] lives in f32
// accumulators. Subgroup sg owns the state row tiles sg, sg + 8, ... (each with all BV / 16 column tiles).
#define CB (BV / 16)
#define RB (DK / 16)
#define S_ROWS ((RB + NSG - 1) / NSG)
#define ROWS_A (DK > CH ? DK : CH)
#define FS (BV + 8)                 // f16 tile row stride (halves)
#define XS BV                       // f32 staging row stride
#define X_ROWS (DK > 2 * CH ? DK : 2 * CH)
#define V_PER (CH * BV / NTC)
groupshared half fA[PARTS * ROWS_A * FS];   // S^T [DK][BV] planes, later U' = diag(exp(G_last - G)) U [64][BV]
groupshared half fB[PARTS * CH * FS];       // R [64][BV], later U
groupshared float fX[X_ROWS * XS];          // [K; Q] S^T, then U, then P U (two parts: first the f32 state)
groupshared float fG[CH];

// Writes x as f16 parts into a planes array (plane stride `planeSize`).
#define STORE_PARTS(arr, planeSize, idx, x) { half hi_, lo_; split2(x, hi_, lo_); arr[idx] = hi_; if (PARTS > 1) arr[(planeSize) + (idx)] = lo_; }

// Loads G (thread t < 64: row t) and V (V_PER values per thread) of the chunk starting at run row c.
#define LOAD_V_G(c)                                                                                                   \
    {                                                                                                                 \
        uint r0_ = uint(run.y) + (c), nv_ = min(uint(CH), n - (c));                                                   \
        if (tid < CH) gNext = p.gc[uint64_t(r0_ + min(tid, nv_ - 1)) * HV + h];                                       \
        [ForceUnroll]                                                                                                 \
        for (uint i_ = 0; i_ < V_PER; i_++)                                                                          \
        {                                                                                                             \
            uint e_ = tid + i_ * NTC, t_ = e_ / BV, j_ = e_ % BV;                                                     \
            vr[i_] = t_ < nv_ ? p.v[uint64_t(r0_ + t_) * VS0 + uint64_t(h) * VS1 + j0 + j_] : 0.0;                  \
        }                                                                                                             \
    }
[shader("compute")]
[numthreads(NTC, 1, 1)]
void delta_chunk_state(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ChunkParams p)
{
    uint h = gid.x, hk = h % HK, j0 = gid.y * BV;
    uint tid = ltid.x, sg = tid / 32;
    int4 run = chunk_run(p, gid.z);
    uint n = uint(run.z);
    if (n == 0 || n < CHUNK_MIN) return;
    bool fresh = run.w != 0;
    STATE_T* st = p.state + ((uint64_t(run.x) * HV + h) * DV + j0) * DK;   // S^T(i, j) at st[j * DK + i]

    MC S[S_ROWS][CB];
#if STATE_FMT
    // 16-bit state: widened into the f32 staging tile as S^T [DK][BV] (fX is free until step 1).
    if (!fresh)
    {
        for (uint e = tid; e < DK * BV; e += NTC)
        {
            uint i = e % DK, j = e / DK;
            fX[i * XS + j] = state_load(st[j * DK + i]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    [ForceUnroll]
    for (uint m = 0; m < S_ROWS; m++)
    {
        uint ri = sg + m * NSG;
        [ForceUnroll]
        for (uint ci = 0; ci < CB; ci++)
        {
#if STATE_FMT
            S[m][ci] = fresh || ri >= RB ? MC(0.0) : MC.Load<CoopMatMatrixLayout.RowMajor>(fX, ri * 16 * XS + ci * 16, XS);
#else
            S[m][ci] = fresh || ri >= RB ? MC(0.0) : MC.Load<CoopMatMatrixLayout.ColumnMajor>(st, ci * 16 * DK + ri * 16, DK);
#endif
        }
    }
#if STATE_FMT
    GroupMemoryBarrierWithGroupSync();   // fX is rewritten by step 1
#endif

    const uint64_t qkStride = uint64_t(HK) * DK;
    const uint64_t tpStride = uint64_t(HV) * CH;
    // G and V of the next chunk are loaded while the current one is processed.
    float gNext = 0;
    float vr[V_PER];
    LOAD_V_G(0)
    for (uint c0 = 0; c0 < n; c0 += CH)
    {
        uint row0 = uint(run.y) + c0;
        uint nv = min(uint(CH), n - c0);
        bool more = c0 + CH < n;
        if (tid < CH) fG[tid] = gNext;

        // 1. S^T -> f16 planes fA [DK][BV]. Two parts: staged in f32 and split in scalar code (a converted accumulator
        // can come back from the driver unrounded, which would leave the lo part zero).
        [ForceUnroll]
        for (uint m = 0; m < S_ROWS; m++)
        {
            uint ri = sg + m * NSG;
            if (ri >= RB) continue;
            [ForceUnroll]
            for (uint ci = 0; ci < CB; ci++)
            {
#if PARTS > 1
                S[m][ci].Store<CoopMatMatrixLayout.RowMajor>(fX, ri * 16 * XS + ci * 16, XS);
#else
                MCH hi = MCH(S[m][ci]);
                hi.Store<CoopMatMatrixLayout.RowMajor>(fA, ri * 16 * FS + ci * 16, FS);
#endif
            }
        }
#if PARTS > 1
        GroupMemoryBarrierWithGroupSync();
        for (uint e = tid; e < DK * BV; e += NTC)
        {
            uint i = e / BV, j = e % BV;
            STORE_PARTS(fA, ROWS_A * FS, i * FS + j, fX[i * XS + j]);
        }
#endif
        GroupMemoryBarrierWithGroupSync();
        float gLast = fG[CH - 1];

        // 2. [K; Q] S^T -> fX rows 0..63 (K), 64..127 (Q): subgroup sg owns the row tiles sg, sg + NSG, ...
        [ForceUnroll]
        for (uint tt = 0; tt < 8 / NSG; tt++)
        {
            uint tile = sg + tt * NSG;
            uint which = tile < 4 ? 1 : 0;
            uint r = row0 + (tile % 4) * 16;
            MC acc[CB], cor[CB];
            [ForceUnroll]
            for (uint ci = 0; ci < CB; ci++)
            {
                acc[ci] = MC(0.0);
                cor[ci] = MC(0.0);
            }
            [ForceUnroll]
            for (uint kk = 0; kk < RB; kk++)
            {
                DECL_FRAGS(MA, aA, 1)
                LOAD_FRAG(MA, aA, 0, CoopMatMatrixLayout.RowMajor, p.qk, qk_off(0, which, r, hk) + kk * 16, qk_off(1, which, r, hk) + kk * 16, qkStride);
                [ForceUnroll]
                for (uint ci = 0; ci < CB; ci++)
                {
                    MB bH = MB.Load<CoopMatMatrixLayout.RowMajor>(fA, kk * 16 * FS + ci * 16, FS);
#if PARTS > 1
                    MB bL = MB.Load<CoopMatMatrixLayout.RowMajor>(fA, ROWS_A * FS + kk * 16 * FS + ci * 16, FS);
#endif
                    MMA_PARTS(acc[ci], cor[ci], aAH[0], aAL[0], bH, bL);
                }
            }
            [ForceUnroll]
            for (uint ci = 0; ci < CB; ci++)
            {
                MMA_DONE(acc[ci], cor[ci]);
                acc[ci].Store<CoopMatMatrixLayout.RowMajor>(fX, tile * 16 * XS + ci * 16, XS);
            }
        }
        // Operands of the 64 x 64 products, row tile sg % 4: with 8 subgroups, subgroups 0-3 compute U = Tb R (step 4) and
        // 4-7 P U (step 6a); with 4, every subgroup computes both. The first operand is loaded now so that its latency
        // overlaps steps 2 and 3.
        bool doU = NSG == 4 || sg < 4, doPU = NSG == 4 || sg >= 4;
        uint tileRow = row0 + (sg % 4) * 16;
        DECL_FRAGS(MA, aT, 4)
        {
            uint which = doU ? 0 : 1;
            [ForceUnroll]
            for (uint kk = 0; kk < 4; kk++)
            {
                LOAD_FRAG(MA, aT, kk, CoopMatMatrixLayout.RowMajor, p.tp, tp_off(0, which, tileRow, h) + kk * 16, tp_off(1, which, tileRow, h) + kk * 16, tpStride);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. R = V - diag(exp(G)) K S^T -> fB.
        [ForceUnroll]
        for (uint i = 0; i < V_PER; i++)
        {
            uint e = tid + i * NTC, t = e / BV, j = e % BV;
            float r = t < nv ? vr[i] - exp(fG[t]) * fX[t * XS + j] : 0.0;
            STORE_PARTS(fB, CH * FS, t * FS + j, r);
        }
        if (more) LOAD_V_G(c0 + CH)
        GroupMemoryBarrierWithGroupSync();

        // 4. U = Tb R -> fX rows 0..63.
        if (doU)
        {
            [ForceUnroll]
            for (uint ci = 0; ci < CB; ci++)
            {
                MC acc = MC(0.0), cor = MC(0.0);
                [ForceUnroll]
                for (uint kk = 0; kk < 4; kk++)
                {
                    MB bH = MB.Load<CoopMatMatrixLayout.RowMajor>(fB, kk * 16 * FS + ci * 16, FS);
#if PARTS > 1
                    MB bL = MB.Load<CoopMatMatrixLayout.RowMajor>(fB, CH * FS + kk * 16 * FS + ci * 16, FS);
#endif
                    MMA_PARTS(acc, cor, aTH[kk], aTL[kk], bH, bL);
                }
                MMA_DONE(acc, cor);
                acc.Store<CoopMatMatrixLayout.RowMajor>(fX, (sg % 4) * 16 * XS + ci * 16, XS);
            }
        }
        if (NSG == 4)
        {
            // P for step 6a.
            [ForceUnroll]
            for (uint kk = 0; kk < 4; kk++)
            {
                LOAD_FRAG(MA, aT, kk, CoopMatMatrixLayout.RowMajor, p.tp, tp_off(0, 1, tileRow, h) + kk * 16, tp_off(1, 1, tileRow, h) + kk * 16, tpStride);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 5. U -> fB, diag(exp(G_last - G)) U -> fA (rows past the chunk end are zero).
        for (uint e = tid; e < CH * BV; e += NTC)
        {
            uint t = e / BV, j = e % BV;
            float u = t < nv ? fX[t * XS + j] : 0.0;
            STORE_PARTS(fB, CH * FS, t * FS + j, u);
            STORE_PARTS(fA, ROWS_A * FS, t * FS + j, u * exp(gLast - fG[t]));
        }
        GroupMemoryBarrierWithGroupSync();

        // 6a. P U -> fX rows 0..63.
        if (doPU)
        {
            [ForceUnroll]
            for (uint ci = 0; ci < CB; ci++)
            {
                MC acc = MC(0.0), cor = MC(0.0);
                [ForceUnroll]
                for (uint kk = 0; kk < 4; kk++)
                {
                    MB bH = MB.Load<CoopMatMatrixLayout.RowMajor>(fB, kk * 16 * FS + ci * 16, FS);
#if PARTS > 1
                    MB bL = MB.Load<CoopMatMatrixLayout.RowMajor>(fB, CH * FS + kk * 16 * FS + ci * 16, FS);
#endif
                    MMA_PARTS(acc, cor, aTH[kk], aTL[kk], bH, bL);
                }
                MMA_DONE(acc, cor);
                acc.Store<CoopMatMatrixLayout.RowMajor>(fX, (sg % 4) * 16 * XS + ci * 16, XS);
            }
        }
        // 6b. S^T = exp(G_last) S^T + K^T U'. Its K^T operand is loaded only now: preloading it next to the step 4 / 6a
        // operands would cost the registers of a second resident workgroup.
        DECL_FRAGS(MA, aK, 4 * S_ROWS)
        [ForceUnroll]
        for (uint m = 0; m < S_ROWS; m++)
        {
            uint ri = min(sg + m * NSG, uint(RB - 1));
            [ForceUnroll]
            for (uint kk = 0; kk < 4; kk++)
            {
                LOAD_FRAG(MA, aK, m * 4 + kk, CoopMatMatrixLayout.ColumnMajor, p.qk, qk_off(0, 1, row0 + kk * 16, hk) + ri * 16, qk_off(1, 1, row0 + kk * 16, hk) + ri * 16, qkStride);
            }
        }
        MC decay = MC(exp(gLast));
        [ForceUnroll]
        for (uint m = 0; m < S_ROWS; m++)
        {
            if (sg + m * NSG >= RB) continue;
            [ForceUnroll]
            for (uint ci = 0; ci < CB; ci++)
            {
                MC acc = S[m][ci] * decay, cor = MC(0.0);
                [ForceUnroll]
                for (uint kk = 0; kk < 4; kk++)
                {
                    MB bH = MB.Load<CoopMatMatrixLayout.RowMajor>(fA, kk * 16 * FS + ci * 16, FS);
#if PARTS > 1
                    MB bL = MB.Load<CoopMatMatrixLayout.RowMajor>(fA, ROWS_A * FS + kk * 16 * FS + ci * 16, FS);
#endif
                    MMA_PARTS(acc, cor, aKH[m * 4 + kk], aKL[m * 4 + kk], bH, bL);
                }
                MMA_DONE(acc, cor);
                S[m][ci] = acc;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 7. O = SCALE diag(exp(G)) Q S^T + P U.
        for (uint e = tid; e < nv * BV; e += NTC)
        {
            uint t = e / BV, j = e % BV;
            p.o[(uint64_t(row0 + t) * HV + h) * DV + j0 + j] = SCALE * exp(fG[t]) * fX[(CH + t) * XS + j] + fX[t * XS + j];
        }
        GroupMemoryBarrierWithGroupSync();   // fG and fX are rewritten by the next chunk
    }

    [ForceUnroll]
    for (uint m = 0; m < S_ROWS; m++)
    {
        uint ri = sg + m * NSG;
        if (ri >= RB) continue;
        [ForceUnroll]
        for (uint ci = 0; ci < CB; ci++)
        {
#if STATE_FMT
            S[m][ci].Store<CoopMatMatrixLayout.RowMajor>(fX, ri * 16 * XS + ci * 16, XS);
#else
            S[m][ci].Store<CoopMatMatrixLayout.ColumnMajor>(st, ci * 16 * DK + ri * 16, DK);
#endif
        }
    }
#if STATE_FMT
    // 16-bit state: narrowed from the staging tile, rounded once for the whole run.
    GroupMemoryBarrierWithGroupSync();
    for (uint e = tid; e < DK * BV; e += NTC)
    {
        uint i = e % DK, j = e / DK;
        st[j * DK + i] = state_store(state_round(fX[i * XS + j]));
    }
#endif
}
#endif
