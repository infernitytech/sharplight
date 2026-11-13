// Token sampling over logits rows (StandardOps.Sample, semantics in SamplingHead.cs). One workgroup per row; only workgroup
// barriers and shared-memory integer atomics are used (no subgroup operations), so the kernel runs on every Vulkan device.
// Per row: max / arg-max pass. Sampled rows with top-k (top-p implies top-k <= NC) count, for a few windows below the maximum, how
// many logits fall inside; when the smallest window holding k logits holds at most NC, those logits are compacted into shared
// memory and sorted (bitonic), which gives the k-th largest and the sorted candidates in two more passes. Otherwise the k-th largest
// comes from a radix select over order-preserving keys. Then: Gumbel-max over the kept tokens, or (top-p) the nucleus cut over the
// sorted candidates followed by a Gumbel-max over the kept prefix. The noise of token i is a hash of (seed, counter, i); the CPU
// implementation (SampleOp.cs, SamplingReference.cs) uses the same functions and the same rules.
// Macros: V (vocabulary), WG (threads per workgroup, power of two, 256), NC (candidate capacity, SamplingHead.MaxNucleusCandidates),
// VEC (1 when V is divisible by 4: float4 loads).
#include "common.slang"

#ifndef WG
#define WG 256
#endif
#ifndef NC
#define NC 1024
#endif

#define U 8                 // logits loaded per thread and loop iteration (independent loads in flight)
#define NBINS 2048          // radix histogram bins (11 bits)
#define NWIN 8              // windows m - 0.25 * 2^j below the maximum
#define FLAG_ACTIVE 1
#define FLAG_GREEDY 2
#define FLAG_LOGPROB 4
#define NONE 0xFFFFFFFFu
#define NEG_INF (-1.0 / 0.0)

struct SampleParams { float* x; int* prm; int* tok; float* lp; uint R; };

groupshared float sf[WG];
groupshared uint su[WG];
groupshared uint hist[NBINS];   // radix histogram; window counts; candidate probabilities (as float bits)
groupshared uint ckey[NC];
groupshared uint cidx[NC];
groupshared uint bcast[2];

// Order-preserving key of a float (larger float -> larger key).
uint okey(float f) { uint u = asuint(f); return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u); }
float ofloat(uint k) { return asfloat((k & 0x80000000u) != 0 ? (k & 0x7FFFFFFFu) : ~k); }

uint mix32(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint stream_of(uint seed, uint counter) { return mix32(seed ^ mix32(counter ^ 0x6a09e667u)); }

float gumbel(uint stream, uint i)
{
    uint h = mix32(stream ^ mix32(i + 0x9e3779b9u));
    float u = (float(h >> 8) + 0.5) * (1.0 / 16777216.0);
    return -log(-log(u));
}

float window_of(uint j) { return 0.25 * float(1u << j); }

// Thread t owns the logits t, t + WG, t + 2 WG, ... (VEC = 0) or the float4 groups t, t + WG, ... (VEC = 1, V divisible by 4), in
// increasing order; a block is the next U of them (E logits), loaded before use so that the loads are in flight together.
#if VEC
#define NV (V / 4)
#define E (U * 4)
uint elem(uint b, uint e) { return 4 * (b + (e / 4) * WG) + (e % 4); }
void load_block(float* xr, uint b, out float v[E])
{
    [ForceUnroll] for (uint u = 0; u < U; u++)
    {
        uint q = b + u * WG;
        float4 f = q < NV ? ((float4*)xr)[q] : float4(NEG_INF);
        v[4 * u] = f.x; v[4 * u + 1] = f.y; v[4 * u + 2] = f.z; v[4 * u + 3] = f.w;
    }
}
// Logits owned by thread t.
uint owned(uint t) { return t < NV ? ((NV - 1 - t) / WG + 1) * 4 : 0; }
#else
#define NV V
#define E U
uint elem(uint b, uint e) { return b + e * WG; }
void load_block(float* xr, uint b, out float v[E])
{
    [ForceUnroll] for (uint e = 0; e < E; e++) { uint i = elem(b, e); v[e] = i < V ? xr[i] : NEG_INF; }
}
uint owned(uint t) { return t < V ? (V - 1 - t) / WG + 1 : 0; }
#endif

// (value, index) arg-max over the workgroup; ties go to the lower index. Every thread gets the result.
void reduce_argmax(uint t, inout float v, inout uint i)
{
    sf[t] = v; su[t] = i;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            float b = sf[t + s]; uint ib = su[t + s];
            if (b > sf[t] || (b == sf[t] && ib < su[t])) { sf[t] = b; su[t] = ib; }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    v = sf[0]; i = su[0];
    GroupMemoryBarrierWithGroupSync();
}

float reduce_sum(uint t, float v)
{
    sf[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s) sf[t] += sf[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float r = sf[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

uint reduce_min(uint t, uint v)
{
    su[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s) su[t] = min(su[t], su[t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = su[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Exclusive prefix sum over threads (sum of v over threads 0..t-1); `total` receives the sum over all threads.
uint exclusive_scan(uint t, uint v, out uint total)
{
    su[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < WG; off <<= 1)
    {
        uint add = t >= off ? su[t - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        su[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = su[t] - v;
    total = su[WG - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Inclusive prefix sum over threads (sum of v over threads 0..t).
float prefix_sum(uint t, float v)
{
    sf[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < WG; off <<= 1)
    {
        float add = t >= off ? sf[t - off] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        sf[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    float r = sf[t];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Index of the smallest window holding at least k logits (NWIN when none does); `mine` receives this thread's count in it and
// `total` the count of the whole row.
uint choose_window(uint t, float* xr, float m, uint k, out uint mine, out uint total)
{
    uint c[NWIN];
    [ForceUnroll] for (uint j = 0; j < NWIN; j++) c[j] = 0;
    for (uint b = t; b < NV; b += U * WG)
    {
        float v[E];
        load_block(xr, b, v);
        [ForceUnroll] for (uint e = 0; e < E; e++)
        {
            // Same predicate as the compaction (v >= m - w), padding entries are -inf.
            [ForceUnroll] for (uint j = 0; j < NWIN; j++) c[j] += v[e] >= m - window_of(j) ? 1u : 0u;
        }
    }
    [ForceUnroll] for (uint j = 0; j < NWIN; j++) hist[t * NWIN + j] = c[j];
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            [ForceUnroll] for (uint j = 0; j < NWIN; j++) hist[t * NWIN + j] += hist[(t + s) * NWIN + j];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint w = NWIN;
    for (int j = NWIN - 1; j >= 0; j--)
    {
        if (hist[j] >= k) w = uint(j);
    }
    total = w < NWIN ? hist[w] : V;
    mine = owned(t);   // every entry of this thread when no window is wide enough
    [ForceUnroll] for (uint j = 0; j < NWIN; j++) mine = j == w ? c[j] : mine;
    GroupMemoryBarrierWithGroupSync();
    return w;
}

// Key of the k-th largest logit (1-based k <= number of logits >= lo); radix select over 11/11/10-bit digits.
// Only logits >= lo take part, which keeps the shared-memory atomics few.
uint select_kth(uint t, float* xr, uint k, float lo)
{
    uint prefix = 0, mask = 0;
    for (uint pass = 0; pass < 3; pass++)
    {
        uint shift = pass == 0 ? 21u : pass == 1 ? 10u : 0u;
        uint nb = pass == 2 ? 1024u : 2048u;
        for (uint b = t; b < nb; b += WG) hist[b] = 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint b = t; b < NV; b += U * WG)
        {
            float v[E];
            load_block(xr, b, v);
            [ForceUnroll] for (uint e = 0; e < E; e++)
            {
                uint key = okey(v[e]);
                if (elem(b, e) < V && v[e] >= lo && (key & mask) == prefix)
                {
                    InterlockedAdd(hist[(key >> shift) & (nb - 1)], 1u);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        // Each thread owns `per` consecutive bins; find the bin where the count from the top reaches k.
        uint per = nb / WG;
        uint local = 0;
        for (uint b = 0; b < per; b++) local += hist[t * per + b];
        uint all;
        uint below = exclusive_scan(t, local, all);
        uint above = all - below - local;
        if (above < k && k <= above + local)
        {
            for (int b = int(per) - 1; b >= 0; b--)
            {
                uint h = hist[t * per + uint(b)];
                if (k <= above + h) { bcast[0] = t * per + uint(b); bcast[1] = above; break; }
                above += h;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint bin = bcast[0];
        k -= bcast[1];
        prefix |= bin << shift;
        mask |= (nb - 1) << shift;
        bool whole = k == hist[bin];   // the k-th largest is the smallest of this bin: every key >= prefix is in
        GroupMemoryBarrierWithGroupSync();
        if (whole) break;
    }
    // After an early exit `prefix` may lie below lo; logits below lo were not counted, so they must not pass the threshold.
    return max(prefix, okey(lo));
}

// Writes the logits with key >= minKey (at most NC, `mine` of them from this thread) to the candidate arrays in thread-major order.
// Returns the number written.
uint compact(uint t, float* xr, uint minKey, uint mine)
{
    uint all;
    uint pos = exclusive_scan(t, mine, all);
    for (uint b = t; b < NV && pos < NC; b += U * WG)
    {
        float v[E];
        load_block(xr, b, v);
        [ForceUnroll] for (uint e = 0; e < E; e++)
        {
            uint i = elem(b, e);
            uint key = okey(v[e]);
            if (i < V && key >= minKey && pos < NC) { ckey[pos] = key; cidx[pos] = i; pos++; }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return min(all, uint(NC));
}

// Sorts the first n candidates by descending key, then ascending index (bitonic sort over the next power of two).
void sort_candidates(uint t, uint n)
{
    uint ns = 1;
    while (ns < n) ns <<= 1;
    for (uint j = n + t; j < ns; j += WG) { ckey[j] = 0; cidx[j] = NONE; }
    GroupMemoryBarrierWithGroupSync();
    for (uint size = 2; size <= ns; size <<= 1)
    {
        for (uint stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (uint i = t; i < ns; i += WG)
            {
                uint l = i ^ stride;
                if (l > i)
                {
                    uint ki = ckey[i], kl = ckey[l], ii = cidx[i], il = cidx[l];
                    bool iFirst = ki > kl || (ki == kl && ii < il);
                    if (((i & size) == 0) != iFirst)
                    {
                        ckey[i] = kl; ckey[l] = ki; cidx[i] = il; cidx[l] = ii;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

// Number of sorted candidates with key >= minKey (a prefix).
uint count_at_least(uint t, uint n, uint minKey)
{
    uint first = n;
    for (uint j = t; j < n; j += WG) { if (ckey[j] < minKey) first = min(first, j); }
    return reduce_min(t, first);
}

// Gumbel-max over sorted candidates [0, n) whose logit is >= minLogit; NONE when there is none.
uint gumbel_candidates(uint t, uint n, float m, float invT, float minLogit, uint stream)
{
    float best = NEG_INF;
    uint bi = NONE;
    for (uint j = t; j < n; j += WG)
    {
        float v = ofloat(ckey[j]);
        uint i = cidx[j];
        if (v >= minLogit)
        {
            float s = (v - m) * invT + gumbel(stream, i);
            if (bi == NONE || s > best || (s == best && i < bi)) { best = s; bi = i; }
        }
    }
    reduce_argmax(t, best, bi);
    return bi;
}

// Nucleus cut over sorted candidates [0, n): min-p prefix, then the shortest prefix whose mass reaches topP of the candidates' mass.
uint nucleus_cut(uint t, uint n, float m, float invT, float topP, float minP)
{
    const uint PER = NC / WG;
    float local = 0.0;
    uint keep = n;
    for (uint q = 0; q < PER; q++)
    {
        uint j = t * PER + q;
        float pj = j < n ? exp((ofloat(ckey[j]) - m) * invT) : 0.0;
        hist[j] = asuint(pj);
        local += pj;
        if (j < n && j > 0 && pj < minP) keep = min(keep, j);
    }
    float z = reduce_sum(t, local);
    uint cut = reduce_min(t, keep);

    float mass = 0.0;
    for (uint q = 0; q < PER; q++)
    {
        uint j = t * PER + q;
        mass += j < cut ? asfloat(hist[j]) : 0.0;
    }
    float before = prefix_sum(t, mass) - mass;
    uint first = cut;
    for (uint q = 0; q < PER; q++)
    {
        uint j = t * PER + q;
        if (j < cut)
        {
            before += asfloat(hist[j]);
            if (before >= topP * z) { first = min(first, j + 1); }
        }
    }
    return reduce_min(t, first);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void sample(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform SampleParams p)
{
    uint row = gid.x;
    uint t = ltid.x;
    if (row >= p.R) return;
    int* pr = p.prm + row * 8;
    uint flags = uint(pr[0]);
    if ((flags & FLAG_ACTIVE) == 0)
    {
        if (t == 0) { p.tok[row] = -1; p.lp[row] = 0.0; }
        return;
    }
    float* xr = p.x + uint64_t(row) * V;

    // Maximum and first arg-max.
    float m = NEG_INF;
    uint am = NONE;
    for (uint b = t; b < NV; b += U * WG)
    {
        float v[E];
        load_block(xr, b, v);
        [ForceUnroll] for (uint e = 0; e < E; e++)
        {
            uint i = elem(b, e);
            if (i < V && (v[e] > m || am == NONE)) { m = v[e]; am = i; }
        }
    }
    reduce_argmax(t, m, am);

    uint token = am;
    if ((flags & FLAG_GREEDY) == 0)
    {
        float temp = asfloat(pr[2]);
        float invT = 1.0 / temp;
        uint topK = uint(pr[1]);
        float topP = asfloat(pr[3]);
        float minP = asfloat(pr[4]);
        uint stream = stream_of(uint(pr[5]), uint(pr[6]));
        bool nucleus = topP > 0.0 && topP < 1.0;
        uint k = nucleus && (topK == 0 || topK > NC) ? NC : topK;
        bool useTopK = k > 0 && k < V;
        float minLogit = minP > 0.0 ? m + temp * log(minP) : NEG_INF;

        // Top-k threshold; when the logits near the maximum fit, they become the sorted candidates right away.
        uint kthKey = 0;
        uint n = 0;   // sorted candidates holding every logit with key >= kthKey, or 0
        if (useTopK)
        {
            uint mine, total;
            uint w = choose_window(t, xr, m, k, mine, total);
            float lo = w < NWIN ? m - window_of(w) : NEG_INF;
            if (total <= NC)
            {
                uint count = compact(t, xr, okey(lo), mine);
                sort_candidates(t, count);
                kthKey = ckey[k - 1];
                n = count_at_least(t, count, kthKey);
            }
            else
            {
                kthKey = select_kth(t, xr, k, lo);
            }
        }

        if (!nucleus && n > 0)
        {
            uint bi = gumbel_candidates(t, n, m, invT, minLogit, stream);
            token = bi == NONE ? am : bi;
        }
        else if (!nucleus)
        {
            // Gumbel-max over every token above both thresholds.
            float best = NEG_INF;
            uint bi = NONE;
            for (uint b = t; b < NV; b += U * WG)
            {
                float v[E];
                load_block(xr, b, v);
                [ForceUnroll] for (uint e = 0; e < E; e++)
                {
                    uint i = elem(b, e);
                    if (i < V && okey(v[e]) >= kthKey && v[e] >= minLogit)
                    {
                        float s = (v[e] - m) * invT + gumbel(stream, i);
                        if (s > best || bi == NONE) { best = s; bi = i; }
                    }
                }
            }
            reduce_argmax(t, best, bi);
            token = bi == NONE ? am : bi;
        }
        else
        {
            if (n == 0)
            {
                // Candidates = logits with key >= kthKey (all of them when there is no top-k), at most NC in thread-major order.
                uint mine = 0;
                for (uint b = t; b < NV; b += U * WG)
                {
                    float v[E];
                    load_block(xr, b, v);
                    [ForceUnroll] for (uint e = 0; e < E; e++) mine += elem(b, e) < V && okey(v[e]) >= kthKey ? 1u : 0u;
                }
                n = compact(t, xr, kthKey, mine);
                sort_candidates(t, n);
            }
            uint cut = nucleus_cut(t, n, m, invT, topP, minP);
            uint bi = gumbel_candidates(t, cut, m, invT, NEG_INF, stream);
            token = bi == NONE ? am : bi;
        }
    }

    float logprob = 0.0;
    if ((flags & FLAG_LOGPROB) != 0)
    {
        float s = 0.0;
        for (uint b = t; b < NV; b += U * WG)
        {
            float v[E];
            load_block(xr, b, v);
            [ForceUnroll] for (uint e = 0; e < E; e++) s += elem(b, e) < V ? exp(v[e] - m) : 0.0;
        }
        s = reduce_sum(t, s);
        logprob = xr[token] - m - log(s);
    }
    if (t == 0)
    {
        p.tok[row] = int(token);
        p.lp[row] = logprob;
    }
}
