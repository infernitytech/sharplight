// Token sampling over logits rows (StandardOps.Sample, semantics in SamplingHead.cs). One workgroup per row; only workgroup
// barriers and shared-memory integer atomics are used (no subgroup operations), so the kernel runs on every Vulkan device.
// Per row: max / arg-max pass; for sampled rows an optional top-k threshold (radix select over order-preserving keys, counting only
// elements inside a window below the maximum), then either a Gumbel-max pass over all tokens above the threshold, or (top-p) a
// compaction of the top-k candidates into shared memory, a bitonic sort, the nucleus cut and a Gumbel-max over the kept prefix.
// The random noise of token i is a hash of (seed, counter, i): the CPU implementation (SampleOp.cs) uses the same functions.
// Macros: V (vocabulary), WG (threads per workgroup, power of two, 256), NC (top-p candidate capacity, SamplingHead.MaxNucleusCandidates).
#include "common.slang"

#ifndef WG
#define WG 256
#endif
#ifndef NC
#define NC 1024
#endif

#define NBINS 2048          // radix histogram bins (11 bits)
#define NWIN 8              // windows m - 0.5 * 2^j below the maximum tried to bound the top-k threshold
#define FLAG_ACTIVE 1
#define FLAG_GREEDY 2
#define FLAG_LOGPROB 4
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

// Inclusive suffix sum over threads (sum of v over threads t..WG-1).
uint suffix_sum(uint t, uint v)
{
    su[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < WG; off <<= 1)
    {
        uint add = t + off < WG ? su[t + off] : 0;
        GroupMemoryBarrierWithGroupSync();
        su[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = su[t];
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

// Smallest window w = 0.5 * 2^j such that at least k logits are >= m - w; +inf when none is wide enough.
float choose_window(uint t, float* xr, float m, uint k)
{
    uint c[NWIN];
    [ForceUnroll] for (int j = 0; j < NWIN; j++) c[j] = 0;
    for (uint i = t; i < V; i += WG)
    {
        float d = m - xr[i];
        [ForceUnroll] for (int j = 0; j < NWIN; j++) c[j] += d <= 0.5 * float(1u << j) ? 1u : 0u;
    }
    [ForceUnroll] for (int j = 0; j < NWIN; j++) hist[t * NWIN + j] = c[j];
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            [ForceUnroll] for (int j = 0; j < NWIN; j++) hist[t * NWIN + j] += hist[(t + s) * NWIN + j];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float w = 1.0 / 0.0;
    for (int j = NWIN - 1; j >= 0; j--)
    {
        if (hist[j] >= k) w = 0.5 * float(1u << j);
    }
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
        for (uint i = t; i < V; i += WG)
        {
            float v = xr[i];
            uint key = okey(v);
            if (v >= lo && (key & mask) == prefix)
            {
                InterlockedAdd(hist[(key >> shift) & (nb - 1)], 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        // Each thread owns `per` consecutive bins; find the bin where the count from the top reaches k.
        uint per = nb / WG;
        uint local = 0;
        for (uint b = 0; b < per; b++) local += hist[t * per + b];
        uint above = suffix_sum(t, local) - local;
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
    uint am = 0xFFFFFFFFu;
    for (uint i = t; i < V; i += WG)
    {
        float v = xr[i];
        if (v > m || am == 0xFFFFFFFFu) { m = v; am = i; }
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
        uint kthKey = 0;
        if (useTopK)
        {
            float w = choose_window(t, xr, m, k);
            kthKey = select_kth(t, xr, k, m - w);
        }
        float minLogit = minP > 0.0 ? m + temp * log(minP) : NEG_INF;

        if (!nucleus)
        {
            // Gumbel-max over every token above both thresholds.
            float best = NEG_INF;
            uint bi = 0xFFFFFFFFu;
            for (uint i = t; i < V; i += WG)
            {
                float v = xr[i];
                if (okey(v) >= kthKey && v >= minLogit)
                {
                    float s = (v - m) * invT + gumbel(stream, i);
                    if (s > best || bi == 0xFFFFFFFFu) { best = s; bi = i; }
                }
            }
            reduce_argmax(t, best, bi);
            token = bi == 0xFFFFFFFFu ? am : bi;
        }
        else
        {
            // Compact the top-k candidates (deterministic order: by thread, then by index), capped at NC.
            uint mine = 0;
            for (uint i = t; i < V; i += WG) mine += okey(xr[i]) >= kthKey ? 1u : 0u;
            uint total = suffix_sum(t, mine);   // suffix sums give each thread the count of the threads after it
            uint all = su[0];
            GroupMemoryBarrierWithGroupSync();
            uint n = min(all, uint(NC));
            uint pos = all - total;             // candidates of threads before t
            for (uint i = t; i < V && pos < NC; i += WG)
            {
                uint key = okey(xr[i]);
                if (key >= kthKey) { ckey[pos] = key; cidx[pos] = i; pos++; }
            }
            uint ns = 1;
            while (ns < n) ns <<= 1;
            for (uint j = n + t; j < ns; j += WG) { ckey[j] = 0; cidx[j] = 0xFFFFFFFFu; }
            GroupMemoryBarrierWithGroupSync();

            // Bitonic sort: descending key, ascending index.
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

            // Probabilities relative to the best token, the top-k mass and the min-p cut (a prefix of the sorted list).
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

            // Nucleus: first j < cut whose inclusive mass reaches topP * z.
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
            cut = reduce_min(t, first);

            float best = NEG_INF;
            uint bi = 0xFFFFFFFFu;
            for (uint j = t; j < cut; j += WG)
            {
                float s = (ofloat(ckey[j]) - m) * invT + gumbel(stream, cidx[j]);
                if (s > best || bi == 0xFFFFFFFFu || (s == best && cidx[j] < bi)) { best = s; bi = cidx[j]; }
            }
            reduce_argmax(t, best, bi);
            token = bi == 0xFFFFFFFFu ? am : bi;
        }
    }

    float logprob = 0.0;
    if ((flags & FLAG_LOGPROB) != 0)
    {
        float s = 0.0;
        for (uint i = t; i < V; i += WG) s += exp(xr[i] - m);
        s = reduce_sum(t, s);
        logprob = xr[token] - m - log(s);
    }
    if (t == 0)
    {
        p.tok[row] = int(token);
        p.lp[row] = logprob;
    }
}
