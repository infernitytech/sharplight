// Detection ops: NonMaxSuppression, RoiAlign and the argmax indices of max pooling. Plain compute: shared memory, barriers and
// 64-bit integers only (no subgroup operations).
//   nms_class:   one workgroup of WG threads per (batch, class): candidates above the score threshold, bitonic sort (best score
//                first, ties by lower box index), then greedy suppression in blocks of WG candidates. Writes the kept box indices
//                and their number per (batch, class).
//   nms_gather:  one workgroup: concatenates the kept boxes of all (batch, class) pairs into (batch, class, box) rows and writes
//                their number (the data-dependent count of the output).
//   roi_align:   one thread per output element (ONNX RoiAlign, avg or max of bilinear samples per bin).
//   pool_argmax: one thread per output element (flat input index of the max-pooling maximum, first maximum wins).
#include "common.slang"

#ifndef WG
#define WG 256
#endif

#if defined(KERNEL_NMS_CLASS)
// Macros: C (classes), MAXK (boxes kept per class at most), KCAP (kept-box slots per class, >= MAXK), CAP (sort slots per class,
// a power of two >= the box count), LOCAL (sort slots in shared memory, a power of two; larger candidate sets sort in the global
// slots), IOU, SCORE (thresholds; SCORE may be -inf), CENTER (1: boxes are [cx, cy, w, h]).
struct NmsParams { float* boxes; float* scores; float* keys; int* ids; int* keptIds; float4* keptBoxes; int* keptCount; uint groups; uint S; };

#define MASK_WORDS (WG / 32)

groupshared uint sCount;
groupshared uint sKept;
groupshared float sKey[LOCAL];
groupshared int sIdx[LOCAL];
groupshared float4 sBox[WG];
groupshared int sBoxId[WG];
groupshared uint sDead[WG];
groupshared uint sMask[WG * MASK_WORDS];

// (y1, x1, y2, x2) with y1 <= y2 and x1 <= x2, as the CPU provider computes it.
float4 nms_corners(float* box)
{
#if CENTER
    precise float halfWidth = box[2] / 2.0;
    precise float halfHeight = box[3] / 2.0;
    precise float y1 = box[1] - halfHeight;
    precise float x1 = box[0] - halfWidth;
    precise float y2 = box[1] + halfHeight;
    precise float x2 = box[0] + halfWidth;
    return float4(y1, x1, y2, x2);
#else
    return float4(min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3]));
#endif
}

// Normalized significand (2^23 <= m < 2^24) and exponent of a finite x > 0: x = m * 2^e.
void nms_split(float x, out uint m, out int e)
{
    uint bits = asuint(x);
    int biased = int(bits >> 23);
    m = bits & 0x7FFFFFu;
    if (biased == 0)
    {
        e = -149;
        while (m < 0x800000u)
        {
            m <<= 1;
            e--;
        }
    }
    else
    {
        m |= 0x800000u;
        e = biased - 150;
    }
}

// Whether the correctly rounded quotient fl(num / den) exceeds t, for num, den > 0. The hardware division may be off by an ulp,
// so the test is exact in integers: IoUs on the threshold (an ONNX conformance case) decide like the CPU provider.
bool nms_quotient_above(float num, float den, float t)
{
    float q = num / den;
    if (!(t >= 1.17549435e-38) || isinf(t) || isinf(num) || isinf(den) || isnan(num) || isnan(den))
    {
        return q > t;   // negative, zero, subnormal, infinite or NaN operands
    }

    // Away from the threshold the approximate quotient decides (its error is a few ulps at most).
    if (abs(q - t) > t * 1e-6)
    {
        return q > t;
    }

    uint mn, md, mt;
    int en, ed, et;
    nms_split(num, mn, en);
    nms_split(den, md, ed);
    nms_split(t, mt, et);
    // fl(q) > t  <=>  q > t + ulp(t) / 2, or q is that midpoint and rounding to even goes up (t's significand is odd).
    // q against (2 mt + 1) 2^(et - 1):  mn 2^(en - ed - et + 1)  against  (2 mt + 1) md  (the right side is below 2^49).
    int d = en - ed - et + 1;
    if (d < 0)
    {
        return false;
    }

    if (d >= 26)
    {
        return true;
    }

    uint64_t left = uint64_t(mn) << uint(d);
    uint64_t right = uint64_t(2u * mt + 1u) * uint64_t(md);
    return left > right || (left == right && (mt & 1u) != 0u);
}

// The CPU provider's IoU test: whether kept box a suppresses candidate b (corner form). `precise` keeps the products and sums
// unfused, rounded as on the CPU.
bool nms_suppresses(float4 a, float4 b)
{
    precise float width = min(a.w, b.w) - max(a.y, b.y);
    precise float height = min(a.z, b.z) - max(a.x, b.x);
    if (width <= 0.0 || height <= 0.0)
    {
        return false;
    }

    precise float intersection = width * height;
    precise float areaA = (a.w - a.y) * (a.z - a.x);
    precise float areaB = (b.w - b.y) * (b.z - b.x);
    precise float sum = areaA + areaB;
    precise float unionArea = sum - intersection;
    if (areaA <= 0.0 || areaB <= 0.0 || unionArea <= 0.0)
    {
        return false;
    }

    return nms_quotient_above(intersection, unionArea, IOU);
}

float nms_key(bool local, float* keys, uint i) { return local ? sKey[i] : keys[i]; }
int nms_id(bool local, int* ids, uint i) { return local ? sIdx[i] : ids[i]; }

[shader("compute")]
[numthreads(WG, 1, 1)]
void nms_class(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform NmsParams p)
{
    uint g = gid.y * 65535u + gid.x;
    if (g >= p.groups)
    {
        return;
    }

    uint tid = ltid.x;
    uint batch = g / C;
    float* scores = p.scores + uint64_t(g) * p.S;
    float* boxes = p.boxes + uint64_t(batch) * p.S * 4u;
    float* keys = p.keys + uint64_t(g) * CAP;
    int* ids = p.ids + uint64_t(g) * CAP;
    int* keptIds = p.keptIds + uint64_t(g) * KCAP;
    float4* keptBoxes = p.keptBoxes + uint64_t(g) * KCAP;

    // 1. Candidates above the score threshold (NaN never is), in any order: the sort orders them.
    if (tid == 0)
    {
        sCount = 0u;
        sKept = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = tid; j < p.S; j += WG)
    {
        float s = scores[j];
        if (s > SCORE)
        {
            uint slot;
            InterlockedAdd(sCount, 1u, slot);
            keys[slot] = s;
            ids[slot] = int(j);
        }
    }
    AllMemoryBarrierWithGroupSync();
    uint n = sCount;
    uint slots = 2u;
    while (slots < n)
    {
        slots <<= 1;
    }

    // 2. Bitonic sort of the padded candidates: score descending, then index ascending; the padding (-inf) sorts last.
    bool local = slots <= LOCAL;
    for (uint i = tid; i < slots; i += WG)
    {
        float key = i < n ? keys[i] : -1.0 / 0.0;
        int id = i < n ? ids[i] : 0x7FFFFFFF;
        if (local)
        {
            sKey[i] = key;
            sIdx[i] = id;
        }
        else if (i >= n)
        {
            keys[i] = key;
            ids[i] = id;
        }
    }
    AllMemoryBarrierWithGroupSync();
    for (uint size = 2u; size <= slots; size <<= 1)
    {
        for (uint stride = size >> 1; stride > 0u; stride >>= 1)
        {
            for (uint i = tid; i < slots; i += WG)
            {
                uint partner = i ^ stride;
                if (partner > i)
                {
                    float ka = nms_key(local, keys, i);
                    float kb = nms_key(local, keys, partner);
                    int ia = nms_id(local, ids, i);
                    int ib = nms_id(local, ids, partner);
                    bool partnerBetter = kb > ka || (kb == ka && ib < ia);
                    bool bestFirst = (i & size) == 0u;
                    if (partnerBetter == bestFirst)
                    {
                        if (local)
                        {
                            sKey[i] = kb;
                            sKey[partner] = ka;
                            sIdx[i] = ib;
                            sIdx[partner] = ia;
                        }
                        else
                        {
                            keys[i] = kb;
                            keys[partner] = ka;
                            ids[i] = ib;
                            ids[partner] = ia;
                        }
                    }
                }
            }
            AllMemoryBarrierWithGroupSync();
        }
    }

    // 3. Greedy suppression, WG candidates at a time: each candidate is tested in parallel against the boxes kept by earlier
    //    blocks and against the earlier candidates of its block (a bit mask); one thread then walks the block in order.
    for (uint start = 0u; start < n; start += WG)
    {
        uint keptBefore = sKept;
        if (keptBefore >= MAXK)
        {
            break;
        }

        uint i = start + tid;
        bool dead = i >= n;
        int id = dead ? 0 : nms_id(local, ids, i);
        float4 box = dead ? float4(0.0) : nms_corners(boxes + uint64_t(id) * 4u);
        for (uint k = 0u; k < keptBefore && !dead; k++)
        {
            dead = nms_suppresses(keptBoxes[k], box);
        }

        sBox[tid] = box;
        sBoxId[tid] = id;
        sDead[tid] = dead ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint w = 0u; w < MASK_WORDS; w++)
        {
            uint bits = 0u;
            uint first = w * 32u;
            for (uint s = first; s < min(first + 32u, tid) && !dead; s++)
            {
                if (sDead[s] == 0u && nms_suppresses(sBox[s], box))
                {
                    bits |= 1u << (s - first);
                }
            }
            sMask[tid * MASK_WORDS + w] = bits;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0u)
        {
            uint keptBits[MASK_WORDS];
            for (uint w = 0u; w < MASK_WORDS; w++)
            {
                keptBits[w] = 0u;
            }

            uint kept = keptBefore;
            for (uint t = 0u; t < WG && kept < MAXK; t++)
            {
                if (sDead[t] != 0u)
                {
                    continue;
                }

                bool suppressed = false;
                for (uint w = 0u; w < MASK_WORDS; w++)
                {
                    suppressed = suppressed || (sMask[t * MASK_WORDS + w] & keptBits[w]) != 0u;
                }

                if (!suppressed)
                {
                    keptBits[t / 32u] |= 1u << (t % 32u);
                    keptIds[kept] = sBoxId[t];
                    keptBoxes[kept] = sBox[t];
                    kept++;
                }
            }
            sKept = kept;
        }
        AllMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        p.keptCount[g] = int(sKept);
    }
}
#endif

#if defined(KERNEL_NMS_GATHER)
// Macros: C (classes), KCAP (kept-box slots per class).
struct NmsGatherParams { int* keptIds; int* keptCount; int* offsets; int* y; int* count; uint groups; };

[shader("compute")]
[numthreads(WG, 1, 1)]
void nms_gather(uint3 ltid : SV_GroupThreadID, uniform NmsGatherParams p)
{
    uint tid = ltid.x;
    if (tid == 0u)
    {
        int total = 0;
        for (uint g = 0u; g < p.groups; g++)
        {
            p.offsets[g] = total;
            total += p.keptCount[g];
        }
        p.count[0] = total;
    }
    AllMemoryBarrierWithGroupSync();
    for (uint g = 0u; g < p.groups; g++)
    {
        uint kept = uint(p.keptCount[g]);
        uint64_t row = uint64_t(p.offsets[g]);
        for (uint r = tid; r < kept; r += WG)
        {
            int* y = p.y + (row + r) * 3u;
            y[0] = int(g / C);
            y[1] = int(g % C);
            y[2] = p.keptIds[uint64_t(g) * KCAP + r];
        }
    }
}
#endif

#if defined(KERNEL_ROI_ALIGN)
// Macros: C, H, W (feature map), XS_N, XS_C, XS_H, XS_W (its strides), OH, OW (bins), RATIO (sampling ratio, 0: adaptive),
// SCALE (spatial scale), MODE_MAX, HALF_PIXEL.
struct RoiAlignParams { float* x; float* rois; int* batch; float* y; uint total; };

// Bilinear sample at (sy, sx) as the CPU provider computes it: corner offsets and weights, all zero outside [-1, size].
float roi_sample(float* image, float sy, float sx, bool takeMax)
{
    if (sy < -1.0 || sy > float(H) || sx < -1.0 || sx > float(W))
    {
        return 0.0;
    }

    sy = max(sy, 0.0);
    sx = max(sx, 0.0);
    int yLow = int(sy);
    int xLow = int(sx);
    int yHigh;
    int xHigh;
    if (yLow >= H - 1)
    {
        yHigh = yLow = H - 1;
        sy = float(yLow);
    }
    else
    {
        yHigh = yLow + 1;
    }

    if (xLow >= W - 1)
    {
        xHigh = xLow = W - 1;
        sx = float(xLow);
    }
    else
    {
        xHigh = xLow + 1;
    }

    precise float ly = sy - float(yLow);
    precise float lx = sx - float(xLow);
    precise float hy = 1.0 - ly;
    precise float hx = 1.0 - lx;
    precise float v0 = (hy * hx) * image[uint64_t(yLow) * XS_H + uint64_t(xLow) * XS_W];
    precise float v1 = (hy * lx) * image[uint64_t(yLow) * XS_H + uint64_t(xHigh) * XS_W];
    precise float v2 = (ly * hx) * image[uint64_t(yHigh) * XS_H + uint64_t(xLow) * XS_W];
    precise float v3 = (ly * lx) * image[uint64_t(yHigh) * XS_H + uint64_t(xHigh) * XS_W];
    if (takeMax)
    {
        return max(max(v0, v1), max(v2, v3));
    }

    precise float s01 = v0 + v1;
    precise float s012 = s01 + v2;
    precise float sum = s012 + v3;
    return sum;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void roi_align(uint3 dtid : SV_DispatchThreadID, uniform RoiAlignParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total)
    {
        return;
    }

    uint pw = idx % OW;
    uint t = idx / OW;
    uint ph = t % OH;
    t /= OH;
    uint c = t % C;
    uint r = t / C;
    float* box = p.rois + uint64_t(r) * 4u;
    float offset = HALF_PIXEL ? 0.5 : 0.0;
    precise float x0 = box[0] * SCALE - offset;
    precise float y0 = box[1] * SCALE - offset;
    precise float x1 = box[2] * SCALE - offset;
    precise float y1 = box[3] * SCALE - offset;
    precise float roiW = x1 - x0;
    precise float roiH = y1 - y0;
    if (!HALF_PIXEL)
    {
        roiW = max(roiW, 1.0);
        roiH = max(roiH, 1.0);
    }

    precise float binH = roiH / float(OH);
    precise float binW = roiW / float(OW);
    int gridH = RATIO > 0 ? RATIO : int(ceil(roiH / float(OH)));
    int gridW = RATIO > 0 ? RATIO : int(ceil(roiW / float(OW)));
    int count = max(gridH * gridW, 1);
    float* image = p.x + uint64_t(p.batch[r]) * XS_N + uint64_t(c) * XS_C;
    precise float sum = 0.0;
    float best = -1.0 / 0.0;
    for (int iy = 0; iy < gridH; iy++)
    {
        precise float sy = y0 + float(ph) * binH + (float(iy) + 0.5) * binH / float(gridH);
        for (int ix = 0; ix < gridW; ix++)
        {
            precise float sx = x0 + float(pw) * binW + (float(ix) + 0.5) * binW / float(gridW);
            float v = roi_sample(image, sy, sx, MODE_MAX != 0);
            if (MODE_MAX != 0)
            {
                best = max(best, v);
            }
            else
            {
                sum += v;
            }
        }
    }

    p.y[idx] = MODE_MAX == 0 ? sum / float(count) : gridH * gridW == 0 ? 0.0 : best;
}
#endif

#if defined(KERNEL_POOL_ARGMAX)
// Macros: C, H, W, XS_N, XS_C, XS_H, XS_W, HO, WO, KH, KW, SH, SW, PH, PW (leading pads), DH, DW, COLUMN_MAJOR (storage_order 1).
struct PoolArgMaxParams { float* x; int* y; uint total; };

[shader("compute")]
[numthreads(256, 1, 1)]
void pool_argmax(uint3 dtid : SV_DispatchThreadID, uniform PoolArgMaxParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total)
    {
        return;
    }

    uint ow = idx % WO;
    uint t = idx / WO;
    uint oh = t % HO;
    uint plane = t / HO;
    float* xp = p.x + uint64_t(plane / C) * XS_N + uint64_t(plane % C) * XS_C;
    int hs = int(oh) * SH - PH;
    int ws = int(ow) * SW - PW;
    float best = 0.0;
    int bh = -1;
    int bw = -1;
    for (int kh = 0; kh < KH; kh++)
    {
        int ih = hs + kh * DH;
        if (uint(ih) >= uint(H))
        {
            continue;
        }

        for (int kw = 0; kw < KW; kw++)
        {
            int iw = ws + kw * DW;
            if (uint(iw) >= uint(W))
            {
                continue;
            }

            float v = xp[uint64_t(ih) * XS_H + uint64_t(iw) * XS_W];
            if (bh < 0 || v > best)
            {
                best = v;
                bh = ih;
                bw = iw;
            }
        }
    }

    int local = bh < 0 ? 0 : COLUMN_MAJOR != 0 ? bh + bw * H : bh * W + bw;
    p.y[idx] = int(plane) * (H * W) + local;
}
#endif
