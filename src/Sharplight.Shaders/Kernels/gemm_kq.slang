// K-quant decode GEMM on cooperative matrices for targets without CUDA C++ (Vulkan; DeviceFeatures.MatrixMmaLayout): the counterpart
// of gemm_kq_native.cu. y[t, n] = sum_k x[t, k] * W[n, k] for batches of at most KQ_BM token rows per token tile (decode steps of
// 9-64 sequences), with f16 activations (the generated GEMMs' f16 copy of x, dense rows of K halves) and Q4_K / Q5_K / Q6_K weights in
// the GGUF block layout (KQ_FMT 4 / 5 / 6), f32 accumulation and output.
//
// A workgroup covers KQ_BM (16, 32, 48 or 64) token rows and 64 weight rows with 8 subgroups: subgroup s owns the 16 weight rows of
// fragment s % 4 and the K half s / 4 of every 256-value superblock, for all token rows, so each weight value is dequantized once and
// each staged activation is read by four subgroups. A K step is one superblock: the step's activation rows (KQ_BM x 512 bytes, 16-byte
// chunks XOR-swizzled by row parity) and the raw blocks of the 64 weight rows (each row's 16-byte-aligned window around its block) are
// staged in shared memory from registers that were loaded during the previous step's products (KQ_PF register sets: that many steps
// ahead), so the loads of the next superblock are in flight while the current one is computed. KQ_KH 1: 4 subgroups, each over both K
// halves. KQ_NT 2: 128 weight rows, 8 subgroups of one fragment each over both K halves, the activations staged half a superblock at a
// time (half steps; the weight windows at the first half of each superblock), so a staged activation serves 128 weight rows.
//
// The fragments are filled from registers: on devices whose 16x16 f16 fragments have the element layout of the 16x8x16 MMA instructions
// (element e of lane (g, c): row (A) / column (B) g + 8 (e >> 2), K slot 2c + (e & 1) + 8 ((e >> 1) & 1)), a lane may hold any K
// values as long as A and B agree, so lane (g, c) takes, of quarter u (32 values) of its K half, the 8 values 8c .. 8c + 7: one 16-byte
// read of each activation row and one 8-byte read of the weight row's codes (Q4_K / Q5_K: 8 code bytes of one nibble plane; Q6_K: 8
// low-nibble bytes and 8 high-bit bytes) serve two MMAs. Each code becomes an exact f16 integer (the 0x6400 bias), scaled with one f16
// FMA by the sub-block scale and minimum, the arithmetic of gemm_cm2's decode. The kernel verifies the layout once and stores NaN where
// it differs, so a wrong detection fails loudly.
//
// The two K halves' accumulators are summed through shared memory at the end. KQ_SPLIT > 1 splits the superblocks between the
// workgroups of grid z: each writes its partial sums to slice z of part ([KQ_SPLIT][T][KQ_PN]), publishes them and counts its arrival in
// arrivals[tile]; the last workgroup of a tile sums the slices in slice order (z = 0 first, from zero), stores the outputs and resets
// the counter (every launch leaves the counters zero). Workgroups never wait for each other.
//
// MULTI (2 or 3): the multi-output form over the concatenated rows of two or three matrices of one format and K (q / k / v), each with
// its own output (dense rows of its width) and, with H_BIAS, its own bias: rows [0, MULTI_O1) are w's into y, [MULTI_O1, MULTI_O2)
// w_1's into y_1, the rest w_2's into y_2 (MULTI_O1, MULTI_O2 multiples of 64). KQ_DUAL: a gate / up pair of one format feeding one
// elementwise group: the tile's even rows are w's rows n0 / 2 + r / 2, the odd ones w_1's, so the two columns of an accumulator pair
// are both products of one output, which the store combines with epi2(gate, up) (defined by the including generated source).
// H_BIAS: the store adds bias[n]. H_XRS: x is a row-ranged f16 copy whose scale words start at byte H_XRS of x (common.slang,
// f16_range_scale); the store multiplies each row's sums by the inverse of its scale (before the bias).
//
// Specialization macros: K (a multiple of 256), N (all rows with MULTI; outputs with KQ_DUAL), Y_STRIDE, KQ_FMT, MULTI, MULTI_O1,
// MULTI_O2, KQ_DUAL, KQ_BM, KQ_SPLIT, H_BIAS, H_XRS. Requires 16-byte aligned x and weights; weight rows may start at any even byte (Q6_K
// rows of 210-byte blocks). Grid: x = token tiles, y = weight tiles (64 rows), z = K splits. 32-wide subgroups.
#include "common.slang"
using namespace linalg;

#ifndef H_BIAS
#define H_BIAS 0
#endif
#ifndef MULTI
#define MULTI 0
#endif
#ifndef KQ_DUAL
#define KQ_DUAL 0
#endif
#if KQ_DUAL && (MULTI || H_BIAS)
#error "gemm_kq: the dual form has neither members nor biases"
#endif
#ifndef KQ_SPLIT
#define KQ_SPLIT 1
#endif
// Harness-only variants (SHARPLIGHT_COOP_KQ_MACROS) that skip one part of the work: KQ_X_NOCOPY (stage only the first superblock),
// KQ_X_NODQ (constant weight fragments), KQ_X_NOA (constant activation fragments). Their results are wrong.
#ifndef KQ_X_NOCOPY
#define KQ_X_NOCOPY 0
#endif
// KQ_PF: superblocks whose loads are in flight in registers during the products (1 or 2).
#ifndef KQ_PF
#define KQ_PF 1
#endif
#ifndef KQ_X_NOSTS
#define KQ_X_NOSTS 0
#endif
#ifndef KQ_X_EARLY
#define KQ_X_EARLY 0
#endif
#ifndef KQ_X_NOSTAGE
#define KQ_X_NOSTAGE 0
#endif
#ifndef KQ_X_NOXCOPY
#define KQ_X_NOXCOPY 0
#endif
#ifndef KQ_X_NOWCOPY
#define KQ_X_NOWCOPY 0
#endif
#ifndef KQ_X_NODQ
#define KQ_X_NODQ 0
#endif
#ifndef KQ_X_NOA
#define KQ_X_NOA 0
#endif
#ifndef KQ_BM
#define KQ_BM 64
#endif
#if KQ_FMT == 4
#define KQ_BB 144
#elif KQ_FMT == 5
#define KQ_BB 176
#elif KQ_FMT == 6
#define KQ_BB 210
#else
#error "gemm_kq: KQ_FMT 4, 5 or 6"
#endif
// KQ_NT: 64-row weight tiles per workgroup. 2: 128 weight rows, one 16-row fragment per subgroup, each over the whole superblock, and the
// activations staged half a superblock (128 values) at a time, so each staged activation serves twice the weight rows (half the
// activation bytes per weight byte) in the same shared memory.
#ifndef KQ_NT
#define KQ_NT 1
#endif
#if KQ_NT != 1 && (KQ_NT != 2 || MULTI)
#error "gemm_kq: KQ_NT 1, or 2 without MULTI"
#endif
#define KQ_BN (64 * KQ_NT)
#if KQ_NT == 2
#define KQ_XHALF 1
#define KQ_XCH 16                                           // 16-byte activation chunks per staged row
#else
#define KQ_XHALF 0
#define KQ_XCH 32
#endif
// KQ_KH: subgroups per weight fragment, each taking its share of every superblock's K halves (2: 8 subgroups, the halves' sums
// added through shared memory at the end; 1: 4 subgroups, each over the whole superblock, so more workgroups fit an SM). KQ_NT 2 takes 1.
#ifndef KQ_KH
#if KQ_NT == 2
#define KQ_KH 1
#else
#define KQ_KH 2
#endif
#endif
#if KQ_NT == 2 && KQ_KH != 1
#error "gemm_kq: KQ_NT 2 takes KQ_KH 1"
#endif
#define KQ_NSG (KQ_BN / 16 * KQ_KH)                          // subgroups
#define KQ_THREADS (KQ_NSG * 32)
// K halves of a superblock each kq_compute call covers.
#if KQ_KH == 2 || KQ_XHALF
#define KQ_KHN 1
#else
#define KQ_KHN 2
#endif
#define KQ_FM (KQ_BM / 16)
#define KQ_SB (K / 256)
#define KQ_ROWB (uint64_t(KQ_SB) * KQ_BB)
// 16-byte chunks of one weight row's window: the block itself when every block starts 16-byte aligned, else the block plus up to 14
// bytes before it; and the row stride in shared memory (KQ_WSC chunks: 4 rows of 8-byte reads at the same offset fall in distinct
// bank groups, a stride of 32 or 96 bytes modulo 128).
#if KQ_BB % 16 == 0 && (KQ_SB * KQ_BB) % 16 == 0
#define KQ_WCH (KQ_BB / 16)
#define KQ_ALIGNED 1
#else
#define KQ_WCH ((KQ_BB + 14 + 15) / 16)
#define KQ_ALIGNED 0
#endif
#if KQ_WCH % 8 == 2 || KQ_WCH % 8 == 6
#define KQ_WSC KQ_WCH
#elif KQ_WCH % 8 == 1 || KQ_WCH % 8 == 5
#define KQ_WSC (KQ_WCH + 1)
#elif KQ_WCH % 8 == 3 || KQ_WCH % 8 == 4
#define KQ_WSC (KQ_WCH + 6 - KQ_WCH % 8)
#else
#define KQ_WSC (KQ_WCH + 10 - KQ_WCH % 8)
#endif
#define KQ_XPT (KQ_BM * KQ_XCH / KQ_THREADS)                   // activation chunks per thread and step
#define KQ_WPT ((KQ_BN * KQ_WCH + KQ_THREADS - 1) / KQ_THREADS)  // weight chunks per thread and step
#if KQ_DUAL
#define KQ_PN (2 * N)
#else
#define KQ_PN N
#endif

typealias KqA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias KqB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias KqC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

struct Params
{
    uint8_t* x;
    uint8_t* w;
    float* y;
#if H_BIAS
    float* bias;
#endif
#if MULTI || KQ_DUAL
    uint8_t* w_1;
#endif
#if MULTI
#if MULTI > 2
    uint8_t* w_2;
#endif
    float* y_1;
#if MULTI > 2
    float* y_2;
#endif
#if H_BIAS
    float* bias_1;
#if MULTI > 2
    float* bias_2;
#endif
#endif
#endif
    float* part;      // KQ_SPLIT > 1: the partial slices
    uint* arrivals;   // KQ_SPLIT > 1: one counter per output tile
    uint T;
};

// The step's activation rows (chunk ch of row r at r * 32 + (ch ^ 4 (r & 1))); after the K loop, the K halves' partial tiles.
groupshared uint4 kqX[KQ_BM * KQ_XCH];
// The step's weight windows as 8-byte words (row r at r * 2 KQ_WSC).
groupshared uint2 kqW[KQ_BN * KQ_WSC * 2];
groupshared uint kqLast;

// The matrix a workgroup's weight tile lies in (MULTI: its member's), with that member's output, bias, row count and output row stride.
struct KqMember
{
    uint8_t* w;
    float* y;
#if H_BIAS
    float* bias;
#endif
    uint rows;
    uint stride;
};

// The member of weight row n0 (a tile's first), and n0 made relative to it.
KqMember kq_member(Params p, inout uint n0)
{
    KqMember m;
    m.w = p.w;
    m.y = p.y;
#if H_BIAS
    m.bias = p.bias;
#endif
    m.rows = N;
    m.stride = Y_STRIDE;
#if MULTI
    m.rows = MULTI_O1;
    if (n0 >= MULTI_O1)
    {
#if MULTI > 2
        bool third = n0 >= MULTI_O2;
        m.w = third ? p.w_2 : p.w_1;
        m.y = third ? p.y_2 : p.y_1;
#if H_BIAS
        m.bias = third ? p.bias_2 : p.bias_1;
#endif
        m.rows = third ? N - MULTI_O2 : MULTI_O2 - MULTI_O1;
        n0 -= third ? MULTI_O2 : MULTI_O1;
#else
        m.w = p.w_1;
        m.y = p.y_1;
#if H_BIAS
        m.bias = p.bias_1;
#endif
        m.rows = N - MULTI_O1;
        n0 -= MULTI_O1;
#endif
    }
    m.stride = m.rows;
#endif
    return m;
}

// The address of tile row r's weight row (rows past the matrix clamped to its last) and the end of that row's matrix. The dual form
// interleaves its matrices: even tile rows are w's rows n0 / 2 + r / 2, odd ones w_1's.
uint64_t kq_row(Params p, KqMember m, uint n0, uint r, out uint64_t end)
{
#if KQ_DUAL
    uint64_t w = (r & 1) != 0 ? uint64_t(p.w_1) : uint64_t(p.w);
    end = w + uint64_t(N) * KQ_ROWB;
    return w + uint64_t(min(n0 / 2 + (r >> 1), uint(N) - 1)) * KQ_ROWB;
#else
    end = uint64_t(m.w) + uint64_t(m.rows) * KQ_ROWB;
    return uint64_t(m.w) + uint64_t(min(n0 + r, m.rows - 1)) * KQ_ROWB;
#endif
}

// 16 bytes at a 16-byte aligned address of which only the bytes before `end` (2-byte aligned) may be read; the others read as zero.
uint4 kq_load16(uint64_t a, uint64_t end)
{
    if (a + 16 <= end)
    {
        return *(uint4*)a;
    }

    uint4 v = uint4(0, 0, 0, 0);
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
        if (a + 2 * i + 2 <= end)
        {
            uint h = uint(*(uint16_t*)(a + 2 * i));
            v[i >> 1] |= h << (16 * (i & 1));
        }
    }
    return v;
}

// The 8 bytes at byte offset o (even) of the staged weight words.
uint2 kq_w8(uint o)
{
#if KQ_ALIGNED
    return kqW[o >> 3];
#else
    uint2 a = kqW[o >> 3];
    uint s = 8 * (o & 7);
    if (s == 0)
    {
        return a;
    }

    uint2 b = kqW[(o >> 3) + 1];
    uint64_t lo = uint64_t(a.x) | (uint64_t(a.y) << 32), hi = uint64_t(b.x) | (uint64_t(b.y) << 32);
    uint64_t r = (lo >> s) | (hi << (64 - s));
    return uint2(uint(r), uint(r >> 32));
#endif
}

// The 16-bit word at byte offset o (even) of the staged weight words.
uint kq_w2(uint o)
{
    uint2 a = kqW[o >> 3];
    uint v = (o & 4) != 0 ? a.y : a.x;
    return (v >> (8 * (o & 2))) & 0xFFFFu;
}

// Two byte codes (bytes 0, 1 of v) as an f16 pair of exact integers 1024 + code.
uint kq_spread_lo(uint v)
{
    return (v & 0xFFu) | ((v & 0xFF00u) << 8) | 0x64006400u;
}

// Bytes 2, 3 of v likewise.
uint kq_spread_hi(uint v)
{
    return ((v >> 16) & 0xFFu) | ((v >> 8) & 0xFF0000u) | 0x64006400u;
}

// True when the device's 16x16 f16 fragments have the element layout this kernel assumes (uniform over the workgroup). Loads both
// fragments from a matrix whose element (r, k) / (k, n) holds 16 r + k / 16 n + k (staged in kqX, which the K loop overwrites after).
bool kq_layout(uint tid, uint lane)
{
    if (tid < 32)
    {
        uint4 v;
        [ForceUnroll]
        for (uint i = 0; i < 4; i++)
        {
            uint e = 8 * tid + 2 * i;
            v[i] = f32tof16(float(e)) | (f32tof16(float(e + 1)) << 16);
        }
        kqX[tid] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    KqA pa = KqA.Load<CoopMatMatrixLayout.RowMajor>(kqX, 0, 2);
    KqB pb = KqB.Load<CoopMatMatrixLayout.ColumnMajor>(kqX, 0, 2);
    uint g = lane / 4, c = lane % 4;
    bool match = true;
    [ForceUnroll]
    for (uint e = 0; e < 8; e++)
    {
        uint expected = (g + 8 * (e >> 2)) * 16 + 2 * c + (e & 1) + 8 * ((e >> 1) & 1);
        match = match && uint(float(pa[e])) == expected && uint(float(pb[e])) == expected;
    }
    match = WaveActiveAllTrue(match);
    GroupMemoryBarrierWithGroupSync();
    return match;
}

// The workgroup barrier of the K loop. KQ_X_BAR (experiments): 1 an execution barrier without memory semantics, 2 an execution barrier
// between a workgroup-memory release and acquire.
#ifndef KQ_X_BAR
#define KQ_X_BAR 0
#endif
void kq_sync()
{
#if KQ_X_BAR == 1
    spirv_asm { OpControlBarrier $(2u) $(2u) $(0u) };
#elif KQ_X_BAR == 2
    spirv_asm { OpMemoryBarrier $(2u) $(0x104u) };
    spirv_asm { OpControlBarrier $(2u) $(2u) $(0u) };
    spirv_asm { OpMemoryBarrier $(2u) $(0x102u) };
#else
    GroupMemoryBarrierWithGroupSync();
#endif
}

// The products of superblock sb staged in kqX / kqW for the subgroup's K halves (khOwn with KQ_KH = 2, both with 1): lane (g, c) of weight
// fragment wf, whose weight rows' windows start at byte wBase[h] of kqW and whose blocks are at the addresses wAlign[h] + sb * KQ_BB.
void kq_compute(uint sb, uint khFirst, uint g, uint c, uint wBase[2], uint64_t wAlign[2], inout KqC acc[KQ_FM])
{
    // The K halves of the superblock this call covers (KQ_KHN from khFirst).
    for (uint kq = 0; kq < KQ_KHN; kq++)
    {
    uint kh = khFirst + kq;
    // The lane's two weight rows' block starts in the staged windows.
    uint wb[2];
    [ForceUnroll]
    for (uint h = 0; h < 2; h++)
    {
        wb[h] = wBase[h] + uint((wAlign[h] + uint64_t(sb) * KQ_BB) & 15ull);
    }

#if KQ_FMT == 6
    // Q6_K, K half kh = 128-value half n of the block: quarter u is value group qd = u, codes from ql (nibble u >> 1 of the bytes
    // 64 kh + 32 (u & 1) + 8c ..) and qh (bits 2u of the bytes 128 + 32 kh + 8c ..), scale sc[8 kh + 2u + c / 2] * d, value
    // (code - 32) * scale.
    half2 dsc[2][4];
    uint2 qh[2];
    [ForceUnroll]
    for (uint h = 0; h < 2; h++)
    {
        float d = f16tof32(kq_w2(wb[h] + 208));
        uint2 scw = kq_w8(wb[h] + 192 + 8 * kh);
        [ForceUnroll]
        for (uint u = 0; u < 4; u++)
        {
            uint b = 2 * u + (c >> 1);
            int s = int(((b < 4 ? scw.x : scw.y) >> (8 * (b & 3))) << 24) >> 24;
            dsc[h][u] = half2(half(d * float(s)));
        }
        qh[h] = kq_w8(wb[h] + 128 + 32 * kh + 8 * c);
    }
    [ForceUnroll]
    for (uint uh = 0; uh < 2; uh++)
    {
        uint2 ql[2];
        [ForceUnroll] for (uint h = 0; h < 2; h++) ql[h] = kq_w8(wb[h] + 64 * kh + 32 * uh + 8 * c);
        [ForceUnroll]
        for (uint un = 0; un < 2; un++)
        {
            uint u = 2 * un + uh;
#else
    // Q4_K / Q5_K, K half kh = chunks 2 kh, 2 kh + 1 of 64 values: quarter u is nibble u & 1 of chunk 2 kh + (u >> 1), codes from the
    // bytes 32 j + 8c .. of qs (Q5_K: plus bit 4 kh + u of the bytes 8c .. of qh), sub-block 4 kh + u.
#if KQ_FMT == 5
    const uint qsOffset = 48;
#else
    const uint qsOffset = 16;
#endif
    half2 dsc[2][4], dmn[2][4];
#if KQ_FMT == 5
    uint2 qh[2];
#endif
    [ForceUnroll]
    for (uint h = 0; h < 2; h++)
    {
        uint2 h0 = kq_w8(wb[h]), h1 = kq_w8(wb[h] + 8);
        uint3 scales = uint3(h0.y, h1.x, h1.y);
        float d = f16tof32(h0.x & 0xFFFFu), dmin = f16tof32(h0.x >> 16);
        [ForceUnroll]
        for (uint u = 0; u < 4; u++)
        {
            float sc, mn;
            scale_min_k4_words(scales, 4 * kh + u, sc, mn);
            dsc[h][u] = half2(half(d * sc));
            dmn[h][u] = half2(half(-dmin * mn));
        }
#if KQ_FMT == 5
        qh[h] = kq_w8(wb[h] + 16 + 8 * c);
#endif
    }
    [ForceUnroll]
    for (uint uh = 0; uh < 2; uh++)
    {
        uint2 ql[2];
        [ForceUnroll] for (uint h = 0; h < 2; h++) ql[h] = kq_w8(wb[h] + qsOffset + 32 * (2 * kh + uh) + 8 * c);
        [ForceUnroll]
        for (uint un = 0; un < 2; un++)
        {
            uint u = 2 * uh + un;
#endif
            // The activations of quarter u: 8 values 128 kh + 32u + 8c .. of rows 16i + g + 8h.
            uint4 xa[KQ_FM][2];
            [ForceUnroll]
            for (uint i = 0; i < KQ_FM; i++)
            {
                [ForceUnroll]
                for (uint h = 0; h < 2; h++)
                {
                    uint r = 16 * i + g + 8 * h, ch = (KQ_XHALF ? 0 : 16 * kh) + 4 * u + c;
                    xa[i][h] = kqX[r * KQ_XCH + (ch ^ ((r & 1) << 2))];
                }
            }
            [ForceUnroll]
            for (uint q = 0; q < 2; q++)
            {
                // MMA q of the quarter: the lane's values 4q .. 4q + 3 (bytes 4q .. 4q + 3 of the codes, word q).
                KqB b;
                [ForceUnroll]
                for (uint h = 0; h < 2; h++)
                {
                    uint word = q == 0 ? ql[h].x : ql[h].y;
#if KQ_FMT == 6
                    uint hw = q == 0 ? qh[h].x : qh[h].y;
                    uint code = ((word >> (4 * (u >> 1))) & 0x0F0F0F0Fu) | (((hw >> (2 * u)) & 0x03030303u) << 4);
                    half2 lo = (bit_cast<half2>(kq_spread_lo(code)) - half2(1056.0h, 1056.0h)) * dsc[h][u];
                    half2 hi = (bit_cast<half2>(kq_spread_hi(code)) - half2(1056.0h, 1056.0h)) * dsc[h][u];
#else
                    uint code = (word >> (4 * (u & 1))) & 0x0F0F0F0Fu;
#if KQ_FMT == 5
                    uint hw = q == 0 ? qh[h].x : qh[h].y;
                    code |= ((hw >> (4 * kh + u)) & 0x01010101u) << 4;
#endif
                    half2 lo = (bit_cast<half2>(kq_spread_lo(code)) - half2(1024.0h, 1024.0h)) * dsc[h][u] + dmn[h][u];
                    half2 hi = (bit_cast<half2>(kq_spread_hi(code)) - half2(1024.0h, 1024.0h)) * dsc[h][u] + dmn[h][u];
#endif
#if KQ_X_NODQ
                    lo = half2(half(h), half(q));
                    hi = lo;
#endif
                    b[4 * h] = lo.x;
                    b[4 * h + 1] = lo.y;
                    b[4 * h + 2] = hi.x;
                    b[4 * h + 3] = hi.y;
                }
                [ForceUnroll]
                for (uint i = 0; i < KQ_FM; i++)
                {
                    KqA a;
                    [ForceUnroll]
                    for (uint h = 0; h < 2; h++)
                    {
#if KQ_X_NOA
                        uint w0 = 0x3C003C00u + i, w1 = 0x3C003C00u + h;
#else
                        uint w0 = xa[i][h][2 * q], w1 = xa[i][h][2 * q + 1];
#endif
                        half2 v0 = bit_cast<half2>(w0), v1 = bit_cast<half2>(w1);
                        a[4 * h] = v0.x;
                        a[4 * h + 1] = v0.y;
                        a[4 * h + 2] = v1.x;
                        a[4 * h + 3] = v1.y;
                    }
                    acc[i] = coopMatMulAdd<float, false>(a, b, acc[i]);
                }
            }
        }
    }
    }
}

[shader("compute")]
[numthreads(KQ_THREADS, 1, 1)]
void gemm_kq(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint tid = ltid.x, lane = tid % 32, sg = tid / 32;
    uint wf = sg % (KQ_BN / 16), khOwn = sg / (KQ_BN / 16);
    uint g = lane / 4, c = lane % 4;
    uint t0 = gid.x * KQ_BM, ng = gid.y * KQ_BN;
    uint n0 = ng;
    KqMember m = kq_member(p, n0);
    uint sb0 = gid.z * KQ_SB / KQ_SPLIT, sb1 = (gid.z + 1) * KQ_SB / KQ_SPLIT, steps = sb1 - sb0;
    bool layoutOk = kq_layout(tid, lane);

    // The loads of one superblock into registers (activation rows clamped to the batch; weight windows zero past their matrix).
    uint4 xr[KQ_PF][KQ_XPT];
    uint4 wr[KQ_PF][KQ_WPT];
    uint64_t xRow[KQ_XPT];
    [ForceUnroll]
    for (uint it = 0; it < KQ_XPT; it++)
    {
        uint idx = tid + it * KQ_THREADS, r = idx / KQ_XCH, ch = idx % KQ_XCH;
        xRow[it] = uint64_t(p.x) + uint64_t(min(t0 + r, p.T - 1)) * (uint64_t(K) * 2) + ch * 16;
    }
    // The weight chunks of each thread: the address of its row's first block (0: no chunk) and the chunk's offset in the window; for
    // windows that may end past their matrix (blocks not 16-byte aligned), the matrix's end.
    uint64_t wRow[KQ_WPT];
    uint wCh[KQ_WPT];
#if !KQ_ALIGNED
    uint64_t wEnd[KQ_WPT];
#endif
    [ForceUnroll]
    for (uint it = 0; it < KQ_WPT; it++)
    {
        uint idx = tid + it * KQ_THREADS;
        uint r = min(idx / KQ_WCH, uint(KQ_BN) - 1);
        uint64_t wend = 0;
        wRow[it] = (KQ_BN * KQ_WCH) % KQ_THREADS == 0 || idx < KQ_BN * KQ_WCH ? kq_row(p, m, n0, r, wend) : 0;
        wCh[it] = (idx % KQ_WCH) * 16;
#if !KQ_ALIGNED
        wEnd[it] = wend;
#endif
    }
    // The loads of half `half` of superblock sb's activations (both halves unless KQ_XHALF) and, with withW, of its weight windows.
    #define KQ_FETCH(set, sb, half, withW)                                                              \
    {                                                                                                   \
        if (!KQ_X_NOXCOPY || (sb) == sb0) { [ForceUnroll] for (uint it = 0; it < KQ_XPT; it++) xr[set][it] = *(uint4*)(xRow[it] + uint64_t(sb) * 512 + (half) * 256); } \
        [ForceUnroll]                                                                                   \
        for (uint it = 0; it < KQ_WPT; it++)                                                            \
        {                                                                                               \
            if ((withW) && (!KQ_X_NOWCOPY || (sb) == sb0))                                              \
            {                                                                                           \
                uint64_t src = ((wRow[it] + uint64_t(sb) * KQ_BB) & ~15ull) + wCh[it];                  \
                if (KQ_ALIGNED)                                                                         \
                {                                                                                       \
                    wr[set][it] = uint4(0, 0, 0, 0);                                                    \
                    if ((KQ_BN * KQ_WCH) % KQ_THREADS == 0 || wRow[it] != 0) wr[set][it] = *(uint4*)src; \
                }                                                                                       \
                else                                                                                    \
                {                                                                                       \
                    KQ_FETCH_TAIL(set, it, src)                                                         \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
    }
#if KQ_ALIGNED
    #define KQ_FETCH_TAIL(set, it, src)
#else
    #define KQ_FETCH_TAIL(set, it, src)                                                                 \
    {                                                                                                   \
        wr[set][it] = uint4(0, 0, 0, 0);                                                                \
        if (wRow[it] != 0 && src + 16 <= wEnd[it]) wr[set][it] = *(uint4*)src;                          \
        else if (wRow[it] != 0 && src < wEnd[it]) wr[set][it] = kq_load16(src, wEnd[it]);               \
    }
#endif
    uint kqDummy = 0;
#if KQ_X_NOSTS
    #define KQ_STAGE(set, withW)                                                                        \
    {                                                                                                   \
        [ForceUnroll] for (uint it = 0; it < KQ_XPT; it++) kqDummy ^= xr[set][it].x ^ xr[set][it].w;      \
        [ForceUnroll] for (uint it = 0; it < KQ_WPT; it++) kqDummy ^= wr[set][it].x ^ wr[set][it].w;      \
    }
#else
    #define KQ_STAGE(set, withW)                                                                        \
    {                                                                                                   \
        [ForceUnroll]                                                                                   \
        for (uint it = 0; it < KQ_XPT; it++)                                                            \
        {                                                                                               \
            uint idx = tid + it * KQ_THREADS, r = idx / KQ_XCH, ch = idx % KQ_XCH;                      \
            kqX[r * KQ_XCH + (ch ^ ((r & 1) << 2))] = xr[set][it];                                      \
        }                                                                                               \
        [ForceUnroll]                                                                                   \
        for (uint it = 0; it < KQ_WPT; it++)                                                            \
        {                                                                                               \
            uint idx = tid + it * KQ_THREADS;                                                           \
            if ((withW) && ((KQ_BN * KQ_WCH) % KQ_THREADS == 0 || idx < KQ_BN * KQ_WCH))                \
            {                                                                                           \
                uint r = idx / KQ_WCH, ch = idx % KQ_WCH, o = (r * KQ_WSC + ch) * 2;                    \
                kqW[o] = wr[set][it].xy;                                                                \
                kqW[o + 1] = wr[set][it].zw;                                                            \
            }                                                                                           \
        }                                                                                               \
    }
#endif

    // Byte offsets of the lane's two weight rows (g, g + 8 of fragment wf) in the staged windows, without the block's alignment offset.
    uint wBase[2];
    uint64_t wAlign[2];
    [ForceUnroll]
    for (uint h = 0; h < 2; h++)
    {
        uint r = 16 * wf + g + 8 * h;
        wBase[h] = r * KQ_WSC * 16;
        uint64_t wend;
        wAlign[h] = kq_row(p, m, n0, r, wend);
    }

    KqC acc[KQ_FM];
    [ForceUnroll] for (uint i = 0; i < KQ_FM; i++) acc[i] = KqC(0.0);

    // KQ_PF register sets hold the loads of the next KQ_PF superblocks: set (step % KQ_PF) is staged at step `step` and then reloaded with
    // superblock step + KQ_PF, so each load has KQ_PF steps of products to arrive.
#if KQ_XHALF
    // Half steps: half 0 of each superblock stages its activations' first half and the weight windows, half 1 the second half of the
    // activations; the loads of the next half step are in flight during this one's products.
    uint hsteps = 2 * steps;
    if (hsteps > 0)
    {
        KQ_FETCH(0, sb0, 0, true)
    }
    for (uint hs = 0; hs < hsteps; hs++)
    {
        uint sb = sb0 + hs / 2, half = hs & 1;
        kq_sync();
        KQ_STAGE(0, half == 0)
        kq_sync();
        if (hs + 1 < hsteps)
        {
            uint nh = (hs + 1) & 1;
            KQ_FETCH(0, sb0 + (hs + 1) / 2, nh, nh == 0)
        }
        kq_compute(sb, half, g, c, wBase, wAlign, acc);
    }
#else
    [ForceUnroll]
    for (uint s = 0; s < KQ_PF; s++)
    {
        if (s < steps)
        {
            KQ_FETCH(s, sb0 + s, 0, true)
        }
    }
    #define KQ_STEP(set)                                                                                \
    {                                                                                                   \
        uint sb = sb0 + step;                                                                           \
        if (!KQ_X_NOCOPY || step == 0)                                                                  \
        {                                                                                               \
            kq_sync();                                                                                  \
            if (!KQ_X_NOSTAGE || step == 0)                                                             \
            {                                                                                           \
                KQ_STAGE(set, true)                                                                     \
            }                                                                                           \
            if (KQ_X_EARLY && step + KQ_PF < steps && !KQ_X_NOCOPY && !KQ_X_NOSTAGE)                    \
            {                                                                                           \
                KQ_FETCH(set, sb + KQ_PF, 0, true)                                                      \
            }                                                                                           \
            kq_sync();                                                                                  \
            if (!KQ_X_EARLY && step + KQ_PF < steps && !KQ_X_NOCOPY && !KQ_X_NOSTAGE)                   \
            {                                                                                           \
                KQ_FETCH(set, sb + KQ_PF, 0, true)                                                      \
            }                                                                                           \
        }                                                                                               \
        kq_compute(sb, KQ_KH == 2 ? khOwn : 0, g, c, wBase, wAlign, acc);                               \
    }
    for (uint step = 0; step < steps;)
    {
        // Every subgroup is done with the previous superblock: stage this one, then load the one KQ_PF ahead into its registers.
        [ForceUnroll]
        for (uint s = 0; s < KQ_PF; s++)
        {
            if (step < steps)
            {
                KQ_STEP(s)
                step++;
            }
        }
    }
#endif
    #undef KQ_STEP
    #undef KQ_FETCH
    #undef KQ_STAGE

    // The K halves' sums: the second half's subgroups leave their accumulators in kqX (element e of lane l of fragment i of weight
    // fragment f at word ((f KQ_FM + i) 2 + e / 4) 32 + l), the first half's add them.
#if KQ_KH == 2
    GroupMemoryBarrierWithGroupSync();
    if (khOwn == 1)
    {
        [ForceUnroll]
        for (uint i = 0; i < KQ_FM; i++)
        {
            [ForceUnroll]
            for (uint q = 0; q < 2; q++)
            {
                kqX[((wf * KQ_FM + i) * 2 + q) * 32 + lane] = uint4(asuint(acc[i][4 * q]), asuint(acc[i][4 * q + 1]), asuint(acc[i][4 * q + 2]),
                    asuint(acc[i][4 * q + 3]));
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float v[KQ_FM][8];
    [ForceUnroll]
    for (uint i = 0; i < KQ_FM; i++)
    {
        [ForceUnroll]
        for (uint q = 0; q < 2; q++)
        {
            uint4 o = kqX[((wf * KQ_FM + i) * 2 + q) * 32 + lane];
            [ForceUnroll] for (uint e = 0; e < 4; e++) v[i][4 * q + e] = acc[i][4 * q + e] + asfloat(o[e]);
        }
    }
#else
    float v[KQ_FM][8];
    [ForceUnroll] for (uint i = 0; i < KQ_FM; i++) [ForceUnroll] for (uint e = 0; e < 8; e++) v[i][e] = acc[i][e];
#endif

    // Element e of fragment i: token row t0 + 16i + g + 8 (e >> 2), tile column 16 wf + 2c + 8 ((e >> 1) & 1) + (e & 1).
#if KQ_SPLIT > 1
    if (khOwn == 0)
    {
        [ForceUnroll]
        for (uint i = 0; i < KQ_FM; i++)
        {
            [ForceUnroll]
            for (uint e = 0; e < 8; e += 2)
            {
                uint t = t0 + 16 * i + g + 8 * (e >> 2);
                uint col = ng + 16 * wf + 2 * c + 8 * ((e >> 1) & 1);
                if (t < p.T)
                {
                    float* pr = p.part + (uint64_t(gid.z) * p.T + t) * KQ_PN;
                    if (col + 1 < KQ_PN)
                    {
                        *(float2*)(pr + col) = float2(v[i][e], v[i][e + 1]);
                    }
                    else if (col < KQ_PN)
                    {
                        pr[col] = v[i][e];
                    }
                }
            }
        }
    }
    DeviceMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    uint tile = gid.y * ((p.T + KQ_BM - 1) / KQ_BM) + gid.x;
    if (tid == 0)
    {
        uint arrived;
        // Indexed, not through a dereferenced pointer: Slang's SPIR-V of an atomic on `*ptr` does not reach the buffer.
        InterlockedAdd(p.arrivals[tile], 1u, arrived);
        bool last = arrived + 1 == KQ_SPLIT;
        if (last)
        {
            p.arrivals[tile] = 0;
        }
        kqLast = last ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (kqLast == 0)
    {
        return;
    }
    DeviceMemoryBarrier();
#endif
    if (khOwn != 0)
    {
        return;
    }

    [ForceUnroll]
    for (uint i = 0; i < KQ_FM; i++)
    {
        [ForceUnroll]
        for (uint e = 0; e < 8; e += 2)
        {
            uint t = t0 + 16 * i + g + 8 * (e >> 2);
            if (t >= p.T)
            {
                continue;
            }

            uint n = n0 + 16 * wf + 2 * c + 8 * ((e >> 1) & 1);
#if KQ_SPLIT > 1
            uint col = ng + 16 * wf + 2 * c + 8 * ((e >> 1) & 1);
            float v0 = 0.0, v1 = 0.0;
            for (uint z = 0; z < KQ_SPLIT; z++)
            {
                float* pr = p.part + (uint64_t(z) * p.T + t) * KQ_PN;
                v0 += pr[min(col, uint(KQ_PN) - 1)];
                v1 += pr[min(col + 1, uint(KQ_PN) - 1)];
            }
#else
            float v0 = v[i][e], v1 = v[i][e + 1];
#endif
#ifdef H_XRS
            float rowInverse = f16_range_inverse(*(uint*)(uint64_t(p.x) + H_XRS + uint64_t(t) * 4));
            v0 *= rowInverse;
            v1 *= rowInverse;
#endif
#if H_BIAS
            v0 += m.bias[min(n, m.rows - 1)];
            v1 += m.bias[min(n + 1, m.rows - 1)];
#endif
            if (!layoutOk || kqDummy == 0x9E3779B9u)
            {
                v0 = asfloat(0x7FC00000u);
                v1 = v0;
            }
            float* yr = m.y + uint64_t(t) * m.stride;
#if KQ_DUAL
            // v0 is w's row n / 2, v1 w_1's: the pair's activation product.
            if (n / 2 < uint(N))
            {
                yr[n / 2] = epi2(v0, v1);
            }
#else
            if (n + 1 < m.rows)
            {
                *(float2*)(yr + n) = float2(v0, v1);
            }
            else if (n < m.rows)
            {
                yr[n] = v0;
            }
#endif
        }
    }
}
