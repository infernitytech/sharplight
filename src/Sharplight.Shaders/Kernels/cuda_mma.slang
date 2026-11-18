// CUDA forms of the cooperative-matrix fragments (included by the tensor-core kernels when SL_TARGET_CUDA is set, i.e. only for
// the CUDA C++ target; SPIR-V and MSL keep Slang's CoopMat). The kernels name their fragment types through typealiases, which
// select these structs on CUDA, and call the same API: T.Load<layout>(array or pointer, element, stride), Store, coopMatMulAdd,
// construction from a scalar and element access.
//
// Every fragment holds its elements in registers in the layout of the 16x8x16 (f16, bf16) and 16x8x32 (int8, E4M3) mma.sync
// instructions, ordered like the NVIDIA Vulkan driver orders cooperative-matrix elements (DeviceFeatures.MatrixMmaLayout): lane
// l = 4 g + c holds element e of a 16x16 matrix (16-bit A, B and every accumulator) at row g + 8 (e >> 2), column
// 2 c + (e & 1) + 8 ((e >> 1) & 1) (B: column g + 8 (e >> 2), row 2 c + ...). Fragments of 16-bit and 8-bit operands load from
// shared memory with ldmatrix (one x4 per fragment, .trans for the transposed layouts), from device memory element by element.
//
// Register order: A (16x16 f16 / bf16, 16x32 int8 / e4m3): r = (a0, a1, a2, a3) of mma.sync (rows 0-7 / 8-15 of K block 0, then
// of K block 1). B (16x16, 32x16): r = (b0, b1) of the columns 0-7, then (b0, b1) of the columns 8-15. Accumulators: t0 = the
// c0..c3 of the columns 0-7, t1 = those of the columns 8-15.
#pragma once
using namespace linalg;

#define CU_MMA_PRELUDE R"(
#ifndef SL_MMA_PRELUDE
#define SL_MMA_PRELUDE
__device__ __forceinline__ unsigned sl_lane() { unsigned l; asm("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }
template<class T, size_t N> __device__ __forceinline__ const char* sl_bytes(const FixedArray<T, N>* a) { return (const char*)a->m_data; }
template<class T> __device__ __forceinline__ const char* sl_bytes(const T* p) { return (const char*)p; }
template<class T, size_t N> __device__ __forceinline__ char* sl_wbytes(FixedArray<T, N>* a) { return (char*)a->m_data; }
template<class T> __device__ __forceinline__ char* sl_wbytes(T* p) { return (char*)p; }
template<class T, size_t N> __device__ __forceinline__ constexpr unsigned sl_esize(const FixedArray<T, N>*) { return sizeof(T); }
template<class T> __device__ __forceinline__ constexpr unsigned sl_esize(const T*) { return sizeof(T); }
template<class T, size_t N> __device__ __forceinline__ constexpr bool sl_is_shared(const FixedArray<T, N>*) { return true; }
template<class T> __device__ __forceinline__ constexpr bool sl_is_shared(const T*) { return false; }

// A 16x16 matrix of 16-bit values (or 16x32 bytes) at `base` with row pitch `pitch` bytes as four 8x8 b16 matrices: matrix q
// covers stored rows 8 rb(q) .. and stored 16-bit columns 8 cb(q) .. with rb(q) = q & 1, cb(q) = q >> 1 (rowsLow) or
// rb(q) = q >> 1, cb(q) = q & 1. Without trans, lane l gets the 32-bit word (row l / 4, columns 2 (l % 4), +1) of each matrix;
// with trans, the values (rows 2 (l % 4), +1 of column l / 4) packed. ldmatrix for 16-byte aligned shared rows, else loads.
template<bool Trans, bool RowsLow, class P>
__device__ __forceinline__ uint4 sl_ld16(P* arr, unsigned element, unsigned stride)
{
    const unsigned es = sl_esize(arr);
    const char* base = sl_bytes(arr) + (size_t)element * es;
    const unsigned pitch = stride * es;
    const unsigned l = sl_lane();
    uint4 r;
    if (sl_is_shared(arr))
    {
        unsigned sa = (unsigned)__cvta_generic_to_shared(base);
        if (((sa | pitch) & 15) == 0)
        {
            unsigned q = l >> 3;
            unsigned rb = RowsLow ? (q & 1) : (q >> 1), cb = RowsLow ? (q >> 1) : (q & 1);
            unsigned addr = sa + (rb * 8 + (l & 7)) * pitch + cb * 16;
            if (Trans)
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr));
            else
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr));
            return r;
        }
    }
    unsigned v[4];
#pragma unroll
    for (unsigned q = 0; q < 4; q++)
    {
        unsigned rb = RowsLow ? (q & 1) : (q >> 1), cb = RowsLow ? (q >> 1) : (q & 1);
        if (Trans)
        {
            const char* p = base + (rb * 8 + 2 * (l & 3)) * pitch + (cb * 8 + (l >> 2)) * 2;
            v[q] = (unsigned)*(const unsigned short*)p | ((unsigned)*(const unsigned short*)(p + pitch) << 16);
        }
        else
        {
            const char* p = base + (rb * 8 + (l >> 2)) * pitch + cb * 16 + (l & 3) * 4;
            v[q] = (unsigned)*(const unsigned short*)p | ((unsigned)*(const unsigned short*)(p + 2) << 16);
        }
    }
    return make_uint4(v[0], v[1], v[2], v[3]);
}

// Accumulator tile `tile` (columns 8 tile ..) of a 16x16 matrix of 32-bit values: (row g, col 2c), (g, 2c + 1), (g + 8, 2c), (g + 8, 2c + 1).
template<class V, class P>
__device__ __forceinline__ V sl_acc_ld(P* arr, unsigned element, unsigned stride, unsigned tile, bool colMajor)
{
    const unsigned es = sl_esize(arr);
    const char* base = sl_bytes(arr) + (size_t)element * es;
    const unsigned pitch = stride * es;
    const unsigned l = sl_lane(), g = l >> 2, c = 8 * tile + 2 * (l & 3);
    V v;
    if (colMajor)
    {
        v.x = *(const decltype(v.x)*)(base + c * pitch + g * 4);
        v.y = *(const decltype(v.x)*)(base + (c + 1) * pitch + g * 4);
        v.z = *(const decltype(v.x)*)(base + c * pitch + (g + 8) * 4);
        v.w = *(const decltype(v.x)*)(base + (c + 1) * pitch + (g + 8) * 4);
    }
    else
    {
        v.x = *(const decltype(v.x)*)(base + g * pitch + c * 4);
        v.y = *(const decltype(v.x)*)(base + g * pitch + c * 4 + 4);
        v.z = *(const decltype(v.x)*)(base + (g + 8) * pitch + c * 4);
        v.w = *(const decltype(v.x)*)(base + (g + 8) * pitch + c * 4 + 4);
    }
    return v;
}

template<class V, class P>
__device__ __forceinline__ void sl_acc_st(P* arr, unsigned element, unsigned stride, unsigned tile, bool colMajor, V v)
{
    typedef decltype(v.x) E;
    const unsigned es = sl_esize(arr);
    char* base = sl_wbytes(arr) + (size_t)element * es;
    const unsigned pitch = stride * es;
    const unsigned l = sl_lane(), g = l >> 2, c = 8 * tile + 2 * (l & 3);
    if (colMajor)
    {
        *(E*)(base + c * pitch + g * 4) = v.x;
        *(E*)(base + (c + 1) * pitch + g * 4) = v.y;
        *(E*)(base + c * pitch + (g + 8) * 4) = v.z;
        *(E*)(base + (c + 1) * pitch + (g + 8) * 4) = v.w;
        return;
    }
    char* p0 = base + g * pitch + c * 4;
    char* p1 = base + (g + 8) * pitch + c * 4;
    if ((((size_t)p0 | pitch) & 7) == 0)
    {
        *(uint2*)p0 = make_uint2(*(unsigned*)&v.x, *(unsigned*)&v.y);
        *(uint2*)p1 = make_uint2(*(unsigned*)&v.z, *(unsigned*)&v.w);
    }
    else
    {
        *(E*)p0 = v.x;
        *(E*)(p0 + 4) = v.y;
        *(E*)p1 = v.z;
        *(E*)(p1 + 4) = v.w;
    }
}

// Accumulator tile of 16-bit values: v.x = (row g, cols 2c, 2c + 1) packed, v.y = (row g + 8, ...).
template<class P>
__device__ __forceinline__ uint2 sl_acch_ld(P* arr, unsigned element, unsigned stride, unsigned tile, bool colMajor)
{
    const unsigned es = sl_esize(arr);
    const char* base = sl_bytes(arr) + (size_t)element * es;
    const unsigned pitch = stride * es;
    const unsigned l = sl_lane(), g = l >> 2, c = 8 * tile + 2 * (l & 3);
    unsigned h[4];
#pragma unroll
    for (unsigned i = 0; i < 4; i++)
    {
        unsigned row = g + 8 * (i >> 1), col = c + (i & 1);
        h[i] = *(const unsigned short*)(base + (colMajor ? col * pitch + row * 2 : row * pitch + col * 2));
    }
    return make_uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
}

template<class P>
__device__ __forceinline__ void sl_acch_st(P* arr, unsigned element, unsigned stride, unsigned tile, bool colMajor, uint2 v)
{
    const unsigned es = sl_esize(arr);
    char* base = sl_wbytes(arr) + (size_t)element * es;
    const unsigned pitch = stride * es;
    const unsigned l = sl_lane(), g = l >> 2, c = 8 * tile + 2 * (l & 3);
    if (!colMajor && ((((size_t)base + c * 2) | pitch) & 3) == 0)
    {
        *(unsigned*)(base + g * pitch + c * 2) = v.x;
        *(unsigned*)(base + (g + 8) * pitch + c * 2) = v.y;
        return;
    }
    unsigned h[4] = { v.x & 0xFFFFu, v.x >> 16, v.y & 0xFFFFu, v.y >> 16 };
#pragma unroll
    for (unsigned i = 0; i < 4; i++)
    {
        unsigned row = g + 8 * (i >> 1), col = c + (i & 1);
        *(unsigned short*)(base + (colMajor ? col * pitch + row * 2 : row * pitch + col * 2)) = (unsigned short)h[i];
    }
}

__device__ __forceinline__ float4 sl_mma_f16(uint4 a, uint2 b, float4 c)
{
    float4 d;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d.x), "=f"(d.y), "=f"(d.z), "=f"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
    return d;
}

__device__ __forceinline__ float4 sl_mma_bf16(uint4 a, uint2 b, float4 c)
{
    float4 d;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d.x), "=f"(d.y), "=f"(d.z), "=f"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
    return d;
}

__device__ __forceinline__ int4 sl_mma_s8(uint4 a, uint2 b, int4 c)
{
    int4 d;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
    return d;
}

__device__ __forceinline__ float4 sl_mma_e4m3(uint4 a, uint2 b, float4 c)
{
    float4 d;
#if __CUDA_ARCH__ >= 890
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d.x), "=f"(d.y), "=f"(d.z), "=f"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
#else
    __trap();
#endif
    return d;
}
#endif
)"

// Intrinsics over the array types the kernels stage fragments in (groupshared arrays of any element type, passed by reference)
// and over device pointers. ARow: A row-major (and B column-major, whose 8x8 matrices come in the order rows first);
// ACol / BRow / BCol likewise.
#define CU_LD16(NAME, ASM)                                                                                                       \
    uint4 NAME<CuE, let CuN : int>(__constref groupshared CuE[CuN] data, uint element, uint stride)                                      \
    {                                                                                                                             \
        __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm ASM; }      \
    }                                                                                                                             \
    uint4 NAME<CuE>(CuE* data, uint element, uint stride)                                                                             \
    {                                                                                                                             \
        __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm ASM; }      \
    }
CU_LD16(cu_ld_arow, "sl_ld16<false, true>($0, $1, $2)")
CU_LD16(cu_ld_acol, "sl_ld16<true, false>($0, $1, $2)")
CU_LD16(cu_ld_bcol, "sl_ld16<false, false>($0, $1, $2)")
CU_LD16(cu_ld_brow, "sl_ld16<true, true>($0, $1, $2)")

#define CU_ACC(V, LD, ST, LDNAME, STNAME)                                                                                          \
    V LDNAME<CuE, let CuN : int>(__constref groupshared CuE[CuN] data, uint element, uint stride, uint tile, bool colMajor)               \
    {                                                                                                                             \
        __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm LD; }            \
    }                                                                                                                             \
    V LDNAME<CuE>(CuE* data, uint element, uint stride, uint tile, bool colMajor)                                                    \
    {                                                                                                                             \
        __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm LD; }            \
    }                                                                                                                             \
    void STNAME<CuE, let CuN : int>(__ref groupshared CuE[CuN] data, uint element, uint stride, uint tile, bool colMajor, V v)            \
    {                                                                                                                             \
        __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm ST; }        \
    }                                                                                                                             \
    void STNAME<CuE>(CuE* data, uint element, uint stride, uint tile, bool colMajor, V v)                                             \
    {                                                                                                                             \
        __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm ST; }        \
    }
CU_ACC(float4, "sl_acc_ld<float4>($0, $1, $2, $3, $4)", "sl_acc_st<float4>($0, $1, $2, $3, $4, $5)", cu_accf_ld, cu_accf_st)
CU_ACC(int4, "sl_acc_ld<int4>($0, $1, $2, $3, $4)", "sl_acc_st<int4>($0, $1, $2, $3, $4, $5)", cu_acci_ld, cu_acci_st)
CU_ACC(uint2, "sl_acch_ld($0, $1, $2, $3, $4)", "sl_acch_st($0, $1, $2, $3, $4, $5)", cu_acch_ld, cu_acch_st)

float4 cu_mma_f16(uint4 a, uint2 b, float4 c)
{
    __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm "sl_mma_f16($0, $1, $2)"; }
}
float4 cu_mma_bf16(uint4 a, uint2 b, float4 c)
{
    __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm "sl_mma_bf16($0, $1, $2)"; }
}
int4 cu_mma_s8(uint4 a, uint2 b, int4 c)
{
    __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm "sl_mma_s8($0, $1, $2)"; }
}
float4 cu_mma_e4m3(uint4 a, uint2 b, float4 c)
{
    __target_switch { case cuda: __requirePrelude(CU_MMA_PRELUDE); __intrinsic_asm "sl_mma_e4m3($0, $1, $2)"; }
}

// Register index of element e: A (rows g, g + 8 in registers 0 / 1, K block in 2), B (K block in register 1, column block in 2).
uint cu_a_reg(uint e) { return (e >> 2) + 2 * ((e >> 1) & 1); }
uint cu_b_reg(uint e) { return 2 * (e >> 2) + ((e >> 1) & 1); }
// Accumulators: tile (e >> 1) & 1, component (e & 1) + 2 (e >> 2).
uint cu_acc_comp(uint e) { return (e & 1) + 2 * (e >> 2); }

uint cu_set_h(uint w, uint e, uint bits) { return (e & 1) != 0 ? (w & 0xFFFFu) | (bits << 16) : (w & 0xFFFF0000u) | bits; }
uint cu_get_h(uint w, uint e) { return (e & 1) != 0 ? w >> 16 : w & 0xFFFFu; }

// 16-bit operand fragments (A 16x16, B 16x16) of half or bfloat16 values. LDR / LDC: the loads for row- and column-major storage.
#define CU_FRAG16(NAME, ELEM, TOBITS, FROMBITS, REG, LDR, LDC)                                                                     \
    struct NAME                                                                                                                   \
    {                                                                                                                             \
        uint4 r;                                                                                                                  \
        __init() { r = uint4(0); }                                                                                                \
        __init(ELEM v) { uint w = TOBITS(v); w |= w << 16; r = uint4(w); }                                                       \
        static NAME Load<let CuL : CoopMatMatrixLayout, CuE, let CuN : int>(__constref groupshared CuE[CuN] data, uint element, uint stride) \
        {                                                                                                                         \
            NAME f;                                                                                                               \
            f.r = CuL == CoopMatMatrixLayout.RowMajor ? LDR(data, element, stride) : LDC(data, element, stride);                   \
            return f;                                                                                                             \
        }                                                                                                                         \
        static NAME Load<let CuL : CoopMatMatrixLayout, CuE>(CuE* data, uint element, uint stride)                                     \
        {                                                                                                                         \
            NAME f;                                                                                                               \
            f.r = CuL == CoopMatMatrixLayout.RowMajor ? LDR(data, element, stride) : LDC(data, element, stride);                   \
            return f;                                                                                                             \
        }                                                                                                                         \
        __subscript(uint e) -> ELEM                                                                                               \
        {                                                                                                                         \
            get { return FROMBITS(cu_get_h(r[REG(e)], e)); }                                                                      \
            set { r[REG(e)] = cu_set_h(r[REG(e)], e, TOBITS(newValue)); }                                                         \
        }                                                                                                                         \
    }

uint cu_h_bits(half v) { return uint(asuint16(v)); }
half cu_h_from(uint b) { return asfloat16(uint16_t(b)); }
uint cu_bf_bits(BFloat16 v) { return uint(bit_cast<uint16_t>(v)); }
BFloat16 cu_bf_from(uint b) { return bit_cast<BFloat16>(uint16_t(b)); }

CU_FRAG16(CuA16, half, cu_h_bits, cu_h_from, cu_a_reg, cu_ld_arow, cu_ld_acol);
CU_FRAG16(CuB16, half, cu_h_bits, cu_h_from, cu_b_reg, cu_ld_brow, cu_ld_bcol);
CU_FRAG16(CuA16bf, BFloat16, cu_bf_bits, cu_bf_from, cu_a_reg, cu_ld_arow, cu_ld_acol);
CU_FRAG16(CuB16bf, BFloat16, cu_bf_bits, cu_bf_from, cu_b_reg, cu_ld_brow, cu_ld_bcol);

// 8-bit operand fragments (A 16x32, B 32x16) of int8 or E4M3 values; loaded row-major (A) / column-major (B) only.
#define CU_FRAG8(NAME, LD)                                                                                                         \
    struct NAME                                                                                                                   \
    {                                                                                                                             \
        uint4 r;                                                                                                                  \
        static NAME Load<let CuL : CoopMatMatrixLayout, CuE, let CuN : int>(__constref groupshared CuE[CuN] data, uint element, uint stride) \
        {                                                                                                                         \
            NAME f;                                                                                                               \
            f.r = LD(data, element, stride);                                                                                      \
            return f;                                                                                                             \
        }                                                                                                                         \
        static NAME Load<let CuL : CoopMatMatrixLayout, CuE>(CuE* data, uint element, uint stride)                                     \
        {                                                                                                                         \
            NAME f;                                                                                                               \
            f.r = LD(data, element, stride);                                                                                      \
            return f;                                                                                                             \
        }                                                                                                                         \
    }
CU_FRAG8(CuA8i, cu_ld_arow);
CU_FRAG8(CuB8i, cu_ld_bcol);
CU_FRAG8(CuA8f, cu_ld_arow);
CU_FRAG8(CuB8f, cu_ld_bcol);

// Accumulators of 32-bit values (float, int): t0 / t1 the c0..c3 registers of the column blocks 0 / 1.
#define CU_ACC32(NAME, ELEM, VEC, LD, ST)                                                                                          \
    struct NAME                                                                                                                   \
    {                                                                                                                             \
        VEC t0, t1;                                                                                                               \
        __init() { t0 = VEC(0); t1 = VEC(0); }                                                                                    \
        __init(ELEM v) { t0 = VEC(v); t1 = VEC(v); }                                                                              \
        static NAME Load<let CuL : CoopMatMatrixLayout, CuE, let CuN : int>(__constref groupshared CuE[CuN] data, uint element, uint stride) \
        {                                                                                                                         \
            NAME f;                                                                                                               \
            f.t0 = LD(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor);                                              \
            f.t1 = LD(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor);                                              \
            return f;                                                                                                             \
        }                                                                                                                         \
        static NAME Load<let CuL : CoopMatMatrixLayout, CuE>(CuE* data, uint element, uint stride)                                     \
        {                                                                                                                         \
            NAME f;                                                                                                               \
            f.t0 = LD(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor);                                              \
            f.t1 = LD(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor);                                              \
            return f;                                                                                                             \
        }                                                                                                                         \
        void Store<let CuL : CoopMatMatrixLayout, CuE, let CuN : int>(__ref groupshared CuE[CuN] data, uint element, uint stride)           \
        {                                                                                                                         \
            ST(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor, t0);                                                  \
            ST(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor, t1);                                                  \
        }                                                                                                                         \
        void Store<let CuL : CoopMatMatrixLayout, CuE>(CuE* data, uint element, uint stride)                                           \
        {                                                                                                                         \
            ST(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor, t0);                                                  \
            ST(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor, t1);                                                  \
        }                                                                                                                         \
        __subscript(uint e) -> ELEM                                                                                               \
        {                                                                                                                         \
            get { return ((e >> 1) & 1) != 0 ? t1[cu_acc_comp(e)] : t0[cu_acc_comp(e)]; }                                         \
            set { if (((e >> 1) & 1) != 0) t1[cu_acc_comp(e)] = newValue; else t0[cu_acc_comp(e)] = newValue; }                  \
        }                                                                                                                         \
    }
CU_ACC32(CuAccF, float, float4, cu_accf_ld, cu_accf_st);
CU_ACC32(CuAccI, int, int4, cu_acci_ld, cu_acci_st);

// Accumulator of half values: t0 / t1 = (row g, row g + 8) of the column blocks 0 / 1, two halves per register.
struct CuAccH
{
    uint2 t0, t1;
    __init() { t0 = uint2(0); t1 = uint2(0); }
    __init(half v) { uint w = cu_h_bits(v); w |= w << 16; t0 = uint2(w); t1 = uint2(w); }
    __init(CuAccF a)
    {
        t0 = uint2(f32tof16(a.t0.x) | (f32tof16(a.t0.y) << 16), f32tof16(a.t0.z) | (f32tof16(a.t0.w) << 16));
        t1 = uint2(f32tof16(a.t1.x) | (f32tof16(a.t1.y) << 16), f32tof16(a.t1.z) | (f32tof16(a.t1.w) << 16));
    }
    static CuAccH Load<let CuL : CoopMatMatrixLayout, CuE, let CuN : int>(__constref groupshared CuE[CuN] data, uint element, uint stride)
    {
        CuAccH f;
        f.t0 = cu_acch_ld(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor);
        f.t1 = cu_acch_ld(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor);
        return f;
    }
    static CuAccH Load<let CuL : CoopMatMatrixLayout, CuE>(CuE* data, uint element, uint stride)
    {
        CuAccH f;
        f.t0 = cu_acch_ld(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor);
        f.t1 = cu_acch_ld(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor);
        return f;
    }
    void Store<let CuL : CoopMatMatrixLayout, CuE, let CuN : int>(__ref groupshared CuE[CuN] data, uint element, uint stride)
    {
        cu_acch_st(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor, t0);
        cu_acch_st(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor, t1);
    }
    void Store<let CuL : CoopMatMatrixLayout, CuE>(CuE* data, uint element, uint stride)
    {
        cu_acch_st(data, element, stride, 0, CuL != CoopMatMatrixLayout.RowMajor, t0);
        cu_acch_st(data, element, stride, 1, CuL != CoopMatMatrixLayout.RowMajor, t1);
    }
    __subscript(uint e) -> half
    {
        get { uint2 t = ((e >> 1) & 1) != 0 ? t1 : t0; return cu_h_from(cu_get_h(t[e >> 2], e)); }
        set
        {
            if (((e >> 1) & 1) != 0) t1[e >> 2] = cu_set_h(t1[e >> 2], e, cu_h_bits(newValue));
            else t0[e >> 2] = cu_set_h(t0[e >> 2], e, cu_h_bits(newValue));
        }
    }
}

// coopMatMulAdd with the generic parameters of linalg's (result type, saturation); two MMAs per 16x16 product.
CuAccF coopMatMulAdd<CuR, let CuSat : bool>(CuA16 a, CuB16 b, CuAccF c)
{
    CuAccF d;
    d.t0 = cu_mma_f16(a.r, b.r.xy, c.t0);
    d.t1 = cu_mma_f16(a.r, b.r.zw, c.t1);
    return d;
}
CuAccF coopMatMulAdd<CuR, let CuSat : bool>(CuA16bf a, CuB16bf b, CuAccF c)
{
    CuAccF d;
    d.t0 = cu_mma_bf16(a.r, b.r.xy, c.t0);
    d.t1 = cu_mma_bf16(a.r, b.r.zw, c.t1);
    return d;
}
CuAccI coopMatMulAdd<CuR, let CuSat : bool>(CuA8i a, CuB8i b, CuAccI c)
{
    CuAccI d;
    d.t0 = cu_mma_s8(a.r, b.r.xy, c.t0);
    d.t1 = cu_mma_s8(a.r, b.r.zw, c.t1);
    return d;
}
CuAccF coopMatMulAdd<CuR, let CuSat : bool>(CuA8f a, CuB8f b, CuAccF c)
{
    CuAccF d;
    d.t0 = cu_mma_e4m3(a.r, b.r.xy, c.t0);
    d.t1 = cu_mma_e4m3(a.r, b.r.zw, c.t1);
    return d;
}
