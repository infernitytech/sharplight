using System.Collections.Concurrent;

namespace Sharplight.Tensors;

/// <summary>
/// Element storage type of a tensor. Plain scalar types have <see cref="BlockSize"/> == 1;
/// quantized formats store <see cref="BlockSize"/> logical elements in <see cref="BlockBytes"/> bytes.
/// </summary>
/// <remarks>
/// Data types are open-ended: new (quantized) formats can be registered at runtime via
/// <see cref="Register"/>. A backend that does not know how to read a format simply reports the
/// consuming op as unsupported and the planner will route it elsewhere.
/// </remarks>
public sealed class DataType : IEquatable<DataType>
{
    private static readonly ConcurrentDictionary<string, DataType> s_byName = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Lower-case name, as used in saved packages and op attributes (for example "f32" or "q4_k").</summary>
    public string Name { get; }
    /// <summary>Logical elements per storage block (1 for plain scalar types).</summary>
    public int BlockSize { get; }
    /// <summary>Bytes per storage block.</summary>
    public int BlockBytes { get; }
    /// <summary>Whether the type is a block-quantized format.</summary>
    public bool IsQuantized => BlockSize > 1 || Kind == DataTypeKind.Quantized;
    /// <summary>Class of the type.</summary>
    public DataTypeKind Kind { get; }

    private DataType(string name, DataTypeKind kind, int blockSize, int blockBytes)
    {
        Name = name;
        Kind = kind;
        BlockSize = blockSize;
        BlockBytes = blockBytes;
    }

    /// <summary>Registers (or returns the existing) data type with the given layout.</summary>
    public static DataType Register(string name, DataTypeKind kind, int blockSize, int blockBytes)
    {
        var dt = s_byName.GetOrAdd(name, n => new DataType(n, kind, blockSize, blockBytes));
        if (dt.BlockSize != blockSize || dt.BlockBytes != blockBytes || dt.Kind != kind)
        {
            throw new InvalidOperationException($"Data type '{name}' already registered with a different layout.");
        }

        return dt;
    }

    /// <summary>Looks up a type by case-insensitive name; <see langword="null"/> when unknown.</summary>
    public static DataType? TryGet(string name) => s_byName.TryGetValue(name, out var dt) ? dt : null;
    /// <summary>
    /// Looks up a type by case-insensitive name.
    /// </summary>
    /// <param name="name">Type name.</param>
    /// <exception cref="KeyNotFoundException">No such type is registered.</exception>
    public static DataType Get(string name) => TryGet(name) ?? throw new KeyNotFoundException($"Unknown data type '{name}'.");
    /// <summary>All registered types.</summary>
    public static IEnumerable<DataType> All => s_byName.Values;

    /// <summary>Bytes needed to store <paramref name="elements"/> elements (must be a multiple of the block size).</summary>
    public long ByteSize(long elements)
    {
        if (elements % BlockSize != 0)
        {
            throw new ArgumentException($"{elements} elements is not a multiple of the {Name} block size {BlockSize}.");
        }

        return elements / BlockSize * BlockBytes;
    }

    /// <summary>Size of one element in bytes, only valid for non-block types.</summary>
    public int ElementSize => BlockSize == 1 ? BlockBytes : throw new InvalidOperationException($"{Name} is a block format.");

    // ---- well-known types ------------------------------------------------------------------
    /// <summary>32-bit IEEE float.</summary>
    public static readonly DataType F32 = Register("f32", DataTypeKind.Float, 1, 4);
    /// <summary>16-bit IEEE half-precision float.</summary>
    public static readonly DataType F16 = Register("f16", DataTypeKind.Float, 1, 2);
    /// <summary>bfloat16 (the upper 16 bits of an f32).</summary>
    public static readonly DataType BF16 = Register("bf16", DataTypeKind.Float, 1, 2);
    /// <summary>64-bit IEEE float.</summary>
    public static readonly DataType F64 = Register("f64", DataTypeKind.Float, 1, 8);
    /// <summary>Signed 8-bit integer.</summary>
    public static readonly DataType I8 = Register("i8", DataTypeKind.Int, 1, 1);
    /// <summary>Unsigned 8-bit integer.</summary>
    public static readonly DataType U8 = Register("u8", DataTypeKind.Int, 1, 1);
    /// <summary>Signed 16-bit integer.</summary>
    public static readonly DataType I16 = Register("i16", DataTypeKind.Int, 1, 2);
    /// <summary>Signed 32-bit integer (index and id tensors).</summary>
    public static readonly DataType I32 = Register("i32", DataTypeKind.Int, 1, 4);
    /// <summary>Unsigned 32-bit integer.</summary>
    public static readonly DataType U32 = Register("u32", DataTypeKind.Int, 1, 4);
    /// <summary>Signed 64-bit integer.</summary>
    public static readonly DataType I64 = Register("i64", DataTypeKind.Int, 1, 8);
    /// <summary>Boolean stored as one byte.</summary>
    public static readonly DataType Bool = Register("bool", DataTypeKind.Bool, 1, 1);

    // ggml-style block quantization formats (layouts are identical across backends).
    /// <summary>ggml 4-bit blocks of 32 with an f16 scale (18 bytes).</summary>
    public static readonly DataType Q4_0 = Register("q4_0", DataTypeKind.Quantized, 32, 18);
    /// <summary>ggml 4-bit blocks of 32 with f16 scale and minimum (20 bytes).</summary>
    public static readonly DataType Q4_1 = Register("q4_1", DataTypeKind.Quantized, 32, 20);
    /// <summary>ggml 5-bit blocks of 32 with an f16 scale (22 bytes).</summary>
    public static readonly DataType Q5_0 = Register("q5_0", DataTypeKind.Quantized, 32, 22);
    /// <summary>ggml 5-bit blocks of 32 with f16 scale and minimum (24 bytes).</summary>
    public static readonly DataType Q5_1 = Register("q5_1", DataTypeKind.Quantized, 32, 24);
    /// <summary>ggml 8-bit blocks of 32 with an f16 scale (34 bytes).</summary>
    public static readonly DataType Q8_0 = Register("q8_0", DataTypeKind.Quantized, 32, 34);
    /// <summary>ggml 2-bit k-quant super-blocks of 256 (84 bytes).</summary>
    public static readonly DataType Q2_K = Register("q2_k", DataTypeKind.Quantized, 256, 84);
    /// <summary>ggml 3-bit k-quant super-blocks of 256 (110 bytes).</summary>
    public static readonly DataType Q3_K = Register("q3_k", DataTypeKind.Quantized, 256, 110);
    /// <summary>ggml 4-bit k-quant super-blocks of 256 (144 bytes).</summary>
    public static readonly DataType Q4_K = Register("q4_k", DataTypeKind.Quantized, 256, 144);
    /// <summary>ggml 5-bit k-quant super-blocks of 256 (176 bytes).</summary>
    public static readonly DataType Q5_K = Register("q5_k", DataTypeKind.Quantized, 256, 176);
    /// <summary>ggml 6-bit k-quant super-blocks of 256 (210 bytes).</summary>
    public static readonly DataType Q6_K = Register("q6_k", DataTypeKind.Quantized, 256, 210);
    /// <summary>ggml 8-bit k-quant super-blocks of 256 (292 bytes).</summary>
    public static readonly DataType Q8_K = Register("q8_k", DataTypeKind.Quantized, 256, 292);
    /// <summary>ggml non-linear 4-bit blocks of 32 (18 bytes).</summary>
    public static readonly DataType IQ4_NL = Register("iq4_nl", DataTypeKind.Quantized, 32, 18);
    /// <summary>ggml non-linear 4-bit super-blocks of 256 (136 bytes).</summary>
    public static readonly DataType IQ4_XS = Register("iq4_xs", DataTypeKind.Quantized, 256, 136);
    /// <summary>Microscaling FP4 blocks of 32 with a shared 8-bit exponent (17 bytes).</summary>
    public static readonly DataType MXFP4 = Register("mxfp4", DataTypeKind.Quantized, 32, 17);
    /// <summary>ggml 2.06-bit super-blocks of 256: f16 scale, E8 lattice grid indices and sign/scale words (66 bytes).</summary>
    public static readonly DataType IQ2_XXS = Register("iq2_xxs", DataTypeKind.Quantized, 256, 66);
    /// <summary>ggml 2.31-bit super-blocks of 256: f16 scale, 9-bit grid indices with 7-bit signs, 4-bit scales (74 bytes).</summary>
    public static readonly DataType IQ2_XS = Register("iq2_xs", DataTypeKind.Quantized, 256, 74);
    /// <summary>ggml 2.56-bit super-blocks of 256: f16 scale, 10-bit grid indices, explicit sign bytes, 4-bit scales (82 bytes).</summary>
    public static readonly DataType IQ2_S = Register("iq2_s", DataTypeKind.Quantized, 256, 82);
    /// <summary>ggml 3.06-bit super-blocks of 256: f16 scale, 8-bit grid indices (four values each), sign/scale words (98 bytes).</summary>
    public static readonly DataType IQ3_XXS = Register("iq3_xxs", DataTypeKind.Quantized, 256, 98);
    /// <summary>ggml 3.44-bit super-blocks of 256: f16 scale, 9-bit grid indices (four values each), explicit signs, 4-bit scales (110 bytes).</summary>
    public static readonly DataType IQ3_S = Register("iq3_s", DataTypeKind.Quantized, 256, 110);
    /// <summary>ggml 1.56-bit super-blocks of 256: f16 scale, 11-bit grid indices (eight values each) with a 3-bit scale and delta sign per 32 (50 bytes).</summary>
    public static readonly DataType IQ1_S = Register("iq1_s", DataTypeKind.Quantized, 256, 50);
    /// <summary>ggml 1.75-bit super-blocks of 256: like <see cref="IQ1_S"/> with per-16 delta signs and a scale split over the 3-bit block scales (56 bytes).</summary>
    public static readonly DataType IQ1_M = Register("iq1_m", DataTypeKind.Quantized, 256, 56);
    /// <summary>ggml ternary (BitNet) super-blocks of 256 packing five base-3 digits per byte, f16 scale (54 bytes).</summary>
    public static readonly DataType TQ1_0 = Register("tq1_0", DataTypeKind.Quantized, 256, 54);
    /// <summary>ggml ternary (BitNet) super-blocks of 256 with 2-bit codes, f16 scale (66 bytes).</summary>
    public static readonly DataType TQ2_0 = Register("tq2_0", DataTypeKind.Quantized, 256, 66);
    /// <summary>NVIDIA FP4 blocks of 64: four UE4M3 sub-block scales and 4-bit E2M1 values (36 bytes).</summary>
    public static readonly DataType NVFP4 = Register("nvfp4", DataTypeKind.Quantized, 64, 36);

    // Group-wise 1-bit / ternary formats (Bonsai): one f16 scale per group, w = scale * code.
    /// <summary>1 bit per weight, group 128: { f16 d; u8 bits[16] }, bit j (LSB first) set = +d, clear = -d (18 bytes).</summary>
    public static readonly DataType Q1_0 = Register("q1_0", DataTypeKind.Quantized, 128, 18);
    /// <summary>2-bit codes, group 64: { f16 d; u8 qs[16] }, code j in bits 2(j%4) of byte j/4, w = (code-1)*d (code 3 = +2d; ternary uses 0..2) (18 bytes).</summary>
    public static readonly DataType Q2_0 = Register("q2_0", DataTypeKind.Quantized, 64, 18);
    /// <summary>Same 2-bit code as <see cref="Q2_0"/> with a scale per 128 weights: { f16 d; u8 qs[32] } (34 bytes).</summary>
    public static readonly DataType Q2_0_G128 = Register("q2_0_g128", DataTypeKind.Quantized, 128, 34);
    /// <summary>
    /// Dense base-3 ternary, group 128: { u8 qs[24]; u8 qh[2]; f16 d } (28 bytes). qs bytes pack 5 trits each (digit n of byte
    /// b is ((b*3^n mod 256)*3)>>8), in stages: bytes 0..15 give values n*16+m, bytes 16..23 give 80+n*8+m; the two qh bytes give 4 trits
    /// each, values 120+n*2+h. w = (trit-1)*d.
    /// </summary>
    public static readonly DataType Tern1_G128 = Register("tern1_g128", DataTypeKind.Quantized, 128, 28);

    // 8-bit floating point (OCP FP8). The scalar types are storage formats read like weights (decoded on load), so they are
    // registered as Quantized with a block size of 1: only the ops that decode weight formats accept them.
    /// <summary>
    /// OCP FP8 E4M3FN, one byte per value: sign bit, 4 exponent bits (bias 7), 3 mantissa bits; exponent 0 is subnormal
    /// (m * 2^-9), there is no infinity, 0x7F / 0xFF are NaN, the largest finite magnitude is 448.
    /// </summary>
    public static readonly DataType F8E4M3 = Register("f8e4m3", DataTypeKind.Quantized, 1, 1);
    /// <summary>OCP FP8 E5M2, one byte per value: the upper byte of an IEEE half (sign, 5 exponent bits with bias 15, 2 mantissa bits; inf and NaN as in half).</summary>
    public static readonly DataType F8E5M2 = Register("f8e5m2", DataTypeKind.Quantized, 1, 1);
    /// <summary>
    /// Block-scaled FP8, 32 values along the row (the contiguous K dimension): { f32 d; u8 q[32] } (36 bytes, 4-byte aligned when
    /// rows are), value i = d * e4m3(q[i]) with <see cref="F8E4M3"/> codes. Carries per-tensor, per-channel or block scales of any
    /// granularity that is a multiple of 32 along K.
    /// </summary>
    public static readonly DataType F8E4M3_B32 = Register("f8e4m3_b32", DataTypeKind.Quantized, 32, 36);
    /// <summary>
    /// Block-scaled FP8, 128 values along the row: { f32 d; u8 q[128] } (132 bytes), value i = d * e4m3(q[i]). The layout of
    /// 128x128 block-quantized checkpoints (one scale per 128 x 128 tile, repeated in each of the tile's 128 rows).
    /// </summary>
    public static readonly DataType F8E4M3_B128 = Register("f8e4m3_b128", DataTypeKind.Quantized, 128, 132);

    /// <inheritdoc/>
    public bool Equals(DataType? other) => ReferenceEquals(this, other);
    /// <inheritdoc/>
    public override bool Equals(object? obj) => ReferenceEquals(this, obj);
    /// <inheritdoc/>
    public override int GetHashCode() => Name.GetHashCode(StringComparison.OrdinalIgnoreCase);
    /// <inheritdoc/>
    public override string ToString() => Name;
}
