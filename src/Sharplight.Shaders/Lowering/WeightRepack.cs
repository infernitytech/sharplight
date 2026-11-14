namespace Sharplight.Shaders.Lowering;

/// <summary>Device-friendly weight layouts the lowering may choose (same byte size as the source format).</summary>
public enum WeightRepack
{
    /// <summary>No repacking: the weight is used in its stored layout.</summary>
    None,
    /// <summary>
    /// Q8_0 rows → [K/32 f16 scales][K int8 quants]: quants become 8-byte aligned so kernels use wide loads.
    /// Requires K % 128 == 0. Kernels see the type <see cref="ShaderTypes.Q8_0R"/>.
    /// </summary>
    Q8_0Split,
    /// <summary>
    /// MXFP4 rows in groups of 16 → [16 rows of K/2 code bytes][16 rows of K/32 E8M0 scale bytes]: every row's codes start
    /// 16-byte aligned, so kernels read one 16-byte word per 32-value block instead of unaligned 17-byte blocks. The
    /// repacked unit is the 16-row group (<see cref="WeightRepacker.RowBytes"/>); requires a row count divisible by 16 per
    /// matrix. Kernels see the type <see cref="ShaderTypes.MXFP4R"/>.
    /// </summary>
    Mxfp4Split,
    /// <summary>
    /// Block-scaled FP8 rows (blocks of { f32 scale; B codes }, B = 32) → [K codes][K/B f32 scales]: the codes of every row
    /// start 16-byte aligned, so kernels read them with 16-byte loads instead of 4-byte loads of 36-byte blocks. Requires
    /// K % 128 == 0 (a row size divisible by 16). Kernels see <see cref="ShaderTypes.F8E4M3_B32R"/>.
    /// </summary>
    Fp8B32Split,
    /// <summary>
    /// The <see cref="Fp8B32Split"/> layout for 128-value blocks (132-byte blocks). Requires K % 512 == 0. Kernels see
    /// <see cref="ShaderTypes.F8E4M3_B128R"/>.
    /// </summary>
    Fp8B128Split,
}
