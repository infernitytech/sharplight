namespace Sharplight.Shaders.Lowering;

/// <summary>Device-friendly weight layouts the lowering may choose (same byte size as the source format).</summary>
public enum WeightRepack
{
    None,
    /// <summary>
    /// Q8_0 rows → [K/32 f16 scales][K int8 quants]: quants become 8-byte aligned so kernels use wide loads.
    /// Requires K % 128 == 0. Kernels see the type <see cref="ShaderTypes.Q8_0R"/>.
    /// </summary>
    Q8_0Split,
}
