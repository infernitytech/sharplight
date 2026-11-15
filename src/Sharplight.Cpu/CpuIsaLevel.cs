namespace Sharplight.Cpu;

/// <summary>
/// Instruction-set tiers of the CPU kernels, ordered from the portable reference to the widest path. Above
/// <see cref="Vector128"/> there are two families, Arm64 (<see cref="Neon"/>, <see cref="NeonDot"/>) and x86 (<see cref="Avx2"/>
/// and up), of which a host supports only one; within a family a higher tier only adds instructions. Every kernel has a path for
/// every tier at or below the host's, and the scalar tier is the reference the faster paths are tested against.
/// </summary>
internal enum CpuIsaLevel
{
    /// <summary>Plain scalar code, no SIMD (reference).</summary>
    Scalar,

    /// <summary>128-bit cross-platform vectors (SSE4 / Arm64 AdvSimd).</summary>
    Vector128,

    /// <summary>Arm64 Advanced SIMD (NEON) kernels: 128-bit vectors with fused multiply-add by lane.</summary>
    Neon,

    /// <summary>NEON plus the Armv8.2 dot-product instructions (<c>sdot</c>, FEAT_DotProd) for int8 dot products.</summary>
    NeonDot,

    /// <summary>256-bit vectors with AVX2 + FMA; int8 dot products through <c>vpmaddubsw</c>.</summary>
    Avx2,

    /// <summary>AVX2 plus 256-bit AVX-VNNI int8 dot products (<c>vpdpbusd ymm</c>).</summary>
    AvxVnni,

    /// <summary>512-bit vectors with AVX-512 F/BW/VL; int8 dot products through 512-bit <c>vpmaddubsw</c>.</summary>
    Avx512,

    /// <summary>AVX-512 plus 512-bit VNNI int8 dot products (<c>vpdpbusd zmm</c>).</summary>
    Avx512Vnni,
}
