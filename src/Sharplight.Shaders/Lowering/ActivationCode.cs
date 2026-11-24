namespace Sharplight.Shaders.Lowering;

/// <summary>Encoding of the activation copies the tensor-core matmul kernels read (shared by the matmuls reading the same rows).</summary>
internal enum ActivationCode
{
    /// <summary>q8_1: per row K int8 codes, then K / 32 (d, d * sum) float pairs (<c>quantize_q8_1</c>, int8 kernels).</summary>
    Q8_1,

    /// <summary>Per row K E4M3 codes, then K / 128 f32 scales (<c>quantize_fp8</c>, the fp8 tensor-core GEMM).</summary>
    Fp8,

    /// <summary>Dense f16 rows of K values (<c>to_f16</c>, the f16 tensor-core GEMMs).</summary>
    F16,

    /// <summary>
    /// Per row K f16 hi parts, K f16 lo parts and the f32 inverse of the row's power-of-two scale, 4K + 16 bytes
    /// (<c>split_rows</c>, the split-f16 GEMM).
    /// </summary>
    SplitF16,

    /// <summary>
    /// Per row K f16 hi parts and the f32 inverse of the row's power-of-two scale, 2K + 16 bytes (<c>split_rows</c> with
    /// SPLIT_PARTS 1, the single-product GEMM of fast matrix precision).
    /// </summary>
    ScaledF16,

    /// <summary>
    /// Per row, for each 256 values of K a 160-byte record of 128 E2M1 code bytes, 16 E4M3 scales (one per 16 values) and 16 bytes of
    /// padding, then 32 bytes with the row's f32 global scale (<c>quantize_fp4</c> in gemm_fp4_native.cu, the FP4 GEMM).
    /// </summary>
    Fp4,
}
