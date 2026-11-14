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
}
