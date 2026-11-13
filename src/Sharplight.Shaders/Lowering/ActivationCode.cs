namespace Sharplight.Shaders.Lowering;

/// <summary>Encoding of the quantized activation copies the 8-bit matmul kernels read.</summary>
internal enum ActivationCode
{
    /// <summary>q8_1: per row K int8 codes, then K / 32 (d, d * sum) float pairs (<c>quantize_q8_1</c>, int8 kernels).</summary>
    Q8_1,

    /// <summary>Per row K E4M3 codes, then K / 128 f32 scales (<c>quantize_fp8</c>, the fp8 tensor-core GEMM).</summary>
    Fp8,
}
