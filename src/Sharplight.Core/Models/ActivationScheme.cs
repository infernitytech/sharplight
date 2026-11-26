namespace Sharplight.Models;

/// <summary>
/// What the quantization format of a weight defines for the activations its products read (<see cref="WeightInfo.Activations"/>):
/// weight-only formats (W4A16, W8A16) multiply full-precision activations, others (W8A8, W4A4, with static or dynamic activation scales)
/// quantize them too. Kernels that quantize activations to a low-precision float format (E4M3, E2M1 with block scales) serve only the
/// weights whose format allows it.
/// </summary>
public enum ActivationScheme
{
    /// <summary>The source does not say (unquantized weights, formats without an activation scheme): kernels choose as for any weight.</summary>
    Unspecified = 0,

    /// <summary>
    /// Weight-only: the format multiplies full-precision activations, so products of this weight do not quantize them to FP8 or FP4
    /// (unless the compile allows reduced precision, <see cref="Runtime.CompileOptions.AllowReducedPrecision"/>).
    /// </summary>
    Unquantized = 1,

    /// <summary>The format quantizes the activations too (static or dynamic scales): kernels may quantize them to a low-precision format.</summary>
    Quantized = 2,
}
