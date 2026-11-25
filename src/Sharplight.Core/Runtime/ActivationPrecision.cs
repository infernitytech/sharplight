namespace Sharplight.Runtime;

/// <summary>
/// The 16-bit format reduced precision (<see cref="CompileOptions.AllowReducedPrecision"/>) gives the activations entering matrix
/// products, convolutions and attention (<see cref="CompileOptions.ActivationPrecision"/>). Products always accumulate in f32; the
/// choice only decides the range and the significant bits of each activation in a product.
/// </summary>
public enum ActivationPrecision
{
    /// <summary>
    /// The default: f16 (11 significant bits, largest value 65504), the fastest form on devices with f16 matrix products. A
    /// component whose activations exceed f16's range produces infinities under it (<see cref="ActivationRangeProbe"/> detects such
    /// components).
    /// </summary>
    Half,

    /// <summary>
    /// For components whose activations exceed f16's range: bf16 (8 significant bits, f32's range) in the kernels that have a bf16 form
    /// on devices with bf16 matrix products (<see cref="DeviceFeatures.MatrixBF16"/>), f32 activations with the strict products
    /// everywhere else.
    /// </summary>
    BFloat16,
}
