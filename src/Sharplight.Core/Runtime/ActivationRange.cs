namespace Sharplight.Runtime;

/// <summary>
/// The largest magnitude among the activations a graph feeds into matrix products, convolutions and attention, measured by
/// <see cref="ActivationRangeProbe"/>.
/// </summary>
/// <param name="MaxMagnitude">The largest |value|; <see cref="float.PositiveInfinity"/> when a probed value was not finite.</param>
/// <param name="Location">The name of the graph value holding it, or <see langword="null"/> when the graph has no probed value.</param>
public sealed record ActivationRange(float MaxMagnitude, string? Location)
{
    /// <summary>
    /// The largest magnitude that still counts as f16-safe: a sixteenth of f16's largest value (65504, so about 4094). Activation
    /// magnitudes depend on the input and the image size (SDXL's VAE decoder: 2e4 for Gaussian latents at 128 × 128 pixels, 1.4e5 for
    /// uniform ones at 1024 × 1024), so a small probe needs a wide margin; f16-safe decoders stay far below it (FLUX.1's and SD 1.5's
    /// at most 1.1e3).
    /// </summary>
    public const float HalfSafeLimit = 65504f / 16;

    /// <summary>Whether the activations stay within <see cref="HalfSafeLimit"/>.</summary>
    public bool FitsHalf => MaxMagnitude <= HalfSafeLimit;

    /// <summary>The activation precision for the measured graph: <see cref="ActivationPrecision.BFloat16"/> unless it <see cref="FitsHalf"/>.</summary>
    public ActivationPrecision Precision => FitsHalf ? ActivationPrecision.Half : ActivationPrecision.BFloat16;
}
