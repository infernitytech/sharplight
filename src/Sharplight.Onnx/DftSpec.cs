namespace Sharplight.Onnx;

/// <summary>One ONNX DFT along an axis, resolved to sizes.</summary>
/// <param name="Length">DFT length L (the input is truncated or zero-padded to it).</param>
/// <param name="InLength">Input extent along the axis.</param>
/// <param name="InComponents">1 for a real input, 2 for interleaved real/imaginary.</param>
/// <param name="OutBins">Output extent along the axis (L, L/2 + 1 for a one-sided forward transform).</param>
/// <param name="OutComponents">2, or 1 for the real output of an inverse one-sided transform.</param>
/// <param name="Inverse">Inverse transform (scaled by 1/L).</param>
internal sealed record DftSpec(int Length, int InLength, int InComponents, int OutBins, int OutComponents, bool Inverse)
{
    /// <summary>Inverse of a one-sided spectrum (numpy irfft): real output.</summary>
    public bool InverseOneSided => Inverse && OutComponents == 1;

    /// <summary>Resolves the ONNX attributes; <paramref name="length"/> null takes the ONNX default.</summary>
    public static DftSpec Create(long inLength, long inComponents, long? length, bool inverse, bool oneSided)
    {
        if (inComponents is not (1 or 2))
        {
            throw new InvalidDataException($"DFT input must end in a dimension of 1 (real) or 2 (complex), got {inComponents}");
        }

        long l = length ?? (inverse && oneSided ? 2 * (inLength - 1) : inLength);
        if (l <= 0 || l > int.MaxValue / 4)
        {
            throw new InvalidDataException($"invalid DFT length {l}");
        }

        int len = (int)l;
        return inverse && oneSided
            ? new DftSpec(len, (int)inLength, (int)inComponents, len, 1, true)
            : new DftSpec(len, (int)inLength, (int)inComponents, oneSided ? (len / 2) + 1 : len, 2, inverse);
    }
}
