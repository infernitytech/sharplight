using Sharplight.Tensors;

namespace Sharplight.Graph;

// Signal processing.
public static partial class StandardOps
{
    /// <summary>
    /// Discrete Fourier transform along the second-to-last axis: x f32 <c>[..., N, C]</c> (C = 1 real, 2 interleaved real/imaginary) →
    /// <c>[..., out_bins, 2]</c>, or <c>[..., out_bins, 1]</c> with <c>real_output</c>. Attributes: <c>length</c> (DFT length L; the input is
    /// truncated or zero-padded to it), <c>out_bins</c> (bins kept, L or L/2 + 1), <c>inverse</c> (bool: e^{+2πi kn/L} / L),
    /// <c>real_output</c> (bool: inverse of a one-sided spectrum, the real signal of length out_bins whose spectrum is the Hermitian
    /// extension of the first L/2 + 1 input bins).
    /// </summary>
    public static readonly OpSchema Dft = new("Dft", OpCategory.Composite, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        if (x.Rank < 2 || x[-1].Fixed is not (1 or 2))
        {
            throw new ArgumentException($"Dft expects x [..., N, 1|2], got {x}.");
        }

        var dims = x.Dims.ToArray();
        dims[^2] = a.Get<long>("out_bins");
        dims[^1] = a.Get("real_output", false) ? 1 : 2;
        return [new(DataType.F32, new Shape(dims))];
    });
}
