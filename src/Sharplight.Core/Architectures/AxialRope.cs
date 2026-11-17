using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Multi-axis rotary position embedding of diffusion transformers (FLUX: (index, row, column); Wan: (frame, row, column)): the head
/// width is split into one section per position axis, and each section rotates with its own frequency series θ^(−2j / axis dims).
/// The <c>Rope</c> op computes one geometric series over the whole head (θ^(−2i / head dims)) with per-section positions; constant
/// per-frequency divisors turn it into the per-axis series.
/// </summary>
internal static class AxialRope
{
    /// <summary>The per-frequency divisors [head dim / 2] of the op's series that give each axis its own series.</summary>
    /// <param name="b">The graph.</param>
    /// <param name="name">Constant name.</param>
    /// <param name="headDim">Width per head.</param>
    /// <param name="axesDim">Rotated dimensions per axis; they sum to <paramref name="headDim"/>.</param>
    /// <param name="theta">The RoPE base.</param>
    /// <exception cref="InvalidDataException">The axes do not cover the head width.</exception>
    public static Value FrequencyFactors(GraphBuilder b, string name, int headDim, IReadOnlyList<int> axesDim, float theta)
    {
        var factors = new float[headDim / 2];
        int pair = 0;
        foreach (int axisDims in axesDim)
        {
            for (int j = 0; j < axisDims / 2; j++, pair++)
            {
                double op = Math.Pow(theta, -2.0 * pair / headDim);
                double axis = Math.Pow(theta, -2.0 * j / axisDims);
                factors[pair] = (float)(op / axis);
            }
        }

        if (pair != headDim / 2)
        {
            throw new InvalidDataException($"RoPE axes {string.Join("+", axesDim)} do not cover the head width {headDim}.");
        }

        return b.Constant(name, DataType.F32, new Shape(headDim / 2), MemoryMarshal.AsBytes(factors.AsSpan()).ToArray());
    }

    /// <summary>The <c>Rope</c> attributes of interleaved (GPT-J style) rotation over three or four axes (FLUX.2: four).</summary>
    /// <param name="headDim">Width per head.</param>
    /// <param name="axesDim">Rotated dimensions of the axes.</param>
    /// <param name="theta">The RoPE base.</param>
    public static Attributes Attributes(int headDim, IReadOnlyList<int> axesDim, float theta)
        => Graph.Attributes.From(("dims", headDim), ("base", theta), ("style", "gptj"),
            ("sections", new[] { axesDim[0] / 2, axesDim[1] / 2, axesDim[2] / 2, axesDim.Count > 3 ? axesDim[3] / 2 : 0 }));
}
