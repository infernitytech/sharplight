using Sharplight.Gguf;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Collects the canonical weights of a latent-diffusion source and normalizes their storage: vectors (norm scales, biases) become
/// f32 whatever they were stored as (stable-diffusion.cpp stores 768-wide LayerNorm scales as Q8_0, for instance), and 1×1
/// convolution kernels that act as linear maps are viewed as matrices.
/// </summary>
internal sealed class CanonicalWeights
{
    private readonly Dictionary<string, WeightInfo> _weights = new(StringComparer.Ordinal);

    /// <summary>The weights by canonical name.</summary>
    public IReadOnlyDictionary<string, WeightInfo> Weights => _weights;

    /// <summary>Adds <paramref name="stored"/> under <paramref name="name"/>; vectors are converted to f32.</summary>
    public void Add(string name, WeightInfo stored)
    {
        if (stored.Shape.Rank == 1 && stored.Type != DataType.F32)
        {
            _weights[name] = ToF32(name, stored);
            return;
        }

        _weights[name] = stored.As(name);
    }

    /// <summary>Adds a weight that is a linear map, viewing a [O, I, 1, 1] convolution kernel as [O, I].</summary>
    /// <exception cref="InvalidDataException">The kernel is larger than 1×1.</exception>
    public void AddLinear(string name, WeightInfo stored)
    {
        if (stored.Shape.Rank != 4)
        {
            Add(name, stored);
            return;
        }

        if (stored.Shape[2].Fixed != 1 || stored.Shape[3].Fixed != 1)
        {
            throw new InvalidDataException($"{stored.Name}: expected a 1x1 convolution kernel, got {stored.Shape}.");
        }

        _weights[name] = stored.As(name, new Shape(stored.Shape[0], stored.Shape[1]));
    }

    /// <summary>Adds the two halves of a stacked projection [2N, K] as separate [N, K] weights (zero-copy row slices).</summary>
    public void AddHalves(string first, string second, WeightInfo stored)
    {
        long half = stored.Shape[0].Fixed / 2;
        Add(first, stored.RowSlice(first, 0, half));
        Add(second, stored.RowSlice(second, half, half));
    }

    /// <summary>Whether a canonical weight of that name was added.</summary>
    public bool Contains(string name) => _weights.ContainsKey(name);

    /// <summary>A canonical weight added before.</summary>
    /// <exception cref="InvalidDataException">The checkpoint provided no such weight.</exception>
    public WeightInfo Get(string name) => _weights.TryGetValue(name, out var weight)
        ? weight : throw new InvalidDataException($"The checkpoint has no weight for '{name}'.");

    private static WeightInfo ToF32(string name, WeightInfo stored) => WeightInfo.Derived(name, DataType.F32, stored.Shape, () =>
    {
        var values = GgmlBlockDecoder.Decode(stored.Type, stored.ReadAll());
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    });
}
