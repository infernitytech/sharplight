using Sharplight.Import.Gguf;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Collects the canonical weights of a latent-diffusion source and normalizes their storage so the graphs see one layout whatever
/// the checkpoint stored: vectors (norm scales, biases) and position tables become f32 (stable-diffusion.cpp stores 768-wide
/// LayerNorm scales as Q4_0, fp16 checkpoints store biases as f16), convolution kernels are float (a block-quantized kernel is
/// decoded to f32), and 1×1 convolution kernels that act as linear maps are viewed as matrices. Unchanged tensors stay zero-copy
/// views of the memory-mapped files.
/// </summary>
internal sealed class CanonicalWeights
{
    private readonly Dictionary<string, WeightInfo> _weights = new(StringComparer.Ordinal);

    /// <summary>The weights by canonical name.</summary>
    public IReadOnlyDictionary<string, WeightInfo> Weights => _weights;

    /// <summary>Adds <paramref name="stored"/> under <paramref name="name"/>: vectors become f32, quantized convolution kernels f32.</summary>
    /// <param name="name">The canonical name.</param>
    /// <param name="stored">The stored tensor.</param>
    public void Add(string name, WeightInfo stored)
    {
        if (stored.Shape.Rank == 1 && stored.Type != DataType.F32)
        {
            _weights[name] = ToF32(name, stored, stored.Shape);
            return;
        }

        if (stored.Shape.Rank == 4 && stored.Type.IsQuantized)
        {
            _weights[name] = ToF32(name, stored, stored.Shape);
            return;
        }

        _weights[name] = stored.As(name);
    }

    /// <summary>Adds a table the graphs add to activations (e.g. a position embedding) as f32.</summary>
    /// <param name="name">The canonical name.</param>
    /// <param name="stored">The stored tensor.</param>
    public void AddF32(string name, WeightInfo stored)
        => _weights[name] = stored.Type == DataType.F32 ? stored.As(name) : ToF32(name, stored, stored.Shape);

    /// <summary>Adds a weight that is a linear map, viewing a [O, I, 1, 1] convolution kernel as [O, I].</summary>
    /// <param name="name">The canonical name.</param>
    /// <param name="stored">The stored tensor.</param>
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

        var matrix = new Shape(stored.Shape[0], stored.Shape[1]);
        _weights[name] = stored.As(name, matrix);
    }

    /// <summary>Whether a canonical weight of that name was added.</summary>
    /// <param name="name">The canonical name.</param>
    public bool Contains(string name) => _weights.ContainsKey(name);

    private static WeightInfo ToF32(string name, WeightInfo stored, Shape shape) => WeightInfo.Derived(name, DataType.F32, shape, () =>
    {
        var values = GgmlBlockDecoder.Decode(stored.Type, stored.ReadAll());
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    });
}
