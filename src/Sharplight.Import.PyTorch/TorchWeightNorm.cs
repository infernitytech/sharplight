using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.PyTorch;

/// <summary>Helpers for checkpoints whose convolutions use weight normalization (<c>weight_g</c> / <c>weight_v</c>, or the parametrization's <c>original0</c> / <c>original1</c>).</summary>
public static class TorchWeightNorm
{
    /// <summary>The shape of a tensor entry.</summary>
    /// <param name="entry">The entry.</param>
    public static Shape ShapeOf(TorchTensorEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        return new Shape(entry.Shape.Select(d => Dim.Of(d)));
    }

    /// <summary>A zero-copy f32 weight over a tensor of <paramref name="file"/>.</summary>
    /// <param name="file">The opened checkpoint (must stay open while the weight is used).</param>
    /// <param name="name">The weight name.</param>
    /// <param name="entry">The tensor.</param>
    /// <param name="shape">The shape to give it, or <see langword="null"/> for the tensor's own.</param>
    public static WeightInfo View(TorchCheckpointFile file, string name, TorchTensorEntry entry, Shape? shape = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(entry);
        if (entry.DType != TorchDType.Float32)
        {
            throw new NotSupportedException($"Tensor '{entry.Name}' is {entry.DType}; only float32 checkpoints are supported here.");
        }

        return new WeightInfo { Name = name, Type = DataType.F32, Shape = shape ?? ShapeOf(entry), Source = new TorchTensorWeightSource(file, entry) };
    }

    /// <summary>
    /// The effective weight <c>g · v / ‖v‖</c> of a weight-normalized layer (the norm per row of the first dimension), computed on first read.
    /// </summary>
    /// <param name="file">The opened checkpoint.</param>
    /// <param name="name">The name of the folded weight.</param>
    /// <param name="gain">The gain tensor <c>g</c> (one value per row of <paramref name="direction"/>).</param>
    /// <param name="direction">The direction tensor <c>v</c>.</param>
    public static WeightInfo Fold(TorchCheckpointFile file, string name, TorchTensorEntry gain, TorchTensorEntry direction)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(gain);
        ArgumentNullException.ThrowIfNull(direction);
        return WeightInfo.Derived(name, DataType.F32, ShapeOf(direction), () =>
        {
            var g = Floats(file, gain);
            var v = Floats(file, direction);
            long rows = direction.Shape[0];
            long per = v.Length / rows;
            for (long r = 0; r < rows; r++)
            {
                double norm = 0;
                for (long i = 0; i < per; i++)
                {
                    norm += (double)v[(r * per) + i] * v[(r * per) + i];
                }

                float scale = (float)(g[r] / Math.Sqrt(norm));
                for (long i = 0; i < per; i++)
                {
                    v[(r * per) + i] *= scale;
                }
            }

            return MemoryMarshal.AsBytes(v.AsSpan()).ToArray();
        });
    }

    /// <summary>The sum of two same-shaped tensors (for example the two bias vectors of an LSTM direction), computed on first read.</summary>
    /// <param name="file">The opened checkpoint.</param>
    /// <param name="name">The weight name.</param>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    public static WeightInfo Sum(TorchCheckpointFile file, string name, TorchTensorEntry a, TorchTensorEntry b)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        return WeightInfo.Derived(name, DataType.F32, ShapeOf(a), () =>
        {
            var x = Floats(file, a);
            var y = Floats(file, b);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += y[i];
            }

            return MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
        });
    }

    private static float[] Floats(TorchCheckpointFile file, TorchTensorEntry entry)
    {
        if (entry.DType != TorchDType.Float32)
        {
            throw new NotSupportedException($"Tensor '{entry.Name}' is {entry.DType}; only float32 checkpoints are supported here.");
        }

        var values = new float[entry.ByteLength / sizeof(float)];
        new TorchTensorWeightSource(file, entry).Read(0, MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }
}
