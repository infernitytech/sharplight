using Sharplight.Import.PyTorch;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Kokoro;

/// <summary>
/// The weights of a Kokoro checkpoint under canonical names: the <c>module.</c> wrapper of the training run is dropped, every weight-normalized
/// convolution (<c>weight_g</c> and <c>weight_v</c>) becomes its effective <c>weight</c> (<c>g · v / ‖v‖</c> per output row, computed on first read),
/// the two LSTM bias vectors of a direction are added into one, and the single-element ALBERT pooler is skipped. Everything else is a zero-copy
/// view of the memory-mapped file.
/// </summary>
internal static class KokoroWeights
{
    /// <summary>The component sections of the checkpoint (the top-level keys of its state dictionary).</summary>
    private static readonly string[] s_sections = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"];

    /// <summary>Maps the tensors of <paramref name="file"/>.</summary>
    /// <param name="file">The opened checkpoint.</param>
    public static Dictionary<string, WeightInfo> Map(TorchCheckpointFile file)
    {
        var tensors = new Dictionary<string, TorchTensorEntry>(StringComparer.Ordinal);
        foreach (var (key, entry) in file.Tensors)
        {
            string name = key.Replace(".module.", ".", StringComparison.Ordinal);
            if (!s_sections.Any(s => name.StartsWith(s + ".", StringComparison.Ordinal)))
            {
                throw new NotSupportedException($"Kokoro: unexpected tensor '{key}'.");
            }

            tensors[name] = entry;
        }

        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, entry) in tensors)
        {
            if (name.StartsWith("bert.pooler.", StringComparison.Ordinal))
            {
                continue;
            }

            if (name.EndsWith(".weight_g", StringComparison.Ordinal))
            {
                string stem = name[..^"_g".Length];
                weights[stem] = Folded(file, stem, entry, tensors[stem + "_v"]);
            }
            else if (name.EndsWith(".weight_v", StringComparison.Ordinal))
            {
                continue;
            }
            else if (name.Contains(".bias_ih_l0", StringComparison.Ordinal))
            {
                string stem = name.Replace(".bias_ih_l0", ".bias_l0", StringComparison.Ordinal);
                weights[stem] = Summed(file, stem, entry, tensors[name.Replace("bias_ih", "bias_hh", StringComparison.Ordinal)]);
            }
            else if (name.Contains(".bias_hh_l0", StringComparison.Ordinal))
            {
                continue;
            }
            else
            {
                var shape = new Shape(entry.Shape);
                if (name.Contains(".alpha", StringComparison.Ordinal))
                {
                    shape = new Shape(entry.Shape[1]);
                }

                weights[name] = View(file, name, entry, shape);
            }
        }

        return weights;
    }

    private static WeightInfo View(TorchCheckpointFile file, string name, TorchTensorEntry entry, Shape shape)
        => new() { Name = name, Type = DataType.F32, Shape = shape, Source = new KokoroTorchWeightSource(file, entry) };

    private static float[] Floats(TorchCheckpointFile file, TorchTensorEntry entry)
    {
        if (entry.DType != TorchDType.Float32)
        {
            throw new NotSupportedException($"Kokoro: tensor '{entry.Name}' is {entry.DType}, only float32 is supported.");
        }

        var values = new float[entry.ByteLength / sizeof(float)];
        new KokoroTorchWeightSource(file, entry).Read(0, System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    private static WeightInfo Folded(TorchCheckpointFile file, string name, TorchTensorEntry g, TorchTensorEntry v)
    {
        var shape = new Shape(v.Shape);
        return WeightInfo.Derived(name, DataType.F32, shape, () =>
        {
            var gain = Floats(file, g);
            var direction = Floats(file, v);
            long rows = v.Shape[0];
            long per = direction.Length / rows;
            for (long r = 0; r < rows; r++)
            {
                double norm = 0;
                for (long i = 0; i < per; i++)
                {
                    norm += (double)direction[(r * per) + i] * direction[(r * per) + i];
                }

                float scale = (float)(gain[r] / Math.Sqrt(norm));
                for (long i = 0; i < per; i++)
                {
                    direction[(r * per) + i] *= scale;
                }
            }

            return System.Runtime.InteropServices.MemoryMarshal.AsBytes(direction.AsSpan()).ToArray();
        });
    }

    private static WeightInfo Summed(TorchCheckpointFile file, string name, TorchTensorEntry a, TorchTensorEntry b)
        => WeightInfo.Derived(name, DataType.F32, new Shape(a.Shape), () =>
        {
            var x = Floats(file, a);
            var y = Floats(file, b);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += y[i];
            }

            return System.Runtime.InteropServices.MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
        });
}
