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
                weights[stem] = TorchWeightNorm.Fold(file, stem, entry, tensors[stem + "_v"]);
            }
            else if (name.EndsWith(".weight_v", StringComparison.Ordinal))
            {
                continue;
            }
            else if (name.Contains(".bias_ih_l0", StringComparison.Ordinal))
            {
                string stem = name.Replace(".bias_ih_l0", ".bias_l0", StringComparison.Ordinal);
                weights[stem] = TorchWeightNorm.Sum(file, stem, entry, tensors[name.Replace("bias_ih", "bias_hh", StringComparison.Ordinal)]);
            }
            else if (name.Contains(".bias_hh_l0", StringComparison.Ordinal))
            {
                continue;
            }
            else
            {
                // The snake parameters are stored as [1, channels, 1].
                weights[name] = TorchWeightNorm.View(file, name, entry, name.Contains(".alpha", StringComparison.Ordinal) ? new Shape(entry.Shape[1]) : null);
            }
        }

        return weights;
    }
}
